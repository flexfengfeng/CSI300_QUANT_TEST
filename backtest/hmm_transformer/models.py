"""模型: HMM 市场状态 + Transformer 收益预测。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hmmlearn.hmm import GaussianHMM

from .data import FEATURE_COLS

# 恐慌性抛售硬覆盖 (逆向加仓层): 突破布林下轨 + 放量1.5倍 → 至少 110% 仓位
PANIC_BB_STD = 2.0            # 布林带下轨: 20日均线 - 2倍标准差
PANIC_VOL_MA_WINDOW = 20      # 成交量均值窗口
PANIC_VOLUME_SURGE_MULT = 1.5 # 成交量异常放大阈值 (当日量 > 20日均量 × 1.5)
PANIC_FLOOR_POSITION = 1.10   # 恐慌触发时的最低仓位 (逆向加仓搏反弹)

# 动态硬切换门控 (vol_target 模式): HMM 是无监督模型, 状态索引排序每次训练随机,
# 绝不硬编码 0/1/2 —— 由 state_position_map 动态定义 牛/熊/震荡 (仓位数最高=激进, 最低=避险)
VOL_TARGET_LOOKBACK = 20      # 滚动波动率窗口 (交易日)
VOL_TARGET_DEFAULT = 0.25     # 目标年化波动率 (中证500历史中枢约 25%~30%, 取 25%)
# 旧杠杆硬约束常量 (0.4/1.5) 已由新公式取代, 保留仅为兼容旧调用方
VOL_TARGET_LOWER = 0.4        # [弃用] 旧: 波动率直接定仓下限
VOL_TARGET_UPPER = 1.5        # [弃用] 旧: 波动率直接定仓上限

# 新公式 (2026-08, v3 动态硬切换门控)
VOL_BUDGET_LOWER = 0.5        # 波动率预算下限: 极端高波动时预算降至 50%
VOL_BUDGET_UPPER = 1.5        # 波动率预算上限: 极端低波动时预算至多 150%
REGIME_BULL_LOWER = 0.8       # 牛市门控: 完全听方向(无视波动率压制), 最低 80%
REGIME_BULL_UPPER = 1.5       # 牛市门控: 最高 150% 杠杆
REGIME_BEAR_LOWER = 0.3       # 熊市门控: 完全听波动率强制砍仓, 底线 30%
REGIME_BEAR_UPPER = 0.9       # 熊市门控: 波动率预算再高也 ≤ 90%
REGIME_NEUTRAL_CORE_W = 0.4   # 震荡门控: 方向权重 0.4
REGIME_NEUTRAL_VOL_W = 0.6    # 震荡门控: 波动率权重 0.6 (偏防守)
REGIME_NEUTRAL_LOWER = 0.4    # 震荡门控: 下限 40%
REGIME_NEUTRAL_UPPER = 1.1    # 震荡门控: 上限 110% (整体偏防守)
REGIME_COLD_LOWER = 0.6       # 冷启动保护(前20日 NaN): 强制中性 60%
REGIME_COLD_UPPER = 1.0       # 冷启动保护: 上限 100%, 避免误判熊市误砍仓
GATE_FINAL_LOWER = 0.3        # 最终输出截断下限
GATE_FINAL_UPPER = 1.5        # 最终输出截断上限


# ---------------------------------------------------------------------------
# HMM: 市场状态识别 (上涨/震荡/下跌)
# ---------------------------------------------------------------------------
class MarketHMM:
    """基于 hmmlearn 的高斯 HMM, 输出市场状态与状态概率。

    输入特征: 每日对数收益、波动率、动量。
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        # 状态标签: 按均值收益排序后映射, 0=低收益(熊), 1=中(震荡), 2=高(牛)
        self.state_labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        """X: (N, F) 特征矩阵, 用 [ret, vol_5, mom_20] 作为 HMM 观测。"""
        obs = X[:, [FEATURE_COLS.index("ret_1"),
                    FEATURE_COLS.index("vol_5"),
                    FEATURE_COLS.index("mom_20")]]
        # 处理 NaN(理论上已消除, 防御性填充)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=1e-4,
        )
        self.model.fit(obs)
        # 状态排序: 用每个状态观测的均值收益(第0维)排序, 0=熊, 末=牛
        means = self.model.means_[:, 0]
        order = np.argsort(means)
        self.state_labels_ = order
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, F), 返回 (N, n_states) 状态概率(按 熊/震荡/牛 排序)。"""
        obs = X[:, [FEATURE_COLS.index("ret_1"),
                    FEATURE_COLS.index("vol_5"),
                    FEATURE_COLS.index("mom_20")]]
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        prob = self.model.predict_proba(obs)  # (N, n_states) 原始顺序
        return prob[:, self.state_labels_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Transformer: 收益方向/幅度预测
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MarketTransformer(nn.Module):
    """轻量 Transformer Encoder 回归模型: 输入 (B, T, F) 输出 (B,) 下期收益预测。"""

    def __init__(
        self,
        n_features: int = len(FEATURE_COLS),
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x) * (self.d_model ** 0.5)
        h = self.pos_enc(h)
        h = self.encoder(h)
        # 取最后时间步(最新信息) 作为预测依据
        last = h[:, -1]                      # (B, d_model)
        return self.head(last).squeeze(-1)


def train_transformer(
    model: MarketTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """训练 Transformer。返回折内训练/验证指标。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device)

    n = len(X_t)
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_batches += 1
        scheduler.step()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_v), y_v).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} train_loss={epoch_loss / max(n, 1):.5f} val_loss={val_loss:.5f}")
        else:
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} train_loss={epoch_loss / max(n, 1):.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 指标: 方向准确率与 IC
    with torch.no_grad():
        pred_train = model(X_t).cpu().numpy()
        corr_train = float(np.corrcoef(pred_train, y_train)[0, 1]) if len(y_train) > 1 else 0.0
        acc_train = float(np.mean((pred_train > 0) == (y_train > 0)))
    metrics = {"train_acc": acc_train, "train_ic": corr_train, "train_loss": epoch_loss / max(n, 1)}
    if X_val is not None:
        with torch.no_grad():
            pred_val = model(X_v).cpu().numpy()
            corr_val = float(np.corrcoef(pred_val, y_val)[0, 1]) if len(y_val) > 1 else 0.0
            acc_val = float(np.mean((pred_val > 0) == (y_val > 0)))
        metrics.update({"val_acc": acc_val, "val_ic": corr_val, "val_loss": best_val_loss})
    return metrics


# ---------------------------------------------------------------------------
# 组合策略: HMM 状态门控 + Transformer 方向
# ---------------------------------------------------------------------------
class HMMTransformerStrategy:
    """综合策略。

    规则:
      1. HMM 输出当日市场状态概率, 定义牛态概率 bull_prob = P(牛|数据)。
      2. Transformer 输出下日收益预测 pred。
      3. 信号 = pred * (bull_prob - 0.5) * 2  (方向加权)
      4. 持仓: 信号 > 0 且 HMM 非熊 → 做多; 信号显著 < 0 → 空仓。
      5. 仓位 = min(|信号| / 阈值, 1) 缩放, 施加转换壁垒避免抖动。
    """

    def __init__(
        self,
        signal_threshold: float = 0.25,
        hold_threshold: float = 0.1,
        max_position: float = 1.2,
        transaction_cost: float = 0.0005,
        transformer_kwargs: dict | None = None,
        state_position_map: tuple = (1.2, 0.8, 1.0),
        duration_penalty: list | None = None,
        vol_target: float | None = None,
        vol_lookback: int = VOL_TARGET_LOOKBACK,
        vol_lower: float = VOL_TARGET_LOWER,
        vol_upper: float = VOL_TARGET_UPPER,
    ):
        """state_position_map: 对应 HMM 状态 [熊市, 震荡, 牛市] 的目标仓位。

        反转映射:
          熊市 → 120% 杠杆 (逆势抄底)
          震荡 → 80% 仓位
          牛市 → 100% 仓位

        duration_penalty: 熊市状态持续期惩罚因子, 形如 [(天数上限, 仓位), ...]。
          熊市持续天数 ≤ 上限 → 对应仓位; 超过最后一个上限 → 使用最后档仓位。
          例: [(20, 1.20), (40, 0.80)] → 首月 120%, 次月 80%, 之后 50%(末档兜底)。
          为 None 时不做时间衰减, 维持 state_position_map 固定仓位。

        vol_target: 动态硬切换门控 (hard-switch regime gating)。
          HMM 无监督 → 状态索引每次训练排列随机, 绝不硬编码 0/1/2;
          由 state_position_map 动态定义 牛/熊/震荡 (系数最高=激进做多, 最低=避险):
            牛市 → 完全听方向: clip(core_position, 0.8, 1.5)      (无视波动率压制)
            熊市 → 完全听波动率: clip(vol_budget, 0.3, 0.9)        (强制砍仓保命)
            震荡 → 4:6 加权: clip(0.4×core + 0.6×vol_budget, 0.4, 1.1)
          冷启动(前20日 realized_vol=NaN) → 强制中性 clip(core, 0.6, 1.0)。
          最终输出 clip(0.3, 1.5); None 时不启用。
          (vol_lower/vol_upper 已由 REGIME_*/GATE_* 约束取代, 保留仅为兼容)
        """
        self.signal_threshold = signal_threshold
        self.hold_threshold = hold_threshold
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.transformer_kwargs = transformer_kwargs or {}
        self.state_position_map = state_position_map
        self.duration_penalty = duration_penalty
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.vol_lower = vol_lower
        self.vol_upper = vol_upper
        self.hmm = MarketHMM()
        self.transformer = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, device: str = "cpu", seed: int = 0,
            verbose: bool = False, epochs: int = 30, batch_size: int = 64,
            lr: float = 1e-3, weight_decay: float = 1e-4):
        """X: (N, T, F) 序列; y: (N,) 下期收益。训练 HMM + Transformer。"""
        # HMM 用最后一时间步特征(当日特征)
        X_last = X[:, -1, :]
        self.hmm.fit(X_last)
        # 加入 HMM 状态概率作为额外输入特征
        prob = self.hmm.predict_proba(X_last)  # (N, n_states)
        # 直接把概率拼接进每个时间步(简化: 复制到所有时间步)
        n_states = prob.shape[1]
        prob_exp = np.repeat(prob[:, np.newaxis, :], X.shape[1], axis=1)  # (N, T, n_states)
        X_aug = np.concatenate([X, prob_exp], axis=-1)  # (N, T, F+n_states)

        self.transformer = MarketTransformer(
            n_features=X_aug.shape[-1],
            **self.transformer_kwargs,
        )
        if X_val is not None:
            prob_v = self.hmm.predict_proba(X_val[:, -1, :])
            prob_v_exp = np.repeat(prob_v[:, np.newaxis, :], X_val.shape[1], axis=1)
            X_val_aug = np.concatenate([X_val, prob_v_exp], axis=-1)
        else:
            X_val_aug = None

        metrics = train_transformer(
            self.transformer, X_aug, y, X_val_aug, y_val,
            device=device, seed=seed, verbose=verbose,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        )
        self._fitted = True
        return metrics

    def predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X: (N, T, F) → (transformer_pred, bull_prob)"""
        prob = self.hmm.predict_proba(X[:, -1, :])
        bull_prob = prob[:, -1]  # 牛态概率(状态排序后最后一个)
        n_states = prob.shape[1]
        prob_exp = np.repeat(prob[:, np.newaxis, :], X.shape[1], axis=1)
        X_aug = np.concatenate([X, prob_exp], axis=-1)
        self.transformer.eval()
        with torch.no_grad():
            pred = self.transformer(torch.tensor(X_aug, dtype=torch.float32)).numpy()
        return pred, bull_prob

    def positions(self, X: np.ndarray, initial_position: float = 0.0,
                  pe_pct: np.ndarray | None = None,
                  closes: np.ndarray | None = None,
                  volumes: np.ndarray | None = None) -> np.ndarray:
        """给定全部特征序列 X, 返回每日目标仓位。

        状态反转映射 (state-driven) + 可选持续期惩罚 + 可选 PE 估值硬覆盖:
          HMM 三态概率 prob = [熊市, 震荡, 牛市] (按收益排序)。

          无 duration_penalty 时 (默认):
            target = Σ prob_i * state_position_map_i
            熊市 → 1.20 / 震荡 → 0.80 / 牛市 → 1.00

          启用 duration_penalty 时 (如 CSI300 救赎方案):
            取每日最大概率状态 s; 若 s == 熊市, 统计连续熊市天数 n:
              n ≤ 20  → 120%  (第一波恐慌盘, 保持抄底杠杆)
              20 < n ≤ 40 → 80%  (中期降杠杆防阴跌)
              n > 40   → 50%  (磨底期保留现金等政策), 由末档兜底
            非熊市状态用 state_position_map 固定仓位 (震荡 80% / 牛市 100%)。

          启用 pe_pct (沪深300 PE_TTM 滚动10年分位数, 与 X 等长) 时,
          在 duration_penalty 输出之上叠加硬覆盖:
           pe_pct < 0.30 → 最低仓位提升至 0.80 (极度低估区, 无视持续惩罚)
           pe_pct < 0.50 → 最低仓位提升至 0.65 (中等估值区)
           pe_pct ≥ 0.50 → 完全交由 duration_penalty (最低 0.50)

          动态硬切换门控 (启用 vol_target 时): 由 state_position_map 动态定义牛/熊/震荡,
          绝不硬编码状态索引 (HMM 无监督, 索引顺序每次训练随机):
            1. bull_regime = argmax(state_position_map)    (激进做多)
               bear_regime = argmin(state_position_map)    (极端避险)
               neutral_regime = 3 - bull - bear            (仅适配 3 状态)
            2. 方向性核心仓位 core_position (HMM 概率加权 + PE + 恐慌, 不含 duration_penalty)
            3. 波动率预算 vol_budget = clip(vol_target / realized_vol, 0.5, 1.5)
            4. 每日主宰状态 current_regime = argmax(prob), 硬切换:
               牛市 → clip(core_position, 0.8, 1.5)          [完全听方向]
               熊市 → clip(vol_budget, 0.3, 0.9)             [完全听波动率]
               震荡 → clip(0.4×core + 0.6×vol_budget, 0.4, 1.1)
            5. 冷启动(前20日 realized_vol=NaN) → clip(core_position, 0.6, 1.0)
            6. 最终输出 clip(0.3, 1.5)
          恐慌性抛售硬覆盖作为核心仓位的一部分叠加生效 (急跌放量时 max(仓位, 110%))。

          目标经 max_position 上限裁剪; 变更超过 hold_threshold 才切换, 降低换手。
        """
        prob = self.hmm.predict_proba(X[:, -1, :])          # (N, 3) [熊, 震荡, 牛]
        states = np.argmax(prob, axis=1)                    # 0=熊, 1=震荡, 2=牛

        # ---- 恐慌性抛售检测 (逆向硬覆盖前置): 跌破布林下轨 + 放量异常 ----
        panic_mask = None
        if closes is not None and volumes is not None:
            close_s = pd.Series(closes)
            vol_s = pd.Series(volumes)
            sma20 = close_s.rolling(PANIC_VOL_MA_WINDOW, min_periods=PANIC_VOL_MA_WINDOW).mean().to_numpy()
            std20 = close_s.rolling(PANIC_VOL_MA_WINDOW, min_periods=PANIC_VOL_MA_WINDOW).std().to_numpy()
            bb_lower = sma20 - PANIC_BB_STD * std20
            ma_vol20 = vol_s.rolling(PANIC_VOL_MA_WINDOW, min_periods=PANIC_VOL_MA_WINDOW).mean().to_numpy()
            # numpy: 与 NaN 比较恒为 False → 冷启动(不足20日)自动不触发恐慌
            panic_mask = (closes < bb_lower) & (volumes > ma_vol20 * PANIC_VOLUME_SURGE_MULT)

        # ---- 动态硬切换门控 (vol_target 模式): HMM 无监督 → 由 state_position_map 动态定义牛/熊/震荡 ----
        if self.vol_target is not None:
            if closes is None:
                raise ValueError("启用 vol_target 时必须提供 closes 收盘价序列")
            # 1. 方向性核心仓位: 来自 HMM + PE + 恐慌覆盖, 不含 duration_penalty
            map_arr = np.array(self.state_position_map, dtype=np.float64)
            core_position = prob @ map_arr                       # HMM 状态概率加权 (核心仓位)
            # PE 估值硬覆盖: 分位数越低, 最低核心仓位门槛越高
            if pe_pct is not None:
                if len(pe_pct) != len(core_position):
                    raise ValueError(f"pe_pct 长度 {len(pe_pct)} != 仓位 {len(core_position)}")
                floor = np.where(pe_pct < 0.30, 0.80,
                                 np.where(pe_pct < 0.50, 0.65, 0.50))
                core_position = np.maximum(core_position, floor)
            # 恐慌性抛售逆向硬覆盖: 越跌越买, 核心仓位最低 110%
            if panic_mask is not None:
                core_position[panic_mask] = np.maximum(core_position[panic_mask],
                                                       PANIC_FLOOR_POSITION)
            # 2. 波动率预算: 目标波动 / 实现波动, 限 [0.5, 1.5]
            close_s = pd.Series(closes)
            daily_ret = close_s.pct_change()
            # 滚动 vol_lookback(默认20)日收益率标准差 → 年化波动率
            realized_vol = daily_ret.rolling(self.vol_lookback, min_periods=self.vol_lookback).std() * np.sqrt(252)
            realized_vol = realized_vol.to_numpy()
            # 冷启动(NaN)时预算保持 1.0
            raw_budget = np.where(np.isfinite(realized_vol) & (realized_vol > 0),
                                  self.vol_target / realized_vol,
                                  1.0)
            vol_budget = np.clip(raw_budget, VOL_BUDGET_LOWER, VOL_BUDGET_UPPER)
            # 3. 动态映射牛/熊/震荡索引 (绝不硬编码 0/1/2):
            #    仓位系数最高 = 激进做多(bull), 最低 = 极端避险(bear), 剩余 = 震荡
            #    (该映射在每折训练后都会随 HMM 状态索引重新推导, 天然适应随机排列)
            bull_regime = int(np.argmax(map_arr))
            bear_regime = int(np.argmin(map_arr))
            neutral_regime = 3 - bull_regime - bear_regime        # 仅适配 3 状态
            # 4. 每日主宰状态 = argmax(HMM 概率), 按状态硬切换门控
            current_regime = np.argmax(prob, axis=1)
            bull_mask = current_regime == bull_regime
            bear_mask = current_regime == bear_regime
            neutral_mask = ~(bull_mask | bear_mask)
            cold_mask = ~np.isfinite(realized_vol)                # 冷启动保护(前20日)
            final_position = np.empty_like(core_position)
            # 牛市/趋势上涨: 完全听方向, 无视波动率压制, 满仓甚至加杠杆 80%~150%
            act = bull_mask & ~cold_mask
            final_position[act] = np.clip(core_position[act], REGIME_BULL_LOWER, REGIME_BULL_UPPER)
            # 熊市/高波动暴跌: 完全听波动率, 强制砍仓保命 30%~90%
            act = bear_mask & ~cold_mask
            final_position[act] = np.clip(vol_budget[act], REGIME_BEAR_LOWER, REGIME_BEAR_UPPER)
            # 震荡市: 方向与波动率 4:6 加权, 偏防守, 总仓 ≤110%
            act = neutral_mask & ~cold_mask
            final_position[act] = np.clip(REGIME_NEUTRAL_CORE_W * core_position[act]
                                          + REGIME_NEUTRAL_VOL_W * vol_budget[act],
                                          REGIME_NEUTRAL_LOWER, REGIME_NEUTRAL_UPPER)
            # 冷启动保护: 前 20 日 realized_vol=NaN → 强制中性仓位, 避免误判熊市误砍仓
            final_position[cold_mask] = np.clip(core_position[cold_mask],
                                                REGIME_COLD_LOWER, REGIME_COLD_UPPER)
            # 5. 最终输出截断 [0.3, 1.5]
            soft_target = np.clip(final_position, GATE_FINAL_LOWER, GATE_FINAL_UPPER)
        else:
            # ---- 原方向模式: HMM → duration_penalty → PE → 恐慌 → max_position 裁剪 ----
            if self.duration_penalty is None:
                weights = np.array(self.state_position_map, dtype=np.float64)
                soft_target = prob @ weights                    # 概率加权目标仓位
            else:
                # 按档位排序 [(上限, 仓位), ...]
                tiers = sorted(self.duration_penalty)
                fallback_pos = tiers[-1][1]                     # 超过最长档 → 末档仓位
                soft_target = np.empty(len(states), dtype=np.float64)
                bear_days = 0
                for i, s in enumerate(states):
                    if s == 0:
                        bear_days += 1
                        target = fallback_pos
                        for limit, pos_ in tiers:
                            if bear_days <= limit:
                                target = pos_
                                break
                        # 恐慌性抛售是"急跌"非"阴跌": 重设持续期计数器, 时间惩罚归零;
                        # 且按需求 max(final_position, 1.10) 强制至少 110%, 不覆盖更高仓位
                        if panic_mask is not None and panic_mask[i]:
                            bear_days = 0
                            target = max(target, PANIC_FLOOR_POSITION)
                    else:
                        bear_days = 0
                        target = self.state_position_map[s]
                    soft_target[i] = target
                # PE 估值硬覆盖: 分位数越低, 最低仓位门槛越高 (成长股价值底)
                if pe_pct is not None:
                    if len(pe_pct) != len(soft_target):
                        raise ValueError(f"pe_pct 长度 {len(pe_pct)} != 仓位 {len(soft_target)}")
                    floor = np.where(pe_pct < 0.30, 0.80,
                                     np.where(pe_pct < 0.50, 0.65, 0.50))
                    soft_target = np.maximum(soft_target, floor)

            # 恐慌性抛售逆向硬覆盖: 无视 HMM 状态/持续期惩罚, 越跌越买
            if panic_mask is not None:
                soft_target[panic_mask] = np.maximum(soft_target[panic_mask],
                                                     PANIC_FLOOR_POSITION)

            soft_target = np.clip(soft_target, 0.0, self.max_position)

        # 滞回: 目标偏离当前仓位超过 hold_threshold 才切换
        pos = np.empty_like(soft_target)
        cur = initial_position
        for i, t in enumerate(soft_target):
            if abs(t - cur) > self.hold_threshold:
                cur = t
            pos[i] = cur
        return pos


# ---------------------------------------------------------------------------
# 风险平价 + 资产轮动 (2026-08 v4: 彻底放弃 HMM/Transformer 方向择时)
# ---------------------------------------------------------------------------
class RiskParityStrategy:
    """风险平价 + 资产轮动 (波动率目标 + PE估值调节 + 国债ETF轮动)。

    核心逻辑是完全确定性的公式, 不包含任何 HMM/Transformer 方向预测:

      1. realized_vol = closes.pct_change().rolling(20).std() * sqrt(252)
      2. dynamic_target = f(pe_percentile)
         PE 分位越低(低估) → 目标波动率越高 (敢加杠杆吃大涨);
         PE 分位越高(高估) → 目标波动率越低 (强制保守躲大跌)。
         锚点: pe=0.0 → 20%, pe=0.3 → 18%, pe=0.7 → 12%, pe=1.0 → 10%
      3. stock_weight = clip(dynamic_target / realized_vol, 0.3, 1.2)
      4. bond_weight  = clip(1 - stock_weight, 0.0, 0.7)   # 剩余资金买国债ETF
      5. 组合收益 = stock_weight × 股票收益 + bond_weight × 国债收益

    该策略无需训练, 直接对价格/估值序列调用 positions() 得到每日权重。
    """

    def __init__(
        self,
        vol_lookback: int = 20,          # 实际波动率滚动窗口 (交易日)
        vol_annual: int = 252,           # 年化交易日数
        pe_anchors: list | None = None,  # [(PE分位, 目标波动率)] 分段线性锚点
        stock_weight_lower: float = 0.3, # 股票权重下限 (极端高波动留30%)
        stock_weight_upper: float = 1.2, # 股票权重上限 (极端低波动可加杠杆)
        bond_weight_lower: float = 0.0,  # 债券权重下限
        bond_weight_upper: float = 0.7,  # 债券权重上限 (股票跌破30%时最多70%债券)
        hold_threshold: float = 0.05,    # 滞回门槛: 目标偏离当前仓位超此值才切换, 降低换手
        transaction_cost: float = 0.0005 # 单边换仓成本
    ):
        if pe_anchors is None:
            # 默认锚点 (贴合蓝图正文):
            #   PE < 30% → 目标波动率 18%~20% (低估, 敢承担风险)
            #   PE > 70% → 目标波动率 10%~12% (高估, 强制保守)
            pe_anchors = [(0.00, 0.20), (0.30, 0.18), (0.70, 0.12), (1.00, 0.10)]
        self.vol_lookback = vol_lookback
        self.vol_annual = vol_annual
        self.pe_anchors = np.array(sorted(pe_anchors), dtype=np.float64)
        self.stock_weight_lower = stock_weight_lower
        self.stock_weight_upper = stock_weight_upper
        self.bond_weight_lower = bond_weight_lower
        self.bond_weight_upper = bond_weight_upper
        self.hold_threshold = hold_threshold
        self.transaction_cost = transaction_cost
        self._fitted = True  # 纯规则策略, 无需训练

    # -- 纯规则 API, 与 HMMTransformerStrategy.fit 兼容 (空操作) ----
    def fit(self, *args, **kwargs) -> dict:
        """纯确定性规则, 无需训练。返回空指标。"""
        self._fitted = True
        return {"train_acc": float("nan"), "train_ic": float("nan")}

    # ---------------------------------------------------------------
    def dynamic_target_vol(self, pe_pct: np.ndarray) -> np.ndarray:
        """PE 分位数 → 动态目标年化波动率 (分段线性插值, 0.10~0.20)。"""
        pe_pct = np.asarray(pe_pct, dtype=np.float64)
        pe_pct = np.clip(np.nan_to_num(pe_pct, nan=0.5), 0.0, 1.0)
        return np.interp(pe_pct, self.pe_anchors[:, 0], self.pe_anchors[:, 1])

    def realized_vol(self, closes: np.ndarray) -> np.ndarray:
        """过去 vol_lookback 日实际年化波动率 (冷启动前20日返回 NaN)。"""
        close_s = pd.Series(closes)
        daily_ret = close_s.pct_change()
        vol = daily_ret.rolling(self.vol_lookback, min_periods=self.vol_lookback).std() * np.sqrt(self.vol_annual)
        return vol.to_numpy()

    def positions(
        self,
        closes: np.ndarray,
        pe_pct: np.ndarray | None = None,
        initial_position: float = 0.0,
        warmup: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算每日股票/债券权重。

        参数:
            closes: (N,) 股票(如沪深300)收盘价序列。
            pe_pct: (N,) 股票指数 PE_TTM 滚动10年分位数 (0~1); None=中性 0.5。
            initial_position: 初始股票仓位 (滞回起点)。
            warmup: 前 warmup 日不计收益 (冷启动保护, 与 WalkForward 对齐)。

        返回:
            (stock_weight, bond_weight) 两个 (N,) float 数组。
        """
        if closes is None or len(closes) == 0:
            raise ValueError("RiskParityStrategy.positions 必须提供 closes 收盘价序列")
        n = len(closes)
        if pe_pct is None:
            pe_pct = np.full(n, 0.5, dtype=np.float64)
        if len(pe_pct) != n:
            raise ValueError(f"pe_pct 长度 {len(pe_pct)} != closes 长度 {n}")

        vol = self.realized_vol(closes)                       # (N,)
        target = self.dynamic_target_vol(pe_pct)              # (N,) 0.10~0.20
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(np.isfinite(vol) & (vol > 0), target / vol, np.nan)
        # 冷启动 (前20日 NaN) → 中性权重 1.0 (之后由滞回平滑接管)
        raw = np.where(np.isfinite(raw), raw, 1.0)
        stock_t = np.clip(raw, self.stock_weight_lower, self.stock_weight_upper)
        bond_t = np.clip(1.0 - stock_t, self.bond_weight_lower, self.bond_weight_upper)

        # 滞回: 目标偏离当前仓位超过 hold_threshold 才切换 (降低换手)
        sw = np.empty_like(stock_t)
        cur = initial_position
        for i, t in enumerate(stock_t):
            if abs(t - cur) > self.hold_threshold:
                cur = t
            sw[i] = cur
        bw = np.clip(1.0 - sw, self.bond_weight_lower, self.bond_weight_upper)
        return sw, bw
