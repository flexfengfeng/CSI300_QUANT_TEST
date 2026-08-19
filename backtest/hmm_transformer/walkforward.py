"""WalkForward 滚动训练-验证-测试框架 + 回测引擎 (支持多进程并行折)。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .models import HMMTransformerStrategy, RiskParityStrategy

BACKTEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# worker 进程共享全局 (Pool initializer 填充)
_WORKER = {}


@dataclass
class WalkForwardConfig:
    """WalkForward 配置。训练/测试均以「交易日」为单位。"""
    train_days: int = 1260      # 训练窗口 ~5 年
    val_days: int = 252         # 验证窗口 ~1 年 (用于早停选择最优 epoch)
    test_days: int = 252        # 每折测试窗口 ~1 年
    step_days: int = 63         # 滚动步长 ~1 季度 (测试窗重叠, 拼接时新折覆盖旧折)
    warmup_days: int = 60       # 每折测试期前 60 天作为模型预热(不计入回测收益)

    # 策略参数 (反转映射: 熊市120%杠杆, 震荡80%, 牛市100%)
    signal_threshold: float = 0.25
    hold_threshold: float = 0.1
    transaction_cost: float = 0.0005
    max_position: float = 1.2
    state_position_map: tuple = (1.2, 0.8, 1.0)
    # 状态持续期惩罚: [(天数上限, 仓位), ...]; None=不启用
    # 例: 熊市 ≤20日→120%(首波恐慌), ≤40日→80%(防阴跌), >40日→50%(磨底留现金)
    duration_penalty: list | None = None
    # 动态硬切换门控: HMM无监督→由 state_position_map 动态定义牛/熊/震荡 (绝不硬编码索引).
    # 牛市=完全听方向 clip(core,0.8,1.5); 熊市=完全听波动率 clip(vol_budget,0.3,0.9);
    # 震荡=4:6加权 clip(0.4×core+0.6×vol_budget,0.4,1.1); 冷启动=中性 clip(core,0.6,1.0);
    # 最终 clip(0.3,1.5); 波动率预算=clip(Target/Realized,0.5,1.5); None=不启用
    vol_target: float | None = None
    vol_lookback: int = 20
    vol_lower: float = 0.4
    vol_upper: float = 1.5

    # Transformer 参数
    transformer_kwargs: dict = field(default_factory=lambda: {
        "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1,
    })
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4

    device: str = "cpu"          # 并行模式下 worker 自动降级为 cpu
    seed: int = 42
    verbose: bool = False
    num_workers: int = 0         # 并行折数; 0=自动(None->min(8, cpu)), 1=串行
    # 沪深300 PE_TTM 滚动10年分位数 (与 X 等长); None=不启用估值硬覆盖
    pe_pct_series: np.ndarray | None = None
    # 恐慌性抛售逆向硬覆盖 (越跌越买); False=关闭(用于 A/B 对比)
    enable_panic_override: bool = True


def walkforward_split(n: int, cfg: WalkForwardConfig):
    """生成 (train_start, train_end, val_start, val_end, test_start, test_end) 折序列。

    锚定式滚动: 训练窗始终从 0 到 train_days, 验证窗紧随其后, 测试窗每次前移 step_days。
    测试窗之间允许重叠(步长 < 窗口), 拼接时后折覆盖前折。
    [start, end) 左闭右开。
    """
    folds = []
    test_start = cfg.train_days + cfg.val_days
    while test_start + cfg.test_days <= n:
        folds.append((
            0, cfg.train_days,
            cfg.train_days, cfg.train_days + cfg.val_days,
            test_start, test_start + cfg.test_days,
        ))
        test_start += cfg.step_days
    if test_start < n - cfg.warmup_days:
        folds.append((
            0, cfg.train_days,
            cfg.train_days, cfg.train_days + cfg.val_days,
            test_start, n,
        ))
    return folds


# ---------------------------------------------------------------------------
# 单折执行 (供多进程 worker 调用, 必须模块级可 pickle)
# ---------------------------------------------------------------------------
def _init_worker(shared: dict):
    """Pool initializer: 在 worker 进程内建立共享数据与配置。"""
    _WORKER.update(shared)


def _run_fold(fi: int):
    """在 worker 进程内执行第 fi 折: 训练 HMM+Transformer, 对测试窗生成仓位与收益。

    返回 (fi, rows, metrics)。rows 为该折测试期每日记录。
    """
    X = _WORKER["X"]
    y = _WORKER["y"]
    dates = _WORKER["dates"]
    closes = _WORKER.get("closes", None)     # (N,) 原始收盘价
    volumes = _WORKER.get("volumes", None)   # (N,) 原始成交量或 None
    cfg = _WORKER["cfg"]
    folds = _WORKER["folds"]
    pe_pct = _WORKER.get("pe_pct", None)   # 全量分位数数组或 None

    tr_s, tr_e, va_s, va_e, te_s, te_e = folds[fi]

    # 并行场景 worker 内用 cpu (MPS 多进程不稳定); 串行用原配置
    device = cfg.device if cfg.num_workers == 1 else "cpu"

    strategy = HMMTransformerStrategy(
        signal_threshold=cfg.signal_threshold,
        hold_threshold=cfg.hold_threshold,
        max_position=cfg.max_position,
        transaction_cost=cfg.transaction_cost,
        transformer_kwargs=cfg.transformer_kwargs,
        state_position_map=cfg.state_position_map,
        duration_penalty=cfg.duration_penalty,
        vol_target=cfg.vol_target,
        vol_lookback=cfg.vol_lookback,
        vol_lower=cfg.vol_lower,
        vol_upper=cfg.vol_upper,
    )

    metrics = strategy.fit(
        X[tr_s:tr_e], y[tr_s:tr_e],
        X[va_s:va_e], y[va_s:va_e],
        device=device, seed=cfg.seed + fi, verbose=cfg.verbose,
        epochs=cfg.epochs, batch_size=cfg.batch_size,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    metrics["fold"] = fi + 1
    metrics["test_start"] = pd.Timestamp(dates[te_s])
    metrics["test_end"] = pd.Timestamp(dates[te_e - 1])

    # 测试窗: 含 warmup 前缀保证仓位状态连续, 收益只取 warmup 之后
    test_window = X[te_s - cfg.warmup_days:te_e]
    pred, bull = strategy.predict_components(test_window)
    # 若提供 PE 分位数序列, 截取对应窗口传入 positions() 做估值硬覆盖
    pe_window = pe_pct[te_s - cfg.warmup_days:te_e] if pe_pct is not None else None
    # 恐慌性抛售硬覆盖需要原始收盘价/成交量 (与 test_window 对齐);
    # 波动率目标硬覆盖仅需收盘价。两者任一启用且有数据时传入 closes。
    use_panic = cfg.enable_panic_override and closes is not None and volumes is not None
    use_vol_target = cfg.vol_target is not None
    # 传入收盘价的条件: 恐慌覆盖 或 波动率目标 (两者都需要 closes)
    need_closes = use_panic or use_vol_target
    close_window = closes[te_s - cfg.warmup_days:te_e] if (need_closes and closes is not None) else None
    vol_window = volumes[te_s - cfg.warmup_days:te_e] if use_panic else None
    test_positions = strategy.positions(test_window, initial_position=0.0,
                                        pe_pct=pe_window,
                                        closes=close_window, volumes=vol_window)

    keep = cfg.warmup_days
    p_keep = test_positions[keep:]
    pred_keep = pred[keep:]
    bull_keep = bull[keep:]
    y_keep = y[te_s:te_e]

    # 收益: 当日仓位 × 下日收益 - 换仓成本×|Δ仓位|
    pos_prev = np.concatenate([[0.0], p_keep[:-1]])
    turnover = np.abs(p_keep - pos_prev)
    ret_keep = p_keep * y_keep - cfg.transaction_cost * turnover

    rows = [(dates[te_s + i], p_keep[i], pred_keep[i], bull_keep[i],
             ret_keep[i], y_keep[i]) for i in range(len(y_keep))]

    metrics.update({
        "test_pos_mean": float(np.mean(p_keep)),
        "test_ret_sum": float(np.sum(ret_keep)),
        "test_acc": float(np.mean((pred_keep > 0) == (y_keep > 0))),
    })
    return fi, rows, metrics


def run_walkforward(
    X: np.ndarray,             # (N, T, F) 特征序列
    y: np.ndarray,             # (N,) 下期对数收益
    dates: np.ndarray,         # (N,) 预测日
    closes: np.ndarray,        # (N,) 当日收盘价
    next_closes: np.ndarray,   # (N,) 下一交易日收盘价
    cfg: WalkForwardConfig,
    volumes: np.ndarray | None = None,   # (N,) 当日成交量; None=无成交量(自动跳过恐慌硬覆盖)
) -> dict:
    """执行 WalkForward 回测。

    每折独立训练+预测, 支持多进程并行 (num_workers>1)。
    各折结果按日期拼接, 重叠部分由更晚的折覆盖。
    """
    t0 = time.time()
    folds = walkforward_split(len(X), cfg)

    # 并行度: 0=自动, 1=串行
    if cfg.num_workers == 0:
        num_workers = min(8, os.cpu_count() or 1, len(folds))
    else:
        num_workers = cfg.num_workers

    shared = {
        "X": X, "y": y, "dates": dates, "cfg": cfg, "folds": folds,
        "pe_pct": getattr(cfg, "pe_pct_series", None),
        "closes": closes,
        "volumes": volumes,
    }

    fold_results = []   # (fi, rows, metrics)
    if num_workers <= 1:
        _init_worker(shared)
        for fi in range(len(folds)):
            print(f"=== Fold {fi + 1}/{len(folds)} | "
                  f"train {pd.Timestamp(dates[folds[fi][0]]).date()}~"
                  f"{pd.Timestamp(dates[folds[fi][1] - 1]).date()} | "
                  f"test {pd.Timestamp(dates[folds[fi][4]]).date()}~"
                  f"{pd.Timestamp(dates[folds[fi][5] - 1]).date()} ===")
            res = _run_fold(fi)
            fold_results.append(res)
            print(f"  fold val_acc={res[2].get('val_acc', float('nan')):.3f} "
                  f"test_ret_sum={res[2]['test_ret_sum']:.4f}")
    else:
        print(f"并行 WalkForward: {len(folds)} 折 × {num_workers} workers (设备: cpu)")
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(shared,)) as pool:
            for fi, rows, metrics in pool.imap_unordered(_run_fold, range(len(folds))):
                fold_results.append((fi, rows, metrics))
                print(f"  Fold {fi + 1} 完成 | val_acc={metrics.get('val_acc', float('nan')):.3f} "
                      f"test_ret_sum={metrics['test_ret_sum']:.4f}")

    # 按折序号排序, 保证拼接时新的折覆盖旧的
    fold_results.sort(key=lambda t: t[0])
    fold_metrics = [m for _, _, m in fold_results]

    # 组装回测 DataFrame: 按日期排序, 重叠日由更晚的折覆盖
    all_rows = []
    for _, rows, _ in fold_results:
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows, columns=[
        "date", "position", "transformer_pred", "bull_prob", "strategy_ret", "market_ret",
    ])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

    result = compute_metrics(df)
    result["folds"] = fold_metrics
    result["df"] = df
    result["elapsed_sec"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# 风险平价 + 资产轮动 (无 HMM/Transformer, 纯确定性规则回测)
# ---------------------------------------------------------------------------
def run_riskparity(
    dates: np.ndarray,             # (N,) 交易日
    stock_closes: np.ndarray,      # (N,) 股票(如沪深300)收盘价
    bond_closes: np.ndarray,       # (N,) 债券(如国债ETF 511010)收盘价
    pe_pct: np.ndarray | None = None,   # (N,) 股票 PE 分位数; None=中性 0.5
    strategy: RiskParityStrategy | None = None,
    warmup: int = 60,              # 前 warmup 日为仓位预热, 不计入回测收益
    transaction_cost: float = 0.0005,
) -> dict:
    """执行风险平价回测: 返回与 compute_metrics 兼容的结果 dict。

    收益合成 (蓝图公式 5 行核心代码):
      stock_ret[i] = stock_closes[i+1] / stock_closes[i] - 1
      bond_ret[i]  = bond_closes[i+1]  / bond_closes[i]  - 1
      strategy_ret[i] = stock_weight[i] × stock_ret[i] + bond_weight[i] × bond_ret[i]

    基准 market_ret = stock_ret (沪深300 买入持有)。
    """
    if strategy is None:
        strategy = RiskParityStrategy(transaction_cost=transaction_cost)

    n = len(stock_closes)
    if len(bond_closes) != n:
        raise ValueError(f"bond_closes 长度 {len(bond_closes)} != stock_closes 长度 {n}")
    if len(dates) != n:
        raise ValueError(f"dates 长度 {len(dates)} != stock_closes 长度 {n}")

    sw, bw = strategy.positions(stock_closes, pe_pct=pe_pct, initial_position=1.0)

    # 下日收益 (最后一天无下一日收益 → 0)
    stock_ret = np.zeros(n)
    bond_ret = np.zeros(n)
    if n > 1:
        stock_ret[:-1] = stock_closes[1:] / stock_closes[:-1] - 1.0
        bond_ret[:-1] = bond_closes[1:] / bond_closes[:-1] - 1.0

    gross = sw * stock_ret + bw * bond_ret

    # 换手成本: 当日权重相对前一日的变化 (含从预热期到首个收益日的切换)
    prev_sw = np.concatenate([[sw[max(0, warmup - 1)]], sw[warmup:-1]])
    turnover = np.abs(sw[warmup:] - prev_sw)
    net_ret = gross[warmup:] - transaction_cost * turnover

    df = pd.DataFrame({
        "date": dates[warmup:],
        "position": sw[warmup:],
        "bond_weight": bw[warmup:],
        "strategy_ret": net_ret,
        "market_ret": stock_ret[warmup:],
    })

    result = compute_metrics(df)
    result["df"] = df
    result["avg_bond_weight"] = float(bw[warmup:].mean())
    result["start_date"] = result["start_date"]
    return result


def save_riskparity_report(result: dict, ticker: str = "CSI300",
                           out_dir: str | None = None) -> str:
    """输出风险平价回测报告 (CSV + TXT)。返回报告基路径。"""
    out_dir = out_dir or backtest_dir()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"riskparity_{ticker.replace('.', '_')}")

    df = result["df"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.to_csv(base + "_daily.csv", index=False)

    lines = [f"===== 风险平价 + 资产轮动 回测报告 | {ticker} ====="]
    lines.append(f"回测区间: {pd.Timestamp(result['start_date']).date()} ~ "
                 f"{pd.Timestamp(result['end_date']).date()} "
                 f"({result['days']} 日, {result['years']:.1f} 年)")
    lines.append("")
    lines.append("---- 组合 (股票×波动率目标 + 国债ETF) ----")
    for k, v in result["strategy"].items():
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append("  胜率: {:.4f}".format(result["win_rate"]))
    lines.append("  平均股票仓位: {:.4f}".format(result["avg_position"]))
    lines.append("  平均债券仓位: {:.4f}".format(result.get("avg_bond_weight", float("nan"))))
    lines.append("  平均换手(日): {:.4f}".format(result["avg_turnover"]))
    lines.append("  超额收益(组合 vs 沪深300): {:+.4f}".format(result["excess_return"]))
    lines.append("")
    lines.append("---- 基准 (沪深300 买入持有) ----")
    for k, v in result["market"].items():
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report = "\n".join(lines)
    print(report)
    with open(base + "_report.txt", "w") as f:
        f.write(report + "\n")
    return base


def compute_metrics(df: pd.DataFrame) -> dict:
    """计算回测指标: CAGR, 最大回撤, 夏普, 索提诺, 胜率, 换手率等。"""
    if len(df) < 2:
        return {"error": "insufficient data"}

    df = df.copy()
    df["strategy_eq"] = np.exp(df["strategy_ret"].cumsum())
    df["market_eq"] = np.exp(df["market_ret"].cumsum())

    n = len(df)
    years = n / 252

    def _stats(eq: np.ndarray, ret: np.ndarray) -> dict:
        cagr = eq[-1] ** (1 / years) - 1 if years > 0 else np.nan
        roll_max = np.maximum.accumulate(eq)
        drawdown = eq / roll_max - 1
        mdd = float(drawdown.min())
        ann_vol = float(np.std(ret) * np.sqrt(252))
        sharpe = float(np.mean(ret) * 252 / (np.std(ret) * np.sqrt(252))) if np.std(ret) > 0 else np.nan
        downside = np.std(ret[ret < 0]) * np.sqrt(252) if (ret < 0).any() else np.nan
        sortino = float(np.mean(ret) * 252 / downside) if downside and downside > 0 else np.nan
        return {
            "cagr": cagr,
            "total_return": float(eq[-1] - 1),
            "mdd": mdd,
            "sharpe": sharpe,
            "sortino": sortino,
            "ann_vol": ann_vol,
        }

    strat_stats = _stats(df["strategy_eq"].to_numpy(), df["strategy_ret"].to_numpy())
    mkt_stats = _stats(df["market_eq"].to_numpy(), df["market_ret"].to_numpy())

    excess = np.exp((df["strategy_ret"] - df["market_ret"]).cumsum()).iloc[-1] - 1

    result = {
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1],
        "days": n,
        "years": years,
        "strategy": strat_stats,
        "market": mkt_stats,
        "excess_return": excess,
        "win_rate": float((df["strategy_ret"] > 0).mean()),
        "avg_turnover": float(np.abs(np.diff(df["position"])).mean()),
        "avg_position": float(df["position"].mean()),
    }
    if "bull_prob" in df.columns:
        result["avg_bull_prob"] = float(df["bull_prob"].mean())
    return result


def save_report(result: dict, ticker: str, out_dir: str | None = None) -> str:
    """输出 CSV 报告 + 控制台摘要。返回报告基路径。"""
    out_dir = out_dir or backtest_dir()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"hmm_transformer_{ticker.replace('.', '_')}")

    df = result["df"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.to_csv(base + "_daily.csv", index=False)

    lines = [f"===== HMM + Transformer WalkForward 回测报告 | {ticker} ====="]
    lines.append(f"回测区间: {pd.Timestamp(result['start_date']).date()} ~ "
                 f"{pd.Timestamp(result['end_date']).date()} "
                 f"({result['days']} 日, {result['years']:.1f} 年)")
    if "elapsed_sec" in result:
        lines.append(f"耗时: {result['elapsed_sec']:.0f} 秒")
    lines.append("")
    lines.append("---- 策略 ----")
    for k, v in result["strategy"].items():
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append("  胜率: {:.4f}".format(result["win_rate"]))
    lines.append("  平均仓位: {:.4f}".format(result["avg_position"]))
    lines.append("  平均换手(日): {:.4f}".format(result["avg_turnover"]))
    lines.append("  超额收益: {:.4f}".format(result["excess_return"]))
    lines.append("")
    lines.append("---- 基准 (买入持有) ----")
    for k, v in result["market"].items():
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append("")
    lines.append("---- 每折指标 ----")
    for m in result["folds"]:
        lines.append(
            f"  Fold {m['fold']}: val_acc={m.get('val_acc', float('nan')):.3f} "
            f"val_ic={m.get('val_ic', float('nan')):.3f} "
            f"test_ret_sum={m['test_ret_sum']:.4f} "
            f"test_acc={m['test_acc']:.3f}"
        )

    report = "\n".join(lines)
    print(report)
    with open(base + "_report.txt", "w") as f:
        f.write(report + "\n")
    return base


def backtest_dir() -> str:
    return BACKTEST_DIR