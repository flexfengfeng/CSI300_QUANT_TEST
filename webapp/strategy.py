# -*- coding: utf-8 -*-
"""风险平价 + 资产轮动 策略核心 (自包含, 供 Streamlit WebApp 使用)。

三大支柱 (与 backtest/hmm_transformer/models.py RiskParityStrategy 逻辑一致):
  1. 波动率目标: stock_weight = clip(target_vol / realized_vol, lower, upper)
  2. PE 估值锚定: 目标波动率随 PE 分位动态调节 (低估→高目标/高估→低目标)
  3. 国债轮动:   bond_weight = clip(1 - stock_weight, 0, bond_upper)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# 腾讯财经行情接口: 用于每日增量刷新 (失败自动回退底库)
KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SYMBOLS = {
    "CSI300": "sh000300",
    "BOND": "sh511010",
}


@dataclass
class RiskParityParams:
    """策略参数 (默认 = 实盘最优参数, 来自 192 组参数扫描)。"""
    target_vol: float = 0.20          # 基准目标年化波动率
    pe_scale: float = 0.05            # PE 调节幅度 (目标波动 = 基准 ± PE分位×幅度)
    vol_lookback: int = 40            # 实际波动率滚动窗口 (交易日)
    stock_lower: float = 0.40         # 股票权重下限 (极端高波动)
    stock_upper: float = 1.00         # 股票权重上限 (不主动加杠杆)
    bond_upper: float = 0.60          # 债券权重上限
    hold_threshold: float = 0.10      # 滞回门槛 (降低换手)
    warmup: int = 60                  # 预热天数 (不计入收益)

    # 由参数派生的 PE→目标波动率 分段锚点
    def pe_anchors(self) -> list:
        t, s = self.target_vol, self.pe_scale
        return [
            (0.00, min(0.35, t + s)),
            (0.30, t + 0.5 * s),
            (0.70, t - 0.5 * s),
            (1.00, max(0.05, t - s)),
        ]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_pe_ttm() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "csi300_pe_ttm.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_daily_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df["time_key"] = pd.to_datetime(df["time_key"])
    return df.sort_values("time_key").reset_index(drop=True)


def fetch_recent_kline(symbol: str, days: int = 5) -> list:
    """从腾讯接口拉取最近 K 线增量 (用于当日最新估值), 失败返回空列表。"""
    try:
        r = requests.get(KLINE_URL, params={
            "param": f"{symbol},day,,,{days},qfq",
        }, timeout=10)
        r.raise_for_status()
        d = r.json().get("data", {})
        kd = d.get(symbol, {})
        seg = kd.get("qfqday") or kd.get("day") or []
        return seg
    except Exception:
        return []


def load_market_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 沪深300 / 国债ETF / PE 序列。

    refresh=True 时尝试用腾讯接口补齐最近交易日 (线性外推 PE)。
    返回 (stock_df, bond_df, pe_df)。
    """
    stock = load_daily_csv("daily_CN_000300.csv")
    bond = load_daily_csv("daily_CN_511010.csv")
    pe = load_pe_ttm()

    if refresh:
        # 补齐股票/债券最新 K 线
        for sym, key, df in [("sh000300", "CSI300", stock), ("sh511010", "BOND", bond)]:
            seg = fetch_recent_kline(sym)
            if not seg:
                continue
            last_local = df["time_key"].iloc[-1].date()
            new_rows = []
            for it in seg:
                d = pd.to_datetime(it[0])
                if d.date() > last_local:
                    new_rows.append({
                        "time_key": d,
                        "open": float(it[1]),
                        "close": float(it[2]),
                        "high": float(it[3]),
                        "low": float(it[4]),
                        "volume": float(it[5]),
                    })
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).sort_values("time_key").reset_index(drop=True)

        # PE 线性外推补齐最新日期 (估值变化慢, 用最近 PE 直充)
        last_pe_date = pe["date"].iloc[-1]
        if stock["time_key"].iloc[-1] > last_pe_date:
            tail_dates = stock.loc[stock["time_key"] > last_pe_date, "time_key"]
            last_pe = pe["pe_ttm"].iloc[-1]
            pe = pd.concat([
                pe,
                pd.DataFrame({"date": tail_dates.values, "pe_ttm": last_pe}),
            ], ignore_index=True).sort_values("date").reset_index(drop=True)
    return stock, bond, pe


# ---------------------------------------------------------------------------
# 策略核心
# ---------------------------------------------------------------------------
def pe_percentile_series(pe_df: pd.DataFrame, dates: np.ndarray,
                         lookback_years: int = 10) -> np.ndarray:
    """计算每个日期 PE 的滚动 lookback_years 年分位数 (0~1)。

    逐日实现与 backtest/hmm_transformer/valuation.py 语义完全一致:
      窗口 = (d - lookback_years, d] 闭区间, 至少 60 条, 分位 = mean(PETTM <= 当天PETTM)。
    """
    pe_arr = pe_df["pe_ttm"].to_numpy(dtype=np.float64)
    pe_dates = pe_df["date"].to_numpy()                       # datetime64[ns]
    span_ns = np.timedelta64(int(lookback_years * 365.25) * 24 * 3600, "s").astype("timedelta64[ns]")
    out = np.full(len(dates), 0.5, dtype=np.float64)
    for i, d in enumerate(dates):
        d64 = np.datetime64(pd.Timestamp(d), "ns")
        start = d64 - span_ns
        i0 = np.searchsorted(pe_dates, start, side="right")   # 严格 > start
        i1 = np.searchsorted(pe_dates, d64, side="right")     # 含当天 <= d
        if i1 - i0 < 60:
            continue
        win = pe_arr[i0:i1]
        cur = pe_arr[i1 - 1]                                  # 窗口最后一天 = 当天 PE
        out[i] = float(np.mean(win <= cur))
    return out


def realized_vol(closes: np.ndarray, lookback: int = 40) -> np.ndarray:
    s = pd.Series(closes)
    vol = s.pct_change().rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
    return vol.to_numpy()


def dynamic_target(pe_pct: np.ndarray, anchors: list) -> np.ndarray:
    a = np.array(sorted(anchors), dtype=np.float64)
    p = np.clip(np.nan_to_num(pe_pct, nan=0.5), 0.0, 1.0)
    return np.interp(p, a[:, 0], a[:, 1])


def compute_weights(closes: np.ndarray, pe_pct: np.ndarray, p: RiskParityParams,
                    initial_position: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (stock_weight, bond_weight)。含滞回平滑。"""
    vol = realized_vol(closes, p.vol_lookback)
    target = dynamic_target(pe_pct, p.pe_anchors())
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(np.isfinite(vol) & (vol > 0), target / vol, np.nan)
    raw = np.where(np.isfinite(raw), raw, 1.0)
    st = np.clip(raw, p.stock_lower, p.stock_upper)
    # 滞回
    sw = np.empty_like(st)
    cur = initial_position
    for i, t in enumerate(st):
        if abs(t - cur) > p.hold_threshold:
            cur = t
        sw[i] = cur
    bw = np.clip(1.0 - sw, 0.0, p.bond_upper)
    return sw, bw


def backtest(stock_df: pd.DataFrame, bond_df: pd.DataFrame, pe_df: pd.DataFrame,
             p: RiskParityParams, cost: float = 0.0005) -> pd.DataFrame:
    """运行全样本风险平价回测, 返回逐日 DataFrame (含各指标计算所需列)。"""
    merged = pd.merge(
        stock_df[["time_key", "close"]].rename(columns={"close": "sc"}),
        bond_df[["time_key", "close"]].rename(columns={"close": "bc"}),
        on="time_key", how="inner",
    ).sort_values("time_key").reset_index(drop=True)

    dates = merged["time_key"].to_numpy()
    sc = merged["sc"].to_numpy(dtype=np.float64)
    bc = merged["bc"].to_numpy(dtype=np.float64)
    pe_pct = pe_percentile_series(pe_df, dates)

    sw, bw = compute_weights(sc, pe_pct, p, initial_position=1.0)

    n = len(sc)
    stock_ret = np.zeros(n)
    bond_ret = np.zeros(n)
    if n > 1:
        stock_ret[:-1] = sc[1:] / sc[:-1] - 1.0
        bond_ret[:-1] = bc[1:] / bc[:-1] - 1.0
    gross = sw * stock_ret + bw * bond_ret

    warm = p.warmup
    prev_sw = np.concatenate([[sw[max(0, warm - 1)]], sw[warm:-1]])
    turnover = np.abs(sw[warm:] - prev_sw)
    net_ret = gross[warm:] - cost * turnover

    return pd.DataFrame({
        "date": pd.to_datetime(dates[warm:]),
        "stock_weight": sw[warm:],
        "bond_weight": bw[warm:],
        "strategy_ret": net_ret,
        "market_ret": stock_ret[warm:],
    })


def compute_metrics(df: pd.DataFrame) -> dict:
    """计算回测指标 (与 backtest/walkforward.py compute_metrics 一致)。"""
    if len(df) < 2:
        return {}
    eq = np.exp(df["strategy_ret"].cumsum())
    mkt = np.exp(df["market_ret"].cumsum())
    n = len(df)
    years = n / 252

    def _stats(eq, ret):
        roll_max = np.maximum.accumulate(eq)
        dd = eq / roll_max - 1
        vol = float(np.std(ret) * np.sqrt(252))
        return {
            "cagr": float(eq[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "mdd": float(dd.min()),
            "sharpe": float(np.mean(ret) * 252 / vol) if vol > 0 else np.nan,
            "ann_vol": vol,
        }

    strat = _stats(eq.to_numpy(), df["strategy_ret"].to_numpy())
    mkt_s = _stats(mkt.to_numpy(), df["market_ret"].to_numpy())
    excess = float(np.exp((df["strategy_ret"] - df["market_ret"]).cumsum()).iloc[-1] - 1)
    return {
        **{f"s_{k}": v for k, v in strat.items()},
        **{f"m_{k}": v for k, v in mkt_s.items()},
        "excess": excess,
        "win_rate": float((df["strategy_ret"] > 0).mean()),
        "avg_stock": float(df["stock_weight"].mean()),
        "avg_bond": float(df["bond_weight"].mean()),
        "avg_turnover": float(np.abs(np.diff(df["stock_weight"])).mean()),
        "years": years,
        "days": n,
        "start": df["date"].iloc[0],
        "end": df["date"].iloc[-1],
    }


def latest_signal(df: pd.DataFrame) -> dict:
    """返回最新一天的仓位信号 (供仪表盘展示)。"""
    last = df.iloc[-1]
    return {
        "date": last["date"],
        "stock_weight": float(last["stock_weight"]),
        "bond_weight": float(last["bond_weight"]),
        "cash_weight": float(max(0.0, 1 - last["stock_weight"] - last["bond_weight"])),
        "position_pct": float(last["stock_weight"] * 100),
    }


def robustness_check(stock_df, bond_df, pe_df,
                     targets=(18, 19, 20, 21, 22),
                     lookbacks=(30, 40, 50)) -> pd.DataFrame:
    """参数鲁棒性扫描: 返回 CAGRs 表 (供过拟合诊断)。"""
    rows = []
    for t in targets:
        for lb in lookbacks:
            p = RiskParityParams(target_vol=t / 100.0, vol_lookback=lb)
            df = backtest(stock_df, bond_df, pe_df, p)
            m = compute_metrics(df)
            rows.append({"target_vol": t, "lookback": lb, "cagr": m.get("s_cagr", np.nan)})
    return pd.DataFrame(rows).pivot(index="target_vol", columns="lookback", values="cagr")