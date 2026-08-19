"""沪深300 PE_TTM 估值分位数计算 (滚动10年) 与硬覆盖逻辑。"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

BACKTEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PE_CSV = os.path.join(BACKTEST_DIR, "csi300_pe_ttm.csv")

# 指数 → PE_TTM CSV 映射 (可扩展: 中证500 等)
PE_CSV_BY_TICKER = {
    "CSI300": os.path.join(BACKTEST_DIR, "csi300_pe_ttm.csv"),
    "CSI500": os.path.join(BACKTEST_DIR, "csi500_pe_ttm.csv"),
}


def load_pe_ttm(ticker: str = "CSI300") -> pd.DataFrame:
    """加载指数 PE_TTM 序列 (date, pe_ttm), 按日期升序。默认为沪深300。"""
    csv_path = PE_CSV_BY_TICKER.get(ticker.upper(), PE_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"缺 PE 数据: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def rolling_pe_percentile(pe_df: pd.DataFrame, current_date: pd.Timestamp,
                          lookback_years: int = 10) -> float:
    """计算某日 PE_TTM 在过去 lookback_years 年内的分位数 (0~1)。

    窗口取 (current - lookback, current]，至少 60 个交易日；不足返回 0.5 中性。
    """
    start = current_date - pd.Timedelta(days=int(lookback_years * 365.25))
    window = pe_df[(pe_df["date"] > start) & (pe_df["date"] <= current_date)]
    if len(window) < 60:
        return 0.5
    cur = float(window.iloc[-1]["pe_ttm"])
    vals = window["pe_ttm"].to_numpy()
    return float(np.mean(vals <= cur))


def build_pe_percentile_series(pe_df: pd.DataFrame,
                               dates: np.ndarray,
                               lookback_years: int = 10) -> np.ndarray:
    """对给定日期数组批量计算滚动 PE 分位数。返回 (N,) float。"""
    pcts = np.empty(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        pcts[i] = rolling_pe_percentile(pe_df, pd.Timestamp(d), lookback_years)
    return pcts