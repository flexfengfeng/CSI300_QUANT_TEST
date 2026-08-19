"""数据加载与特征工程: 读取 SPY/QQQ 日线, 构造 HMM + Transformer 输入特征。"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.dirname(BASE_DIR)

TICKERS = {
    "SPY": "daily_US_SPY.csv",
    "QQQ": "daily_US_QQQ.csv",
    "CSI300": "daily_CN_000300.csv",
    "CSI500": "daily_CN_000905.csv",
    "CSA10": "daily_CN_511010.csv",   # 十年国债ETF (511010.SH), 现金替代资产
}

# 债券类资产 (用于风险平价/资产轮动; 需与股票指数按日期对齐)
BOND_TICKERS = {"CSA10": "daily_CN_511010.csv"}

FEATURE_COLS = [
    "ret_1",       # 昨日对数收益率
    "vol_5",       # 5 日滚动波动率
    "vol_20",      # 20 日滚动波动率
    "mom_5",       # 5 日动量
    "mom_20",      # 20 日动量
    "rsi_14",      # 14 日 RSI
    "bb_pos",      # 布林带位置 (0~1)
    "hl_range",    # (High-Low)/Close
    "gap_open",    # (Open - PrevClose)/PrevClose
]


def load_daily(ticker: str = "SPY") -> pd.DataFrame:
    """加载日线数据, 返回按时间升序的 DataFrame。

    支持指数 (SPY/QQQ/CSI300/CSI500) 与债券ETF (CSA10 = 511010.SH 十年国债ETF)。
    债券ETF 无成交量 或 成交量列缺失时, 由调用方自行处理。
    """
    key = ticker.upper()
    if key not in TICKERS:
        raise ValueError(f"未知 ticker: {ticker}, 可选: {list(TICKERS)}")
    path = os.path.join(BACKTEST_DIR, TICKERS[key])
    df = pd.read_csv(path)
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造模型特征。返回含 time_key, close, volume(可选), 以及 FEATURE_COLS 的 DataFrame。

    volume 仅当原始数据含 volume 列时携带(如沪深300; SPY/QQQ 无成交量), 用于恐慌性抛售硬覆盖。
    """
    out = pd.DataFrame({"time_key": df["time_key"], "close": df["close"]})
    if "volume" in df.columns:
        out["volume"] = df["volume"]

    log_close = np.log(df["close"])
    out["ret_1"] = log_close.diff()
    out["vol_5"] = log_close.diff().rolling(5).std() * np.sqrt(252)
    out["vol_20"] = log_close.diff().rolling(20).std() * np.sqrt(252)
    out["mom_5"] = log_close.diff(5)
    out["mom_20"] = log_close.diff(20)
    out["rsi_14"] = _rsi(df["close"], 14)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    out["bb_pos"] = (df["close"] - mid) / (2 * std).replace(0, np.nan)
    out["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    out["gap_open"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)

    # 去掉前 20 行冷启动
    out = out.iloc[20:].reset_index(drop=True)

    # 全体标准化(离线回测简化处理; WalkForward 折内会重新 fit)
    feat = out[FEATURE_COLS].copy()
    mu, sigma = feat.mean(), feat.std()
    feat = (feat - mu) / sigma.replace(0, 1e-12)
    for c in FEATURE_COLS:
        out[c] = feat[c]
    return out


def make_sequences(features: pd.DataFrame, window: int = 30):
    """把特征表转成 Transformer 所需的 (样本, 窗口, 特征数) 序列数据集。

    返回:
        X: (N, window, F) 序列
        y: (N,) 下一日对数收益 (回归目标)
        dates: (N,) 每个样本的预测日期
        closes: (N,) 每个样本当天的收盘价 (用于回测收益)
        next_closes: (N,) 下一个交易日的收盘价
    """
    arr = features[FEATURE_COLS].to_numpy(dtype=np.float32)
    closes = features["close"].to_numpy(dtype=np.float64)
    dates = features["time_key"].to_numpy()
    log_close = np.log(closes)

    vol_arr = features["volume"].to_numpy(dtype=np.float64) if "volume" in features.columns else None

    X_list, y_list, d_list, c_list, nc_list, v_list = [], [], [], [], [], []
    for i in range(window, len(features) - 1):
        X_list.append(arr[i - window:i])
        y_list.append(log_close[i + 1] - log_close[i])
        d_list.append(dates[i])          # 预测日: 样本窗口最后一天
        c_list.append(closes[i])
        nc_list.append(closes[i + 1])
        v_list.append(vol_arr[i] if vol_arr is not None else np.nan)
    volumes = np.array(v_list, dtype=np.float64) if vol_arr is not None else None
    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.float32),
        np.array(d_list),
        np.array(c_list),
        np.array(nc_list),
        volumes,
    )
