#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险平价 + 资产轮动 (v4): 彻底放弃 HMM 方向择时, 改为确定性公式。

核心公式 (仅 5 行):
  1. realized_vol = closes.pct_change().rolling(20).std() * sqrt(252)
  2. dynamic_target = 0.15 + 0.05 * (1 - pe_percentile)   # PE 分位越低目标波动率越高
  3. stock_weight = clip(dynamic_target / realized_vol, 0.3, 1.2)
  4. bond_weight  = clip(1 - stock_weight, 0.0, 0.7)      # 剩余资金买国债ETF(511010)
  5. 组合收益 = stock_weight × 股票收益 + bond_weight × 国债收益

用法:
  python run_riskparity.py                     # CSI300 + 国债ETF 风险平价回测
  python run_riskparity.py --plot              # 输出权益曲线 PNG
  python run_riskparity.py --target 15 --pe-scale 5   # 自定义目标波动率/PE调节幅度
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_transformer.data import load_daily
from hmm_transformer.models import RiskParityStrategy
from hmm_transformer.walkforward import run_riskparity, save_riskparity_report
try:
    from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series
    HAS_VALUATION = True
except ImportError:
    HAS_VALUATION = False

STOCK_TICKER = "CSI300"
BOND_TICKER = "CSA10"   # 511010.SH 十年国债ETF


def main():
    parser = argparse.ArgumentParser(description="风险平价 + 资产轮动回测 (CSI300 + 国债ETF)")
    # 最优参数 (v3 参数扫描 192 组确定): CAGR 8.82% / MDD -38.6% / Sharpe 0.50 / 超额 +7.9%
    parser.add_argument("--vol-lookback", type=int, default=40, help="实际波动率滚动窗口(交易日); 默认40(最优)")
    parser.add_argument("--target", type=float, default=None,
                        help="基准目标年化波动率(%); 默认 20, 由 PE 分位动态调节 15~25")
    parser.add_argument("--pe-scale", type=float, default=None,
                        help="PE 调节幅度(%); 默认 5, 目标波动率 = 基准 + 调节×(1-PE分位)")
    parser.add_argument("--lower", type=float, default=0.4, help="股票权重下限(极端高波动); 默认0.4(最优)")
    parser.add_argument("--upper", type=float, default=1.0, help="股票权重上限(极端低波动); 默认1.0(最优)")
    parser.add_argument("--bond-upper", type=float, default=0.6, help="债券权重上限; 默认0.6(最优)")
    parser.add_argument("--hold-threshold", type=float, default=0.10, help="滞回门槛(降低换手); 默认0.10(最优)")
    parser.add_argument("--warmup", type=int, default=60, help="预热天数(不计入收益)")
    parser.add_argument("--cost", type=float, default=0.0005, help="单边换仓成本")
    parser.add_argument("--no-pe", action="store_true", help="关闭 PE 估值调节(固定目标波动率)")
    parser.add_argument("--plot", action="store_true", help="输出权益曲线 PNG")
    args = parser.parse_args()

    # ---- 加载股票 (沪深300) 与 债券 (国债ETF 511010) ----
    print(f"加载 {STOCK_TICKER} 数据 ...")
    stock_df = load_daily(STOCK_TICKER)
    print(f"  {len(stock_df)} 根日线, {stock_df['time_key'].iloc[0].date()} ~ "
          f"{stock_df['time_key'].iloc[-1].date()}")

    print(f"加载 {BOND_TICKER} (511010.SH 十年国债ETF) 数据 ...")
    bond_df = load_daily(BOND_TICKER)
    print(f"  {len(bond_df)} 根日线, {bond_df['time_key'].iloc[0].date()} ~ "
          f"{bond_df['time_key'].iloc[-1].date()}")

    # ---- 按共同交易日对齐 (inner join) ----
    merged = pd.merge(
        stock_df[["time_key", "close"]].rename(columns={"close": "stock_close"}),
        bond_df[["time_key", "close"]].rename(columns={"close": "bond_close"}),
        on="time_key", how="inner",
    ).sort_values("time_key").reset_index(drop=True)
    print(f"共同交易日: {len(merged)} 个 ({merged['time_key'].iloc[0].date()} ~ "
          f"{merged['time_key'].iloc[-1].date()})")

    dates = merged["time_key"].to_numpy()
    stock_closes = merged["stock_close"].to_numpy(dtype=np.float64)
    bond_closes = merged["bond_close"].to_numpy(dtype=np.float64)

    # ---- PE 分位数 (CSI300) ----
    pe_pct = None
    if not args.no_pe and HAS_VALUATION:
        try:
            pe_df = load_pe_ttm(STOCK_TICKER)
            pe_pct = build_pe_percentile_series(pe_df, dates, lookback_years=10)
            print(f"已加载 {STOCK_TICKER} PE_TTM 滚动10年分位数: {len(pe_pct)} 个交易日")
            print(f"  PE 分位区间: [{pe_pct.min():.3f}, {pe_pct.max():.3f}]")
        except Exception as e:
            print(f"PE 估值加载失败(忽略, 目标波动率固定): {e}")
            pe_pct = None

    # ---- 构建风险平价策略 ----
    target = 0.20 if args.target is None else args.target / 100.0
    pe_scale = 0.05 if args.pe_scale is None else args.pe_scale / 100.0
    if args.no_pe:
        # 固定目标波动率 (等价于 pe_anchors 全水平平): 构造两个同值锚点
        pe_anchors = [(0.0, target), (1.0, target)]
        print(f"PE 调节: 关闭 | 固定目标波动率 = {target:.1%}")
    else:
        lo_t = max(0.05, target - pe_scale)
        hi_t = min(0.35, target + pe_scale)
        pe_anchors = [(0.00, hi_t), (0.30, target + 0.5 * pe_scale),
                      (0.70, target - 0.5 * pe_scale), (1.00, lo_t)]
        print(f"PE 调节: 开启 | 目标波动率 {lo_t:.1%} (PE=100%) ~ {hi_t:.1%} (PE=0%)")

    strategy = RiskParityStrategy(
        vol_lookback=args.vol_lookback,
        pe_anchors=pe_anchors,
        stock_weight_lower=args.lower,
        stock_weight_upper=args.upper,
        bond_weight_upper=args.bond_upper,
        hold_threshold=args.hold_threshold,
        transaction_cost=args.cost,
    )

    # ---- 回测 ----
    print(f"开始风险平价回测 | 窗口={args.vol_lookback} 预热={args.warmup} "
          f"滞回={args.hold_threshold} 成本={args.cost}")
    result = run_riskparity(
        dates, stock_closes, bond_closes,
        pe_pct=pe_pct,
        strategy=strategy,
        warmup=args.warmup,
        transaction_cost=args.cost,
    )

    # ---- 报告 ----
    base = save_riskparity_report(result, STOCK_TICKER)
    print(f"\n报告已保存: {base}_report.txt / {base}_daily.csv")

    # ---- 绘图 ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = [
                "PingFang HK", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"
            ]
            plt.rcParams["axes.unicode_minus"] = False

            df = result["df"]
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(df["date"], np.exp(df["strategy_ret"].cumsum()), label="风险平价组合 (股票+国债)")
            axes[0].plot(df["date"], np.exp(df["market_ret"].cumsum()), label="沪深300 买入持有", alpha=0.7)
            axes[0].set_ylabel("净值")
            axes[0].legend()
            axes[0].set_title("风险平价 + 资产轮动 | CSI300 + 国债ETF(511010)")

            axes[1].plot(df["date"], df["position"], drawstyle="steps-post",
                         label="股票仓位", color="tab:purple")
            axes[1].plot(df["date"], df["bond_weight"], drawstyle="steps-post",
                         label="国债ETF仓位", color="tab:orange")
            axes[1].set_ylabel("权重")
            axes[1].legend()

            if pe_pct is not None:
                # dates 与 df["date"] 均为 datetime64; 用 pandas 对齐避免类型比较问题
                pe_series = pd.Series(pe_pct, index=pd.to_datetime(dates))
                pe_aligned = pe_series.reindex(pd.to_datetime(df["date"])).to_numpy()
                axes[2].plot(df["date"], pe_aligned, label="PE 分位数", color="tab:green")
                axes[2].set_ylabel("PE 分位")
                axes[2].legend()

            plt.tight_layout()
            png = base + "_equity.png"
            plt.savefig(png, dpi=120)
            print(f"图表已保存: {png}")
        except Exception as e:
            print(f"绘图失败(忽略): {e}")


if __name__ == "__main__":
    main()