#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 backtest 与 webapp 两个实现的回测差异, 定位不一致根因。

用同一份数据 (backtest 目录 CSV):
  A. backtest 路径: hmm_transformer.models.RiskParityStrategy + walkforward.run_riskparity
  B. webapp 路径:   webapp.strategy.backtest (RiskParityParams 默认)
输出两份逐日权重/策略收益的差异统计。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webapp"))

from hmm_transformer.data import load_daily
from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series
from hmm_transformer.models import RiskParityStrategy
from hmm_transformer.walkforward import run_riskparity

import strategy as stg


def main():
    # ---------- 加载同一份数据 (backtest 目录) ----------
    stock_df = load_daily("CSI300")
    bond_df = load_daily("CSA10")
    pe_df = load_pe_ttm("CSI300")
    print(f"数据: 股票 {len(stock_df)} / 国债 {len(bond_df)} / PE {len(pe_df)}")

    merged = pd.merge(
        stock_df[["time_key", "close"]].rename(columns={"close": "sc"}),
        bond_df[["time_key", "close"]].rename(columns={"close": "bc"}),
        on="time_key", how="inner",
    ).sort_values("time_key").reset_index(drop=True)
    dates = merged["time_key"].to_numpy()
    sc = merged["sc"].to_numpy(dtype=np.float64)
    bc = merged["bc"].to_numpy(dtype=np.float64)

    # ---------- A. backtest 路径 ----------
    pe_pct_backtest = build_pe_percentile_series(pe_df, dates, lookback_years=10)
    strategy = RiskParityStrategy(
        vol_lookback=40,
        pe_anchors=[(0.00, 0.25), (0.30, 0.225), (0.70, 0.175), (1.00, 0.15)],
        stock_weight_lower=0.4,
        stock_weight_upper=1.0,
        bond_weight_upper=0.6,
        hold_threshold=0.10,
        transaction_cost=0.0005,
    )
    res_a = run_riskparity(dates, sc, bc, pe_pct=pe_pct_backtest, strategy=strategy, warmup=60)
    df_a = res_a["df"][["date", "position", "strategy_ret"]].rename(
        columns={"position": "sw_a", "strategy_ret": "ret_a"})

    # ---------- B. webapp 路径 ----------
    pe_df_w = pe_df.copy()
    pe_df_w["date"] = pd.to_datetime(pe_df_w["date"])
    stock_w = stock_df.rename(columns={"time_key": "time_key"})
    bond_w = bond_df.rename(columns={"time_key": "time_key"})
    df_b_raw = stg.backtest(stock_w, bond_w, pe_df_w, stg.RiskParityParams())
    m_b = stg.compute_metrics(df_b_raw)
    df_b = df_b_raw[["date", "stock_weight", "strategy_ret"]].rename(
        columns={"stock_weight": "sw_b", "strategy_ret": "ret_b"})

    print(f"\nA(backtest): CAGR={res_a['strategy']['cagr']*100:.4f}% MDD={res_a['strategy']['mdd']*100:.4f}% "
          f"Sharpe={res_a['strategy']['sharpe']:.4f} excess={res_a['excess_return']*100:.4f}pp")
    print(f"B(webapp)  : CAGR={m_b['s_cagr']*100:.4f}% MDD={m_b['s_mdd']*100:.4f}% "
          f"Sharpe={m_b['s_sharpe']:.4f} excess={m_b['excess']*100:.4f}pp")

    # ---------- 逐日对比 ----------
    comp = df_a.merge(df_b, on="date", how="inner")
    print(f"\n共同交易日: {len(comp)}")
    max_sw = np.abs(comp["sw_a"] - comp["sw_b"]).max()
    mean_sw = np.abs(comp["sw_a"] - comp["sw_b"]).mean()
    ret_diff = (comp["ret_a"] - comp["ret_b"]).abs()
    print(f"股票权重差异: max={max_sw:.6f} mean={mean_sw:.6f}")
    print(f"策略日收益差异: max={ret_diff.max():.8f} mean={ret_diff.mean():.8f}")

    # 找出权重差异最大日期
    comp["d_sw"] = (comp["sw_a"] - comp["sw_b"]).abs()
    top = comp.nlargest(5, "d_sw")[["date", "sw_a", "sw_b", "d_sw"]]
    print("\n权重差异最大的 5 个交易日:")
    print(top.to_string(index=False))

    # ---------- PE 分位序列对比 ----------
    pe_pct_web = stg.pe_percentile_series(pe_df_w, dates)
    d_pe = np.abs(pe_pct_backtest - pe_pct_web)
    print(f"\nPE分位差异: max={d_pe.max():.6f} mean={d_pe.mean():.6f}, "
          f"不同天数={int((d_pe > 1e-9).sum())}/{len(dates)}")


if __name__ == "__main__":
    main()