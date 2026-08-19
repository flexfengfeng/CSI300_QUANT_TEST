#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险平价策略参数扫描 v3: 扩展维度寻找 MDD≤-40% & Sharpe≥0.55 的参数。"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_transformer.data import load_daily
from hmm_transformer.models import RiskParityStrategy
from hmm_transformer.walkforward import run_riskparity
from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series


def main():
    t0 = time.time()
    print("加载数据 ...")
    stock_df = load_daily("CSI300")
    bond_df = load_daily("CSA10")
    merged = pd.merge(
        stock_df[["time_key", "close"]].rename(columns={"close": "sc"}),
        bond_df[["time_key", "close"]].rename(columns={"close": "bc"}),
        on="time_key", how="inner",
    ).sort_values("time_key").reset_index(drop=True)
    dates = merged["time_key"].to_numpy()
    sc = merged["sc"].to_numpy(dtype=np.float64)
    bc = merged["bc"].to_numpy(dtype=np.float64)

    print("计算 PE 分位数 (滚动10年) ...")
    pe_df = load_pe_ttm("CSI300")
    pe_pct = build_pe_percentile_series(pe_df, dates, lookback_years=10)

    header = (f"{'target':>7} {'pe_scale':>8} {'lower':>5} {'upper':>5} {'bond_up':>7} "
              f"{'lkbk':>4} {'hold':>4} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Sortino':>7} "
              f"{'excess':>7}")
    print("\n" + header)
    results = []
    targets = [0.10, 0.11, 0.12, 0.13, 0.14, 0.19, 0.20, 0.22]
    lookbacks = [20, 40, 60]
    holds = [0.05, 0.10]
    bounds_list = [(0.3, 1.2, 0.7), (0.35, 1.1, 0.65), (0.4, 1.0, 0.6), (0.5, 1.0, 0.5)]
    for target in targets:
        for pe_scale in [0.05]:
            lo = max(0.05, target - pe_scale)
            hi = min(0.35, target + pe_scale)
            anchors = [(0.00, hi), (0.30, target + 0.5 * pe_scale),
                       (0.70, target - 0.5 * pe_scale), (1.00, lo)]
            for lb in lookbacks:
                for hold in holds:
                    for lower, upper, bup in bounds_list:
                        strat = RiskParityStrategy(
                            vol_lookback=lb,
                            pe_anchors=anchors,
                            stock_weight_lower=lower,
                            stock_weight_upper=upper,
                            bond_weight_upper=bup,
                            hold_threshold=hold,
                            transaction_cost=0.0005,
                        )
                        r = run_riskparity(dates, sc, bc, pe_pct=pe_pct, strategy=strat, warmup=60)
                        s = r["strategy"]
                        results.append((target, lower, upper, bup, lb, hold, s, r))
                        print(f"{target*100:>6.0f}% {pe_scale*100:>7.0f}% {lower:>5.1f} {upper:>5.1f} "
                              f"{bup:>7.1f} {lb:>4d} {hold:>4.2f} {s['cagr']*100:>6.2f}% "
                              f"{s['mdd']*100:>6.1f}% {s['sharpe']:>7.3f} {s['sortino']:>7.3f} "
                              f"{r['excess_return']*100:>6.1f}%")
                        sys.stdout.flush()

    print("\n=== 目标一: MDD <= -40% 且 超额>0 (跑赢 + 回撤可控) ===")
    c1 = [x for x in results if x[6]["mdd"] >= -0.40 and x[7]["excess_return"] > 0]
    for target, lower, upper, bup, lb, hold, s, r in sorted(c1, key=lambda x: x[7]["excess_return"], reverse=True)[:8]:
        print(f"  target={target*100:.0f}% lower={lower} upper={upper} bond_up={bup} lkbk={lb} hold={hold}: "
              f"CAGR={s['cagr']*100:.2f}% MDD={s['mdd']*100:.1f}% Sharpe={s['sharpe']:.3f} "
              f"excess={r['excess_return']*100:+.1f}%")

    print("\n=== 目标二: Sharpe 最高 (不限超额) ===")
    for target, lower, upper, bup, lb, hold, s, r in sorted(results, key=lambda x: x[6]["sharpe"], reverse=True)[:8]:
        print(f"  target={target*100:.0f}% lower={lower} upper={upper} bond_up={bup} lkbk={lb} hold={hold}: "
              f"CAGR={s['cagr']*100:.2f}% MDD={s['mdd']*100:.1f}% Sharpe={s['sharpe']:.3f} "
              f"excess={r['excess_return']*100:+.1f}%")

    print("\n=== 目标三: 超额最高 (跑赢最多) ===")
    for target, lower, upper, bup, lb, hold, s, r in sorted(results, key=lambda x: x[7]["excess_return"], reverse=True)[:8]:
        print(f"  target={target*100:.0f}% lower={lower} upper={upper} bond_up={bup} lkbk={lb} hold={hold}: "
              f"CAGR={s['cagr']*100:.2f}% MDD={s['mdd']*100:.1f}% Sharpe={s['sharpe']:.3f} "
              f"excess={r['excess_return']*100:+.1f}%")

    print(f"\n共 {len(results)} 组, 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()