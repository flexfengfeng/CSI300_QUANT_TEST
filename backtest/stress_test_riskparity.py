#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险平价策略实盘前压力测试。

测试1: 参数鲁棒性检查 — 在最优参数(目标波动20%/窗口40天)邻域扫描
       (目标 18%~22% × 窗口 30~50天, 共25组), 验证策略是"高原型"而非"尖峰型"。
       合格标准: 参数小幅变化时 CAGR 稳定在 7.5%~9.0%, 且持续跑赢沪深300基准(8.16%)。

测试2: 国债ETF缺位应急预案 — 流动性危机时(2020.3美元荒/2013.6钱荒)国债ETF无法交易,
       将 bond_weight 部分强行换成货币基金/逆回购(固定年化 1.5%/2.0%/2.5%),
       观察策略 CAGR 跌幅。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_transformer.data import load_daily
from hmm_transformer.models import RiskParityStrategy
from hmm_transformer.walkforward import compute_metrics
from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series

BENCHMARK_CAGR = 0.0816   # 同期沪深300买入持有 CAGR (回测区间 2013-11-04 ~ 2026-08-18)


def load_aligned():
    """加载沪深300 + 国债ETF(511010) 按共同交易日对齐。"""
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
    pe_df = load_pe_ttm("CSI300")
    pe_pct = build_pe_percentile_series(pe_df, dates, lookback_years=10)
    return dates, sc, bc, pe_pct


def run_rp(dates, sc, bc, pe_pct, target, lkbk, hold=0.10,
           lower=0.4, upper=1.0, bup=0.6,
           bond_flat_ret=None, warmup=60, cost=0.0005):
    """运行风险平价回测 (复刻 run_riskparity 逻辑)。

    bond_flat_ret 非 None 时, 债券收益用固定日收益替代 (货币基金/逆回购场景, 国债ETF缺位)。
    """
    target_v = target / 100.0
    lo_t = max(0.05, target_v - 0.05)
    hi_t = min(0.35, target_v + 0.05)
    anchors = [(0.00, hi_t), (0.30, target_v + 0.025),
               (0.70, target_v - 0.025), (1.00, lo_t)]
    strat = RiskParityStrategy(
        vol_lookback=lkbk,
        pe_anchors=anchors,
        stock_weight_lower=lower,
        stock_weight_upper=upper,
        bond_weight_upper=bup,
        hold_threshold=hold,
        transaction_cost=cost,
    )
    n = len(sc)
    sw, bw = strat.positions(sc, pe_pct=pe_pct, initial_position=1.0)

    stock_ret = np.zeros(n)
    bond_ret = np.zeros(n)
    if n > 1:
        stock_ret[:-1] = sc[1:] / sc[:-1] - 1.0
        if bond_flat_ret is None:
            bond_ret[:-1] = bc[1:] / bc[:-1] - 1.0
        else:
            bond_ret[:-1] = bond_flat_ret     # 固定日收益 (货币基金/逆回购)

    gross = sw * stock_ret + bw * bond_ret
    prev_sw = np.concatenate([[sw[max(0, warmup - 1)]], sw[warmup:-1]])
    turnover = np.abs(sw[warmup:] - prev_sw)
    net_ret = gross[warmup:] - cost * turnover

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
    return result


def main():
    t0 = time.time()
    print("加载数据 (沪深300 + 国债ETF 511010 + PE分位) ...")
    dates, sc, bc, pe_pct = load_aligned()

    # ================= 测试1: 参数鲁棒性 =================
    print("\n" + "=" * 84)
    print("测试1: 参数鲁棒性检查 | 最优解邻域: 目标波动 18%~22% × 波动窗口 30~50 天")
    print("固定参数: lower=0.4 upper=1.0 bond_up=0.6 hold=0.10 | 沪深300基准CAGR=8.16%")
    print("=" * 84)
    header = (f"{'target':>7} {'lkbk':>4} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} "
              f"{'Sortino':>7} {'excess':>7}")
    print(header)
    results_nb = []
    for target in [18, 19, 20, 21, 22]:
        for lkbk in [30, 35, 40, 45, 50]:
            r = run_rp(dates, sc, bc, pe_pct, target, lkbk)
            s = r["strategy"]
            results_nb.append((target, lkbk, s, r))
            beat = "YES" if s["cagr"] > BENCHMARK_CAGR else "no"
            print(f"{target:>6.0f}% {lkbk:>4d} {s['cagr']*100:>6.2f}% {s['mdd']*100:>6.1f}% "
                  f"{s['sharpe']:>7.3f} {s['sortino']:>7.3f} {r['excess_return']*100:>6.1f}%  {beat}")

    cagrs = [x[2]["cagr"] for x in results_nb]
    midds = [x[2]["mdd"] for x in results_nb]
    sharps = [x[2]["sharpe"] for x in results_nb]
    pct_beat = 100.0 * np.mean([x[2]["cagr"] > BENCHMARK_CAGR for x in results_nb])
    print("\n---- 鲁棒性汇总 (25 组) ----")
    print(f"  CAGR: 均值 {np.mean(cagrs)*100:.2f}% | 中位 {np.median(cagrs)*100:.2f}% | "
          f"最差 {np.min(cagrs)*100:.2f}% | 最好 {np.max(cagrs)*100:.2f}%")
    print(f"  MDD:  均值 {np.mean(midds)*100:.1f}% | 最差(最深) {np.min(midds)*100:.1f}%")
    print(f"  Sharpe: 均值 {np.mean(sharps):.3f}")
    print(f"  跑赢沪深300基准(8.16%) 的组合占比: {pct_beat:.0f}%")
    if np.min(cagrs) > 0.075 and pct_beat >= 80:
        print("  ✅ 判定: 高原型 (参数小幅扰动下 CAGR 稳定 >7.5%, 绝大多数组合跑赢基准)")
    else:
        print("  ⚠️ 判定: 存在尖峰风险 (部分邻域参数跌破阈值, 需谨慎)")

    # ================= 测试2: 国债缺位应急预案 =================
    print("\n" + "=" * 84)
    print("测试2: 国债ETF缺位(流动性危机)应急预案")
    print("场景: 2020.3 美元荒 / 2013.6 钱荒时国债ETF无法正常买卖,")
    print("      bond_weight 强行换成 货币基金/逆回购 (固定年化收益, 无价格波动)")
    print("=" * 84)
    base_r = run_rp(dates, sc, bc, pe_pct, 20, 40)
    bs = base_r["strategy"]
    print(f"\n{'债券场景':>14} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'excess':>7}  备注")
    print(f"{'国债ETF(511010)':>14} {bs['cagr']*100:>6.2f}% {bs['mdd']*100:>6.1f}% "
          f"{bs['sharpe']:>7.3f} {base_r['excess_return']*100:>6.1f}%  实盘默认配置")
    for ann_ret in [0.015, 0.02, 0.025]:
        daily_ret = ann_ret / 252
        r2 = run_rp(dates, sc, bc, pe_pct, 20, 40, bond_flat_ret=daily_ret)
        s2 = r2["strategy"]
        drop = bs["cagr"] - s2["cagr"]
        note = f"CAGR降 {drop*100:.2f}pp"
        if s2["cagr"] > BENCHMARK_CAGR:
            note += " | 仍跑赢基准"
        else:
            note += " | ⚠️ 跌破基准"
        print(f"{('货币基金 ' + format(ann_ret*100, '.1f') + '%'):>16} {s2['cagr']*100:>6.2f}% "
              f"{s2['mdd']*100:>6.1f}% {s2['sharpe']:>7.3f} "
              f"{r2['excess_return']*100:>6.1f}%  {note}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()