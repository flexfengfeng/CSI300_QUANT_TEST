# -*- coding: utf-8 -*-
"""webapp 冒烟测试: 验证 strategy.py 回测结果与 backtest 一致, 并定位性能瓶颈。"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
import strategy as stg


def main():
    t0 = time.time()
    print("1. 加载底库数据 ...", flush=True)
    stock, bond, pe = stg.load_market_data(refresh=False)
    print(f"   股票 {len(stock)} / 国债 {len(bond)} / PE {len(pe)} 条 "
          f"({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    print("2. 默认参数回测 ...", flush=True)
    p = stg.RiskParityParams()
    df = stg.backtest(stock, bond, pe, p)
    print(f"   完成 ({time.time()-t1:.1f}s), {len(df)} 行", flush=True)

    t2 = time.time()
    m = stg.compute_metrics(df)
    print(f"3. 指标 ({time.time()-t2:.1f}s)", flush=True)
    print(f"   回测区间: {m['start'].date()} ~ {m['end'].date()} "
          f"({m['days']}日, {m['years']:.1f}年)")
    print(f"   CAGR   : {m['s_cagr']*100:.2f}%  (基准 {m['m_cagr']*100:.2f}%)")
    print(f"   MDD    : {m['s_mdd']*100:.2f}%   (基准 {m['m_mdd']*100:.2f}%)")
    print(f"   Sharpe : {m['s_sharpe']:.4f} | 超额: {m['excess']*100:+.2f}pp")
    print(f"   平均仓位: 股票{m['avg_stock']*100:.1f}% 国债{m['avg_bond']*100:.1f}%")

    sig = stg.latest_signal(df)
    print(f"4. 最新信号 ({sig['date'].date()}): "
          f"股票 {sig['stock_weight']*100:.1f}%, "
          f"国债 {sig['bond_weight']*100:.1f}%, "
          f"现金 {sig['cash_weight']*100:.1f}%")

    t3 = time.time()
    print("5. 鲁棒性扫描 ...", flush=True)
    rob = stg.robustness_check(stock, bond, pe)
    print(f"   完成 ({time.time()-t3:.1f}s)")
    print((rob * 100).round(2).to_string())

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()