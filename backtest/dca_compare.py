#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCA 定投对比 —— 腾讯(HK.00700) vs 恒生指数(盈富基金 HK.02800)
================================================================
区间与策略回测一致: 2014-01-02 ~ 2026-06-02 (12.4年)

DCA 规则:
  - 每月第一个交易日投入 10,000 HKD
  - 按当日收盘价买入, 允许碎股(券商定投机制), 计入全部费用(佣金+税+滑点)
  - 期末按最后收盘价卖出计费

年化口径:
  - XIRR: 把"每月-10000"与"期末+市值"作为现金流, 解内部收益率(年化)
  - 对照组: 期初一次性投入全部资金并持有到期末的 CAGR

数据: backtest/daily_00700.csv / backtest/daily_02800.csv (qfq 前复权)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
START, END = '2014-01-02', '2026-06-02'
PER_MONTH = 10_000.0

UNIVERSE = {
    'HK.00700': dict(name='腾讯(个股)', csv='daily_00700.csv', stamp=0.001, slip=0.0005),
    'HK.02800': dict(name='盈富基金(恒指ETF)', csv='daily_02800.csv', stamp=0.0, slip=0.0002),
}
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5


def cost_of(notional, stamp):
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return (commission + notional * stamp + notional * TRADE_FEE_R
            + notional * LEVY_R + SYS_FEE)


def xirr(cashflows, lo=-0.9999, hi=20.0):
    """按 (天数偏移, 金额) 现金流解年化IRR, 二分法。"""
    def npv(r):
        return sum(amt / (1 + r) ** (days / 365.0) for days, amt in cashflows)

    f_lo = npv(lo)
    f_hi = npv(hi)
    if f_lo * f_hi > 0:
        return float('nan')
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2


def run_dca(code):
    m = UNIVERSE[code]
    df = pd.read_csv(os.path.join(BASE, m['csv']), parse_dates=['time_key'])
    df = df[(df['time_key'] >= START) & (df['time_key'] <= END)].reset_index(drop=True)
    df['ym'] = df['time_key'].dt.to_period('M')

    # 每月第一个交易日
    first_of_month = df.groupby('ym').head(1)
    close_px = df['close'].to_numpy()
    tkeys = df['time_key'].to_numpy()
    t0 = tkeys[0]

    cashflows = []
    shares = 0.0
    invested = 0.0
    n_buys = 0
    slip = m['slip']

    for _, row in first_of_month.iterrows():
        px = row['close'] * (1 + slip)
        fee = cost_of(PER_MONTH, m['stamp'])
        buy_shares = (PER_MONTH - fee) / px
        if buy_shares <= 0:
            continue
        shares += buy_shares
        invested += PER_MONTH
        n_buys += 1
        days = (row['time_key'] - pd.Timestamp(t0)).days
        cashflows.append((days, -PER_MONTH))

    # 期末卖出
    final_px = close_px[-1] * (1 - slip)
    final_val = shares * final_px - cost_of(shares * final_px, m['stamp'])
    days_end = (pd.Timestamp(tkeys[-1]) - pd.Timestamp(t0)).days
    cashflows.append((days_end, final_val))

    x = xirr(cashflows)
    total_ret = final_val / invested - 1

    # 对照: 期初一次性 buy&hold
    bh_shares = (10_000 * 12 * (n_buys / 12) - 0) / (close_px[0] * (1 + slip)) - 0  # 简化, 用同样本金
    total_bh = invested
    bh_shares = (total_bh - cost_of(total_bh, m['stamp'])) / (close_px[0] * (1 + slip))
    bh_final = bh_shares * final_px - cost_of(bh_shares * final_px, m['stamp'])
    years = days_end / 365.25
    bh_cagr = (bh_final / total_bh) ** (1 / years) - 1 if bh_final > 0 else -1.0

    # 期末指数点位比率(大致)
    return dict(
        name=m['name'], code=code,
        start=pd.Timestamp(t0).date(), end=pd.Timestamp(tkeys[-1]).date(),
        years=years, months=n_buys,
        invested=invested, final_val=final_val,
        total_ret=total_ret, xirr=x,
        bh_cagr=bh_cagr, bh_final=bh_final,
    )


def main():
    print('=' * 100)
    print('DCA 定投对比 · 每月首交易日投入 10,000 HKD · 允许碎股 · 含全部费用+滑点')
    print(f'区间 {START} ~ {END}  (12.4 年)  标的=腾讯 / 恒生指数(盈富ETF代理)')
    print('=' * 100)
    header = '%-22s %7s %9s %9s %11s %10s %10s %10s' % (
        '标的', '月数', '累计投入', '期末市值', '总收益%', 'DCA年化XIRR', '买入持有CAGR', '期末HKD')
    print(header)
    print('-' * 100)
    for code in UNIVERSE:
        r = run_dca(code)
        print('%-22s %7d %9.0f %9.0f %+10.1f%% %+9.2f%% %+9.2f%% %10.0f' % (
            r['name'], r['months'], r['invested'], r['final_val'],
            r['total_ret'] * 100, r['xirr'] * 100, r['bh_cagr'] * 100, r['final_val']))

    print()
    print('解读:')
    print(' - DCA年化XIRR = 现金流内部收益率(每月-1万, 期末+市值), 是定投的真实年化收益率')
    print(' - 买入持有CAGR = 同期一次性全投并持有到期的年化(对照)')
    print(' - 腾讯/盈富均用前复权价(含红利再投资的近似); 定投与一次性的本金相同')
    print(' - 对照策略回测: 腾讯MA10/30只多 CAGR +13.7% (全仓复利)')


if __name__ == '__main__':
    main()