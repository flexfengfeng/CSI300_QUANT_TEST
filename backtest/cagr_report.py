#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周线趋势策略 —— 复利净值模拟 & 年复合回报率(CAGR)报告
================================================
口径说明：
  早期回测按「每笔固定10万名义」统计, 赚的钱不滚入下一次下单 → 累计盈亏金额。
  本脚本改为标准复利口径: 初始资金10万, 每次信号出现时按【当前净值】全仓买入
  (盈利再投资), 空仓期间持有现金 → 得到真实净值曲线 → 计算 CAGR / 最大回撤 /
  年度收益分解。

对标标的: 腾讯 HK.00700 (推荐) / 盈富 HK.02800 (对比)
信号: 与 weekly_trend.py 完全一致(昨日收盘MA判断 + 今日开盘成交, 无未来函数)
推荐配置: MA10/30 只多、MA5/20 只多 (跨周/月持有, 不做空, 不加额外止损)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
INIT_CASH = 100_000.0

UNIVERSE = {
    'HK.00700': dict(name='腾讯', csv='daily_00700.csv', stamp=0.001, slip=0.0005, lot=100),
    'HK.02800': dict(name='盈富', csv='daily_02800.csv', stamp=0.0, slip=0.0002, lot=500),
}
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0


def cost_of_trade(code, notional):
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return (commission + notional * UNIVERSE[code]['stamp']
            + notional * TRADE_FEE_R + notional * LEVY_R + SYS_FEE)


def simulate(code, fast, slow):
    """复利全仓净值模拟。返回 (净值序列df, 年度收益df, 交易列表)。"""
    df = pd.read_csv(os.path.join(BASE, UNIVERSE[code]['csv']), parse_dates=['time_key'])
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    tkeys = df['time_key'].to_numpy()
    n = len(df)

    ma_fast = pd.Series(closes).rolling(fast).mean().to_numpy()
    ma_slow = pd.Series(closes).rolling(slow).mean().to_numpy()
    slip = UNIVERSE[code]['slip']
    lot = UNIVERSE[code]['lot']

    cash = INIT_CASH
    qty = 0
    entry_px = 0.0
    entry_cost = 0.0
    entry_idx = None
    trades = []
    equity_curve = []

    def mark(i):
        px = closes[i]
        equity_curve.append((pd.Timestamp(tkeys[i]), cash + qty * px))

    for i in range(n):
        mark(i)
        if i == 0 or np.isnan(ma_slow[i - 1]):
            continue
        sig = 1 if ma_fast[i - 1] >= ma_slow[i - 1] else -1

        o = opens[i]

        # 平仓
        if qty > 0 and sig == -1:
            exit_px = o * (1 - slip)
            close_cost = cost_of_trade(code, exit_px * qty)
            gross = (exit_px - entry_px) * qty
            net = gross - entry_cost - close_cost
            cash += exit_px * qty - close_cost
            trades.append({
                'buy_time': pd.Timestamp(tkeys[entry_idx]), 'sell_time': pd.Timestamp(tkeys[i]),
                'hold_days': i - entry_idx, 'gross': gross, 'net': net,
            })
            qty = 0
            entry_px, entry_cost, entry_idx = 0.0, 0.0, None

        # 开仓(全仓复利)
        if qty == 0 and sig == 1:
            equity = cash
            # 留出费用余量, 按净值全仓
            raw = equity / (o * (1 + slip))
            buy_qty = int(raw / lot) * lot
            if buy_qty < lot:
                buy_qty = lot
                # 资金不足一手则跳过
                if equity < (o * (1 + slip)) * lot * 1.02:
                    continue
            entry_px = o * (1 + slip)
            entry_cost = cost_of_trade(code, entry_px * buy_qty)
            if entry_cost > cash:
                continue
            cash -= entry_px * buy_qty + entry_cost
            qty = buy_qty
            entry_idx = i

    # 期末平仓
    if qty > 0:
        exit_px = closes[-1] * (1 - slip)
        close_cost = cost_of_trade(code, exit_px * qty)
        gross = (exit_px - entry_px) * qty
        net = gross - entry_cost - close_cost
        cash += exit_px * qty - close_cost
        trades.append({
            'buy_time': pd.Timestamp(tkeys[entry_idx]),
            'sell_time': pd.Timestamp(tkeys[-1]),
            'hold_days': n - 1 - entry_idx, 'gross': gross, 'net': net,
        })

    eq = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    # 年度收益(自然年)
    eq['year'] = eq['time'].dt.year
    ygroup = eq.groupby('year')['equity']
    yfirst = ygroup.first()
    ylast = ygroup.last()
    annual = pd.DataFrame({
        'year': yfirst.index,
        'start_equity': yfirst.values,
        'end_equity': ylast.values,
    })
    annual['return_pct'] = annual['end_equity'] / annual['start_equity'] - 1
    annual['n_trade'] = [sum(1 for t in trades if pd.Timestamp(t['buy_time']).year == y)
                         for y in annual['year']]

    return eq, annual, pd.DataFrame(trades)


def metrics(code, fast, slow):
    eq, annual, trades = simulate(code, fast, slow)
    eq = eq.set_index('time')['equity']
    final = eq.iloc[-1]
    init = eq.iloc[0]
    total_ret = final / init - 1

    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (final / init) ** (1 / years) - 1 if final > 0 else -1.0

    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    max_dd = dd.min()

    daily_ret = eq.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else float('nan')
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else float('nan')

    n_trades = len(trades)
    trades_np = trades['net'].to_numpy() if n_trades else np.array([])
    wins = trades_np[trades_np > 0]
    loss = trades_np[trades_np <= 0]
    wr = len(wins) / n_trades if n_trades else 0.0
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float('inf')

    return {
        'uni': UNIVERSE[code]['name'], 'code': code, 'cfg': f'MA{fast}/{slow} 只多',
        'start': eq.index[0].date(), 'end': eq.index[-1].date(),
        'years': years, 'days_in_market': float((eq.index.to_series().diff().fillna(pd.Timedelta(0)) > pd.Timedelta(0)).sum()),
        'init': init, 'final': final,
        'total_ret': total_ret, 'cagr': cagr,
        'max_dd': max_dd, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'n_trades': n_trades, 'win_rate': wr, 'pf': pf,
        'avg_hold_days': trades['hold_days'].mean() if n_trades else 0.0,
    }, eq, annual, trades


def main():
    cfgs = [
        ('HK.00700', 5, 20), ('HK.00700', 10, 30),
        ('HK.02800', 5, 20), ('HK.02800', 10, 30),
    ]
    results = {}
    for code, fast, slow in cfgs:
        m, eq, annual, trades = metrics(code, fast, slow)
        results[(code, fast, slow)] = (m, eq, annual, trades)

    print('=' * 118)
    print('周线趋势策略 · 复利净值模拟 · 初始资金 100,000 HKD · 全仓再投资 · 含全部成本')
    print('=' * 118)

    for (code, fast, slow), (m, eq, annual, trades) in results.items():
        print(f'\n### {m["uni"]} {code}  {m["cfg"]}')
        print(f'  区间 {m["start"]} ~ {m["end"]}  ({m["years"]:.1f} 年)')
        print(f'  初始 {m["init"]:,.0f} → 期末 {m["final"]:,.0f} HKD')
        print(f'  总收益率 {m["total_ret"]*100:+.1f}%')
        print(f'  ★ 年复合回报率 CAGR = {m["cagr"]*100:+.2f}%')
        print(f'  最大回撤 {m["max_dd"]*100:.1f}%   年化波动 {m["ann_vol"]*100:.1f}%   '
              f'夏普 {m["sharpe"]:.2f}')
        print(f'  交易 {m["n_trades"]} 笔   胜率(按净) {m["win_rate"]*100:.1f}%   '
              f'盈亏比 {m["pf"]:.2f}   平均持仓 {m["avg_hold_days"]:.0f} 交易日')
        print(f'  --- 分年度收益 ---')
        for _, row in annual.iterrows():
            print('     %d: 净值 %.0f → %.0f   %+7.2f%%   (开仓%d笔)'
                  % (row['year'], row['start_equity'], row['end_equity'],
                     row['return_pct'] * 100, row['n_trade']))

    print()
    print('注: CAGR=(期末/期初)^(1/年数)-1, 年数按自然日/365.25; '
          '全仓复利口径, 空仓期持现金无收息; 与实际可获得的回报存在执行差异')


if __name__ == '__main__':
    main()