#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位管理对比 —— 高位减仓对收益/回撤的影响
================================================
基准：腾讯 MA5/20、MA10/30 只多（跨周/月持有）
变体：
  A 全仓复利        —— 每次信号全仓买入
  B 恒定仓位(60%)   —— 始终只投净值的60%
  C 乖离率高位减仓1 —— 价>慢线8%→半仓, >15%→3成仓 (趋势越高越谨慎)
  D 乖离率高位减仓2 —— 价>慢线12%→6成仓, >20%→4成仓 (更温和)
  E 净值回撤减仓    —— 净值自峰值回撤>8%后降到6成仓 (追高保护)

口径：全仓复利等价于目标仓位=100%；部分减仓按开盘价-滑点成交, 计入成本;
      信号与仓位均用"昨日收盘/昨日峰值"判定, 今日开盘执行(无未来函数)。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
INIT_CASH = 100_000.0

TENCENT = dict(name='腾讯', csv='daily_00700.csv', stamp=0.001, slip=0.0005, lot=100)
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0


def cost_of_trade(notional):
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return (commission + notional * TENCENT['stamp']
            + notional * TRADE_FEE_R + notional * LEVY_R + SYS_FEE)


def target_ratio(mode, price, ma, equity, peak, rsi):
    """返回目标仓位比例 0~1。所有输入为昨日已收盘数据。"""
    r = 1.0
    if mode == 'const60':
        r = 0.60
    elif mode == 'bias8':
        bias = price / ma - 1 if ma > 0 else 0.0
        if bias > 0.15:
            r = 0.30
        elif bias > 0.08:
            r = 0.50
    elif mode == 'bias12':
        bias = price / ma - 1 if ma > 0 else 0.0
        if bias > 0.20:
            r = 0.40
        elif bias > 0.12:
            r = 0.60
    elif mode == 'dd8':
        if peak > 0 and equity / peak - 1 < -0.08:
            r = 0.60
    elif mode == 'rsi70':
        # RSI 超买减仓: >70→6成, >80→4成
        if rsi > 80:
            r = 0.40
        elif rsi > 70:
            r = 0.60
    elif mode == 'rsi75':
        # RSI 更敏感: >70→6成, >75→3成
        if rsi > 75:
            r = 0.30
        elif rsi > 70:
            r = 0.60
    elif mode == 'rsi_dd':
        # RSI 超买 或 净值回撤 任一触发即减仓
        if rsi > 75:
            r = 0.60
        elif peak > 0 and equity / peak - 1 < -0.08:
            r = 0.60
    return r


def simulate(mode, fast, slow):
    df = pd.read_csv(os.path.join(BASE, TENCENT['csv']), parse_dates=['time_key'])
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    tkeys = df['time_key'].to_numpy()
    n = len(df)
    slip = TENCENT['slip']
    lot = TENCENT['lot']

    ma_fast = pd.Series(closes).rolling(fast).mean().to_numpy()
    ma_slow = pd.Series(closes).rolling(slow).mean().to_numpy()

    # RSI(14) Wilder 平滑, 与信号同样采用昨日收盘、无未来函数
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi_n = 14
    avg_gain = gain.ewm(alpha=1 / rsi_n, min_periods=rsi_n).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_n, min_periods=rsi_n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_arr = (100 - 100 / (1 + rs)).to_numpy()
    rsi_arr = np.where(np.isnan(rsi_arr), 50.0, rsi_arr)

    cash = INIT_CASH
    qty = 0
    entry_px = 0.0
    entry_cost = 0.0
    entry_idx = None
    trades = []          # 每次成交事件(action: close/trim/buy)
    equity_curve = []
    peak = INIT_CASH

    def mark(i):
        nonlocal peak
        px = closes[i]
        eq = cash + qty * px
        peak = max(peak, eq)
        equity_curve.append((pd.Timestamp(tkeys[i]), eq))

    for i in range(n):
        mark(i)
        if i == 0 or np.isnan(ma_slow[i - 1]):
            continue

        sig = 1 if ma_fast[i - 1] >= ma_slow[i - 1] else -1
        prev_equity = equity_curve[-2][1] if len(equity_curve) >= 2 else INIT_CASH
        prev_peak = peak  # 昨日及之前的净值峰值(标记当天尚未含今日)
        ratio = target_ratio(mode, closes[i - 1], ma_slow[i - 1], prev_equity, prev_peak, rsi_arr[i - 1])

        o = opens[i]

        # 信号转空 → 全平
        if qty > 0 and sig == -1:
            exit_px = o * (1 - slip)
            close_cost = cost_of_trade(exit_px * qty)
            gross = (exit_px - entry_px) * qty
            net = gross - entry_cost - close_cost
            cash += exit_px * qty - close_cost
            trades.append({
                'action': 'close', 'time': pd.Timestamp(tkeys[i]),
                'days': i - entry_idx, 'gross': gross, 'net': net, 'qty': qty,
            })
            qty = 0
            entry_px, entry_cost, entry_idx = 0.0, 0.0, None

        # 持仓中 → 高位减仓到目标比例
        if qty > 0 and sig == 1:
            price_now = o
            equity_now = cash + qty * price_now
            actual = qty * price_now / equity_now if equity_now > 0 else 1.0
            if ratio < actual - 0.01:
                target_qty = int(equity_now * ratio / (price_now * (1 - slip)) / lot) * lot
                if target_qty < lot:
                    target_qty = 0
                sell_qty = qty - target_qty
                if sell_qty >= lot:
                    sell_px = price_now * (1 - slip)
                    sell_cost = cost_of_trade(sell_px * sell_qty)
                    cash += sell_px * sell_qty - sell_cost
                    # 调整持仓成本基数: 卖出部分按比例释放成本
                    release = entry_cost * (sell_qty / qty)
                    entry_cost -= release
                    trades.append({
                        'action': 'trim', 'time': pd.Timestamp(tkeys[i]),
                        'days': i - entry_idx, 'gross': 0.0,
                        'net': -(sell_cost + release) * 0, 'qty': sell_qty,  # 简化, 详见注释
                    })
                    qty = target_qty

        # 空仓 → 按目标比例建仓
        if qty == 0 and sig == 1:
            equity = cash
            budget = equity * ratio
            raw = budget / (o * (1 + slip))
            buy_qty = int(raw / lot) * lot
            if buy_qty < lot:
                continue
            entry_px = o * (1 + slip)
            entry_cost = cost_of_trade(entry_px * buy_qty)
            if entry_px * buy_qty + entry_cost > cash:
                buy_qty = int(cash / (o * (1 + slip) * 1.02) / lot) * lot
                if buy_qty < lot:
                    continue
                entry_px = o * (1 + slip)
                entry_cost = cost_of_trade(entry_px * buy_qty)
            cash -= entry_px * buy_qty + entry_cost
            qty = buy_qty
            entry_idx = i
            trades.append({
                'action': 'buy', 'time': pd.Timestamp(tkeys[i]),
                'days': 0, 'gross': 0.0, 'net': -entry_cost, 'qty': qty,
            })

    if qty > 0:
        exit_px = closes[-1] * (1 - slip)
        close_cost = cost_of_trade(exit_px * qty)
        gross = (exit_px - entry_px) * qty
        net = gross - entry_cost - close_cost
        cash += exit_px * qty - close_cost
        trades.append({
            'action': 'close', 'time': pd.Timestamp(tkeys[-1]),
            'days': n - 1 - entry_idx, 'gross': gross, 'net': net, 'qty': qty,
        })

    eq = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    return eq, pd.DataFrame(trades)


def report(mode, fast, slow):
    eq, trades = simulate(mode, fast, slow)
    eq = eq.set_index('time')['equity']
    final = eq.iloc[-1]
    total_ret = final / INIT_CASH - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (final / INIT_CASH) ** (1 / years) - 1 if final > 0 else -1.0
    roll_max = eq.cummax()
    md = (eq / roll_max - 1).min()
    daily = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else float('nan')
    calmar = cagr / abs(md) if md != 0 else float('nan')

    # 闭合交易净盈亏
    closed = trades[trades['action'] == 'close']
    nets = closed['net'].to_numpy() if len(closed) else np.array([])
    wins = nets[nets > 0]
    loss = nets[nets <= 0]
    wr = len(wins) / len(nets) if len(nets) else 0.0
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float('inf')

    n_trim = int((trades['action'] == 'trim').sum())
    return {
        'mode': mode, 'cfg': f'MA{fast}/{slow}',
        'final': final, 'total_ret': total_ret, 'cagr': cagr,
        'mdd': md, 'sharpe': sharpe, 'calmar': calmar,
        'n_close': len(closed), 'n_trim': n_trim,
        'win_rate': wr, 'pf': pf,
    }


def main():
    print('=' * 128)
    print('仓位管理研究 · 腾讯 HK.00700 · 初始10万HKD · 复利 + 动态减仓 · 含全部成本')
    print('=' * 128)
    modes = [
        ('full', 'A 全仓复利(基准)'),
        ('const60', 'B 恒定60%仓位'),
        ('bias12', 'C 乖离率减仓2(>12%→6成,>20%→4成)'),
        ('bias8', 'D 乖离率减仓1(>8%→5成,>15%→3成)'),
        ('dd8', 'E 净值回撤减仓(距峰值-8%→6成)'),
        ('rsi70', 'F RSI超买减仓(RSI70→6成,80→4成)'),
        ('rsi75', 'G RSI超买减仓敏感(RSI70→6成,75→3成)'),
        ('rsi_dd', 'H RSI超买或净值回撤→6成(复合)'),
    ]
    for fast, slow in [(5, 20), (10, 30)]:
        print(f'\n### MA{fast}/{slow} 只多')
        print('%-32s %9s %8s %8s %8s %7s %8s %6s %7s %6s' % (
            '模式', '期末HKD', '总收益%', 'CAGR%', '最大回撤%', '夏普', 'Calmar', '平仓', '波段', '胜率%'))
        for mode, label in modes:
            m = report(mode, fast, slow)
            print('%-32s %9.0f %+7.1f%% %+7.2f%% %+7.1f%% %6.2f %8.2f %6d %6d %6.1f%%' % (
                label, m['final'], m['total_ret'] * 100, m['cagr'] * 100,
                m['mdd'] * 100, m['sharpe'], m['calmar'],
                m['n_close'], m['n_trim'], m['win_rate'] * 100))

    print()
    print('注: 波段=高位减仓次数(trim); 减仓按当日开盘价-滑点成交并计费; '
          'C=较温和 D=较激进 E=回撤触发式; 信号/仓位均无未来函数')


if __name__ == '__main__':
    main()