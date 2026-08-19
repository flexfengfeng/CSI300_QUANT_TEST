#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_s1.ipynb 策略回测
============================================
策略：HK.00700 1分钟K线短线趋势跟踪
  - 快线 = 最新价(近似新bar开盘价)，慢线 = 最近3根已收盘K线收盘价均值
  - 无持仓且 快线>=慢线 → 按开盘价买入1手
  - 持多且 快线<慢线  → 按开盘价全部平仓(不开空/不加仓)
成本：富途港股标准费率 + 默认5bps滑点
数据：moomoo OpenD 历史K线(自动分页拉取并缓存为CSV)
"""

import os
import sys
import math
import numpy as np
import pandas as pd

from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)

# ============================ 参数 ============================
HOST, PORT = '127.0.0.1', 11111
CODE = 'HK.00700'
KLINE = KLType.K_1M
START_DATE = '2025-06-01'   # 历史数据起始日
END_DATE = None             # None = 最新
CACHE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hk_00700_1m.csv')
FORCE_REFRESH = False

LOT_SIZE = 100            # 每手股数(HK.00700 = 100)
INIT_CASH = 100_000.0     # 初始资金 HKD

# 交易成本（富途港股标准费率）
COMMISSION_RATE = 0.0003    # 佣金 0.03%
COMMISSION_MIN = 5.0        # 最低佣金 HKD/笔
STAMP_DUTY = 0.001          # 印花税 0.1%
TRADE_FEE = 0.0000565       # 交易费 0.00565%
LEVY = 0.000027             # 交易征费 0.0027%
SYSTEM_FEE = 0.5            # 交易系统使用费 HKD/笔
SLIPPAGE = 0.0005           # 滑点默认 5 bps

FAST_N = 1   # 快线=最新价
SLOW_N = 3   # 慢线=N根均值

# ============================ 数据拉取 ============================
def download_history():
    """从 OpenD 分页拉取 1 分钟 K 线，缓存为 CSV。"""
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    frames, page_key = [], None
    page = 0
    while True:
        page += 1
        ret, data, page_key = ctx.request_history_kline(
            code=CODE, start=START_DATE, end=END_DATE, ktype=KLINE,
            autype=AuType.QFQ,
            fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                    KL_FIELD.LOW, KL_FIELD.CLOSE, KL_FIELD.TRADE_VOL,
                    KL_FIELD.TRADE_VAL],
            max_count=1000, page_req_key=page_key)
        if ret != RET_OK:
            print(f'[下载] 第{page}页失败: {data}', file=sys.stderr)
            break
        frames.append(data)
        print(f'[下载] 第{page}页: {len(data)} 根, 范围 {data["time_key"].iloc[0]} ~ {data["time_key"].iloc[-1]}')
        if page_key is None:
            break
    ctx.close()

    if not frames:
        raise RuntimeError('未拉到任何K线数据')

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.to_csv(CACHE_CSV, index=False)
    print(f'[下载] 共 {len(df)} 根K线, 缓存到 {CACHE_CSV}')
    return df


def load_data():
    if (not FORCE_REFRESH) and os.path.exists(CACHE_CSV):
        df = pd.read_csv(CACHE_CSV, parse_dates=['time_key'])
        print(f'[加载] 使用缓存 {CACHE_CSV}: {len(df)} 根K线')
        return df
    return download_history()


# ============================ 交易成本 ============================
def cost_of_trade_price(notional: float) -> float:
    """计算单笔交易的各项成本合计(HKD)。"""
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    stamp = notional * STAMP_DUTY
    fee = notional * TRADE_FEE
    levy = notional * LEVY
    return commission + stamp + fee + levy + SYSTEM_FEE


# ============================ 回测主逻辑 ============================
def run_backtest(df: pd.DataFrame):
    """逐根1分钟K线复刻策略。返回净值序列与交易记录。"""
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    times = df['time_key'].to_numpy()

    n = len(df)
    cash = INIT_CASH
    pos = 0                 # 持仓股数(0 或 LOT_SIZE)
    cost_basis = 0.0        # 当前持仓总成本
    realized_pnl = 0.0      # 已实现盈亏
    trades = []             # 完整交易(开→平)记录
    fills = []              # 每笔成交记录
    equity_curve = []       # (time, equity)
    open_trade = None       # 当前未平交易信息

    def mark_equity(ts, px):
        equity_curve.append((ts, cash + pos * px))

    for i in range(n):
        ts, o, c = times[i], opens[i], closes[i]
        # 每分钟记录一次净值(收盘价计)
        mark_equity(ts, c)

        # 需要至少 SLOW_N+1 根K线才能计算慢线
        if i < SLOW_N:
            continue

        # 信号(忠实复刻 test_s1.ipynb 的 calculate_bull_bear)：
        # K线反转后取 list[1:fast+1] 与 list[1:slow+1]
        # 即 快线=上一根已收盘K线收盘价；慢线=前3根已收盘K线收盘均值(不含当前新bar)
        slow = float(np.mean(closes[i - SLOW_N:i]))
        fast = closes[i - 1]
        signal = 1 if fast >= slow else -1

        if pos == 0:
            if signal == 1:
                # 开仓：按开盘价+滑点买入1手
                buy_px = o * (1 + SLIPPAGE)
                qty = LOT_SIZE
                notional = buy_px * qty
                cost = cost_of_trade_price(notional)
                if notional + cost <= cash:
                    cash -= (notional + cost)
                    pos = qty
                    cost_basis = notional
                    realized_pnl -= cost
                    fills.append({'time': ts, 'action': 'BUY', 'price': round(buy_px, 2),
                                  'qty': qty, 'cost': round(cost, 2),
                                  'cash': round(cash, 2), 'notional': round(notional, 2)})
                    open_trade = {'buy_time': ts, 'buy_px': buy_px, 'qty': qty,
                                  'buy_cost': cost}
                # 资金不足则什么都不做
        else:
            if signal == -1:
                # 平仓：按开盘价-滑点全部卖出
                sell_px = o * (1 - SLIPPAGE)
                qty = pos
                notional = sell_px * qty
                cost = cost_of_trade_price(notional)
                cash += (notional - cost)
                gross_pnl = notional - cost_basis
                realized_pnl += gross_pnl - cost
                fills.append({'time': ts, 'action': 'SELL', 'price': round(sell_px, 2),
                              'qty': qty, 'cost': round(cost, 2),
                              'cash': round(cash, 2), 'notional': round(notional, 2)})
                hold_seconds = (pd.Timestamp(ts) - pd.Timestamp(open_trade['buy_time'])).total_seconds()
                trades.append({
                    'buy_time': pd.Timestamp(open_trade['buy_time']),
                    'sell_time': ts,
                    'buy_px': round(open_trade['buy_px'], 2),
                    'sell_px': round(sell_px, 2),
                    'qty': qty,
                    'gross_pnl': round(gross_pnl, 2),
                    'total_cost': round(open_trade['buy_cost'] + cost, 2),
                    'net_pnl': round(gross_pnl - cost - open_trade['buy_cost'], 2),
                    'ret_pct': round((gross_pnl - cost - open_trade['buy_cost']) / INIT_CASH * 100, 3),
                    'hold_min': int(hold_seconds // 60),
                })
                pos = 0
                cost_basis = 0.0
                open_trade = None
            # 多头信号不加仓

    # 回测结束：若仍持仓，按最后收盘价强制平仓(计入统计)
    if pos > 0:
        ts = times[-1]
        o = closes[-1]
        sell_px = o * (1 - SLIPPAGE)
        notional = sell_px * pos
        cost = cost_of_trade_price(notional)
        cash += (notional - cost)
        gross_pnl = notional - cost_basis
        realized_pnl += gross_pnl - cost
        fills.append({'time': ts, 'action': 'SELL(末日平仓)', 'price': round(sell_px, 2),
                      'qty': pos, 'cost': round(cost, 2),
                      'cash': round(cash, 2), 'notional': round(notional, 2)})
        hold_seconds = (pd.Timestamp(ts) - pd.Timestamp(open_trade['buy_time'])).total_seconds()
        trades.append({
            'buy_time': pd.Timestamp(open_trade['buy_time']), 'sell_time': ts,
            'buy_px': round(open_trade['buy_px'], 2), 'sell_px': round(sell_px, 2),
            'qty': pos, 'gross_pnl': round(gross_pnl, 2),
            'total_cost': round(open_trade['buy_cost'] + cost, 2),
            'net_pnl': round(gross_pnl - cost - open_trade['buy_cost'], 2),
            'ret_pct': round((gross_pnl - cost - open_trade['buy_cost']) / INIT_CASH * 100, 3),
            'hold_min': int(hold_seconds // 60),
        })
        pos = 0

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    fills_df = pd.DataFrame(fills)
    trades_df = pd.DataFrame(trades)
    return equity_df, fills_df, trades_df, cash, realized_pnl


# ============================ 绩效指标 ============================
def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                    final_cash: float, realized_pnl: float):
    eq = equity_df.set_index('time')['equity']
    total_ret = eq.iloc[-1] / INIT_CASH - 1

    # 年化：按交易日(约252天/年)计算
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
    years = max(days / 365.0, 1e-9)
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1.0

    # 最大回撤
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1
    max_dd = drawdown.min()

    # 日收益 -> 波动率 / 夏普
    daily_eq = eq.resample('1D').last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = np.nan
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(252)

    # 交易统计
    n_trades = len(trades_df)
    wins = trades_df[trades_df['net_pnl'] > 0] if n_trades else pd.DataFrame()
    win_rate = len(wins) / n_trades if n_trades else 0.0
    avg_win = wins['net_pnl'].mean() if len(wins) else 0.0
    losses = trades_df[trades_df['net_pnl'] <= 0] if n_trades else pd.DataFrame()
    avg_loss = losses['net_pnl'].mean() if len(losses) else 0.0
    profit_factor = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
                     if len(losses) and losses['net_pnl'].sum() != 0 else math.inf if len(wins) else 0.0)
    avg_hold = trades_df['hold_min'].mean() if n_trades else 0.0

    return {
        'start': eq.index[0], 'end': eq.index[-1],
        'days': days, 'n_kline': len(eq),
        'init_cash': INIT_CASH, 'final_equity': eq.iloc[-1],
        'total_ret': total_ret, 'annual_ret': annual_ret,
        'max_drawdown': max_dd, 'sharpe': sharpe,
        'n_trades': n_trades, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_hold_min': avg_hold, 'realized_pnl': realized_pnl,
    }


def plot_equity(equity_df: pd.DataFrame, title: str, save_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print('[绘图] matplotlib 未安装，跳过图表生成')
        return

    eq = equity_df.set_index('time')['equity']
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    # 净值曲线
    axes[0].plot(eq.index, eq.values, lw=1.1, color='#1f77b4')
    axes[0].axhline(INIT_CASH, color='gray', lw=0.8, ls='--', alpha=0.7)
    axes[0].set_title(title, fontsize=12)
    axes[0].set_ylabel('Equity (HKD)')
    axes[0].grid(alpha=0.3)

    # 回撤
    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    axes[1].fill_between(dd.index, dd.values, 0, color='#d62728', alpha=0.4)
    axes[1].set_ylabel('Drawdown')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f'[绘图] 净值曲线已保存: {save_path}')


# ============================ 主流程 ============================
def main():
    print('=' * 70)
    print('回测 test_s1 策略：价格 vs 3分钟均线 短线趋势跟踪 (HK.00700 1分钟线)')
    print('=' * 70)

    df = load_data()
    print(f'[数据] 范围 {df["time_key"].iloc[0]} ~ {df["time_key"].iloc[-1]}, '
          f'共 {len(df)} 根K线, {df["time_key"].dt.date.nunique()} 个交易日')

    equity_df, fills_df, trades_df, final_cash, realized_pnl = run_backtest(df)
    metrics = compute_metrics(equity_df, trades_df, final_cash, realized_pnl)

    # ---- 打印交易明细 ----
    print('\n' + '=' * 70)
    print(f'成交明细 (前24条, 共{len(fills_df)}条; 全部见 fills.csv)')
    print('=' * 70)
    if len(fills_df):
        preview = fills_df.head(24)
        print(preview.to_string(index=False))
        if len(fills_df) > 24:
            print(f'... (其余 {len(fills_df)-24} 条已省略)')
    else:
        print('(无任何成交)')

    print('\n' + '=' * 70)
    print('完整交易记录 (开→平)')
    print('=' * 70)
    if len(trades_df):
        cols = ['buy_time', 'sell_time', 'buy_px', 'sell_px', 'qty',
                'gross_pnl', 'total_cost', 'net_pnl', 'hold_min']
        print(trades_df[cols].to_string(index=False))
    else:
        print('(无完整交易)')

    # ---- 绩效指标 ----
    print('\n' + '=' * 70)
    print('绩效指标')
    print('=' * 70)
    rows = [
        ('回测区间', f'{metrics["start"]} ~ {metrics["end"]}'),
        ('K线数量', f'{metrics["n_kline"]:,}'),
        ('回测天数', f'{metrics["days"]:.1f} 天'),
        ('初始资金', f'{metrics["init_cash"]:,.0f} HKD'),
        ('期末净值', f'{metrics["final_equity"]:,.2f} HKD'),
        ('已实现盈亏(含成本)', f'{metrics["realized_pnl"]:+,.2f} HKD'),
        ('总收益率', f'{metrics["total_ret"]*100:+.2f}%'),
        ('年化收益率', f'{metrics["annual_ret"]*100:+.2f}%'),
        ('最大回撤', f'{metrics["max_drawdown"]*100:.2f}%'),
        ('夏普比率(日频年化)', f'{metrics["sharpe"]:.2f}' if not math.isnan(metrics["sharpe"]) else 'N/A'),
        ('交易次数(完整开平)', f'{metrics["n_trades"]}'),
        ('胜率', f'{metrics["win_rate"]*100:.1f}%'),
        ('平均盈利/笔', f'{metrics["avg_win"]:+,.2f} HKD'),
        ('平均亏损/笔', f'{metrics["avg_loss"]:+,.2f} HKD'),
        ('盈亏比(Profit Factor)', f'{metrics["profit_factor"]:.2f}'),
        ('平均持仓时长', f'{metrics["avg_hold_min"]:.1f} 分钟'),
    ]
    for k, v in rows:
        print(f'  {k:<18} {v}')

    # ---- 保存结果 ----
    base = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(base, 'equity_curve.csv'), index=False)
    if len(fills_df):
        fills_df.to_csv(os.path.join(base, 'fills.csv'), index=False)
    if len(trades_df):
        trades_df.to_csv(os.path.join(base, 'trades.csv'), index=False)
    pdf = pd.DataFrame([metrics])
    pdf.to_csv(os.path.join(base, 'metrics.csv'), index=False)
    print(f'\n[保存] 结果已写入 {base}/ 下的 equity_curve.csv / fills.csv / trades.csv / metrics.csv')

    # 绘图
    plot_equity(equity_df,
                title=f'test_s1 Strategy Backtest: {CODE} 1m  (ret={metrics["total_ret"]*100:+.2f}%, '
                      f'MDD={metrics["max_drawdown"]*100:.2f}%, trades={metrics["n_trades"]})',
                save_path=os.path.join(base, 'equity_curve.png'))


if __name__ == '__main__':
    main()