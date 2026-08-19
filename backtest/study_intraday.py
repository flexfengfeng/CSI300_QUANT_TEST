#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内策略研究（HK.00700, 强制当日平仓）
============================================
目标：在"当天必须平仓"约束下，探索高胜率策略的可行路径。
对比维度：交易次数 / 胜率(按净盈亏) / 总盈亏 / 单笔期望 / 盈亏比 / 出场分布。

成本：佣金0.03%(最低5) + 卖出印花税0.1% + 交易费0.00565% + 征费0.0027%
      + 系统费0.5 + 双边滑点各5bps
  → 每手(100股, 名义约4.9万)往返摩擦 ≈ 147 HKD ≈ 名义金额 0.28%

数据：backtest/hk_00700_1m.csv (2025-06-02 ~ 2026-06-01, 1分钟)
约束：所有持仓在每一交易日的最后一根 bar(或15:58) 强制平仓, 永不过夜。
"""
import os
import datetime as dt
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hk_00700_1m.csv')
LOT = 100
INIT_CASH = 100_000.0
EST_PRICE = 490.0

COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0
STAMP_DUTY, TRADE_FEE, LEVY, SYSTEM_FEE = 0.001, 0.0000565, 0.000027, 0.5
SLIPPAGE = 0.0005


def cost_of_trade(notional: float) -> float:
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return (commission + notional * STAMP_DUTY + notional * TRADE_FEE
            + notional * LEVY + SYSTEM_FEE)


def is_trading_time(ts):
    t = ts.time()
    return (dt.time(9, 30) <= t <= dt.time(12, 0)) or (dt.time(13, 0) <= t <= dt.time(16, 0))


def load_df():
    df = pd.read_csv(DATA, parse_dates=['time_key'])
    return df


def aggregate(df, freq):
    """按日分组重采样到 freq(如'5T'), 丢弃跨休市/夜间的bridging bar。"""
    if freq is None:
        df = df.copy()
    else:
        groups = []
        for _, g in df.groupby(df['time_key'].dt.date):
            agg = (g.set_index('time_key').resample(freq)
                     .agg({'open': 'first', 'high': 'max', 'low': 'min',
                           'close': 'last', 'volume': 'sum', 'turnover': 'sum'})
                     .dropna(subset=['open']))
            groups.append(agg)
        df = pd.concat(groups).reset_index()
    df = df[df['time_key'].map(is_trading_time)].reset_index(drop=True)
    return df


# ---------------------- 信号 ----------------------
def signal_ma(cfg, i, closes):
    fast_n = cfg.get('fast', 1)
    fast = closes[i - fast_n] if fast_n > 1 else closes[i - 1]
    slow = float(np.mean(closes[i - cfg['slow']:i]))
    return 1 if fast >= slow else -1


def signal_meanrev(cfg, i, closes):
    ma = float(np.mean(closes[i - cfg['ma_n']:i]))
    dev = (closes[i - 1] - ma) / ma
    if dev <= -cfg['rev_dev']:
        return 1
    if dev >= cfg['rev_dev']:
        return -1
    return 0


def signal_orb(cfg, i, closes, orb_high, orb_low, orb_done):
    if not orb_done:
        return 0
    if closes[i - 1] > orb_high:
        return 1
    if closes[i - 1] < orb_low:
        return -1
    return 0


# ---------------------- 通用回测引擎 ----------------------
def run_cfg(df, cfg):
    data = aggregate(df, cfg.get('agg'))
    closes = data['close'].to_numpy()
    highs = data['high'].to_numpy()
    lows = data['low'].to_numpy()
    opens = data['open'].to_numpy()
    tkeys = data['time_key'].to_numpy()
    n = len(data)

    # 每日最后一根 bar 标记(用于强制当日平仓)
    days = pd.to_datetime(tkeys).date
    last_of_day = np.zeros(n, dtype=bool)
    for k in range(n - 1):
        if days[k] != days[k + 1]:
            last_of_day[k] = True
    last_of_day[-1] = True

    # 每个交易日的开盘价 / 前一交易日收盘价(跳空过滤用)
    day_first_idx = {}
    for k in range(n):
        if days[k] not in day_first_idx:
            day_first_idx[days[k]] = k
    prev_close = {}
    day_list = list(day_first_idx)
    closes_by_day = {}
    for k in range(n):
        closes_by_day.setdefault(days[k], []).append(closes[k])
    for j, d in enumerate(day_list):
        if j > 0:
            prev_close[d] = closes_by_day[day_list[j - 1]][-1]

    warmup = cfg.get('slow', 3) if cfg['signal'] == 'ma' else cfg.get('ma_n', 15)
    open_from = dt.time.fromisoformat(cfg['open_win'][0])
    open_until = dt.time.fromisoformat(cfg['open_win'][1])
    close_t = dt.time.fromisoformat(cfg['close'])

    trades = []
    pos = 0
    entry_px = 0.0
    entry_cost = 0.0
    tp_px = 0.0
    sl_px = 0.0
    best = 0.0
    trail_armed = False
    day = None
    day_count = 0
    orb_high, orb_low, orb_done = 0.0, 1e18, False

    for i in range(n):
        ts = pd.Timestamp(tkeys[i])
        d = ts.date()
        tod = ts.time()
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        if d != day:
            day = d
            day_count = 0
            orb_high, orb_low, orb_done = 0.0, 1e18, False

        if cfg['signal'] == 'orb':
            if tod < dt.time(9, 45):
                orb_high = max(orb_high, h)
                orb_low = min(orb_low, l)
            elif not orb_done and orb_high > 0 and orb_low < 1e18:
                orb_done = True

        if i < warmup:
            continue

        # ---------- 平仓 (独立判断, 优先级 SL>TP>TRAIL>反转>EOD) ----------
        if pos != 0:
            long = pos > 0
            exit_reason, exit_px = None, None

            if cfg['sl'] and ((long and l <= sl_px) or (not long and h >= sl_px)):
                exit_reason, exit_px = 'SL', sl_px
            if exit_reason is None and cfg['tp'] and ((long and h >= tp_px) or (not long and l <= tp_px)):
                exit_reason, exit_px = 'TP', tp_px
            if exit_reason is None and cfg.get('trailing'):
                if long and c > best:
                    best = c
                    if (best - entry_px) / entry_px >= cfg['trailing']['activate']:
                        trail_armed = True
                if long and trail_armed and (best - c) / best >= cfg['trailing']['pullback']:
                    exit_reason, exit_px = 'TRAIL', c
            if exit_reason is None and cfg.get('reverse_exit') and cfg['signal'] == 'ma':
                sig = signal_ma(cfg, i, closes)
                if (long and sig == -1) or (not long and sig == 1):
                    exit_reason, exit_px = 'SIGNAL', o * (1 - SLIPPAGE if long else 1 + SLIPPAGE)
            if exit_reason is None and (tod >= close_t or last_of_day[i]):
                exit_reason, exit_px = 'EOD', o * (1 - SLIPPAGE if long else 1 + SLIPPAGE)

            if exit_reason is not None:
                close_cost = cost_of_trade(abs(exit_px * pos))
                gross = (exit_px - entry_px) * pos
                net = gross - entry_cost - close_cost
                trades.append({'exit': exit_reason, 'gross': round(gross, 2),
                               'net': round(net, 2)})
                pos, best, trail_armed = 0, 0.0, False

        # ---------- 开仓 ----------
        if pos == 0:
            can_open = (day_count < cfg['max_trades']
                        and open_from <= tod <= open_until)
            if cfg['signal'] == 'ma':
                sig = signal_ma(cfg, i, closes)
            elif cfg['signal'] == 'meanrev':
                sig = signal_meanrev(cfg, i, closes)
            else:
                sig = signal_orb(cfg, i, closes, orb_high, orb_low, orb_done)

            want_long = sig == 1 and cfg['direction'] in ('long_only', 'both')
            want_short = sig == -1 and cfg['direction'] == 'both'

            gap_ok = True
            if cfg.get('gap_up_only'):
                gap_ok = d in prev_close and o > prev_close[d]

            if can_open and gap_ok and (want_long or want_short):
                long = sig == 1
                entry_px = o * (1 + SLIPPAGE if long else 1 - SLIPPAGE)
                entry_cost = cost_of_trade(entry_px * LOT)
                pos = LOT if long else -LOT
                best = entry_px
                trail_armed = False
                if cfg['tp']:
                    tp_px = entry_px * (1 + cfg['tp'] * (1 if long else -1))
                if cfg['sl']:
                    sl_px = entry_px * (1 - cfg['sl'] * (1 if long else -1))
                day_count += 1

    # 末段兜底平仓
    if pos != 0:
        long = pos > 0
        exit_px = closes[-1] * (1 - SLIPPAGE if long else 1 + SLIPPAGE)
        close_cost = cost_of_trade(abs(exit_px * pos))
        gross = (exit_px - entry_px) * pos
        trades.append({'exit': 'EOD(end)', 'gross': round(gross, 2),
                       'net': round(gross - entry_cost - close_cost, 2)})

    return trades


# ---------------------- 统计 ----------------------
def summarize(cfg, trades):
    n = len(trades)
    if n == 0:
        return cfg['name'], 0, 0.0, 0.0, 0.0, 0.0, 0.0
    nets = np.array([t['net'] for t in trades])
    wins = nets[nets > 0]
    loss = nets[nets <= 0]
    wr = len(wins) / n
    total = nets.sum()
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float('inf')
    by_reason = {}
    for t in trades:
        by_reason[t['exit']] = by_reason.get(t['exit'], 0) + 1
    return cfg['name'], n, wr * 100, total, total / n, pf, by_reason


# ---------------------- 配置 ----------------------
CONFIGS = [
    dict(name='① 原版MA(1,3)+信号反转平|15:58平', signal='ma', slow=3,
         direction='long_only', reverse_exit=True, tp=0, sl=0,
         open_win=('09:30', '15:58'), max_trades=999, close='15:58:00'),
    dict(name='② MA(1,3)+TP0.2%/SL0.5% (窄止盈冲胜率)', signal='ma', slow=3,
         direction='long_only', reverse_exit=False, tp=0.002, sl=0.005,
         open_win=('09:30', '15:58'), max_trades=999, close='15:58:00'),
    dict(name='③ MA(1,3)+TP0.3%/SL0.3% (对称)', signal='ma', slow=3,
         direction='long_only', reverse_exit=False, tp=0.003, sl=0.003,
         open_win=('09:30', '15:58'), max_trades=999, close='15:58:00'),
    dict(name='④ MA(1,3)+TP0.5%/SL0.2% (宽止盈)', signal='ma', slow=3,
         direction='long_only', reverse_exit=False, tp=0.005, sl=0.002,
         open_win=('09:30', '15:58'), max_trades=999, close='15:58:00'),
    dict(name='⑤ 开盘60分MA+TP0.4%/SL0.25%+1次/天', signal='ma', slow=3,
         direction='long_only', reverse_exit=False, tp=0.004, sl=0.0025,
         open_win=('09:30', '10:30'), max_trades=1, close='15:58:00'),
    dict(name='⑥ 开盘60分MA+移止(0.3%/0.2%)+SL0.25%+1次/天', signal='ma', slow=3,
         direction='long_only', reverse_exit=False, tp=0, sl=0.0025,
         trailing={'activate': 0.003, 'pullback': 0.002},
         open_win=('09:30', '10:30'), max_trades=1, close='15:58:00'),
    dict(name='⑦ ORB前15分+TP0.4%/SL0.4% 双向', signal='orb',
         direction='both', reverse_exit=False, tp=0.004, sl=0.004,
         open_win=('09:45', '15:00'), max_trades=2, close='15:58:00'),
    dict(name='⑧ 均值回归MA15偏0.25%+TP0.3%/SL0.5% 双向', signal='meanrev', ma_n=15,
         rev_dev=0.0025, direction='both', reverse_exit=False, tp=0.003, sl=0.005,
         open_win=('09:30', '15:30'), max_trades=6, close='15:58:00'),
    # ---- 日内趋势 (5分钟bar, 大毛收益目标) ----
    dict(name='⑨ 5分MA60上穿+TP1.0%/SL0.35% 多向 1次/天 (5m)', signal='ma', fast=1, slow=60,
         direction='long_only', reverse_exit=False, tp=0.010, sl=0.0035,
         open_win=('09:30', '15:00'), max_trades=1, close='15:58:00', agg='5min'),
    dict(name='⑩ 5分MA60+移止(0.8%/0.3%)+SL0.4% 1次/天 (5m, 不止盈)', signal='ma', fast=1, slow=60,
         direction='long_only', reverse_exit=False, tp=0, sl=0.004,
         trailing={'activate': 0.008, 'pullback': 0.003},
         open_win=('09:30', '15:00'), max_trades=1, close='15:58:00', agg='5min'),
    dict(name='⑪ 同⑨但仅跳空高开日做多 (5m, 趋势过滤)', signal='ma', fast=1, slow=60,
         direction='long_only', reverse_exit=False, tp=0.010, sl=0.0035,
         open_win=('09:30', '15:00'), max_trades=1, close='15:58:00',
         agg='5min', gap_up_only=True),
]


def main():
    df = load_df()
    print('= 日内策略对比研究 · HK.00700 1分钟 (%s ~ %s, %d 根K线) ='
          % (df['time_key'].iloc[0], df['time_key'].iloc[-1], len(df)))
    one_hand_rt = cost_of_trade(EST_PRICE * LOT) * 2
    print('  每手(100股)往返摩擦成本 ≈ %.0f HKD ≈ %.3f%% 名义金额 (含5bps滑点)'
          % (one_hand_rt, one_hand_rt / (EST_PRICE * LOT) * 100))
    print()

    print('%-62s %6s %7s %10s %9s %7s %-24s' % (
        '策略', '交易数', '胜率%', '总盈亏HKD', '均值/笔', '盈亏比', '出场分布'))
    print('-' * 136)

    for cfg in CONFIGS:
        trades = run_cfg(df, cfg)
        name, n, wr, total, avg, pf, reasons = summarize(cfg, trades)
        rstr = ','.join(f'{k}:{v}' for k, v in sorted(reasons.items()))
        print('%-62s %6d %6.1f%% %+10.0f %+9.1f %7.2f %-24s'
              % (name, n, wr, total, avg, pf, rstr))

    print()
    print('解读：')
    print(' - 胜率按【净盈亏>0】计算(已扣全部费用+滑点)')
    print(' - 出场分布: TP=止盈 SL=止损 TRAIL=移动止盈 EOD=收盘前强平 SIGNAL=信号反转平仓')
    print(' - 所有策略在每交易日最后一根bar前强制平仓, 无隔夜持仓')
    print(' - (5m)=在5分钟bar上运行, 其余为1分钟bar')


if __name__ == '__main__':
    main()