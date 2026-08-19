#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盈富基金(HK.02800) 日内策略研究 —— 强制当日平仓
=================================================
ETF 相比个股 HK.00700 的成本优势：
  1. ETF 免印花税 (单边0.1% × 2 → 0)
  2. 一手仅 500 股(~1.2万), 可加大名义金额摊薄最低佣金
  3. 日均成交额 154亿, 流动性全港最佳 → 滑点可设 2bps(股票用5bps)

成本模型(假设每笔名义≈5万港元=4手):
  佣金  max(0.03%*NV, 5)   + 交易费 0.00565%*NV + 征费 0.0027%*NV + 系统费0.5/笔
  往返 ≈ 0.079% + 滑点 2bps×2=0.04%  ≈ 总摩擦 0.12%
  (对比股票 HK.00700: 总摩擦 ≈ 0.28%)
"""
import os
import datetime as dt
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hk_02800_1m.csv')
LOT = 500                # 盈富基金 500 股/手
TARGET_NV = 50_000.0     # 每笔目标名义金额 HKD
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0
TRADE_FEE, LEVY, SYSTEM_FEE = 0.0000565, 0.000027, 0.5   # 无印花税!
SLIPPAGE = 0.0002        # 2 bps/边 (流动性极好)

# 敏感性: 最乐观成本情景 (大客户佣金 + 超低滑点)
BEST_COMMISSION, BEST_SLIPPAGE, BEST_SYS = 0.0001, 0.0001, 0.0


def cost_of_trade(notional: float, best_case=False, free=False) -> float:
    """ETF 成本: 免印花税。best_case=大客户费率; free=零成本(诊断用)。"""
    if free:
        return 0.0
    if best_case:
        commission = max(notional * BEST_COMMISSION, 5.0)
        return commission + notional * TRADE_FEE + notional * LEVY + BEST_SYS
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return commission + notional * TRADE_FEE + notional * LEVY + SYSTEM_FEE


def round_lot(px, nv):
    """按目标名义金额计算股数并归整到手。"""
    qty = int(round(nv / px / LOT)) * LOT
    return max(qty, LOT)


def load():
    df = pd.read_csv(DATA, parse_dates=['time_key'])
    return df


def is_trading_time(ts):
    t = ts.time()
    return (dt.time(9, 30) <= t <= dt.time(12, 0)) or (dt.time(13, 0) <= t <= dt.time(16, 0))


def aggregate(df, freq):
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
def sig_ma(cfg, i, closes):
    fast = closes[i - (cfg.get('fast', 1) if cfg.get('fast', 1) > 1 else 1)]
    slow = float(np.mean(closes[i - cfg['slow']:i]))
    return 1 if fast >= slow else -1


def sig_meanrev(cfg, i, closes):
    ma = float(np.mean(closes[i - cfg['ma_n']:i]))
    dev = (closes[i - 1] - ma) / ma
    if dev <= -cfg['rev_dev']:
        return 1
    if dev >= cfg['rev_dev']:
        return -1
    return 0


def sig_orb(cfg, i, closes, orb_high, orb_low, orb_done):
    if not orb_done:
        return 0
    if closes[i - 1] > orb_high:
        return 1
    if closes[i - 1] < orb_low:
        return -1
    return 0


# ---------------------- 引擎 ----------------------
def run_cfg(df, cfg):
    data = aggregate(df, cfg.get('agg'))
    closes = data['close'].to_numpy()
    highs = data['high'].to_numpy()
    lows = data['low'].to_numpy()
    opens = data['open'].to_numpy()
    tkeys = data['time_key'].to_numpy()
    n = len(data)

    days = pd.to_datetime(tkeys).date
    last_of_day = np.zeros(n, dtype=bool)
    for k in range(n - 1):
        if days[k] != days[k + 1]:
            last_of_day[k] = True
    last_of_day[-1] = True

    # 当日开盘(9:30)~10:00 振幅 (波动率过滤)
    opn_amp = {}
    opn_idx = {}
    for k in range(n):
        if days[k] not in opn_idx:
            opn_idx[days[k]] = k
    for k in range(n):
        if days[k] in opn_idx and opn_idx[days[k]] == k:
            pass
    # 计算每个交易日的 9:30-10:00 振幅
    for d, idx in opn_idx.items():
        base = highs[idx]
        hi, lo = base, base
        j = idx
        while j < n and days[j] == d and pd.Timestamp(tkeys[j]).time() <= dt.time(10, 0):
            hi = max(hi, highs[j])
            lo = min(lo, lows[j])
            j += 1
        opn_amp[d] = (hi - lo) / base if base > 0 else 0.0

    warmup = cfg.get('slow', 3) if cfg['signal'] == 'ma' else cfg.get('ma_n', 15)
    open_from = dt.time.fromisoformat(cfg['open_win'][0])
    open_until = dt.time.fromisoformat(cfg['open_win'][1])
    close_t = dt.time.fromisoformat(cfg['close'])

    trades = []
    pos = 0
    qty = 0
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

        # ---- 平仓 (SL>TP>TRAIL>SIGNAL>EOD) ----
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
                    if (best - entry_px) / entry_px >= cfg['trailing'][0]:
                        trail_armed = True
                if long and trail_armed and (best - c) / best >= cfg['trailing'][1]:
                    exit_reason, exit_px = 'TRAIL', c
            if exit_reason is None and cfg.get('reverse_exit') and cfg['signal'] == 'ma':
                sig = sig_ma(cfg, i, closes)
                if (long and sig == -1) or (not long and sig == 1):
                    exit_reason, exit_px = 'SIGNAL', o * (1 - SLIPPAGE if long else 1 + SLIPPAGE)
            if exit_reason is None and (tod >= close_t or last_of_day[i]):
                exit_reason, exit_px = 'EOD', o * (1 - SLIPPAGE if long else 1 + SLIPPAGE)

            if exit_reason is not None:
                slip = 0.0 if cfg.get('free') else (BEST_SLIPPAGE if cfg.get('best_case') else SLIPPAGE)
                close_px_eff = exit_px * (1 - slip if long else 1 + slip)
                close_cost = cost_of_trade(abs(close_px_eff * qty),
                                           best_case=cfg.get('best_case'),
                                           free=cfg.get('free'))
                gross = (close_px_eff - entry_px) * qty
                net = gross - entry_cost - close_cost
                trades.append({'exit': exit_reason, 'gross': round(gross, 2),
                               'net': round(net, 2)})
                pos, qty, best, trail_armed = 0, 0, 0.0, False

        # ---- 开仓 ----
        if pos == 0:
            can_open = (day_count < cfg['max_trades']
                        and open_from <= tod <= open_until)
            if cfg['signal'] == 'ma':
                sig = sig_ma(cfg, i, closes)
            elif cfg['signal'] == 'meanrev':
                sig = sig_meanrev(cfg, i, closes)
            else:
                sig = sig_orb(cfg, i, closes, orb_high, orb_low, orb_done)

            want_long = sig == 1 and cfg['direction'] in ('long_only', 'both')
            want_short = sig == -1 and cfg['direction'] == 'both'

            vol_ok = True
            if cfg.get('vol_filter'):
                vol_ok = opn_amp.get(d, 0.0) >= cfg['vol_filter']

            if can_open and vol_ok and (want_long or want_short):
                long = sig == 1
                qty = round_lot(o, TARGET_NV)
                slip = 0.0 if cfg.get('free') else (BEST_SLIPPAGE if cfg.get('best_case') else SLIPPAGE)
                entry_px = o * (1 + slip if long else 1 - slip)
                entry_cost = cost_of_trade(entry_px * qty,
                                           best_case=cfg.get('best_case'),
                                           free=cfg.get('free'))
                pos = qty if long else -qty
                best = entry_px
                trail_armed = False
                if cfg['tp']:
                    tp_px = entry_px * (1 + cfg['tp'] * (1 if long else -1))
                if cfg['sl']:
                    sl_px = entry_px * (1 - cfg['sl'] * (1 if long else -1))
                day_count += 1

    if pos != 0:
        long = pos > 0
        exit_px = closes[-1] * (1 - SLIPPAGE if long else 1 + SLIPPAGE)
        close_cost = cost_of_trade(abs(exit_px * qty))
        gross = (exit_px - entry_px) * pos
        trades.append({'exit': 'EOD(end)', 'gross': round(gross, 2),
                       'net': round(gross - entry_cost - close_cost, 2)})
    return trades


def summarize(cfg, trades):
    n = len(trades)
    if n == 0:
        return cfg['name'], 0, 0.0, 0.0, 0.0, 0.0, ''
    nets = np.array([t['net'] for t in trades])
    wins = nets[nets > 0]
    loss = nets[nets <= 0]
    wr = len(wins) / n
    total = nets.sum()
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float('inf')
    by = {}
    for t in trades:
        by[t['exit']] = by.get(t['exit'], 0) + 1
    return cfg['name'], n, wr * 100, total, total / n, pf, ','.join(f'{k}:{v}' for k, v in sorted(by.items()))


def grid_meanrev():
    """均值回归 TP/SL 网格。"""
    cfgs = []
    for tp in (0.003, 0.005, 0.008):
        for sl in (0.002, 0.003, 0.005):
            cfgs.append(dict(
                name=f'MR TP{tp*100:.1f}%/SL{sl*100:.2f}% 双向',
                signal='meanrev', ma_n=15, rev_dev=0.0025, direction='both',
                reverse_exit=False, tp=tp, sl=sl,
                open_win=('09:30', '15:30'), max_trades=6, close='15:58:00'))
    return cfgs


def main():
    df = load()
    px = df['close'].mean()
    print('= 盈富基金 HK.02800 日内研究 · %s ~ %s · %d 根K线 =' %
          (df['time_key'].iloc[0], df['time_key'].iloc[-1], len(df)))
    print('  均价 %.2f, 每手500股≈%.0f HKD, 每笔名义≈%.0f股/%.0f HKD' %
          (px, px * LOT, TARGET_NV / px / LOT * LOT, TARGET_NV))
    one = cost_of_trade(px * int(round(TARGET_NV / px / LOT)) * LOT)
    print('  单边成本 ≈ %.1f HKD ; 往返摩擦 ≈ %.3f%% (标准费率+2bps滑点)' %
          (one, (one + one + 0.0002 * 2 * TARGET_NV) / TARGET_NV * 100))
    one_best = cost_of_trade(px * int(round(TARGET_NV / px / LOT)) * LOT, best_case=True)
    print('  最优成本情景(佣金0.01 + 1bp滑点 + 无系统费): 往返摩擦 ≈ %.3f%%' %
          ((one_best + one_best + 0.0001 * 2 * TARGET_NV) / TARGET_NV * 100))
    print()

    cfgs = []
    cfgs.append(dict(name='① 原版MA(1,3)信号反转平|当日平', signal='ma', slow=3,
                     direction='long_only', reverse_exit=True, tp=0, sl=0,
                     open_win=('09:30', '15:58'), max_trades=999, close='15:58:00'))
    cfgs.append(dict(name='② 开盘60分MA+移止(0.3/0.2)+SL0.25% 1次/天', signal='ma', slow=3,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.0025,
                     trailing=(0.003, 0.002), open_win=('09:30', '10:30'),
                     max_trades=1, close='15:58:00'))
    cfgs.append(dict(name='③ ORB前15分+TP0.4%/SL0.4% 双向', signal='orb',
                     direction='both', reverse_exit=False, tp=0.004, sl=0.004,
                     open_win=('09:45', '15:00'), max_trades=2, close='15:58:00'))
    cfgs.append(dict(name='④ 5分MA60+移止(0.8/0.3)+SL0.4% 1次/天 (5m)', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min'))
    cfgs.extend(grid_meanrev())

    # ---- 诊断: 零成本(无任何费用/滑点) → 纯信号毛期望 ----
    cfgs.append(dict(name='◆ 零成本诊断: ORB+TP0.4/SL0.4 双向', signal='orb',
                     direction='both', reverse_exit=False, tp=0.004, sl=0.004,
                     open_win=('09:45', '15:00'), max_trades=2, close='15:58:00',
                     free=True))
    cfgs.append(dict(name='◆ 零成本诊断: MR TP0.5/SL0.3 双向', signal='meanrev', ma_n=15,
                     rev_dev=0.0025, direction='both', reverse_exit=False, tp=0.005, sl=0.003,
                     open_win=('09:30', '15:30'), max_trades=6, close='15:58:00',
                     free=True))
    cfgs.append(dict(name='◆ 零成本诊断: 5分MA60移止(0.8/0.3)+SL0.4', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min', free=True))
    cfgs.append(dict(name='◆ 零成本: 5分MA60移止 + 波动过滤(≥0.3%)', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min', free=True,
                     vol_filter=0.003))
    cfgs.append(dict(name='◆ 零成本: 5分MA60移止 + 波动过滤(≥0.5%)', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min', free=True,
                     vol_filter=0.005))

    # ---- 天花板测试: 最乐观成本 ----
    cfgs.append(dict(name='★ 最优成本: ORB+TP0.4/SL0.4 双向 [佣金0.01/滑点1bp]', signal='orb',
                     direction='both', reverse_exit=False, tp=0.004, sl=0.004,
                     open_win=('09:45', '15:00'), max_trades=2, close='15:58:00',
                     best_case=True))
    cfgs.append(dict(name='★ 最优成本: MR TP0.5/SL0.3 双向 [佣金0.01/滑点1bp]', signal='meanrev', ma_n=15,
                     rev_dev=0.0025, direction='both', reverse_exit=False, tp=0.005, sl=0.003,
                     open_win=('09:30', '15:30'), max_trades=6, close='15:58:00',
                     best_case=True))
    cfgs.append(dict(name='★ 最优成本: 5分MA60移止(0.8/0.3)+SL0.4 [佣金0.01/滑点1bp]', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min', best_case=True))
    cfgs.append(dict(name='★ 最优成本: 5分MA60移止 + 波动过滤(≥0.3%) [佣金0.01/滑点1bp]', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min', best_case=True,
                     vol_filter=0.003))
    cfgs.append(dict(name='⑤ 标准成本: 5分MA60移止 + 波动过滤(≥0.3%) (5m)', signal='ma', fast=1, slow=60,
                     direction='long_only', reverse_exit=False, tp=0, sl=0.004,
                     trailing=(0.008, 0.003), open_win=('09:30', '15:00'),
                     max_trades=1, close='15:58:00', agg='5min',
                     vol_filter=0.003))

    print('%-52s %6s %7s %10s %9s %7s %s' % (
        '策略', '交易数', '胜率%', '总盈亏HKD', '均值/笔', '盈亏比', '出场分布'))
    print('-' * 140)
    for cfg in cfgs:
        trades = run_cfg(df, cfg)
        name, n, wr, total, avg, pf, rstr = summarize(cfg, trades)
        print('%-52s %6d %6.1f%% %+10.0f %+9.1f %7.2f %s'
              % (name, n, wr, total, avg, pf, rstr))

    print()
    print('注: 所有策略 15:58 前强制平仓, 无隔夜; 胜率按净盈亏>0 计; 成本已含免印花税后的ETF费率+2bps滑点')


if __name__ == '__main__':
    main()