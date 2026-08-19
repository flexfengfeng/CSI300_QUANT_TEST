#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周线级持仓趋势回测 —— 腾讯(HK.00700) & 盈富(HK.02800)
====================================================
问题：不要求日内平仓，持仓周期拉长到一周级别，结果如何？

设计：
  - 日线 K 线, 覆盖约4年(2022-06 ~ 至今, 视OpenD数据深度)
  - 趋势信号: 快线MA(fast) 上穿/站上 慢线MA(slow) → 做多; 下穿 → 平多/反手
  - 允许持仓跨日/跨周/跨月, 唯一离场 = 信号反转 或 可选止盈止损
  - 成本: 开+平两笔(腾讯含印花税0.1%/边, 盈富免); 每笔名义10万HKD
  - 对比: 只多 vs 双向, 不同快慢MA, 是否加SL

数据: 首次运行自动从 OpenD 拉取并缓存为CSV (backtest/daily_00700.csv / daily_02800.csv)
"""
import os
import sys
import math
import numpy as np
import pandas as pd

from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)

BASE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = '127.0.0.1', 11111
START_DATE = '2014-01-01'
END_DATE = '2026-06-02'

# ---------------- 标的参数 ----------------
UNIVERSE = {
    'HK.00700': dict(name='腾讯', csv='daily_00700.csv', commission=0.0003,
                     stamp=0.001, slip=0.0005, min_comm=5.0),
    'HK.02800': dict(name='盈富', csv='daily_02800.csv', commission=0.0003,
                     stamp=0.0, slip=0.0002, min_comm=5.0),
}
TARGET_NV = 100_000.0   # 每笔目标名义
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5

# ---------------- 数据 ----------------
def fetch(codes):
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    for code in codes:
        meta = UNIVERSE[code]
        path = os.path.join(BASE, meta['csv'])
        if os.path.exists(path):
            print(f'[缓存] {code} 使用 {path}')
            continue
        frames, page_key = [], None
        while True:
            ret, data, page_key = ctx.request_history_kline(
                code=code, start=START_DATE, end=END_DATE,
                ktype=KLType.K_DAY, autype=AuType.QFQ,
                fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                        KL_FIELD.LOW, KL_FIELD.CLOSE, KL_FIELD.TRADE_VOL,
                        KL_FIELD.TRADE_VAL],
                max_count=None, page_req_key=page_key)
            if ret != RET_OK:
                print(f'[下载] {code} 失败: {data}', file=sys.stderr)
                break
            frames.append(data)
            print(f'[下载] {code} 第{len(frames)}页: {len(data)}根 '
                  f'{data["time_key"].iloc[0]} ~ {data["time_key"].iloc[-1]}')
            if page_key is None:
                break
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
            df['time_key'] = pd.to_datetime(df['time_key'])
            df.to_csv(path, index=False)
            print(f'[下载] {code} 共{len(df)}根 已缓存 {path}')
    ctx.close()


# ---------------- 成本 ----------------
def cost_of_trade(code, notional):
    m = UNIVERSE[code]
    commission = max(notional * m['commission'], m['min_comm'])
    return commission + notional * m['stamp'] + notional * TRADE_FEE_R \
        + notional * LEVY_R + SYS_FEE


def round_lot(code, px, nv):
    lot = 100 if code == 'HK.00700' else 500
    qty = int(round(nv / px / lot)) * lot
    return max(qty, lot)


# ---------------- 回测 ----------------
def run_trend(code, df, fast, slow, allow_short=False, sl=None):
    closes = df['close'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    opens = df['open'].to_numpy()
    tkeys = df['time_key'].to_numpy()
    n = len(df)

    ma_fast = pd.Series(closes).rolling(fast).mean().to_numpy()
    ma_slow = pd.Series(closes).rolling(slow).mean().to_numpy()

    trades = []
    pos = 0
    entry_px = 0.0
    entry_cost = 0.0
    entry_idx = None
    sl_px = 0.0

    for i in range(1, n):
        # 用上一根收盘确定信号(无未来函数), 以当日开盘成交
        if np.isnan(ma_slow[i - 1]):
            continue
        sig = 1 if ma_fast[i - 1] >= ma_slow[i - 1] else -1

        o, h, l = opens[i], highs[i], lows[i]

        # 平仓
        if pos != 0:
            long = pos > 0
            exit_reason, exit_px = None, None
            if sl and ((long and l <= sl_px) or (not long and h >= sl_px)):
                exit_reason, exit_px = 'SL', sl_px
            elif (long and sig == -1) or (not long and sig == 1):
                exit_reason = 'SIGNAL'
                exit_px = o * (1 - (UNIVERSE[code]['slip'] if long else -UNIVERSE[code]['slip']))
                # 严格无未来: 用当日开盘平
            if exit_reason is not None:
                qty = abs(pos)
                close_cost = cost_of_trade(code, abs(exit_px * qty))
                gross = (exit_px - entry_px) * pos
                net = gross - entry_cost - close_cost
                hold_days = max((i - entry_idx), 1)
                trades.append({
                    'exit': exit_reason, 'gross': round(gross, 2),
                    'net': round(net, 2), 'hold_days': hold_days,
                })
                pos = 0

        # 开仓/反手
        if pos == 0:
            want_long = sig == 1
            want_short = sig == -1 and allow_short
            if want_long or want_short:
                code_l = UNIVERSE[code]['name']
                slip = UNIVERSE[code]['slip']
                qty = round_lot(code, o, TARGET_NV)
                entry_px = o * (1 + slip if want_long else 1 - slip)
                entry_cost = cost_of_trade(code, entry_px * qty)
                pos = qty if want_long else -qty
                entry_idx = i
                if sl:
                    sl_px = entry_px * (1 - sl if want_long else 1 + sl)

    # 末段平仓
    if pos != 0:
        long = pos > 0
        exit_px = closes[-1] * (1 - UNIVERSE[code]['slip'] if long else 1 + UNIVERSE[code]['slip'])
        close_cost = cost_of_trade(code, abs(exit_px * pos))
        gross = (exit_px - entry_px) * pos
        trades.append({'exit': 'EOD', 'gross': round(gross, 2),
                       'net': round(gross - entry_cost - close_cost, 2),
                       'hold_days': max((n - 1 - entry_idx), 1)})

    return trades


def summarize(code, tag, trades):
    n = len(trades)
    if n == 0:
        return (f'{UNIVERSE[code]["name"]} {tag}', 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'x')
    nets = np.array([t['net'] for t in trades])
    gross = np.array([t['gross'] for t in trades])
    wins = nets[nets > 0]
    loss = nets[nets <= 0]
    wr = len(wins) / n
    total = nets.sum()
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float('inf')
    avg_hold = np.mean([t['hold_days'] for t in trades])
    by = {}
    for t in trades:
        by[t['exit']] = by.get(t['exit'], 0) + 1
    rstr = ','.join(f'{k}:{v}' for k, v in sorted(by.items()))
    return (f'{UNIVERSE[code]["name"]} {tag}', n, wr * 100, total,
            total / n, pf, avg_hold, gross.mean(), rstr)


def main():
    fetch(list(UNIVERSE.keys()))

    cfgs = []
    # (fast, slow, allow_short, sl, tag)
    cfgs.append((3, 10, False, None, 'MA3/10 只多'))
    cfgs.append((5, 20, False, None, 'MA5/20 只多  ← 周线持仓'))
    cfgs.append((10, 30, False, None, 'MA10/30 只多'))
    cfgs.append((5, 20, True, None, 'MA5/20 双向'))
    cfgs.append((5, 20, False, 0.05, 'MA5/20 只多 +SL5%'))
    cfgs.append((5, 20, True, 0.05, 'MA5/20 双向 +SL5%'))

    print('=' * 130)
    print('周线级持仓趋势回测 · 日线 · 每笔名义10万HKD · 成本=双边佣金+税+滑点')
    print('=' * 130)

    for code in UNIVERSE:
        df = pd.read_csv(os.path.join(BASE, UNIVERSE[code]['csv']), parse_dates=['time_key'])
        px = df['close'].mean()
        one = cost_of_trade(code, TARGET_NV)
        print(f'\n### {UNIVERSE[code]["name"]} {code}  {df["time_key"].iloc[0].date()} ~ '
              f'{df["time_key"].iloc[-1].date()}  {len(df)}根日线  '
              f'往返摩擦≈{one*2/TARGET_NV*100:.3f}%')
        print('%-22s %5s %6s %11s %9s %7s %7s %9s %s' % (
            '策略', '交易', '胜率%', '总盈亏', '均值/笔', '盈亏比', '持仓天', '毛均值', '出场'))
        for fast, slow, allow_short, sl, tag in cfgs:
            trades = run_trend(code, df, fast, slow, allow_short, sl)
            name, n, wr, total, avg, pf, hold, gavg, rstr = summarize(code, tag, trades)
            print('%-22s %5d %6.1f %+11.0f %+9.1f %7.2f %7.1f %+9.1f %s' % (
                name, n, wr, total, avg, pf, hold, gavg, rstr))

    print()
    print('注: 信号用昨日收盘均线判断、今日开盘成交(无未来函数); 持仓可跨日/周/月; '
          'SL=固定止损; SIGNAL=均线反手; EOD=数据末端强制平仓')


if __name__ == '__main__':
    main()