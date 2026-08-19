#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取盈富基金(HK.02800) 1分钟历史K线 缓存为CSV, 并打印流动性/手数信息。"""
import os
import sys
import pandas as pd

from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)

HOST, PORT = '127.0.0.1', 11111
CODE = 'HK.02800'
KLINE = KLType.K_1M
START_DATE = '2025-06-01'
END_DATE = None
QUEUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hk_02800_1m.csv')
FORCE_REFRESH = '--force' in sys.argv

ctx = OpenQuoteContext(host=HOST, port=PORT)

# 快照: 手数 / 成交额
if not os.path.exists(QUEUE_PATH) or FORCE_REFRESH:
    ret, snap = ctx.get_market_snapshot([CODE])
    if ret == RET_OK:
        print('[快照]', CODE, snap[['lot_size', 'last_price', 'turnover']].iloc[0].to_dict())
    else:
        print('[快照] 失败:', snap)

    frames, page_key = [], None
    while True:
        ret, data, page_key = ctx.request_history_kline(
            code=CODE, start=START_DATE, end=END_DATE, ktype=KLINE,
            autype=AuType.QFQ,
            fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                    KL_FIELD.LOW, KL_FIELD.CLOSE, KL_FIELD.TRADE_VOL,
                    KL_FIELD.TRADE_VAL],
            max_count=1000, page_req_key=page_key)
        if ret != RET_OK:
            print('[下载] 失败:', data, file=sys.stderr)
            break
        frames.append(data)
        print('[下载] 第%d页: %d 根  %s ~ %s' % (
            len(frames), len(data), data['time_key'].iloc[0], data['time_key'].iloc[-1]))
        if page_key is None:
            break

    ctx.close()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.to_csv(QUEUE_PATH, index=False)
    print('[下载] 共 %d 根K线, 缓存到 %s' % (len(df), QUEUE_PATH))

df = pd.read_csv(QUEUE_PATH, parse_dates=['time_key'])
print('[加载] %s: %d 根K线, %s ~ %s, %d 个交易日' % (
    QUEUE_PATH, len(df), df['time_key'].iloc[0], df['time_key'].iloc[-1],
    df['time_key'].dt.date.nunique()))

# 流动性概览
daily = df.groupby(df['time_key'].dt.date).agg(
    day_turnover=('turnover', 'sum'), day_vol=('volume', 'sum'))
print('[流动性] 日均成交额: %.0f 港元, 日均成交量: %.0f 股' % (
    daily['day_turnover'].mean(), daily['day_vol'].mean()))