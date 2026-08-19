#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 SPY/QQQ 2006-08 至今 QFQ 日线(含 OHLC) 覆盖旧缓存, 作为 benchmark。"""
import os
import pandas as pd
from moomoo import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK

BASE = os.path.dirname(os.path.abspath(__file__))
START, END = '2006-01-01', '2026-08-01'

for code, name in [('US.SPY', 'SPDR S&P 500 ETF'), ('US.QQQ', 'Invesco QQQ Trust')]:
    path = os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv')
    ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, _ = ctx.request_history_kline(
        code=code, start=START, end=END, ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                KL_FIELD.LOW, KL_FIELD.CLOSE],
        max_count=None, page_req_key=None)
    ctx.close()
    if ret != RET_OK:
        print('%s 失败: %s' % (code, data))
        continue
    df = data.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    if 'code' not in df.columns:
        df.insert(0, 'code', code)
    if 'name' not in df.columns:
        df.insert(1, 'name', name)
    df.to_csv(path, index=False)
    print('%s %d 根 %s ~ %s → %s' % (
        code, len(df), df['time_key'].iloc[0].date(), df['time_key'].iloc[-1].date(), path))

print('完成')