#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: moomoo OpenD 对美股的历史日线 / 历史年报(财务)可得范围。
确认 20 年(约2006)回测数据是否可用。"""
import os
import numpy as np
import pandas as pd
from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

BASE = os.path.dirname(os.path.abspath(__file__))
CODES = ['US.AAPL', 'US.KO', 'US.BRK.B', 'US.V', 'US.MCO', 'US.AXP']

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print('===== 1. 美股实时快照 =====')
try:
    ret, snap = ctx.get_market_snapshot(CODES)
    print('ret =', ret)
    if ret == RET_OK:
        keep = [c for c in snap.columns if any(k in c.lower() for k in
                ('code', 'name', 'price', 'pe_ttm', 'pb', 'market_val', 'dividend'))]
        print(snap[keep].to_string())
except Exception as e:
    print('快照异常:', repr(e))

print('\n===== 2. 美股日线最早可用日期 (2011-01-01 起) =====')
try:
    ret, data, _ = ctx.request_history_kline(
        code='US.AAPL', start='2011-01-01', end='2026-08-01',
        ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE], max_count=None)
    print('ret =', ret, ' 行数 =', 0 if ret != RET_OK else len(data))
    if ret == RET_OK:
        dt = pd.to_datetime(data['time_key'])
        print('最早:', dt.min().date(), ' 最新:', dt.max().date(), ' 总日数:', len(dt))
except Exception as e:
    print('日线异常:', repr(e))

print('\n===== 3. 美股日线最早可用日期 (2000-01-01 起, 测20年) =====')
try:
    ret, data, _ = ctx.request_history_kline(
        code='US.AAPL', start='2000-01-01', end='2026-08-01',
        ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE], max_count=None)
    print('ret =', ret, ' 行数 =', 0 if ret != RET_OK else len(data))
    if ret == RET_OK:
        dt = pd.to_datetime(data['time_key'])
        print('最早:', dt.min().date(), ' 最新:', dt.max().date(), ' 总日数:', len(dt))
except Exception as e:
    print('日线(2000)异常:', repr(e))

print('\n===== 4. 美股财务 MainIndex 历史年报 =====')
try:
    ret, d = ctx.get_financials_statements(
        'US.AAPL', statement_type=FinancialStatementsType_MainIndex,
        num=50, next_key=None)
    print('ret =', ret)
    if ret == RET_OK and d is not None:
        reports = d.get('report_list', [])
        print('报告期数:', len(reports))
        annual = [r for r in reports if r.get('financial_type') == 'ANNUAL']
        print('年报期数:', len(annual))
        for r in annual[:5]:
            print('   ', r.get('date_time_str'), r.get('financial_type'))
        if annual:
            last = annual[0]
            print('最新年报日期:', last.get('date_time_str'))
            for item in last.get('item_list', [])[:15]:
                print('     id=%s %s = %s' % (item['field_id'], item['display_name'], item['data']))
except Exception as e:
    print('财务异常:', repr(e))

print('\n===== 5. 伯克希尔BRK.B / 喜诗糖果未上市检测 =====')
for code in ['US.BRK.B', 'US.KO', 'US.V', 'US.MCO']:
    try:
        ret, snap = ctx.get_market_snapshot([code])
        if ret == RET_OK and len(snap):
            r = snap.iloc[0]
            print('%-10s %-12s price=%s pe=%s pb=%s'
                  % (code, r['name'], r.get('last_price'), r.get('pe_ttm_ratio'), r.get('pb_ratio')))
        else:
            print('%-10s 无快照: %s' % (code, snap))
    except Exception as e:
        print('%-10s 异常: %s' % (code, repr(e)))

ctx.close()
print('\n探测完成')