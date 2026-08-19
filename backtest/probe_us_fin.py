#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: 美股 MainIndex 历史年报分页获取能力 + item 字段格式。"""
import os
import pandas as pd
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print('===== 1. MainIndex 分页拉取全部历史 (next_key) =====')
all_reports = []
next_key = None
for page in range(6):
    ret, d = ctx.get_financials_statements(
        'US.AAPL', statement_type=FinancialStatementsType_MainIndex,
        num=50, next_key=next_key)
    if ret != RET_OK:
        print('页%d失败: %s' % (page + 1, d))
        break
    rl = d.get('report_list', [])
    all_reports.extend(rl)
    nk = d.get('next_key')
    print('页%d: +%d 条 (累计 %d)  next_key=%s' % (page + 1, len(rl), len(all_reports), bool(nk)))
    if not nk:
        break
    next_key = nk

annual = [r for r in all_reports if r.get('financial_type') == 'ANNUAL']
print('\n总报告 %d, 年报 %d 期' % (len(all_reports), len(annual)))
for r in annual:
    print('   ', r.get('date_time_str'), r.get('financial_type'))

print('\n===== 2. 最新年报 item 字段格式 =====')
if annual:
    last = annual[0]
    item = last.get('item_list', [])[0]
    print('item keys:', list(item.keys()))
    print('item sample:', {k: v for k, v in item.items() if k != 'children'})

print('\n===== 3. 另测几只美股的年报可得性 =====')
for code in ['US.KO', 'US.BRK.B', 'US.V', 'US.MCO', 'US.AXP']:
    try:
        ret, d = ctx.get_financials_statements(
            code, statement_type=FinancialStatementsType_MainIndex, num=50)
        if ret != RET_OK:
            print('%-10s 失败: %s' % (code, d))
            continue
        rl = d.get('report_list', [])
        ann = [r for r in rl if r.get('financial_type') == 'ANNUAL']
        years = [r.get('date_time_str') for r in ann]
        print('%-10s 报告%d 年报%d 最早=%s 最新=%s'
              % (code, len(rl), len(ann), years[-1] if years else '-', years[0] if years else '-'))
    except Exception as e:
        print('%-10s 异常: %s' % (code, repr(e)))

ctx.close()
print('\n完成')