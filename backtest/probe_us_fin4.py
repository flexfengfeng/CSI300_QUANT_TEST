#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: 美股 ANNUAL 年报报告的完整字段, 确认 EPS/BPS/营收/净利等字段。"""
import json
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 分页拿历史年报
all_reports = []
next_key = None
for _ in range(6):
    ret, d = ctx.get_financials_statements(
        'US.AAPL', statement_type=FinancialStatementsType_MainIndex,
        num=50, next_key=next_key)
    if ret != RET_OK:
        print('失败:', d)
        break
    all_reports.extend(d.get('report_list', []))
    nk = d.get('next_key')
    if not nk:
        break
    next_key = nk

annual = [r for r in all_reports if r.get('financial_type') == 'ANNUAL']
print('年报期数:', len(annual))

# 打印最近一份年报和最早一份年报的完整字段
for label, rep in [('最新年报', annual[0]), ('最早年报', annual[-1])]:
    print('\n===== %s %s =====' % (label, rep.get('date_time_str')))
    for it in rep.get('item_list', []):
        print('id=%-6s %-35s data=%s' % (
            it['field_id'], it['display_name'], it.get('data')))

ctx.close()
print('\n完成')