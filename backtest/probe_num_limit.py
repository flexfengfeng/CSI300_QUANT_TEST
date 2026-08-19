#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: MainIndex num 参数上限 + 分页行为(KO/AAPL)。"""
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

for num in (50, 100):
    ret, d = ctx.get_financials_statements(
        'US.KO', statement_type=FinancialStatementsType_MainIndex, num=num)
    print('num=%d ret=%d 报告数=%s' % (num, ret,
          len(d.get('report_list', [])) if ret == RET_OK else d))

print('\n--- num=50 分页拿全量 ---')
next_key = None
total = 0
for page in range(6):
    ret, d = ctx.get_financials_statements(
        'US.KO', statement_type=FinancialStatementsType_MainIndex,
        num=50, next_key=next_key)
    if ret != RET_OK:
        print('页%d失败: %s' % (page + 1, d))
        break
    rl = d.get('report_list', [])
    total += len(rl)
    nk = d.get('next_key')
    ann = [r for r in rl if r.get('financial_type') == 'ANNUAL']
    print('页%d +%d (累计%d, 年报%d) next_key=%s' % (page + 1, len(rl), total, len(ann), bool(nk)))
    if not nk:
        break
    next_key = nk

print('\n--- 一次 num=50 返回的报告 date_time_str 范围 ---')
ret, d = ctx.get_financials_statements(
    'US.KO', statement_type=FinancialStatementsType_MainIndex, num=50)
if ret == RET_OK:
    rl = d.get('report_list', [])
    dts = [r.get('date_time_str') for r in rl]
    print('报告数:', len(rl), ' 最早:', min(dts), ' 最新:', max(dts))
    print('financial_type 分布:', {t: sum(1 for r in rl if r.get('financial_type') == t) for t in set(r.get('financial_type') for r in rl)})

ctx.close()
print('\n完成')