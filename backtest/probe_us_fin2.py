#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: 美股 MainIndex item 的完整结构(数值藏在哪)。"""
import json
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret, d = ctx.get_financials_statements(
    'US.AAPL', statement_type=FinancialStatementsType_MainIndex, num=3)
print('ret =', ret)
if ret == RET_OK:
    rl = d.get('report_list', [])
    print('报告期数:', len(rl))
    r0 = rl[0]
    print('report keys:', list(r0.keys()))
    print('report 部分内容:')
    for k, v in r0.items():
        if k == 'item_list':
            continue
        print('   %s = %s' % (k, v))
    items = r0.get('item_list', [])
    print('\nitem_list 长度:', len(items))
    print('\n前 3 个 item 完整 JSON:')
    for it in items[:3]:
        print(json.dumps(it, ensure_ascii=False, default=str))
    print('\nitem_list 最后 5 个:')
    for it in items[-5:]:
        print(json.dumps(it, ensure_ascii=False, default=str))

ctx.close()
print('\n完成')