#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测: 打印美股 MainIndex 全部字段ID↔名称映射, 供选股指标使用。"""
import json
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret, d = ctx.get_financials_statements(
    'US.AAPL', statement_type=FinancialStatementsType_MainIndex, num=3)
if ret == RET_OK:
    r0 = d.get('report_list', [])[0]
    for it in r0.get('item_list', []):
        print('id=%-6s %-35s data=%s' % (
            it['field_id'], it['display_name'], it.get('data')))
else:
    print('失败:', d)

ctx.close()
print('\n完成')