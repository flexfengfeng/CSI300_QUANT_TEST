"""探测2: 批量快照估值 + 资产负债表字段(算 ROE 用) + MainIndex(加权ROE)。"""
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import (
    FinancialStatementsType_BalanceSheet,
    FinancialStatementsType_Income,
    FinancialStatementsType_MainIndex,
)

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print('===== 资产负债表(BalanceSheet) =====')
try:
    ret2, d2 = ctx.get_financials_statements(
        'HK.00700', statement_type=FinancialStatementsType_BalanceSheet, num=4)
    print('ret2 =', ret2)
    if ret2 == RET_OK and d2 is not None:
        sl = d2.get('structure_list', [])
        print('字段数 =', len(sl))
        for f in sl:
            print('   ', f['field_id'], f['display_name'])
        rl = d2.get('report_list', [])
        print('期数 =', len(rl))
        if rl:
            r0 = rl[0]
            print('最新期:', r0.get('date_time_str'), r0.get('period_text'), r0.get('currency_code'))
            for item in r0.get('item_list', []):
                print('     id=%s %s = %s' % (item['field_id'], item['display_name'], item['data']))
except Exception as e:
    print('异常:', repr(e))

print('\n===== MainIndex(主要指标,含ROE) =====')
try:
    ret3, d3 = ctx.get_financials_statements(
        'HK.00700', statement_type=FinancialStatementsType_MainIndex, num=6)
    print('ret3 =', ret3)
    if ret3 == RET_OK and d3 is not None:
        sl = d3.get('structure_list', [])
        print('字段数 =', len(sl))
        keep = []
        for f in sl:
            fn = f['display_name']
            if any(k in fn.lower() for k in ('roe', 'net margin', 'gross margin',
                    'debt', 'asset', 'equity', 'cash', 'div', 'eps', 'growth')):
                keep.append((f['field_id'], fn))
        print('与基本面相关字段:')
        for f in sl:
            print('   ', f['field_id'], f['display_name'])
        rl = d3.get('report_list', [])
        print('期数 =', len(rl))
        if rl:
            r0 = rl[0]
            print('最新期:', r0.get('date_time_str'), r0.get('period_text'))
            for item in r0.get('item_list', []):
                print('     id=%s %s = %s' % (item['field_id'], item['display_name'], item['data']))
except Exception as e:
    print('异常:', repr(e))

ctx.close()
print('完成')