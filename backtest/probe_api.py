"""探测 moomoo OpenD 的财务/估值/列表接口，确认筛选器数据来源与格式。"""
from moomoo import (OpenQuoteContext, Market, SecurityType, RET_OK)

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print('===== 1. 港股股票列表 =====')
ret, data = ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)
print('ret =', ret, ' 数量 =', 0 if ret != RET_OK else len(data))
if ret == RET_OK:
    print(data[['code', 'name', 'lot_size']].head(5).to_string())

print('\n===== 2. 财务摘要(测试) =====')
# 尝试不同参数组合
try:
    ret2, d2 = ctx.get_financials_statements('HK.00700')
    print('ret2 =', ret2, type(d2))
    if ret2 == RET_OK and d2 is not None:
        if hasattr(d2, 'shape'):
            print('shape =', d2.shape)
            print('columns =', list(d2.columns)[:30])
        else:
            print(d2)
except Exception as e:
    print('财务接口异常:', repr(e))

print('\n===== 3. 实时估值/行情(测试) =====')
try:
    ret3, q3 = ctx.get_stock_quote(['HK.00700', 'HK.02800'])
    print('ret3 =', ret3)
    if ret3 == RET_OK:
        cols = [c for c in q3.columns if any(k in c.lower() for k in
                ('price', 'pe', 'pb', 'roe', 'turnover', 'market', 'name'))]
        print('相关列:', cols)
        print(q3[cols].head(10).to_string())
except Exception as e:
    print('行情异常:', repr(e))

print('\n===== 4. 港股自选股池(手工候选池) =====')
# 先用手工构建的港股优质蓝筹池探测财务/行情接口是否可批量
candidates = [
    'HK.00700', 'HK.02800', 'HK.00005', 'HK.00939', 'HK.01299',
    'HK.00388', 'HK.02318', 'HK.01093', 'HK.02020', 'HK.01810',
    'HK.09988', 'HK.00941', 'HK.00288', 'HK.00388', 'HK.01113',
]
ret4, q4 = ctx.get_stock_quote(candidates)
if ret4 == RET_OK:
    print('批量行情 ok, 行数 =', len(q4))

ctx.close()
print('\n探测完成')