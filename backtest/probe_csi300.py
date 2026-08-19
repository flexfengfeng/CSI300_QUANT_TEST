"""探测 moomoo 接口获取沪深300 (000300) 指数日线的可行性。"""
import pandas as pd
from moomoo import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 候选代码: moomoo A股指数常见格式
candidates = ['SH.000300', 'SH.000001']

for code in candidates:
    print(f'===== {code} =====')
    ret, data, _ = ctx.request_history_kline(
        code=code, start='2020-01-01', end='2026-01-01',
        ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                KL_FIELD.LOW, KL_FIELD.CLOSE],
        max_count=10, page_req_key=None)
    if ret != RET_OK:
        print(f'  失败: {data}')
        continue
    print(f'  成功, 返回 {len(data)} 条')
    df = data[['time_key', 'open', 'close']]
    print(df.head(3).to_string(index=False))

ctx.close()
print('\n探测完成')