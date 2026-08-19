"""测试从 moomoo OpenD 拉取 HK.00700 1 分钟历史 K 线，验证返回格式与分页。"""
from moomoo import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK

HOST = '127.0.0.1'
PORT = 11111
CODE = 'HK.00700'
KLINE = KLType.K_1M

ctx = OpenQuoteContext(host=HOST, port=PORT)

# 先拉最近数据，看返回结构和最大请求条数
ret, data, page_req_key = ctx.request_history_kline(
    code=CODE,
    ktype=KLINE,
    autype=AuType.QFQ,
    fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH, KL_FIELD.LOW, KL_FIELD.CLOSE,
            KL_FIELD.TRADE_VOL, KL_FIELD.TRADE_VAL],
    max_count=1000,
)
print('ret =', ret)
if ret != RET_OK:
    print('data =', data)
else:
    print('rows =', len(data))
    print('columns =', list(data.columns))
    print('--- first 3 rows ---')
    print(data.head(3).to_string())
    print('--- last 3 rows ---')
    print(data.tail(3).to_string())
    print('page_req_key =', page_req_key)
    print('dtypes:')
    print(data.dtypes)

ctx.close()