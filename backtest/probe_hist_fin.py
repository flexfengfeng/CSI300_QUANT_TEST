"""探测财务历史深度: next_key 分页能拉多少期历史(判断能否做季度滚动重建池)。"""
from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

code = 'HK.00700'
next_key = None
total = 0
for page in range(1, 6):  # 最多 5 页
    ret, d = ctx.get_financials_statements(
        code, statement_type=FinancialStatementsType_MainIndex,
        num=50, next_key=next_key)
    if ret != RET_OK:
        print('第%d页失败: %s' % (page, d))
        break
    rl = d.get('report_list', [])
    total += len(rl)
    next_key = d.get('next_key')
    if rl:
        oldest = rl[-1].get('date_time_str')
        newest = rl[0].get('date_time_str')
        print('第%d页: %d 期  最新=%s 最旧=%s' % (page, len(rl), newest, oldest))
    if not next_key:
        break

print('共拉取 %d 个报告期' % total)
ctx.close()