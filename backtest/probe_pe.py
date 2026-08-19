"""探测沪深300 PE_TTM 数据源 (中证指数 / 乐咕乐股)。"""
import akshare as ak

# 1. 中证指数官网
try:
    df = ak.stock_zh_index_value_csindex(symbol="000300")
    print("csindex OK, 行数:", len(df))
    print(df.head(2).to_string())
except Exception as e:
    print("csindex 失败:", repr(e)[:150])

# 2. 乐咕乐股 PE_TTM
try:
    df2 = ak.stock_index_pe_lg(symbol="沪深300")
    print("\nlg OK, 行数:", len(df2))
    print("列:", list(df2.columns))
    print(df2.head(2).to_string())
    print(df2.tail(2).to_string())
except Exception as e2:
    print("\nlg 失败:", repr(e2)[:150])