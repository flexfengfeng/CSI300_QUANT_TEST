"""诊断 rolling_quality 开仓为0的问题。"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rolling_quality as rq

# 用缓存的2020年年报测 score_at 是否正常
cache = rq.load_cache()
print('缓存标的:', len(cache))
asof = pd.Timestamp('2015-01-02')
for code in ['HK.00700', 'HK.02020', 'HK.02382', 'HK.00941']:
    ann = cache.get(code, {}).get('annual', {}) if cache.get(code, {}).get('done') else None
    if ann:
        sc = rq.score_at(ann, asof)
        print('score_at(%s, %s) = %s' % (code, asof.date(), sc))
    else:
        print('score_at(%s) 无年报缓存' % code)

# 直接检查: 重建后某标的某天的 sig
df = pd.read_csv('/Users/fengfeng/Dev/moomoo/backtest/daily_HK_00700.csv', parse_dates=['time_key'])
closes = df['close']
mf = pd.Series(closes).rolling(10).mean()
ms = pd.Series(closes).rolling(30).mean()
# 2014-03-15 附近的一天
target = pd.Timestamp('2014-03-17')
idx = df.index[df['time_key'] == target][0]
print('\nHK.00700 2014-03-17 idx=%d' % idx)
print('  mf[idx-1]=%s ms[idx-1]=%s  sig=%d' % (
    mf.iloc[idx-1], ms.iloc[idx-1], 1 if mf.iloc[idx-1] >= ms.iloc[idx-1] else -1))

# 2014 池中新鸿基
df2 = pd.read_csv('/Users/fengfeng/Dev/moomoo/backtest/daily_HK_00016.csv', parse_dates=['time_key'])
print('\n新鸿基 2014-03-17 存在?', (df2['time_key'] == target).any())
if (df2['time_key'] == target).any():
    idx2 = df2.index[df2['time_key'] == target][0]
    c2 = df2['close']
    mf2 = pd.Series(c2).rolling(10).mean()
    ms2 = pd.Series(c2).rolling(30).mean()
    print('  open=%.2f lot=1000 预算16667 → raw=%.0f qty=%d' % (
        df2['open'].iloc[idx2], 16667/(df2['open'].iloc[idx2]*1.0005),
        int(16667/(df2['open'].iloc[idx2]*1.0005)/1000)*1000))
    print('  mf2[idx-1]=%s ms2[idx-1]=%s sig=%d' % (
        mf2.iloc[idx2-1], ms2.iloc[idx2-1],
        1 if mf2.iloc[idx2-1] >= ms2.iloc[idx2-1] else -1))