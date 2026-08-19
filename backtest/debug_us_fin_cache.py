#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试: 检查 us_annual_fin_cache.json 结构 + quality_score 输出。"""
import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FIN_CACHE = os.path.join(BASE, 'us_annual_fin_cache.json')

cache = {}
if os.path.exists(FIN_CACHE):
    with open(FIN_CACHE) as f:
        cache = json.load(f)
print('缓存存在, 标的数:', len(cache))

for code in ['US.KO', 'US.AAPL', 'US.PG', 'US.CVX']:
    entry = cache.get(code)
    if entry is None:
        print('\n%-8s 不在缓存' % code)
        continue
    annual = entry.get('annual', {})
    print('\n%-8s done=%s 年报期数=%d' % (code, entry.get('done'), len(annual)))
    if annual:
        keys = sorted(annual.keys())
        print('  最早 %s  最新 %s' % (keys[0], keys[-1]))
        # 看最近一期和 2006 期
        for ds in ([keys[-1], '2006-12-31'] if '2006-12-31' in annual else [keys[-1]]):
            d = annual[ds]
            print('  期 %s 字段数=%d 示例: %s' % (ds, len(d), dict(list(d.items())[:6])))
            # 关键字段
            for fid in ['14029', '14005', '14002', '14019', '14042']:
                print('      field %s = %s' % (fid, d.get(fid)))


def annual_at(annual, asof, pad_days=90):
    best, best_dt = None, None
    pad = pd.Timedelta(days=pad_days)
    for ds, d in annual.items():
        dt_ = pd.Timestamp(ds)
        if dt_ + pad <= asof and (best_dt is None or dt_ > best_dt):
            best, best_dt = d, dt_
    return best, best_dt


print('\n===== 2007-01-02 复盘时点可用年报 =====')
for code in ['US.KO', 'US.AAPL', 'US.PG', 'US.CVX']:
    annual = cache.get(code, {}).get('annual', {})
    best, best_dt = annual_at(annual, pd.Timestamp('2007-01-02'))
    if best is None:
        print('%-8s 无可用年报' % code)
    else:
        print('%-8s 可用年报=%s ROE=%s 净利率=%s' % (
            code, best_dt,
            best.get('14029', best.get(14029)),
            best.get('14005', best.get(14005))))
print('\n完成')