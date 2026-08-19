#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巴菲特-芒格 量化筛选器 (QMJ+价值+低波动 三因子打分)
====================================================
数据来源: moomoo OpenD (实时快照 + 财务MainIndex)
评分框架 (Frazzini, Kabiller & Pedersen 2018 "Buffett's Alpha"):
  - 质量 QMJ: 高ROE + 高净利/毛利 + 稳定增长 + 低负债
  - 价值 Value: 低PE / 低PB (便宜)

限速处理: OpenD 财务接口 30秒/30次; 采用【缓存+自动等待重试】:
  - 财务结果缓存到 financial_cache.json(避免重复请求)
  - 失败(限流)时 sleep 35s 后重试该只, 直至拉完
"""
import os
import json
import time
import numpy as np
import pandas as pd

from moomoo import OpenQuoteContext, RET_OK
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

HOST, PORT = '127.0.0.1', 11111
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, 'financial_cache.json')

CANDIDATES = list(dict.fromkeys([
    'HK.00700', 'HK.09988', 'HK.00005', 'HK.00939', 'HK.01299', 'HK.00388',
    'HK.02318', 'HK.01810', 'HK.02020', 'HK.01093', 'HK.00288', 'HK.02382',
    'HK.01088', 'HK.00883', 'HK.01211', 'HK.00941', 'HK.00386', 'HK.00981',
    'HK.01800', 'HK.06618', 'HK.02331', 'HK.00669', 'HK.02628', 'HK.00027',
    'HK.01113', 'HK.00016', 'HK.00001', 'HK.00002',
]))

# 最新期(年报)相关字段(避免单季误导): MainIndex 字段ID
FIELDS = {
    5002: '净资产/股', 5003: '基本EPS', 5005: '经营现金流/股',
    5010: '毛利率%', 5012: '净利率%', 5014: 'ROE%', 5015: 'ROA%',
    5017: '流动比率', 5030: '营收3Y增速%', 5034: '净利3Y增速%',
    5043: '资产负债率%', 5045: '产权比率', 5055: '股息率%',
}


def load_cache():
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE, 'w') as f:
        json.dump(cache, f)


def fetch_financials(ctx, code, cache, retries=6):
    """拉取单只 MainIndex 并缓存; 遇限流等待后重试。返回缓存条目或 None。"""
    if code in cache:
        return cache[code]
    for attempt in range(retries):
        try:
            ret, d = ctx.get_financials_statements(
                code, statement_type=FinancialStatementsType_MainIndex, num=1)
        except Exception as e:
            print('[财务] %s 异常 %s' % (code, e))
            time.sleep(35)
            continue
        if ret == RET_OK and d is not None:
            rl = d.get('report_list', [])
            entry = {}
            if rl:
                for item in rl[0].get('item_list', []):
                    fid = item.get('field_id')
                    v = item.get('value') if 'value' in item else item.get('data')
                    entry[str(fid)] = v
            cache[code] = entry
            save_cache(cache)
            return entry
        # 限流 → 等待 35s 后重试
        print('[限流] %s 等待35s重试(%d/%d)...' % (code, attempt + 1, retries))
        time.sleep(35)
    print('[放弃] %s 财务数据拉取失败' % code)
    return None


def main():
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    cache = load_cache()

    print('=' * 120)
    print('巴菲特-芒格 量化筛选器 · 候选池 %d 只港股蓝筹(OpenD 实时数据)' % len(CANDIDATES))
    print('=' * 120)

    ret, snap = ctx.get_market_snapshot(CANDIDATES)
    if ret != RET_OK:
        print('快照失败:', snap)
        ctx.close()
        return
    snap = snap.drop_duplicates(subset='code').set_index('code', drop=False)
    print('[快照] %d 只估值/价数据' % len(snap))

    rows = []
    for idx, code in enumerate(CANDIDATES):
        if code not in snap.index:
            continue
        fin = fetch_financials(ctx, code, cache)
        s = snap.loc[code]
        # 年化ROE = EPS / 每股净资产(快照TTM口径, 避免单季误导)
        eps, naps = s['earning_per_share'], s['net_asset_per_share']
        roe_yearly = (eps / naps * 100) if (eps == eps and naps and naps > 0) else np.nan
        def g(fid):
            return fin.get(str(fid)) if fin else None
        rows.append({
            'code': code, 'name': s['name'], 'price': s['last_price'],
            'pe_ttm': s['pe_ttm_ratio'], 'pb': s['pb_ratio'],
            'mkt_val_bn': s['total_market_val'] / 1e9 if s['total_market_val'] == s['total_market_val'] else np.nan,
            'roe_yearly': roe_yearly,
            'net_margin': g(5012), 'gross_margin': g(5010),
            'debt_ratio': g(5043), 'profit_gr3y': g(5034),
            'rev_gr3y': g(5030), 'flow_per_share': g(5005),
            'div': g(5055),
        })

    df = pd.DataFrame(rows)

    # 过滤: 需有有效ROE/估值/毛利 → 排除银行保险等金融与缺数据者
    df = df[df['roe_yearly'].notna() & df['pe_ttm'].notna() & (df['pe_ttm'] > 0)
            & (df['pb'] > 0) & df['gross_margin'].notna()].copy()
    df['roe'] = df['roe_yearly'].fillna(df['net_margin'] * 0)

    def pct(series, higher_better=True):
        r = series.rank(pct=True).fillna(0.5)
        return r if higher_better else 1 - r

    df['quality'] = (
        pct(df['roe']) * 3 +
        pct(df['net_margin']) * 2 +
        pct(df['gross_margin']) * 1 +
        pct(df['flow_per_share']) * 1 +
        (1 - pct(df['debt_ratio'])) * 1.5
    )
    df['value'] = pct(df['pe_ttm'], True) + pct(df['pb'], True)
    df['score'] = df['quality'] * 0.6 + df['value'] * 0.4
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    cols = ['code', 'name', 'price', 'pe_ttm', 'pb', 'roe', 'net_margin',
            'gross_margin', 'debt_ratio', 'profit_gr3y', 'flow_per_share',
            'div', 'quality', 'value', 'score']
    show = df[cols].copy()

    print('\n%-8s %-12s %8s %6s %6s %7s %7s %6s %7s %8s %6s | %6s %5s %5s' % (
        '代码', '名称', '价格', 'PE', 'PB', 'ROE%', '净利%', '毛利%', '负债%', '净利3Y%', '股息%',
        '质量', '价值', '总分'))
    for _, r in show.iterrows():
        print('%-8s %-12s %8.2f %6.1f %6.2f %7.1f %6.1f %6.1f %7.1f %8.1f %6.2f | %6.2f %5.2f %5.3f' % (
            r['code'], r['name'], r['price'], r['pe_ttm'], r['pb'],
            r['roe'] or 0, r['net_margin'] or 0, r['gross_margin'] or 0,
            r['debt_ratio'] or 0, r['profit_gr3y'] or 0, r['div'] or 0,
            r['quality'], r['value'], r['score']))

    print('\n[Top 10 — 巴菲特-芒格打分]')
    for i, r in show.head(10).iterrows():
        print('  %2d. %-8s %-12s  ROE=%5.1f%%  净利率=%5.1f%%  负债率=%5.1f%%  PE=%5.1f  PB=%4.2f  总分=%.3f'
              % (i + 1, r['code'], r['name'], r['roe'] or 0, r['net_margin'] or 0,
                 r['debt_ratio'] or 0, r['pe_ttm'], r['pb'], r['score']))

    out = os.path.join(BASE, 'buffett_screen_result.csv')
    show.to_csv(out, index=False)
    print('\n[已保存] %s' % out)
    print('[缓存] 财务缓存 %s 共 %d 只' % (CACHE, len(cache)))

    ctx.close()


if __name__ == '__main__':
    main()