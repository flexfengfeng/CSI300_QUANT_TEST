#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季度滚动重建池回测 —— 消除前视偏差
========================================================
相比 quality_pool_backtest.py (用当前财报选10只, 前视上偏) 的修正:
  1. 拉取每只标的【全部历史年报】(MainIndex, ~25年)
  2. 每个季度第1个交易日重建池: 只用【当时已披露】的最近年报打分
     披露日近似 = 财务期末 + 4个月 (港股年报3-4月披露上年度)
  3. 每季从候选宇宙(28只)打分取 Top 6, 池内等权预算(10万/6), MA10/30 择时
  4. 出池标的平仓; 全程无未来信息 → 结果可实盘参考

净值模型: 净值 = 初始资金 + 已实现盈亏(现金增减) + 持仓市值
          逐标的独立 MA 择时; 池外标的持币(预算闲置)。

候选宇宙 = buffett_screener 的池子(固定, 但宇宙本身也是"今天可见"的,
          严格无偏需在2014年就定义宇宙, 此处如实标注为近似)。
"""
import os
import json
import time
import numpy as np
import pandas as pd

from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)
from moomoo.common.pb.Qot_Common_pb2 import FinancialStatementsType_MainIndex

HOST, PORT = '127.0.0.1', 11111
BASE = os.path.dirname(os.path.abspath(__file__))
START, END = '2014-01-01', '2026-06-02'
INIT = 100_000.0
TOP_N = 6                 # 每期池大小
DISCLOSE_PAD_DAYS = 120   # 财务期末 +120天 ≈ 年报披露日
FAST_MA, SLOW_MA = 10, 30
FIN_CACHE = os.path.join(BASE, 'rolling_fin_cache.json')

UNIVERSE = [
    ('HK.00700', '腾讯', 0.001, 100, 0.0005),
    ('HK.09988', '阿里', 0.001, 100, 0.0005),
    ('HK.00005', '汇丰', 0.001, 400, 0.0005),
    ('HK.00939', '建行', 0.001, 1000, 0.0005),
    ('HK.01299', '友邦', 0.001, 200, 0.0005),
    ('HK.00388', '港交所', 0.001, 100, 0.0005),
    ('HK.02318', '平安', 0.001, 500, 0.0005),
    ('HK.01810', '小米', 0.001, 200, 0.0005),
    ('HK.02020', '安踏', 0.001, 200, 0.0005),
    ('HK.01093', '石药', 0.001, 2000, 0.0005),
    ('HK.00288', '万洲', 0.001, 1000, 0.0005),
    ('HK.02382', '舜宇', 0.001, 100, 0.0005),
    ('HK.01088', '神华', 0.001, 500, 0.0005),
    ('HK.00883', '中海油', 0.001, 1000, 0.0005),
    ('HK.01211', '比亚迪', 0.001, 500, 0.0005),
    ('HK.00941', '中移动', 0.001, 500, 0.0005),
    ('HK.00386', '中石化', 0.001, 2000, 0.0005),
    ('HK.00981', '中芯', 0.001, 500, 0.0005),
    ('HK.01800', '中交建', 0.001, 1000, 0.0005),
    ('HK.06618', '京东健康', 0.001, 100, 0.0005),
    ('HK.02331', '李宁', 0.001, 500, 0.0005),
    ('HK.00669', '创科', 0.001, 500, 0.0005),
    ('HK.02628', '国寿', 0.001, 1000, 0.0005),
    ('HK.00027', '银河', 0.001, 1000, 0.0005),
    ('HK.01113', '长实', 0.001, 500, 0.0005),
    ('HK.00016', '新鸿基', 0.001, 1000, 0.0005),
    ('HK.00001', '长和', 0.001, 500, 0.0005),
    ('HK.00002', '中电', 0.001, 500, 0.0005),
]
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5


def cost_of(notional, stamp):
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return commission + notional * stamp + notional * TRADE_FEE_R + notional * LEVY_R + SYS_FEE


def load_cache():
    if os.path.exists(FIN_CACHE):
        try:
            with open(FIN_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(c):
    with open(FIN_CACHE, 'w') as f:
        json.dump(c, f)


def fetch_annual_financials(code, cache, retries=12):
    """拉取全部历史 MainIndex, 提取 ANNUAL 报告 (date → 指标dict), 缓存。
    遇限流(30次/30秒)等待 35s 重试, 直到成功或放弃。"""
    if code in cache and cache[code].get('done'):
        return cache[code]['annual']
    for attempt in range(retries):
        ctx = OpenQuoteContext(host=HOST, port=PORT)
        next_key, all_reports = None, []
        ok = True
        for _ in range(4):
            ret, d = ctx.get_financials_statements(
                code, statement_type=FinancialStatementsType_MainIndex,
                num=50, next_key=next_key)
            if ret != RET_OK:
                msg = str(d)
                if 'high frequency' in msg:
                    ok = False
                else:
                    print('  [财务] %s 失败: %s' % (code, msg))
                    ok = False
                break
            reports = d.get('report_list', [])
            all_reports.extend(reports)
            next_key = d.get('next_key')
            if not next_key:
                break
        ctx.close()
        if ok and all_reports:
            break
        if not ok:
            print('  [限流] %s 第%d次等待35s...' % (code, attempt + 1))
        else:
            # 接口返回OK但无报告(部分标的无MainIndex数据) → 标为不可用, 不再重试
            cache[code] = {'done': True, 'annual': {}}
            save_cache(cache)
            return {}
        time.sleep(35)
    else:
        return {}

    annual = {}
    for rep in all_reports:
        if rep.get('financial_type') != 'ANNUAL':
            continue
        ds = rep.get('date_time_str', '')
        if not ds:
            continue
        d = {}
        for item in rep.get('item_list', []):
            v = item.get('value') if 'value' in item else item.get('data')
            d[item['field_id']] = v
        annual[ds] = d
    cache[code] = {'done': True, 'annual': annual}
    save_cache(cache)
    newest = sorted(annual)[-1] if annual else '-'
    print('  [财务] %s 年报 %d 期 (最近 %s)' % (code, len(annual), newest))
    return annual


def annual_at(annual, asof):
    """asof 时点最近可用年报 (期末+120天已披露)。返回 dict 或 None。"""
    best, best_dt = None, None
    pad = pd.Timedelta(days=DISCLOSE_PAD_DAYS)
    for ds, d in annual.items():
        dt_ = pd.Timestamp(ds)
        if dt_ + pad <= asof and (best_dt is None or dt_ > best_dt):
            best, best_dt = d, dt_
    return best


def download_daily(code):
    path = os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['time_key'])
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    ret, data, page_key = ctx.request_history_kline(
        code=code, start=START, end=END, ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                KL_FIELD.LOW, KL_FIELD.CLOSE],
        max_count=None, page_req_key=None)
    ctx.close()
    if ret != RET_OK:
        return None
    df = data.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.to_csv(path, index=False)
    return df


def score_at(annual, asof):
    """asof 时点单只打分: 用最近年报; 缺关键字段返回 None。"""
    d = annual_at(annual, asof)
    if d is None:
        return None

    def g(fid):
        # JSON缓存读回后 field_id 键变为字符串, 兼容两种
        v = d.get(fid)
        if v is None:
            v = d.get(str(fid))
        return v if isinstance(v, (int, float)) and v == v else None

    roe, nm, gm, debt, cfo = g(5014), g(5012), g(5010), g(5043), g(5005)
    if roe is None or nm is None or gm is None:
        return None
    return {'roe': roe, 'nm': nm, 'gm': gm, 'debt': debt, 'cfo': cfo}


def main():
    print('=' * 120)
    print('季度滚动重建池回测 · 消除前视偏差 · 候选%d只 → 每季Top%d · 池内MA10/30择时' % (len(UNIVERSE), TOP_N))
    print('=' * 120)

    daily = {}
    annuals = {}
    cache = load_cache()
    for code, name, stamp, lot, slip in UNIVERSE:
        df = download_daily(code)
        if df is None or len(df) < 60:
            print('[跳过] %s %s 数据不足' % (code, name))
            continue
        # qfq 前复权后历史价格 <=0 → 标的上市日晚于回测起点或前复权数据缺陷 → 剔除
        # (2014年回测时根本不可能持有这些尚未上市的标的, 这也是消除前视偏差的一部分)
        bad = df[(df['open'] <= 0) | (df['close'] <= 0)]
        if len(bad) > 0:
            print('[剔除] %s %s 存在非正价格(首个 %s) → 视为当时未上市/数据缺陷' % (
                code, name, bad['time_key'].iloc[0].date()))
            continue
        ann = fetch_annual_financials(code, cache)
        daily[code] = dict(df=df, name=name, stamp=stamp, lot=lot, slip=slip)
        annuals[code] = ann

    codes = list(daily.keys())
    print('[标的] 剔除未上市/无有效价后: %d 只 = %s' % (
        len(codes), '、'.join(meta_name=0)) if False else '[标的] 剔除后 %d 只: %s' % (
        len(codes), '、'.join(daily[c]['name'] for c in codes)))
    meta = {c: daily[c] for c in codes}
    budget = INIT / TOP_N

    print('[数据] 可用标的 %d 只' % len(codes))

    # 交易日序列(以第一只标的为准)
    first_df = daily[codes[0]]['df']
    all_dates = first_df['time_key'].reset_index(drop=True)

    df_by_code = {c: daily[c]['df'] for c in codes}
    ma = {}
    for c in codes:
        closes = daily[c]['df']['close']
        ma[c] = (pd.Series(closes).rolling(FAST_MA).mean(),
                 pd.Series(closes).rolling(SLOW_MA).mean())

    idx = {c: 0 for c in codes}
    positions = {}        # code -> dict(qty, entry_px, entry_cost)
    cash = INIT           # 标准现金账户: 开仓扣本金+费, 平仓加回卖出净额
    pool = []
    cur_quarter = None
    nav_dates, nav_values = [], []
    n_open = 0

    def sell_position(c, p, price):
        """平仓: 卖出净额(卖价*股数-费用) 回到现金。"""
        nonlocal cash
        sell_px = price * (1 - meta[c]['slip'])
        close_cost = cost_of(sell_px * p['qty'], meta[c]['stamp'])
        cash += sell_px * p['qty'] - close_cost

    for di in range(len(all_dates)):
        today = all_dates.iloc[di]
        q = today.to_period('Q')

        # 季度首日 → 重建池(用当时已披露年报)
        if q != cur_quarter:
            cur_quarter = q
            scored = []
            for c in codes:
                sc = score_at(annuals[c], today)
                if sc:
                    scored.append((c, sc))
            scored.sort(key=lambda x: (
                x[1]['roe'] * 3 + x[1]['nm'] * 2 - (x[1]['debt'] or 0) * 1.5), reverse=True)
            new_pool = [c for c, _ in scored[:TOP_N]]
            # 平掉出池持仓(按当日或最近开盘价卖)
            for c in list(positions.keys()):
                if c not in new_pool:
                    p = positions.pop(c)
                    if idx[c] < len(df_by_code[c]):
                        px = df_by_code[c].iloc[idx[c]]['open']
                        if px is None or (isinstance(px, float) and (px != px or px <= 0)):
                            # 停牌日 → 用最近有效收盘价
                            prev_valid = df_by_code[c].iloc[:idx[c] + 1]
                            px = prev_valid[prev_valid['open'] > 0]['close'].iloc[-1]
                    else:
                        px = df_by_code[c].iloc[-1]['close']
                    sell_position(c, p, px)
            pool = new_pool
            print('  [重建] %s 池=%s 现金=%.0f' % (today.date(),
                  '、'.join(meta[c]['name'] for c in pool), cash))

        # 逐标的推进
        for c in codes:
            dfc = df_by_code[c]
            while idx[c] < len(dfc) and dfc['time_key'].iloc[idx[c]] < today:
                idx[c] += 1
            if idx[c] >= len(dfc) or dfc['time_key'].iloc[idx[c]] != today:
                continue
            row = dfc.iloc[idx[c]]
            o = row['open']
            # 停牌/无成交日 open=0 → 跳过本日全部操作, 防止0元清仓
            if o is None or (isinstance(o, float) and (o != o or o <= 0)):
                continue
            mf, ms = ma[c]
            if idx[c] < SLOW_MA or np.isnan(ms.iloc[idx[c] - 1]):
                continue
            sig = 1 if mf.iloc[idx[c] - 1] >= ms.iloc[idx[c] - 1] else -1
            in_pool = c in pool

            # 平仓: 信号转空
            if c in positions and sig == -1:
                p = positions.pop(c)
                sell_position(c, p, o)

            # 开仓: 池内 & 多头信号 & 现金足够1手 & 池内未满
            if in_pool and sig == 1 and c not in positions and len(positions) < TOP_N:
                cap = INIT / TOP_N          # 每只固定预算上限
                eff = min(cap, cash)
                raw = eff / (o * (1 + meta[c]['slip']))
                buy_qty = int(raw / meta[c]['lot']) * meta[c]['lot']
                if buy_qty > 0:
                    px = o * (1 + meta[c]['slip'])
                    entry_cost = cost_of(px * buy_qty, meta[c]['stamp'])
                    if px * buy_qty + entry_cost <= cash:
                        cash -= px * buy_qty + entry_cost
                        positions[c] = {'qty': buy_qty, 'entry_px': px,
                                        'entry_cost': entry_cost}
                        n_open += 1

        # 净值 = 现金 + 持仓市值
        nav = cash
        for c, p in positions.items():
            row = df_by_code[c].iloc[min(idx[c], len(df_by_code[c]) - 1)]
            nav += p['qty'] * row['close']
        nav_values.append(nav)
        nav_dates.append(today)

    eq = pd.DataFrame({'time': nav_dates, 'equity': nav_values}).set_index('time')
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = eq['equity'].iloc[-1]
    cagr = (final / INIT) ** (1 / years) - 1
    mdd = (eq['equity'] / eq['equity'].cummax() - 1).min()
    daily_r = eq['equity'].pct_change().dropna()
    sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else float('nan')

    print('\n' + '=' * 120)
    print('滚动池结果 (无前视偏差版本)')
    print('=' * 120)
    print('  区间     %s ~ %s  (%.1f年)' % (eq.index[0].date(), eq.index[-1].date(), years))
    print('  期末     %.0f HKD   (初始 %.0f)' % (final, INIT))
    print('  总收益   %+.1f%%' % ((final / INIT - 1) * 100))
    print('  ★ CAGR  %+.2f%%' % (cagr * 100))
    print('  最大回撤 %.1f%%' % (mdd * 100))
    print('  夏普     %.2f' % sharpe)
    print('  开仓次数 %d' % n_open)

    eq.reset_index().to_csv(os.path.join(BASE, 'rolling_pool_equity.csv'), index=False)
    print('\n[已保存] rolling_pool_equity.csv')


if __name__ == '__main__':
    main()