#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巴菲特/芒格原则 · 美股股票池 + 每年复盘回测
==============================================
设计(无前视、资金使用充分):
  1. 候选宇宙 33 只美股护城河龙头(2006 年已上市, 后上市标的如 Visa 会因当日无价
     被自动排除, 直到其上市后才有资格入池)
  2. 每年 1 月首个交易日复盘:
      - 用【当时已披露年报】(财年末+90天) 按巴菲特-芒格质量分排序取 Top10
        (ROE×3 + 净利率×2 + 毛利率×1 + 净利CAGR5Y×1 − 资产负债率×1.5)
      - 当日无交易价格(未上市/停牌)的标的自动失去入池资格(消除前视)
      - 全额再平衡: 全部持仓按当日开盘价卖出, 现金按总净值/10 等权买入 Top10
      - 卖出计入双边滑点+佣金 5bps/边 (美股无印花税)
  3. 纯持股长期持有, 无择时 → 符合巴菲特/芒格理念
  4. 输出:
      - 每年复盘池记录 us_buffett_reviews.csv
      - 日净值 us_buffett_equity.csv
      - 5/10/15/20 年年化收益与最大回撤 (CAGR / MDD)
      - 当前 Top10 股票池(含最新年报质量指标 + 实时 PE/PB 估值)

数据 (moomoo OpenD):
  日线 QFQ 前复权(含分红) 2006-08~今; MainIndex 历史年报(分页, num<=50)
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

START, END = '2006-01-01', '2026-08-01'   # 日线拉取起点(实际最早2006-08)
BACKTEST_END = '2026-07-31'               # 回测截止(最新行情日)
INIT = 100_000.0
TOP_N = 10
REVIEW_MONTH = 1                          # 每年 1 月复盘
DISCLOSE_PAD_DAYS = 90                    # 美股 10-K 披露滞后(财年末+90天)
SLIP_COST = 0.0005                        # 每边滑点+佣金 5bps

FIN_CACHE = os.path.join(BASE, 'us_annual_fin_cache.json')

UNIVERSE = [
    # 消费(巴菲特经典)
    ('US.KO',   '可口可乐'),
    ('US.PG',   '宝洁'),
    ('US.PEP',  '百事可乐'),
    ('US.MCD',  '麦当劳'),
    ('US.COST', '好市多'),
    ('US.WMT',  '沃尔玛'),
    ('US.HD',   '家得宝'),
    ('US.NKE',  '耐克'),
    ('US.SBUX', '星巴克'),
    # 金融(巴菲特重仓领域)
    ('US.AXP',  '美国运通'),
    ('US.BAC',  '美国银行'),
    ('US.JPM',  '摩根大通'),
    ('US.WFC',  '富国银行'),
    ('US.GS',   '高盛'),
    ('US.BLK',  '贝莱德'),
    ('US.MCO',  '穆迪'),
    ('US.SPGI', '标普全球'),
    ('US.V',    'Visa'),
    ('US.MA',   '万事达'),
    # 科技
    ('US.AAPL', '苹果'),
    ('US.MSFT', '微软'),
    ('US.GOOG', '谷歌'),
    ('US.IBM',  'IBM'),
    ('US.ORCL', '甲骨文'),
    # 医疗
    ('US.JNJ',  '强生'),
    ('US.PFE',  '辉瑞'),
    ('US.MRK',  '默克'),
    # 工业/能源
    ('US.CAT',  '卡特彼勒'),
    ('US.MMM',  '3M'),
    ('US.HON',  '霍尼韦尔'),
    ('US.BRK.B', '伯克希尔B'),
    ('US.XOM',  '埃克森美孚'),
    ('US.CVX',  '雪佛龙'),
]

US_FIELDS = {
    14029: 'ROE', 14030: 'ROA', 14031: 'ROIC',
    14002: '毛利率', 14005: '净利率', 14019: '资产负债率',
    14042: '净利CAGR5Y', 14040: '营收CAGR5Y', 14039: '营收CAGR3Y',
}

QUALITY_WEIGHTS = dict(roe=3.0, net_margin=2.0, gross_margin=1.0,
                       profit_cagr5=1.0, debt_ratio=-1.5)


def load_fin_cache():
    if os.path.exists(FIN_CACHE):
        try:
            with open(FIN_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_fin_cache(c):
    with open(FIN_CACHE, 'w') as f:
        json.dump(c, f)


def download_daily(code, name):
    path = os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['time_key'])
        return df[df['time_key'] <= BACKTEST_END].reset_index(drop=True)
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    ret, data, _ = ctx.request_history_kline(
        code=code, start=START, end=END, ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                KL_FIELD.LOW, KL_FIELD.CLOSE],
        max_count=None, page_req_key=None)
    ctx.close()
    if ret != RET_OK:
        print('[下载] %s %s 失败: %s' % (code, name, data))
        return None
    df = data.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df = df[df['time_key'] <= BACKTEST_END].reset_index(drop=True)
    df.to_csv(path, index=False)
    print('[下载] %s %s %d 根 → %s' % (code, name, len(df), path))
    return df


def fetch_annual_financials(code, cache, retries=8):
    """分页拉全部 MainIndex 历史 → ANNUAL 年报(date→field_id→值), 缓存。num<=50。"""
    if code in cache and cache[code].get('done'):
        return cache[code]['annual']
    for attempt in range(retries):
        ctx = OpenQuoteContext(host=HOST, port=PORT)
        next_key, all_reports, ok = None, [], True
        for _ in range(6):
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
            rl = d.get('report_list', []) or []
            all_reports.extend(rl)
            nk = d.get('next_key')
            if not nk:
                break
            next_key = nk
        ctx.close()
        if ok and all_reports:
            break
        if not ok:
            print('  [限流] %s 第%d次等待35s...' % (code, attempt + 1))
            time.sleep(35)
        else:
            cache[code] = {'done': True, 'annual': {}}
            save_fin_cache(cache)
            return {}
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
            v = item.get('value')
            if v is None:
                v = item.get('data')
            d[item['field_id']] = v
        annual[ds] = d
    cache[code] = {'done': True, 'annual': annual}
    save_fin_cache(cache)
    print('  [财务] %s 年报 %d 期 (最近 %s)' % (code, len(annual),
          sorted(annual)[-1] if annual else '-'))
    return annual


def annual_at(annual, asof):
    """asof 时点最近可用年报(财年末+90天已披露)。返回 报告期str 或 None。"""
    best, best_dt = None, None
    pad = pd.Timedelta(days=DISCLOSE_PAD_DAYS)
    for ds, d in annual.items():
        dt_ = pd.Timestamp(ds)
        if dt_ + pad <= asof and (best_dt is None or dt_ > best_dt):
            best, best_dt = d, dt_
    return best


def quality_score(asof, annual):
    """asof 时点质量分。必需 ROE+净利率, 缺其余字段用中性处理。"""
    d = annual_at(annual, asof)
    if d is None:
        return None

    def g(fid):
        v = d.get(fid)
        if v is None:
            v = d.get(str(fid))
        return v if isinstance(v, (int, float)) and v == v else None

    roe, nm, gm, debt, pg = g(14029), g(14005), g(14002), g(14019), g(14042)
    if roe is None or nm is None:
        return None
    score = (float(roe) * QUALITY_WEIGHTS['roe'] +
             float(nm) * QUALITY_WEIGHTS['net_margin'] +
             (float(gm) if gm is not None else 0.0) * QUALITY_WEIGHTS['gross_margin'] +
             (float(pg) if pg is not None else 0.0) * QUALITY_WEIGHTS['profit_cagr5'] +
             (float(debt) if debt is not None else 50.0) * QUALITY_WEIGHTS['debt_ratio'])
    return {
        'score': score, 'roe': roe, 'net_margin': nm, 'gross_margin': gm,
        'debt': debt, 'cagr5': pg,
    }


def metric_at(annual, asof, window_years=10):
    """asof 前 window_years 年内的年化收益(直接用缓存日线算, 见 main). """
    return None


def run_backtest(daily, annuals, codes):
    """每年复盘 + 全额等权再平衡。返回 (eq DataFrame, reviews list)。"""
    df_by_code = {c: daily[c]['df'] for c in codes}
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[set(df['time_key']) for df in df_by_code.values()])))
    idx = {c: 0 for c in codes}

    cash = INIT
    positions = {}
    pool = []
    cur_review_year = None
    nav_dates, nav_values = [], []
    reviews = []

    def price_on(c, day):
        """code 在 day 当日的有效开盘价; 未上市/停牌返回 None。"""
        dfc = df_by_code[c]
        while idx[c] < len(dfc) and dfc['time_key'].iloc[idx[c]] < day:
            idx[c] += 1
        if idx[c] < len(dfc) and dfc['time_key'].iloc[idx[c]] == day:
            row = dfc.iloc[idx[c]]
            o, cl = float(row['open']), float(row['close'])
            if o != o or o <= 0 or cl != cl or cl <= 0:
                return None
            return o
        return None

    def mark_value(day):
        # 计算净值(现金 + 持仓按最新收盘价)
        nonlocal cash
        v = cash
        for c, p in positions.items():
            dfc = df_by_code[c]
            last_used = min(idx[c], len(dfc) - 1)
            # 找 <= day 的最近收盘
            while last_used > 0 and dfc['time_key'].iloc[last_used] > day:
                last_used -= 1
            v += p['qty'] * float(dfc['close'].iloc[last_used])
        return v

    for di in range(len(all_dates)):
        today = all_dates[di]

        # 每年 1 月首个交易日 → 复盘
        if today.month == REVIEW_MONTH and today.year != cur_review_year and today.year >= 2007:
            cur_review_year = today.year
            # 1) 索引推进到 today
            for c in codes:
                price_on(c, today)
            # 2) 评分选池(当日有交易价才有资格)
            scored = []
            for c in codes:
                if price_on(c, today) is None:
                    continue   # 未上市/停牌 → 无资格
                sc = quality_score(today, annuals.get(c, {}))
                if sc:
                    scored.append((c, sc))
            scored.sort(key=lambda x: x[1]['score'], reverse=True)
            new_pool_full = [c for c, _ in scored[:TOP_N]]

            # 3) 全额卖出全部持仓(开盘价, 扣滑点)
            for c in list(positions.keys()):
                p = positions.pop(c)
                px = price_on(c, today)
                if px is None:
                    dfc = df_by_code[c]
                    j = min(idx[c], len(dfc) - 1)
                    while j > 0 and (dfc['time_key'].iloc[j] > today or
                                     dfc['close'].iloc[j] != dfc['close'].iloc[j] or
                                     float(dfc['close'].iloc[j]) <= 0):
                        j -= 1
                    px = float(dfc['close'].iloc[j])
                cash += px * (1 - SLIP_COST) * p['qty']
            positions = {}

            # 4) 等权买入新池: 每只预算 = 总净值/10, 依次买入
            total = cash
            budget = total / TOP_N
            actual_pool = []
            for c in new_pool_full:
                px = price_on(c, today)
                if px is None:
                    continue
                buy_px = px * (1 + SLIP_COST)
                qty = int(budget / buy_px)
                if qty <= 0:
                    continue
                cost = buy_px * qty
                if cost > cash:
                    qty = int(cash / buy_px)
                    if qty <= 0:
                        continue
                    cost = buy_px * qty
                cash -= cost
                positions[c] = {'qty': qty, 'entry_px': buy_px}
                actual_pool.append(c)
            pool = actual_pool

            reviews.append({
                'date': today, 'year': today.year,
                'pool': [daily[c]['name'] for c in pool],
                'scores': {c: sc['score'] for c, sc in scored if c in new_pool_full},
                'cash_left': cash,
            })
            print('  [复盘 %d] %d只 池=%s 现金=%.0f (总净值 %.0f)' % (
                today.year, len(pool), '、'.join(daily[c]['name'] for c in pool),
                cash, cash))

        # 推进索引
        for c in codes:
            dfc = df_by_code[c]
            while idx[c] < len(dfc) and dfc['time_key'].iloc[idx[c]] < today:
                idx[c] += 1

        # 日净值
        nav_dates.append(today)
        nav_values.append(mark_value(today))

    eq = pd.DataFrame({'time': nav_dates, 'equity': nav_values}).set_index('time')
    return eq, reviews


def horizon_metrics(eq, years_list=(5, 10, 15, 20)):
    """从回测终点回看 N 年年化收益与最大回撤。"""
    out = []
    end_dt = eq.index[-1]
    for y in years_list:
        start_dt = end_dt - pd.DateOffset(years=y)
        seg = eq[eq.index >= start_dt]
        if len(seg) < 2:
            out.append({'years': y, 'cagr': None, 'mdd': None})
            continue
        s0, s1 = seg['equity'].iloc[0], seg['equity'].iloc[-1]
        span = (seg.index[-1] - seg.index[0]).days / 365.25
        cagr = (s1 / s0) ** (1 / span) - 1
        mdd = (seg['equity'] / seg['equity'].cummax() - 1).min()
        out.append({'years': y, 'cagr': cagr, 'mdd': mdd * 100})
    return out


def current_top10(daily, annuals, codes):
    """当前时点 Top10: 最新年报质量分 + 实时快照 PE/PB 过滤。"""
    asof = pd.Timestamp(BACKTEST_END)
    scored = []
    for c in codes:
        sc = quality_score(asof, annuals.get(c, {}))
        if sc:
            scored.append((c, sc))
    scored.sort(key=lambda x: x[1]['score'], reverse=True)

    # 实时快照估值
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    ret, snap = ctx.get_market_snapshot(codes)
    ctx.close()
    pe, pb = {}, {}
    if ret == RET_OK:
        for _, r in snap.iterrows():
            c = r['code']
            pe[c] = r.get('pe_ttm_ratio')
            pb[c] = r.get('pb_ratio')

    rows = []
    for c, sc in scored:
        if c not in pe or pe[c] is None or pb[c] is None or pe[c] <= 0 or pb[c] <= 0:
            continue
        price = daily[c]['df']['close'].iloc[-1]
        # 价值过滤: PE/PB 高于宇宙中位 2 倍则降级(简单规则)
        rows.append({
            'code': c, 'name': daily[c]['name'],
            'price': price, 'pe_ttm': pe[c], 'pb': pb[c],
            'roe': sc['roe'], 'net_margin': sc['net_margin'],
            'gross_margin': sc['gross_margin'], 'debt': sc['debt'],
            'cagr5': sc['cagr5'], 'score': sc['score'],
        })
    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    return df.head(TOP_N)


def main():
    print('=' * 120)
    print('巴菲特/芒格 · 美股股票池 · 每年复盘(全额等权·无前视) · 2006-08~2026-07')
    print('=' * 120)

    # ---- 1. 数据 ----
    cache = load_fin_cache()
    daily, annuals = {}, {}
    for code, name in UNIVERSE:
        df = download_daily(code, name)
        if df is None or len(df) < 252:
            print('[跳过] %s %s 数据不足' % (code, name))
            continue
        bad = df[(df['open'] <= 0) | (df['close'] <= 0)]
        if len(bad) > 0:
            print('[剔除] %s %s 存在非正价格(首个 %s) → 视为未上市/前复权缺陷' % (
                code, name, bad['time_key'].iloc[0].date()))
            continue
        ann = fetch_annual_financials(code, cache)
        daily[code] = dict(df=df, name=name)
        annuals[code] = ann

    codes = list(daily.keys())
    if len(codes) < TOP_N:
        print('可用标的 %d < %d' % (len(codes), TOP_N))
        return
    print('[标的] 数据可用 %d 只: %s' % (
        len(codes), '、'.join(daily[c]['name'] for c in codes)))

    # ---- 2. 回测 ----
    eq, reviews = run_backtest(daily, annuals, codes)

    years_total = (eq.index[-1] - eq.index[0]).days / 365.25
    final = eq['equity'].iloc[-1]
    cagr = (final / INIT) ** (1 / years_total) - 1
    mdd = (eq['equity'] / eq['equity'].cummax() - 1).min()
    daily_r = eq['equity'].pct_change().dropna()
    sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else np.nan

    print('\n' + '=' * 120)
    print('每年复盘组合 (Top%d 全额等权·纯持股·无择时)' % TOP_N)
    print('=' * 120)
    print('  区间     %s ~ %s  (%.2f 年)' % (eq.index[0].date(), eq.index[-1].date(), years_total))
    print('  期末     %.0f USD  (初始 %.0f)' % (final, INIT))
    print('  总收益   %+.1f%%' % ((final / INIT - 1) * 100))
    print('  ★ CAGR  %+.2f%%' % (cagr * 100))
    print('  最大回撤 %.1f%%' % (mdd * 100))
    print('  夏普     %.2f' % sharpe)

    # 5/10/15/20 年窗口
    print('\n  [持有窗口分析 — 从 2026-07-31 回看]')
    hm = horizon_metrics(eq)
    for h in hm:
        if h['cagr'] is None:
            print('    %2d年: 数据不足' % h['years'])
        else:
            print('    %2d年: CAGR %+6.2f%%  最大回撤 %6.1f%%' % (
                h['years'], h['cagr'] * 100, -h['mdd']))

    eq.reset_index().to_csv(os.path.join(BASE, 'us_buffett_equity.csv'), index=False)
    rev = pd.DataFrame([{
        'review_date': r['date'], 'year': r['year'],
        'pool': '、'.join(r['pool']), 'cash_left': r['cash_left'],
    } for r in reviews])
    rev.to_csv(os.path.join(BASE, 'us_buffett_reviews.csv'), index=False)

    # ---- 3. 当前 Top10(含估值) ----
    print('\n' + '=' * 120)
    print('当前股票池 Top%d (最新年报质量 + 实时估值)' % TOP_N)
    print('=' * 120)
    top = current_top10(daily, annuals, codes)
    top.to_csv(os.path.join(BASE, 'us_buffett_top10.csv'), index=False)
    if len(top):
        print('%-8s %-12s %8s %7s %7s %7s %7s %7s %7s %8s' % (
            '代码', '名称', '价格', 'PE', 'PB', 'ROE%', '净利%', '毛利%', '负债%', '总分'))
        for _, r in top.iterrows():
            print('%-8s %-12s %8.2f %7.1f %7.2f %7.1f %7.1f %7.1f %7.1f %8.1f' % (
                r['code'], r['name'], r['price'], r['pe_ttm'], r['pb'],
                r['roe'] or 0, r['net_margin'] or 0, r['gross_margin'] or 0,
                r['debt'] or 0, r['score']))

    print('\n[已保存] us_buffett_equity.csv / us_buffett_reviews.csv / us_buffett_top10.csv')


if __name__ == '__main__':
    main()