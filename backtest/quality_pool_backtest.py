#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量池组合回测 —— 巴菲特筛选 Top10 + MA10/30 择时
====================================================
标的池 = buffett_screener.py 输出的 Top 10 港股(应用户要求):
  腾讯/安踏/中海油/石药/创科/神华/新鸿基/舜宇/李宁/中移动

方法:
  - 每只股票下载 2014-01 ~ 今 日线(qfq), 缓存 CSV
  - MA10/30 只多择时: 昨日收盘 MA10>=MA30 → 今日持有该股, 否则空仓(持币)
  - 组合=等权子账户: 初始资金 10万 / 10只 = 每子账户 1万
  - 子账户按自己信号持有/持币; 信号翻转日按当日开盘价±滑点成交, 计入全部费用
  - 组合净值 = Σ子账户净值; 输出 CAGR/回撤/分年度/各标的贡献

偏差说明: 标的池基于"当前时点"财报筛选(前视), 回测收益上偏;
        用于展示"质量池+择时"框架可行性, 非可实盘的保证。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

from moomoo import (OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK)

HOST, PORT = '127.0.0.1', 11111
BASE = os.path.dirname(os.path.abspath(__file__))
START, END = '2014-01-01', '2026-06-02'
INIT = 100_000.0

# 质量池 Top10 (代码, 名称, 卖出印花税率, 手数, 滑点)
POOL = [
    ('HK.00700', '腾讯',     0.001, 100, 0.0005),
    ('HK.02020', '安踏体育', 0.001, 200, 0.0005),
    ('HK.00883', '中海油',   0.001, 1000, 0.0005),
    ('HK.01093', '石药集团', 0.001, 2000, 0.0005),
    ('HK.00669', '创科实业', 0.001, 500, 0.0005),
    ('HK.01088', '中国神华', 0.001, 500, 0.0005),
    ('HK.00016', '新鸿基',   0.001, 1000, 0.0005),
    ('HK.02382', '舜宇光学', 0.001, 100, 0.0005),
    ('HK.02331', '李宁',     0.001, 500, 0.0005),
    ('HK.00941', '中国移动', 0.001, 500, 0.0005),
]
COMMISSION_RATE, COMMISSION_MIN = 0.0003, 5.0
TRADE_FEE_R, LEVY_R, SYS_FEE = 0.0000565, 0.000027, 0.5
FAST_MA, SLOW_MA = 10, 30


def cost_of(notional, stamp):
    commission = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    return commission + notional * stamp + notional * TRADE_FEE_R + notional * LEVY_R + SYS_FEE


def download(code):
    path = os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['time_key'])
        print(f'[缓存] {code} {len(df)}根  {df["time_key"].iloc[0].date()}~{df["time_key"].iloc[-1].date()}')
        return df
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    ret, data, page_key = ctx.request_history_kline(
        code=code, start=START, end=END, ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                KL_FIELD.LOW, KL_FIELD.CLOSE, KL_FIELD.TRADE_VOL, KL_FIELD.TRADE_VAL],
        max_count=None, page_req_key=None)
    ctx.close()
    if ret != RET_OK:
        print(f'[下载] {code} 失败: {data}')
        return None
    df = data.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.to_csv(path, index=False)
    print(f'[下载] {code} {len(df)}根 → {path}')
    return df


def run_single(code, name, stamp, lot, slip, df):
    """单标的自有账户: 初始 1万, MA10/30 只多, 信号翻转日成交计费。"""
    dates = df['time_key']
    opens = df['open'].to_numpy()
    closes = df['close'].to_numpy()
    n = len(df)

    ma_fast = pd.Series(closes).rolling(FAST_MA).mean().to_numpy()
    ma_slow = pd.Series(closes).rolling(SLOW_MA).mean().to_numpy()

    cash = INIT / len(POOL)
    qty = 0
    entry_px = 0.0
    nav = []                     # (date, nav) 仅翻转日记录
    trades = []

    def mark(i):
        nav.append((dates.iloc[i], cash + qty * closes[i] if qty else cash))

    for i in range(n):
        mark(i)  # 每日记录净值
        if i == 0 or np.isnan(ma_slow[i - 1]):
            continue
        sig = 1 if ma_fast[i - 1] >= ma_slow[i - 1] else -1
        o = opens[i]
        if o is None or (isinstance(o, float) and (o != o or o <= 0)):
            continue  # 停牌/无成交日

        # 平仓(用收盘价估值时已含 open>0 保护)
        if qty > 0 and sig == -1:
            sell_px = o * (1 - slip)
            close_cost = cost_of(sell_px * qty, stamp)
            gross = (sell_px - entry_px) * qty
            net = gross - entry_cost - close_cost
            cash += sell_px * qty - close_cost
            trades.append(net)
            qty = 0

        # 开仓(等价权预算; 复利按子账户净值重算)
        if qty == 0 and sig == 1:
            equity_now = cash
            raw = equity_now / (o * (1 + slip))
            buy_qty = int(raw / lot) * lot
            if buy_qty > 0:
                entry_px = o * (1 + slip)
                entry_cost = cost_of(entry_px * buy_qty, stamp)
                if entry_px * buy_qty + entry_cost <= cash:
                    cash -= entry_px * buy_qty + entry_cost
                    qty = buy_qty

    # 期末平仓
    if qty > 0:
        sell_px = closes[-1] * (1 - slip)
        close_cost = cost_of(sell_px * qty, stamp)
        gross = (sell_px - entry_px) * qty
        trades.append(gross - entry_cost - close_cost)
        cash += sell_px * qty - close_cost
        qty = 0

    return cash, nav, trades


def main():
    print('=' * 120)
    print('质量池组合回测 · 巴菲特Top10 + MA10/30 只多择时 · 等权子账户 · 初始 10万HKD')
    print('=' * 120)

    results = {}
    for code, name, stamp, lot, slip in POOL:
        df = download(code)
        if df is None or len(df) < 60:
            print(f'[跳过] {code} 数据不足')
            continue
        final, nav, trades = run_single(code, name, stamp, lot, slip, df)
        results[code] = dict(name=name, final=final, nav=nav, trades=trades,
                             n_trade=len(trades))
        print(f'  {code} {name:<8s} 期末={final:>10,.0f}  交易={len(trades)}笔')

    # 组合净值: 各子账户日净值相加(日期对齐)
    nav_all = {}
    for code, r in results.items():
        for d, v in r['nav']:
            nav_all.setdefault(d, {}).setdefault(code, v)
    dates = sorted(nav_all)
    equity = []
    for d in dates:
        daymap = nav_all[d]
        # 子账户无记录日 → 用其初值(起投); 近似
        val = sum(daymap.get(c, INIT / len(POOL)) for c in results)
        equity.append((d, val))
    eq = pd.DataFrame(equity, columns=['time', 'equity']).set_index('time')

    # 指标
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = eq['equity'].iloc[-1]
    cagr = (final / INIT) ** (1 / years) - 1
    mdd = (eq['equity'] / eq['equity'].cummax() - 1).min()
    daily = eq['equity'].pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float('nan')

    print('\n' + '=' * 120)
    print('组合结果')
    print('=' * 120)
    print(f'  标的池: {len(results)} 只 ({"、".join(r["name"] for r in results.values())})')
    print(f'  区间:   {eq.index[0].date()} ~ {eq.index[-1].date()}  ({years:.1f} 年)')
    print(f'  期末:   {final:,.0f} HKD   (初始 {INIT:,.0f})')
    print(f'  总收益: {final/INIT-1:+.1%}')
    print(f'  ★ CAGR: {cagr:+.2%}')
    print(f'  最大回撤: {mdd:.1%}')
    print(f'  夏普:   {sharpe:.2f}')

    # 分年度
    eqy = eq.assign(year=eq.index.year).groupby('year')['equity']
    print('\n  分年度:')
    prev = INIT
    for y, g in eqy:
        start = g.iloc[0]
        end = g.iloc[-1]
        ret = end / start - 1
        print('    %d: %+6.1f%%   (净值 %.0f -> %.0f)' % (y, ret * 100, start, end))

    # 各标的贡献
    print('\n  各标的期末(子账户)与交易次数:')
    for code, r in results.items():
        print('    %-8s %-8s 期末=%10.0f  交易=%3d笔  均净/笔=%+8.0f' % (
            code, r['name'], r['final'], r['n_trade'],
            np.mean(r['trades']) if r['trades'] else 0))

    # 保存
    eq.reset_index().to_csv(os.path.join(BASE, 'quality_pool_equity.csv'), index=False)
    print('\n[已保存] quality_pool_equity.csv')


if __name__ == '__main__':
    main()