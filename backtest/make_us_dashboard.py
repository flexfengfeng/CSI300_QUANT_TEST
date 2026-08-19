#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「巴菲特/芒格 · 美股股票池」交互式 HTML Dashboard
======================================================
数据源(均在 backtest/ 目录):
  us_buffett_equity.csv    每年复盘组合日净值
  daily_US_SPY.csv         SPY 标普500 (QFQ 含分红)
  daily_US_QQQ.csv         QQQ 纳指100 (QFQ 含分红)
  us_buffett_top10.csv     当前 Top10 股票池(质量+估值)
  us_buffett_reviews.csv   每年复盘池记录

核心板块:
  ① 概览卡片(期末净值/总收益/CAGR/最大回撤)
  ② 净值对比曲线(策略 vs SPY vs QQQ, 线性/对数切换)
  ③ 回撤面积图
  ④ 5/10/15/20 年年化收益×最大回撤矩阵(核心要求)
  ⑤ 分年度收益柱状图
  ⑥ 当前 Top10 股票池表格(ROE/净利率/毛利率/负债/PE/PB)
  ⑦ 每年复盘池记录表

输出: backtest/us_buffett_dashboard.html
"""
import os
import json
import numpy as np
import pandas as pd

BASE = '/Users/fengfeng/Dev/moomoo/backtest'
INIT = 100_000.0
STRATEGY_COLOR = '#F4A261'
SPY_COLOR = '#38B26B'
QQQ_COLOR = '#6C9EF1'


def load_eq(path):
    df = pd.read_csv(os.path.join(BASE, path), parse_dates=['time'])
    return df.set_index('time')['equity']


def load_etf_close(code):
    df = pd.read_csv(os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv'),
                     parse_dates=['time_key'])
    s = df.set_index('time_key')['close']
    return s / s.iloc[0] * INIT


def build_series(name, eq, color, label):
    eq = eq[~eq.index.duplicated(keep='last')].sort_index()
    eq = eq[eq.index >= '2006-08-03']
    init_v = float(eq.iloc[0])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = float(eq.iloc[-1])
    cagr = (final / init_v) ** (1 / years) - 1
    total = final / init_v - 1
    dd = (eq / eq.cummax() - 1) * 100
    mdd = float(dd.min())

    # 分年度收益
    yearly = eq.resample('YE').last()
    y_ret = (yearly / yearly.shift(1) - 1).dropna() * 100
    first_yr = eq.index[0].year
    first_ret = (eq[eq.index.year == first_yr].iloc[-1] / init_v - 1) * 100
    y_ser = pd.concat([
        pd.Series([first_ret], index=[pd.Timestamp(f'{first_yr}-12-31')]),
        y_ret,
    ])

    return {
        'name': name, 'label': label, 'color': color,
        'final': final, 'init': init_v, 'cagr': float(cagr * 100),
        'total': float(total * 100), 'mdd': mdd, 'years': years,
        'dates': [d.strftime('%Y-%m-%d') for d in eq.index],
        'equity': [round(float(x), 2) if x == x else None for x in eq],
        'dd': [round(float(x), 2) if x == x else None for x in dd],
        'year_labels': [int(y.year) for y in y_ser.index],
        'year_ret': [round(float(x), 2) if x == x else None for x in y_ser],
    }


def horizon_table():
    """从 2026-07-31 回看 5/10/15/20 年 CAGR×MDD, 策略 vs SPY vs QQQ。"""
    eq = load_eq('us_buffett_equity.csv')
    spy = load_etf_close('US.SPY')
    qqq = load_etf_close('US.QQQ')
    out = []
    end_dt = eq.index[-1]
    for y in (5, 10, 15, 20):
        start_dt = end_dt - pd.DateOffset(years=y)
        row = {'years': y}
        for key, s in [('strategy', eq), ('spy', spy), ('qqq', qqq)]:
            seg = s[s.index >= start_dt]
            if len(seg) < 2:
                row[key + '_cagr'], row[key + '_mdd'] = None, None
                continue
            s0, s1 = seg.iloc[0], seg.iloc[-1]
            span = (seg.index[-1] - seg.index[0]).days / 365.25
            row[key + '_cagr'] = ((s1 / s0) ** (1 / span) - 1) * 100
            row[key + '_mdd'] = (seg / seg.cummax() - 1).min() * 100
        out.append(row)
    return out


def yearly_returns_table(eq, spy, qqq):
    """对齐区间(2006-08 起)的分年度收益对比。"""
    dfs = []
    for name, s in [('策略', eq), ('SPY', spy), ('QQQ', qqq)]:
        t = pd.DataFrame({'v': s})
        t['year'] = t.index.year
        groups = []
        for y, g in t.groupby('year'):
            ret = (g['v'].iloc[-1] / g['v'].iloc[0] - 1) * 100
            groups.append({'year': y, name: round(float(ret), 1)})
        dfs.append(pd.DataFrame(groups).set_index('year'))
    df = pd.concat(dfs, axis=1)
    return df.reset_index().to_dict('records')


def top10_table():
    path = os.path.join(BASE, 'us_buffett_top10.csv')
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict('records')


def reviews_table():
    path = os.path.join(BASE, 'us_buffett_reviews.csv')
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=['review_date'])
    df['review_date'] = df['review_date'].dt.strftime('%Y-%m-%d')
    return df.to_dict('records')


def build():
    strat = build_series('策略', load_eq('us_buffett_equity.csv'),
                         STRATEGY_COLOR, '巴菲特池 Top10（每年复盘）')
    spy = build_series('SPY', load_etf_close('US.SPY'), SPY_COLOR, '标普500 SPY')
    qqq = build_series('QQQ', load_etf_close('US.QQQ'), QQQ_COLOR, '纳指100 QQQ')

    series = {'strategy': strat, 'spy': spy, 'qqq': qqq}
    horizon = horizon_table()
    yearly = yearly_returns_table(
        load_eq('us_buffett_equity.csv'), load_etf_close('US.SPY'), load_etf_close('US.QQQ'))
    top10 = top10_table()
    reviews = reviews_table()

    payload = json.dumps({
        'series': series,
        'horizon': horizon,
        'yearly': yearly,
        'top10': [{k: (None if (isinstance(v, float) and v != v) else v)
                   for k, v in r.items()} for r in top10],
        'reviews': reviews,
        'init': INIT,
    }, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>巴菲特/芒格 · 美股股票池 · 每年复盘回测</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#0f1420; --card:#151c2c; --line:#22304a; --txt:#dfe6f2; --muted:#8fa3c0; --acc:#4ea1ff; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--txt); font-family:-apple-system,'SF Pro Display','PingFang SC','Microsoft YaHei',sans-serif; padding:24px; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:24px; font-weight:700; margin-bottom:4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:22px; line-height:1.6; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
  .card .k { font-size:11px; color:var(--muted); margin-bottom:6px; }
  .card .v { font-size:22px; font-weight:700; }
  .card .d { font-size:11px; color:var(--muted); margin-top:4px; }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  .seg { display:flex; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { background:transparent; color:var(--muted); border:none; padding:7px 14px; font-size:12px; cursor:pointer; }
  .seg button.on { background:var(--acc); color:#fff; }
  .plot { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:6px; margin-bottom:20px; }
  .plot .t { font-size:13px; color:var(--muted); padding:8px 12px 0; }
  .section { margin-bottom:26px; }
  .section h2 { font-size:16px; margin-bottom:10px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; }
  tr:hover td { background:rgba(78,161,255,0.06); }
  .pos { color:#57d98a; } .neg { color:#f26d6d; }
  .legend { display:flex; gap:18px; font-size:12px; color:var(--muted); margin-bottom:8px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-2px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:6px; }
  .badge.hot { background:rgba(244,162,97,.18); color:#F4A261; }
  .footer { color:var(--muted); font-size:11px; line-height:1.8; margin-top:10px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>巴菲特/芒格 · 美股股票池</h1>
  <div class="sub">候选宇宙 33 只美股护城河龙头 → 每年 1 月按「当时已披露年报」质量评分(ROE×3 + 净利率×2 + 毛利率×1 + 净利CAGR5Y×1 − 负债率×1.5)取 Top10 等权持有，全年无择时 · QFQ 前复权(含分红) · 双边成本 5bps/边 · 2006-08 ~ 2026-07（20 年）</div>

  <div class="cards" id="cards"></div>

  <div class="section">
    <div class="legend">
      <span><i style="background:#F4A261"></i>巴菲特池 Top10</span>
      <span><i style="background:#38B26B"></i>标普500 SPY</span>
      <span><i style="background:#6C9EF1"></i>纳指100 QQQ</span>
    </div>
    <div class="plot"><div class="t">① 净值对比曲线（1 万美元→ 每期等权再平衡）</div><div id="g1"></div></div>
    <div class="plot"><div class="t">② 回撤（水下面积 %，与上图 x 轴联动）</div><div id="g2"></div></div>
  </div>

  <div class="section">
    <h2>③ 持有窗口 · 年化收益 × 最大回撤（从 2026-07 回看）</h2>
    <div style="overflow-x:auto"><table id="horizonTab"></table></div>
  </div>

  <div class="section">
    <h2>④ 分年度收益（%）</h2>
    <div class="plot"><div id="g3"></div></div>
  </div>

  <div class="section">
    <h2>⑤ 当前股票池 Top10（最新年报质量 + 实时估值）</h2>
    <div style="overflow-x:auto"><table id="topTab"></table></div>
  </div>

  <div class="section">
    <h2>⑥ 每年复盘记录</h2>
    <div style="overflow-x:auto; max-height:360px; overflow-y:auto"><table id="revTab"></table></div>
  </div>

  <div class="footer">
    口径：每年 1 月首个交易日复盘，用财年末+90 天(10-K 披露)以内的最近年报评分；当日无交易价(未上市/停牌)的标的不具备入池资格，消除前视。<br>
    净值按每股子货币 USD，初始 100,000 × 每年全额再平衡（平仓余额 ÷10 等权买入），含 0.05% 单边滑点+佣金；SPY/QQQ 为同期 QFQ 日线归一化对照（含分红再投资）。<br>
    回撤 = 净值/历史峰值 − 1；CAGR 按 365.25 天/年复利；本页为历史回测演示，不构成投资建议。
  </div>
</div>

<script>
const D = __DATA__;
const S = D.series;
const grid = {
  margin:{l:64,r:12,t:14,b:44}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(21,28,44,0.55)',
  font:{color:'#8fa3c0', size:11}
};

function fmt(n, d=0){ return n == null ? '—' : n.toLocaleString('en-US',{maximumFractionDigits:d}); }
function fmtPct(n, d=1, sign=false){ if(n == null) return '—'; const s = sign ? (n>=0?'+':'') : (n>0?'':'') ; return s + n.toFixed(d) + '%'; }
function fmtCagr(n){ return n == null ? '—' : (n>=0?'+':'') + n.toFixed(2) + '%'; }

// ① 概览卡片
(function(){
  const st = S.strategy;
  const cards = [
    ['期末净值', '$' + fmt(Math.round(st.final)), '初始 $100,000 · ' + st.years.toFixed(1) + ' 年'],
    ['累计总收益', '+' + st.total.toFixed(0) + '%', '扣除全部交易成本'],
    ['年复合 CAGR', '+' + st.cagr.toFixed(2) + '%', '复利年化'],
    ['最大回撤', st.mdd.toFixed(1) + '%', '峰谷最深跌幅'],
  ];
  document.getElementById('cards').innerHTML = cards.map(([k,v,d]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
})();

// ② 净值 + 回撤
(function(){
  const traces = ['strategy','spy','qqq'].map(k => ({
    x: S[k].dates, y: S[k].equity, name: S[k].label, type:'scatter', mode:'lines',
    line:{color:S[k].color, width: k==='strategy'?2.6:1.4},
    hovertemplate:'%{y:,.0f} <extra>'+S[k].label+'</extra>'
  }));
  Plotly.newPlot('g1', traces, {
    ...grid,
    xaxis:{type:'date', tickformat:'%Y-%m', gridcolor:'#22304a', zeroline:false},
    yaxis:{type:'log', title:'净值 $', gridcolor:'#22304a', zeroline:false},
    hovermode:'x unified', showlegend:true, legend:{orientation:'h', y:1.12, x:0}
  }, {responsive:true, displayModeBar:true});

  const ddTr = ['strategy','spy','qqq'].map(k => ({
    x: S[k].dates, y: S[k].dd, name: S[k].label + ' 回撤', type:'scatter', mode:'lines',
    line:{color:S[k].color === '#F4A261' ? '#e05d5d' : S[k].color, width: k==='strategy'?1.8:1.2, dash: k==='strategy'?'solid':'dot'},
    fill: k==='strategy' ? 'tozeroy' : 'none',
    fillcolor: k==='strategy' ? 'rgba(224,93,93,0.20)' : undefined,
    hovertemplate:'%{y:.1f}%<extra>'+S[k].label+'</extra>'
  }));
  Plotly.newPlot('g2', ddTr, {
    ...grid,
    xaxis:{type:'date', tickformat:'%Y-%m', gridcolor:'#22304a', zeroline:false},
    yaxis:{title:'回撤 (%)', range:[-56, 5], gridcolor:'#22304a', zeroline:true, zerolinecolor:'#3a4a6a'},
    hovermode:'x unified', showlegend:true, legend:{orientation:'h', y:1.12, x:0}
  }, {responsive:true, displayModeBar:false});

  document.getElementById('g1').on('plotly_relayout', evt => {
    const r0 = evt['xaxis.range[0]'], r1 = evt['xaxis.range[1]'];
    if (r0 !== undefined || r1 !== undefined)
      Plotly.relayout('g2', {'xaxis.range':[r0,r1], 'xaxis.autorange':false});
  });
})();

// ③ 持有窗口表
(function(){
  const rows = D.horizon.map(h => {
    const cell = (c, m) => {
      if (h[c+'_cagr'] == null || h[c+'_mdd'] == null) return '<td>—</td><td>—</td>';
      return `<td class="${h[c+'_cagr']>=0?'pos':'neg'}">${fmtCagr(h[c+'_cagr'])}</td>
              <td class="${h[c+'_mdd']<0?'neg':'pos'}">${fmtPct(h[c+'_mdd'])}</td>`;
    };
    const label = h.years === 20 ? '20 年（2006-08 起）' : h.years + ' 年';
    return `<tr><td>${label}</td>${cell('strategy')}${cell('spy')}${cell('qqq')}</tr>`;
  }).join('');
  document.getElementById('horizonTab').innerHTML =
    `<tr><th>窗口</th><th colspan="2">巴菲特池 CAGR / MDD</th><th colspan="2">SPY CAGR / MDD</th><th colspan="2">QQQ CAGR / MDD</th></tr>` + rows;
})();

// ④ 分年度柱状
(function(){
  const bar = [];
  ['strategy','spy','qqq'].forEach((k, i) => {
    const c = S[k].color;
    bar.push({
      x: S[k].year_labels, y: S[k].year_ret, name: S[k].label, type:'bar',
      marker:{color: S[k].year_ret.map(v => v>=0 ? c : '#e05d5d')},
      opacity: i===0 ? 1 : 0.55,
      hovertemplate:'%{y:.1f}%<extra>'+S[k].label+'</extra>'
    });
  });
  Plotly.newPlot('g3', bar, {
    ...grid,
    xaxis:{type:'category', gridcolor:'#22304a'},
    yaxis:{title:'年度收益 (%)', gridcolor:'#22304a', zeroline:true, zerolinecolor:'#3a4a6a'},
    barmode:'group', hovermode:'x unified', showlegend:true, legend:{orientation:'h', y:1.12, x:0}
  }, {responsive:true, displayModeBar:false});
})();

// ⑤ Top10 表
(function(){
  const rows = D.top10.map((r, i) => `<tr>
    <td>${i+1}</td>
    <td>${r['code']||''}</td>
    <td>${r['name']||''}</td>
    <td>${fmt(r['price'], 2)}</td>
    <td>${fmt(r['pe_ttm'], 1)}</td>
    <td>${fmt(r['pb'], 2)}</td>
    <td>${fmt(r['roe'], 1)}%</td>
    <td>${fmt(r['net_margin'], 1)}%</td>
    <td>${fmt(r['gross_margin'], 1)}%</td>
    <td>${fmt(r['debt'], 1)}%</td>
    <td><span class="badge hot">${fmt(r['score'], 0)}</span></td>
  </tr>`).join('');
  document.getElementById('topTab').innerHTML =
    `<tr><th>#</th><th>代码</th><th>名称</th><th>价格$</th><th>PE(TTM)</th><th>PB</th><th>ROE%</th><th>净利率%</th><th>毛利率%</th><th>负债率%</th><th>质量总分</th></tr>` + rows;
})();

// ⑥ 复盘记录表
(function(){
  const rows = D.reviews.map(r => `<tr>
    <td>${r['year']}</td>
    <td>${r['review_date']||''}</td>
    <td>${r['pool']||''}</td>
    <td>$${fmt(Math.round(r['cash_left']||0))}</td>
  </tr>`).join('');
  document.getElementById('revTab').innerHTML =
    `<tr><th>年份</th><th>复盘日</th><th>入选 Top10（质量评分）</th><th>复盘后剩余现金</th></tr>` + rows;
})();
</script>
</body>
</html>"""

    html = html.replace('__DATA__', payload)
    out = os.path.join(BASE, 'us_buffett_dashboard.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('已生成:', out)
    st = series['strategy']
    print('  策略: CAGR %+.2f%%  MDD %.1f%%  期末 $%.0f' % (st['cagr'], st['mdd'], st['final']))
    for h in horizon:
        print('  %2d年: 策略CAGR %+6.2f%% / MDD %6.1f%%   SPY %+6.2f%%   QQQ %+6.2f%%' % (
            h['years'], h.get('strategy_cagr') or 0, h.get('strategy_mdd') or 0,
            h.get('spy_cagr') or 0, h.get('qqq_cagr') or 0))


if __name__ == '__main__':
    build()