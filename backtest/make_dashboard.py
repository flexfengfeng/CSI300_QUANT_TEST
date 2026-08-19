#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成交互式投资仪表盘 HTML（净值/收益/回撤联动）
=================================================
数据源:
  滚动池   backtest/rolling_pool_equity.csv   (季度滚动池, 无前视, 修复后)
  标普500  US.SPY   (同期 QFQ 前复权, 含分红再投资)
  纳指100  US.QQQ   (同期 QFQ 前复权, 含分红再投资)

美股基准口径: 以100,000初始资金按比重买入持有至期末(归一化净值),
             使用与策略回测相同的初始资金, 便于直接对比。
功能: 4 系列主视图切换 + 任意两两对比, 净值(线性/对数)、回撤面积、
      年度收益柱状图, hover 联动, 选框缩放。
输出: backtest/investment_dashboard.html
"""
import os
import json
import numpy as np
import pandas as pd

from moomoo import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK

BASE = os.path.dirname(os.path.abspath(__file__))
START, END = '2014-01-02', '2026-06-02'
INIT = 100_000.0

STRATEGIES = [
    ('rolling_pool_equity.csv', '季度滚动池（无前视·可实盘参考）', '#2E86AB'),
]
US_INDEXES = [
    ('US.SPY', '标普500 SPY（含分红再投资）', '#38B26B'),
    ('US.QQQ', '纳指100 QQQ（含分红再投资）', '#F4A261'),
]


def load(eq):
    if isinstance(eq, str):
        df = pd.read_csv(os.path.join(BASE, eq), parse_dates=['time'])
        return df.set_index('time')['equity']
    return eq


def fetch_us(code):
    path = os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['time_key'])
        return df
    ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, _ = ctx.request_history_kline(
        code=code, start=START, end=END, ktype=KLType.K_DAY, autype=AuType.QFQ,
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE], max_count=None)
    ctx.close()
    if ret != RET_OK:
        raise RuntimeError(f'{code} 拉取失败: {data}')
    df = data.drop_duplicates(subset='time_key').sort_values('time_key').reset_index(drop=True)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.to_csv(path, index=False)
    return df


def build_series(equity, label, color):
    if isinstance(equity, str):
        eq = load(equity)
    else:
        eq = equity
    init = float(eq.iloc[0])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = float(eq.iloc[-1])
    cagr = (final / init) ** (1 / years) - 1
    total = final / init - 1
    dd = (eq / eq.cummax() - 1) * 100
    mdd = float(dd.min())

    monthly = eq.resample('YE').last()
    y_ret = (monthly / monthly.shift(1) - 1).dropna() * 100
    first_yr = eq.index[0].year
    first_ret = (eq[eq.index.year == first_yr].iloc[-1] / init - 1) * 100
    y_ser = pd.concat([
        pd.Series([float(first_ret)], index=[pd.Timestamp(f'{first_yr}-12-31')]),
        y_ret,
    ])

    return {
        'label': label, 'color': color, 'init': init,
        'cagr': float(cagr * 100), 'total': float(total * 100), 'mdd': mdd, 'final': final,
        'dates': [d.strftime('%Y-%m-%d') for d in eq.index],
        'equity': [round(float(x), 2) if x == x else None for x in eq],
        'dd': [round(float(x), 2) if x == x else None for x in dd],
        'year_labels': [int(y.year) for y in y_ser.index],
        'year_ret': [round(float(x), 2) if x == x else None for x in y_ser],
    }


def build():
    datas = {}

    for csv, label, color in STRATEGIES:
        datas[csv] = build_series(csv, label, color)

    for code, label, color in US_INDEXES:
        df = fetch_us(code)
        close = df.set_index('time_key')['close']
        # 归一化: 初始资金 100,000 全仓买入持有(前复权含分红)
        eq = close / close.iloc[0] * INIT
        datas[code] = build_series(eq, label, color)

    payload = json.dumps(datas, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>巴菲特质量池 · 标普500 · 纳指100 · 收益总览</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#0f1420; --card:#151c2c; --line:#22304a; --txt:#dfe6f2; --muted:#8fa3c0; --acc:#4ea1ff; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--txt); font-family:-apple-system,'SF Pro Display','PingFang SC','Microsoft YaHei',sans-serif; padding:24px; }
  .wrap { max-width:1200px; margin:0 auto; }
  h1 { font-size:22px; font-weight:700; margin-bottom:4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
  .card .k { font-size:11px; color:var(--muted); margin-bottom:6px; }
  .card .v { font-size:20px; font-weight:700; }
  .card .d { font-size:11px; color:var(--muted); margin-top:4px; }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  .seg { display:flex; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; flex-wrap:wrap; }
  .seg button { background:transparent; color:var(--muted); border:none; padding:7px 14px; font-size:12px; cursor:pointer; white-space:nowrap; }
  .seg button.on { background:var(--acc); color:#fff; }
  .sw { display:flex; gap:16px; margin-left:auto; align-items:center; flex-wrap:wrap; }
  .sw label { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; cursor:pointer; }
  .sw select { background:var(--card); color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:5px 8px; font-size:12px; }
  .plot { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:6px; margin-bottom:20px; }
  .plot .t { font-size:13px; color:var(--muted); padding:8px 12px 0; }
  .footer { color:var(--muted); font-size:11px; line-height:1.7; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>巴菲特质量池 × MA10/30 择时 · 同期大盘对比</h1>
  <div class="sub">港股21只候选 · 季度滚动重建池（无前视·修复后版）vs 标普500/纳指100 · 2014-01 ~ 2026-06（12.4年）· 同初始 100,000 HKD · 含全部港股费用</div>

  <div class="cards" id="cards"></div>

  <div class="controls">
    <div class="seg" id="seg"></div>
    <div class="sw">
      <label><input type="checkbox" id="showCompare"> 叠加对比：
        <select id="cmpSel"></select>
      </label>
      <div class="seg" id="scaleSeg">
        <button data-s="log">对数</button>
        <button data-s="linear" class="on">线性</button>
      </div>
    </div>
  </div>

  <div class="plot"><div class="t">① 净值曲线（拖动/滚轮缩放 · hover 十字线）</div><div id="g1"></div></div>
  <div class="plot"><div class="t">② 回撤（水下面积，与上图 x 轴联动）</div><div id="g2"></div></div>
  <div class="plot"><div class="t">③ 分年度收益率（% / 年）</div><div id="g3"></div></div>

  <div class="footer">
    口径：季度滚动池每季用「当时已披露」年报打分取 Top6，池内 MA10/30 只多择时（无前视，已剔除记账 bug 与未上市/负价标的）。<br>
    标普500 SPY / 纳指100 QQQ 为同期 QFQ 前复权日线（已含分红再投资），按同初始资金 100,000 归一化持有；港股策略已扣佣金/印花税/交易费/滑点。<br>
    回撤 = 净值/历史峰值 − 1；CAGR 按 365.25 天/年复利。
  </div>
</div>

<script>
const DATAS = __DATA__;
const KEYS = Object.keys(DATAS);
let sel = 0, showCmp = false, cmpSel = 1, scale = 'linear';

const grid = {
  margin:{l:60,r:12,t:14,b:42}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(21,28,44,0.55)',
  font:{color:'#8fa3c0', size:11}, showlegend:true, legend:{orientation:'h', y:1.12, x:0}
};

function initUI(){
  const seg = document.getElementById('seg');
  seg.innerHTML = KEYS.map((k,i) => `<button data-v="${i}" class="${i===0?'on':''}">${DATAS[k].label}</button>`).join('');
  const cmp = document.getElementById('cmpSel');
  cmp.innerHTML = KEYS.map((k,i) => `<option value="${i}">${DATAS[k].label}</option>`).join('');
}
function renderCards(){
  const d = DATAS[KEYS[sel]];
  const cards = [
    ['期末净值', 'HK$ ' + Math.round(d.final).toLocaleString(),
      '×' + (d.final/d.init).toFixed(2) + '（初始 ' + d.init.toLocaleString() + '）'],
    ['总收益率', '+' + d.total.toFixed(1) + '%', '12.4 年累计'],
    ['年复合回报（CAGR）', '+' + d.cagr.toFixed(2) + '%', '复利年化'],
    ['最大回撤', d.mdd.toFixed(1) + '%', '峰值至谷底'],
  ];
  document.getElementById('cards').innerHTML = cards.map(([k,v,sub]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${sub}</div></div>`
  ).join('');
}
function getPair(){
  return { main: DATAS[KEYS[sel]], cmp: showCmp ? DATAS[KEYS[cmpSel]] : null };
}
function render(){
  const { main, cmp } = getPair();

  const traces = [];
  if (cmp) traces.push({
    x: cmp.dates, y: cmp.equity, name: cmp.label, type:'scatter', mode:'lines',
    line:{color:cmp.color, width:1.3, dash:'dot'}, opacity:0.85,
    hovertemplate:'%{y:,.0f} <extra>' + cmp.label + '</extra>'
  });
  traces.push({
    x: main.dates, y: main.equity, name: main.label, type:'scatter', mode:'lines',
    line:{color:main.color, width:2.2},
    fill: cmp ? 'none' : 'tozeroy', fillcolor: main.color + '14',
    hovertemplate:'%{y:,.0f} <extra>' + main.label + '</extra>'
  });
  Plotly.newPlot('g1', traces, {
    ...grid,
    xaxis:{type:'date', tickformat:'%Y-%m', gridcolor:'#22304a', zeroline:false, tickfont:{size:10}},
    yaxis:{type:scale, title:'净值 (HK$ 等值)', gridcolor:'#22304a', zeroline:false, tickfont:{size:10}},
    hovermode:'x unified'
  }, {responsive:true, displayModeBar:true});

  const ddTr = [];
  if (cmp) ddTr.push({
    x: cmp.dates, y: cmp.dd, name: cmp.label + ' 回撤', type:'scatter', mode:'lines',
    line:{color:cmp.color, width:1.2, dash:'dot'}, opacity:0.6,
    hovertemplate:'%{y:.1f}%<extra>' + cmp.label + '</extra>'
  });
  ddTr.push({
    x: main.dates, y: main.dd, name: main.label + ' 回撤', type:'scatter', mode:'lines',
    line:{color:'#e05d5d', width:1.6}, fill:'tozeroy', fillcolor:'rgba(224,93,93,0.20)',
    hovertemplate:'%{y:.1f}%<extra>' + main.label + '</extra>'
  });
  Plotly.newPlot('g2', ddTr, {
    ...grid,
    xaxis:{type:'date', tickformat:'%Y-%m', gridcolor:'#22304a', zeroline:false, tickfont:{size:10}},
    yaxis:{title:'回撤 (%)', range:[-105, 5], gridcolor:'#22304a', zeroline:true, zerolinecolor:'#3a4a6a', tickfont:{size:10}},
    hovermode:'x unified'
  }, {responsive:true, displayModeBar:false});

  const bar = [];
  if (cmp) bar.push({
    x: cmp.year_labels, y: cmp.year_ret, name: cmp.label, type:'bar', opacity:0.55,
    marker:{color: cmp.year_ret.map(v => v>=0 ? cmp.color : '#4a5a78')},
    hovertemplate:'%{y:.1f}%<extra>' + cmp.label + '</extra>'
  });
  bar.push({
    x: main.year_labels, y: main.year_ret, name: main.label, type:'bar',
    marker:{color: main.year_ret.map(v => v>=0 ? main.color : '#e05d5d')},
    hovertemplate:'%{y:.1f}%<extra>' + main.label + '</extra>'
  });
  Plotly.newPlot('g3', bar, {
    ...grid,
    xaxis:{type:'category', gridcolor:'#22304a', tickfont:{size:10}},
    yaxis:{title:'年度收益 (%)', gridcolor:'#22304a', zeroline:true, zerolinecolor:'#3a4a6a', tickfont:{size:10}},
    barmode:'group', hovermode:'x unified'
  }, {responsive:true, displayModeBar:false});

  document.getElementById('g1').on('plotly_relayout', evt => {
    const r0 = evt['xaxis.range[0]'], r1 = evt['xaxis.range[1]'];
    if (r0 !== undefined || r1 !== undefined) {
      Plotly.relayout('g2', { 'xaxis.range': [r0, r1], 'xaxis.autorange': false });
    }
  });
  renderCards();
}

document.getElementById('seg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  document.querySelectorAll('#seg button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); sel = Number(b.dataset.v); render();
});
document.getElementById('showCompare').addEventListener('change', e => { showCmp = e.target.checked; render(); });
document.getElementById('cmpSel').addEventListener('change', e => { cmpSel = Number(e.target.value); render(); });
document.getElementById('scaleSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  document.querySelectorAll('#scaleSeg button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); scale = b.dataset.s; render();
});

initUI();
render();
</script>
</body>
</html>"""

    html = html.replace('__DATA__', payload)
    out = os.path.join(BASE, 'investment_dashboard.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)

    print('已生成: %s' % out)
    for k, v in datas.items():
        print('  %-12s %-28s CAGR=%+6.2f%%  MDD=%6.1f%%  期末=%10.0f  (init %.0f)'
              % (k, v['label'], v['cagr'], v['mdd'], v['final'], v['init']))


if __name__ == '__main__':
    build()