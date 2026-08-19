# -*- coding: utf-8 -*-
"""风险平价 + 资产轮动 | CSI300 网页版仪表盘 (Streamlit)。

功能:
  - 今日仓位信号牌 (股票/国债/现金)
  - 回测净值曲线 + 关键指标 (CAGR/MDD/夏普/超额)
  - 历史仓位走势 + PE 分位
  - 参数鲁棒性扫描 (过拟合诊断)
  - 说明与免责声明

国际化:
  - 自动检测浏览器语言 (navigator.language)
  - 中文浏览器 → 中文界面; 其余语言 → 英文界面
  - 检测结果写入 URL 查询参数 `?lang=zh|en`, 首次访问自动重载一次

部署: Streamlit Community Cloud / GitHub
运行: streamlit run app.py
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import strategy as stg


# ---------------------------------------------------------------------------
# 国际化 (i18n): 英文为默认, 中文浏览器自动切换为中文
# ---------------------------------------------------------------------------
LANG_ZH = "zh"
LANG_EN = "en"

# 浏览器语言探测脚本: 读取 navigator.language, 把 ?lang= 写入 URL 后重载一次。
# 之后每次运行都能从查询参数直接拿到语言, 不再重复重定向。
# 只有当 URL 中没有 lang 参数时才会触发重定向, 因此不会造成循环。
_LANG_DETECTION_JS = """
<script>
(function() {
  if (window.location.search.indexOf('lang=') >= 0) return;
  var lang = (navigator.language || navigator.userLanguage || 'en').toLowerCase();
  var code = lang.indexOf('zh') === 0 ? 'zh' : 'en';
  if (code === 'zh') { document.title = '股债风险平价助手'; }
  else { document.title = 'Risk Parity Assistant'; }
  var url = new URL(window.location.href);
  url.searchParams.set('lang', code);
  window.location.replace(url.toString());
})();
</script>
"""

def _embed_html(html: str) -> None:
    """注入零尺寸 HTML/JS 片段。

    优先使用新版 st.html (Streamlit >= 1.42, 需允许 JS),
    旧版本自动回退到 streamlit.components.v1.html。
    """
    try:
        st.html(html, unsafe_allow_javascript=True)
    except (AttributeError, TypeError):
        import streamlit.components.v1 as components
        components.html(html, height=0, width=0)


# 各语言下的浏览器标签页标题 (set_page_config 使用)
_PAGE_TITLES = {
    LANG_ZH: "股债风险平价助手",
    LANG_EN: "Risk Parity Assistant",
}


def _query_lang() -> Optional[str]:
    """只读查询参数中的语言 (在 set_page_config 之前可安全调用)。"""
    raw = st.query_params.get("lang")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return raw if raw in (LANG_ZH, LANG_EN) else None


def detect_language() -> str:
    """检测浏览器语言 → 'zh' (中文) 或 'en' (默认英文)。

    优先级: URL 查询参数 (右上角手动切换 / JS 探测结果) > 会话内已确定 > 探测脚本。
    首次无参数访问时注入 JS 读取 navigator.language 并写入 `?lang=` 重载。
    """
    qlang = _query_lang()
    if qlang is not None:
        st.session_state["lang"] = qlang
        return qlang
    if "lang" in st.session_state:
        return st.session_state["lang"]
    # 首次访问: 注入零尺寸 JS 组件探测浏览器语言并重定向 (无 JS 环境回退英文)
    _embed_html(_LANG_DETECTION_JS)
    st.session_state["lang"] = LANG_EN
    return LANG_EN


# 在页面配置前读取一次查询参数, 使浏览器标签页标题也能跟随语言
_initial_lang = _query_lang() or LANG_EN
st.set_page_config(
    page_title=_PAGE_TITLES[_initial_lang],
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# 文案资源 (中/英)
# ---------------------------------------------------------------------------
T = {
    LANG_ZH: {
        "app_title": "📊 风险平价 + 资产轮动 | 沪深300 网页版",
        "app_caption": "波动率目标 × PE 估值锚定 × 国债轮动 —— 不再预测涨跌，只做确定性配置",

        # 侧边栏
        "params_header": "⚙️ 参数",
        "refresh_label": "尝试刷新最新行情 (联网)",
        "refresh_help": "开启后从腾讯财经接口补齐最近交易日数据",
        "strategy_params": "策略参数",
        "target_vol_label": "目标年化波动率 (%)",
        "target_vol_help": "PE 低估时上调, 高估时下调 (调节幅度见下)",
        "pe_scale_label": "PE 调节幅度 (%)",
        "lookback_label": "波动率滚动窗口 (日)",
        "pos_constraints": "仓位约束",
        "stock_lower_label": "股票权重下限",
        "stock_upper_label": "股票权重上限",
        "bond_upper_label": "债券权重上限",
        "hold_label": "滞回门槛",
        "data_caption": "数据截至: 以底库为准, `尝试刷新` 可选联网补齐",
        "readme_link": "部署指南 → README",

        # 数据与信号
        "loading_data": "加载数据...",
        "signal_title": "🎯 今日配置信号",
        "stock_position": "股票仓位",
        "bond_position": "国债ETF仓位",
        "cash_position": "现金",
        "signal_date": "信号日期",

        # 回测表现
        "backtest_title": "📈 回测表现",
        "cagr": "CAGR",
        "mdd": "最大回撤",
        "sharpe": "夏普",
        "ann_vol": "年化波动",
        "excess": "超额收益",
        "avg_position": "平均仓位",
        "benchmark": "沪深300",
        "vs_benchmark": "vs 沪深300",
        "avg_stock_label": "股票",
        "avg_bond_label": "国债",

        # 图表
        "equity_subtitle": "净值 (策略 vs 沪深300)",
        "weights_subtitle": "资产权重",
        "pe_subtitle": "PE 分位数",
        "portfolio_name": "风险平价组合",
        "buy_hold_name": "沪深300 买入持有",
        "stock_name": "股票",
        "bond_name": "国债ETF",
        "strategy_mpl": "策略",
        "market_mpl": "沪深300",
        "net_value": "净值",
        "weights": "权重",
        "pe_pctile": "PE 分位",
        "cagr_colorbar": "CAGR %",

        # 回测明细
        "details_expander": "📋 回测明细 (最近 10 个交易日)",
        "detail_cols": ["日期", "股票%", "国债%", "策略日收益%", "沪深300日收益%"],
        "download_btn": "⬇️ 下载完整回测 CSV",

        # 参数鲁棒性
        "robust_title": "🧪 参数鲁棒性检查 (过拟合诊断)",
        "robust_caption": "固定其余参数, 扫描 目标波动率(18~22%) × 波动窗口(30/40/50天) 的 CAGR。"
                          "若整表稳定在 8%±1 且均跑赢沪深300(≈8.2%), 说明策略为『高原型』而非『尖峰型』。",
        "scanning": "扫描中...",
        "robust_plot_title": "参数鲁棒性 | CAGR% (目标波动 × 波动窗口)",
        "robust_x_axis": "波动率窗口 (交易日)",
        "robust_y_axis": "目标波动率 (%)",
        "window_col_fmt": "{}日窗口",

        # 说明与页脚
        "strategy_logic_expander": "📖 策略原理与风险提示",
        "strategy_logic_md": """
**三大支柱 (无任何方向预测):**

1. **波动率目标 (风险预算)**: `股票仓位 = 目标波动率 ÷ 过去40日实际波动率`
   - 市场平静 → 波动低 → 自动加仓吃慢牛
   - 市场恐慌 → 波动飙升 → 自动减仓避险

2. **PE 估值锚定 (风险偏好)**: 目标波动率随沪深300 PE 10年分位动态调节
   - PE 分位 < 30% (低估) → 目标波动率上调 → 敢承担风险
   - PE 分位 > 70% (高估) → 目标波动率下调 → 强制保守

3. **国债轮动 (现金替代)**: 股票仓位不足 100% 时, 剩余资金买十年国债ETF(511010)

**历史回测 (2013.11 ~ 今, 默认参数):** CAGR 8.8%, 最大回撤 -38.6%, 夏普 0.50, 跑赢沪深300 约 +7.9pp。

**压力测试:**
- 参数邻域 25 组 100% 跑赢基准 → 高原型, 抗过拟合
- 国债缺位改持货币基金 → CAGR 仍 8.4~8.5%

---
⚠️ **风险提示**: 本工具仅用于研究学习, 不构成任何投资建议。历史业绩不代表未来表现。
股市有风险, 投资需谨慎。请使用闲置资金, 控制杠杆。
""",
        "footer": "数据来源: 腾讯财经行情接口 + 内置底库 (沪深300 PE_TTM / 日线)。"
                  "扫码或打开下方链接访问部署版 → [Streamlit Community Cloud](https://share.streamlit.io)",
    },

    LANG_EN: {
        "app_title": "📊 Risk Parity + Asset Rotation | CSI 300",
        "app_caption": "Volatility targeting × PE valuation anchor × bond rotation — "
                       "no direction prediction, only disciplined allocation",

        # Sidebar
        "params_header": "⚙️ Parameters",
        "refresh_label": "Try refreshing latest quotes (online)",
        "refresh_help": "Fetch missing recent trading days from the Tencent Finance API when enabled",
        "strategy_params": "Strategy Parameters",
        "target_vol_label": "Target Annual Volatility (%)",
        "target_vol_help": "Raised when PE is cheap, lowered when expensive (adjustment below)",
        "pe_scale_label": "PE Adjustment (%)",
        "lookback_label": "Volatility Lookback Window (days)",
        "pos_constraints": "Position Constraints",
        "stock_lower_label": "Stock Weight Lower Bound",
        "stock_upper_label": "Stock Weight Upper Bound",
        "bond_upper_label": "Bond Weight Upper Bound",
        "hold_label": "Hysteresis Threshold",
        "data_caption": "Data as of: local dataset; 'Try refreshing' optionally fetches online",
        "readme_link": "Deployment Guide → README",

        # Data & signal
        "loading_data": "Loading data...",
        "signal_title": "🎯 Today's Allocation Signal",
        "stock_position": "Stock Position",
        "bond_position": "Treasury Bond ETF",
        "cash_position": "Cash",
        "signal_date": "Signal Date",

        # Backtest performance
        "backtest_title": "📈 Backtest Performance",
        "cagr": "CAGR",
        "mdd": "Max Drawdown",
        "sharpe": "Sharpe",
        "ann_vol": "Annual Volatility",
        "excess": "Excess Return",
        "avg_position": "Avg Position",
        "benchmark": "CSI 300",
        "vs_benchmark": "vs CSI 300",
        "avg_stock_label": "Equity ",
        "avg_bond_label": "Bond ",

        # Charts
        "equity_subtitle": "Net Value (Strategy vs CSI 300)",
        "weights_subtitle": "Asset Weights",
        "pe_subtitle": "PE Percentile",
        "portfolio_name": "Risk Parity Portfolio",
        "buy_hold_name": "CSI 300 Buy & Hold",
        "stock_name": "Equity",
        "bond_name": "Treasury ETF",
        "strategy_mpl": "Strategy",
        "market_mpl": "CSI 300",
        "net_value": "Net Value",
        "weights": "Weights",
        "pe_pctile": "PE Pctile",
        "cagr_colorbar": "CAGR %",

        # Backtest details
        "details_expander": "📋 Backtest Details (last 10 trading days)",
        "detail_cols": ["Date", "Equity %", "Bond %", "Strategy Daily Ret %", "CSI 300 Daily Ret %"],
        "download_btn": "⬇️ Download Full Backtest CSV",

        # Parameter robustness
        "robust_title": "🧪 Parameter Robustness Check (Overfitting Diagnosis)",
        "robust_caption": "Holding other parameters fixed, scan target volatility (18–22%) × "
                          "lookback window (30/40/50 days) and compare CAGR. If the whole table "
                          "stays within 8%±1 and consistently beats CSI 300 (≈8.2%), the strategy "
                          "is a 'plateau' rather than a 'spike'.",
        "scanning": "Scanning...",
        "robust_plot_title": "Parameter Robustness | CAGR% (Target Volatility × Lookback Window)",
        "robust_x_axis": "Volatility Window (trading days)",
        "robust_y_axis": "Target Volatility (%)",
        "window_col_fmt": "{}-day window",

        # Strategy logic & footer
        "strategy_logic_expander": "📖 Strategy Logic & Risk Disclosure",
        "strategy_logic_md": """
**Three pillars (no directional prediction):**

1. **Volatility Targeting (Risk Budget)**: `Stock position = target volatility ÷ realized volatility over the past 40 days`
   - Calm market → low volatility → position scales up to ride a steady bull
   - Panic market → volatility spikes → position scales down for safety

2. **PE Valuation Anchor (Risk Appetite)**: target volatility adjusts dynamically with the CSI 300's 10-year PE percentile
   - PE percentile < 30% (cheap) → raise target volatility → willing to take risk
   - PE percentile > 70% (expensive) → lower target volatility → forced to be conservative

3. **Bond Rotation (Cash Replacement)**: when the stock position is under 100%, remaining capital buys the 10-year Treasury ETF (511010)

**Historical backtest (2013.11 – present, default params):** CAGR 8.8%, max drawdown -38.6%, Sharpe 0.50, beating CSI 300 by about +7.9pp.

**Stress tests:**
- All 25 parameter-neighborhood combinations beat the benchmark → plateau type, robust to overfitting
- Replacing bonds with a money-market fund → CAGR still 8.4–8.5%

---
⚠️ **Risk disclosure**: This tool is for research and education only and does not constitute investment advice.
Past performance does not guarantee future results. Investing involves risk; invest prudently.
Use idle funds only and control leverage.
""",
        "footer": "Data sources: Tencent Finance quotes API + built-in database (CSI 300 PE_TTM / daily). "
                  "Scan the QR code or open the link below for the deployed version → [Streamlit Community Cloud](https://share.streamlit.io)",
    },
}


# ---------------------------------------------------------------------------
# 数据加载 (带缓存)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return stg.load_market_data(refresh=refresh)


@st.cache_data(show_spinner=False)
def run_backtest(stock, bond, pe, target, pe_scale, lkbk, lower, upper, bup, hold, warmup):
    p = stg.RiskParityParams(
        target_vol=target / 100.0,
        pe_scale=pe_scale / 100.0,
        vol_lookback=int(lkbk),
        stock_lower=lower,
        stock_upper=upper,
        bond_upper=bup,
        hold_threshold=hold,
        warmup=int(warmup),
    )
    df = stg.backtest(stock, bond, pe, p)
    m = stg.compute_metrics(df)
    return df, m


@st.cache_data(show_spinner=False)
def run_robustness(stock, bond, pe):
    return stg.robustness_check(stock, bond, pe)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_equity(df: pd.DataFrame, t: dict):
    """净值曲线 + 沪深300基准 (Plotly 双 Y 主图 + 下方权重/PE 子图)。"""
    if HAS_PLOTLY:
        x = df["date"]
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=(t["equity_subtitle"], t["weights_subtitle"], t["pe_subtitle"]),
        )
        fig.add_trace(go.Scatter(x=x, y=np.exp(df["strategy_ret"].cumsum()),
                                 name=t["portfolio_name"], line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.exp(df["market_ret"].cumsum()),
                                 name=t["buy_hold_name"], line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["stock_weight"],
                                 name=t["stock_name"], line=dict(color="#9467bd"), fill="tozeroy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bond_weight"],
                                 name=t["bond_name"], line=dict(color="#2ca02c"), fill="tozeroy"), row=2, col=1)
        fig.update_layout(height=680, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(l=40, r=20, t=60, b=30), hovermode="x unified")
        fig.update_yaxes(title_text=t["net_value"], row=1, col=1)
        fig.update_yaxes(title_text=t["weights"], range=[0, 1.2], row=2, col=1)
        return fig
    # matplotlib 回退
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df["date"], np.exp(df["strategy_ret"].cumsum()),
                 label=t["strategy_mpl"], color="#1f77b4")
    axes[0].plot(df["date"], np.exp(df["market_ret"].cumsum()),
                 label=t["market_mpl"], color="#ff7f0e")
    axes[0].set_ylabel(t["net_value"]); axes[0].legend()
    axes[1].plot(df["date"], df["stock_weight"], label=t["stock_name"], color="#9467bd")
    axes[1].plot(df["date"], df["bond_weight"], label=t["bond_name"], color="#2ca02c")
    axes[1].set_ylabel(t["weights"]); axes[1].legend(); axes[1].set_ylim(0, 1.2)
    axes[2].set_ylabel(t["pe_pctile"])
    plt.tight_layout()
    return fig


def plot_robustness(robust_df: pd.DataFrame, t: dict):
    """参数鲁棒性热力图 (CAGR%)。"""
    labels = robust_df.copy()
    labels = (labels * 100).round(2)
    if HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=labels.to_numpy(),
            x=[str(c) for c in labels.columns],
            y=[str(r) for r in labels.index],
            text=labels.to_numpy(),
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=8.0,
            colorbar=dict(title=t["cagr_colorbar"]),
        ))
        fig.update_layout(
            title=t["robust_plot_title"],
            xaxis_title=t["robust_x_axis"], yaxis_title=t["robust_y_axis"],
            height=420, margin=dict(l=60, r=20, t=60, b=30),
        )
        return fig
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(labels.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels.columns)), [str(c) for c in labels.columns])
    ax.set_yticks(range(len(labels.index)), [str(r) for r in labels.index])
    ax.set_xlabel(t["robust_x_axis"]); ax.set_ylabel(t["robust_y_axis"])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            ax.text(j, i, f"{labels.iloc[i, j]:.2f}", ha="center", va="center")
    plt.colorbar(im, label=t["cagr_colorbar"])
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _lang_href(target: str) -> str:
    """构造带 ?lang= 的链接, 保留除 lang 之外的所有查询参数。"""
    others = []
    for k, v in st.query_params.items():
        if str(k) == "lang" or str(k).startswith("$"):
            continue
        values = v if isinstance(v, list) else [v]
        for item in values:
            others.append(f"{k}={item}")
    suffix = ("&" + "&".join(others)) if others else ""
    return f"?lang={target}{suffix}"


def _render_lang_switcher(lang: str) -> None:
    """在页面右上角固定显示 中文 / EN 切换链接 (当前语言高亮)。"""
    zh_cls = "active" if lang == LANG_ZH else ""
    en_cls = "active" if lang == LANG_EN else ""
    html = f"""
    <style>
    .lang-switcher {{
      position: fixed;
      top: 14px;
      right: 18px;
      z-index: 99999;
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(255,255,255,0.92);
      border: 1px solid #e3e6ed;
      border-radius: 999px;
      padding: 4px 6px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.10);
      font-size: 13px;
      line-height: 1;
    }}
    .lang-switcher a {{
      text-decoration: none;
      color: #6b7280;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 600;
    }}
    .lang-switcher a.active {{
      background: #0f172a;
      color: #ffffff;
    }}
    @media (max-width: 767px) {{
      .lang-switcher {{
        top: 10px;
        right: 10px;
        font-size: 12px;
      }}
      .lang-switcher a {{
        padding: 3px 8px;
      }}
    }}
    </style>
    <div class="lang-switcher">
      <a href="{_lang_href(LANG_ZH)}" class="{zh_cls}">中文</a>
      <a href="{_lang_href(LANG_EN)}" class="{en_cls}">EN</a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_card(st, label, value, delta=None, color: str | None = None,
                suffix: str = ""):
    """渲染单指标卡片。"""
    delta_html = (
        f'<div style="font-size:13px;color:#16a34a;margin-top:2px">{delta}</div>'
        if delta else ""
    )
    # 注意: HTML 必须拼接为单行, 不能包含换行。
    # 若开始标签被换行拆开, markdown 引擎无法识别整段原始 HTML,
    # 结尾的 </div> 会被当作纯文本显示出来。
    html = (
        f'<div style="background:{color or "#f7f8fa"};padding:16px;'
        f'border-radius:12px;border:1px solid #e3e6ed;margin-bottom:8px">'
        f'<div style="font-size:13px;color:#6b7280">{label}</div>'
        f'<div style="font-size:26px;font-weight:700;color:#111827;margin-top:4px">'
        f'{value}{suffix}</div>'
        f'{delta_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def main():
    # ---- 语言检测: 中文浏览器 → zh, 其余 → en ----
    lang = detect_language()
    t = T[lang]

    # ---- 移动端响应式: 窄屏(<768px)时所有多列布局堆叠为单列 ----
    st.markdown(
        """
        <style>
        @media (max-width: 767px) {
          /* 让 st.columns 的内部 flex 容器纵向排列, 每列占满整行 */
          div[data-testid="column"] {
            width: 100% !important;
            flex: 0 0 100% !important;
            max-width: 100% !important;
          }
          div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
          }
          /* 指标卡片字号微调, 手机更易读 */
          div[data-testid="stMetricValue"] {
            font-size: 1.3rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- 右上角语言切换 (中文 / EN) ----
    _render_lang_switcher(lang)

    st.title(t["app_title"])
    st.caption(t["app_caption"])

    with st.sidebar:
        st.header(t["params_header"])
        refresh = st.checkbox(t["refresh_label"], value=False, help=t["refresh_help"])
        st.divider()
        st.subheader(t["strategy_params"])
        target = st.slider(t["target_vol_label"], 10, 25, 20, 1, help=t["target_vol_help"])
        pe_scale = st.slider(t["pe_scale_label"], 1, 8, 5, 1)
        lkbk = st.slider(t["lookback_label"], 20, 60, 40, 5)
        st.divider()
        st.subheader(t["pos_constraints"])
        lower = st.slider(t["stock_lower_label"], 0.30, 0.60, 0.40, 0.05)
        upper = st.slider(t["stock_upper_label"], 0.80, 1.20, 1.00, 0.05)
        bup = st.slider(t["bond_upper_label"], 0.30, 0.70, 0.60, 0.05)
        hold = st.slider(t["hold_label"], 0.03, 0.20, 0.10, 0.01)
        st.divider()
        st.caption(t["data_caption"])
        st.markdown(f"[{t['readme_link']}](README.md)")

    # ---- 加载数据 ----
    with st.spinner(t["loading_data"]):
        stock, bond, pe = load_data(refresh)
    df, m = run_backtest(stock, bond, pe, target, pe_scale, lkbk, lower, upper, bup, hold, 60)
    sig = stg.latest_signal(df)

    # ---- 今日信号牌 ----
    st.subheader(t["signal_title"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(st, t["stock_position"], f"{sig['position_pct']:.0f}%",
                    color="#eef2ff", suffix="")
    with col2:
        metric_card(st, t["bond_position"], f"{sig['bond_weight']*100:.0f}%", color="#ecfdf5")
    with col3:
        metric_card(st, t["cash_position"], f"{sig['cash_weight']*100:.0f}%", color="#f9fafb")
    with col4:
        metric_card(st, t["signal_date"], sig["date"].strftime("%Y-%m-%d"), color="#fefce8")

    st.markdown("---")

    # ---- 回测表现 (2 行 x 3 列, 保证列宽足够, 平均仓位不被截断) ----
    st.subheader(t["backtest_title"])
    r1a, r1b, r1c = st.columns(3)
    r1a.metric(t["cagr"], f"{m['s_cagr']*100:.2f}%",
               f"{t['benchmark']} {m['m_cagr']*100:.2f}%")
    r1b.metric(t["mdd"], f"{m['s_mdd']*100:.1f}%",
               f"{t['benchmark']} {m['m_mdd']*100:.1f}%")
    r1c.metric(t["sharpe"], f"{m['s_sharpe']:.2f}",
               f"{t['benchmark']} {m['m_sharpe']:.2f}")
    r2a, r2b, r2c = st.columns(3)
    r2a.metric(t["ann_vol"], f"{m['s_ann_vol']*100:.1f}%",
               f"{t['benchmark']} {m['m_ann_vol']*100:.1f}%")
    r2b.metric(t["excess"], f"{m['excess']*100:+.1f}pp", t["vs_benchmark"])
    # st.metric 单行会在列宽不足时截断 (如 "股票87% / 国债xx%"),
    # 故平均仓位改用自定义卡片分两行展示, 保证股票/国债比例完整可见。
    with r2c:
        metric_card(st, t["avg_position"],
                    f"{t['avg_stock_label']}{m['avg_stock']*100:.0f}%<br>"
                    f"{t['avg_bond_label']}{m['avg_bond']*100:.0f}%",
                    color="#f7f8fa")

    fig = plot_equity(df, t)
    if HAS_PLOTLY:
        st.plotly_chart(fig, width="stretch")
    else:
        st.pyplot(fig)

    with st.expander(t["details_expander"]):
        detail = df.tail(10).copy()
        detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
        detail["stock_weight"] = (detail["stock_weight"] * 100).round(1)
        detail["bond_weight"] = (detail["bond_weight"] * 100).round(1)
        detail["strategy_ret"] = (detail["strategy_ret"] * 100).round(3)
        detail["market_ret"] = (detail["market_ret"] * 100).round(3)
        detail.columns = t["detail_cols"]
        st.dataframe(detail, width="stretch", hide_index=True)

        buf = io.BytesIO()
        export = df.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(buf, index=False)
        st.download_button(t["download_btn"], buf.getvalue(),
                           file_name="riskparity_csi300_backtest.csv", mime="text/csv")

    st.markdown("---")

    # ---- 参数鲁棒性 ----
    st.subheader(t["robust_title"])
    st.caption(t["robust_caption"])
    with st.spinner(t["scanning"]):
        rob = run_robustness(stock, bond, pe)
    fig2 = plot_robustness(rob, t)
    if HAS_PLOTLY:
        st.plotly_chart(fig2, width="stretch")
    else:
        st.pyplot(fig2)
    st.dataframe((rob * 100).round(2).rename(columns=lambda c: t["window_col_fmt"].format(c)),
                 width="stretch")

    st.markdown("---")

    # ---- 说明 ----
    with st.expander(t["strategy_logic_expander"]):
        st.markdown(t["strategy_logic_md"])

    st.markdown("---")
    st.caption(t["footer"])


if __name__ == "__main__":
    main()