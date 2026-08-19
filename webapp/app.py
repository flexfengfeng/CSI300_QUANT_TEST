# -*- coding: utf-8 -*-
"""风险平价 + 资产轮动 | CSI300 网页版仪表盘 (Streamlit)。

功能:
  - 今日仓位信号牌 (股票/国债/现金)
  - 回测净值曲线 + 关键指标 (CAGR/MDD/夏普/超额)
  - 历史仓位走势 + PE 分位
  - 参数鲁棒性扫描 (过拟合诊断)
  - 说明与免责声明

部署: Streamlit Community Cloud / GitHub
运行: streamlit run app.py
"""
from __future__ import annotations

import io

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

st.set_page_config(
    page_title="股债风险平价助手",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


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
def plot_equity(df: pd.DataFrame):
    """净值曲线 + 沪深300基准 (Plotly 双 Y 主图 + 下方权重/PE 子图)。"""
    if HAS_PLOTLY:
        x = df["date"]
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=("净值 (策略 vs 沪深300)", "资产权重", "PE 分位数"),
        )
        fig.add_trace(go.Scatter(x=x, y=np.exp(df["strategy_ret"].cumsum()),
                                 name="风险平价组合", line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.exp(df["market_ret"].cumsum()),
                                 name="沪深300 买入持有", line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["stock_weight"],
                                 name="股票", line=dict(color="#9467bd"), fill="tozeroy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bond_weight"],
                                 name="国债ETF", line=dict(color="#2ca02c"), fill="tozeroy"), row=2, col=1)
        fig.update_layout(height=680, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(l=40, r=20, t=60, b=30), hovermode="x unified")
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="权重", range=[0, 1.2], row=2, col=1)
        return fig
    # matplotlib 回退
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df["date"], np.exp(df["strategy_ret"].cumsum()), label="策略", color="#1f77b4")
    axes[0].plot(df["date"], np.exp(df["market_ret"].cumsum()), label="沪深300", color="#ff7f0e")
    axes[0].set_ylabel("净值"); axes[0].legend()
    axes[1].plot(df["date"], df["stock_weight"], label="股票", color="#9467bd")
    axes[1].plot(df["date"], df["bond_weight"], label="国债ETF", color="#2ca02c")
    axes[1].set_ylabel("权重"); axes[1].legend(); axes[1].set_ylim(0, 1.2)
    axes[2].set_ylabel("PE 分位")
    plt.tight_layout()
    return fig


def plot_robustness(robust_df: pd.DataFrame):
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
            colorbar=dict(title="CAGR %"),
        ))
        fig.update_layout(
            title="参数鲁棒性 | CAGR% (目标波动 × 波动窗口)",
            xaxis_title="波动率窗口 (交易日)", yaxis_title="目标波动率 (%)",
            height=420, margin=dict(l=60, r=20, t=60, b=30),
        )
        return fig
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(labels.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels.columns)), [str(c) for c in labels.columns])
    ax.set_yticks(range(len(labels.index)), [str(r) for r in labels.index])
    ax.set_xlabel("波动率窗口 (交易日)"); ax.set_ylabel("目标波动率 (%)")
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            ax.text(j, i, f"{labels.iloc[i, j]:.2f}", ha="center", va="center")
    plt.colorbar(im, label="CAGR %")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def metric_card(st, label, value, delta=None, color: str | None = None,
                suffix: str = ""):
    """渲染单指标卡片。"""
    st.markdown(
        f"""<div style="background:{color or '#f7f8fa'};padding:16px;border-radius:12px;
            border:1px solid #e3e6ed;margin-bottom:8px">
            <div style="font-size:13px;color:#6b7280">{label}</div>
            <div style="font-size:26px;font-weight:700;color:#111827;margin-top:4px">
                {value}{suffix}</div>
            {f'<div style="font-size:13px;color:#16a34a;margin-top:2px">{delta}</div>' if delta else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def main():
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

    st.title("📊 风险平价 + 资产轮动 | 沪深300 网页版")
    st.caption("波动率目标 × PE 估值锚定 × 国债轮动 —— 不再预测涨跌，只做确定性配置")

    with st.sidebar:
        st.header("⚙️ 参数")
        refresh = st.checkbox("尝试刷新最新行情 (联网)", value=False,
                              help="开启后从腾讯财经接口补齐最近交易日数据")
        st.divider()
        st.subheader("策略参数")
        target = st.slider("目标年化波动率 (%)", 10, 25, 20, 1,
                           help="PE 低估时上调, 高估时下调 (调节幅度见下)")
        pe_scale = st.slider("PE 调节幅度 (%)", 1, 8, 5, 1)
        lkbk = st.slider("波动率滚动窗口 (日)", 20, 60, 40, 5)
        st.divider()
        st.subheader("仓位约束")
        lower = st.slider("股票权重下限", 0.30, 0.60, 0.40, 0.05)
        upper = st.slider("股票权重上限", 0.80, 1.20, 1.00, 0.05)
        bup = st.slider("债券权重上限", 0.30, 0.70, 0.60, 0.05)
        hold = st.slider("滞回门槛", 0.03, 0.20, 0.10, 0.01)
        st.divider()
        st.caption("数据截至: 以底库为准, `尝试刷新` 可选联网补齐")
        st.markdown("[部署指南 → README](README.md)")

    # ---- 加载数据 ----
    with st.spinner("加载数据..."):
        stock, bond, pe = load_data(refresh)
    df, m = run_backtest(stock, bond, pe, target, pe_scale, lkbk, lower, upper, bup, hold, 60)
    sig = stg.latest_signal(df)

    # ---- 今日信号牌 ----
    st.subheader("🎯 今日配置信号")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(st, "股票仓位", f"{sig['position_pct']:.0f}%",
                    color="#eef2ff", suffix="")
    with col2:
        metric_card(st, "国债ETF仓位", f"{sig['bond_weight']*100:.0f}%", color="#ecfdf5")
    with col3:
        metric_card(st, "现金", f"{sig['cash_weight']*100:.0f}%", color="#f9fafb")
    with col4:
        metric_card(st, "信号日期", sig["date"].strftime("%Y-%m-%d"), color="#fefce8")

    st.markdown("---")

    # ---- 回测表现 ----
    st.subheader("📈 回测表现")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", f"{m['s_cagr']*100:.2f}%",
              f"沪深300 {m['m_cagr']*100:.2f}%")
    c2.metric("最大回撤", f"{m['s_mdd']*100:.1f}%",
              f"沪深300 {m['m_mdd']*100:.1f}%")
    c3.metric("夏普", f"{m['s_sharpe']:.2f}",
              f"沪深300 {m['m_sharpe']:.2f}")
    c4.metric("年化波动", f"{m['s_ann_vol']*100:.1f}%",
              f"沪深300 {m['m_ann_vol']*100:.1f}%")
    c5.metric("超额收益", f"{m['excess']*100:+.1f}pp", "vs 沪深300")
    c6.metric("平均仓位", f"股票{m['avg_stock']*100:.0f}% / 国债{m['avg_bond']*100:.0f}%")

    fig = plot_equity(df)
    if HAS_PLOTLY:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)

    with st.expander("📋 回测明细 (最近 10 个交易日)"):
        detail = df.tail(10).copy()
        detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
        detail["stock_weight"] = (detail["stock_weight"] * 100).round(1)
        detail["bond_weight"] = (detail["bond_weight"] * 100).round(1)
        detail["strategy_ret"] = (detail["strategy_ret"] * 100).round(3)
        detail["market_ret"] = (detail["market_ret"] * 100).round(3)
        detail.columns = ["日期", "股票%", "国债%", "策略日收益%", "沪深300日收益%"]
        st.dataframe(detail, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        export = df.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(buf, index=False)
        st.download_button("⬇️ 下载完整回测 CSV", buf.getvalue(),
                           file_name="riskparity_csi300_backtest.csv", mime="text/csv")

    st.markdown("---")

    # ---- 参数鲁棒性 ----
    st.subheader("🧪 参数鲁棒性检查 (过拟合诊断)")
    st.caption("固定其余参数, 扫描 目标波动率(18~22%) × 波动窗口(30/40/50天) 的 CAGR。"
               "若整表稳定在 8%±1 且均跑赢沪深300(≈8.2%), 说明策略为『高原型』而非『尖峰型』。")
    with st.spinner("扫描中..."):
        rob = run_robustness(stock, bond, pe)
    fig2 = plot_robustness(rob)
    if HAS_PLOTLY:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.pyplot(fig2)
    st.dataframe((rob * 100).round(2).rename(columns=lambda c: f"{c}日窗口"),
                 use_container_width=True)

    st.markdown("---")

    # ---- 说明 ----
    with st.expander("📖 策略原理与风险提示"):
        st.markdown("""
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
""")

    st.markdown("---")
    st.caption(
        "数据来源: 腾讯财经行情接口 + 内置底库 (沪深300 PE_TTM / 日线)。"
        "扫码或打开下方链接访问部署版 → [Streamlit Community Cloud](https://share.streamlit.io)"
    )


if __name__ == "__main__":
    main()