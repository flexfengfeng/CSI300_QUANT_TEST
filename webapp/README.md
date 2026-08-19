# 📊 风险平价 + 资产轮动 | CSI300 网页版

「风险平价 + 资产轮动」策略的网页版仪表盘，通过 **Streamlit Community Cloud** 部署，给朋友和自己使用。

> 不再预测涨跌（彻底放弃 HMM/Transformer 方向择时），只做确定性配置：
> **波动率目标 × PE 估值锚定 × 国债轮动**。

---

## ✨ 功能

- 🎯 **今日配置信号**：股票 / 国债ETF / 现金 三栏仓位牌
- 📈 **回测表现**：净值曲线 (策略 vs 沪深300) + CAGR / 最大回撤 / 夏普 / 超额收益
- 🧪 **参数鲁棒性检查**：目标波动率 × 波动窗口 的 CAGR 热力图（过拟合诊断）
- ⚙️ **实时调参**：侧边栏滑块调整目标波动率、PE调节、仓位约束，即时刷新
- 📋 **明细导出**：下载完整回测 CSV
- 🌐 **可选联网刷新**：从腾讯财经接口补齐最近交易日数据

## 🧠 策略核心（三大支柱，纯确定性公式）

```python
# 1. 实际波动率 (40日滚动, 年化)
realized_vol = closes.pct_change().rolling(40).std() * sqrt(252)

# 2. 动态目标波动率 (随 PE 10年分位调节: 低估→高目标, 高估→低目标)
dynamic_target = 基准(20%) ± PE分位 × 调节幅度(5%)

# 3. 股票仓位 (风险平价核心) + 国债仓位 (现金替代, 511010)
stock_weight = clip(dynamic_target / realized_vol, 0.4, 1.0)
bond_weight  = clip(1 - stock_weight, 0.0, 0.6)

# 4. 组合收益
returns = stock_weight * 股票收益 + bond_weight * 国债收益
```

**历史回测 (2013.11 ~ 今, 默认参数):**

| 指标 | 风险平价组合 | 沪深300 买入持有 |
|---|---|---|
| CAGR | **8.82%** | 8.16% |
| 最大回撤 | **-38.6%** | -43.0% |
| 夏普 | **0.50** | 0.37 |
| 年化波动 | **16.8%** | 21.3% |
| 超额收益 | **+7.9pp** | — |

**压力测试结论：**
- 参数邻域 25 组（目标波动18~22% × 窗口30~50天）**100% 跑赢基准** → 高原型，抗过拟合
- 国债缺位改持货币基金 → CAGR 仍 8.4~8.5%（仅降 0.3~0.5pp）

---

## 🚀 本地运行

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

## ☁️ 部署到 Streamlit Community Cloud

### 前提
1. 一个 GitHub 账号（免费）
2. 把本 `webapp/` 目录推送到一个 GitHub 仓库（公开或私有均可）

### 步骤

1. **推送到 GitHub**
   ```bash
   # 在本项目根目录 (moomoo) 执行, 假设已 git init
   git add webapp/
   git commit -m "feat: 风险平价 CSI300 网页版"
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   git push -u origin main
   ```

2. **登录 Streamlit Community Cloud**
   打开 https://share.streamlit.io ，用 GitHub 账号登录（点击 "Sign in with GitHub"）。

3. **创建新应用**
   点击左上角 **"New app"** →
   - **Repository**：选择你刚推送的仓库
   - **Branch**：`main`
   - **Main file path**：`webapp/app.py`
   - 点击 **"Deploy"**

4. **完成** 🎉
   约 1~2 分钟后，你会获得一个公开链接：
   `https://<你的用户名>-<仓库名>-<随机串>.streamlit.app`
   把它发给朋友即可。

### 后续更新
每次 `git push` 到 main 分支，Streamlit 会自动重新部署（约 1 分钟）。

---

## 📁 目录结构

```
webapp/
├── app.py               # Streamlit 页面入口
├── strategy.py          # 策略核心 (自包含, 与回测脚本同逻辑)
├── requirements.txt     # Python 依赖
├── README.md            # 本文件
└── data/                # 内置数据底库
    ├── csi300_pe_ttm.csv       # 沪深300 PE_TTM (2005~今)
    ├── daily_CN_000300.csv     # 沪深300 日线 (2006~今)
    └── daily_CN_511010.csv     # 十年国债ETF 511010 日线 (2013~今)
```

## 📖 数据说明

- **内置底库**：随仓库提交，保证离线/首屏即时可用
- **联网刷新**（可选）：勾选侧边栏 "尝试刷新最新行情" 后，通过腾讯财经接口补齐最近交易日 K 线；PE 用最近值线性外推（估值变化慢）。底库建议每月随 `git push` 更新一次，或手工替换 `data/` 下 CSV。
- **刷新底库脚本**（可选）：
  ```bash
  # 重新拉取 PE:   python backtest/fetch_pe.py
  # 重新拉取日线:  python backtest/fetch_csi300.py
  #                python backtest/fetch_bond_etf.py
  ```

---

## ⚠️ 风险提示

本工具仅用于研究学习，**不构成任何投资建议**。历史业绩不代表未来表现。股市有风险，投资需谨慎。请使用闲置资金，控制杠杆。

## 🧪 完整验证记录

对应回测与压力测试脚本位于 `backtest/` 目录：
- `backtest/run_riskparity.py` —— 回测入口（默认最优参数）
- `backtest/scan_riskparity.py` —— 192 组参数扫描
- `backtest/stress_test_riskparity.py` —— 参数鲁棒性 + 国债缺位应急预案