# 📊 风险平价 + 资产轮动 | CSI300

基于 **沪深300 + 波动率目标 + PE 估值锚定 + 国债轮动** 的量化配置策略与 Streamlit 网页版。

> 彻底放弃在单指数上做"方向择时"（HMM），转向"风险平价 + 资产轮动"——从"赌大小"到"做配置"的质变。
> 国债（511010.SH）作为现金替代，填补闲置资金的复利损耗。

## 🧠 策略核心（三大支柱，纯确定性公式）

| 支柱 | 逻辑 |
|---|---|
| 1. 波动率目标 | `股票仓位 = 目标波动率 ÷ 过去40日实际波动率`（市场平静→加仓；恐慌→自动减仓） |
| 2. PE 估值锚定 | PE 10年分位 < 30% → 目标波动率上调；> 70% → 下调（低估敢扛、高估保守） |
| 3. 国债轮动 | 股票仓位不足 100% 时，剩余资金买十年国债ETF(511010)，对冲回撤 + 填现金损耗 |

## 📈 回测结果（2013.11 ~ 今，默认最优参数）

| 指标 | 风险平价组合 | 沪深300 买入持有 |
|---|---|---|
| CAGR | **8.82%** | 8.16% |
| 最大回撤 | **-38.6%** | -43.0% |
| 夏普 | **0.50** | 0.37 |
| 年化波动 | **16.8%** | 21.3% |
| 超额收益 | **+7.9pp** | — |

**压力测试结论**
- 参数邻域 25 组（目标波动18~22% × 窗口30~50天）**100% 跑赢基准** → 高原型，抗过拟合
- 国债缺位改持货币基金 → CAGR 仍 8.4~8.5%（仅降 0.3~0.5pp）

## ✨ 网页版（webapp/）

Streamlit 仪表盘：今日仓位信号牌 / 回测净值曲线 / 参数鲁棒性热力图 / 实时调参 / CSV 导出。

**本地运行：**
```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py   # http://localhost:8501
```

**部署 Streamlit Community Cloud：**
1. https://share.streamlit.io → GitHub 登录
2. New app → 仓库 `flexfengfeng/CSI300_QUANT_TEST` → Branch=`main` → Main file path=`webapp/app.py`
3. Deploy → 获得公开链接

## 🔄 数据更新机制

**每日自动更新（GitHub Actions）**：`.github/workflows/update_data.yml`
- 每个交易日 **16:15（北京时间）** 自动运行：抓取沪深300 + 国债ETF 最新日线 → 同步到 `webapp/data/` → 自动 commit+push → Streamlit Cloud 自动重新部署
- 也可**手动触发**：GitHub 仓库 → **Actions** → `Update Market Data` → **Run workflow**（`workflow_dispatch`）
- 数据无变化时（节假日/已最新）自动跳过提交，不产生空 commit

**网页版可选联网刷新**：打开页面侧边栏的"尝试刷新最新行情"复选框，可在会话内临时补齐最近交易日数据（不写入仓库）。

## ⚠️ 边界说明（重要）

| 项 | 状态 | 说明 |
|---|---|---|
| 沪深300 / 国债ETF 日线 | ✅ 每日自动更新 | 腾讯财经接口，依赖仅 `requests`，CI 稳定 |
| PE_TTM 估值数据 | ⚠️ **需手动更新** | legulegu 接口需 `akshare` + `py_mini_racer`，未纳入 CI。PE 变化慢，建议**每月**手工刷新一次（见下） |
| 联网刷新失败 | ✅ 自动降级 | 页面勾选联网刷新时若不成功，自动回退使用仓库内置底库 |
| 历史数据完整性 | ✅ 覆盖全 | 沪深300(2006~)、国债ETF(2013~)、PE(2005~) |

## 🛠️ 需要手动操作的部分

1. **每月：刷新 PE_TTM 底库**
   ```bash
   # 本地执行 (需 akshare + py_mini_racer)
   python backtest/fetch_pe.py
   cp backtest/csi300_pe_ttm.csv webapp/data/csi300_pe_ttm.csv
   git add webapp/data/csi300_pe_ttm.csv
   git commit -m "chore(data): refresh PE_TTM [$(date +%Y-%m)]"
   git push
   ```
   push 后 Streamlit Cloud 自动重新部署。

2. **按需：手动触发每日数据更新**
   GitHub 仓库 → **Actions** → `Update Market Data` → **Run workflow**（数据文件缺失或想立即更新时）

3. **首次部署 Streamlit**：见上方"部署 Streamlit Community Cloud"步骤（一次性）

4. **本地跑回测/压力测试**（可选）：
   ```bash
   cd backtest
   python run_riskparity.py --plot        # 回测
   python stress_test_riskparity.py       # 压力测试
   ```

## 📁 目录结构

```
webapp/      Streamlit 网页版（含数据底库 data/，部署必需）
backtest/   策略研究：回测 / 参数扫描 / 压力测试脚本
  ├─ run_riskparity.py        回测入口（默认最优参数）
  ├─ scan_riskparity.py       192 组参数扫描
  ├─ stress_test_riskparity.py 鲁棒性 + 国债缺位应急预案
  └─ hmm_transformer/         核心模型包（RiskParityStrategy）
```

## ⚠️ 免责声明

本仓库仅用于研究学习，不构成任何投资建议。历史业绩不代表未来表现。股市有风险，投资需谨慎。