"""检查 position=0.5 的日期的 PE 分位是否低于 30%（验证硬覆盖是否应生效）。"""
import numpy as np
import pandas as pd
from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series

df = pd.read_csv("hmm_transformer_CSI300_daily.csv")
df["date"] = pd.to_datetime(df["date"])
pe_df = load_pe_ttm()

low = df[df["position"] == 0.5]
print(f"position=0.5 的天数: {len(low)}")
if len(low):
    pcts = build_pe_percentile_series(pe_df, low["date"].to_numpy())
    for d, p in zip(low["date"], pcts):
        print(f"  {d.date()} PE分位={p:.3f}")
    print("PE分位 <0.3 的 0.5 仓位天数:", int((np.array(pcts) < 0.3).sum()))