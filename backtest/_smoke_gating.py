"""确定性验证动态硬切换门控 + 置换不变性 (绝不硬编码状态索引)."""
import numpy as np
import pandas as pd
from hmm_transformer.models import (
    HMMTransformerStrategy,
    VOL_BUDGET_LOWER, VOL_BUDGET_UPPER,
    REGIME_BULL_LOWER, REGIME_BULL_UPPER,
    REGIME_BEAR_LOWER, REGIME_BEAR_UPPER,
    REGIME_NEUTRAL_CORE_W, REGIME_NEUTRAL_VOL_W,
    REGIME_NEUTRAL_LOWER, REGIME_NEUTRAL_UPPER,
    REGIME_COLD_LOWER, REGIME_COLD_UPPER,
    GATE_FINAL_LOWER, GATE_FINAL_UPPER,
)

np.random.seed(2)
N = 100
X = np.random.randn(N, 30, 9).astype(np.float32)
close = np.full(N, 100.0)
close[20:40] = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.002, 20)))
close[40:60] = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.03, 20)))
close[60:] = close[59] * np.exp(np.cumsum(np.random.normal(0, 0.002, 40)))
volumes = np.full(N, 1e8)

MAP = (1.2, 0.8, 1.0)
map_arr = np.array(MAP, dtype=np.float64)
bull = int(np.argmax(map_arr)); bear = int(np.argmin(map_arr)); neutral = 3 - bull - bear
print(f"动态映射: bull_idx={bull} bear_idx={bear} neutral_idx={neutral}")

prob = np.zeros((N, 3))
prob[:20] = [0.34, 0.33, 0.33]
prob[20:40] = [0.80, 0.10, 0.10]
prob[40:60] = [0.10, 0.80, 0.10]
prob[60:80] = [0.10, 0.10, 0.80]
prob[80:] = [0.85, 0.05, 0.10]

s = HMMTransformerStrategy(vol_target=0.25, hold_threshold=0.0)
s.hmm.predict_proba = lambda _: prob
pos = s.positions(X, closes=close, volumes=volumes)

ret = pd.Series(close).pct_change()
rv = ret.rolling(20, min_periods=20).std().to_numpy() * np.sqrt(252)
raw_budget = np.where(np.isfinite(rv) & (rv > 0), 0.25 / rv, 1.0)
vol_budget = np.clip(raw_budget, VOL_BUDGET_LOWER, VOL_BUDGET_UPPER)
core = prob @ map_arr
expected = np.concatenate([
    np.clip(core[:20], REGIME_COLD_LOWER, REGIME_COLD_UPPER),
    np.clip(core[20:40], REGIME_BULL_LOWER, REGIME_BULL_UPPER),
    np.clip(vol_budget[40:60], REGIME_BEAR_LOWER, REGIME_BEAR_UPPER),
    np.clip(REGIME_NEUTRAL_CORE_W * core[60:80] + REGIME_NEUTRAL_VOL_W * vol_budget[60:80],
            REGIME_NEUTRAL_LOWER, REGIME_NEUTRAL_UPPER),
    np.clip(core[80:], REGIME_BULL_LOWER, REGIME_BULL_UPPER),
])
d = np.abs(pos - expected).max()
print(f"门控 vs 独立期望 最大差异: {d:.2e}")
assert d < 1e-9, d
print(f"  冷启动均值={pos[:20].mean():.3f} | 牛市={pos[20:40].mean():.3f} | 熊市={pos[40:60].mean():.3f} | "
      f"震荡={pos[60:80].mean():.3f} | 牛市低波={pos[80:].mean():.3f}")
assert pos.min() >= GATE_FINAL_LOWER - 1e-9 and pos.max() <= GATE_FINAL_UPPER + 1e-9
assert np.all(pos[40:60] <= REGIME_BEAR_UPPER + 1e-9), "熊市上限<=0.9"

perm = [2, 0, 1]
s2 = HMMTransformerStrategy(vol_target=0.25, hold_threshold=0.0, state_position_map=tuple(map_arr[perm]))
s2.hmm.predict_proba = lambda _: prob[:, perm]
dp = np.abs(s2.positions(X, closes=close, volumes=volumes) - pos).max()
print(f"置换不变性(打乱索引): 最大差异 {dp:.2e}")
assert dp < 1e-9, dp

print("GATING_SMOKE_OK")