"""验证 PE 估值硬覆盖在真实长熊区间生效。"""
import numpy as np
import pandas as pd
from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series
from hmm_transformer.data import load_daily, build_features, make_sequences
from hmm_transformer.models import HMMTransformerStrategy


def main():
    raw = load_daily("CSI300")
    feat = build_features(raw)
    X, y, dates, closes, nc, volumes = make_sequences(feat, window=30)
    pe_df = load_pe_ttm()

    # 训练策略
    X_tr, y_tr = X[500:1200], y[500:1200]
    X_va, y_va = X[1200:1400], y[1200:1400]
    strat = HMMTransformerStrategy(duration_penalty=[(20, 1.2), (40, 0.8), (9999, 0.5)])
    strat.fit(X_tr, y_tr, X_va, y_va, epochs=5)

    # 扫描关键时点 (2015-2024 各阶段), 对比 有/无 PE 硬覆盖
    print(f'{"日期":<12}{"PE分位":>8}{"无PE仓":>8}{"有PE仓":>8}{"状态":>6}')
    for idx in [1600, 1750, 1900, 2050, 2400, 2700, 2900, 3100, 3300]:
        pe_pct = build_pe_percentile_series(pe_df, dates[idx:idx + 1])[0]
        w = X[idx - 60:idx + 1]
        c_w = closes[idx - 60:idx + 1]
        v_w = volumes[idx - 60:idx + 1]
        pos_wo = strat.positions(w, initial_position=0.0, pe_pct=None,
                                 closes=c_w, volumes=v_w)
        pos_w = strat.positions(w, initial_position=0.0,
                                pe_pct=np.full(len(w), pe_pct),
                                closes=c_w, volumes=v_w)
        prob = strat.hmm.predict_proba(w[-1:, -1, :])[0]
        if prob[0] > prob[1] and prob[0] > prob[2]:
            st = "熊"
        elif prob[2] > prob[1]:
            st = "牛"
        else:
            st = "震荡"
        print(f"{str(pd.Timestamp(dates[idx]).date()):<12}"
              f"{pe_pct:>8.3f}{pos_wo[-1]:>8.2f}{pos_w[-1]:>8.2f}{st:>6}")
    print()
    print("结论: 若低估区(PE分位<0.3)且长熊段, 有PE仓应显著高于无PE仓")


if __name__ == "__main__":
    main()