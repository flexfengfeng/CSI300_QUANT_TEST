#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMM + Transformer + WalkForward 量化验证 | SPY/QQQ 近 20 年日线。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_transformer.data import load_daily, build_features, make_sequences
from hmm_transformer.walkforward import WalkForwardConfig, run_walkforward, save_report
try:
    from hmm_transformer.valuation import load_pe_ttm, build_pe_percentile_series
    HAS_VALUATION = True
except ImportError:
    HAS_VALUATION = False


def main():
    parser = argparse.ArgumentParser(description="HMM+Transformer WalkForward 回测")
    parser.add_argument("--ticker", choices=["SPY", "QQQ", "CSI300", "CSI500"], default="SPY")
    parser.add_argument("--window", type=int, default=30, help="Transformer 序列窗口(交易日)")
    parser.add_argument("--train-days", type=int, default=1260)
    parser.add_argument("--val-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--duration-penalty", default=None,
                        help="熊市持续期惩罚, 格式 '20:1.2,40:0.8,9999:0.5' 表示 "
                             "≤20日→120%, ≤40日→80%, 更久→50%. 默认不启用")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="动态硬切换门控: HMM无监督→由 state_position_map 动态定义牛/熊/震荡 "
                             "(绝不硬编码状态索引). 牛市=完全听方向 clip(core,0.8,1.5); "
                             "熊市=完全听波动率 clip(vol_budget,0.3,0.9); "
                             "震荡=4:6加权 clip(0.4×core+0.6×vol_budget,0.4,1.1); "
                             "冷启动(前20日)=中性 clip(core,0.6,1.0); 最终 clip(0.3,1.5). "
                             "波动率预算 = clip(目标波动/实现波动, 0.5, 1.5). "
                             "例: 0.25 = 25% 年化目标波动 (中证500历史中枢). 默认不启用")
    parser.add_argument("--vol-lookback", type=int, default=20,
                        help="波动率预算滚动窗口(交易日), 默认 20")
    parser.add_argument("--vol-lower", type=float, default=0.4,
                        help="[兼容保留, 已弃用] 旧波动率直接定仓下限, 新公式不再使用")
    parser.add_argument("--vol-upper", type=float, default=1.5,
                        help="[兼容保留, 已弃用] 旧波动率直接定仓上限, 新公式不再使用")
    parser.add_argument("--no-panic-override", action="store_true",
                        help="关闭恐慌性抛售逆向硬覆盖(越跌越买); 用于与开启时做 A/B 对比")
    parser.add_argument("--device", default=None,
                        help="训练设备: auto(默认, 自动选MPS/cuda/cpu), mps, cpu, cuda")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行折数: 4=默认(实测本机最优), 1=串行, N=指定N进程")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true", help="输出权益曲线 PNG")
    args = parser.parse_args()

    # 自动选择设备: MPS (Apple Silicon) > CUDA > CPU
    if args.device is None or args.device == "auto":
        import torch
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"训练设备: {args.device}")

    print(f"加载 {args.ticker} 数据 ...")
    raw = load_daily(args.ticker)
    print(f"  {len(raw)} 根日线, {raw['time_key'].iloc[0].date()} ~ {raw['time_key'].iloc[-1].date()}")

    print("构造特征 ...")
    features = build_features(raw)
    X, y, dates, closes, next_closes, volumes = make_sequences(features, window=args.window)
    print(f"  序列样本: {X.shape}"
          + (" | 含成交量(恐慌硬覆盖启用)" if volumes is not None
             else " | 无成交量数据(恐慌硬覆盖跳过)"))

    # 解析持续时间惩罚: "20:1.2,40:0.8,9999:0.5" → [(20,1.2),(40,0.8),(9999,0.5)]
    duration_penalty = None
    if args.duration_penalty:
        duration_penalty = []
        for seg in args.duration_penalty.split(","):
            limit, pos = seg.split(":")
            duration_penalty.append((int(limit), float(pos)))

    # CSI300/CSI500 可选: 加载对应指数 PE_TTM 并计算滚动10年分位数 (估值硬覆盖)
    pe_pct_series = None
    if args.ticker in ("CSI300", "CSI500") and HAS_VALUATION:
        try:
            pe_df = load_pe_ttm(args.ticker)
            pe_pct_series = build_pe_percentile_series(pe_df, dates, lookback_years=10)
            print(f"已加载 {args.ticker} PE_TTM 估值分位数: {len(pe_pct_series)} 个交易日 (滚动10年)")
        except Exception as e:
            print(f"PE 估值加载失败(忽略, 不启用硬覆盖): {e}")

    cfg = WalkForwardConfig(
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
        duration_penalty=duration_penalty,
        vol_target=args.vol_target,
        vol_lookback=args.vol_lookback,
        vol_lower=args.vol_lower,
        vol_upper=args.vol_upper,
        num_workers=args.workers,
        pe_pct_series=pe_pct_series,
        enable_panic_override=not args.no_panic_override,
    )

    vol_desc = (f" | 动态硬切换门控: target={cfg.vol_target} "
                f"lookback={cfg.vol_lookback} 牛市[0.8,1.5]/熊市[0.3,0.9]/震荡[0.4,1.1]/最终[0.3,1.5]"
                if cfg.vol_target is not None else "")
    print(f"开始 WalkForward 回测 | 折配置: train={cfg.train_days} val={cfg.val_days} "
          f"test={cfg.test_days} step={cfg.step_days}{vol_desc}")
    result = run_walkforward(X, y, dates, closes, next_closes, cfg, volumes=volumes)

    base = save_report(result, args.ticker)
    print(f"\n报告已保存: {base}_report.txt / {base}_daily.csv")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            # 修复中文 legend 显示为方块: 配置中文字体
            plt.rcParams["font.sans-serif"] = [
                "PingFang HK", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"
            ]
            plt.rcParams["axes.unicode_minus"] = False

            df = result["df"]
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(df["date"], np.exp(df["strategy_ret"].cumsum()), label="策略")
            axes[0].plot(df["date"], np.exp(df["market_ret"].cumsum()), label="买入持有", alpha=0.7)
            axes[0].set_ylabel("净值")
            axes[0].legend()
            axes[0].set_title(f"HMM+Transformer WalkForward | {args.ticker}")

            axes[1].plot(df["date"], df["position"], drawstyle="steps-post", label="仓位", color="tab:purple")
            axes[1].set_ylabel("仓位")
            axes[1].legend()

            axes[2].plot(df["date"], df["bull_prob"], label="HMM 牛态概率", color="tab:green")
            axes[2].plot(df["date"], df["transformer_pred"], label="Transformer 预测", color="tab:orange", alpha=0.7)
            axes[2].set_ylabel("信号")
            axes[2].legend()

            plt.tight_layout()
            png = base + "_equity.png"
            plt.savefig(png, dpi=120)
            print(f"图表已保存: {png}")
        except Exception as e:
            print(f"绘图失败(忽略): {e}")


if __name__ == "__main__":
    main()