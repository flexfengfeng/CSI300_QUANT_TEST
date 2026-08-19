#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已保存的 _daily.csv 重绘 HMM+Transformer 三合一图表 (不重新训练模型)。"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))


def plot_daily(csv_path: str, png_path: str, ticker: str):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 中文字体配置 (修复 legend 方块)
    plt.rcParams["font.sans-serif"] = [
        "PingFang HK", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df["date"], np.exp(df["strategy_ret"].cumsum()), label="策略", lw=1.5)
    axes[0].plot(df["date"], np.exp(df["market_ret"].cumsum()), label="买入持有", lw=1.2, alpha=0.7)
    axes[0].set_ylabel("净值")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"HMM + Transformer WalkForward | {ticker}")

    axes[1].plot(df["date"], df["position"], drawstyle="steps-post", label="仓位", color="tab:purple")
    axes[1].set_ylabel("仓位")
    axes[1].set_ylim(0, 1.3)
    axes[1].legend(loc="upper left")

    axes[2].plot(df["date"], df["bull_prob"], label="HMM 牛态概率", color="tab:green")
    axes[2].plot(df["date"], df["transformer_pred"], label="Transformer 预测", color="tab:orange", alpha=0.7)
    axes[2].set_ylabel("信号")
    axes[2].legend(loc="upper left")

    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"已重绘 → {png_path}")


def main():
    parser = argparse.ArgumentParser(description="重绘 HMM+Transformer 图表")
    parser.add_argument("--ticker", choices=["SPY", "QQQ", "CSI300"], nargs="+",
                        default=["SPY", "QQQ", "CSI300"])
    args = parser.parse_args()

    for ticker in args.ticker:
        csv_path = os.path.join(BASE, f"hmm_transformer_{ticker}_daily.csv")
        png_path = os.path.join(BASE, f"hmm_transformer_{ticker}_equity.png")
        if not os.path.exists(csv_path):
            print(f"跳过 {ticker}: 缺 {csv_path}")
            continue
        plot_daily(csv_path, png_path, ticker)


if __name__ == "__main__":
    main()