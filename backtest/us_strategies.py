#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股指数量化策略对比 —— SPY(标普500) / QQQ(纳指100)
========================================================
数据: 2014-01-02 ~ 2026-06-02 日线 QFQ 前复权(含分红), 已缓存
      backtest/daily_US_SPY.csv / daily_US_QQQ.csv
成本: 美股ETF极低摩擦 → 双边佣金+滑点合计 5 bps(0.0005/边×2)
      (对比港股策略约0.28%, 约 1/56)

策略矩阵(全部日线, 初始 100,000, 复利, 只多, 无杠杆):
  A 买入持有(基准)
  B MA10/30 只多择时(与港股策略同构, 直接迁移)
  C MA50/200 只多择时(经典趋势)
  D 价格 > 200日均线 持有(200日线上方持有, Faber)
  E RSI(14) 超卖买/超买卖 (均值回归: <25买, >75 卖出)
  F 波动率过滤: MA10/30 + 20日年化波动 < 30% 才持仓

指标: CAGR / 最大回撤 / 夏普 / Calmar / 在市天数
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
INIT = 100_000.0
SLIP_COST = 0.0005   # 每边滑点+佣金合计 (5bps), 双边 = 0.001

INDEXES = [
    ('US.SPY', '标普500 SPY'),
    ('US.QQQ', '纳指100 QQQ'),
]


def load(code):
    df = pd.read_csv(os.path.join(BASE, f'daily_{code.replace(".", "_")}.csv'),
                     parse_dates=['time_key'])
    return df


def run_strategy(df, strategy):
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    dates = df['time_key']
    n = len(df)

    close_s = pd.Series(closes)
    ma10 = close_s.rolling(10).mean().to_numpy()
    ma30 = close_s.rolling(30).mean().to_numpy()
    ma50 = close_s.rolling(50).mean().to_numpy()
    ma200 = close_s.rolling(200).mean().to_numpy()
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100/(1+rs)).to_numpy()
    rsi = np.where(np.isnan(rsi), 50, rsi)
    # 20日年化波动
    vol20 = (close_s.pct_change().rolling(20).std() * np.sqrt(252) * 100).to_numpy()

    def sig(i):
        if strategy == 'B':
            return 1 if ma10[i-1] >= ma30[i-1] else -1
        if strategy == 'C':
            return 1 if ma50[i-1] >= ma200[i-1] else -1
        if strategy == 'D':
            if np.isnan(ma200[i-1]):
                return -1
            return 1 if closes[i-1] >= ma200[i-1] else -1
        if strategy == 'E':
            if rsi[i-1] <= 25:
                return 1
            if rsi[i-1] >= 75:
                return -1
            return 0
        if strategy == 'F':
            if vol20[i-1] > 30:
                return -1
            return 1 if ma10[i-1] >= ma30[i-1] else -1
        return 1  # A: 一直持有

    cash = INIT
    qty = 0
    entry_px = 0.0
    for i in range(n):
        o = opens[i]
        if o is None or (isinstance(o, float) and (o != o or o <= 0)):
            continue
        if i < 2 or np.isnan(ma30[i-1]):
            continue
        s = sig(i)

        # 平仓
        if qty > 0 and s == -1:
            sell_px = o * (1 - SLIP_COST)
            cash += sell_px * qty
            qty = 0
        # 开仓
        if qty == 0 and s == 1:
            qty = int(cash / (o * (1 + SLIP_COST)))
            if qty > 0:
                cash -= o * (1 + SLIP_COST) * qty

    # 期末估值
    final = cash + qty * closes[-1]
    # 净值曲线(粗, 只记月末)
    return final


def metrics(df, strategy):
    closes = df['close'].to_numpy()
    opens = closes   # 缓存仅含close, 日线低频趋势用收盘价成交(近似)
    n = len(df)
    close_s = pd.Series(closes)
    ma10 = close_s.rolling(10).mean().to_numpy()
    ma30 = close_s.rolling(30).mean().to_numpy()
    ma50 = close_s.rolling(50).mean().to_numpy()
    ma200 = close_s.rolling(200).mean().to_numpy()
    delta = close_s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = np.where(np.isnan((100 - 100/(1+rs)).to_numpy()), 50, (100 - 100/(1+rs)).to_numpy())
    vol20 = (close_s.pct_change().rolling(20).std() * np.sqrt(252) * 100).to_numpy()

    def sig(i):
        if strategy == 'B':
            return 1 if ma10[i-1] >= ma30[i-1] else -1
        if strategy == 'C':
            return 1 if ma50[i-1] >= ma200[i-1] else -1
        if strategy == 'D':
            if np.isnan(ma200[i-1]):
                return -1
            return 1 if closes[i-1] >= ma200[i-1] else -1
        if strategy == 'E':
            if rsi[i-1] <= 25:
                return 1
            if rsi[i-1] >= 75:
                return -1
            return 0
        if strategy == 'F':
            if vol20[i-1] > 30:
                return -1
            return 1 if ma10[i-1] >= ma30[i-1] else -1
        return 1

    cash = INIT
    qty = 0
    equity_curve = []
    in_market = 0
    for i in range(n):
        o, c = opens[i], closes[i]
        if o is None or (isinstance(o, float) and (o != o or o <= 0)):
            o = c
        if i >= 2 and not np.isnan(ma30[i-1]):
            s = sig(i)
            if qty > 0 and s == -1:
                cash += o * (1 - SLIP_COST) * qty
                qty = 0
            if qty == 0 and s == 1:
                qty = int(cash / (o * (1 + SLIP_COST)))
                if qty > 0:
                    cash -= o * (1 + SLIP_COST) * qty
        if qty > 0:
            in_market += 1
        equity_curve.append(cash + qty * c)

    eq = pd.Series(equity_curve, index=df['time_key'])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = eq.iloc[-1]
    cagr = (final / INIT) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    daily_r = eq.pct_change().dropna()
    sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else np.nan
    calmar = cagr / abs(dd) if dd != 0 else np.nan
    return cagr, total_ret(final), dd, sharpe, calmar, in_market / n


def total_ret(final):
    return final / INIT - 1


def main():
    print('=' * 130)
    print('美股指数量化策略对比 · SPY / QQQ · 2014-01 ~ 2026-06 · 初始 100,000 · 复利 · 双边成本5bps')
    print('=' * 130)
    for code, label in INDEXES:
        df = load(code)
        print('\n### %s  %s' % (code, label))
        print('%-14s %8s %10s %8s %7s %8s %8s' % (
            '策略', 'CAGR%', '总收益%', '最大回撤%', '夏普', 'Calmar', '在市%'))
        for name, strat in [
            ('A 买入持有', 'A'),
            ('B MA10/30', 'B'),
            ('C MA50/200', 'C'),
            ('D 200日线持有', 'D'),
            ('E RSI均值回归', 'E'),
            ('F MA10/30+波动过滤', 'F'),
        ]:
            cagr, total, dd, sharpe, calmar, inmkt = metrics(df, strat)
            print('%-14s %+7.2f %+9.1f %+7.1f %6.2f %8.2f %7.1f%%' % (
                name, cagr * 100, total * 100, dd * 100, sharpe, calmar, inmkt * 100))

    print('\n解读:')
    print(' - 成本5bps/边(0.05%), 远低于港股(≈0.14%/边)')
    print(' - MA10/30 与港股策略同构可直接迁移; 200日线持有是Faber经典策略')
    print(' - RSI均值回归专门测试指数超买超卖有效性')


if __name__ == '__main__':
    main()