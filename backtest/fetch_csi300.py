#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从腾讯财经接口下载沪深300 (sh000300) 指数日线, 覆盖 2006-08 ~ 2026-08。

接口语义: `symbol,day,,{end},{count},qfq` 返回 end 之前最近 count 条。
用连续翻页: 每次取 800 条, 下轮 end = 已取首条日期 - 1 天, 直到覆盖 2006 年。
输出: daily_CN_000300.csv (time_key, open, high, low, close, volume)
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SYMBOL = "sh000300"
OUT = os.path.join(BASE, "daily_CN_000300.csv")
START_DATE = "2006-08-01"


def fetch_batch(end: str, count: int = 800) -> list:
    """拉取 end 之前最近 count 条。返回 [date, open, close, high, low, volume] 列表。"""
    params = {"param": f"{SYMBOL},day,,{end},{count},qfq"}
    r = requests.get(URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    d = data.get("data")
    if not isinstance(d, dict) or SYMBOL not in d:
        print(f"  [{end}] 接口异常: {data.get('msg')}")
        return []
    kd = d[SYMBOL]
    return kd.get("qfqday") or kd.get("day") or []


def main():
    rows = {}
    end = datetime(2026, 8, 18).strftime("%Y-%m-%d")
    page = 0
    while True:
        page += 1
        seg = fetch_batch(end)
        if not seg:
            print(f"第 {page} 页 ({end}) 为空, 停止")
            break
        for item in seg:
            date = item[0]
            if len(item) >= 6:
                rows[date] = {
                    "time_key": pd.Timestamp(date),
                    "open": float(item[1]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "volume": float(item[5]),
                }
        first_date = seg[0][0]
        print(f"第 {page} 页: {first_date} ~ {seg[-1][0]} ({len(seg)} 条)")
        if first_date <= START_DATE:
            break
        # 下一轮: end = 首条日期前一天
        prev = datetime.strptime(first_date, "%Y-%m-%d") - timedelta(days=1)
        end = prev.strftime("%Y-%m-%d")
        time.sleep(0.4)

    df = pd.DataFrame.from_dict(rows, orient="index")
    # DataFrame 用 date 字符串做索引, 先转 datetime 排序
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index >= START_DATE].reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\n共 {len(df)} 条 → {OUT}")
    print(f"区间: {df['time_key'].iloc[0].date()} ~ {df['time_key'].iloc[-1].date()}")


if __name__ == "__main__":
    main()