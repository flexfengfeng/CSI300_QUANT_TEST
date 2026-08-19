#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载沪深300 (000300) 历史 PE_TTM 序列存 CSV。

复用 akshare 内部辅助 (hash_code / get_cookie_csrf) 绕过 JS 混淆,
但手动解析数据以兼容旧数据中字符串日期 (akshare 的 unit='ms' 会崩溃)。
输出: csi300_pe_ttm.csv (date, pe_ttm)
"""
import os
import json

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "csi300_pe_ttm.csv")
LOG = os.path.join(BASE, "fetch_pe.log")
URL = "https://legulegu.com/api/stockdata/index-basic-pe"
REFERER = "https://legulegu.com/stockdata/sz50-ttm-lyr"

log_lines = []


def log(msg):
    log_lines.append(str(msg))
    print(msg, flush=True)


def main():
    try:
        # 复用 akshare 内部 token 生成与 cookie 获取
        from akshare.stock_feature.stock_a_pe_and_pb import hash_code, get_cookie_csrf
        import py_mini_racer
        from datetime import datetime

        js = py_mini_racer.MiniRacer()
        js.eval(hash_code)
        token = js.call("hex", datetime.now().date().isoformat()).lower()

        params = {"token": token, "indexCode": "000300.SH"}
        # get_cookie_csrf 返回 {"cookies":..., "headers":...}, 用 ** 展开
        r = requests.get(URL, params=params, timeout=30, **get_cookie_csrf(url=REFERER))
        log(f"HTTP: {r.status_code}")
        r.raise_for_status()
        data_json = r.json()
        items = data_json.get("data", [])
        log(f"原始条数: {len(items)}")
        if not items:
            log(f"返回片段: {json.dumps(data_json, ensure_ascii=False)[:500]}")
            return
        log(f"首条字段: {list(items[0].keys())}")

        rows = []
        for it in items:
            d = it.get("date")
            # 兼容毫秒时间戳与 'YYYY-MM-DD' 字符串
            if isinstance(d, (int, float)):
                date = pd.to_datetime(d, unit="ms", utc=True) \
                    .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
            else:
                date = str(d)[:10]
            pe = it.get("ttmPe") or it.get("ttm_pe") or None
            if pe is None:
                continue
            try:
                rows.append({"date": date, "pe_ttm": float(pe)})
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        df = df.dropna().sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
        df.to_csv(OUT, index=False)
        log(f"已保存 {len(df)} 条 → {OUT}")
        if len(df):
            log(f"区间: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            log(f"最新 3 条:\n{df.tail(3).to_string(index=False)}")
            log(f"最早 3 条:\n{df.head(3).to_string(index=False)}")
            # 描述统计, 快速验证 10 年窗口可用
            log(f"\n近 10 年缺失 PE 天数检查: "
                f"{df[df['date'] >= '2016-01-01'].shape[0]} 条 / "
                f"2520 交易日")
    except Exception as e:
        log(f"异常: {repr(e)[:600]}")

    with open(LOG, "w") as f:
        f.write("\n".join(log_lines))
    log(f"日志: {LOG}")


if __name__ == "__main__":
    main()