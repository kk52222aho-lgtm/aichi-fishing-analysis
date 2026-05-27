"""潮汐データ取得。

優先順:
  1. ローカルキャッシュ data/tide/{port}_{year}.csv（手動配置 or 過去取得分）
  2. 気象庁の年次潮位表 https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{port}.txt
     固定幅フォーマット（1日1行、24時間×3桁の毎時潮位[cm]）

潮位は cm 単位、観測点基準面（DL）からの値。
潮の流れ（流速・流向）は Open-Meteo Marine の ocean_current_* を使うため、ここでは扱わない。

CLI:
    python -m src.tide_fetcher --port IO --year 2026
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from . import config

_TIMEOUT = 30


def _cache_path(port: str, year: int) -> Path:
    return config.TIDE_DIR / f"{port}_{year}.csv"


def _parse_jma_text(text: str, year: int) -> pd.DataFrame:
    """JMA 固定幅潮位データをパース。

    各行: 24h × 3桁の潮位(cm) + ' YYMMDD' + 港コード(2文字) + 4組(高低潮)の "HHMMHHH"
    本実装は毎時潮位のみ抽出（高低潮の正確な解析は省略）。
    """
    records: list[dict] = []
    for line in text.splitlines():
        if len(line) < 78:
            continue
        try:
            hourly = [int(line[i*3:(i+1)*3]) for i in range(24)]
            yymmdd = line[72:78].strip()
            if len(yymmdd) != 6 or not yymmdd.isdigit():
                continue
            yy = int(yymmdd[0:2])
            mm = int(yymmdd[2:4])
            dd = int(yymmdd[4:6])
            full_year = 2000 + yy if yy < 80 else 1900 + yy
            if full_year != year:
                continue
            for h, v in enumerate(hourly):
                records.append({
                    "datetime": datetime(full_year, mm, dd, h),
                    "tide_cm": int(v),
                })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)


def fetch_tide_year(port: str, year: int, force: bool = False) -> pd.DataFrame:
    """ある港・年の毎時潮位を取得。

    Args:
        port: JMA港コード（例: "IO"=伊良湖, "NA"=名古屋）
        year: 西暦
        force: True ならキャッシュを無視して再取得
    """
    cache = _cache_path(port, year)
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["datetime"])

    url = f"https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{port}.txt"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = _parse_jma_text(resp.text, year)
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def fetch_tide_range(port: str, start: date, end: date) -> pd.DataFrame:
    """期間（複数年にまたがってもOK）の潮位を返す。"""
    years = sorted({start.year, end.year})
    frames = []
    for y in years:
        try:
            frames.append(fetch_tide_year(port, y))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["datetime", "tide_cm"])
    df = pd.concat(frames, ignore_index=True)
    mask = (df["datetime"].dt.date >= start) & (df["datetime"].dt.date <= end)
    return df.loc[mask].sort_values("datetime").reset_index(drop=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="JMAから年次潮位データを取得")
    parser.add_argument("--port", required=True, help="JMA港コード (例: IO, NA)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = fetch_tide_year(args.port, args.year, force=args.force)
    print(f"{len(df)} 行取得 ({args.port} {args.year})")
    print(df.head())


if __name__ == "__main__":
    _cli()
