"""Open-Meteo API から愛知近海の気象・海象データを取得する。

無料・登録不要・JSON返却。
  - Forecast API: 気温、風、気圧、降水（直近 ~92日 + 未来16日）
  - Archive API:  気温、風、気圧、降水（古い過去、5日以上前まで遡れる）
  - Marine API:   波高、波向、波周期、海水面温度、海流（過去も含めて取得可）

要求された期間が長くて Forecast の遡及限界を超える場合は、
自動的に「過去分は Archive、直近+未来は Forecast」に分割して取得する。

使い方:
    from src.weather_fetcher import fetch
    df = fetch(site="irago", start="2025-11-01", end="2026-05-13")
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from . import config

_TIMEOUT = 30

# Forecast API の過去遡及限界（実測: ~92日）。安全に 60 日に設定し、
# それ以前は Archive にフォールバック。境界は十分余裕を持たせる。
_FORECAST_PAST_DAYS = 60
# Archive はデータ取り込みに数日かかるので、直近 5 日以内は Forecast に任せる。
_ARCHIVE_DELAY_DAYS = 5


def _request(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _to_dataframe(payload: dict, source: str) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["source"] = source
    return df


def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def _normalize_dates(start: str | date, end: str | date) -> tuple[str, str]:
    s, e = _to_date(start), _to_date(end)
    return s.isoformat(), e.isoformat()


def _fetch_hourly(
    url: str, site: config.Site, start: date, end: date,
    hourly_vars: list[str], source_label: str,
) -> pd.DataFrame:
    if start > end:
        return pd.DataFrame()
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "hourly": ",".join(hourly_vars),
        "timezone": config.TIMEZONE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    try:
        return _to_dataframe(_request(url, params), source_label)
    except requests.HTTPError as e:
        print(f"⚠️ {source_label} {start}->{end}: {e}")
        return pd.DataFrame()


def fetch_forecast(site: config.Site, start: str | date, end: str | date) -> pd.DataFrame:
    """Forecast 範囲のみ（直近+未来）。範囲外は呼び出し側で別途取得。"""
    return _fetch_hourly(
        config.OPEN_METEO_FORECAST_URL, site, _to_date(start), _to_date(end),
        config.FORECAST_HOURLY_VARS, "forecast",
    )


def fetch_archive(site: config.Site, start: str | date, end: str | date) -> pd.DataFrame:
    """Archive 範囲のみ（古い過去）。"""
    return _fetch_hourly(
        config.OPEN_METEO_ARCHIVE_URL, site, _to_date(start), _to_date(end),
        config.FORECAST_HOURLY_VARS, "archive",
    )


def fetch_atmosphere(site: config.Site, start: str | date, end: str | date) -> pd.DataFrame:
    """期間を自動分割して気象（気温・風・気圧等）を返す。

    過去 ~60日より古い部分は Archive API、それ以降は Forecast API。
    """
    s, e = _to_date(start), _to_date(end)
    today = date.today()
    forecast_cutoff = today - timedelta(days=_FORECAST_PAST_DAYS)

    frames: list[pd.DataFrame] = []
    # 1) 古い過去 → Archive
    if s < forecast_cutoff:
        a_end = min(e, today - timedelta(days=_ARCHIVE_DELAY_DAYS))
        if s <= a_end:
            frames.append(fetch_archive(site, s, a_end))

    # 2) 直近+未来 → Forecast
    if e >= forecast_cutoff:
        f_start = max(s, forecast_cutoff)
        if f_start <= e:
            frames.append(fetch_forecast(site, f_start, e))

    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def fetch_marine(site: config.Site, start: str | date, end: str | date) -> pd.DataFrame:
    """Marine API は過去も含めて広い範囲を一発で返す（実測）。
    エラー時は空 DataFrame を返してアプリ側を落とさない。
    """
    return _fetch_hourly(
        config.OPEN_METEO_MARINE_URL, site, _to_date(start), _to_date(end),
        config.MARINE_HOURLY_VARS, "marine",
    )


def fetch(site: str, start: str | date, end: str | date) -> pd.DataFrame:
    """指定地点・期間の気象＋海象を時刻でマージして返す。"""
    if site not in config.SITES:
        raise ValueError(f"unknown site: {site}. available: {list(config.SITES)}")
    s = config.SITES[site]

    atmosphere = fetch_atmosphere(s, start, end).drop(columns=["source"], errors="ignore")
    marine = fetch_marine(s, start, end).drop(columns=["source"], errors="ignore")

    if atmosphere.empty and marine.empty:
        return pd.DataFrame()
    if atmosphere.empty:
        merged = marine
    elif marine.empty:
        merged = atmosphere
    else:
        merged = pd.merge(atmosphere, marine, on="time", how="outer")
    merged["site"] = site
    merged["site_name_ja"] = s.name_ja
    return merged.sort_values("time").reset_index(drop=True)


def cache_path(site: str, start: str, end: str) -> Path:
    return config.WEATHER_DIR / f"{site}_{start}_{end}.parquet"


def fetch_and_cache(site: str, start: str | date, end: str | date) -> pd.DataFrame:
    s, e = _normalize_dates(start, end)
    p = cache_path(site, s, e)
    if p.exists():
        return pd.read_parquet(p)
    df = fetch(site, s, e)
    if not df.empty:
        df.to_parquet(p, index=False)
    return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Open-Meteoから気象・海象を取得")
    parser.add_argument("--site", default="shinojima", choices=list(config.SITES))
    parser.add_argument("--start", default=(date.today() - timedelta(days=7)).isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", type=Path, default=None, help="保存先 (.parquet)")
    args = parser.parse_args()

    df = fetch_and_cache(args.site, args.start, args.end)
    print(f"{len(df)} 行取得 ({args.site} {args.start} -> {args.end})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"saved: {args.out}")
    else:
        print(df.head())


if __name__ == "__main__":
    _cli()
