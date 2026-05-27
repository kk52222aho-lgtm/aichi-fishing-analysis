"""クロロフィルa・海色データの取得（NASA OceanColor / JAXA Himawari）。

データソース:
    1. NASA OceanColor ERDDAP - https://oceandata.sci.gsfc.nasa.gov/
       MODIS-Aqua / VIIRS の L3 日次プロダクト
       例: chlor_a (mg/m^3), Kd_490 (1/m, 透明度)
    2. JAXA Himawari Monitor https://www.eorc.jaxa.jp/ptree/
       静止衛星なので雲が抜ければ日次取得可能
    3. Copernicus Marine - 要登録 https://marine.copernicus.eu/

伊勢湾・三河湾は内湾でピクセル特性が陸地に近く、衛星アルゴリズムの誤差大きいので
ピクセル選定は注意。沿岸補正された L2 LAC 推奨。

実装方針:
    - ERDDAP の griddap でCSV取得（時間範囲 × box中央値）
    - 各サイト周辺3km四方の中央値を抽出
    - キャッシュは parquet ベース

特徴量:
    chlor_a              : サイト直近ピクセルの chlor_a (mg/m^3)
    chlor_a_anomaly_30d  : 30日平均との偏差（赤潮警戒指標）
    chlor_a_d7d          : 7日変化率（プランクトン増減）
    water_transparency   : Kd_490 から計算した透明度（1/Kd_490）
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CHLOR_DIR = config.DATA_DIR / "chlorophyll"
CHLOR_DIR.mkdir(parents=True, exist_ok=True)

_ERDDAP_BASE = "https://oceandata.sci.gsfc.nasa.gov/erddap/griddap/"
_DATASET_CHLOR = "erdMH1chla1day"
_DATASET_KD490 = "erdMH1kd4901day"


def _cache_path(site: str, year: int, kind: str) -> Path:
    return CHLOR_DIR / f"{site}_{kind}_{year}.parquet"


def fetch_site_pixel(site_code: str, start: date, end: date,
                     kind: str = "chlor_a", timeout: int = 60) -> pd.DataFrame:
    if site_code not in config.SITES:
        raise ValueError(f"unknown site: {site_code}")
    s = config.SITES[site_code]
    lat, lon = s.latitude, s.longitude

    cache = _cache_path(site_code, start.year, kind)
    if cache.exists():
        df = pd.read_parquet(cache)
        return df[(df["date"] >= start) & (df["date"] <= end)]

    try:
        import requests
        dataset = _DATASET_CHLOR if kind == "chlor_a" else _DATASET_KD490
        var = "chlorophyll" if kind == "chlor_a" else "Kd_490"
        d_lat = 0.03
        d_lon = 0.03
        url = (
            f"{_ERDDAP_BASE}{dataset}.csv?"
            f"{var}[({start.isoformat()}T00:00:00Z):({end.isoformat()}T00:00:00Z)]"
            f"[({lat - d_lat}):({lat + d_lat})]"
            f"[({lon - d_lon}):({lon + d_lon})]"
        )
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), skiprows=[1])
        if "time" not in df.columns or var not in df.columns:
            return pd.DataFrame(columns=["date", "value"])
        df["date"] = pd.to_datetime(df["time"]).dt.date
        daily = df.groupby("date")[var].median().reset_index()
        daily = daily.rename(columns={var: "value"})
        daily.to_parquet(cache, index=False)
        return daily
    except Exception as e:
        print(f"⚠️ chlor fetch failed for {site_code}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def compute_features_for_catches(catches: pd.DataFrame) -> pd.DataFrame:
    cols = ["chlor_a", "chlor_a_anomaly_30d", "chlor_a_d7d", "water_transparency"]
    out = pd.DataFrame({c: [np.nan] * len(catches) for c in cols}, index=catches.index)

    if "site" not in catches.columns or "datetime" not in catches.columns:
        return out

    catches_dt = pd.to_datetime(catches["datetime"]).dt.normalize()

    for site_code in catches["site"].dropna().unique():
        if site_code not in config.SITES:
            continue
        mask = catches["site"] == site_code
        rows_idx = catches.index[mask]
        if len(rows_idx) == 0:
            continue

        d_min = catches_dt[mask].min().date() - timedelta(days=30)
        d_max = catches_dt[mask].max().date()

        chlor = fetch_site_pixel(site_code, d_min, d_max, kind="chlor_a")
        kd = fetch_site_pixel(site_code, d_min, d_max, kind="kd_490")

        if chlor.empty:
            continue
        chlor = chlor.sort_values("date").set_index("date")["value"]
        for i in rows_idx:
            target_d = catches_dt.loc[i].date()
            if target_d in chlor.index:
                out.loc[i, "chlor_a"] = float(chlor.loc[target_d])
            else:
                window = chlor[(chlor.index >= target_d - timedelta(days=3))
                               & (chlor.index <= target_d + timedelta(days=3))]
                if not window.empty:
                    out.loc[i, "chlor_a"] = float(window.mean())

            past30 = chlor[(chlor.index >= target_d - timedelta(days=30))
                           & (chlor.index < target_d)]
            if not past30.empty and not np.isnan(out.loc[i, "chlor_a"]):
                out.loc[i, "chlor_a_anomaly_30d"] = out.loc[i, "chlor_a"] - past30.mean()

            d_lag7 = target_d - timedelta(days=7)
            if d_lag7 in chlor.index and not np.isnan(out.loc[i, "chlor_a"]):
                out.loc[i, "chlor_a_d7d"] = out.loc[i, "chlor_a"] - float(chlor.loc[d_lag7])

        if not kd.empty:
            kd = kd.sort_values("date").set_index("date")["value"]
            for i in rows_idx:
                target_d = catches_dt.loc[i].date()
                if target_d in kd.index:
                    k = float(kd.loc[target_d])
                    if k > 0:
                        out.loc[i, "water_transparency"] = 1.0 / k

    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="クロロフィル取得")
    parser.add_argument("--site", required=True, choices=list(config.SITES))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--kind", default="chlor_a", choices=["chlor_a", "kd_490"])
    args = parser.parse_args()

    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)
    df = fetch_site_pixel(args.site, s, e, kind=args.kind)
    print(df.head())
    print(f"{len(df)} 行 取得")


if __name__ == "__main__":
    _cli()
