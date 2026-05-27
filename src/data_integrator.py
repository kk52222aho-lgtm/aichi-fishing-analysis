"""釣果ログに気象・海象・潮汐・天文・河川・黒潮・市場・上流・海色を結合する。"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import astronomy, config, derived_features, fishing_loader, tide_fetcher, weather_fetcher

try:
    from . import river_flow_fetcher
    _HAS_RIVER = True
except Exception:
    _HAS_RIVER = False

try:
    from . import kuroshio_fetcher
    _HAS_KUROSHIO = True
except Exception:
    _HAS_KUROSHIO = False

try:
    from . import market_price_fetcher
    _HAS_MARKET = True
except Exception:
    _HAS_MARKET = False

try:
    from . import upstream_catch_fetcher
    _HAS_UPSTREAM = True
except Exception:
    _HAS_UPSTREAM = False

try:
    from . import chlorophyll_fetcher
    _HAS_CHLOR = True
except Exception:
    _HAS_CHLOR = False

_DERIVED_PADDING_DAYS = 8


def _strip_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def _weather_for(site: str, start, end, derive: bool = True) -> pd.DataFrame:
    df = weather_fetcher.fetch_and_cache(site, start, end)
    if df.empty:
        return df
    df = df.copy()
    df["time"] = _strip_tz(df["time"])
    df = df.drop(columns=["site", "site_name_ja"], errors="ignore")
    df = df.sort_values("time").reset_index(drop=True)
    if derive:
        df = derived_features.enrich_weather(df)
    return df


def _tide_for(port: str, start, end, derive: bool = True) -> pd.DataFrame:
    df = tide_fetcher.fetch_tide_range(port, start, end)
    if df.empty:
        return df
    df = df.copy()
    df["datetime"] = _strip_tz(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    if derive:
        df = derived_features.enrich_tide(df)
    return df.rename(columns={"datetime": "tide_time"})


def integrate(
    catches: pd.DataFrame,
    padding_days: int = _DERIVED_PADDING_DAYS,
    with_rivers: bool = False,
    with_market: bool = True,
    with_upstream: bool = True,
    with_chlorophyll: bool = False,
) -> pd.DataFrame:
    """釣果DFに全外部データを結合して返す。"""
    if catches.empty:
        return catches.copy()

    catches = catches.copy()
    catches["datetime"] = _strip_tz(catches["datetime"])

    enriched_chunks: list[pd.DataFrame] = []
    for site, group in catches.groupby("site"):
        if not site or site not in config.SITES:
            enriched_chunks.append(group)
            continue
        site_obj = config.SITES[site]
        start = (group["datetime"].min() - timedelta(days=padding_days)).date()
        end = (group["datetime"].max() + timedelta(days=1)).date()

        weather = _weather_for(site, start, end)
        if not weather.empty:
            group = pd.merge_asof(
                group.sort_values("datetime"), weather,
                left_on="datetime", right_on="time",
                direction="nearest", tolerance=pd.Timedelta(hours=1),
            )

        tide = _tide_for(site_obj.tide_port, start, end)
        if not tide.empty:
            group = pd.merge_asof(
                group.sort_values("datetime"), tide,
                left_on="datetime", right_on="tide_time",
                direction="nearest", tolerance=pd.Timedelta(hours=1),
            )

        if with_rivers and _HAS_RIVER:
            try:
                rivers = river_flow_fetcher.fetch_site_rivers(site, start, end, kind="discharge")
                if not rivers.empty:
                    rivers = rivers.copy()
                    rivers["time"] = pd.to_datetime(rivers["time"])
                    if getattr(rivers["time"].dt, "tz", None) is not None:
                        rivers["time"] = rivers["time"].dt.tz_localize(None)
                    rivers = rivers.sort_values("time")
                    group = pd.merge_asof(
                        group.sort_values("datetime"), rivers,
                        left_on="datetime", right_on="time",
                        direction="backward", tolerance=pd.Timedelta(hours=24),
                        suffixes=("", "_river"),
                    )
            except Exception as e:
                print(f"⚠️ river: {e}")

        group = astronomy.annotate(group, site_obj.latitude, site_obj.longitude)
        enriched_chunks.append(group)

    out = pd.concat(enriched_chunks, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    if _HAS_KUROSHIO and not out.empty:
        try:
            s = out["datetime"].min().date()
            e = out["datetime"].max().date()
            ks = kuroshio_fetcher.kuroshio_state_for_range(s, e)
            if not ks.empty:
                ks["time"] = pd.to_datetime(ks["time"])
                out = pd.merge_asof(
                    out.sort_values("datetime"), ks.sort_values("time"),
                    left_on="datetime", right_on="time",
                    direction="nearest", tolerance=pd.Timedelta(hours=24),
                    suffixes=("", "_kuro"),
                )
                out = out.drop(columns=["time_kuro"], errors="ignore")
        except Exception as e:
            print(f"⚠️ kuroshio: {e}")

    if with_market and _HAS_MARKET:
        try:
            mp = market_price_fetcher.compute_features_for_catches(out)
            out = pd.concat([out, mp], axis=1)
        except Exception as e:
            print(f"⚠️ market: {e}")

    if with_upstream and _HAS_UPSTREAM:
        try:
            up_f = upstream_catch_fetcher.compute_features_for_catches(out)
            out = pd.concat([out, up_f], axis=1)
        except Exception as e:
            print(f"⚠️ upstream: {e}")

    if with_chlorophyll and _HAS_CHLOR:
        try:
            ch = chlorophyll_fetcher.compute_features_for_catches(out)
            out = pd.concat([out, ch], axis=1)
        except Exception as e:
            print(f"⚠️ chlorophyll: {e}")

    out = derived_features.add_catch_lags(out)
    out = derived_features.add_calendar_features(out)
    return out


def integrate_from_path(
    catches_path: str | Path,
    output: str | Path | None = None,
    with_rivers: bool = False,
    with_market: bool = True,
    with_upstream: bool = True,
    with_chlorophyll: bool = False,
) -> pd.DataFrame:
    catches = fishing_loader.load_auto(catches_path)
    integrated = integrate(
        catches,
        with_rivers=with_rivers,
        with_market=with_market,
        with_upstream=with_upstream,
        with_chlorophyll=with_chlorophyll,
    )
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            integrated.to_csv(out, index=False)
        else:
            integrated.to_parquet(out, index=False)
    return integrated


def build_inference_row(
    site: str,
    target_dt: pd.Timestamp,
    species: str,
    boat: str | None = None,
    anglers: int | None = None,
    target_species: str | None = None,
    tackle: str | None = None,
    departure_hour: int | None = None,
    catches_history: pd.DataFrame | None = None,
    padding_days: int = _DERIVED_PADDING_DAYS,
    with_rivers: bool = False,
    with_market: bool = True,
    with_upstream: bool = True,
    with_chlorophyll: bool = False,
) -> pd.DataFrame:
    if site not in config.SITES:
        raise ValueError(f"unknown site: {site}")
    site_obj = config.SITES[site]
    target_dt = pd.to_datetime(target_dt)
    if getattr(target_dt, "tz", None) is not None:
        target_dt = target_dt.tz_localize(None)

    d = target_dt.date()
    start = d - timedelta(days=padding_days)
    end = d

    weather = _weather_for(site, start, end)
    tide = _tide_for(site_obj.tide_port, start, end)

    row = pd.DataFrame([{
        "datetime": target_dt, "site": site, "species": species,
        "count": 0, "boat": boat, "anglers": anglers,
        "target_species": target_species or species, "tackle": tackle,
        "departure_hour": departure_hour if departure_hour is not None else target_dt.hour,
    }])

    if not weather.empty:
        row = pd.merge_asof(row.sort_values("datetime"), weather,
                            left_on="datetime", right_on="time",
                            direction="nearest", tolerance=pd.Timedelta(hours=1))
    if not tide.empty:
        row = pd.merge_asof(row.sort_values("datetime"), tide,
                            left_on="datetime", right_on="tide_time",
                            direction="nearest", tolerance=pd.Timedelta(hours=1))
    if with_rivers and _HAS_RIVER:
        try:
            rivers = river_flow_fetcher.fetch_site_rivers(site, start, end, kind="discharge")
            if not rivers.empty:
                rivers["time"] = pd.to_datetime(rivers["time"])
                if getattr(rivers["time"].dt, "tz", None) is not None:
                    rivers["time"] = rivers["time"].dt.tz_localize(None)
                row = pd.merge_asof(row.sort_values("datetime"),
                                    rivers.sort_values("time"),
                                    left_on="datetime", right_on="time",
                                    direction="backward", tolerance=pd.Timedelta(hours=24),
                                    suffixes=("", "_river"))
        except Exception as e:
            print(f"⚠️ river: {e}")

    row = astronomy.annotate(row, site_obj.latitude, site_obj.longitude)

    if _HAS_KUROSHIO:
        try:
            ks = kuroshio_fetcher.kuroshio_state_for_range(target_dt.date(), target_dt.date())
            if not ks.empty:
                row["kuroshio_state"] = ks["kuroshio_state"].iloc[0]
        except Exception:
            pass

    if with_market and _HAS_MARKET:
        try:
            row = pd.concat([row.reset_index(drop=True),
                            market_price_fetcher.compute_features_for_catches(row).reset_index(drop=True)],
                           axis=1)
        except Exception:
            pass

    if with_upstream and _HAS_UPSTREAM:
        try:
            row = pd.concat([row.reset_index(drop=True),
                            upstream_catch_fetcher.compute_features_for_catches(row).reset_index(drop=True)],
                           axis=1)
        except Exception:
            pass

    if with_chlorophyll and _HAS_CHLOR:
        try:
            row = pd.concat([row.reset_index(drop=True),
                            chlorophyll_fetcher.compute_features_for_catches(row).reset_index(drop=True)],
                           axis=1)
        except Exception:
            pass

    lag_values = derived_features.lookup_catch_lags_for_inference(
        site=site, boat=boat, species=species, target_dt=target_dt,
        catches_history=catches_history,
    )
    for k, v in lag_values.items():
        row[k] = v

    row = derived_features.add_calendar_features(row)
    return row


def _cli() -> None:
    parser = argparse.ArgumentParser(description="釣果に全外部データを結合")
    parser.add_argument("--catches", type=Path,
                        default=config.FISHING_DIR / "catches_template.csv")
    parser.add_argument("--out", type=Path,
                        default=config.INTEGRATED_DIR / "integrated.parquet")
    parser.add_argument("--with-rivers", action="store_true")
    parser.add_argument("--with-chlorophyll", action="store_true")
    args = parser.parse_args()

    df = integrate_from_path(args.catches, args.out,
                              with_rivers=args.with_rivers,
                              with_chlorophyll=args.with_chlorophyll)
    print(df.head())
    print(f"{len(df)} 行 -> {args.out}")
    print(f"列数: {len(df.columns)}")


if __name__ == "__main__":
    _cli()
