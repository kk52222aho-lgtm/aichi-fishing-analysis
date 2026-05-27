"""釣果予測モデル用の特徴量エンジニアリング。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import derived_features as df_mod

_BASE_NUMERIC_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl",
    "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "precipitation",
    "wave_height", "wave_period", "swell_wave_height", "swell_wave_period",
    "ocean_current_velocity", "sea_surface_temperature", "sea_level_height_msl",
]

RIVER_DERIVED_COLS = [
    "river_total_discharge",
    "river_total_discharge_3d",
    "river_total_discharge_7d",
    "river_flood_flag",
]

MARKET_COLS = [
    "market_price_lag1d", "market_price_lag7d", "market_price_ma7d",
    "market_volume_lag1d", "market_volume_ma7d", "market_price_change_7d",
]

UPSTREAM_COLS = [
    "upstream_cpue_lag1d", "upstream_cpue_lag3d", "upstream_cpue_lag7d",
    "upstream_cpue_ma7d", "upstream_reports_lag3d",
]

CHLOROPHYLL_COLS = [
    "chlor_a", "chlor_a_anomaly_30d", "chlor_a_d7d", "water_transparency",
]

NUMERIC_WEATHER_COLS = (
    _BASE_NUMERIC_WEATHER_COLS
    + list(df_mod.DERIVED_WEATHER_COLS)
    + list(df_mod.DERIVED_TIDE_COLS)
    + list(df_mod.DERIVED_CATCH_LAG_COLS)
    + ["is_long_weekend", "is_school_break", "hours_since_oshio", "days_since_new_moon"]
    + RIVER_DERIVED_COLS
    + MARKET_COLS
    + UPSTREAM_COLS
    + CHLOROPHYLL_COLS
)

CIRCULAR_COLS = [
    ("wind_direction_10m", "wind_dir"),
    ("wave_direction", "wave_dir"),
    ("ocean_current_direction", "current_dir"),
]

ASTRO_COLS = [
    "moon_age", "is_morning_mazume", "is_evening_mazume",
    "sunrise_hour", "sunset_hour",
]

TIDE_PHASE_VALUES = ["大潮", "中潮", "小潮", "長潮", "若潮"]
MOON_PHASE_VALUES = ["新月", "三日月", "上弦", "十三夜", "満月", "十六夜", "下弦", "有明"]
TACKLE_VALUES = ["サビキ", "ジギング", "エギング", "投げ", "落とし込み", "胴突き", "天秤", "ルアー", "その他"]
KUROSHIO_STATE_VALUES = ["大蛇行", "小蛇行A", "小蛇行B", "小蛇行C", "直進"]

SITE_CODES = list(config.SITES.keys())


def _circular_encode(deg, prefix):
    rad = np.deg2rad(deg.astype(float).fillna(0))
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(rad).values,
        f"{prefix}_cos": np.cos(rad).values,
    })


def _time_features(dt):
    dt = pd.to_datetime(dt)
    hour = dt.dt.hour + dt.dt.minute / 60.0
    return pd.DataFrame({
        "month": dt.dt.month.values,
        "day_of_year": dt.dt.dayofyear.values,
        "weekday": dt.dt.weekday.values,
        "is_weekend": (dt.dt.weekday >= 5).astype(int).values,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0).values,
        "hour_cos": np.cos(2 * np.pi * hour / 24.0).values,
        "doy_sin": np.sin(2 * np.pi * dt.dt.dayofyear / 365.25).values,
        "doy_cos": np.cos(2 * np.pi * dt.dt.dayofyear / 365.25).values,
    })


def _one_hot(series, values, prefix):
    return pd.DataFrame({
        f"{prefix}_{v}": (series.fillna("").astype(str) == v).astype(int).values
        for v in values
    })


def _site_one_hot(site): return _one_hot(site, SITE_CODES, "site")


def _tide_phase_features(df):
    if "tide_phase" in df.columns:
        return _one_hot(df["tide_phase"], TIDE_PHASE_VALUES, "tidep")
    return pd.DataFrame({f"tidep_{v}": [0] * len(df) for v in TIDE_PHASE_VALUES})


def _moon_phase_features(df):
    if "moon_phase" in df.columns:
        return _one_hot(df["moon_phase"], MOON_PHASE_VALUES, "moon")
    return pd.DataFrame({f"moon_{v}": [0] * len(df) for v in MOON_PHASE_VALUES})


def _tackle_features(df):
    if "tackle" in df.columns:
        return _one_hot(df["tackle"], TACKLE_VALUES, "tackle")
    return pd.DataFrame({f"tackle_{v}": [0] * len(df) for v in TACKLE_VALUES})


def _kuroshio_features(df):
    if "kuroshio_state" in df.columns:
        return _one_hot(df["kuroshio_state"], KUROSHIO_STATE_VALUES, "kuro")
    return pd.DataFrame({f"kuro_{v}": [0] * len(df) for v in KUROSHIO_STATE_VALUES})


def _booking_features(df):
    out = pd.DataFrame(index=df.index)
    out["anglers"] = pd.to_numeric(df.get("anglers"), errors="coerce").fillna(0).astype(float).values
    out["departure_hour"] = pd.to_numeric(df.get("departure_hour"), errors="coerce").fillna(6).astype(float).values
    out["target_match"] = (
        df.get("species", pd.Series([""] * len(df))).astype(str)
        == df.get("target_species", pd.Series([""] * len(df))).astype(str)
    ).astype(int).values
    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    parts.append(_time_features(df["datetime"]))
    parts.append(_site_one_hot(df["site"].fillna("")))
    parts.append(_booking_features(df))

    for col in NUMERIC_WEATHER_COLS:
        if col in df.columns:
            parts.append(df[[col]].astype(float).reset_index(drop=True))
        else:
            parts.append(pd.DataFrame({col: [np.nan] * len(df)}))

    for src_col, prefix in CIRCULAR_COLS:
        if src_col in df.columns:
            parts.append(_circular_encode(df[src_col], prefix))
        else:
            parts.append(pd.DataFrame({f"{prefix}_sin": [0.0] * len(df), f"{prefix}_cos": [0.0] * len(df)}))

    if "tide_cm" in df.columns:
        parts.append(df[["tide_cm"]].astype(float).reset_index(drop=True))
    else:
        parts.append(pd.DataFrame({"tide_cm": [np.nan] * len(df)}))

    for col in ASTRO_COLS:
        if col in df.columns:
            parts.append(df[[col]].astype(float).reset_index(drop=True))
        else:
            parts.append(pd.DataFrame({col: [np.nan] * len(df)}))

    parts.append(_tide_phase_features(df))
    parts.append(_moon_phase_features(df))
    parts.append(_tackle_features(df))
    parts.append(_kuroshio_features(df))

    feats = pd.concat(parts, axis=1)
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)
    return feats


def feature_columns(df):
    return list(build_features(df.head(1)).columns)
