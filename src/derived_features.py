"""派生特徴量。

既存データ（気象・潮汐・釣果ログ）から、時系列コンテキストを使う派生特徴量を計算する。
すべて「過去データのみ参照」（未来参照禁止）なので、出船前推論でも使える。

設計:
  - enrich_weather(weather_df): 気圧変化/水温偏差/降水積算/波合成/風向安定性 等
  - enrich_tide(tide_df): 潮位変化速度/潮位差/潮目フラグ
  - add_catch_lags(catches_df): 同船・同魚種の直近N日前釣果（自己ラグ）
  - add_calendar_features(df): 連休/学校休暇/大潮からの経過時間

呼び出し位置（data_integrator.py で）:
  weather = enrich_weather(_weather_for(...))     # merge_asof の前
  tide    = enrich_tide(_tide_for(...))            # merge_asof の前
  → 結合後に add_catch_lags / add_calendar_features

推論時:
  build_inference_row では weather/tide を 14日 padding で取得して
  最新時刻まで含めた window で enrich してから merge_asof で1点抜き出す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ===================================================================
# 公開する派生列名（features.py から参照されることを想定）
# ===================================================================

DERIVED_WEATHER_COLS: list[str] = [
    # 気圧変化
    "pressure_d6h",
    "pressure_d24h",
    "pressure_d48h",
    "hours_since_pressure_min_7d",
    "pressure_anomaly_7d",
    # 水温
    "sst_d24h",
    "sst_d7d",
    "sst_anomaly_7d",
    # 気温
    "temp_d24h",
    # 降水積算
    "precip_24h",
    "precip_3d",
    "precip_7d",
    # 波・うねり
    "wave_severity",
    "flat_sea_flag",
    "rough_sea_flag",
    "swell_residual",
    # 風
    "wind_dir_stability_6h",
    "wind_speed_d6h",
    "wind_speed_std_6h",
    # 海面高度
    "ssh_d24h",
    "ssh_anomaly_7d",
    # 海流
    "current_speed_avg6h",
]

DERIVED_TIDE_COLS: list[str] = [
    "tide_dcm_dh",
    "tide_dcm_dh_abs",
    "tide_range_24h",
    "tide_running_avg_24h",
]

DERIVED_CATCH_LAG_COLS: list[str] = [
    "count_lag1d",
    "count_lag3d",
    "count_lag7d",
    "count_lag14d",
    "count_avg7d",
    "count_avg14d",
    "days_since_last_record",
    "boat_records_30d",
]

DERIVED_CALENDAR_COLS: list[str] = [
    "is_long_weekend",
    "is_school_break",
    "hours_since_oshio",
    "days_since_new_moon",
]

# 同魚種 × 同月±1 の過去 trip 上位/中央値（マダイ等 wide-distribution 魚種で
# 「大漁日」の signal を統計モデルに与える。LLM 側 Step 3.3 と同じ発想）
DERIVED_SIMILAR_PAST_COLS: list[str] = [
    "past_max_same_month",
    "past_p75_same_month",
    "past_median_same_month",
    "past_n_same_month",
]

# 同魚種の直近 7日/30日 max/mean — 「今その魚が活きてるか」signal
# 船宿を跨いで集計するので「地域全体の今週の調子」が取れる
DERIVED_SPECIES_RECENT_COLS: list[str] = [
    "species_recent7d_max",
    "species_recent7d_mean",
    "species_recent7d_n",
    "species_recent30d_max",
    "species_recent30d_mean",
    "species_recent30d_n",
]


# ===================================================================
# 気象派生
# ===================================================================

def enrich_weather(df: pd.DataFrame) -> pd.DataFrame:
    """weather DataFrame に派生列を追加。

    入力前提: `time` 列でソート可能、毎時刻分解能（Open-Meteo の hourly 出力）。
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = df.sort_values("time").reset_index(drop=True).copy()

    # ---- 気圧 ----
    if "pressure_msl" in df.columns:
        p = df["pressure_msl"].astype(float)
        df["pressure_d6h"] = p.diff(6)
        df["pressure_d24h"] = p.diff(24)
        df["pressure_d48h"] = p.diff(48)
        p_avg7d = p.rolling(24 * 7, min_periods=24).mean()
        df["pressure_anomaly_7d"] = p - p_avg7d
        df["hours_since_pressure_min_7d"] = _hours_since_rolling_min(
            p, window=24 * 7, min_periods=12
        )

    # ---- 気温 ----
    if "temperature_2m" in df.columns:
        df["temp_d24h"] = df["temperature_2m"].astype(float).diff(24)

    # ---- 水温 ----
    if "sea_surface_temperature" in df.columns:
        sst = df["sea_surface_temperature"].astype(float)
        df["sst_d24h"] = sst.diff(24)
        df["sst_d7d"] = sst.diff(24 * 7)
        sst_avg7d = sst.rolling(24 * 7, min_periods=24).mean()
        df["sst_anomaly_7d"] = sst - sst_avg7d

    # ---- 降水積算 ----
    if "precipitation" in df.columns:
        precip = df["precipitation"].astype(float).fillna(0)
        df["precip_24h"] = precip.rolling(24, min_periods=1).sum()
        df["precip_3d"] = precip.rolling(24 * 3, min_periods=1).sum()
        df["precip_7d"] = precip.rolling(24 * 7, min_periods=1).sum()

    # ---- 波・うねり ----
    if {"wave_height", "wave_period"}.issubset(df.columns):
        wh = df["wave_height"].astype(float)
        wp = df["wave_period"].astype(float)
        df["wave_severity"] = wh * wp
        df["flat_sea_flag"] = (wh < 0.5).astype(int)
        df["rough_sea_flag"] = (wh >= 1.5).astype(int)

    if {"swell_wave_height", "wave_height"}.issubset(df.columns):
        df["swell_residual"] = (
            df["swell_wave_height"].astype(float) - df["wave_height"].astype(float)
        )

    # ---- 風 ----
    if "wind_direction_10m" in df.columns:
        rad = np.deg2rad(df["wind_direction_10m"].astype(float))
        sin_r = np.sin(rad)
        cos_r = np.cos(rad)
        mean_sin = sin_r.rolling(6, min_periods=2).mean()
        mean_cos = cos_r.rolling(6, min_periods=2).mean()
        df["wind_dir_stability_6h"] = np.sqrt(mean_sin ** 2 + mean_cos ** 2)

    if "wind_speed_10m" in df.columns:
        ws = df["wind_speed_10m"].astype(float)
        df["wind_speed_d6h"] = ws.diff(6)
        df["wind_speed_std_6h"] = ws.rolling(6, min_periods=2).std()

    # ---- 海面高度 ----
    if "sea_level_height_msl" in df.columns:
        ssh = df["sea_level_height_msl"].astype(float)
        df["ssh_d24h"] = ssh.diff(24)
        ssh_avg7d = ssh.rolling(24 * 7, min_periods=24).mean()
        df["ssh_anomaly_7d"] = ssh - ssh_avg7d

    # ---- 海流 ----
    if "ocean_current_velocity" in df.columns:
        df["current_speed_avg6h"] = (
            df["ocean_current_velocity"].astype(float).rolling(6, min_periods=2).mean()
        )

    return df


def _hours_since_rolling_min(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """rolling window 内で「最小値からの経過時間（行数）」を返す。"""
    def _f(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.all(np.isnan(arr)):
            return np.nan
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        idx = int(np.nanargmin(arr))
        return float(len(arr) - 1 - idx)
    return s.rolling(window, min_periods=min_periods).apply(_f, raw=True)


# ===================================================================
# 潮汐派生
# ===================================================================

def enrich_tide(df: pd.DataFrame) -> pd.DataFrame:
    """tide DataFrame に派生列を追加。"""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = df.sort_values("datetime").reset_index(drop=True).copy()

    if "tide_cm" in df.columns:
        t = df["tide_cm"].astype(float)
        df["tide_dcm_dh"] = t.diff(1)
        df["tide_dcm_dh_abs"] = df["tide_dcm_dh"].abs()
        df["tide_range_24h"] = (
            t.rolling(24, min_periods=12).max() - t.rolling(24, min_periods=12).min()
        )
        df["tide_running_avg_24h"] = t.rolling(24, min_periods=12).mean()

    return df


# ===================================================================
# 自己ラグ（同船・同魚種の直近釣果）
# ===================================================================

def add_catch_lags(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("site", "boat", "species"),
    lag_days: tuple[int, ...] = (1, 3, 7, 14),
) -> pd.DataFrame:
    """同一 group の直近N日前釣果を自己ラグとして付与。

    複数記録/日は合計してから shift。当日の count は使わない（未来参照防止）。
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if "count" not in df.columns:
        return df

    df = df.sort_values("datetime").reset_index(drop=True).copy()
    df["_date"] = pd.to_datetime(df["datetime"]).dt.normalize()

    grp_list = list(group_cols)
    valid = df[grp_list].notna().all(axis=1)
    sub = df.loc[valid].copy()

    if sub.empty:
        for lag in lag_days:
            df[f"count_lag{lag}d"] = np.nan
        df["count_avg7d"] = np.nan
        df["count_avg14d"] = np.nan
        df["days_since_last_record"] = np.nan
        df["boat_records_30d"] = 0.0
        df = df.drop(columns=["_date"])
        return df

    daily = (
        sub.groupby(grp_list + ["_date"], dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values(grp_list + ["_date"])
        .reset_index(drop=True)
    )

    for lag in lag_days:
        col = f"count_lag{lag}d"
        daily[col] = daily.groupby(grp_list)["count"].shift(lag)

    shifted = daily.groupby(grp_list)["count"].shift(1)
    daily["count_avg7d"] = (
        shifted.groupby([daily[c] for c in grp_list])
        .rolling(7, min_periods=1).mean()
        .reset_index(level=list(range(len(grp_list))), drop=True)
    )
    daily["count_avg14d"] = (
        shifted.groupby([daily[c] for c in grp_list])
        .rolling(14, min_periods=1).mean()
        .reset_index(level=list(range(len(grp_list))), drop=True)
    )

    keep_cols = grp_list + ["_date"] + [f"count_lag{l}d" for l in lag_days] + ["count_avg7d", "count_avg14d"]
    df = df.merge(daily[keep_cols], on=grp_list + ["_date"], how="left")

    df_sorted = df.sort_values(grp_list + ["datetime"]).reset_index()
    prev_dt = df_sorted.groupby(grp_list)["datetime"].shift(1)
    df_sorted["days_since_last_record"] = (
        (pd.to_datetime(df_sorted["datetime"]) - pd.to_datetime(prev_dt))
        .dt.total_seconds() / 86400
    )
    df = (
        df_sorted.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    )

    df["boat_records_30d"] = _rolling_count_per_group(
        df, group_cols=("site", "boat"), window_days=30
    )

    df = df.drop(columns=["_date"])
    return df


def _rolling_count_per_group(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    window_days: int,
) -> pd.Series:
    """group ごとの「過去window_days日以内の記録数（当日含まない）」を返す。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    dt = pd.to_datetime(df["datetime"])
    grouped = df.groupby(list(group_cols), dropna=False).groups
    for keys, idx_list in grouped.items():
        sub_dt = dt.loc[idx_list].sort_values()
        sub_idx = sub_dt.index
        sub_vals = sub_dt.values.astype("datetime64[ns]")
        counts = np.zeros(len(sub_vals), dtype=float)
        for i, t in enumerate(sub_vals):
            cutoff = t - np.timedelta64(window_days, "D")
            counts[i] = float(((sub_vals[:i] >= cutoff) & (sub_vals[:i] < t)).sum())
        out.loc[sub_idx] = counts
    return out.fillna(0.0)


# ===================================================================
# 暦・天文派生
# ===================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """暦系の派生特徴量。"""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    month = dt.dt.month
    day = dt.dt.day
    wd = dt.dt.weekday

    df["is_long_weekend"] = (
        (wd == 5) | (wd == 6) | ((wd == 0) & day.isin([1, 2, 3]))
    ).astype(int)

    df["is_school_break"] = (
        ((month == 7) & (day >= 20))
        | (month == 8)
        | ((month == 12) & (day >= 25))
        | ((month == 1) & (day <= 7))
        | ((month == 3) & (day >= 25))
        | ((month == 4) & (day <= 7))
    ).astype(int)

    if "tide_phase" in df.columns:
        df_s = df.sort_values("datetime").reset_index().copy()
        is_oshio = (df_s["tide_phase"].astype(str) == "大潮").astype(int)
        last_oshio = df_s["datetime"].where(is_oshio == 1).ffill()
        df_s["hours_since_oshio"] = (
            (pd.to_datetime(df_s["datetime"]) - pd.to_datetime(last_oshio))
            .dt.total_seconds() / 3600
        )
        df = (
            df_s.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
        )
    else:
        df["hours_since_oshio"] = np.nan

    if "moon_age" in df.columns:
        df["days_since_new_moon"] = df["moon_age"].astype(float)
    else:
        df["days_since_new_moon"] = np.nan

    return df


# ===================================================================
# 推論時用ヘルパ: 過去釣果ログを引いてラグ列を埋める
# ===================================================================

def lookup_catch_lags_for_inference(
    site: str,
    boat: str | None,
    species: str,
    target_dt: pd.Timestamp,
    catches_history: pd.DataFrame | None,
    lag_days: tuple[int, ...] = (1, 3, 7, 14),
) -> dict[str, float]:
    """推論行に埋める catch lag 値を計算。

    catches_history は (datetime, site, boat, species, count) を含む過去ログ。
    boat が None なら site×species だけで集約。
    """
    out: dict[str, float] = {f"count_lag{l}d": np.nan for l in lag_days}
    out["count_avg7d"] = np.nan
    out["count_avg14d"] = np.nan
    out["days_since_last_record"] = np.nan
    out["boat_records_30d"] = 0.0

    if catches_history is None or catches_history.empty:
        return out

    h = catches_history.copy()
    h["datetime"] = pd.to_datetime(h["datetime"])
    mask = (h["site"] == site) & (h["species"] == species)
    if boat:
        mask = mask & (h["boat"].astype(str) == str(boat))
    h = h.loc[mask & (h["datetime"] < target_dt)]
    if h.empty:
        return out

    h["_date"] = h["datetime"].dt.normalize()
    daily = h.groupby("_date")["count"].sum().sort_index()

    target_date = pd.to_datetime(target_dt).normalize()
    for lag in lag_days:
        d = target_date - pd.Timedelta(days=lag)
        out[f"count_lag{lag}d"] = float(daily.get(d, np.nan))

    last7 = daily[daily.index >= target_date - pd.Timedelta(days=7)]
    last14 = daily[daily.index >= target_date - pd.Timedelta(days=14)]
    out["count_avg7d"] = float(last7.mean()) if not last7.empty else np.nan
    out["count_avg14d"] = float(last14.mean()) if not last14.empty else np.nan

    last_record_dt = h["datetime"].max()
    out["days_since_last_record"] = float(
        (target_dt - last_record_dt).total_seconds() / 86400
    )

    if boat:
        h30 = catches_history.copy()
        h30["datetime"] = pd.to_datetime(h30["datetime"])
        m30 = (
            (h30["site"] == site)
            & (h30["boat"].astype(str) == str(boat))
            & (h30["datetime"] < target_dt)
            & (h30["datetime"] >= target_dt - pd.Timedelta(days=30))
        )
        out["boat_records_30d"] = float(m30.sum())

    return out


# ===================================================================
# 同魚種 × 同月±1 の過去 trip 統計（Step 3.3 LLM 側と同じ signal を統計モデルへ）
# ===================================================================

def add_similar_past_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    label_col: str = "top_per_angler",
    species_col: str = "species",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """各 row の過去 (同魚種 × 同月±1) trip から max/p75/median/n を計算して列追加。

    walk_forward の各 trip i では、過去 trip 0..i-1 のみが参照可能になるよう、
    history_df を別途渡せる（渡さない場合は df 自身を history として使う）。

    Args:
        df: 行ごとに特徴量を追加したい DataFrame（学習でも推論でも可）
        history_df: 過去参照ソース。None なら df 自身。各行 i では
                    history_df の中で datetime < row[i].datetime かつ
                    同魚種 + 月±1 の行を集計する。
        label_col: 集計対象列（default: top_per_angler）

    Returns:
        df の copy に [past_max_same_month, past_p75_same_month,
        past_median_same_month, past_n_same_month] を追加したもの。
        該当 0 件の行は NaN（後段の features.build_features で median fillna）。
    """
    out = df.copy().reset_index(drop=True)
    hist = (history_df if history_df is not None else df).copy()

    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    hist[datetime_col] = pd.to_datetime(hist[datetime_col], errors="coerce")
    if getattr(out[datetime_col].dt, "tz", None) is not None:
        out[datetime_col] = out[datetime_col].dt.tz_localize(None)
    if getattr(hist[datetime_col].dt, "tz", None) is not None:
        hist[datetime_col] = hist[datetime_col].dt.tz_localize(None)

    if label_col not in hist.columns:
        # ラベル列が無ければ全部 NaN
        for c in ("past_max_same_month", "past_p75_same_month",
                  "past_median_same_month", "past_n_same_month"):
            out[c] = np.nan
        return out

    hist = hist.dropna(subset=[label_col])
    hist["_month"] = hist[datetime_col].dt.month

    max_vals, p75_vals, med_vals, n_vals = [], [], [], []
    for i in range(len(out)):
        target_dt = out[datetime_col].iloc[i]
        target_species = out[species_col].iloc[i] if species_col in out.columns else None
        if pd.isna(target_dt) or target_species is None:
            max_vals.append(np.nan); p75_vals.append(np.nan)
            med_vals.append(np.nan); n_vals.append(0)
            continue

        target_month = int(target_dt.month)
        months = {((target_month - 2) % 12) + 1, target_month, (target_month % 12) + 1}

        mask = (
            (hist[datetime_col] < target_dt)
            & (hist[species_col] == target_species)
            & (hist["_month"].isin(months))
        )
        vals = pd.to_numeric(hist.loc[mask, label_col], errors="coerce").dropna()
        if len(vals) >= 1:
            max_vals.append(float(vals.max()))
            p75_vals.append(float(vals.quantile(0.75)))
            med_vals.append(float(vals.median()))
            n_vals.append(int(len(vals)))
        else:
            max_vals.append(np.nan); p75_vals.append(np.nan)
            med_vals.append(np.nan); n_vals.append(0)

    out["past_max_same_month"] = max_vals
    out["past_p75_same_month"] = p75_vals
    out["past_median_same_month"] = med_vals
    out["past_n_same_month"] = n_vals
    return out


def add_species_recent_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    label_col: str = "top_per_angler",
    species_col: str = "species",
    datetime_col: str = "datetime",
    windows: tuple[int, ...] = (7, 30),
) -> pd.DataFrame:
    """各 row の同魚種 × 直近 N 日（船宿を跨ぐ）max/mean/n を列追加。

    add_similar_past_features は同月±1 で「seasonal な大漁日」を捕捉する一方、
    こちらは「今 hot な species か」を捕捉する recency signal。マダイ等の
    spawning peak で他の船宿が大漁出したら今週中はうちも期待できる、等。

    Args と past_only ガードは add_similar_past_features と同じ。
    """
    out = df.copy().reset_index(drop=True)
    hist = (history_df if history_df is not None else df).copy()

    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    hist[datetime_col] = pd.to_datetime(hist[datetime_col], errors="coerce")
    if getattr(out[datetime_col].dt, "tz", None) is not None:
        out[datetime_col] = out[datetime_col].dt.tz_localize(None)
    if getattr(hist[datetime_col].dt, "tz", None) is not None:
        hist[datetime_col] = hist[datetime_col].dt.tz_localize(None)

    if label_col not in hist.columns:
        for w in windows:
            for suf in ("max", "mean", "n"):
                out[f"species_recent{w}d_{suf}"] = np.nan if suf != "n" else 0
        return out

    hist = hist.dropna(subset=[label_col])

    for w in windows:
        max_vals, mean_vals, n_vals = [], [], []
        for i in range(len(out)):
            target_dt = out[datetime_col].iloc[i]
            target_species = out[species_col].iloc[i] if species_col in out.columns else None
            if pd.isna(target_dt) or target_species is None:
                max_vals.append(np.nan); mean_vals.append(np.nan); n_vals.append(0)
                continue
            window_start = target_dt - pd.Timedelta(days=w)
            mask = (
                (hist[datetime_col] >= window_start)
                & (hist[datetime_col] < target_dt)
                & (hist[species_col] == target_species)
            )
            vals = pd.to_numeric(hist.loc[mask, label_col], errors="coerce").dropna()
            if len(vals) >= 1:
                max_vals.append(float(vals.max()))
                mean_vals.append(float(vals.mean()))
                n_vals.append(int(len(vals)))
            else:
                max_vals.append(np.nan); mean_vals.append(np.nan); n_vals.append(0)
        out[f"species_recent{w}d_max"] = max_vals
        out[f"species_recent{w}d_mean"] = mean_vals
        out[f"species_recent{w}d_n"] = n_vals
    return out
