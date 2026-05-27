"""天文関連の特徴量。

提供する量:
  - 日の出 / 日の入り / 薄明（天文・常用）
  - 月齢、月相
  - 潮回り（大潮/中潮/小潮/長潮/若潮）— 月齢ベースの簡易判定
  - 朝マヅメ・夕マヅメ判定（日出±1h, 日没±1h）

外部API不要。すべてローカル計算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import pandas as pd

JST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class SunMoon:
    sunrise: datetime | None
    sunset: datetime | None
    dawn_civil: datetime | None
    dusk_civil: datetime | None
    moon_age: float
    moon_phase: str  # 新月/三日月/上弦/十三夜/満月/十六夜/下弦/有明
    tide_phase: str  # 大潮/中潮/小潮/長潮/若潮


_MOON_REF = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)  # 既知の新月
_SYNODIC_MONTH = 29.530588853


def moon_age(d: date | datetime) -> float:
    """新月起算の月齢（0〜29.53）。"""
    if isinstance(d, datetime):
        dt = d if d.tzinfo else d.replace(tzinfo=JST)
    else:
        dt = datetime.combine(d, time(12, 0), tzinfo=JST)
    delta = (dt - _MOON_REF).total_seconds() / 86400.0
    return delta % _SYNODIC_MONTH


def moon_phase_label(age: float) -> str:
    if age < 1.5 or age >= 28.0:
        return "新月"
    if age < 6.0:
        return "三日月"
    if age < 8.5:
        return "上弦"
    if age < 13.0:
        return "十三夜"
    if age < 16.0:
        return "満月"
    if age < 18.5:
        return "十六夜"
    if age < 23.5:
        return "下弦"
    return "有明"


def tide_phase_label(age: float) -> str:
    """月齢から潮回りを返す（簡易版）。

    新月（age≈0）と満月（age≈14.75）付近が大潮、上弦・下弦付近が小潮。
    潮回り暦と完全一致はしないが、特徴量としては十分機能する。
    """
    a = age % _SYNODIC_MONTH
    if a < 2.0 or a >= 27.5:
        return "大潮"
    if 13.0 <= a < 16.5:
        return "大潮"
    if 6.0 <= a < 8.5 or 20.5 <= a < 23.0:
        return "小潮"
    if 8.5 <= a < 9.5 or 23.0 <= a < 24.0:
        return "長潮"
    if 9.5 <= a < 10.5 or 24.0 <= a < 25.0:
        return "若潮"
    return "中潮"


def _sun_times_astral(lat: float, lon: float, target_date: date) -> dict[str, datetime | None]:
    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError:
        return {"sunrise": None, "sunset": None, "dawn": None, "dusk": None}

    loc = LocationInfo(latitude=lat, longitude=lon, timezone="Asia/Tokyo")
    try:
        s = sun(loc.observer, date=target_date, tzinfo=JST)
        return {
            "sunrise": s["sunrise"],
            "sunset": s["sunset"],
            "dawn": s["dawn"],
            "dusk": s["dusk"],
        }
    except Exception:
        return {"sunrise": None, "sunset": None, "dawn": None, "dusk": None}


def sun_moon(lat: float, lon: float, target_date: date) -> SunMoon:
    s = _sun_times_astral(lat, lon, target_date)
    age = moon_age(target_date)
    return SunMoon(
        sunrise=s["sunrise"],
        sunset=s["sunset"],
        dawn_civil=s["dawn"],
        dusk_civil=s["dusk"],
        moon_age=age,
        moon_phase=moon_phase_label(age),
        tide_phase=tide_phase_label(age),
    )


def is_mazume(dt: datetime, sunrise: datetime | None, sunset: datetime | None,
              window_hours: float = 1.0) -> tuple[int, int]:
    """(朝マヅメ, 夕マヅメ) のフラグを返す。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    morning = 0
    evening = 0
    if sunrise and abs((dt - sunrise).total_seconds()) <= window_hours * 3600:
        morning = 1
    if sunset and abs((dt - sunset).total_seconds()) <= window_hours * 3600:
        evening = 1
    return morning, evening


def annotate(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """DataFrame の各行（datetime列を持つ）に天文・潮回り列を付加。"""
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    cache: dict[date, SunMoon] = {}
    rows = []
    for ts in dt:
        d = ts.date()
        if d not in cache:
            cache[d] = sun_moon(lat, lon, d)
        sm = cache[d]
        morning, evening = is_mazume(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                                     sm.sunrise, sm.sunset)
        rows.append({
            "moon_age": sm.moon_age,
            "moon_phase": sm.moon_phase,
            "tide_phase": sm.tide_phase,
            "is_morning_mazume": morning,
            "is_evening_mazume": evening,
            "sunrise_hour": (sm.sunrise.hour + sm.sunrise.minute / 60.0) if sm.sunrise else float("nan"),
            "sunset_hour": (sm.sunset.hour + sm.sunset.minute / 60.0) if sm.sunset else float("nan"),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
