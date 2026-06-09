"""プロジェクト全体で共有する定数とパス定義。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"
FISHING_DIR = DATA_DIR / "fishing_logs"
INTEGRATED_DIR = DATA_DIR / "integrated"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = DATA_DIR / "models"
TIDE_DIR = DATA_DIR / "tide"

# YOLO 統合モデル (GenesisEngine-v6 でブート、釣果写真からの自動カウント用)
YOLO_MODELS_DIR = MODELS_DIR / "yolo_unified"
YOLO_DEFAULT_WEIGHTS = YOLO_MODELS_DIR / "best.pt"

for _d in (WEATHER_DIR, FISHING_DIR, INTEGRATED_DIR, IMAGES_DIR, MODELS_DIR,
           TIDE_DIR, YOLO_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Site:
    code: str
    name_ja: str
    latitude: float
    longitude: float
    tide_port: str  # JMA 潮汐表の港コード or ローカルCSV名のキー


# JMA 港コード一覧: https://www.data.jma.go.jp/kaiyou/db/tide/suisan/listall.php
# 不明な港は最寄りの観測点で代用（精度は出るので問題なし）。
SITES: dict[str, Site] = {
    "shinojima":     Site("shinojima",     "篠島",           34.6700, 136.9700, "IO"),  # 伊良湖で代用
    "morozaki":      Site("morozaki",      "師崎",           34.7100, 136.9800, "IO"),
    "utsumi_shinko": Site("utsumi_shinko", "内海新港",       34.7000, 136.8800, "IO"),  # 南知多町（伊勢湾側）
    "irago":         Site("irago",         "伊良湖",         34.5800, 137.0200, "IO"),
    "mikawa_bay":    Site("mikawa_bay",    "三河湾中央",     34.7500, 137.0500, "IO"),
    "chita_tip":     Site("chita_tip",     "知多半島先端",   34.6900, 136.9700, "NA"),  # 名古屋
    "akabane":       Site("akabane",       "赤羽根港",       34.6131, 137.2667, "IO"),  # 渥美半島・遠州灘側
    "toyohama":      Site("toyohama",      "豊浜港",         34.7117, 136.9061, "IO"),  # 南知多町豊浜（師崎の北西側）
}

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

FORECAST_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
]

MARINE_HOURLY_VARS = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "swell_wave_height",
    "swell_wave_period",
    "ocean_current_velocity",
    "ocean_current_direction",
    "sea_surface_temperature",
    "sea_level_height_msl",
]

TIMEZONE = "Asia/Tokyo"

# 月齢から潮回りを判定する閾値（簡易版）
TIDE_PHASE_LABELS = ["大潮", "中潮", "小潮", "長潮", "若潮"]
