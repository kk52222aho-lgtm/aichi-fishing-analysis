"""FastAPI サーバー — 釣り船の釣果予測をWeb/アプリに配信。

主な動線:
    GET /predict?site=&species=&date=&hour=&boat=&anglers=&tackle=&target_species=
    GET /sites                           観測地点一覧
    GET /species                         学習済み魚種一覧
    GET /weather?site=&start=&end=       気象データ
    GET /tide?port=&year=                潮汐データ
    GET /catches                         過去の統合データ（学習データ確認）
    GET /report.png                      可視化レポート画像
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, predictor, tide_fetcher, visualizer, weather_fetcher  # noqa: E402

app = FastAPI(
    title="Aichi Fishing Catch Prediction API",
    version="0.2.0",
    description="愛知近海の釣り船向け釣果予測API（実釣果は YOLO で計数、予測は気象/海象/潮汐/天文/予約メタから）",
)


class CatchIn(BaseModel):
    datetime: str
    site: str
    species: str
    count: Optional[int] = None
    boat: Optional[str] = None
    anglers: Optional[int] = None
    target_species: Optional[str] = None
    tackle: Optional[str] = None
    departure_hour: Optional[int] = None
    total_weight_g: Optional[float] = None
    avg_length_cm: Optional[float] = None
    angler: Optional[str] = None
    notes: Optional[str] = None
    image_path: Optional[str] = None


@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "endpoints": ["/predict", "/sites", "/species", "/weather", "/tide", "/catches", "/report.png"],
    }


@app.get("/sites")
def sites() -> list[dict]:
    return [
        {"code": s.code, "name_ja": s.name_ja, "lat": s.latitude, "lon": s.longitude, "tide_port": s.tide_port}
        for s in config.SITES.values()
    ]


@app.get("/species")
def species() -> list[dict]:
    return predictor.list_models()


@app.get("/predict")
def predict_endpoint(
    site: str = Query(..., description="地点コード"),
    species: str = Query(..., description="魚種"),
    date_: date = Query(..., alias="date", description="予測対象日（YYYY-MM-DD）"),
    hour: int = Query(6, ge=0, le=23, description="想定出船時刻"),
    boat: Optional[str] = Query(None, description="船宿/船名"),
    anglers: Optional[int] = Query(None, ge=1, description="乗船人数"),
    tackle: Optional[str] = Query(None, description="仕掛け"),
    target_species: Optional[str] = Query(None, description="その日の狙い魚種"),
) -> dict:
    if site not in config.SITES:
        raise HTTPException(404, f"unknown site: {site}")
    try:
        return predictor.predict(
            site=site, species=species, target_date=date_, hour=hour,
            boat=boat, anglers=anglers, tackle=tackle, target_species=target_species,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"prediction failed: {e}")


@app.get("/weather")
def weather(site: str = Query(...), start: date = Query(...), end: date = Query(...)) -> list[dict]:
    if site not in config.SITES:
        raise HTTPException(404, f"unknown site: {site}")
    df = weather_fetcher.fetch_and_cache(site, start, end)
    return df.to_dict(orient="records")


@app.get("/tide")
def tide(port: str = Query(...), year: int = Query(...)) -> list[dict]:
    df = tide_fetcher.fetch_tide_year(port, year)
    return df.to_dict(orient="records")


@app.get("/catches")
def catches() -> list[dict]:
    p = config.INTEGRATED_DIR / "integrated.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    return df.to_dict(orient="records")


@app.post("/catches")
def add_catch(payload: CatchIn) -> dict:
    target = config.FISHING_DIR / "catches.csv"
    df_new = pd.DataFrame([payload.model_dump()])
    if target.exists():
        df = pd.concat([pd.read_csv(target), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(target, index=False)
    return {"status": "ok", "rows": len(df)}


@app.get("/report.png")
def report():
    p = config.INTEGRATED_DIR / "integrated.parquet"
    if not p.exists():
        raise HTTPException(404, "統合データがありません。data_integratorを先に実行してください。")
    df = pd.read_parquet(p)
    out = config.INTEGRATED_DIR / "report.png"
    visualizer.render_report(df, out)
    return FileResponse(out, media_type="image/png")
