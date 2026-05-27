"""潮上流（黒潮上流）の釣果データ取得 — 回遊魚予測の革命的シグナル。

考え方:
    黒潮系の回遊魚（カツオ、マグロ、サワラ、ブリ、シイラ等）は西から東へ移動する。
    遠州灘・伊勢湾外に来る前に、高知→和歌山→三重で先に釣れる。
    その地域の釣果（CPUE: 1人1日あたりの匹数）を 1〜7日ラグ特徴量化する。

データソース:
    1. Anglers https://anglers.jp/  — 公式 API 無いので HTML スクレイプ
    2. 釣り船予約サイト各社（フィッシングジャパン, 釣割, etc）
    3. つり人オンライン、釣りビジョン
    4. SNS (X) ハッシュタグ集計

実装:
    - 外部 CSV キャッシュを前提（手動入力 or スクレイプ）
    - カラム: date, region, species, cpue (人/竿あたり匹数), n_reports
    - 釣果datetime → 1日, 3日, 7日 ラグ特徴量を計算

地域定義（黒潮上流順）:
    kochi    : 高知（足摺岬・室戸岬）
    wakayama : 和歌山（潮岬・串本）
    mie      : 三重（尾鷲・志摩・大王崎）
    → 愛知 (本予測対象)
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

UPSTREAM_DIR = config.DATA_DIR / "upstream_catch"
UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
UPSTREAM_CSV = UPSTREAM_DIR / "reports.csv"

UPSTREAM_REGIONS = ["kochi", "wakayama", "mie"]

REGION_WEIGHTS = {
    "kochi": 0.3,
    "wakayama": 0.7,
    "mie": 1.0,
}

PELAGIC_SPECIES = [
    "カツオ", "シイラ", "ブリ", "ハマチ", "メジロ",
    "サワラ", "キハダ", "ヨコワ", "マグロ", "アジ",
    "サバ", "イワシ",
]


def load_reports() -> pd.DataFrame:
    if not UPSTREAM_CSV.exists():
        return pd.DataFrame(columns=["date", "region", "species", "cpue", "n_reports"])
    df = pd.read_csv(UPSTREAM_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def save_reports(df: pd.DataFrame) -> Path:
    df.to_csv(UPSTREAM_CSV, index=False)
    return UPSTREAM_CSV


def compute_features_for_catches(catches: pd.DataFrame) -> pd.DataFrame:
    reports = load_reports()
    out = pd.DataFrame(index=catches.index)
    cols = [
        "upstream_cpue_lag1d", "upstream_cpue_lag3d", "upstream_cpue_lag7d",
        "upstream_cpue_ma7d", "upstream_reports_lag3d",
    ]
    for c in cols:
        out[c] = np.nan

    if reports.empty or "species" not in catches.columns:
        return out

    catches_dt = pd.to_datetime(catches["datetime"]).dt.normalize()
    catches_sp = catches["species"].astype(str)

    reports = reports.copy()
    reports["weight"] = reports["region"].map(REGION_WEIGHTS).fillna(0.5)
    reports["cpue_w"] = reports["cpue"].astype(float) * reports["weight"]
    agg = (
        reports.groupby(["date", "species"])
        .agg(cpue_w_sum=("cpue_w", "sum"),
             weight_sum=("weight", "sum"),
             reports=("n_reports", "sum"))
        .reset_index()
    )
    agg["cpue"] = agg["cpue_w_sum"] / agg["weight_sum"].replace(0, np.nan)
    agg = agg[["date", "species", "cpue", "reports"]].sort_values(["species", "date"])

    for i, (target_dt, sp) in enumerate(zip(catches_dt, catches_sp)):
        if sp not in PELAGIC_SPECIES:
            continue
        target_d = target_dt.date()
        sub = agg[agg["species"] == sp]
        if sub.empty:
            continue

        for lag in (1, 3, 7):
            d_lag = target_d - timedelta(days=lag)
            m = sub[sub["date"] == d_lag]
            if not m.empty:
                out.loc[catches.index[i], f"upstream_cpue_lag{lag}d"] = m["cpue"].iloc[0]
                if lag == 3:
                    out.loc[catches.index[i], "upstream_reports_lag3d"] = m["reports"].iloc[0]

        past7 = sub[(sub["date"] >= target_d - timedelta(days=7))
                    & (sub["date"] < target_d)]
        if not past7.empty:
            out.loc[catches.index[i], "upstream_cpue_ma7d"] = past7["cpue"].mean()

    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="潮上流釣果データの確認")
    args = parser.parse_args()
    df = load_reports()
    print(f"records: {len(df)}")
    if not df.empty:
        print(df.tail(20))


if __name__ == "__main__":
    _cli()
