"""魚市場の日次価格・入荷量データの取得。

データソース候補:
    1. 名古屋市中央卸売市場 北部市場 / 南部市場
       https://www.city.nagoya.jp/keizai/category/56-7-7-0-0-0-0-0-0-0.html
       日報PDF（魚種別 平均価格 円/kg, 入荷量 kg）
    2. 豊浜漁港 / 師崎漁港 / 一色漁港 の地元市場速報
       愛知県漁業協同組合連合会 https://www.aichigyoren.or.jp/
    3. 焼津・三崎・那智勝浦等 広域比較用

実装:
    - 直近価格は当日確定しないので、推論には「前日終値」「7日移動平均」を使う
    - 価格急騰 = 供給不足 = 不漁シグナル（逆相関）
    - データは CSV キャッシュ形式 (date, species, price_yen_per_kg, volume_kg)
      → ユーザーが手動で更新 or 別途スクレイパー追加

カラム仕様:
    date              : YYYY-MM-DD
    species           : 魚種（catches.csv の species と一致）
    price_yen_per_kg  : 1日の平均価格（円/kg）
    volume_kg         : 入荷量（kg）
    market            : 市場名（"nagoya_north", "nagoya_south", "toyohama", ...）
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

MARKET_DIR = config.DATA_DIR / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)
MARKET_CSV = MARKET_DIR / "prices.csv"

TRACKED_SPECIES = [
    "アジ", "イワシ", "サバ", "サワラ", "タチウオ",
    "マダイ", "クロダイ", "スズキ", "ヒラメ", "カレイ",
    "アオリイカ", "コウイカ", "マダコ", "ブリ", "ハマチ",
    "メジロ", "イサキ", "アマダイ",
]


def load_prices() -> pd.DataFrame:
    if not MARKET_CSV.exists():
        return pd.DataFrame(columns=["date", "species", "price_yen_per_kg", "volume_kg", "market"])
    df = pd.read_csv(MARKET_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def save_prices(df: pd.DataFrame) -> Path:
    df = df.copy()
    df.to_csv(MARKET_CSV, index=False)
    return MARKET_CSV


def compute_features_for_catches(catches: pd.DataFrame) -> pd.DataFrame:
    prices = load_prices()
    out = pd.DataFrame(index=catches.index)
    cols = [
        "market_price_lag1d", "market_price_lag7d", "market_price_ma7d",
        "market_volume_lag1d", "market_volume_ma7d",
        "market_price_change_7d",
    ]
    for c in cols:
        out[c] = np.nan

    if prices.empty or "datetime" not in catches.columns or "species" not in catches.columns:
        return out

    daily = (
        prices.groupby(["date", "species"])
        .agg(price=("price_yen_per_kg", "mean"),
             volume=("volume_kg", "sum"))
        .reset_index()
    )
    daily = daily.sort_values(["species", "date"]).reset_index(drop=True)

    catches_dt = pd.to_datetime(catches["datetime"]).dt.normalize()
    catches_sp = catches["species"].astype(str)

    for i, (target_dt, sp) in enumerate(zip(catches_dt, catches_sp)):
        target_d = target_dt.date()
        sub = daily[daily["species"] == sp]
        if sub.empty:
            continue

        d_lag1 = target_d - timedelta(days=1)
        m1 = sub[sub["date"] == d_lag1]
        if not m1.empty:
            out.loc[catches.index[i], "market_price_lag1d"] = m1["price"].iloc[0]
            out.loc[catches.index[i], "market_volume_lag1d"] = m1["volume"].iloc[0]

        d_lag7 = target_d - timedelta(days=7)
        m7 = sub[sub["date"] == d_lag7]
        if not m7.empty:
            out.loc[catches.index[i], "market_price_lag7d"] = m7["price"].iloc[0]

        past7 = sub[(sub["date"] >= target_d - timedelta(days=7))
                    & (sub["date"] < target_d)]
        if not past7.empty:
            out.loc[catches.index[i], "market_price_ma7d"] = past7["price"].mean()
            out.loc[catches.index[i], "market_volume_ma7d"] = past7["volume"].mean()

        if not m1.empty and not m7.empty and m7["price"].iloc[0] > 0:
            out.loc[catches.index[i], "market_price_change_7d"] = (
                m1["price"].iloc[0] / m7["price"].iloc[0] - 1
            )

    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="市場価格データの確認")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = load_prices()
    print(f"records: {len(df)}")
    if args.show and not df.empty:
        print(df.tail(20))
        print(f"\n魚種数: {df['species'].nunique()}")
        print(f"市場数: {df['market'].nunique()}")
        print(f"期間: {df['date'].min()} ~ {df['date'].max()}")


if __name__ == "__main__":
    _cli()
