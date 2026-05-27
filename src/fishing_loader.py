"""釣果記録（CSV/Excel/Googleスプレッドシート）の読み込み。

`count` は最終的に YOLO（GenesisEngine-v6）が釣果写真から数える想定。
そのため学習時にも欠損を許す（YOLO処理待ち or 手動入力）。
予測モデルの入力は本ファイルのスキーマ + 気象/海象/潮汐/天文。

期待スキーマ:
    datetime         ISO8601（例: 2026-05-08T05:30:00+09:00）
    site             地点コード（src.config.SITES のキー）
    species          魚種（target_species と一致するか、副次釣果なら別）
    count            匹数（int）— YOLOが入れる or 手動。空欄可
    boat             船宿/船名（任意。one-hotで効く）
    anglers          乗船人数（任意。匹/竿 正規化に使う）
    target_species   その日の狙い魚種（任意。species と異なる行もありうる）
    tackle           仕掛け種別（任意。サビキ/ジギング/エギング 等）
    departure_hour   出船時刻（0-23、任意）
    total_weight_g   合計重量(g)
    avg_length_cm    平均体長(cm)
    angler           釣り人名
    notes            備考
    image_path       釣果写真（YOLO用）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config

REQUIRED_COLUMNS = ("datetime", "site", "species")
OPTIONAL_COLUMNS = (
    "count",            # 船全体の釣果推定（基本は YOLO 画像カウント）
    "count_yolo",       # YOLO の生カウント（count と同値、監査用に明示）
    "top_per_angler",   # 本文「竿頭 N尾」抽出 — 個人最大カウント
    "total_catch",      # 本文「全N匹」抽出 — 船全体の総釣果数
    "max_size_cm",      # 本文「最大Nセンチ」抽出 — 最大魚体長
    "qualitative",      # 本文の定性的評価（絶好調/好調/普通/渋い/厳しい/ボウズ）
    "boat",
    "anglers",
    "target_species",
    "tackle",
    "departure_hour",
    "total_weight_g",
    "avg_length_cm",
    "angler",
    "notes",
    "image_path",
    "entry_title",      # 元ブログのタイトル（target_species の根拠／監査用）
)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必須カラムが不足: {missing}")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["anglers"] = pd.to_numeric(df["anglers"], errors="coerce")
    df["departure_hour"] = pd.to_numeric(df["departure_hour"], errors="coerce")
    unknown_sites = set(df["site"].dropna()) - set(config.SITES)
    if unknown_sites:
        raise ValueError(f"未登録の地点コード: {unknown_sites}")
    return df[list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)]


def load_csv(path: str | Path) -> pd.DataFrame:
    return _validate(pd.read_csv(path))


def load_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    return _validate(pd.read_excel(path, sheet_name=sheet_name))


def load_google_sheet(spreadsheet_id: str, worksheet: str = "釣果",
                      credentials_path: str | Path = "service_account.json") -> pd.DataFrame:
    """Googleスプレッドシートから読み込み（gspread + サービスアカウント認証）。"""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet)
    rows = ws.get_all_records()
    return _validate(pd.DataFrame(rows))


def load_auto(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        return load_excel(p)
    return load_csv(p)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="釣果ファイル読み込み確認")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    df = load_auto(args.path)
    print(df.dtypes)
    print(df.head())
    print(f"{len(df)} 行 / count欠損 {df['count'].isna().sum()} 行（YOLO処理待ち）")


if __name__ == "__main__":
    _cli()
