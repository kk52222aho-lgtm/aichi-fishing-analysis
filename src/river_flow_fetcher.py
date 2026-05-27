"""国交省 水文水質DB から河川流量・水位データを取得する。

データソース:
    http://www1.river.go.jp/  （水文水質データベース）

  - 観測所別の時間値・日値ファイル取得は 2 通り:
      a) CSV ダウンロード形式
         例: http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=3&ID={id}&BGNDATE={YYYYMMDD}&ENDDATE={YYYYMMDD}&KAWABOU=NO
         HTML だが内部に CSV エクスポートリンクあり
      b) テキスト直リンク (一部観測所のみ): /dat/dload/download/{id}.dat

  KIND コード:
      1 = 水位 (m)
      3 = 流量 (m3/s)
      2 = 雨量 (mm)

愛知近海に効く主要河川（用途別）:
    - 伊勢湾奥への注入: 木曽川 / 長良川 / 揖斐川 / 庄内川
    - 三河湾への注入  : 矢作川 / 豊川 / 矢崎川
    - 遠州灘西側     : 天竜川 (外洋ベイト経由でindirectに効く)

観測所コードはサイト変更で動くことがあるため CONFIG として外部から差し替え可能にしている。
ユーザーは config.py の AICHI_RIVERS dict を更新して使う。

使い方:
    from src.river_flow_fetcher import fetch_and_cache_river
    df = fetch_and_cache_river("kiso_inuyama", "2026-04-01", "2026-05-01", kind="discharge")
"""
from __future__ import annotations

import argparse
import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests

from . import config

_TIMEOUT = 30
_BASE = "http://www1.river.go.jp/cgi-bin/DspWaterData.exe"

KIND_MAP = {
    "water_level": 1,   # 水位 (m)
    "rainfall": 2,      # 雨量 (mm)
    "discharge": 3,     # 流量 (m3/s)
}


# ===================================================================
# 観測所定義（愛知の釣り場に影響しそうな主要河川の下流観測所）
# ===================================================================
# IDは10桁数値文字列。実運用前に http://www1.river.go.jp/ で照合してください。
# 緯度経度は概数（伊勢湾/三河湾流域マッピング用）。

AICHI_RIVERS = {
    # 伊勢湾系（北西から流入）
    "kiso_inuyama":      {"name": "木曽川・犬山",     "river": "木曽川",  "bay": "伊勢湾", "id": "305031283311010"},
    "nagara_chusetsu":   {"name": "長良川・忠節",     "river": "長良川",  "bay": "伊勢湾", "id": "305051283414020"},
    "ibi_mangoku":       {"name": "揖斐川・万石",     "river": "揖斐川",  "bay": "伊勢湾", "id": "305071283513070"},
    "shonai_biwajima":   {"name": "庄内川・枇杷島",   "river": "庄内川",  "bay": "伊勢湾", "id": "305131284010230"},
    # 三河湾系
    "yahagi_iwatsu":     {"name": "矢作川・岩津",     "river": "矢作川",  "bay": "三河湾", "id": "304081284206290"},
    "toyokawa_ushikawa": {"name": "豊川・牛川",       "river": "豊川",    "bay": "三河湾", "id": "304041284311050"},
    # 遠州灘外側
    "tenryu_kashima":    {"name": "天竜川・鹿島",     "river": "天竜川",  "bay": "遠州灘", "id": "303081284506050"},
}

# 観測サイト → 関連河川 のマッピング（merge時の代理シグナル）
SITE_RIVERS = {
    "shinojima":     ["kiso_inuyama", "nagara_chusetsu", "ibi_mangoku", "shonai_biwajima"],
    "morozaki":      ["kiso_inuyama", "nagara_chusetsu", "ibi_mangoku", "shonai_biwajima"],
    "utsumi_shinko": ["kiso_inuyama", "nagara_chusetsu", "ibi_mangoku", "shonai_biwajima"],
    "chita_tip":     ["kiso_inuyama", "nagara_chusetsu", "ibi_mangoku"],
    "irago":         ["yahagi_iwatsu", "toyokawa_ushikawa", "tenryu_kashima"],
    "mikawa_bay":    ["yahagi_iwatsu", "toyokawa_ushikawa"],
}

# キャッシュ場所
RIVER_DIR = config.DATA_DIR / "river"
RIVER_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(station_key: str, year: int, kind: str) -> Path:
    return RIVER_DIR / f"{station_key}_{kind}_{year}.csv"


# ===================================================================
# 国交省 DB からの取得
# ===================================================================

def _fetch_yearly(station_id: str, year: int, kind_code: int) -> pd.DataFrame:
    """国交省 cgi-bin から1年分を取得。HTML/テキスト両対応で解析する。"""
    bgn = f"{year}0101"
    end = f"{year}1231"
    params = {
        "KIND": kind_code,
        "ID": station_id,
        "BGNDATE": bgn,
        "ENDDATE": end,
        "KAWABOU": "NO",
    }
    resp = requests.get(_BASE, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    # サーバは Shift_JIS 返却が多い
    resp.encoding = resp.apparent_encoding or "shift_jis"
    text = resp.text
    return _parse_html_response(text, kind_code)


def _parse_html_response(text: str, kind_code: int) -> pd.DataFrame:
    """国交省レスポンスHTMLから時系列値テーブルを抽出。

    観測所により表構造が違うので、複数のパースを試行する。
    1) <pre> 内に CSV 形式（YYYY/MM/DD HH:MM, value）
    2) <table> 内の行（日付列 + 24時間値）
    """
    # 戦略1: <pre> CSV
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", text, flags=re.S | re.I)
    if pre_match:
        try:
            df = pd.read_csv(io.StringIO(pre_match.group(1)), header=None,
                             names=["datetime_str", "value"])
            df["datetime"] = pd.to_datetime(df["datetime_str"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["datetime", "value"]].dropna()
        except Exception:
            pass

    # 戦略2: HTML table — 「年月日」列＋時系列（最近のDB UIで使われる）
    try:
        tables = pd.read_html(io.StringIO(text), header=0)
    except Exception:
        return pd.DataFrame(columns=["datetime", "value"])

    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        # 「日時」「時間」「観測値」のような列を探す
        dt_col = next((c for c in cols if re.search(r"日時|時刻|datetime", str(c))), None)
        val_col = next((c for c in cols if re.search(r"観測|値|流量|水位|雨量|level|discharge", str(c))), None)
        if dt_col and val_col:
            out = pd.DataFrame({
                "datetime": pd.to_datetime(tbl[dt_col], errors="coerce"),
                "value": pd.to_numeric(tbl[val_col], errors="coerce"),
            }).dropna()
            if not out.empty:
                return out
        # フォールバック: 「年月日」+ 1..24時 列の wide 表
        if any(re.search(r"年月日|月日|日付|date", str(c)) for c in cols):
            date_col = cols[0]
            hour_cols = [c for c in cols[1:] if re.fullmatch(r"\d{1,2}|\d{1,2}時", str(c))]
            if hour_cols:
                rows = []
                for _, r in tbl.iterrows():
                    base = pd.to_datetime(r[date_col], errors="coerce")
                    if pd.isna(base):
                        continue
                    for h_col in hour_cols:
                        h = int(re.sub(r"[^0-9]", "", str(h_col)))
                        v = pd.to_numeric(r[h_col], errors="coerce")
                        if not pd.isna(v):
                            rows.append({"datetime": base + pd.Timedelta(hours=h), "value": float(v)})
                if rows:
                    return pd.DataFrame(rows)

    return pd.DataFrame(columns=["datetime", "value"])


def fetch_and_cache_river(
    station_key: str,
    start: str | date,
    end: str | date,
    kind: Literal["water_level", "rainfall", "discharge"] = "discharge",
    force: bool = False,
) -> pd.DataFrame:
    """観測所×期間 の時間値をキャッシュ経由で取得。

    Args:
        station_key: AICHI_RIVERS のキー
        start, end : 取得期間（YYYY-MM-DD or date）
        kind       : "water_level" / "rainfall" / "discharge"
    """
    if station_key not in AICHI_RIVERS:
        raise ValueError(f"unknown station: {station_key}")
    info = AICHI_RIVERS[station_key]
    kind_code = KIND_MAP[kind]

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        cache = _cache_path(station_key, year, kind)
        if cache.exists() and not force:
            df = pd.read_csv(cache, parse_dates=["datetime"])
        else:
            try:
                df = _fetch_yearly(info["id"], year, kind_code)
            except Exception as e:
                print(f"⚠️ {station_key} {year} {kind}: {e}")
                df = pd.DataFrame(columns=["datetime", "value"])
            if not df.empty:
                df.to_csv(cache, index=False)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["datetime", "value", "station_key", "river"])

    out = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    out = out[(out["datetime"].dt.date >= start) & (out["datetime"].dt.date <= end)].copy()
    out["station_key"] = station_key
    out["river"] = info["river"]
    return out


# ===================================================================
# データインテグレータ向けヘルパ
# ===================================================================

def fetch_site_rivers(
    site: str,
    start: date,
    end: date,
    kind: str = "discharge",
) -> pd.DataFrame:
    """サイトに関連する全河川観測所のデータを取得し、wide形式で返す。

    返り値の列:
      time, river_{key}_discharge, river_{key}_water_level, ...

    site_rivers[site] にない観測所は無視。エラー観測所はスキップ。
    """
    if site not in SITE_RIVERS:
        return pd.DataFrame(columns=["time"])
    keys = SITE_RIVERS[site]
    frames: list[pd.DataFrame] = []

    for key in keys:
        df = fetch_and_cache_river(key, start, end, kind=kind)
        if df.empty:
            continue
        col_name = f"river_{key}_{kind}"
        sub = df[["datetime", "value"]].rename(columns={"datetime": "time", "value": col_name})
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["time"])

    # 全 station の outer merge
    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="time", how="outer")
    merged = merged.sort_values("time").reset_index(drop=True)

    # 派生: site合計流量・3日積算
    discharge_cols = [c for c in merged.columns if c.endswith("_discharge")]
    if discharge_cols:
        merged["river_total_discharge"] = merged[discharge_cols].sum(axis=1, min_count=1)
        merged["river_total_discharge_3d"] = (
            merged["river_total_discharge"].rolling(24 * 3, min_periods=24).sum()
        )
        merged["river_total_discharge_7d"] = (
            merged["river_total_discharge"].rolling(24 * 7, min_periods=24).sum()
        )
        # 出水フラグ: 平年(=7日平均)の2倍以上
        baseline = merged["river_total_discharge"].rolling(24 * 7, min_periods=24).mean()
        merged["river_flood_flag"] = (
            (merged["river_total_discharge"] > baseline * 2.0).astype("Int64")
        )

    return merged


# ===================================================================
# CLI
# ===================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(description="国交省 河川データ取得")
    parser.add_argument("--station", required=True, choices=list(AICHI_RIVERS),
                        help="観測所キー")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--kind", default="discharge",
                        choices=list(KIND_MAP))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = fetch_and_cache_river(args.station, args.start, args.end,
                               kind=args.kind, force=args.force)
    print(df.head())
    print(f"{len(df)} 行 取得  ({args.station} {args.kind})")


if __name__ == "__main__":
    _cli()
