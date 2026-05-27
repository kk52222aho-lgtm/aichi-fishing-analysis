"""黒潮流路カテゴリの取得。

気象庁「海面水温・海流」発表をベースに、各日の黒潮蛇行状態を5値カテゴリで返す。

主要カテゴリ（気象庁分類）:
    大蛇行         : 八丈島南方を大きく南偏（黒潮南偏期）
    非大蛇行・小蛇行A : 接岸傾向
    非大蛇行・小蛇行B : 中間
    非大蛇行・小蛇行C : 沖合離岸傾向
    直進型         : 蛇行が小さい状態

参考:
    https://www.data.jma.go.jp/kaiyou/data/shindan/sokuho_kuro/index.html
    https://www.jamstec.go.jp/aplinfo/jcope/  (JCOPE-T, より細かい流軸位置)

実装戦略:
1. 既知期間の状態を `KUROSHIO_HISTORY` にハードコード（バックフィル用）
2. 最新情報は気象庁ページから月次スクレイプ（_fetch_latest）
3. 釣果側 datetime を月次値に merge_asof で結合

実運用での更新:
   - 気象庁が新しいカテゴリを発表したら KUROSHIO_HISTORY に追記
   - 月次パイプラインで kuroshio_state.csv を再生成
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config

# キャッシュ場所
KUROSHIO_DIR = config.DATA_DIR / "kuroshio"
KUROSHIO_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = KUROSHIO_DIR / "history.csv"

KUROSHIO_STATES = ["大蛇行", "小蛇行A", "小蛇行B", "小蛇行C", "直進"]

# ===================================================================
# 過去履歴（気象庁公式発表ベース、月単位）
# 期間: start_date <= d < end_date のときその state
# 必要に応じてユーザーが追記・修正する
# ===================================================================
KUROSHIO_HISTORY: list[dict] = [
    # 2017年8月〜2024年夏: 戦後最長級の大蛇行
    {"start": "2017-08-01", "end": "2024-09-01", "state": "大蛇行"},
    # 2024年秋以降は流路の南偏が縮小傾向（要月次再確認）
    {"start": "2024-09-01", "end": "2025-10-01", "state": "小蛇行B"},
    {"start": "2025-10-01", "end": "2027-01-01", "state": "小蛇行A"},
]


def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def build_daily_history() -> pd.DataFrame:
    """KUROSHIO_HISTORY を日次 DataFrame に展開して返す。"""
    rows: list[dict] = []
    for entry in KUROSHIO_HISTORY:
        s = _to_date(entry["start"])
        e = _to_date(entry["end"])
        days = pd.date_range(s, e - timedelta(days=1), freq="1D")
        for d in days:
            rows.append({"date": d.date(), "kuroshio_state": entry["state"]})
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def save_history_csv() -> Path:
    df = build_daily_history()
    df.to_csv(HISTORY_CSV, index=False)
    return HISTORY_CSV


def load_history(force_rebuild: bool = False) -> pd.DataFrame:
    if HISTORY_CSV.exists() and not force_rebuild:
        df = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return build_daily_history()


# ===================================================================
# 気象庁 黒潮速報の月次スクレイプ（最新確認用）
# ===================================================================

_JMA_URL = "https://www.data.jma.go.jp/kaiyou/data/shindan/sokuho_kuro/index.html"


def fetch_latest_state(timeout: int = 30) -> dict | None:
    """気象庁ページから最新の黒潮状態テキストを取得して状態を推定。

    返り値: {"date": YYYY-MM, "state": "大蛇行" or ...}  or  None
    """
    try:
        import requests
        resp = requests.get(_JMA_URL, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        text = resp.text
    except Exception as e:
        print(f"⚠️ JMA fetch failed: {e}")
        return None

    state = _classify_text(text)
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        ym = f"{int(m.group(1))}-{int(m.group(2)):02d}"
    else:
        ym = datetime.now().strftime("%Y-%m")
    return {"date": ym, "state": state}


def _classify_text(text: str) -> str:
    """ページテキストから黒潮状態を5値で分類。"""
    if "大蛇行" in text:
        return "大蛇行"
    if re.search(r"接岸|北偏", text):
        return "小蛇行A"
    if re.search(r"離岸|南偏", text):
        return "小蛇行C"
    if "小蛇行" in text:
        return "小蛇行B"
    if "直進" in text or "蛇行小" in text:
        return "直進"
    return "小蛇行B"


# ===================================================================
# データインテグレータ向けヘルパ
# ===================================================================

def kuroshio_state_for_range(start: date, end: date) -> pd.DataFrame:
    """指定期間の日次黒潮状態を返す（毎時刻に展開）。"""
    daily = load_history()
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    if daily.empty:
        return pd.DataFrame(columns=["time", "kuroshio_state"])

    rows = []
    for _, r in daily.iterrows():
        base = pd.Timestamp(r["date"])
        for h in range(24):
            rows.append({"time": base + pd.Timedelta(hours=h),
                         "kuroshio_state": r["kuroshio_state"]})
    return pd.DataFrame(rows)


def update_history_with_latest() -> bool:
    """最新の気象庁状態を取得し、必要なら history.csv を更新。"""
    latest = fetch_latest_state()
    if not latest:
        return False

    target_ym = latest["date"]
    target_state = latest["state"]

    df = load_history()
    df["ym"] = df["date"].apply(lambda d: f"{d.year:04d}-{d.month:02d}")
    existing = df[df["ym"] == target_ym]
    if not existing.empty and (existing["kuroshio_state"].iloc[0] == target_state):
        return False

    y, m = map(int, target_ym.split("-"))
    s = date(y, m, 1)
    e = date(y + (m // 12), (m % 12) + 1, 1)
    new_rows = pd.DataFrame({
        "date": pd.date_range(s, e - timedelta(days=1), freq="1D").date,
        "kuroshio_state": target_state,
    })
    df = df.drop(columns=["ym"])
    merged = pd.concat([df[~df["date"].between(s, e - timedelta(days=1))], new_rows],
                      ignore_index=True).sort_values("date")
    merged.to_csv(HISTORY_CSV, index=False)
    return True


def _cli() -> None:
    parser = argparse.ArgumentParser(description="黒潮蛇行状態の取得・更新")
    parser.add_argument("--rebuild", action="store_true",
                        help="KUROSHIO_HISTORY から history.csv を再生成")
    parser.add_argument("--update-latest", action="store_true",
                        help="気象庁から最新月の状態を取得して反映")
    args = parser.parse_args()

    if args.rebuild:
        p = save_history_csv()
        print(f"rebuilt: {p}")
    if args.update_latest:
        ok = update_history_with_latest()
        print(f"updated: {ok}")

    df = load_history()
    print(df.tail(10))
    print(f"\n総日数: {len(df)}")
    print(f"カテゴリ分布:\n{df['kuroshio_state'].value_counts()}")


if __name__ == "__main__":
    _cli()
