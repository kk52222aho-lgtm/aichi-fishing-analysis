"""data/scraped/<entry>/summary.json を catches.csv に集約する。

predict_from_url.run() が吐く summary.json をまとめて読み、
釣果ログ（fishing_loader が期待するスキーマ）に変換する。

日付の決め方:
  1. summary.results[*].image_url の `/YYYYMMDD/` パス（Ameblo の画像CDN形式）
  2. なければ posted_at（ブログ投稿日時）— build_seed_dataset 側で渡される

site / boat は呼び出し側で指定（船宿1軒に絞る運用想定）。
複数日が混じることは原則無いが、安全のため (datetime, site, species) で集約。

使い方:
    from src.scrape_to_catches import aggregate
    df = aggregate(site="irago", boat="maruman2010")

CLI:
    python -m src.scrape_to_catches --site irago --boat maruman2010
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config, title_parser

JST = timezone(timedelta(hours=9))
_IMG_DATE_RE = re.compile(r"/(\d{8})/")
_AMEBLO_BLOG_ID_RE = re.compile(r"ameblo\.jp/([^/]+)/")

DEFAULT_DEPARTURE_HOUR = 5  # 釣り船の平均的な出船時刻（仮）

# 魚種名の表記揺れ正規化マップ（船宿ごとに表記が異なるため aggregate 時に統一）
# LLM が船宿ブログの表記をそのまま使うので、aggregate で正規化する
SPECIES_NORMALIZE_MAP: dict[str, str] = {
    # 漢字 ↔ カタカナ
    "ガンゾウヒラメ":   "ガンゾウビラメ",  # 「ガンゾウ平目」=「ガンゾウ鮃」表記揺れ
    "マアジ":           "アジ",
    "真鯵":             "アジ",
    "真鯛":             "マダイ",       # シーブルー等が「真鯛」表記
    "黒鯛":             "クロダイ",
    "石鯛":             "イシダイ",
    "太刀魚":           "タチウオ",
    "障泥烏賊":         "アオリイカ",
    "鬼笠子":           "オニカサゴ",
    # マダイ系の俗称・サイズ違い表記を統一（学習上は同じ魚として扱う）
    "マイクロ真鯛":     "マダイ",
    "当歳魚(マイクロ真鯛)": "マダイ",     # シーブルー特有の表記
    "小鯛":             "マダイ",       # 若いマダイ
    "大ダイ":           "マダイ",
    "ダイ":             "マダイ",       # LLM の脱字
    "タイ":             "マダイ",       # 漢字なし表記
    # 関西名・若魚名（ブリ族・サワラ族は出世魚なので関東系に統一）
    "ツバス":           "ハマチ",       # 関西「ツバス」 = 関東「ワカシ/ハマチ」
    "ワカシ":           "ハマチ",
    "イナダ":           "ハマチ",       # 関東「イナダ」≒ハマチ
    "シオ":             "カンパチ",     # シオ = カンパチ若魚
    "サゴシ":           "サワラ",       # サゴシ = サワラ若魚
    "メジマグロ":       "クロマグロ",   # メジ = クロマグロ幼魚
    # イカ・タチウオ・ハゼの揺れ
    "ライトヤリイカ":   "ヤリイカ",     # 「ライトヤリイカ便」仕掛け名混入
    "スルメ":           "スルメイカ",
    "ドラゴン":         "タチウオ",     # ドラゴン級タチウオ
    "ドラゴン太刀魚":   "タチウオ",
    # その他
    "鱒":               "サクラマス",   # シーブルーの私的釣行（虹鱒・鱒）
    "いねごち":         "イネゴチ",     # 漢字→カタカナ
    "こういか":         "コウイカ",
    "シーバス":         "スズキ",       # シーバス=スズキ
    "しょうさいフグ":   "シューサイフグ",
    # フグ系は同一とせず保持（種が違うため）：シューサイフグ ≠ トラフグ ≠ フグ
}

# 完全に除外する非魚種・誤抽出・総称（aggregate 時に row ごと drop）
SPECIES_DROP_SET: set[str] = {
    "シーブルー",       # 船宿名が species として誤抽出
    "アチコチ",         # 副詞「あちこち」を誤抽出
    "青物",             # ブリ/ワラサ/サワラ等の総称（個別記載がある時は重複、無い時は分類不能）
    "大型青物",         # 同上
    "タイラバ",         # 仕掛け名の誤抽出
    "虎ハゼ",           # 同定不能なハゼ俗称
}

# blog_id → {boat, site} の対応表を保持する registry ファイル
# build_seed_dataset が build 時に書き込み、aggregate が読み込む
BLOG_REGISTRY_PATH = config.DATA_DIR / "scraped" / "_blog_registry.json"


def _blog_id_from_url(url: str) -> Optional[str]:
    """エントリ URL から ameblo の blog_id（screen_name）を抽出。"""
    if not url:
        return None
    m = _AMEBLO_BLOG_ID_RE.search(url)
    return m.group(1) if m else None


def _match_custom_blog_id(url: str, registry: dict[str, dict]) -> Optional[str]:
    """独自サイトの URL を registry の blog_url とホスト名で突き合わせて blog_id 逆引き。
    ameblo 以外の船宿(石川丸/久六/としかず/大進丸)の boat 解決用。"""
    if not url:
        return None
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower()
    if not host:
        return None
    for bid, entry in registry.items():
        bhost = urlparse(entry.get("blog_url") or "").netloc.lower()
        if bhost and (bhost == host or host.endswith("." + bhost) or bhost.endswith("." + host)):
            return bid
    return None


def load_blog_registry() -> dict[str, dict]:
    """blog_id → {boat, site, ...} の registry を読み込む。"""
    if not BLOG_REGISTRY_PATH.exists():
        return {}
    try:
        with BLOG_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def register_blog(
    blog_id: str,
    boat: str,
    site: str,
    primary_signal: Optional[str] = None,
    secondary_signal: Optional[str] = None,
    **extra: Any,
) -> dict[str, dict]:
    """blog_id を registry に登録（既存ならマージ更新）。

    Args:
        blog_id: ameblo の screen_name
        boat: catches.csv に入れる船宿名
        site: config.SITES のキー
        primary_signal: この船宿の主力データソース
            - "top_per_angler": 本文に「竿頭 N尾」が明示（maruman2010 系）
            - "qualitative":    本文は定性表現中心（ありもと丸 系）
            - "count_yolo":     写真メイン（武丸 系）
            - "total_catch":    本文に船合計記載
            None なら build/aggregate 時にデータ密度から自動判定
        secondary_signal: 補助シグナル（同上）
        extra: 追加メタ
    """
    reg = load_blog_registry()
    entry = reg.get(blog_id, {})
    entry["boat"] = boat
    entry["site"] = site
    if primary_signal is not None:
        entry["primary_signal"] = primary_signal
    if secondary_signal is not None:
        entry["secondary_signal"] = secondary_signal
    for k, v in extra.items():
        if v is not None:
            entry[k] = v
    reg[blog_id] = entry
    BLOG_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BLOG_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    return reg


def get_boat_profile(boat: str) -> dict[str, Any]:
    """船宿名から registry エントリを引く（registry のキーは blog_id だが、
    LLM 予測時など boat 名しか知らない呼び出し側のための逆引きヘルパ）。
    """
    reg = load_blog_registry()
    for blog_id, entry in reg.items():
        if entry.get("boat") == boat:
            return {"blog_id": blog_id, **entry}
    return {}


def _extract_date_from_summary(summary: dict) -> Optional[datetime]:
    """results 内の image_url から /YYYYMMDD/ を拾う。"""
    for r in summary.get("results", []):
        m = _IMG_DATE_RE.search(r.get("image_url", ""))
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=JST)
            except ValueError:
                continue
    return None


def _load_text_extracted(summary_path: Path) -> dict[str, dict]:
    """同一ディレクトリの text_extracted.json から catch_info を読み出す。"""
    p = summary_path.parent / "text_extracted.json"
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("catch_info") or {}
    except Exception:
        return {}


def _load_text_title(summary_path: Path) -> Optional[str]:
    """同一ディレクトリの text_extracted.json から title を読み出す。
    title_map が外部から渡されなくても永続化された title を使えるようにする。
    """
    p = summary_path.parent / "text_extracted.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        t = data.get("title")
        return t.strip() if isinstance(t, str) and t.strip() else None
    except Exception:
        return None


def _load_text_posted_at(summary_path: Path) -> Optional[datetime]:
    """text_extracted.json に永続化された posted_at を読み出す。

    aggregate を単体で呼ぶ場合（posted_at_map が無い場合）に
    summary.results が空でも日付を解決できるようにするフォールバック。
    """
    p = summary_path.parent / "text_extracted.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("posted_at")
        if not s:
            return None
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
        return dt
    except Exception:
        return None


def summary_to_rows(
    summary_path: Path,
    site: str,
    boat: str,
    posted_at: Optional[datetime] = None,
    departure_hour: int = DEFAULT_DEPARTURE_HOUR,
    entry_title: Optional[str] = None,
    blog_registry: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """1 エントリの summary.json + text_extracted.json から釣果ログ行を作る。

    生成ルール:
      - 魚種は Union(YOLO検出, 本文抽出)。
      - count       = YOLO の検出合計（写真ベースの船全体推定）。本文のみ存在ならNaN。
      - top_per_angler = 本文「竿頭 N尾」抽出。無ければ NaN。
      - max_size_cm = 本文「最大 Nセンチ」抽出。無ければ NaN。
      - target_species = タイトル優先、無ければ YOLO 最頻検出魚種。
      - boat/site は source_url の blog_id を blog_registry で解決。
        registry にエントリが無ければ default 引数を使う。
    """
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    yolo_totals: dict[str, int] = summary.get("total_per_species", {})
    text_info: dict[str, dict] = _load_text_extracted(summary_path)

    # YOLO もテキストも空ならエントリ自体をスキップ
    if not yolo_totals and not text_info:
        return []

    # ── boat/site の解決（registry 優先、無ければ引数の default） ──
    source_url = summary.get("source_url", "")
    if blog_registry is None:
        blog_registry = load_blog_registry()
    blog_id = _blog_id_from_url(source_url)
    if not blog_id or blog_id not in blog_registry:
        # ameblo で引けない＝独自サイト → ホスト名で registry 逆引き
        blog_id = _match_custom_blog_id(source_url, blog_registry) or blog_id
    reg_entry = blog_registry.get(blog_id or "", {})
    resolved_boat = reg_entry.get("boat") or boat
    resolved_site = reg_entry.get("site") or site

    # 日付の解決順:
    #   1. summary.results[*].image_url の /YYYYMMDD/（YOLO 経由で取れる場合）
    #   2. 呼び出し側から渡された posted_at（title_map と一緒に渡されることが多い）
    #   3. text_extracted.json に永続化された posted_at（aggregate 単体呼び出し時）
    dt = _extract_date_from_summary(summary) or posted_at or _load_text_posted_at(summary_path)
    if dt is None:
        return []
    dt = dt.replace(hour=departure_hour, minute=30, second=0, microsecond=0)

    # 各魚種の代表画像（YOLO 検出があったもののみ）
    species_image: dict[str, str] = {}
    for r in summary.get("results", []):
        for sp in r.get("count_per_species", {}):
            if sp not in species_image and r.get("local"):
                species_image[sp] = r["local"]

    # entry_title フォールバック: title_map で渡されなくても text_extracted.json から永続化された title を使う
    if not entry_title:
        entry_title = _load_text_title(summary_path)

    # target_species: タイトル優先、無ければ YOLO 最頻検出魚種
    target_from_title = title_parser.extract_target_species(entry_title)
    dominant = (
        max(yolo_totals.items(), key=lambda x: x[1])[0]
        if yolo_totals else None
    )
    target_species = target_from_title or dominant

    # 魚種 Union（YOLO + 本文）
    all_species = set(yolo_totals) | set(text_info)

    rows: list[dict] = []
    for sp in sorted(all_species):
        yolo_count = yolo_totals.get(sp)
        ti = text_info.get(sp, {})
        top = ti.get("top_per_angler")
        total = ti.get("total_catch")
        size_cm = ti.get("max_size_cm")
        qual = ti.get("qualitative")

        # LLM 誤抽出フィルタ: top_per_angler が max_size_cm と同じ値で 25 以上の場合、
        # "N センチ" を尾数として誤抽出した可能性が高いので top をクリアする。
        # 現実的に「竿頭 30+尾」は希少。「サイズ 30+cm」は普通。
        try:
            if (
                top is not None and size_cm is not None
                and int(top) == int(size_cm) and int(top) >= 25
            ):
                top = None
        except (TypeError, ValueError):
            pass

        rows.append({
            "datetime": dt.isoformat(),
            "site": resolved_site,
            "species": sp,
            "count": int(yolo_count) if yolo_count is not None else None,
            "count_yolo": int(yolo_count) if yolo_count is not None else None,
            "top_per_angler": int(top) if top is not None else None,
            "total_catch": int(total) if total is not None else None,
            "max_size_cm": int(size_cm) if size_cm is not None else None,
            "qualitative": qual,
            "boat": resolved_boat,
            "anglers": None,
            "target_species": target_species,
            "tackle": None,
            "departure_hour": departure_hour,
            "total_weight_g": None,
            "avg_length_cm": None,
            "angler": None,
            "notes": f"yolo_scraped: {source_url}",
            "image_path": species_image.get(sp),
            "entry_title": entry_title,
        })
    return rows


def aggregate(
    scraped_dir: Path | str | None = None,
    out_path: Path | str | None = None,
    site: str = "irago",
    boat: str = "maruman2010",
    departure_hour: int = DEFAULT_DEPARTURE_HOUR,
    posted_at_map: Optional[dict[str, datetime]] = None,
    title_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """data/scraped/ 配下の全 summary.json を catches.csv に集約。

    Args:
        scraped_dir: 走査するルート（default: data/scraped）
        out_path: 出力 CSV パス（default: data/fishing_logs/catches.csv）
        site / boat: ログに固定で入れる
        posted_at_map: {entry_url: datetime} — image_url から日付が取れない場合の補完
        title_map: {entry_url: title} — タイトルから target_species を抽出する
    """
    scraped = Path(scraped_dir) if scraped_dir else config.DATA_DIR / "scraped"
    out = Path(out_path) if out_path else config.FISHING_DIR / "catches.csv"

    # 1回だけ registry を読む（各 entry の boat/site 解決に使う）
    blog_registry = load_blog_registry()

    rows: list[dict] = []
    n_summaries = 0
    n_skipped = 0
    n_target_from_title = 0
    boat_distribution: dict[str, int] = {}
    for sp in sorted(scraped.glob("*/summary.json")):
        n_summaries += 1
        # source_url 経由で posted_at と title を解決
        posted_at = None
        entry_title = None
        try:
            with sp.open("r", encoding="utf-8") as f:
                s = json.load(f)
            source_url = s.get("source_url", "")
            if posted_at_map:
                posted_at = posted_at_map.get(source_url)
            if title_map:
                entry_title = title_map.get(source_url)
        except Exception:
            pass

        entry_rows = summary_to_rows(
            sp, site=site, boat=boat,
            posted_at=posted_at, departure_hour=departure_hour,
            entry_title=entry_title,
            blog_registry=blog_registry,
        )
        if not entry_rows:
            n_skipped += 1
            continue
        # タイトル由来 target_species が入ったかを計測
        if entry_title and title_parser.extract_target_species(entry_title):
            n_target_from_title += 1
        # boat 分布も集計
        for r_ in entry_rows:
            boat_distribution[r_.get("boat", "?")] = boat_distribution.get(r_.get("boat", "?"), 0) + 1
        rows.extend(entry_rows)

    if not rows:
        print(f"⚠️ 集約対象なし（{n_summaries} summary 中 {n_skipped} 件スキップ）")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 表記揺れの正規化（船宿によって違う表記を統一）
    if SPECIES_NORMALIZE_MAP:
        df["species"] = df["species"].replace(SPECIES_NORMALIZE_MAP)
        df["target_species"] = df["target_species"].replace(SPECIES_NORMALIZE_MAP)

    # 非魚種・誤抽出・総称を完全に除外
    if SPECIES_DROP_SET:
        n_before = len(df)
        df = df[~df["species"].isin(SPECIES_DROP_SET)].reset_index(drop=True)
        if n_before != len(df):
            print(f"   [drop] 非魚種/誤抽出 {n_before - len(df)} 行を除外 (シーブルー/青物/アチコチ等)")

    # 同一 (datetime, site, species) は count を合算
    # None の数値カラムを sum で 0 にしないよう first を使う
    df = df.groupby(
        ["datetime", "site", "species", "boat"], dropna=False, as_index=False
    ).agg({
        "count": "sum",
        "count_yolo": "sum",
        "top_per_angler": "max",     # 個人最大なので max
        "total_catch": "sum",        # 船全体の合計
        "max_size_cm": "max",
        "qualitative": "first",
        "anglers": "first",
        "target_species": "first",
        "tackle": "first",
        "departure_hour": "first",
        "total_weight_g": "first",
        "avg_length_cm": "first",
        "angler": "first",
        "notes": lambda x: " | ".join(sorted(set(str(v) for v in x if v))),
        "image_path": "first",
        "entry_title": "first",
    })
    df = df.sort_values("datetime").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ {len(df)} 行 / {df['datetime'].nunique()} trip 日付 を {out} に書き出し")
    if title_map:
        print(f"   タイトル由来 target_species: {n_target_from_title}/{n_summaries - n_skipped} エントリ")
    print(f"   船宿別 行数:")
    for b, n in sorted(boat_distribution.items(), key=lambda x: -x[1]):
        print(f"     {b}: {n}")
    # 魚種別サマリ: trips（観測 trip 数）優先、各シグナルの集計値も併記
    # 旧コードは count（YOLO のみ）を sum() していたため、
    # qualitative-only の魚種が「0.0」と表示されてしまっていた。
    sp_grp = df.groupby("species", dropna=False)
    sp_summary = pd.DataFrame({
        "trips": sp_grp.size(),
        "top_max": sp_grp["top_per_angler"].max(),
        "total_sum": sp_grp["total_catch"].sum(min_count=1),
        "yolo_sum": sp_grp["count_yolo"].sum(min_count=1),
        "qual_n": sp_grp["qualitative"].count(),
    }).sort_values("trips", ascending=False)
    print(f"   魚種別サマリ (trips=記録された trip 数):")
    print(sp_summary.head(30).fillna("-").to_string())
    return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="summary.json を catches.csv に集約")
    parser.add_argument("--scraped", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--site", default="irago", choices=list(config.SITES))
    parser.add_argument("--boat", default="maruman2010")
    parser.add_argument("--departure-hour", type=int, default=DEFAULT_DEPARTURE_HOUR)
    args = parser.parse_args()

    aggregate(
        scraped_dir=args.scraped, out_path=args.out,
        site=args.site, boat=args.boat,
        departure_hour=args.departure_hour,
    )


if __name__ == "__main__":
    _cli()
