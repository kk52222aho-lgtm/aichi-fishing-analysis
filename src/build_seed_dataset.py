"""釣り船ブログ 1 軒の過去エントリを巡回して、釣果ログの seed データを作る。

aichi-fishing-analysis 専用の orchestrator。
GenesisEngine-v6（自動モデル学習）とは独立。本モジュールは
GenesisEngine が出した best.pt を「使う」側であって、学習はしない。

処理パイプライン:
    1. blog_scraper.list_entries(blog_id, months_back)
       → ameblo の entrylist から URL/投稿日時を集める
    2. 各エントリで predict_from_url.run(url)
       → data/scraped/<slug>/summary.json + 可視化画像を作る
    3. scrape_to_catches.aggregate(site, boat)
       → data/fishing_logs/catches.csv に集約

Colab で実行する想定:
    from src.build_seed_dataset import build
    df = build(blog_id="maruman2010", site="irago", months_back=6)

CLI（GPU が無いと YOLO 推論で落ちる点に注意）:
    python -m src.build_seed_dataset --blog maruman2010 --site irago --months 6
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from . import blog_scraper, config, scrape_to_catches

# 釣果が載らないエントリのタイトルパターン（予約案内・お知らせ・船長の私的釣行等）
DEFAULT_SKIP_TITLE_KEYWORDS: tuple[str, ...] = (
    "予約状況", "予約", "お知らせ", "休業", "中止", "キャンセル",
    "本日休", "明日", "募集",
    "日帰り", "プライベート", "オフ",  # シーブルー等の私的釣行記事
)


def _filter_entries_by_title(
    entries: list[blog_scraper.Entry],
    skip_keywords: tuple[str, ...] = DEFAULT_SKIP_TITLE_KEYWORDS,
) -> tuple[list[blog_scraper.Entry], int]:
    keep: list[blog_scraper.Entry] = []
    n_skip = 0
    for e in entries:
        title = e.title or ""
        if any(kw in title for kw in skip_keywords):
            n_skip += 1
            continue
        keep.append(e)
    return keep, n_skip


def _slug_for_url(url: str) -> str:
    """predict_from_url._slug_for_url と整合させる。"""
    from .predict_from_url import _slug_for_url as _slug
    return _slug(url)


def _process_entries(
    entries: list[blog_scraper.Entry],
    conf: float,
    skip_existing: bool,
    sleep_sec: float,
    no_viz: bool,
) -> None:
    """各エントリを predict_from_url.run() に流す。"""
    # 遅延 import: build_seed_dataset 自体は YOLO 不要で読めるようにする
    from .predict_from_url import run as predict_url

    scraped_root = config.DATA_DIR / "scraped"
    n = len(entries)
    for i, e in enumerate(entries, 1):
        slug = _slug_for_url(e.url)
        summary_path = scraped_root / slug / "summary.json"
        if skip_existing and summary_path.exists():
            print(f"[{i}/{n}] skip (already scraped): {e.url}")
            continue

        print(f"\n[{i}/{n}] {e.posted_at.date()} {e.title[:60]}")
        try:
            predict_url(e.url, conf=conf, no_viz=no_viz, no_show=True)
        except KeyboardInterrupt:
            print("⏹ 中断。途中までの結果は data/scraped/ に保存済み。")
            raise
        except Exception as exc:
            print(f"  ⚠️ error: {exc}")
        time.sleep(sleep_sec)


def build(
    blog_id: str = "maruman2010",
    site: str = "irago",
    boat: Optional[str] = None,
    months_back: int = 6,
    conf: float = 0.30,
    limit: Optional[int] = None,
    skip_existing: bool = True,
    sleep_sec: float = 1.2,
    no_viz: bool = False,
    out_path: Optional[Path] = None,
    skip_title_keywords: Optional[tuple[str, ...]] = DEFAULT_SKIP_TITLE_KEYWORDS,
    use_llm_extract: bool = False,
    llm_provider: str = "groq",
    llm_model: Optional[str] = None,
    llm_fallback_provider: Optional[str] = None,
    run_yolo: bool = True,
    primary_signal: Optional[str] = None,
    secondary_signal: Optional[str] = None,
) -> pd.DataFrame:
    """エントリ列挙 → YOLO 推論 → catches.csv 集約 までを一気通貫。

    Args:
        blog_id: ameblo ブログ ID（例: "maruman2010"）
        site: 釣果ログに固定で入れる site コード（例: "irago"）
        boat: 船宿名（default: blog_id）
        months_back: 何ヶ月前まで遡るか
        conf: YOLO の conf 閾値
        limit: 処理するエントリ数の上限（テスト用）
        skip_existing: summary.json が既にあるエントリは飛ばす
        sleep_sec: エントリ間のスリープ
        no_viz: bbox 可視化画像を作らない（速い・容量節約）
        skip_title_keywords: タイトルにこのキーワードを含むエントリは
            釣果報告でない（予約状況・お知らせ等）とみなして除外。
            None を渡せば全エントリ処理。

    Returns:
        作成した catches.csv の DataFrame
    """
    boat = boat or blog_id
    if site not in config.SITES:
        raise ValueError(f"unknown site: {site}. available: {list(config.SITES)}")

    # registry に登録（aggregate 時に各エントリで boat/site を正しく解決するため）
    scrape_to_catches.register_blog(
        blog_id, boat=boat, site=site,
        primary_signal=primary_signal,
        secondary_signal=secondary_signal,
    )

    print(f"🔎 {blog_id} の過去 {months_back} ヶ月のエントリを列挙...")
    entries = blog_scraper.list_entries(blog_id, months_back=months_back)
    print(f"   {len(entries)} entries")

    if skip_title_keywords:
        entries, n_skipped = _filter_entries_by_title(entries, skip_title_keywords)
        if n_skipped:
            print(f"   ↳ タイトルフィルタで {n_skipped} 件除外 → 残 {len(entries)} 件")

    if limit:
        entries = entries[:limit]
        print(f"   ↳ limit={limit} で {len(entries)} 件に絞り込み")

    # posted_at と title は summary.json には残らないので、URL→値マップで補助
    posted_at_map = {e.url: e.posted_at for e in entries}
    title_map = {e.url: e.title for e in entries}

    if run_yolo:
        _process_entries(
            entries=entries, conf=conf,
            skip_existing=skip_existing, sleep_sec=sleep_sec, no_viz=no_viz,
        )
    else:
        # YOLO スキップ。summary.json が無い船宿は本文だけ取りに行く動きにする
        # 後段の text extractor が動けば catch_info は得られる
        print("   ↳ run_yolo=False: YOLO 推論をスキップ")
        from . import predict_from_url
        from .predict_from_url import _slug_for_url
        for i, e in enumerate(entries, 1):
            slug = _slug_for_url(e.url)
            sdir = config.DATA_DIR / "scraped" / slug
            sjson = sdir / "summary.json"
            if skip_existing and sjson.exists():
                continue
            # ダミーの summary.json を作る（YOLO 検出 0 件のエントリと同等）
            sdir.mkdir(parents=True, exist_ok=True)
            with sjson.open("w", encoding="utf-8") as f:
                json.dump({
                    "source_url": e.url,
                    "conf_threshold": conf,
                    "n_image_urls": 0,
                    "n_processed": 0,
                    "total_per_species": {},
                    "results": [],
                }, f, ensure_ascii=False, indent=2)

    # 本文 + LLM 抽出（任意）— 後段の aggregate が読む text_extracted.json を作る
    print(f"\n📄 本文抽出（use_llm_fallback={use_llm_extract}）...")
    from . import blog_text_extractor
    urls_to_process = [e.url for e in entries]
    blog_text_extractor.process_batch(
        urls=urls_to_process,
        skip_existing=skip_existing,
        sleep_sec=0.4,
        use_llm_fallback=use_llm_extract,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_fallback_provider=llm_fallback_provider,
        posted_at_map=posted_at_map,
    )

    print(f"\n📦 catches.csv に集約中...")
    df = scrape_to_catches.aggregate(
        site=site, boat=boat,
        out_path=out_path,
        posted_at_map=posted_at_map,
        title_map=title_map,
    )

    # メタ情報も保存（再現用）
    meta_path = config.FISHING_DIR / f"seed_meta_{blog_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "built_at": datetime.now().isoformat(),
            "blog_id": blog_id,
            "site": site,
            "boat": boat,
            "months_back": months_back,
            "conf": conf,
            "n_entries_listed": len(entries),
            "n_rows": int(len(df)),
            "entries": [e.to_dict() for e in entries],
        }, f, ensure_ascii=False, indent=2)
    print(f"📝 meta: {meta_path}")

    return df


def re_aggregate(
    blog_id: str = "maruman2010",
    site: str = "irago",
    boat: Optional[str] = None,
    months_back: int = 12,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """既存スクレイプ済みデータを再集約のみ（YOLO を再実行しない）。

    用途: scrape_to_catches や title_parser を改修した後、
    YOLO 推論を回し直さずに catches.csv だけ作り直したいとき。

    months_back はタイトル取得のためだけに使う（広めに取って取りこぼし防止）。
    """
    boat = boat or blog_id
    print(f"🔎 タイトル取得のため {blog_id} の過去 {months_back} ヶ月をリスト...")
    entries = blog_scraper.list_entries(blog_id, months_back=months_back)
    print(f"   {len(entries)} entries")

    posted_at_map = {e.url: e.posted_at for e in entries}
    title_map = {e.url: e.title for e in entries}

    print(f"\n📦 catches.csv に集約中（YOLO は実行しない）...")
    return scrape_to_catches.aggregate(
        site=site, boat=boat,
        out_path=out_path,
        posted_at_map=posted_at_map,
        title_map=title_map,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="釣り船ブログから seed catches.csv を作る")
    parser.add_argument("--blog", default="maruman2010", help="ameblo blog id")
    parser.add_argument("--site", default=None, choices=list(config.SITES),
                        help="default: registry から自動解決、無ければ irago")
    parser.add_argument("--boat", default=None,
                        help="船宿名 (default: registry から自動解決、無ければ blog_id)")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="既存 summary.json も再処理する")
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--no-viz", action="store_true")
    # LLM 抽出 / YOLO スキップ flags
    parser.add_argument("--use-llm-extract", action="store_true",
                        help="本文からの LLM 抽出を有効化 (top_per_angler 精度向上)")
    parser.add_argument("--llm-provider", default="cerebras",
                        choices=["cerebras", "groq", "gemini", "ollama"],
                        help="LLM 抽出時のプロバイダ (default: cerebras 無料)")
    parser.add_argument("--llm-model", default=None,
                        help="LLM モデル指定 (None なら provider default)")
    parser.add_argument("--llm-fallback", default=None,
                        help="一次失敗時の fallback provider (default: なし、課金 gemini 試行回避)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="YOLO 推論をスキップ (LLM 抽出だけで catches.csv 生成)")
    args = parser.parse_args()

    # registry から site/boat を解決 (CLI 引数が None のときだけ)
    site = args.site
    boat = args.boat
    if site is None or boat is None:
        try:
            reg = scrape_to_catches.load_blog_registry()
            entry = reg.get(args.blog, {})
            site = site or entry.get("site")
            boat = boat or entry.get("boat")
        except Exception:
            pass
    site = site or "irago"

    build(
        blog_id=args.blog, site=site, boat=boat,
        months_back=args.months, conf=args.conf, limit=args.limit,
        skip_existing=not args.no_skip_existing,
        sleep_sec=args.sleep, no_viz=args.no_viz,
        use_llm_extract=args.use_llm_extract,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_fallback_provider=args.llm_fallback,
        run_yolo=not args.no_yolo,
    )


if __name__ == "__main__":
    _cli()
