"""全船宿を一括で最新まで追いつかせる更新スクリプト。

これまで「各船宿を scrape → 最後に全船宿まとめて再集約」という手順が
Colab のセル上にしか存在せず、手を止めると catches.csv が古いまま止まっていた。
（実際 2026-06-26 で 55 日ぶん止まっていた。）その手順をリポジトリ側に固定するのが本モジュール。

ローカルで catches.csv を「作り直しては」いけない:
    scrape_to_catches.aggregate() は data/scraped/ 配下 *全部* を読み直して
    catches.csv を丸ごと書き直す。ところが data/scraped/ は .gitignore 対象で、
    commit されているのは成果物の catches.csv だけ。つまり手元の scrape キャッシュは
    Colab で作った完全版の一部でしかない。
    実測 2026-08-20: 手元 1,863 エントリからの全面再構築は 4,444 行で、
    commit 済み 9,846 行の半分以下だった（石川丸 2,345 行 → 1 行）。
    そのため本モジュールは集約結果を tmp に出し、既存 catches.csv には
    merge_into_catches() で *新しいキーの行だけ追記* する。上書きはしない。

使い方:
    python -m src.update_all                  # 全船宿を過去3ヶ月ぶん更新して追記
    python -m src.update_all --months 6
    python -m src.update_all --blogs ishikawamaru matobaya
    python -m src.update_all --aggregate-only # スクレイプせず集約+追記だけ
    python -m src.update_all --dry-run        # 列挙だけして書き込まない
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from . import blog_scraper, build_seed_dataset, config, scrape_to_catches

def _maps_from_disk() -> tuple[dict[str, datetime], dict[str, str]]:
    """data/scraped/*/text_extracted.json から url -> posted_at / title を復元する。

    aggregate() は data/scraped 配下 *全部* を読むのに、日付は posted_at_map 頼り。
    その map を「今回の実行が何ヶ月遡ったか」に依存させると、窓の外の過去エントリが
    まるごと日付不明で落ちる（months=3 で回して 9,846 行 → 4,444 行になった）。
    抽出時に posted_at/title はディスクに保存済みなので、そこから全期間ぶんを復元する。
    """
    posted: dict[str, datetime] = {}
    titles: dict[str, str] = {}
    root = config.DATA_DIR / "scraped"
    n_bad = 0
    for tp in root.glob("*/text_extracted.json"):
        try:
            with tp.open("r", encoding="utf-8") as f:
                d = json.load(f)
            url = d.get("url")
            if not url:
                continue
            pa = d.get("posted_at")
            if pa:
                posted[url] = datetime.fromisoformat(pa)
            if d.get("title"):
                titles[url] = d["title"]
        except Exception:
            n_bad += 1
    print(f"🗂️ ディスクから復元: posted_at {len(posted)} 件 / title {len(titles)} 件"
          + (f"（読めず {n_bad} 件）" if n_bad else ""))
    return posted, titles


# 追記マージのキー。同じ (日時, 場所, 船宿, 魚種) は既存行を勝たせる。
MERGE_KEY = ["datetime", "site", "boat", "species"]


def _snapshot_seed_meta() -> dict[Path, str]:
    """seed_meta_*.json の中身を退避する。

    build_seed_dataset.build() は seed_meta_<blog>.json を「今回列挙した窓」で
    まるごと上書きする。seed 構築時は 6〜12 ヶ月ぶんの来歴が入っているので、
    3 ヶ月窓の定期更新でそのまま呼ぶと過去の来歴が消える
    （実測: ishikawamaru の meta が 10,304 行ぶん削れた）。
    定期更新は「追いつかせる」のが仕事で来歴を書き換える立場に無いので、退避して戻す。
    """
    return {f: f.read_text(encoding="utf-8")
            for f in config.FISHING_DIR.glob("seed_meta_*.json")}


def _restore_seed_meta(snap: dict[Path, str]) -> None:
    n = 0
    for f, body in snap.items():
        if f.read_text(encoding="utf-8") != body:
            f.write_text(body, encoding="utf-8")
            n += 1
    if n:
        print(f"   ↩ seed_meta を {n} 件復元（定期更新では来歴を書き換えない）")


def rebuild_integrated() -> None:
    """catches.csv から integrated.parquet を作り直す。

    predictor / backtest / llm_predictor / Streamlit アプリが読むのはこの parquet で、
    catches.csv を更新しただけでは画面も予測も古いままになる。
    気象・潮汐は外部 API から取り直すので数分かかる。
    """
    from . import data_integrator
    src_csv = config.FISHING_DIR / "catches.csv"
    out = config.INTEGRATED_DIR / "integrated.parquet"
    print(f"\n🔗 integrated.parquet を再構築中（気象・潮汐を取得するので数分かかる）...")
    df = data_integrator.integrate_from_path(src_csv, out)
    print(f"✅ {out}: {len(df)} 行 / {len(df.columns)} 列")


def merge_into_catches(fresh: pd.DataFrame, out: Path, fallback_path: Path) -> pd.DataFrame:
    """ローカル集約結果を既存 catches.csv に *追記* する。

    なぜ上書きでなく追記か:
        data/scraped/ は .gitignore されている（commit されるのは catches.csv だけ）。
        つまり手元の scrape キャッシュは Colab で作った完全版の一部でしかなく、
        aggregate() の全面再構築をローカルで走らせると必ず行が減る。
        実測 2026-08-20: 手元 1,863 エントリからの再構築は 4,444 行で、
        commit 済み 9,846 行の半分以下だった（石川丸 2,345 行 → 1 行）。
        よって「新しいキーの行だけ足す」以外に安全な更新手段が無い。
    """
    if fresh.empty:
        print("⚠️ 集約結果が空。catches.csv は触らない。")
        return pd.DataFrame()

    if not out.exists():
        shutil.copyfile(fallback_path, out)
        print(f"✅ {out} を新規作成（{len(fresh)} 行）")
        return fresh

    base = pd.read_csv(out)
    n_before = len(base)
    missing = [c for c in MERGE_KEY if c not in base.columns or c not in fresh.columns]
    if missing:
        print(f"🛑 マージキー列が無い: {missing}。catches.csv は触らず {fallback_path} に残す。")
        return fresh

    # dtype 差で取りこぼさないよう、キーは文字列に揃えてから突き合わせる
    def _key(df: pd.DataFrame) -> pd.Series:
        return df[MERGE_KEY].astype(str).agg('\x1f'.join, axis=1)

    known = set(_key(base))
    add = fresh[~_key(fresh).isin(known)]
    if add.empty:
        print(f"   新規行なし（既存 {n_before} 行のまま）")
        return base

    merged = pd.concat([base, add], ignore_index=True)
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged.to_csv(out, index=False, encoding="utf-8")

    d_new = pd.to_datetime(add["datetime"], errors="coerce", utc=True)
    d_all = pd.to_datetime(merged["datetime"], errors="coerce", utc=True)
    print(f"✅ {out}: {n_before} 行 → {len(merged)} 行（+{len(add)}）")
    print(f"   追加分の期間 {d_new.min()} 〜 {d_new.max()}")
    print(f"   全体の最新 {d_all.max()} / 最古 {d_all.min()}")
    print("   追加分の船宿別:")
    for boat, n in add["boat"].value_counts().items():
        print(f"     {boat}: {n}")
    return merged


def update_all(
    blogs: Optional[list[str]] = None,
    months_back: int = 3,
    use_llm_extract: bool = True,
    llm_provider: str = "groq",
    sleep_sec: float = 1.0,
    dry_run: bool = False,
    aggregate_only: bool = False,
) -> pd.DataFrame:
    registry = scrape_to_catches.load_blog_registry()
    blog_ids = [] if aggregate_only else (blogs or list(registry))

    # 先にディスク全体から復元し、その上に今回の列挙結果を重ねる
    posted_at_map, title_map = _maps_from_disk()
    seed_meta_snap = _snapshot_seed_meta()
    tmp_dir = Path(tempfile.mkdtemp(prefix="catches_scratch_"))
    stats: list[tuple[str, int, str]] = []

    try:
        for i, blog_id in enumerate(blog_ids, 1):
            entry = registry.get(blog_id, {})
            site = entry.get("site") or "irago"
            boat = entry.get("boat") or blog_id
            print(f"\n{'='*70}\n[{i}/{len(blog_ids)}] {blog_id} ({boat} / {site})\n{'='*70}")
            try:
                if dry_run:
                    entries = blog_scraper.list_entries(blog_id, months_back=months_back)
                    print(f"   {len(entries)} entries（dry-run: 書き込みなし）")
                    for e in entries:
                        posted_at_map[e.url] = e.posted_at
                        title_map[e.url] = e.title
                    stats.append((blog_id, len(entries), "dry-run"))
                    continue

                # per-blog の集約は捨てる（out_path を tmp に逃がす）。
                # 欲しいのは summary.json / text_extracted.json と posted_at/title。
                build_seed_dataset.build(
                    blog_id=blog_id, site=site, boat=boat,
                    months_back=months_back,
                    skip_existing=True,
                    sleep_sec=sleep_sec,
                    no_viz=True,
                    run_yolo=False,
                    use_llm_extract=use_llm_extract,
                    llm_provider=llm_provider,
                    primary_signal=entry.get("primary_signal"),
                    secondary_signal=entry.get("secondary_signal"),
                    out_path=tmp_dir / f"{blog_id}.csv",
                )
                entries = blog_scraper.list_entries(blog_id, months_back=months_back)
                for e in entries:
                    posted_at_map[e.url] = e.posted_at
                    title_map[e.url] = e.title
                stats.append((blog_id, len(entries), "ok"))
            except Exception as e:
                traceback.print_exc()
                stats.append((blog_id, 0, f"FAIL: {type(e).__name__}: {e}"))

        print(f"\n{'='*70}\n📦 全 {len(blog_ids)} 船宿を 1 回だけ再集約"
              f"（posted_at {len(posted_at_map)} 件）\n{'='*70}")
        for blog_id, n, status in stats:
            print(f"   {blog_id:22s} {n:4d} entries  {status}")

        if dry_run:
            return pd.DataFrame()

        _restore_seed_meta(seed_meta_snap)

        out = config.FISHING_DIR / "catches.csv"
        fresh = scrape_to_catches.aggregate(
            site="irago", boat="UNKNOWN",   # registry で船宿ごとに上書きされる
            out_path=tmp_dir / "catches_new.csv",
            posted_at_map=posted_at_map,
            title_map=title_map,
        )
        return merge_into_catches(fresh, out, tmp_dir / "catches_new.csv")
    finally:
        if not dry_run:
            print(f"（作業ディレクトリ: {tmp_dir}）")


def _cli() -> None:
    p = argparse.ArgumentParser(description="全船宿を最新まで更新して catches.csv を作り直す")
    p.add_argument("--blogs", nargs="*", default=None, help="対象 blog_id（default: registry 全部）")
    p.add_argument("--months", type=int, default=3)
    p.add_argument("--llm-provider", default="groq")
    p.add_argument("--no-llm", action="store_true", help="LLM 抽出を使わない")
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebuild-integrated", action="store_true",
                   help="更新後に integrated.parquet も作り直す（アプリ/予測が読む本体）")
    p.add_argument("--aggregate-only", action="store_true",
                   help="スクレイプせず catches.csv の再集約だけ行う")
    a = p.parse_args()
    update_all(
        blogs=a.blogs, months_back=a.months,
        use_llm_extract=not a.no_llm, llm_provider=a.llm_provider,
        sleep_sec=a.sleep, dry_run=a.dry_run,
        aggregate_only=a.aggregate_only,
    )
    if a.rebuild_integrated and not a.dry_run:
        rebuild_integrated()


if __name__ == "__main__":
    _cli()
