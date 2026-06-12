"""Colab で 1 コマンド実行するための修復 + 大進丸スクレイプスクリプト。

使い方:
    cd /content/aichi-fishing-analysis
    git pull
    python -m src.fix_and_scrape

このスクリプトは:
1. _blog_registry.json を正しい内容に強制上書き
2. catches.csv の boat 名を英→日に訂正 (masahoumaru → 政宝丸 等)
3. daishinmaru 用 custom dispatcher が動くか直接 Python から検証
4. 大進丸ブログを fresh scrape (--no-skip-existing 相当)
5. 結果サマリを print
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

REGISTRY_TARGET = {
    "maruman2010": {
        "boat": "maruman2010", "site": "irago",
        "primary_signal": "top_per_angler", "secondary_signal": "count_yolo",
    },
    "go-arimotomaru": {
        "boat": "ありもと丸", "site": "utsumi_shinko",
        "primary_signal": "qualitative", "secondary_signal": "total_catch",
    },
    "takemaru-gt-4989": {
        "boat": "武丸", "site": "morozaki",
        "primary_signal": "count_yolo",
    },
    "matobaya": {
        "boat": "まとばや", "site": "morozaki",
        "primary_signal": "qualitative",
    },
    "seablue-0908": {
        "boat": "シーブルー", "site": "morozaki",
        "primary_signal": "qualitative",
    },
    "masahoumaru": {
        "boat": "政宝丸", "site": "shinojima",
        "primary_signal": "total_catch", "secondary_signal": "qualitative",
    },
    "posukemaru": {
        "boat": "ぽん助丸", "site": "akabane",
        "primary_signal": "qualitative", "secondary_signal": "total_catch",
    },
    "daishinmaru": {
        "boat": "大進丸", "site": "toyohama",
        "primary_signal": "qualitative", "secondary_signal": "total_catch",
        "blog_url": "https://daishinmaru.jp/fishing/",
        "blog_platform": "custom",
    },
}

BOAT_RENAME = {
    "masahoumaru": "政宝丸",
    "posukemaru": "ぽん助丸",
    "daishinmaru": "大進丸",
}


def step_1_force_registry() -> None:
    from . import config
    reg_path = config.DATA_DIR / "scraped" / "_blog_registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: Force registry ===")
    if reg_path.exists():
        try:
            current = json.loads(reg_path.read_text(encoding="utf-8"))
            print(f"  現状: {len(current)} entries")
            for k in current:
                cur = current[k]
                print(f"    {k}: boat={cur.get('boat')}, "
                      f"platform={cur.get('blog_platform', 'ameblo')}")
        except Exception as e:
            print(f"  読込失敗: {e}")
    else:
        print("  (ファイル無し)")

    reg_path.write_text(
        json.dumps(REGISTRY_TARGET, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  → 上書き完了: {len(REGISTRY_TARGET)} entries")
    print(f"  daishinmaru の blog_platform = "
          f"{REGISTRY_TARGET['daishinmaru']['blog_platform']}")


def step_2_rename_catches() -> None:
    import pandas as pd
    from . import config
    path = config.FISHING_DIR / "catches.csv"

    print("\n=== STEP 2: Rename boat in catches.csv ===")
    if not path.exists():
        print("  catches.csv が無い、スキップ")
        return

    df = pd.read_csv(path)
    total_renamed = 0
    for old, new in BOAT_RENAME.items():
        mask = df["boat"] == old
        n = int(mask.sum())
        if n:
            df.loc[mask, "boat"] = new
            total_renamed += n
            print(f"  {old} → {new}: {n} 行")
    if total_renamed:
        df.to_csv(path, index=False)
        print(f"  合計 {total_renamed} 行訂正")
    else:
        print("  訂正対象なし (既に日本名 or 該当船宿無し)")


def step_3_verify_dispatch() -> None:
    print("\n=== STEP 3: daishinmaru dispatch test ===")
    # モジュールキャッシュ回避
    for mod_name in ("src.scrape_to_catches", "src.blog_scraper",
                     "src.blog_text_extractor"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from . import blog_scraper, blog_text_extractor

    platform, blog_url = blog_scraper._registry_platform("daishinmaru")
    print(f"  _registry_platform('daishinmaru') -> "
          f"platform={platform}, url={blog_url}")
    assert platform == "custom", f"DISPATCH 失敗: platform={platform}"

    # 直接 list_entries を呼んでみる
    entries = blog_scraper.list_entries("daishinmaru", months_back=3)
    print(f"  list_entries: {len(entries)} 件")
    for e in entries[:3]:
        print(f"    {e.posted_at.date()}  {e.entry_id[:50]}")
    assert len(entries) > 0, "entries 0 件 (custom scraper 失敗)"

    # 1 件本文取得テスト
    body = blog_text_extractor.fetch_body(entries[0].url)
    if body:
        print(f"  fetch_body OK: title={body['title'][:50]}, "
              f"body_len={len(body['body_text'])}")
    else:
        print("  ⚠️ fetch_body 失敗")


def step_4_scrape(blog: str, months: int = 6) -> None:
    print(f"\n=== STEP 4: scrape {blog} (months={months}) ===")
    # サブプロセスで実行（モジュールキャッシュ完全リセット）
    result = subprocess.run(
        [
            sys.executable, "-m", "src.build_seed_dataset",
            "--blog", blog,
            "--months", str(months),
            "--use-llm-extract",
            "--llm-provider", "cerebras",
            "--no-yolo",
            "--no-skip-existing",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ⚠️ exit code {result.returncode}")


def step_5_summary() -> None:
    import pandas as pd
    from . import config
    print("\n=== STEP 5: 最終サマリ ===")
    df = pd.read_csv(config.FISHING_DIR / "catches.csv")
    print(f"全体: {len(df)} 行, {df['datetime'].nunique()} trip 日付")
    print("\n船宿別 行数:")
    for boat, n in df.groupby("boat").size().sort_values(ascending=False).items():
        print(f"  {boat}: {n}")

    print("\n新追加 3 船宿:")
    for boat in ("政宝丸", "ぽん助丸", "大進丸"):
        sub = df[df["boat"] == boat]
        n = len(sub)
        n_top = sub["top_per_angler"].notna().sum() if "top_per_angler" in sub else 0
        n_tot = sub["total_catch"].notna().sum() if "total_catch" in sub else 0
        n_qual = sub["qualitative"].notna().sum() if "qualitative" in sub else 0
        print(f"  {boat}: {n} 行 (top_per_angler={n_top}, "
              f"total_catch={n_tot}, qualitative={n_qual})")


def main() -> None:
    step_1_force_registry()
    step_2_rename_catches()
    try:
        step_3_verify_dispatch()
    except AssertionError as e:
        print(f"\n⚠️ STEP 3 失敗: {e}")
        print("daishinmaru の scrape はスキップして STEP 5 へ")
        step_5_summary()
        return

    step_4_scrape("daishinmaru", months=6)
    step_5_summary()


if __name__ == "__main__":
    main()
