"""data/scraped/<slug>/summary.json を v6 best.pt で再計算して上書き。

【目的】
旧 best.pt (5/13 版, 18 クラス) で生成された summary.json を、
v6 best.pt (21 クラス, mAP50=0.611) の検出結果で上書きする。

【対象】
data/scraped/<slug>/images/ に画像があり summary.json が「実 summary」(n_processed > 0)
の dir のみ。run_yolo=False で作られたダミー summary は対象外。

【挙動】
1. 既存 summary.json を summary.json.bak_pre_v6 に退避 (既存があればスキップ)
2. images/ の全画像で v6.detect_all() を実行
3. 元の構造 (source_url, conf_threshold, results 配列) を保ちつつ
   total_per_species と results[].detections を v6 値で書き換える
4. viz 画像は再生成しない (元のは古いが、aggregate には影響しない)

【独立性】
- catches.csv は触らない (user が後で scrape_to_catches.aggregate() を回せば反映)
- 並走ワーカー (features.py / fetchers) と完全に縄張りが違う

【実行】
    python notebooks/rerun_summaries_with_v6.py
    # dry-run で件数だけ見る:
    python notebooks/rerun_summaries_with_v6.py --dry
    # 1 dir だけ試す:
    python notebooks/rerun_summaries_with_v6.py --limit 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# プロジェクトルート (notebooks/ の親)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.yolo_predictor import FishSpeciesPredictor  # noqa: E402

SCRAPED = config.DATA_DIR / "scraped"
BAK_SUFFIX = ".bak_pre_v6"


def is_real_summary(info: dict) -> bool:
    """ダミー summary (run_yolo=False で作成) を除外。"""
    return info.get("n_image_urls", 0) > 0 or info.get("n_processed", 0) > 0


def collect_targets() -> list[Path]:
    """対象 dir = images/ あり & summary.json が real。"""
    targets: list[Path] = []
    for d in SCRAPED.iterdir():
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            info = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not is_real_summary(info):
            continue
        img_dir = d / "images"
        if not img_dir.exists() or not any(img_dir.iterdir()):
            continue
        targets.append(d)
    return targets


def rerun_one(d: Path, predictor: FishSpeciesPredictor, conf: float) -> dict:
    """1 dir の summary.json を v6 で更新。"""
    sj = d / "summary.json"
    bak = sj.with_suffix(sj.suffix + BAK_SUFFIX)
    info = json.loads(sj.read_text(encoding="utf-8"))

    # バックアップ (既存はスキップ = 多重実行で原本を失わない)
    if not bak.exists():
        bak.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    img_dir = d / "images"
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # 元 results の image_url を local パスで引けるよう map 化 (URL 保持のため)
    url_by_local = {}
    for r in info.get("results", []):
        loc = r.get("local")
        if loc:
            url_by_local[Path(loc).name] = r.get("image_url", "")

    new_results = []
    total_per_species: dict[str, int] = {}
    for img in images:
        det = predictor.detect_all(img)
        per_sp = det.count_per_species
        for sp, c in per_sp.items():
            total_per_species[sp] = total_per_species.get(sp, 0) + c
        new_results.append({
            "image_url": url_by_local.get(img.name, ""),
            "local": str(img.relative_to(ROOT)).replace("\\", "/"),
            "total": det.total,
            "count_per_species": per_sp,
            "detections": [
                {
                    "species": p.species,
                    "confidence": p.confidence,
                    "bbox": list(p.bbox) if p.bbox else None,
                }
                for p in det.detections
            ],
        })

    new_info = {
        "source_url": info.get("source_url", ""),
        "conf_threshold": conf,
        "n_image_urls": info.get("n_image_urls", len(images)),
        "n_processed": len(images),
        "total_per_species": total_per_species,
        "results": new_results,
        "_model_version": "v6_finetune_v6_2026-05-22",
    }
    sj.write_text(json.dumps(new_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dir": d.name,
        "n_images": len(images),
        "n_detections": sum(r["total"] for r in new_results),
        "species": list(total_per_species.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=None, help="先頭 N dir のみ処理 (テスト用)")
    ap.add_argument("--dry", action="store_true", help="対象 dir 数だけ表示して終了")
    args = ap.parse_args()

    targets = collect_targets()
    print(f"📁 対象 dir: {len(targets)}")
    if args.dry:
        for t in targets[:5]:
            print(f"  - {t.name}")
        if len(targets) > 5:
            print(f"  ... 他 {len(targets) - 5}")
        return

    if args.limit:
        targets = targets[:args.limit]
        print(f"   ↳ limit={args.limit} で絞り込み")

    pred = FishSpeciesPredictor(
        weights_path=config.YOLO_DEFAULT_WEIGHTS, conf_threshold=args.conf,
    )
    if not pred.is_ready:
        print(f"❌ YOLO モデル未ロード: {pred.weights_path}")
        sys.exit(1)
    print(f"✅ v6 best.pt ロード: {pred.weights_path}")

    t0 = time.time()
    totals = {"images": 0, "detections": 0}
    species_freq: dict[str, int] = {}
    for i, d in enumerate(targets, 1):
        try:
            r = rerun_one(d, pred, args.conf)
            totals["images"] += r["n_images"]
            totals["detections"] += r["n_detections"]
            for sp in r["species"]:
                species_freq[sp] = species_freq.get(sp, 0) + 1
            elapsed = time.time() - t0
            eta = elapsed / i * (len(targets) - i)
            print(f"[{i:3}/{len(targets)}] {r['dir'][:40]:40} "
                  f"imgs={r['n_images']:2} det={r['n_detections']:3} "
                  f"sp={','.join(r['species'])[:30]:30} "
                  f"({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"[{i:3}/{len(targets)}] {d.name}: ❌ {e}", flush=True)

    print()
    print("=" * 70)
    print(f"✅ 完了 {totals['images']} 画像 / {totals['detections']} 検出 / {(time.time()-t0)/60:.1f} 分")
    print()
    print("検出された種 (dir 数):")
    for sp, n in sorted(species_freq.items(), key=lambda x: -x[1]):
        marker = " ⭐ NEW (v6 で初検出)" if sp in {"ホウボウ", "オコゼ", "サワラ"} else ""
        print(f"  {sp:<14} : {n} dir{marker}")
    print()
    print("[次のステップ]")
    print("  python -m src.scrape_to_catches  # catches.csv に反映")


if __name__ == "__main__":
    main()
