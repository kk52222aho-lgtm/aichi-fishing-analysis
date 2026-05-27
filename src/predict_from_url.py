"""URL (釣り船ブログ等) から画像を抽出して YOLO 統合モデルで魚種検出。

ブログ等の釣果写真ページを 1 URL 渡せば、画像をダウンロード → 検出 → 可視化 →
summary.json まで一気に作る。釣果ログ補完の前段として URL 単位で使える。

使い方:
    python -m src.predict_from_url \\
        https://ameblo.jp/maruman2010/entry-12965183599.html

オプション:
    --conf 0.25       conf 閾値 (default 0.30、学習時 val とドメインズレを許容)
    --out  DIR        出力ディレクトリ (default: data/scraped/<url_slug>/)
    --no_viz          可視化を省略
    --min_size 200    取得画像の最小辺 (アイコン除外用、default 200px)

出力構造:
    data/scraped/<slug>/
      ├ images/       ダウンロード画像
      ├ results/      bbox 可視化済み画像 (matplotlib + 日本語フォント)
      └ summary.json  検出結果サマリー
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

from . import config
from .yolo_predictor import FishSpeciesPredictor, ImageDetections

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
MIN_SIZE_DEFAULT = 200


def _slug_for_url(url: str) -> str:
    """URL の末尾セグメントをディレクトリ名に。なければ MD5 短縮で。"""
    parsed = urlparse(url)
    last = parsed.path.rstrip("/").split("/")[-1]
    if last:
        slug = re.sub(r"[^\w.-]", "_", last)[:64]
        if slug:
            return slug
    return "url_" + hashlib.md5(url.encode()).hexdigest()[:12]


def fetch_image_urls(page_url: str) -> list[str]:
    """ページから <img src/data-src> を収集 (重複除去・絶対URL化)."""
    resp = requests.get(page_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen: set[str] = set()
    urls: list[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        full = urljoin(page_url, src)
        if full in seen:
            continue
        seen.add(full)
        urls.append(full)
    return urls


def download_image(img_url: str, dest_path: Path,
                   min_size: int = MIN_SIZE_DEFAULT) -> Path | None:
    """画像 DL。min_size 未満の辺を持つ画像 (アイコン等) は None を返す。"""
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pil = Image.open(io.BytesIO(r.content)).convert("RGB")
        if pil.width < min_size or pil.height < min_size:
            return None
        pil.save(dest_path, "JPEG", quality=92)
        return dest_path
    except Exception:
        return None


def visualize_detection(img_path: Path, det: ImageDetections,
                        out_path: Path) -> bool:
    """matplotlib で bbox + 日本語ラベル描画 (font_setup 経由で日本語フォント解決)."""
    try:
        from . import font_setup
        font_setup.apply()
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return False

    pil = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil)
    ax.axis("off")

    colors = ["lime", "red", "cyan", "yellow", "magenta",
              "orange", "lightgreen", "deepskyblue", "violet"]

    for i, d in enumerate(det.detections):
        if d.bbox is None:
            continue
        x1, y1, x2, y2 = d.bbox
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2.5, edgecolor=color,
                                 facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 5),
            f"{d.species} {d.confidence:.2f}",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=color, alpha=0.85, edgecolor="none"),
            fontsize=11, fontweight="bold", color="black",
            verticalalignment="bottom",
        )

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    return True


def _show_results_inline(viz_paths: list[Path], width: int = 900) -> None:
    """Colab/Jupyter で実行中なら viz 画像をインライン表示。CLI なら何もしない。"""
    if not viz_paths:
        return
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return
        from IPython.display import display, Image as IPImage
    except ImportError:
        return

    print(f"\n🖼️  検出結果画像 ({len(viz_paths)} 枚)")
    for p in viz_paths:
        print(f"  {p.name}")
        display(IPImage(filename=str(p), width=width))


def run(
    url: str,
    conf: float = 0.30,
    out: str | Path | None = None,
    weights: str | Path | None = None,
    no_viz: bool = False,
    no_show: bool = False,
    show_all: bool = False,
    min_size: int = MIN_SIZE_DEFAULT,
) -> dict:
    """URL を処理して検出結果を返す (notebook から呼ぶ用の関数 API)。

    Colab/Jupyter のセル内で
        from src.predict_from_url import run
        run('https://...', conf=0.30)
    のように呼べば、inline 表示が正しく動く (subprocess 経由だと
    IPython カーネルが見えず inline 表示できないため)。

    Returns: summary dict (summary.json と同じ構造)
    """
    # ── 出力先準備 ───────────────────────────────────────
    out_dir = Path(out) if out else (
        config.DATA_DIR / "scraped" / _slug_for_url(url))
    raw_dir = out_dir / "images"
    viz_dir = out_dir / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"🌐 ページ取得: {url}")
    try:
        img_urls = fetch_image_urls(url)
    except Exception as e:
        print(f"❌ ページ取得失敗: {e}")
        return {"source_url": url, "error": str(e)}
    print(f"   {len(img_urls)} 個の <img> URL を抽出")

    # ── モデル準備 ───────────────────────────────────────
    pred = FishSpeciesPredictor(
        weights_path=weights or config.YOLO_DEFAULT_WEIGHTS,
        conf_threshold=conf,
    )
    if not pred.is_ready:
        print(f"❌ YOLO モデル未ロード: {pred.weights_path}")
        print("   ultralytics がインストールされ、weights が存在するか確認")
        return {"source_url": url, "error": "model not loaded"}
    print(f"📂 weights: {pred.weights_path}")

    # ── 各画像を DL → 検出 → 可視化 ─────────────────────
    summary: list[dict] = []
    viz_paths: list[Path] = []
    saved = 0
    for i, img_url in enumerate(img_urls):
        dest = raw_dir / f"img_{i:03d}.jpg"
        path = download_image(img_url, dest, min_size)
        if path is None:
            continue
        saved += 1

        det = pred.detect_all(path)
        print(f"\n[{saved}] {Path(img_url).name}")
        entry = {
            "image_url": img_url,
            "local": str(path.relative_to(config.PROJECT_ROOT)
                         if path.is_absolute() and config.PROJECT_ROOT in path.parents
                         else path),
            "total": det.total,
            "count_per_species": det.count_per_species,
        }
        if det.total == 0:
            print("   → 検出なし")
            if show_all:
                viz_paths.append(path)
        else:
            for d in det.detections:
                print(f"   🐟 {d.species:<6} conf={d.confidence:.3f}")
            entry["detections"] = [
                {"species": d.species, "confidence": d.confidence,
                 "bbox": list(d.bbox) if d.bbox else None}
                for d in det.detections
            ]
            if not no_viz:
                viz_path = viz_dir / f"{path.stem}_viz.jpg"
                ok = visualize_detection(path, det, viz_path)
                if ok:
                    entry["viz"] = str(viz_path)
                    viz_paths.append(viz_path)

        summary.append(entry)

    # ── サマリー出力 ─────────────────────────────────────
    totals: dict[str, int] = {}
    for e in summary:
        for sp, c in e.get("count_per_species", {}).items():
            totals[sp] = totals.get(sp, 0) + c

    result_data = {
        "source_url":     url,
        "conf_threshold": conf,
        "n_image_urls":   len(img_urls),
        "n_processed":    saved,
        "total_per_species": totals,
        "results":        summary,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"✅ 処理完了: {saved}枚処理 ({len(img_urls)}個のURLから)")
    print(f"📝 summary  : {summary_path}")
    print(f"🖼️  raw     : {raw_dir}")
    if not no_viz:
        print(f"🎨 viz     : {viz_dir}")
    if totals:
        print(f"\n📊 全画像合計検出 (種別)")
        for sp, c in sorted(totals.items(), key=lambda x: -x[1]):
            print(f"   {sp:<6} : {c}")
    else:
        print(f"\n⚠️  検出 0 件。conf を下げる (例 0.20) と取れる可能性あり")

    # ── Colab/Jupyter なら viz 画像をインライン表示 ─────
    if not no_show:
        _show_results_inline(viz_paths)

    result_data["viz_paths"] = [str(p) for p in viz_paths]
    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="釣り船ブログ等の URL から画像を抽出 → YOLO 検出")
    parser.add_argument("url", help="対象 URL")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="conf 閾値 (default 0.30 — ドメインズレ許容で緩め)")
    parser.add_argument("--out", default=None,
                        help="出力ディレクトリ (default: data/scraped/<slug>/)")
    parser.add_argument("--no_viz", action="store_true", help="可視化を省略")
    parser.add_argument("--no_show", action="store_true",
                        help="Colab 等でも inline 表示しない")
    parser.add_argument("--show_all", action="store_true",
                        help="検出 0 件の画像も inline 表示")
    parser.add_argument("--min_size", type=int, default=MIN_SIZE_DEFAULT,
                        help=f"取得画像の最小辺 px (default {MIN_SIZE_DEFAULT})")
    parser.add_argument("--weights", default=None,
                        help="YOLO 重みパス (default: config.YOLO_DEFAULT_WEIGHTS)")
    args = parser.parse_args()

    res = run(
        url=args.url,
        conf=args.conf,
        out=args.out,
        weights=args.weights,
        no_viz=args.no_viz,
        no_show=args.no_show,
        show_all=args.show_all,
        min_size=args.min_size,
    )
    if "error" in res:
        sys.exit(1)


if __name__ == "__main__":
    main()
