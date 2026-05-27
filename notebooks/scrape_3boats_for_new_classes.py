"""Colab セル: ホウボウ・サワラ多発 3 船宿を v6 で再 scrape。

【目的】
catches.csv にあるホウボウ 174 件・サワラ 66 件はテキストのみで image_path 空。
新3クラス (cls 18/19/20) を含む v6 モデルを deploy 済 (data/models/yolo_unified/best.pt) なので、
これら船宿の過去エントリを画像取得 + YOLO 推論で再処理し、image_path 付き行を増やす。

【対象 3 船宿 (catches.csv 集計より)】
  まとばや (matobaya / morozaki)     : ホウボウ 73 + サワラ 31 = 104 件
  ありもと丸 (go-arimotomaru / utsumi_shinko) : ホウボウ 50 + サワラ 21 = 72 件
  シーブルー (seablue-0908 / morozaki)        : ホウボウ 51 + サワラ 14 = 65 件
  → 計 241 件の新3クラス候補が画像化される見込み

【ポイント】
  - skip_existing=False  : 既存ダミー summary.json (run_yolo=False で作成) を強制再処理
  - run_yolo=True        : 画像取得 + YOLO 推論を有効
  - no_viz=True          : bbox 可視化は不要 (Drive 容量節約)
  - use_llm_extract=True : 本文 LLM 抽出も並行 (テキスト由来の species/count を保持)
  - conf=0.30            : 新クラスは弱いので低めに

【Colab 実行】
  GPU ランタイム + Drive マウント済み前提
  Runtime → Change runtime type → T4 GPU
"""
# ============================================================
# セル 1: マウント & 環境変数セット
# ============================================================
"""
from google.colab import drive, userdata
drive.mount('/content/drive', force_remount=False)

import os
# subprocess からも見えるよう env に export (GenesisEngine と同じ trick)
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "GOOGLE_API_KEY"):
    try:
        os.environ[k] = userdata.get(k)
        print(f"  {k}: ✅ env に設定")
    except Exception as e:
        print(f"  {k}: ⚠️  {e}")
"""

# ============================================================
# セル 2: 依存ライブラリ確認 (足りなければ pip install)
# ============================================================
"""
!pip install -q ultralytics icrawler beautifulsoup4 lxml requests pandas pyyaml
"""

# ============================================================
# セル 3: progress checkpoint ファイル準備 (Colab セッション切れリカバリ)
# ============================================================
"""
import json
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/aichi-fishing-analysis')
CKPT = ROOT / 'data' / 'progress_scrape_3boats.json'

# 既存進捗を読む (中断後の再開用)
progress = {}
if CKPT.exists():
    progress = json.loads(CKPT.read_text(encoding='utf-8'))
    print(f"📌 既存 checkpoint 発見: {progress}")
else:
    print("📭 新規実行 (checkpoint 無し)")
"""

# ============================================================
# セル 4: 3 船宿を順次再 scrape (本体 + 進捗チェックポイント)
# ============================================================
'''
%cd /content/drive/MyDrive/aichi-fishing-analysis

import sys, json, time
from datetime import datetime
from pathlib import Path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_seed_dataset import build

# (blog_id, site, boat, months_back)
PLANS = [
    ("matobaya",       "morozaki",      "まとばや",   12),
    ("go-arimotomaru", "utsumi_shinko", "ありもと丸", 12),
    ("seablue-0908",   "morozaki",      "シーブルー", 12),
]

def save_progress():
    CKPT.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')

for blog_id, site, boat, months in PLANS:
    # レジューム: 既に完了している船宿はスキップ
    if progress.get(blog_id, {}).get("status") == "done":
        print(f"⏭️  {boat} ({blog_id}): 完了済 (rows={progress[blog_id].get(\"n_rows\")}) — skip")
        continue

    started = datetime.now().isoformat(timespec="seconds")
    print(f"\\n{\'=\'*70}\\n🚢 {boat} ({blog_id}) → site={site} / {months} ヶ月遡る\\n   開始: {started}\\n{\'=\'*70}")
    progress[blog_id] = {"status": "running", "started_at": started}
    save_progress()

    try:
        t0 = time.time()
        df = build(
            blog_id=blog_id,
            site=site,
            boat=boat,
            months_back=months,
            conf=0.30,
            run_yolo=True,         # 画像取得 + YOLO 推論を有効化（最重要）
            skip_existing=False,   # ダミー summary.json を強制再処理
            no_viz=True,           # bbox 可視化画像は不要 (容量節約)
            sleep_sec=1.0,
            use_llm_extract=True,  # 本文 LLM 抽出も並行
            llm_provider="groq",
            llm_fallback_provider="gemini",
        )
        elapsed = time.time() - t0
        progress[blog_id] = {
            "status": "done",
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_min": round(elapsed / 60, 1),
            "n_rows": int(len(df)),
        }
        save_progress()
        print(f"  ✅ {boat}: catches.csv 計 {len(df)} 行 ({elapsed/60:.1f}分)")
    except KeyboardInterrupt:
        progress[blog_id]["status"] = "interrupted"
        save_progress()
        print(f"  ⏹ {boat}: 中断 — 次回再実行で再開")
        break
    except Exception as e:
        progress[blog_id]["status"] = f"error: {e}"
        save_progress()
        print(f"  ❌ {boat}: {e}")

print("\\n" + "="*70)
print("📊 3 船宿 scrape 進捗")
print("="*70)
for blog_id, info in progress.items():
    print(f"  {blog_id}: {info}")
'''

# ============================================================
# セル 5: 新3クラスの取り込み件数を集計
# ============================================================
'''
import pandas as pd
df = pd.read_csv(ROOT / 'data' / 'fishing_logs' / 'catches.csv')
df['has_img'] = df['image_path'].notna() & (df['image_path'].astype(str).str.strip() != '')

print("=== 新3クラス (ホウボウ/オコゼ/サワラ) の image_path 取得状況 ===")
for sp in ['ホウボウ', 'オコゼ', 'サワラ']:
    sub = df[df['species'] == sp]
    n_total = len(sub)
    n_with_img = sub['has_img'].sum()
    pct = (n_with_img / n_total * 100) if n_total else 0
    print(f"  {sp}: {n_with_img}/{n_total} 行に image_path ({pct:.0f}%)")

print("\\n=== 船宿別 image_path カバー率 ===")
boat_summary = df.groupby('boat').agg(
    rows=('species', 'size'),
    with_img=('has_img', 'sum'),
)
boat_summary['pct'] = (boat_summary['with_img'] / boat_summary['rows'] * 100).round(1)
print(boat_summary.sort_values('rows', ascending=False).to_string())
'''

# ============================================================
# セル 6: 新3クラスの bbox 可視化サンプル (検出された画像を 5 枚プレビュー)
# ============================================================
'''
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from src import font_setup  # 日本語フォント設定

scraped = ROOT / 'data' / 'scraped'
TARGET_SP = {'ホウボウ', 'オコゼ', 'サワラ'}
samples = []  # (img_path, species, conf, bbox)

for d in scraped.iterdir():
    if not d.is_dir():
        continue
    sj = d / 'summary.json'
    if not sj.exists():
        continue
    try:
        info = json.loads(sj.read_text(encoding='utf-8'))
    except Exception:
        continue
    for r in info.get('results', []):
        for det in r.get('detections', []):
            if det['species'] in TARGET_SP:
                local = ROOT / r.get('local', '')
                if local.exists():
                    samples.append((local, det['species'], det['confidence'], det['bbox']))
    if len(samples) >= 5:
        break

if not samples:
    print("⚠️ 新3クラス検出サンプルがまだ無い (scrape 未完了 or 検出ゼロ)")
else:
    n = min(5, len(samples))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (p, sp, conf, bbox) in zip(axes, samples[:n]):
        img = Image.open(p)
        ax.imshow(img)
        if bbox:
            x1, y1, x2, y2 = bbox
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none',
            ))
        ax.set_title(f"{sp} ({conf:.2f})")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"📸 {len(samples)} 件中先頭 {n} 件を表示 (新3クラス検出 bbox)")
'''

# ============================================================
# 想定所要時間 (T4 GPU)
# ============================================================
# - エントリ列挙 (blog_scraper.list_entries)        : 各船宿 1-2 分
# - 画像取得 + YOLO 推論 (predict_from_url)         : ~10 秒/エントリ × ~300-400 entries/船宿
# - 本文抽出 (blog_text_extractor)                  : ~3 秒/エントリ × 300-400
# - aggregate                                       : 数秒
# 1 船宿あたり 1-2 時間、3 船宿で 4-6 時間
#
# 進捗 checkpoint (data/progress_scrape_3boats.json) を持つので、Colab セッションが切れても
# 同じセル 4 を再実行すれば 完了済船宿はスキップして残りから再開可能。
