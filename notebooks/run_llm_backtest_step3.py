"""Step 3 改善 (temperature=0.0 + 直近 trip anchor + recent-3 median fallback) を
LLM backtest で実測し、旧結果と比較。

【課金は発生しない】無料枠プロバイダのみ:
    - cerebras (gpt-oss-120b)        : 無料 tier
    - groq     (llama-3.3-70b-versatile): 無料 14400 req/日
Gemini は無料枠 20 req/日と狭く、超過時課金リスクがあるため除外。

【Colab セットアップ】
    # 1. Drive マウント + API キー登録
    from google.colab import drive, userdata
    drive.mount('/content/drive', force_remount=False)

    # 2. プロジェクトに移動
    import os
    PROJ = '/content/drive/MyDrive/aichi-fishing-analysis'
    os.chdir(PROJ)
    !git pull

    # 3. 依存
    !pip install -q lightgbm openai

    # 4. このスクリプトを実行
    !python notebooks/run_llm_backtest_step3.py

API キーは Colab userdata に下記名前で登録:
    CEREBRAS_API_KEY, GROQ_API_KEY

【比較ジョブ】(課金プロバイダは除外)
    - マダイ × cerebras/gpt-oss-120b   (旧 MAE=16.13, +23.9% vs baseline)
    - イサキ × groq/llama-3.3-70b      (旧 MAE=6.40,  +14.3%)
    - ホウボウ × cerebras/gpt-oss-120b (旧 MAE=2.80,  -26.4% ← 改善余地大)

新 CSV は `backtest_llm_{provider}_{species}_{STEP}.csv` に保存し、旧 CSV を
壊さない。最後に comparison table を出力。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（notebooks/ から起動された場合に必要）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import backtest as bt
from src import config

# ------------------------------------------------------------
# 試行バージョン識別子 — 新規実装の度にバンプして過去結果を上書きしない
#   step3   : temperature=0.0 + anchor block (±30% hard constraint)
#   step3_1 : anchor を緩和（参考枠）+ past_max 追加
#   step3_2 : 類似日 (同月±1 + 同 tide_phase) の過去最大を prompt に注入
#   step3_3 : 類似日 block を max/median < 2.5 の narrow species だけ発火
# ------------------------------------------------------------
STEP = "step3_3"

# ------------------------------------------------------------
# 比較対象（旧 backtest と同条件）
# ------------------------------------------------------------
JOBS = [
    # (species, site,         boat,            provider,   model)
    # 無料枠プロバイダのみ。gemini は除外（無料 20 req/日で超過リスク）
    ("マダイ",  "morozaki",   "まとばや",      "cerebras", "gpt-oss-120b"),
    ("イサキ",  "irago",      "maruman2010",   "groq",     "llama-3.3-70b-versatile"),
    ("ホウボウ", "morozaki",   "まとばや",      "cerebras", "gpt-oss-120b"),
]

# 旧 CSV (commit 8462e98 で確認した値) - tuple: (mae, baseline_mae, vs_base_pct, tier_exact, bias)
OLD_RESULTS = {
    ("マダイ",  "cerebras"): (16.13, 21.20,  0.239, 0.17, -4.83),
    ("イサキ",  "groq"):     ( 6.40,  7.46,  0.143, 0.10, -2.20),
    ("ホウボウ", "cerebras"): ( 2.80,  2.22, -0.264, 0.20, +1.40),
}


def _backup_old_csv(species: str, provider: str, model: str) -> Path | None:
    """旧 LLM CSV を _old サフィックスで保存（既存ファイルを壊さないため）。"""
    suffix = provider if model is None else f"{provider}_{model.replace('/', '_')}"
    p_old = config.INTEGRATED_DIR / f"backtest_llm_{suffix}_{species}.csv"
    if not p_old.exists():
        return None
    p_backup = config.INTEGRATED_DIR / f"backtest_llm_{suffix}_{species}_old.csv"
    shutil.copy(p_old, p_backup)
    return p_backup


def _summarize(csv_path: Path) -> dict:
    """CSV から指標を集計（旧 _summarize と整合）。"""
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return {"n": 0}
    mae = float(df["residual"].abs().mean())
    base_mae = float(df["baseline_residual"].abs().mean()) if "baseline_residual" in df else None
    vs_base = (
        round((base_mae - mae) / base_mae, 3)
        if base_mae and base_mae > 0 else None
    )
    return {
        "n": n,
        "mae": round(mae, 2),
        "baseline_mae": round(base_mae, 2) if base_mae else None,
        "vs_baseline_pct": vs_base,
        "rmse": round(float(np.sqrt((df["residual"] ** 2).mean())), 2),
        "bias": round(float(df["residual"].mean()), 2),
        "tier_exact": round(float((df["tier_pred"] == df["tier_actual"]).mean()), 2),
        "tier_within_1": round(float((np.abs(df["tier_pred"] - df["tier_actual"]) <= 1).mean()), 2),
    }


def main():
    print("="*70)
    print("LLM Backtest Step 3 改善検証")
    print("="*70)

    results = []
    for species, site, boat, provider, model in JOBS:
        print(f"\n--- {species} × {provider}/{model} (site={site}, boat={boat}) ---")

        # 旧 CSV をバックアップ
        backup = _backup_old_csv(species, provider, model)
        if backup:
            print(f"  旧 CSV を退避: {backup.name}")

        # 実行（既存ファイル名で上書き保存される）
        # use_cache=False で旧プロンプト時の予測キャッシュを無視 → Step 3 改善を反映
        try:
            summary = bt.run_llm_for_species(
                species, site=site, boat=boat,
                provider=provider, model=model,
                min_train=5, save_csv=True, save_plots=False,
                use_cache=False,
            )
            new_csv = Path(summary["csv_path"])
            # _step3 サフィックスにリネーム
            new_path = new_csv.parent / new_csv.name.replace(".csv", f"_{STEP}.csv")
            shutil.move(new_csv, new_path)
            # バックアップを元の場所に戻す（旧 CSV を保持）
            if backup:
                shutil.move(backup, new_csv)
            print(f"  新 CSV: {new_path.name}")
            new_metrics = _summarize(new_path)
        except Exception as e:
            print(f"  ⚠️ エラー: {e}")
            new_metrics = {"error": str(e)}
            if backup and backup.exists():
                shutil.move(backup, new_csv)

        old = OLD_RESULTS.get((species, provider))
        if old and "error" not in new_metrics:
            old_mae, old_base, old_vs, old_tier, old_bias = old
            delta_mae = round(new_metrics["mae"] - old_mae, 2)
            print(f"  旧 MAE={old_mae:.2f} (vs_base={old_vs:+.1%}) → "
                  f"新 MAE={new_metrics['mae']:.2f} (Δ={delta_mae:+.2f})")
            results.append({
                "species": species, "provider": provider, "model": model,
                "n": new_metrics.get("n"),
                "old_mae": old_mae, "new_mae": new_metrics.get("mae"),
                "delta_mae": delta_mae,
                "old_vs_base": old_vs, "new_vs_base": new_metrics.get("vs_baseline_pct"),
                "old_tier_exact": old_tier, "new_tier_exact": new_metrics.get("tier_exact"),
                "old_bias": old_bias, "new_bias": new_metrics.get("bias"),
            })
        else:
            results.append({
                "species": species, "provider": provider, "model": model,
                **new_metrics, "old": old,
            })

    # 比較表
    out_path = ROOT / f"_llm_{STEP}_comparison.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("\n" + "="*70)
    print(f"Comparison Table (saved to _llm_{STEP}_comparison.csv)")
    print("="*70)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
