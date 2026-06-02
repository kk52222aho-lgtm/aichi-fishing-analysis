"""予測ロガー — /predict の結果を CSV に記録し、後から /feedback で実績を紐付ける。

設計:
    1. 予測時: log_prediction(result) → prediction_id (uuid) を発行して 1 行 append
    2. 出船後: link_feedback(prediction_id, actual_top_per_angler, ...) で
       同じ行の actual_* 列を埋める
    3. compute_summary() で全期間の MAE / tier 一致率 / 件数を返す（運用真の精度）

ファイル: data/predictions_log.csv
    既存スキーマと互換性を保つため、列追加は末尾のみ。
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config

LOG_PATH = config.DATA_DIR / "predictions_log.csv"

# CSV スキーマ（順序固定）
_COLUMNS = [
    "prediction_id",
    "created_at",            # 予測時刻 (UTC ISO8601)
    "target_date",           # 予測対象日 YYYY-MM-DD
    "hour",
    "site",
    "species",
    "boat",
    "anglers",
    "tackle",
    "target_species",
    "engine",                # llm / statistical
    "provider",              # cerebras / groq / ... (LLM のみ)
    "model",                 # gpt-oss-120b 等
    "predicted_top_per_angler",
    "tier",
    "tier_label",
    "confidence",            # high/medium/low (LLM のみ)
    "reasoning_short",       # LLM の reasoning 先頭 300 文字
    # 以下は feedback 時に埋まる
    "actual_top_per_angler",
    "actual_total_catch",
    "actual_qualitative",
    "feedback_at",           # フィードバック受領時刻 UTC ISO8601
    "feedback_notes",
]

_LOCK = threading.Lock()


def _ensure_file() -> None:
    """予測ログファイルが無ければ空ファイルを作る。"""
    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_COLUMNS).to_csv(LOG_PATH, index=False)


_STR_COLUMNS = {
    "prediction_id", "created_at", "target_date", "site", "species",
    "boat", "tackle", "target_species", "engine", "provider", "model",
    "tier_label", "confidence", "reasoning_short",
    "actual_qualitative", "feedback_at", "feedback_notes",
}


def _read_log() -> pd.DataFrame:
    _ensure_file()
    # 文字列列は object 固定（fillna(NaN) で float 化を防ぐ）
    dtype = {c: "object" for c in _STR_COLUMNS}
    df = pd.read_csv(LOG_PATH, dtype=dtype, keep_default_na=False, na_values=[""])
    for c in _COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c not in _STR_COLUMNS else ""
    # str 列の NaN を空文字に
    for c in _STR_COLUMNS:
        df[c] = df[c].fillna("")
    return df[_COLUMNS]


def log_prediction(result: dict[str, Any]) -> str:
    """1 件の予測を CSV に append。prediction_id を返す。

    Args:
        result: api/server.py /predict が返す dict。`prediction` キーの中身と
                `engine`, `model`, `provider` 等の top-level を読む。

    Returns:
        prediction_id (UUID hex)。サーバが response にも含めて返却すべき。
    """
    pid = uuid.uuid4().hex[:16]  # 16桁で十分一意、CSV で扱いやすい
    pred = result.get("prediction", {}) or {}
    model_info = result.get("model", {}) or {}

    reasoning = pred.get("reasoning", "") or ""
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    row = {
        "prediction_id": pid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_date": str(result.get("date", "")),
        "hour": result.get("hour"),
        "site": result.get("site"),
        "species": result.get("species"),
        "boat": result.get("boat"),
        "anglers": result.get("anglers"),
        "tackle": result.get("tackle"),
        "target_species": result.get("target_species"),
        "engine": result.get("engine"),
        "provider": model_info.get("provider"),
        "model": model_info.get("name"),
        "predicted_top_per_angler": pred.get("predicted_top_per_angler",
                                             pred.get("top_per_angler")),
        "tier": pred.get("tier"),
        "tier_label": pred.get("tier_label"),
        "confidence": pred.get("confidence"),
        "reasoning_short": reasoning,
        "actual_top_per_angler": np.nan,
        "actual_total_catch": np.nan,
        "actual_qualitative": "",
        "feedback_at": "",
        "feedback_notes": "",
    }

    with _LOCK:
        _ensure_file()
        # ヘッダー付き append
        pd.DataFrame([row], columns=_COLUMNS).to_csv(
            LOG_PATH, mode="a", header=False, index=False,
        )
    return pid


def link_feedback(
    prediction_id: str,
    actual_top_per_angler: Optional[float] = None,
    actual_total_catch: Optional[float] = None,
    actual_qualitative: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """既存の予測行に実績を書き戻す。

    Returns:
        更新された行の dict、または {"error": ...} 形式。
    """
    with _LOCK:
        df = _read_log()
        mask = df["prediction_id"] == prediction_id
        if not mask.any():
            return {"error": f"prediction_id not found: {prediction_id}"}
        idx = df.index[mask][0]
        if actual_top_per_angler is not None:
            df.at[idx, "actual_top_per_angler"] = float(actual_top_per_angler)
        if actual_total_catch is not None:
            df.at[idx, "actual_total_catch"] = float(actual_total_catch)
        if actual_qualitative is not None:
            df.at[idx, "actual_qualitative"] = str(actual_qualitative)
        if notes is not None:
            df.at[idx, "feedback_notes"] = str(notes)
        df.at[idx, "feedback_at"] = datetime.now(timezone.utc).isoformat()
        df.to_csv(LOG_PATH, index=False)
        return df.iloc[idx].to_dict()


def find_recent_predictions(
    site: Optional[str] = None,
    species: Optional[str] = None,
    boat: Optional[str] = None,
    target_date: Optional[str] = None,
    only_pending_feedback: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """検索: 条件に合う最近の予測ログ。Streamlit でフィードバック対象を選ぶため。"""
    df = _read_log()
    if df.empty:
        return []
    if site:
        df = df[df["site"] == site]
    if species:
        df = df[df["species"] == species]
    if boat:
        df = df[df["boat"] == boat]
    if target_date:
        df = df[df["target_date"] == target_date]
    if only_pending_feedback:
        df = df[df["actual_top_per_angler"].isna()
                | (df["actual_top_per_angler"] == "")]
    df = df.sort_values("created_at", ascending=False).head(limit)
    return df.to_dict("records")


def compute_summary(species: Optional[str] = None) -> dict[str, Any]:
    """フィードバック済み予測から運用真の MAE / tier 一致率を集計。

    Args:
        species: 指定すれば魚種別、None なら全体。

    Returns:
        n_total / n_with_feedback / mae / rmse / bias / tier_exact_match /
        tier_within_1 / by_engine など
    """
    df = _read_log()
    if df.empty:
        return {"n_total": 0, "n_with_feedback": 0}

    if species:
        df = df[df["species"] == species]

    total = len(df)
    df_fb = df.dropna(subset=["actual_top_per_angler"]).copy()
    df_fb = df_fb[df_fb["predicted_top_per_angler"].notna()]
    n_fb = len(df_fb)

    if n_fb == 0:
        return {
            "species": species,
            "n_total": int(total),
            "n_with_feedback": 0,
            "feedback_rate": 0.0,
        }

    df_fb["predicted_top_per_angler"] = df_fb["predicted_top_per_angler"].astype(float)
    df_fb["actual_top_per_angler"] = df_fb["actual_top_per_angler"].astype(float)
    residual = df_fb["predicted_top_per_angler"] - df_fb["actual_top_per_angler"]
    mae = float(residual.abs().mean())
    rmse = float(np.sqrt((residual ** 2).mean()))
    bias = float(residual.mean())

    # tier 一致率（tier 列が欠損していなければ）
    tier_metrics = {}
    if "tier" in df_fb.columns and df_fb["tier"].notna().any():
        # tier_actual はログには無いので、予測時の cutoffs を使えないため
        # 簡易: actual の値から quintile を求める（df_fb 内で再計算）
        try:
            cutoffs = np.quantile(df_fb["actual_top_per_angler"],
                                  [0.2, 0.4, 0.6, 0.8])
            actual_tier = (np.digitize(df_fb["actual_top_per_angler"],
                                       cutoffs, right=True) + 1).clip(1, 5)
            pred_tier = df_fb["tier"].astype(int)
            tier_metrics = {
                "tier_exact_match": float((pred_tier == actual_tier).mean()),
                "tier_within_1": float((np.abs(pred_tier - actual_tier) <= 1).mean()),
            }
        except Exception:
            pass

    # engine 別内訳
    by_engine = {}
    if "engine" in df_fb.columns:
        for eng, grp in df_fb.groupby("engine"):
            grp_res = grp["predicted_top_per_angler"].astype(float) - \
                grp["actual_top_per_angler"].astype(float)
            by_engine[str(eng)] = {
                "n": int(len(grp)),
                "mae": round(float(grp_res.abs().mean()), 2),
                "bias": round(float(grp_res.mean()), 2),
            }

    return {
        "species": species,
        "n_total": int(total),
        "n_with_feedback": int(n_fb),
        "feedback_rate": round(n_fb / total, 3) if total else 0.0,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        **tier_metrics,
        "by_engine": by_engine,
    }
