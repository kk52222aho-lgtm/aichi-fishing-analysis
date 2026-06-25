"""LLM 推論ベースの釣果予測（プロバイダ抽象化済み）。

なぜ LLM か:
  - 学習データが各魚種 20-25 件しかないので統計モデルは baseline 同等
  - 釣り船予測には強い事前知識がある（春の大潮の朝マヅメは爆釣 等）
  - LLM の事前知識 + 過去ログの統計 + 今日の予報 を組み合わせれば実用的

プロバイダ:
  - "groq"     : Groq Llama 3.3 70B（OSS、無料枠 14400 req/日、default）
  - "cerebras" : Cerebras gpt-oss-120b（default、無料枠は他モデルより緩い実測）。
                 Groq より TPM が広く、大量バッチ抽出向き。
                 利用可能モデル: gpt-oss-120b / qwen-3-235b-a22b-instruct-2507 / llama3.1-8b / zai-glm-4.7
                 （qwen-3-235b は free で即 rate_limit になりやすい実測 2026-05-15）
  - "gemini"   : Google Gemini 2.5-flash（課金、reasoning の質が高い）
  - "ollama"   : ローカル Ollama（完全無料、要セットアップ）
  - "cascade"  : Groq でスクリーニング → low conf のみ Gemini に escalate

数値予測の精度は Groq ≒ Gemini（backtest で MAE ほぼ同等）。
Reasoning の深さは Gemini が上。愛知〜中部規模なら Groq 単独で実用。

将来 GPT-5 / Claude 5 等を足すなら _PROVIDER_CALLERS に関数1個追加するだけ。

使い方:
    from src.llm_predictor import predict_with_llm
    r = predict_with_llm(
        site="irago", species="イサキ", target_date="2026-05-15",
        hour=5, boat="maruman2010", anglers=10, tackle="サビキ",
        provider="gemini",  # or "groq", "cascade"
    )
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import config, data_integrator

TIER_LABELS = ("厳しい", "やや渋い", "普通", "好調", "大漁")

# プロバイダごとのデフォルトモデル
_PROVIDER_DEFAULTS = {
    "gemini":    "gemini-2.5-flash",
    "groq":      "llama-3.3-70b-versatile",
    "cerebras":  "gpt-oss-120b",
    "ollama":    "llama3.2:3b",
    # 追加の無料 OpenAI 互換プロバイダ（各社独立枠 → カスケードで総量増）
    "nvidia":    "meta/llama-3.3-70b-instruct",
    "sambanova": "Meta-Llama-3.3-70B-Instruct",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}

# プロバイダごとの API キー候補（Colab userdata / 環境変数の名前）
_PROVIDER_API_KEYS = {
    "gemini": (
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY",
        "GEMINI_KEY", "GOOGLE_AI_API_KEY",
    ),
    "groq":     ("GROQ_API_KEY", "GROQ_KEY"),
    "cerebras": ("CEREBRAS_API_KEY", "CEREBRAS_KEY"),
    "ollama":   (),  # local, no key needed
    "nvidia":   ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY"),
    "sambanova": ("SAMBANOVA_API_KEY", "SAMBA_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
}

# JSON 出力スキーマ（OpenAPI 風、Gemini が直接食う形）
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "predicted_top_per_angler": {"type": "number"},
        "tier": {"type": "integer"},
        "tier_label": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "predicted_top_per_angler", "tier", "tier_label",
        "confidence", "reasoning",
    ],
}


# ============================================================
# API キー解決
# ============================================================

def _get_api_key(provider: str = "gemini") -> str:
    """指定プロバイダの API キーを Colab userdata / 環境変数から解決。"""
    candidates = _PROVIDER_API_KEYS.get(provider, ())
    if not candidates:
        return ""

    try:
        from google.colab import userdata  # type: ignore
        for name in candidates:
            try:
                k = userdata.get(name)
                if k:
                    return k
            except Exception:
                continue
    except Exception:
        pass

    for name in candidates:
        v = os.environ.get(name)
        if v:
            return v
    return ""


def _list_available_keys() -> dict[str, list[str]]:
    """デバッグ用: プロバイダごとに利用可能なキー名を返す。"""
    out: dict[str, list[str]] = {}
    for provider, candidates in _PROVIDER_API_KEYS.items():
        found: list[str] = []
        try:
            from google.colab import userdata  # type: ignore
            for name in candidates:
                try:
                    if userdata.get(name):
                        found.append(f"colab:{name}")
                except Exception:
                    continue
        except Exception:
            pass
        for name in candidates:
            if os.environ.get(name):
                found.append(f"env:{name}")
        out[provider] = found
    return out


# ============================================================
# データ準備（プロバイダ非依存）
# ============================================================

# 数値系シグナル（量データ）と定性系シグナル（カテゴリ）
_NUMERIC_SIGNALS = ("top_per_angler", "count_yolo", "total_catch")
_QUALITATIVE_SIGNALS = ("qualitative",)


def _numeric_signal_stats(values: np.ndarray) -> dict[str, Any]:
    """数値シグナルの統計（mean/median/quintile 等）。"""
    if len(values) == 0:
        return {"n": 0}
    s: dict[str, Any] = {
        "n": int(len(values)),
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "std": round(float(np.std(values)), 2),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": round(float(np.quantile(values, 0.25)), 2),
        "p75": round(float(np.quantile(values, 0.75)), 2),
    }
    if len(values) >= 5:
        s["tier_cutoffs"] = [round(float(x), 2) for x in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]
    return s


def _auto_pick_primary_signal(signals: dict[str, dict], min_n: int = 5) -> Optional[str]:
    """シグナル別データ密度から主力シグナルを自動選択。

    優先順序: top_per_angler > count_yolo > total_catch > qualitative
    密度が `min_n` 未満なら次のシグナルへフォールバック。
    """
    order = ("top_per_angler", "count_yolo", "total_catch", "qualitative")
    for sig in order:
        if signals.get(sig, {}).get("n", 0) >= min_n:
            return sig
    # それでも見つからなければ最大密度のシグナル（min_n 未満でも）
    best = max(signals.items(), key=lambda kv: kv[1].get("n", 0), default=(None, {}))
    return best[0] if best[1].get("n", 0) > 0 else None


def compute_boat_stats(
    species: str,
    boat: Optional[str] = None,
    catches_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """catches.csv から船宿×魚種の過去統計を計算（multi-signal 版）。

    出力は signals dict で 4 種のシグナル別統計を提供:
      - top_per_angler  (個人最大、本文「竿頭 N尾」)
      - count_yolo      (YOLO の画像カウント、船全体推定)
      - total_catch     (船合計、本文「全 N 匹」)
      - qualitative     (絶好調/好調/普通/渋い/厳しい/ボウズ の分布)

    primary_signal は registry に登録されてれば優先、無ければ
    `_auto_pick_primary_signal` がデータ密度から自動選択。

    後方互換: primary signal の統計値を top-level にも複製して、既存の
    predictor / visualizer が変更なく動くようにする。
    """
    csv = Path(catches_path) if catches_path else config.FISHING_DIR / "catches.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    sub = df[df["species"] == species].copy()
    if boat:
        sub = sub[sub["boat"] == boat]
    if sub.empty:
        return {}

    sub["datetime"] = pd.to_datetime(sub["datetime"])
    sub = sub.sort_values("datetime")

    # ── 4 種シグナル別の統計 ──
    signals: dict[str, dict] = {}
    for col in _NUMERIC_SIGNALS:
        if col not in sub.columns:
            signals[col] = {"n": 0}
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").dropna().astype(float).values
        signals[col] = _numeric_signal_stats(vals)

    for col in _QUALITATIVE_SIGNALS:
        if col not in sub.columns:
            signals[col] = {"n": 0, "distribution": {}}
            continue
        qual = sub[col].dropna()
        signals[col] = {
            "n": int(len(qual)),
            "distribution": qual.value_counts().to_dict() if len(qual) else {},
        }

    # ── primary_signal の解決 (registry > 自動判定) ──
    # 遅延 import: scrape_to_catches 経由で循環を避ける
    from . import scrape_to_catches as _stc
    profile = _stc.get_boat_profile(boat) if boat else {}
    primary = profile.get("primary_signal") or _auto_pick_primary_signal(signals)
    secondary = profile.get("secondary_signal")

    # ── 月別の活性傾向（primary が数値系のときだけ）──
    by_month: dict[int, dict] = {}
    if primary in _NUMERIC_SIGNALS:
        sub_valid = sub.dropna(subset=[primary]).copy()
        if not sub_valid.empty:
            sub_valid["month"] = sub_valid["datetime"].dt.month
            monthly = sub_valid.groupby("month")[primary].agg(["mean", "size"]).round(1)
            by_month = {
                int(m): {"mean": float(monthly.loc[m, "mean"]), "n": int(monthly.loc[m, "size"])}
                for m in monthly.index
            }

    # ── 直近 5 trip（全シグナル列込み） ──
    recent_cols = ["datetime"] + [c for c in _NUMERIC_SIGNALS + _QUALITATIVE_SIGNALS + ("entry_title",) if c in sub.columns]
    recent = sub.tail(5)[recent_cols].copy()
    recent["datetime"] = recent["datetime"].dt.strftime("%Y-%m-%d")
    recent_5 = recent.to_dict("records")

    stats: dict[str, Any] = {
        "n_trips_total": int(len(sub)),
        "primary_signal": primary,
        "secondary_signal": secondary,
        "signals": signals,
        "by_month_of_primary": by_month,
        "recent_5_trips": recent_5,
    }

    # ── 後方互換: primary signal の統計値を top-level にも複製 ──
    if primary in _NUMERIC_SIGNALS:
        p_stats = signals.get(primary, {})
        if p_stats.get("n", 0) > 0:
            stats["n_trips"] = p_stats["n"]
            for k in ("mean", "median", "std", "min", "max", "p25", "p75"):
                if k in p_stats:
                    stats[k] = p_stats[k]
            if "tier_cutoffs" in p_stats:
                stats["tier_cutoffs"] = p_stats["tier_cutoffs"]

    return stats


def find_similar_past_trips(
    species: str,
    boat: Optional[str],
    target_dt: pd.Timestamp,
    target_conditions: dict[str, Any],
    catches_path: Optional[Path | str] = None,
    integrated_path: Optional[Path | str] = None,
    label_column: str = "top_per_angler",
    n_top: int = 3,
) -> dict[str, Any]:
    """同船宿×同魚種で、同月±1 + 同 tide_phase の過去 trip の最大/上位を返す。

    マダイ等 wide-distribution 魚種で LLM が median 寄り予測になるのを抑制するため、
    『同条件下では過去 N 尾出ている』という事実を prompt に注入する材料を提供する。

    データソース優先順:
        1. integrated_path (default: data/integrated/integrated.parquet) — backtest と
           同じ「真の」ラベル分布が入っている
        2. catches_path - integrated が無い時のフォールバック

    Args:
        target_dt:  予測対象日時（これより前の trip のみ参照、未来漏洩防止）
        target_conditions: _readable_conditions() の出力。tide_phase をフィルタに使う
    """
    ipath = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    if ipath.exists():
        df = pd.read_parquet(ipath) if ipath.suffix == ".parquet" else pd.read_csv(ipath)
    else:
        csv = Path(catches_path) if catches_path else config.FISHING_DIR / "catches.csv"
        if not csv.exists():
            return {}
        df = pd.read_csv(csv)
    if "datetime" not in df.columns or label_column not in df.columns:
        return {}
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # tz を剥がして比較互換にする（catches.csv は +09:00 付き）
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    target_dt_naive = pd.Timestamp(target_dt)
    if getattr(target_dt_naive, "tz", None) is not None:
        target_dt_naive = target_dt_naive.tz_localize(None)
    # 過去のみ（target 自身は含めない）
    df = df[df["datetime"] < target_dt_naive]
    df = df[df["species"] == species]
    df = df.dropna(subset=[label_column])
    if df.empty:
        return {}

    target_month = int(target_dt.month)
    months = {((target_month - 2) % 12) + 1, target_month, (target_month % 12) + 1}
    df["_month"] = df["datetime"].dt.month
    df_window = df[df["_month"].isin(months)].copy()
    if df_window.empty:
        return {}

    # 1) boat-filtered 試行（厳しい条件）。十分な n が取れたら採用
    target_tide = target_conditions.get("tide_phase")
    scope = "all_boats"
    df_strict = df_window
    if boat and "boat" in df_window.columns:
        df_boat = df_window[df_window["boat"] == boat]
        if len(df_boat) >= 3:
            df_strict = df_boat
            scope = "same_boat"

    # 2) tide_phase で更に絞り込み（その結果が 3 件未満なら剥がす）
    criteria = {"month_window": f"{target_month}±1", "tide_phase": None, "scope": scope}
    if target_tide and "tide_phase" in df_strict.columns:
        df_tide = df_strict[df_strict["tide_phase"] == target_tide]
        if len(df_tide) >= 3:
            df_strict = df_tide
            criteria["tide_phase"] = target_tide

    vals = pd.to_numeric(df_strict[label_column], errors="coerce").dropna().astype(float)
    if len(vals) < 3:
        return {}

    top_trips_df = df_strict.sort_values(label_column, ascending=False).head(n_top)
    top_trips = []
    for r in top_trips_df.to_dict("records"):
        top_trips.append({
            "date": str(r.get("datetime", ""))[:10],
            "boat": r.get("boat"),
            label_column: r.get(label_column),
            "tide_phase": r.get("tide_phase"),
        })

    return {
        "n_similar": int(len(vals)),
        "criteria": criteria,
        "max": float(vals.max()),
        "median": float(vals.median()),
        "p75": float(vals.quantile(0.75)),
        "top_trips": top_trips,
    }


def compute_blowout_probability(
    species: str,
    target_dt: pd.Timestamp,
    boat: Optional[str] = None,
    target_conditions: Optional[dict[str, Any]] = None,
    integrated_path: Optional[Path | str] = None,
    label_column: str = "top_per_angler",
    threshold_quantile: float = 0.75,
) -> dict[str, Any]:
    """過去の類似日 (同月±1) から「大漁日確率」を計算。

    大漁の閾値: その魚種の全期間 top_per_angler の p75（= 過去上位 25%）。
    確率: 類似日 (同月±1) のうち閾値以上だった日の割合。

    マダイのような wide-distribution 魚種で「絶対値予測は外れがちだが、
    今日が大漁日かどうかだけは当てたい」という釣り人ニーズに応える機能。
    """
    ipath = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    if not ipath.exists():
        return {"probability": None, "reason": "integrated.parquet なし"}

    df = pd.read_parquet(ipath) if ipath.suffix == ".parquet" else pd.read_csv(ipath)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    target_dt_naive = pd.Timestamp(target_dt)
    if getattr(target_dt_naive, "tz", None) is not None:
        target_dt_naive = target_dt_naive.tz_localize(None)

    sub = df[df["species"] == species].copy()
    sub = sub.dropna(subset=[label_column])
    if len(sub) < 5:
        return {
            "probability": None,
            "reason": f"全期間 {len(sub)} 件のみ、判定に必要な 5 件未満",
        }

    # 大漁閾値: 全期間 p75
    threshold = float(sub[label_column].quantile(threshold_quantile))

    # 類似日 (同月±1) を過去から抽出
    past = sub[sub["datetime"] < target_dt_naive].copy()
    if len(past) < 3:
        return {
            "probability": None,
            "threshold": round(threshold, 1),
            "reason": f"過去 {len(past)} 件のみ",
        }

    past["_month"] = past["datetime"].dt.month
    target_month = int(target_dt.month)
    months = {((target_month - 2) % 12) + 1, target_month, (target_month % 12) + 1}
    similar = past[past["_month"].isin(months)]
    if len(similar) < 3:
        similar = past
        scope = "all_past"
    else:
        scope = "same_month_±1"

    blowout_mask = similar[label_column] >= threshold
    blowout_count = int(blowout_mask.sum())
    base_probability = blowout_count / len(similar)

    # 直近 7d / 14d 活性で補正
    after_recent, recent_reason = _adjust_prob_with_recent(
        base_probability, past, pd.Timestamp(target_dt), threshold, label_column,
    )
    # SST 7d 変化 / tide_phase で precision を絞り込み
    if target_conditions:
        after_cond, cond_reason = _adjust_prob_with_conditions(
            after_recent, target_conditions, past, threshold, label_column,
        )
    else:
        after_cond = after_recent
        cond_reason = "条件補正なし"

    # 船宿別 prior で更に絞り込み
    adjusted_probability, boat_reason = _adjust_prob_with_boat(
        after_cond, boat, past, threshold, label_column,
    )
    parts = [recent_reason]
    if cond_reason and cond_reason != "条件補正なし":
        parts.append(cond_reason)
    if boat_reason and boat_reason != "船宿補正なし":
        parts.append(boat_reason)
    adj_reason = "; ".join(parts)

    # 直近の大漁日（上位 3 件）
    big_days = similar[blowout_mask].sort_values("datetime", ascending=False)
    examples: list[dict[str, Any]] = []
    for _, r in big_days.head(3).iterrows():
        examples.append({
            "date": str(r["datetime"])[:10],
            "boat": r.get("boat"),
            label_column: float(r[label_column]),
        })

    return {
        "probability": round(adjusted_probability, 3),
        "base_probability": round(base_probability, 3),
        "adjustment": adj_reason,
        "threshold": round(threshold, 1),
        "n_similar": int(len(similar)),
        "n_blowout_in_similar": blowout_count,
        "scope": scope,
        "recent_blowout_examples": examples,
        "label_column": label_column,
    }


def _adjust_prob_with_recent(
    base_prob: float, past: pd.DataFrame, target_dt: pd.Timestamp,
    threshold: float, label_column: str = "top_per_angler",
) -> tuple[float, str]:
    """直近 7d / 14d の species-wide 活性を見て base_prob を補正。

    Returns:
        (調整後確率, 補正理由ラベル)
    """
    recent_7d = past[past["datetime"] >= target_dt - pd.Timedelta(days=7)]
    recent_14d = past[past["datetime"] >= target_dt - pd.Timedelta(days=14)]

    has_7d_blowout = (recent_7d[label_column] >= threshold).any()
    has_14d_blowout = (recent_14d[label_column] >= threshold).any()
    n_recent_14d = len(recent_14d)

    if has_7d_blowout:
        return min(1.0, base_prob + 0.25), "直近 7d に大漁日あり (+25%)"
    if has_14d_blowout:
        return min(1.0, base_prob + 0.12), "直近 14d に大漁日あり (+12%)"
    if n_recent_14d >= 3:
        # 3 trip 以上活動あるのに大漁無し = cold streak
        return max(0.0, base_prob - 0.10), "直近 14d cold streak (-10%)"
    return base_prob, "補正なし (recent 情報少)"


def _adjust_prob_with_boat(
    prob: float, target_boat: Optional[str], past: pd.DataFrame,
    threshold: float, label_column: str = "top_per_angler",
    min_trips: int = 3,
) -> tuple[float, str]:
    """過去その船宿での大漁実績で確率を絞り込む。

    species p75 基準なので overall_rate = 0.25 が比較基準。
    シーブルー型の「大漁実績が偏る船宿」を強く拾うため積極的に補正。
    """
    if not target_boat or "boat" not in past.columns:
        return prob, "船宿補正なし"

    boat_past = past[past["boat"] == target_boat]
    n_boat = len(boat_past)
    if n_boat < min_trips:
        return prob, f"船宿 {target_boat} 過去 {n_boat} 件のみ (補正なし)"

    boat_blowouts = int((boat_past[label_column] >= threshold).sum())
    boat_rate = boat_blowouts / n_boat
    overall_rate = 0.25  # p75 基準の定義

    if boat_blowouts == 0 and n_boat >= 5:
        return (
            max(0.0, prob * 0.20),
            f"{target_boat} は過去 {n_boat} 件で大漁ゼロ (×0.2)",
        )
    if boat_rate >= overall_rate * 2.0:
        return (
            min(1.0, prob * 1.5),
            f"{target_boat} は大漁実績多数 ({boat_blowouts}/{n_boat}, "
            f"率 {boat_rate*100:.0f}%) (×1.5)",
        )
    if boat_rate >= overall_rate * 1.5:
        return (
            min(1.0, prob * 1.25),
            f"{target_boat} は大漁多め ({boat_blowouts}/{n_boat}, "
            f"率 {boat_rate*100:.0f}%) (×1.25)",
        )
    if boat_rate <= overall_rate * 0.5:
        return (
            max(0.0, prob * 0.5),
            f"{target_boat} は大漁少 ({boat_blowouts}/{n_boat}, "
            f"率 {boat_rate*100:.0f}%) (×0.5)",
        )
    return prob, f"{target_boat} は平均的 ({boat_blowouts}/{n_boat})"


def _adjust_prob_with_conditions(
    prob: float, target: dict[str, Any] | pd.Series, past: pd.DataFrame,
    threshold: float, label_column: str = "top_per_angler",
) -> tuple[float, str]:
    """SST 7d 上昇傾向と tide_phase の過去大漁傾向で確率を絞り込み (precision 向上目的)。

    マダイ 9/9 件が中潮/大潮 → 小潮/長潮/若潮は強力に suppress。
    """
    reasons: list[str] = []
    factor = 1.0

    # SST 7d delta — 直近 7d 平均との差で代用（integrated に sst_d7d 列が無いケースに対応）
    sst_d7d = None
    sst_d7d_raw = target.get("sst_d7d") if hasattr(target, "get") else None
    try:
        if sst_d7d_raw is not None:
            sst_d7d = float(sst_d7d_raw)
            if sst_d7d != sst_d7d:  # NaN
                sst_d7d = None
    except (TypeError, ValueError):
        sst_d7d = None

    if sst_d7d is None and "sea_surface_temperature" in past.columns:
        target_sst_raw = target.get("sea_surface_temperature") if hasattr(target, "get") else None
        try:
            target_sst = float(target_sst_raw) if target_sst_raw is not None else None
            if target_sst is not None and target_sst != target_sst:
                target_sst = None
        except (TypeError, ValueError):
            target_sst = None
        if target_sst is not None and "datetime" in past.columns and len(past):
            target_dt = pd.Timestamp(target["datetime"]) if hasattr(target, "get") and target.get("datetime") is not None else None
            if target_dt is not None:
                recent_sst = past[past["datetime"] >= target_dt - pd.Timedelta(days=14)]
                if len(recent_sst) >= 2:
                    past_mean = recent_sst["sea_surface_temperature"].dropna().mean()
                    if past_mean == past_mean:  # not NaN
                        sst_d7d = target_sst - float(past_mean)

    if sst_d7d is not None:
        if sst_d7d >= 0.5:
            factor *= 1.15
            reasons.append(f"SST 7d +{sst_d7d:.1f}℃ (+15%)")
        elif sst_d7d <= -0.5:
            factor *= 0.80
            reasons.append(f"SST 7d {sst_d7d:.1f}℃ (-20%)")

    # Tide phase の大漁傾向 (積極補正)
    today_tide = target.get("tide_phase") if hasattr(target, "get") else None
    if today_tide and "tide_phase" in past.columns:
        past_blowouts = past[past[label_column] >= threshold]
        n_past = len(past)
        n_past_blowouts = len(past_blowouts)
        if n_past >= 5 and n_past_blowouts >= 2:
            all_tide_count = (past["tide_phase"] == today_tide).sum()
            blowout_tide_count = (past_blowouts["tide_phase"] == today_tide).sum()
            if all_tide_count >= 2:
                overall_rate = n_past_blowouts / n_past
                tide_specific_rate = blowout_tide_count / all_tide_count

                if blowout_tide_count == 0 and all_tide_count >= 2:
                    # この潮回りで大漁日ゼロ → 強力 suppression
                    factor *= 0.30
                    reasons.append(f"{today_tide}は過去大漁ゼロ (×0.3)")
                elif tide_specific_rate >= overall_rate * 1.3:
                    factor *= 1.20
                    reasons.append(f"{today_tide}は大漁多 (+20%)")
                elif tide_specific_rate >= overall_rate * 1.1:
                    factor *= 1.10
                    reasons.append(f"{today_tide}はやや大漁多 (+10%)")
                elif tide_specific_rate <= overall_rate * 0.5:
                    factor *= 0.60
                    reasons.append(f"{today_tide}は大漁少 (-40%)")

    adjusted = max(0.0, min(1.0, prob * factor))
    return adjusted, "; ".join(reasons) if reasons else "条件補正なし"


def backtest_blowout_probability(
    species: str,
    integrated_path: Optional[Path | str] = None,
    label_column: str = "top_per_angler",
    threshold_quantile: float = 0.75,
    min_history: int = 5,
    use_recent_adjustment: bool = True,
) -> pd.DataFrame:
    """大漁日確率を過去 trip で再現し、calibration / 的中率を測定する。

    各 trip i に対して:
      - 過去 trip 0..i-1 のみ参照
      - 閾値 = 過去の p75 (= 大漁定義もリアルタイム化)
      - base = 類似日 (同月±1) のうち閾値超過比率
      - 直近 7d / 14d の活性で補正 (+25% / +12% / -10%)
    """
    ipath = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    df = pd.read_parquet(ipath) if ipath.suffix == ".parquet" else pd.read_csv(ipath)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    sub = df[df["species"] == species].copy()
    sub = sub.dropna(subset=[label_column])
    sub = sub.sort_values("datetime").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, target in sub.iterrows():
        target_dt = target["datetime"]
        actual = float(target[label_column])

        past = sub[sub["datetime"] < target_dt].copy()
        if len(past) < min_history:
            continue

        threshold = float(past[label_column].quantile(threshold_quantile))

        past["_month"] = past["datetime"].dt.month
        target_month = int(target_dt.month)
        months = {((target_month - 2) % 12) + 1, target_month, (target_month % 12) + 1}
        similar = past[past["_month"].isin(months)]
        if len(similar) < 3:
            similar = past
            scope = "all_past"
        else:
            scope = "same_month"

        n_similar = len(similar)
        n_blowout = int((similar[label_column] >= threshold).sum())
        base_prob = n_blowout / n_similar

        if use_recent_adjustment:
            after_recent, recent_reason = _adjust_prob_with_recent(
                base_prob, past, target_dt, threshold, label_column,
            )
            after_cond, cond_reason = _adjust_prob_with_conditions(
                after_recent, target, past, threshold, label_column,
            )
            target_boat = target.get("boat") if hasattr(target, "get") else None
            adjusted_prob, boat_reason = _adjust_prob_with_boat(
                after_cond, target_boat, past, threshold, label_column,
            )
            parts = [recent_reason]
            if cond_reason and cond_reason != "条件補正なし":
                parts.append(cond_reason)
            if boat_reason and boat_reason not in ("船宿補正なし",):
                parts.append(boat_reason)
            adj_reason = "; ".join(parts)
        else:
            adjusted_prob = base_prob
            adj_reason = "補正無効"

        actually_blowout = bool(actual >= threshold)

        rows.append({
            "datetime": target_dt,
            "actual": actual,
            "threshold": round(threshold, 1),
            "base_prob": round(base_prob, 3),
            "predicted_prob": round(adjusted_prob, 3),
            "adjustment": adj_reason,
            "actually_blowout": actually_blowout,
            "n_similar": n_similar,
            "n_blowout_in_similar": n_blowout,
            "scope": scope,
        })

    return pd.DataFrame(rows)


def _readable_conditions(row: pd.DataFrame) -> dict[str, Any]:
    """LLM prompt 向けに当日コンディションを抽出。

    重要: Open-Meteo はデフォルトで風速を **km/h** で返すが、日本の釣り人は m/s
    で会話する。LLM 出力の reasoning でも m/s として解釈される前提なので、
    ここで km/h → m/s 換算してから渡す（過去 backtest で 3.6 倍誤読していた）。
    """
    r = row.iloc[0].to_dict()
    out: dict[str, Any] = {}
    keys_round = {
        "temperature_2m": 1, "wind_direction_10m": 0,
        "pressure_msl": 1, "precipitation": 1, "cloud_cover": 0,
        "wave_height": 2, "wave_period": 1, "swell_wave_height": 2,
        "ocean_current_velocity": 2, "ocean_current_direction": 0,
        "sea_surface_temperature": 1, "sea_level_height_msl": 2,
        "sst_d7d": 2, "sst_d24h": 2,
        "tide_cm": 0, "moon_age": 1, "sunrise_hour": 2, "sunset_hour": 2,
    }
    for k, nd in keys_round.items():
        v = r.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[k] = round(float(v), nd)
    # 風速は km/h -> m/s 換算してから提供
    for wind_key in ("wind_speed_10m", "wind_gusts_10m"):
        v = r.get(wind_key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[wind_key] = round(float(v) / 3.6, 1)  # m/s 単位
    out["_wind_unit_note"] = "風速は m/s 単位（Open-Meteo の km/h を換算済み）"
    for k in ("moon_phase", "tide_phase"):
        if r.get(k):
            out[k] = r[k]
    for k in ("is_morning_mazume", "is_evening_mazume"):
        v = r.get(k)
        if v is not None:
            try:
                out[k] = int(v)
            except Exception:
                pass
    return out


_SIGNAL_DESCRIPTION = """
【シグナル種別の意味】
  - top_per_angler: 個人最大釣果（"竿頭 N尾"）。釣り客視点の指標
  - count_yolo:     YOLO 画像カウント。船全体合計の写真ベース推定。ノイズあり
  - total_catch:    船全体合計（本文「全 N 匹」明示）
  - qualitative:    定性カテゴリ（絶好調/好調/普通/渋い/厳しい/ボウズ）

【シグナル使い分け】
  この船の primary_signal を最も信頼すべきシグナルとして扱う。
  - primary_signal が "top_per_angler" → 個人最大の数値を予測ベースに
  - primary_signal が "count_yolo"     → 写真ベース、実数からブレが大きい前提
  - primary_signal が "total_catch"    → 船全体数（anglers で割って個人最大に変換可）
  - primary_signal が "qualitative"    → カテゴリのみ。reasoning と tier 推定が主
  - n_trips_total が極めて少ない時は LLM の事前知識のウェイトを上げる
"""


def _build_prompt(
    species: str, target_date: date, hour: int,
    boat: Optional[str], anglers: Optional[int],
    target_species: Optional[str], tackle: Optional[str],
    stats: dict, conditions: dict,
    include_schema_hint: bool = True,
) -> str:
    schema_hint = ""
    if include_schema_hint:
        schema_hint = """

【出力 JSON スキーマ】
{
  "predicted_top_per_angler": number (整数, 0以上, 単位: 尾),
  "tier": integer (1-5),
  "tier_label": string ("厳しい"|"やや渋い"|"普通"|"好調"|"大漁"),
  "confidence": string ("high"|"medium"|"low"),
  "signal_used": string ("top_per_angler"|"count_yolo"|"total_catch"|"qualitative"|"none"),
  "reasoning": string (2-4文の根拠。primary_signal の特性に触れる),
  "key_factors": array of string (プラス要因, 最大4個),
  "risk_factors": array of string (マイナス要因, 最大3個)
}
"""

    primary = stats.get("primary_signal") or "(none)"
    n_total = stats.get("n_trips_total", 0)
    tier_basis_hint = ""
    if "tier_cutoffs" in stats:
        tier_basis_hint = f"primary_signal の tier_cutoffs={stats['tier_cutoffs']} を基準"
    elif stats.get("signals", {}).get("qualitative", {}).get("n", 0) > 0:
        tier_basis_hint = "primary_signal=qualitative の分布から tier を推定"
    else:
        tier_basis_hint = "過去データ希薄。LLM 事前知識と当日コンディションから推定"

    # ── 直近 trip を予測の参考情報として提示 (制約ではなく材料)
    # 注意: ±30% などの厳しい制約は wide-distribution 魚種 (マダイ等) で
    # 大漁日を過小予測する原因になる。あくまで「参考」「天井/床」として渡す。
    anchor_block = ""
    recent = stats.get("recent_5_trips") or []
    if recent and primary in _NUMERIC_SIGNALS:
        last = recent[-min(3, len(recent)):]
        anchor_lines = []
        for r in last:
            v = r.get(primary)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                anchor_lines.append(f"  - {r.get('datetime', '?')}: {primary}={v}")
        if anchor_lines:
            recent_vals = [r.get(primary) for r in last if r.get(primary) is not None]
            try:
                recent_floats = [float(v) for v in recent_vals]
                anchor_median = float(np.median(recent_floats))
                # 過去最大値（天井把握用、wide-distribution 魚種で過小予測を防ぐ）
                past_max = stats.get("signals", {}).get(primary, {}).get("max")
                past_max_str = f"、過去最大 = {past_max:.0f} 尾" if past_max else ""
                anchor_block = (
                    "\n【直近 trip 参考】\n"
                    + "\n".join(anchor_lines)
                    + f"\n  → 直近 median = {anchor_median:.1f} 尾{past_max_str}。"
                    + "あくまで参考値。大潮・朝マヅメ・好シーズン入り等の好条件、"
                    + "また過去最大級のコンディションでは median を大きく超えて予測してよい。"
                )
            except Exception:
                pass

    # ── 類似日ブロック（Step 3.3: narrow-distribution 魚種限定で発火）
    # 検証結果:
    #   - イサキ (max/median≈1.3): 似日 block で MAE 改善 (6.20 → 5.85)
    #   - ホウボウ (max/median≈4): 似日 max が予測を引き上げ過大予測 → MAE 悪化
    #   - マダイ (max/median≈25): LLM が contradicting signals で median 寄りに hedge
    #     → 改善せず
    # よって max/median < 2.5 の narrow species のみで発火する。
    similar_block = ""
    similar = stats.get("similar_past_trips")
    if similar and similar.get("n_similar", 0) >= 3:
        smax = similar.get("max", 0)
        smedian = similar.get("median", 0)
        spread = (smax / smedian) if smedian > 0 else 999.0
        if spread >= 2.5:
            # wide-distribution: 似日 block を出さない（LLM を混乱させるだけ）
            similar = None
    if similar:
        crit = similar.get("criteria", {})
        scope = crit.get("scope", "all_boats")
        scope_str = "同船宿" if scope == "same_boat" else "全船宿"
        crit_str = f"{scope_str} × 同月±1月"
        if crit.get("tide_phase"):
            crit_str += f" × {crit['tide_phase']}"
        top_lines = []
        for t in similar.get("top_trips", []):
            v = t.get("top_per_angler", "?")
            tide = t.get("tide_phase") or "?"
            boat_str = f" @{t.get('boat','?')}" if scope == "all_boats" else ""
            top_lines.append(f"    {t.get('date','?')}{boat_str}: {v} 尾 ({tide})")
        similar_block = (
            f"\n【類似日 ({crit_str}) の過去実績】 n={similar['n_similar']}\n"
            f"  過去最大: {similar['max']:.0f} 尾、p75: {similar['p75']:.1f}、median: {similar['median']:.1f}\n"
            f"  上位 trip:\n" + "\n".join(top_lines)
            + f"\n  → 類似コンディション下では過去 {similar['max']:.0f} 尾出る日があった。"
            + "今日の海況・潮汐がその時に近ければ、median ではなく上位寄りも合理的予測。"
        )

    return f"""あなたは愛知近海（伊勢湾・三河湾・遠州灘）の釣り船を熟知した予測アシスタントです。
日本語で、釣り船の船長や常連客が頷くような知見を交えて回答してください。

【予測対象】
- 船宿: {boat or "不明"}
- 出船日時: {target_date}  {hour:02d}時出船
- 魚種: {species}
- 本日の狙い: {target_species or species}
- 乗船人数: {anglers if anglers is not None else "不明"}
- 仕掛け: {tackle or "不明"}

【この船 × {species} の過去実績（過去 trip 数: {n_total}、主力シグナル: {primary}）】
{json.dumps(stats, ensure_ascii=False, indent=2, default=str)}
{_SIGNAL_DESCRIPTION}
【今日のコンディション（出船時付近）】
{json.dumps(conditions, ensure_ascii=False, indent=2, default=str)}
{anchor_block}{similar_block}

【予測してほしいこと】
本日この船で釣れる {species} の **竿頭 (個人最大釣果, 尾)** を整数で予測してください。
{tier_basis_hint}。

【5段階評価】
  1 厳しい / 2 やや渋い / 3 普通 / 4 好調 / 5 大漁

confidence:
  - high   : 過去 trip 数 ≥ 20 かつ primary_signal が top_per_angler/count_yolo（数値系）で類似条件あり
  - medium : trip 数が 5〜19、または primary_signal が qualitative
  - low    : trip 数 < 5、または primary_signal が無く LLM 事前知識のみで判断

signal_used: 予測の根拠として最も重視したシグナル。
{schema_hint}
JSON のみで回答してください。
"""


# ============================================================
# プロバイダ別 LLM 呼び出し
# ============================================================

def _call_gemini(
    prompt: str, model: str, api_key: str,
    schema: Optional[dict] = None,
) -> dict:
    """Google Gemini API。schema を渡せば response_schema として強制。"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg_kwargs: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }
    if schema is not None:
        cfg_kwargs["response_schema"] = schema
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    return json.loads(response.text)


def _call_groq(
    prompt: str, model: str, api_key: str,
    schema: Optional[dict] = None,  # Groq OpenAI 互換は response_schema 非対応、プロンプトで誘導
) -> dict:
    """Groq API（OpenAI 互換）— Llama 3.3 70B 等のオープンソースモデル"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1",
                    timeout=20.0, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def _call_cerebras(
    prompt: str, model: str, api_key: str,
    schema: Optional[dict] = None,  # response_format=json_object は対応、schema 渡しは省略
) -> dict:
    """Cerebras Cloud API（OpenAI 互換）— Llama 3.3 70B 等。
    無料 tier の TPM が Groq より広い（30k vs 12k）ので大量バッチ抽出向き。
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1",
                    timeout=20.0, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def _call_ollama(
    prompt: str, model: str, api_key: str = "",
    schema: Optional[dict] = None,
) -> dict:
    """ローカル Ollama API（http://localhost:11434）"""
    import requests
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def _loads_json_lenient(content: str) -> dict:
    """OpenAI 互換でも response_format 非対応な provider 向けの寛容な JSON parse。
    ```json フェンスや前後の散文を除去して最外の {...} を取り出す。"""
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b + 1])
        raise


def _call_openai_compatible(
    base_url: str, prompt: str, model: str, api_key: str,
    use_json_mode: bool = True,
) -> dict:
    """OpenAI 互換エンドポイント共通 caller。"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        # response_format 非対応な model/endpoint は外して再試行
        kwargs.pop("response_format", None)
        response = client.chat.completions.create(**kwargs)
    return _loads_json_lenient(response.choices[0].message.content)


def _call_nvidia(prompt: str, model: str, api_key: str, schema: Optional[dict] = None) -> dict:
    """NVIDIA NIM（integrate.api.nvidia.com, OpenAI 互換）— 無料枠が広い。"""
    return _call_openai_compatible(
        "https://integrate.api.nvidia.com/v1", prompt, model, api_key)


def _call_sambanova(prompt: str, model: str, api_key: str, schema: Optional[dict] = None) -> dict:
    """SambaNova Cloud（api.sambanova.ai, OpenAI 互換）— 高速・無料枠あり。"""
    return _call_openai_compatible(
        "https://api.sambanova.ai/v1", prompt, model, api_key)


def _call_openrouter(prompt: str, model: str, api_key: str, schema: Optional[dict] = None) -> dict:
    """OpenRouter（openrouter.ai, OpenAI 互換）— :free モデル群（日次上限あり）。"""
    return _call_openai_compatible(
        "https://openrouter.ai/api/v1", prompt, model, api_key)


# Provider registry — 新プロバイダ追加はここに1行
# 各 caller は (prompt, model, api_key, schema=None) -> dict のシグネチャ
_PROVIDER_CALLERS: dict[str, Callable[..., dict]] = {
    "gemini":    _call_gemini,
    "groq":      _call_groq,
    "cerebras":  _call_cerebras,
    "ollama":    _call_ollama,
    "nvidia":    _call_nvidia,
    "sambanova": _call_sambanova,
    "openrouter": _call_openrouter,
}


# ============================================================
# 後処理（プロバイダ非依存）
# ============================================================

def _post_process(llm_out: dict, stats: dict) -> dict:
    """tier_label 正規化と vs_boat_avg 計算。"""
    try:
        t = int(llm_out["tier"])
        if 1 <= t <= 5:
            llm_out["tier_label"] = TIER_LABELS[t - 1]
    except (KeyError, ValueError, TypeError):
        pass
    yhat = float(llm_out.get("predicted_top_per_angler", 0.0))
    if stats.get("mean"):
        llm_out["vs_boat_avg"] = round(yhat / stats["mean"], 3)
    if stats.get("median"):
        llm_out["vs_boat_median"] = round(yhat / stats["median"], 3)

    # signal_used が LLM 側から返ってない/不正値なら primary_signal で補完
    sig_used = llm_out.get("signal_used")
    valid_signals = ("top_per_angler", "count_yolo", "total_catch", "qualitative", "none")
    if not isinstance(sig_used, str) or sig_used not in valid_signals:
        llm_out["signal_used"] = stats.get("primary_signal") or "none"

    return llm_out


# プロンプトや入力データの解釈が変わるたびにバンプしてキャッシュを無効化する
# v1 -> v2: 風速を km/h -> m/s 換算してから LLM に渡すよう変更 (2026-06-02)
# v2 -> v3: 大漁日確率を result に含めるよう拡張 (2026-06-03)
# v3 -> v4: 直近 7d/14d activity で確率を補正、result に base_probability/adjustment 追加
# v4 -> v5: SST 7d 変化 + tide_phase 補正を追加して precision を向上
# v5 -> v6: 船宿別 prior (大漁実績多/少) 追加。シーブルー型の偏りを強く反映
_CACHE_VERSION = "v6"


def _cache_path(site: str, species: str, target_date: date, hour: int,
                boat: Optional[str], provider: str, model: str) -> Path:
    safe = lambda s: re.sub(r"[^\w-]", "_", str(s))
    fname = (
        f"{safe(site)}_{safe(species)}_{target_date}_{hour:02d}"
        f"_{safe(boat or 'none')}_{safe(provider)}_{safe(model)}_{_CACHE_VERSION}.json"
    )
    return config.DATA_DIR / "predictions" / fname


# ============================================================
# 中央ディスパッチ
# ============================================================

def predict_with_llm(
    site: str,
    species: str,
    target_date: str | date,
    hour: int = 6,
    boat: Optional[str] = None,
    anglers: Optional[int] = None,
    target_species: Optional[str] = None,
    tackle: Optional[str] = None,
    departure_hour: Optional[int] = None,
    provider: str = "groq",
    model: Optional[str] = None,
    use_cache: bool = True,
    catches_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """LLM ベース釣果予測。

    Args:
        provider: "groq"(default,無料) | "gemini"(課金,高精度reasoning) | "ollama" | "cascade"
        model: 明示指定（None なら provider のデフォルト）
    """
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    # cascade は別ルート
    if provider == "cascade":
        return _predict_cascade(
            site=site, species=species, target_date=target_date, hour=hour,
            boat=boat, anglers=anglers, target_species=target_species,
            tackle=tackle, departure_hour=departure_hour,
            use_cache=use_cache, catches_path=catches_path,
        )

    if provider not in _PROVIDER_CALLERS:
        raise ValueError(
            f"unknown provider: {provider}. "
            f"available: {list(_PROVIDER_CALLERS) + ['cascade']}"
        )

    if model is None:
        model = _PROVIDER_DEFAULTS[provider]

    cache_path = _cache_path(site, species, target_date, hour, boat, provider, model)
    if use_cache and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 1) 過去統計
    stats = compute_boat_stats(species, boat=boat, catches_path=catches_path)

    # 2) 今日のコンディション
    target_dt = pd.Timestamp(datetime.combine(target_date, datetime.min.time()).replace(hour=hour))
    row = data_integrator.build_inference_row(
        site=site, target_dt=target_dt, species=species,
        boat=boat, anglers=anglers,
        target_species=target_species, tackle=tackle,
        departure_hour=departure_hour or hour,
    )
    conditions = _readable_conditions(row)

    # 2.5) 類似コンディション過去 trip lookup（Step 3.2: wide-distribution 魚種の
    #      under-prediction 改善用）
    similar = find_similar_past_trips(
        species=species, boat=boat, target_dt=target_dt,
        target_conditions=conditions, catches_path=catches_path,
    )
    if similar:
        stats["similar_past_trips"] = similar

    # 2.6) 大漁日確率（マダイ等 wide-distribution 魚種の event 検出）
    blowout = compute_blowout_probability(
        species=species, target_dt=target_dt,
        boat=boat, target_conditions=conditions,
    )

    # 3) プロンプト生成
    prompt = _build_prompt(
        species, target_date, hour, boat, anglers,
        target_species, tackle, stats, conditions,
        # Gemini は schema 機構を使うので冗長ヒントは省略可、他は必須
        include_schema_hint=(provider != "gemini"),
    )

    # 4) API キー解決
    api_key = _get_api_key(provider)
    if provider != "ollama" and not api_key:
        tried = ", ".join(_PROVIDER_API_KEYS.get(provider, ()))
        raise RuntimeError(
            f"{provider} の API キーが解決できません。\n"
            f"試した名前: {tried}\n"
            f"Colab: 左サイドバー 🔑 → 該当名で登録 → 「ノートブックでアクセス」ON\n"
            f"既存キー一覧: from src.llm_predictor import _list_available_keys; "
            f"print(_list_available_keys())"
        )

    # 5) LLM 呼び出し
    llm_out = _PROVIDER_CALLERS[provider](prompt, model, api_key, schema=_RESPONSE_SCHEMA)

    # 6) 後処理
    llm_out = _post_process(llm_out, stats)

    result = {
        "site": site,
        "site_name_ja": config.SITES[site].name_ja if site in config.SITES else site,
        "species": species,
        "date": str(target_date),
        "hour": hour,
        "boat": boat,
        "anglers": anglers,
        "tackle": tackle,
        "prediction": llm_out,
        "boat_context": stats,
        "conditions": conditions,
        "blowout": blowout,
        "model": {
            "provider": provider,
            "name": model,
            "label_column": "top_per_angler",
        },
    }

    # キャッシュ書き出し
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass
    return result


def _predict_cascade(
    *, site, species, target_date, hour, boat, anglers, target_species,
    tackle, departure_hour, use_cache, catches_path,
    screen_provider: str = "groq",
    deep_provider: str = "gemini",
    escalate_on: tuple[str, ...] = ("low",),  # low のみ escalate（medium は screen で済ます）
) -> dict:
    """カスケード予測: 安いプロバイダで初回 → 低 conf なら深いプロバイダに escalate。

    閾値設計の指針:
      escalate_on=("low",)          : low だけエスカレ。コスト最小。default
      escalate_on=("low", "medium") : 中程度の confidence もエスカレ。質重視
    """
    # Stage 1: スクリーニング
    try:
        screen = predict_with_llm(
            site=site, species=species, target_date=target_date, hour=hour,
            boat=boat, anglers=anglers, target_species=target_species,
            tackle=tackle, departure_hour=departure_hour,
            provider=screen_provider, use_cache=use_cache, catches_path=catches_path,
        )
    except Exception as e:
        # screen 失敗 → 即 deep へ
        deep = predict_with_llm(
            site=site, species=species, target_date=target_date, hour=hour,
            boat=boat, anglers=anglers, target_species=target_species,
            tackle=tackle, departure_hour=departure_hour,
            provider=deep_provider, use_cache=use_cache, catches_path=catches_path,
        )
        deep["prediction"]["_cascade"] = {
            "escalated": True, "reason": f"screen_error: {e}",
            "screen_provider": screen_provider, "deep_provider": deep_provider,
        }
        deep["model"]["provider"] = "cascade"
        return deep

    screen_conf = screen["prediction"].get("confidence", "low")
    if screen_conf not in escalate_on:
        # high conf → そのまま採用（無料パス）
        screen["prediction"]["_cascade"] = {
            "escalated": False, "screen_confidence": screen_conf,
            "screen_provider": screen_provider, "deep_provider": deep_provider,
        }
        screen["model"]["provider"] = "cascade"
        return screen

    # Stage 2: deep に escalate
    deep = predict_with_llm(
        site=site, species=species, target_date=target_date, hour=hour,
        boat=boat, anglers=anglers, target_species=target_species,
        tackle=tackle, departure_hour=departure_hour,
        provider=deep_provider, use_cache=use_cache, catches_path=catches_path,
    )
    deep["prediction"]["_cascade"] = {
        "escalated": True, "screen_confidence": screen_conf,
        "screen_provider": screen_provider, "deep_provider": deep_provider,
        "screen_prediction": screen["prediction"].get("predicted_top_per_angler"),
        "screen_reasoning": screen["prediction"].get("reasoning"),
    }
    deep["model"]["provider"] = "cascade"
    return deep


# ============================================================
# CLI
# ============================================================

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LLM 釣果予測（プロバイダ選択可）")
    parser.add_argument("--site", required=True, choices=list(config.SITES))
    parser.add_argument("--species", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=5)
    parser.add_argument("--boat", default=None)
    parser.add_argument("--anglers", type=int, default=None)
    parser.add_argument("--tackle", default=None)
    parser.add_argument("--target-species", default=None)
    parser.add_argument("--provider", default="groq",
                        choices=list(_PROVIDER_CALLERS) + ["cascade"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    result = predict_with_llm(
        site=args.site, species=args.species, target_date=args.date,
        hour=args.hour, boat=args.boat, anglers=args.anglers,
        target_species=args.target_species, tackle=args.tackle,
        provider=args.provider, model=args.model,
        use_cache=not args.no_cache,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()
