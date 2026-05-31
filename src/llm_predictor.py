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
    "gemini":   "gemini-2.5-flash",
    "groq":     "llama-3.3-70b-versatile",
    "cerebras": "gpt-oss-120b",
    "ollama":   "llama3.2:3b",
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


def _readable_conditions(row: pd.DataFrame) -> dict[str, Any]:
    r = row.iloc[0].to_dict()
    out: dict[str, Any] = {}
    keys_round = {
        "temperature_2m": 1, "wind_speed_10m": 1, "wind_direction_10m": 0,
        "pressure_msl": 1, "precipitation": 1, "cloud_cover": 0,
        "wave_height": 2, "wave_period": 1, "swell_wave_height": 2,
        "ocean_current_velocity": 2, "ocean_current_direction": 0,
        "sea_surface_temperature": 1, "sea_level_height_msl": 2,
        "tide_cm": 0, "moon_age": 1, "sunrise_hour": 2, "sunset_hour": 2,
    }
    for k, nd in keys_round.items():
        v = r.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[k] = round(float(v), nd)
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

    # ── 直近 trip を予測のアンカーとして強調 (few-shot 効果)
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
                anchor_median = float(np.median([float(v) for v in recent_vals]))
                anchor_block = (
                    "\n【直近 trip アンカー（予測のベースライン）】\n"
                    + "\n".join(anchor_lines)
                    + f"\n  → 直近 median = {anchor_median:.1f} 尾。"
                    + "コンディションが平常なら ±30% 以内に収め、極端な気象/潮汐変化のみ大きく外す。"
                )
            except Exception:
                pass

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
{anchor_block}

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
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
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
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
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


# Provider registry — 新プロバイダ追加はここに1行
# 各 caller は (prompt, model, api_key, schema=None) -> dict のシグネチャ
_PROVIDER_CALLERS: dict[str, Callable[..., dict]] = {
    "gemini":   _call_gemini,
    "groq":     _call_groq,
    "cerebras": _call_cerebras,
    "ollama":   _call_ollama,
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


def _cache_path(site: str, species: str, target_date: date, hour: int,
                boat: Optional[str], provider: str, model: str) -> Path:
    safe = lambda s: re.sub(r"[^\w-]", "_", str(s))
    fname = (
        f"{safe(site)}_{safe(species)}_{target_date}_{hour:02d}"
        f"_{safe(boat or 'none')}_{safe(provider)}_{safe(model)}.json"
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
