"""Streamlit app の共通ユーティリティ。

責務:
- リポジトリルートを sys.path に通す（Colab / ローカル両対応）
- catches.csv / registry を @st.cache_data で読む
- LLM プロバイダの選択肢（環境にあるキーから自動選別）
- predict ラッパー（プロバイダ未指定なら cerebras > groq > gemini の順で fallback）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── リポジトリルートの解決 ──────────────────────────────────
# app/ の親 = リポジトリルート（src/ と同階層）
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent

# src/ をインポートできるようにする
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# src 配下は repo_root をパスに入れた後でないと import 失敗するので遅延配置
from src import config  # noqa: E402
from src.llm_predictor import (  # noqa: E402
    predict_with_llm,
    compute_boat_stats,
    _PROVIDER_DEFAULTS,
    _PROVIDER_API_KEYS,
    _get_api_key,
)
from src.scrape_to_catches import load_blog_registry  # noqa: E402


CATCHES_CSV = config.FISHING_DIR / "catches.csv"


# ── データロード（cache） ────────────────────────────────
@st.cache_data(ttl=60)
def load_catches() -> pd.DataFrame:
    """catches.csv を読む。ファイルが無い時は空 DataFrame。"""
    if not CATCHES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(CATCHES_CSV)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_registry() -> dict[str, dict]:
    """_blog_registry.json を読む。"""
    return load_blog_registry()


def list_boats(min_rows: int = 1) -> list[str]:
    """catches.csv に出現する船宿一覧（行数 desc）。"""
    df = load_catches()
    if df.empty or "boat" not in df.columns:
        return []
    counts = df.groupby("boat").size().sort_values(ascending=False)
    return [b for b, n in counts.items() if n >= min_rows]


def list_species(min_trips: int = 5) -> list[str]:
    """catches.csv に出現する魚種一覧（trip 数 desc, 最低 N trip 以上）。"""
    df = load_catches()
    if df.empty or "species" not in df.columns:
        return []
    counts = df.groupby("species").size().sort_values(ascending=False)
    return [s for s, n in counts.items() if n >= min_trips]


def list_sites() -> list[tuple[str, str]]:
    """(code, name_ja) のタプル一覧。"""
    return [(code, s.name_ja) for code, s in config.SITES.items()]


def boat_to_site(boat: str) -> Optional[str]:
    """船宿名から site code を逆引き（registry 経由）。"""
    reg = load_registry()
    for blog_id, entry in reg.items():
        if entry.get("boat") == boat:
            return entry.get("site")
    return None


def boat_to_blog_id(boat: str) -> Optional[str]:
    """船宿名から blog_id を逆引き。"""
    reg = load_registry()
    for blog_id, entry in reg.items():
        if entry.get("boat") == boat:
            return blog_id
    return None


# ── LLM プロバイダ選択 ────────────────────────────────────
def available_providers() -> list[str]:
    """API キーが解決できる provider 一覧。

    優先順: cerebras (無料) > groq (無料) > ollama (ローカル無料) > gemini (課金リスク)。
    無料枠を使い切るまで gemini にフォールバックしないため意図的にこの順序。
    """
    priority = ["cerebras", "groq", "ollama", "gemini"]
    out = []
    for p in priority:
        if p == "ollama":
            # ollama はキー不要、URL が立ってる前提で常に候補に出す
            out.append(p)
            continue
        try:
            if _get_api_key(p):
                out.append(p)
        except Exception:
            pass
    return out or ["groq"]


def default_provider() -> str:
    """利用可能な provider の中で最も TPM が広いもの。"""
    avail = available_providers()
    return avail[0] if avail else "groq"


def safe_predict(
    site: str,
    species: str,
    target_date,
    *,
    hour: int = 6,
    boat: Optional[str] = None,
    anglers: Optional[int] = None,
    tackle: Optional[str] = None,
    target_species: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """predict_with_llm のラッパー。provider 未指定なら available_providers の先頭。

    rate limit を自動 retry（cerebras → groq → gemini）するため、
    primary 失敗時は次の provider に切り替えてもう一度だけ試す。
    """
    providers_try = [provider] if provider else available_providers()
    last_err: Optional[Exception] = None
    for prov in providers_try:
        if prov is None:
            continue
        try:
            return predict_with_llm(
                site=site, species=species, target_date=target_date,
                hour=hour, boat=boat, anglers=anglers, tackle=tackle,
                target_species=target_species,
                provider=prov, model=model, use_cache=use_cache,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            # rate limit 系なら次のプロバイダへフォールバック
            if "429" in msg or "rate" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                continue
            # それ以外（プロンプト/データ系エラー）は即 raise
            raise
    if last_err:
        raise last_err
    raise RuntimeError("no provider available")


def tier_emoji(tier: int) -> str:
    """tier (1-5) を絵文字で表現。"""
    return {1: "😣", 2: "😕", 3: "😐", 4: "😊", 5: "🎉"}.get(int(tier or 0), "❓")


def confidence_badge(conf: str) -> str:
    """信頼度を色付きバッジ風文字列で（日本語）。"""
    return {"high": "🟢 高", "medium": "🟡 中", "low": "🔴 低"}.get(
        (conf or "").lower(), conf or "?"
    )


# LLM が返す signal_used の機械名を日本語ラベルに変換
_SIGNAL_LABELS_JA = {
    "top_per_angler": "竿頭釣果 (本文抽出)",
    "count_yolo": "YOLO 自動カウント",
    "total_catch": "船合計釣果",
    "qualitative": "定性評価",
    "none": "（なし）",
}


def signal_label_ja(signal: str) -> str:
    """signal_used の機械名を日本語ラベルへ。未知ならそのまま返す。"""
    if not signal:
        return "?"
    return _SIGNAL_LABELS_JA.get(signal, signal)


# ============================================================
# 出船可否判定（台風/強風/高波で予測値より先に欠航見込みを伝える）
# ============================================================

def boat_status(cond: dict[str, Any]) -> dict[str, Any]:
    """当日コンディションから「出船できそうか」を判定。

    Args:
        cond: _readable_conditions() の出力（風速 m/s, 波高 m 等）

    Returns:
        {
            "level":    "ok" | "caution" | "hard" | "no_go",
            "emoji":    "☀️" / "✅" / "⚠️" / "⛔",
            "label":    "ベタ凪" / "出船可能" / "厳しい海況" / "出船困難",
            "reasons":  ["風速 22 m/s（台風級）", "波高 2.5 m"],
            "summary":  "風速 22 m/s と波高 2.5 m。多くの船宿は欠航見込み。",
        }
    """
    if not cond:
        return {
            "level": "unknown", "emoji": "❓", "label": "判定不能",
            "reasons": [], "summary": "コンディション情報なし",
        }
    wind = cond.get("wind_speed_10m")
    gust = cond.get("wind_gusts_10m")
    wave = cond.get("wave_height")
    swell = cond.get("swell_wave_height")
    precip = cond.get("precipitation")

    reasons_no_go: list[str] = []
    reasons_hard: list[str] = []
    reasons_caution: list[str] = []

    # 風速 m/s 基準（一般的な釣り船の出船可否目安）
    try:
        ws = float(wind) if wind is not None else None
    except (TypeError, ValueError):
        ws = None
    if ws is not None:
        if ws >= 15:
            reasons_no_go.append(f"風速 {ws:.0f} m/s（台風級）")
        elif ws >= 12:
            reasons_no_go.append(f"風速 {ws:.0f} m/s（猛烈な強風）")
        elif ws >= 8:
            reasons_hard.append(f"風速 {ws:.0f} m/s（強風）")
        elif ws >= 4:
            reasons_caution.append(f"風速 {ws:.0f} m/s")

    # 突風
    try:
        gs = float(gust) if gust is not None else None
    except (TypeError, ValueError):
        gs = None
    if gs is not None and gs >= 18:
        reasons_no_go.append(f"突風 {gs:.0f} m/s")

    # 波高 m 基準
    try:
        wv = float(wave) if wave is not None else None
    except (TypeError, ValueError):
        wv = None
    if wv is not None:
        if wv >= 2.5:
            reasons_no_go.append(f"波高 {wv:.1f} m")
        elif wv >= 1.5:
            reasons_hard.append(f"波高 {wv:.1f} m")
        elif wv >= 1.0:
            reasons_caution.append(f"波高 {wv:.1f} m")

    # うねり
    try:
        sw = float(swell) if swell is not None else None
    except (TypeError, ValueError):
        sw = None
    if sw is not None and sw >= 2.0:
        reasons_no_go.append(f"うねり {sw:.1f} m")

    # 降水
    try:
        pp = float(precip) if precip is not None else None
    except (TypeError, ValueError):
        pp = None
    if pp is not None and pp >= 20:
        reasons_hard.append(f"降水 {pp:.0f} mm/h（豪雨）")

    if reasons_no_go:
        return {
            "level": "no_go", "emoji": "🚩",
            "label": "多くの船宿が判断に迷う海況",
            "reasons": reasons_no_go + reasons_hard + reasons_caution,
            "summary": "強風・高波・突風で **船宿によって出船判断が分かれます**。"
                       "出船予定なら船宿に**欠航確認の電話**を強く推奨。"
                       "（実測検証: この判定は厳しめに振れる傾向あり）",
        }
    if reasons_hard:
        return {
            "level": "hard", "emoji": "⚠️",
            "label": "厳しい海況",
            "reasons": reasons_hard + reasons_caution,
            "summary": "出船はできても船酔い注意、ポイント制限がかかる可能性。",
        }
    if reasons_caution:
        return {
            "level": "caution", "emoji": "✅",
            "label": "出船可能（やや風あり）",
            "reasons": reasons_caution,
            "summary": "釣り日和。コンディションに大きな問題なし。",
        }
    return {
        "level": "ok", "emoji": "☀️",
        "label": "ベタ凪",
        "reasons": [],
        "summary": "穏やかな海。絶好の釣り日和。",
    }
