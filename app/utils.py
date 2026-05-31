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
    """信頼度を色付きバッジ風文字列で。"""
    return {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}.get(
        (conf or "").lower(), conf or "?"
    )
