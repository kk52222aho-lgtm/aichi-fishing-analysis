"""愛知近海 釣果予測 — Streamlit エントリポイント。

起動:
    streamlit run app/streamlit_app.py

ページ:
    1_🎣_単発予測.py        — 船宿×魚種×日付 から LLM 予測
    2_⚔️_船宿ランキング.py  — 同条件で複数船宿の予測を並べる
    3_📊_データ詳細.py      — 取り込み済み 1273 行の可視化
    4_🔄_最新エントリ取り込み.py — ブログ最新エントリを即取り込み
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit Cloud secrets を環境変数にブリッジ（_get_api_key は env を見るため）
# fly.io / Render では DOCKER の ENV 経由で直接 env に入るので no-op
try:
    for _k in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

from utils import (
    available_providers,
    default_provider,
    list_boats,
    list_species,
    load_catches,
    load_registry,
)

st.set_page_config(
    page_title="愛知近海 釣果予測",
    page_icon="🎣",
    layout="wide",
)

st.title("🎣 愛知近海 釣果予測")
st.caption(
    "伊勢湾 / 三河湾 / 遠州灘 の釣り船向け予測。LLM が過去ログ + 気象 + 潮汐 + 月齢を読んで "
    "魚種別 tier (1-5) と reasoning を返す。"
)

# ── 全体ステータス ─────────────────────────────────────
df = load_catches()
reg = load_registry()
providers = available_providers()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("取り込み済 行数", f"{len(df):,}" if not df.empty else "0")
with col2:
    n_trips = df["datetime"].nunique() if not df.empty else 0
    st.metric("出船日数", f"{n_trips:,}")
with col3:
    # registry が空でも catches に船宿があれば数える
    n_boats = (
        len(reg) if reg
        else (df["boat"].nunique() if (not df.empty and "boat" in df.columns) else 0)
    )
    st.metric("船宿数", n_boats)
with col4:
    st.metric("LLM プロバイダ", default_provider() if providers else "❌ なし")

# ── 現状サマリ ─────────────────────────────────────────
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("船宿別 行数")
    if df.empty:
        st.info("catches.csv が空です。Page 4 から取り込みを実行してください。")
    else:
        boat_counts = (
            df.groupby("boat")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="行数")
            .rename(columns={"boat": "船宿"})
        )
        st.dataframe(boat_counts, use_container_width=True, hide_index=True)

with right:
    st.subheader("魚種別 出船日数 top 10")
    if not df.empty:
        sp_counts = (
            df.groupby("species")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="出船日数")
            .rename(columns={"species": "魚種"})
        )
        st.dataframe(sp_counts, use_container_width=True, hide_index=True)

# ── 利用可能 provider 表示 ───────────────────────────────
st.divider()
st.subheader("⚙️ システム状態")
if providers:
    st.success(
        f"利用可能 LLM プロバイダ: **{', '.join(providers)}**（先頭が default）"
    )
else:
    st.error(
        "LLM プロバイダのキーが解決できません。Streamlit Cloud の Secrets に "
        "`CEREBRAS_API_KEY` または `GROQ_API_KEY` を登録してください。"
    )

# ── ナビゲーション ─────────────────────────────────────
st.divider()
st.markdown(
    """
    ### 📑 各ページ
    左サイドバーから移動できます。

    - **🎣 単発予測** — 1 つの船宿×魚種×日付の予測を詳しく見る
    - **⚔️ 船宿ランキング** — 同条件で複数船宿の予測を比較
    - **📊 データ詳細** — 取り込み済データの可視化（魚種別 trend、月別等）
    - **🔄 最新エントリ取り込み** — ブログから今日の釣果を即反映
    """
)
