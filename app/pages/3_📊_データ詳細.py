"""データ詳細ダッシュボード — 船宿別 trips / 魚種別 trend / 取り込み状況。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import load_catches, load_registry

st.set_page_config(page_title="データ詳細", page_icon="📊", layout="wide")
st.title("📊 データ詳細ダッシュボード")
st.caption("取り込み済 catches.csv を可視化。LLM API は呼ばないので即表示。")

df = load_catches()
if df.empty:
    st.warning("catches.csv が空です。Page 4 から取り込みを実行してください。")
    st.stop()

# ── 全体サマリ ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("総行数", f"{len(df):,}")
with col2:
    st.metric("trip 日付数", f"{df['datetime'].nunique():,}")
with col3:
    st.metric("船宿数", df["boat"].nunique() if "boat" in df.columns else 0)
with col4:
    st.metric("魚種数", df["species"].nunique() if "species" in df.columns else 0)

st.divider()

# ── 船宿別 trip 数 + 直近エントリ日付 ─────────────────────
st.subheader("🚢 船宿別 行数 / 直近エントリ")
boat_summary = (
    df.groupby("boat")
    .agg(
        rows=("species", "size"),
        n_trips=("datetime", "nunique"),
        latest=("datetime", "max"),
        oldest=("datetime", "min"),
    )
    .sort_values("rows", ascending=False)
    .reset_index()
)
boat_summary["latest"] = pd.to_datetime(boat_summary["latest"]).dt.strftime("%Y-%m-%d")
boat_summary["oldest"] = pd.to_datetime(boat_summary["oldest"]).dt.strftime("%Y-%m-%d")
st.dataframe(boat_summary, use_container_width=True, hide_index=True)

st.bar_chart(boat_summary.set_index("boat")["rows"], height=200)

st.divider()

# ── 魚種別 trip 数 top 20 ──────────────────────────────
st.subheader("🐟 魚種別 trip 数 (top 20)")
species_summary = (
    df.groupby("species")
    .agg(
        trips=("datetime", "nunique"),
        top_max=("top_per_angler", "max"),
        total_sum=("total_catch", "sum"),
        yolo_sum=("count_yolo", "sum"),
    )
    .sort_values("trips", ascending=False)
    .head(20)
    .reset_index()
)
st.dataframe(species_summary, use_container_width=True, hide_index=True)
st.bar_chart(species_summary.set_index("species")["trips"], height=250)

st.divider()

# ── 魚種別 月別 trend ──────────────────────────────────
st.subheader("📈 魚種別 月別 trend")
top_species = species_summary["species"].head(8).tolist()
selected = st.multiselect(
    "表示する魚種（top 8 から）",
    species_summary["species"].head(20).tolist(),
    default=top_species[:5],
)

if selected:
    sub = df[df["species"].isin(selected)].copy()
    sub["month"] = sub["datetime"].dt.to_period("M").astype(str)
    trend = (
        sub.groupby(["month", "species"])
        .size()
        .reset_index(name="trips")
        .pivot(index="month", columns="species", values="trips")
        .fillna(0)
    )
    st.line_chart(trend, height=300)
else:
    st.caption("魚種を選択するとグラフが表示されます。")

st.divider()

# ── 船宿 × 魚種 ヒートマップ風 ──────────────────────────
st.subheader("🔥 船宿 × 主要魚種 trip 数")
pivot = (
    df[df["species"].isin(species_summary["species"].head(12).tolist())]
    .groupby(["boat", "species"])
    .size()
    .unstack(fill_value=0)
)
# 主要魚種順に列を並べ替え
pivot = pivot[[s for s in species_summary["species"].head(12) if s in pivot.columns]]
st.dataframe(
    pivot.style.background_gradient(cmap="YlOrRd", axis=None),
    use_container_width=True,
)

st.divider()

# ── 取り込み状況（registry） ────────────────────────────
st.subheader("📋 取り込み状況 (registry)")
reg = load_registry()
if reg:
    reg_df = pd.DataFrame([
        {
            "blog_id": k,
            "boat": v.get("boat"),
            "site": v.get("site"),
            "primary_signal": v.get("primary_signal"),
            "secondary_signal": v.get("secondary_signal"),
        }
        for k, v in reg.items()
    ])
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
else:
    st.caption("registry が空です。")
