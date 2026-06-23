"""釣果カレンダー — 場所×月の魚種構成・平年の釣果・推移を可視化（記述的分析）。

予測ではなく「過去にいつ・どこで・何が・どれくらい釣れてきたか」の実績集計。
日次の釣果良し悪しは出船前情報では予測できないと検証で判明したため、
季節で明確に変わる「魚種の旬」「平年の傾向」を見せることに振った可視化。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import load_catches
from src import config

st.set_page_config(page_title="釣果カレンダー", page_icon="📅", layout="wide")
st.title("📅 釣果カレンダー（季節・場所別の傾向）")
st.caption("過去の釣果ログの実績集計です。予測ではなく『いつ・どこで・何が・どれくらい釣れてきたか』。")

SITE_JA = {code: s.name_ja for code, s in config.SITES.items()}
GOOD = {"好調", "絶好調", "爆釣", "大漁"}
MONTH_COLS = list(range(1, 13))

df = load_catches()
if df.empty:
    st.warning("catches.csv が空です。Page 4 から取り込みを実行してください。")
    st.stop()

df = df.dropna(subset=["datetime", "species"]).copy()
df["month"] = df["datetime"].dt.month
df["ym"] = df["datetime"].dt.to_period("M").astype(str)

# ── 場所フィルタ ───────────────────────────────────────
sites_in_data = [s for s in df["site"].dropna().unique() if s]
site_opts = ["（全体）"] + sorted(sites_in_data, key=lambda c: SITE_JA.get(c, c))
sel = st.selectbox(
    "場所", site_opts,
    format_func=lambda c: "（全体）" if c == "（全体）" else SITE_JA.get(c, c),
)
view = df if sel == "（全体）" else df[df["site"] == sel]

c1, c2, c3, c4 = st.columns(4)
c1.metric("対象行数", f"{len(view):,}")
c2.metric("trip 日数", f"{view['datetime'].dt.date.nunique():,}")
c3.metric("魚種数", view["species"].nunique())
c4.metric("期間", f"{view['datetime'].min():%y/%m}〜{view['datetime'].max():%y/%m}")

st.divider()

# ── ① 旬カレンダー（魚種×月ヒートマップ）────────────────────
st.subheader("① 旬カレンダー（魚種 × 月）")
st.caption("各セル＝その魚種がその月に記録された trip 日数。色は『魚種ごと』に正規化（行内比較）— 各魚種の旬（濃い月）が分かります。")

top_n = st.slider("表示する魚種数（trip 数 上位）", 5, 30, 18)
top_species = (
    view.groupby("species")["datetime"].apply(lambda s: s.dt.date.nunique())
    .sort_values(ascending=False).head(top_n).index.tolist()
)
sub = view[view["species"].isin(top_species)]
pivot = (
    sub.groupby(["species", "month"])["datetime"].apply(lambda s: s.dt.date.nunique())
    .unstack("month").reindex(index=top_species, columns=MONTH_COLS).fillna(0)
)
pivot.columns = [f"{m}月" for m in pivot.columns]
st.dataframe(
    pivot.style.background_gradient(cmap="YlOrRd", axis=1).format("{:.0f}"),
    use_container_width=True,
)

st.divider()

# ── ② 魚種を選んで季節詳細 ──────────────────────────────
st.subheader("② 魚種の季節詳細")
sp = st.selectbox("魚種", top_species)
spv = view[view["species"] == sp].copy()

g = spv.groupby("month")
monthly = pd.DataFrame({
    "trip日数": g["datetime"].apply(lambda s: s.dt.date.nunique()),
}).reindex(MONTH_COLS).fillna(0)

# 好調率（qualitative がある行のうち 好調系の割合）
q = spv.dropna(subset=["qualitative"])
if not q.empty:
    q_good = q.assign(good=q["qualitative"].isin(GOOD).astype(int))
    monthly["好調率"] = q_good.groupby("month")["good"].mean().reindex(MONTH_COLS)
# 平年の匹数（top_per_angler 月別中央値、データがある月のみ）
tpa = spv.dropna(subset=["top_per_angler"])
if not tpa.empty:
    monthly["匹数中央値"] = tpa.groupby("month")["top_per_angler"].median().reindex(MONTH_COLS)

monthly.index = [f"{m}月" for m in MONTH_COLS]

cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**月別 trip 日数（どの月に出る魚か）**")
    st.bar_chart(monthly["trip日数"], height=240)
with cc2:
    if "好調率" in monthly.columns:
        st.markdown("**月別 好調率（好調+絶好調 の割合）**")
        st.bar_chart(monthly["好調率"].fillna(0), height=240)
    else:
        st.info("この魚種は質的評価データがありません。")

st.markdown("**月別サマリ**")
show = monthly.copy()
if "好調率" in show.columns:
    show["好調率"] = (show["好調率"] * 100).round(0)
st.dataframe(show.fillna("-"), use_container_width=True)

# 平年の匹数 全体統計
if not tpa.empty:
    s = tpa["top_per_angler"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("匹数 中央値", f"{s.median():.0f}")
    m2.metric("匹数 上位25%", f"{s.quantile(0.75):.0f}")
    m3.metric("匹数 最大", f"{s.max():.0f}")
    m4.metric("数値記録数", f"{len(s)}")

st.divider()

# ── ③ 月次トレンド（活動量の推移）──────────────────────────
st.subheader("③ 月次トレンド（記録された trip 日数の推移）")
trend = (
    view.groupby("ym")["datetime"].apply(lambda s: s.dt.date.nunique())
    .rename("trip日数").to_frame()
)
st.line_chart(trend, height=240)

st.caption("※ trip 日数は『その月に釣果が記録された日数』。船宿の投稿頻度にも依存します（実釣行数そのものではない）。")
