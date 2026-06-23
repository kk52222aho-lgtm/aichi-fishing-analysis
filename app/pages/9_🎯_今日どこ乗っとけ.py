"""今日どこ乗っとけ — 船を選ばず、日付だけで [厚い船 × その月の主力魚種] を相対tier順に並べる横断ビュー。

このシステム唯一の強み = 複数船宿ブログを横断して持っていること。個々の船は自分のブログしか
出さない。横断ビューはこのデータでしか作れない。だからこのページは「船も魚種も選ばせず、
日付だけ」で、船×魚種を相対tier(高め/並/低め)でグループ表示する。

設計判断:
- 入力は日付のみ。船宿・魚種はユーザーに選ばせない。
- 対象船は catches.csv の行数で自動足切り(厚い船だけ)。船が増えれば自動で対象入り。
- 主力魚種は「その船のその月の trip 数上位」を自動抽出（手動固定は保守増なので後回し）。
- 絶対匹数は画面に出さない（弱いシグナルを誇張しない / 「16匹て言うたやんけ」を防ぐ）。
- tier 内は順位をつけない（順位は弱いシグナルなので主張しない）。行名順で並べるだけ。

既存資産の再利用: utils.list_boats(行数閾値で自動抽出) / safe_predict(船宿比較と同じ予測関数) /
釣果カレンダーと同じ trip 数集計(boat フィルタ版)。新規学習・スクレイプは無し。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import streamlit as st

from utils import (
    boat_to_site,
    default_provider,
    list_boats,
    load_catches,
    safe_predict,
)

st.set_page_config(page_title="今日どこ乗っとけ", page_icon="🎯", layout="wide")
st.title("🎯 今日どこ乗っとけ")
st.caption("船も魚種も選ばず、日付だけ。データの厚い船 × その月の主力魚種を、平年比の相対傾向で並べます。")
st.info(
    "ℹ️ この予測は絶対的な釣果数を当てるものではありません。出船前情報だけでは、その日の釣果数は"
    "統計的に予測できないことが検証で確認されています。本ツールが示すのは「平年比での相対的な期待度"
    "（高め/並/低め）」であり、これも弱い傾向（順位相関 0.1〜0.3 程度）にとどまります。"
    "釣行判断の主役ではなく、参考の一つとしてご利用ください。"
)

# ── パラメータ（後で船が増えたら自動で対象入りするよう閾値式にしている） ──
MIN_BOAT_ROWS = 300     # 「厚い船」の足切り行数
TOP_K_SPECIES = 2       # 各船で拾う主力魚種の数
MIN_SPECIES_TRIPS = 3   # 主力魚種に採用する最低 trip 数（その月）

df = load_catches()
if df.empty:
    st.warning("catches.csv が空です。Page 4 から取り込みを実行してください。")
    st.stop()

target_date = st.date_input("日付", value=date.today() + timedelta(days=1))
run = st.button("この日の傾向を見る", type="primary", use_container_width=True)


def top_species_for(boat: str, month: int) -> list[str]:
    """その船・その月に trip 数が多い主力魚種を上位 K 件返す（釣果カレンダーと同じ集計の boat 版）。"""
    d = df[(df["boat"] == boat) & (df["datetime"].dt.month == month)]
    if d.empty:
        return []
    counts = (
        d.groupby("species")["datetime"].apply(lambda s: s.dt.date.nunique())
        .sort_values(ascending=False)
    )
    counts = counts[counts >= MIN_SPECIES_TRIPS]
    return counts.head(TOP_K_SPECIES).index.tolist()


if run:
    boats = list_boats(min_rows=MIN_BOAT_ROWS)
    if not boats:
        st.warning(f"データが {MIN_BOAT_ROWS} 行以上の船宿がまだありません。")
        st.stop()

    month = target_date.month
    combos = [(b, sp) for b in boats for sp in top_species_for(b, month)]
    if not combos:
        st.warning(f"{month}月の主力魚種データが足りません（各船の同月 trip が少ない）。")
        st.stop()

    st.caption(
        f"対象: 厚い船 {len(boats)} 隻（{MIN_BOAT_ROWS}行以上）× {month}月の主力魚種"
        f" = {len(combos)} 通りを評価"
    )
    provider = default_provider()

    def predict_combo(bs: tuple[str, str]) -> dict:
        boat, sp = bs
        try:
            r = safe_predict(
                site=boat_to_site(boat) or "morozaki",
                species=sp, target_date=target_date,
                boat=boat, provider=provider, use_cache=True,
            )
            return {"boat": boat, "species": sp,
                    "tier": r.get("prediction", {}).get("tier"), "err": None}
        except Exception as e:
            return {"boat": boat, "species": sp, "tier": None, "err": str(e)}

    prog = st.progress(0.0, text="評価中...")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(predict_combo, c): c for c in combos}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            prog.progress(i / len(combos), text=f"評価中... [{i}/{len(combos)}]")
    prog.empty()

    ok = [r for r in results if r["tier"] is not None]
    if not ok:
        st.error(
            "予測を取得できませんでした。予測には LLM API キーが必要です"
            "（アプリの Settings → Secrets に CEREBRAS_API_KEY 等を設定）。"
        )
        errs = [r["err"] for r in results if r["err"]]
        if errs:
            st.caption(f"エラー例: {errs[0][:200]}")
        st.stop()

    def band(t: int) -> str:
        return "high" if t >= 4 else ("mid" if t == 3 else "low")

    groups: dict[str, list[str]] = {"high": [], "mid": [], "low": []}
    for r in sorted(ok, key=lambda r: (r["boat"], r["species"])):  # 行名順・順位なし
        groups[band(int(r["tier"]))].append(f"**{r['boat']}** × {r['species']}")

    st.subheader(f"{target_date.month}/{target_date.day} の \"ここ乗っとけ\"")
    if groups["high"]:
        st.markdown("### 🔴 高め傾向")
        for x in groups["high"]:
            st.markdown(f"- {x}")
    else:
        st.success("今日は平年並み。どこも大きな差はなさそうです。")

    if groups["mid"]:
        st.markdown("### 🟡 並")
        for x in groups["mid"]:
            st.markdown(f"- {x}")

    if groups["low"]:
        with st.expander("🔵 低め傾向（参考）"):
            for x in groups["low"]:
                st.markdown(f"- {x}")

    st.caption("※ tier 内は順位を付けていません（差は弱いため）。絶対匹数は表示しません。")
