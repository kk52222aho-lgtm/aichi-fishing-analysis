"""船宿ランキングページ — 同条件で複数船宿の予測を並列実行してランキング表示。

主な機能:
- ThreadPoolExecutor による並列 LLM 呼び出し (4-5 船宿で 30-60秒 → 10-15秒)
- 過去日付なら catches.csv から実績を引いて predict vs actual を表示
- reasoning / key_factors / risk_factors の全文表示
- ランキング CSV ダウンロード
- データ薄い船宿の警告
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from utils import (
    available_providers,
    boat_to_site,
    confidence_badge,
    list_boats,
    list_species,
    load_catches,
    safe_predict,
    tier_emoji,
)

st.set_page_config(page_title="船宿ランキング", page_icon="⚔️", layout="wide")
st.title("⚔️ 船宿ランキング (vs_other_boats)")
st.caption(
    "同じ魚種・日付・条件で **複数の船宿** に並列予測を投げ、尾数 / tier 順にランキング表示します。"
    " 並列実行により 4-5 船宿で ~10-15 秒。"
)

# ── 入力フォーム ─────────────────────────────────────────
all_boats = list_boats(min_rows=10)  # 学習に足る量があるもののみ
if not all_boats:
    st.error("十分なデータがある船宿がありません。Page 4 から取り込みを実行してください。")
    st.stop()

with st.form("ranking_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        species_list = list_species(min_trips=10)
        default_idx = species_list.index("マダイ") if "マダイ" in species_list else 0
        species = st.selectbox("魚種", species_list, index=default_idx)
        target_date = st.date_input("対象日", value=date.today() + timedelta(days=1))

    with col2:
        hour = st.number_input("出船時刻 (hour)", min_value=0, max_value=23, value=6)
        anglers = st.number_input("乗船人数", min_value=1, max_value=20, value=6)

    with col3:
        tackle = st.text_input("仕掛け (任意)", value="タイラバ" if species == "マダイ" else "")
        providers = available_providers()
        provider = st.selectbox("LLM provider", providers, index=0)

    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        boats_selected = st.multiselect(
            "比較対象の船宿（最大 5 件推奨）",
            all_boats,
            default=all_boats[:4],
        )
    with col_b:
        parallel = st.checkbox("並列実行", value=True, help="OFF にすると順次実行（デバッグ用）")
        max_workers = st.number_input("並列数", min_value=1, max_value=8, value=4)

    submitted = st.form_submit_button("⚔️ ランキング作成", use_container_width=True)


# ── 1 船宿の予測タスク ────────────────────────────────────
def _predict_one(boat: str) -> dict[str, Any]:
    site = boat_to_site(boat) or "morozaki"
    try:
        r = safe_predict(
            site=site,
            species=species,
            target_date=target_date,
            hour=hour,
            boat=boat,
            anglers=int(anglers),
            tackle=tackle or None,
            provider=provider,
            use_cache=True,
        )
        pred = r.get("prediction", {})
        ctx = r.get("boat_context", {})
        return {
            "boat": boat,
            "site": site,
            "予測尾数": pred.get("predicted_top_per_angler"),
            "tier": pred.get("tier"),
            "tier_label": pred.get("tier_label"),
            "confidence": pred.get("confidence"),
            "signal": pred.get("signal_used") or ctx.get("primary_signal"),
            "n_trips": ctx.get("n_trips_total") or ctx.get("n_trips"),
            "reasoning": pred.get("reasoning", ""),
            "key_factors": pred.get("key_factors", []) or [],
            "risk_factors": pred.get("risk_factors", []) or [],
            "_error": None,
        }
    except Exception as e:
        return {
            "boat": boat,
            "site": site,
            "予測尾数": None,
            "tier": None,
            "tier_label": "ERROR",
            "confidence": "error",
            "signal": None,
            "n_trips": None,
            "reasoning": "",
            "key_factors": [],
            "risk_factors": [],
            "_error": str(e),
        }


# ── 過去日付なら実績を引く ──────────────────────────────
def _actual_for(boat: str, target_d: date, species_name: str) -> dict[str, Any] | None:
    """catches.csv から該当 (boat, date, species) の実績行を集計して返す。"""
    df = load_catches()
    if df.empty or "datetime" not in df.columns:
        return None
    mask = (
        (df["boat"] == boat)
        & (df["datetime"].dt.date == target_d)
        & (df["species"] == species_name)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return {
        "top_per_angler": float(rows["top_per_angler"].max()) if "top_per_angler" in rows.columns and rows["top_per_angler"].notna().any() else None,
        "total_catch": float(rows["total_catch"].sum()) if "total_catch" in rows.columns and rows["total_catch"].notna().any() else None,
        "n_rows": len(rows),
    }


# ── 並列予測 ─────────────────────────────────────────────
if submitted:
    if not boats_selected:
        st.warning("船宿を 1 つ以上選んでください。")
        st.stop()

    n = len(boats_selected)
    progress = st.progress(0.0, text=f"予測中... (0/{n})")
    rows: list[dict] = []
    done = 0

    if parallel and n > 1:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            futures = {ex.submit(_predict_one, b): b for b in boats_selected}
            for fut in as_completed(futures):
                rows.append(fut.result())
                done += 1
                progress.progress(done / n, text=f"[{done}/{n}] 完了")
    else:
        for b in boats_selected:
            rows.append(_predict_one(b))
            done += 1
            progress.progress(done / n, text=f"[{done}/{n}] {b} 完了")
    progress.empty()

    # ── 過去実績マージ（target_date < 今日 の時） ──────────
    is_past = target_date < date.today()
    if is_past:
        for r in rows:
            actual = _actual_for(r["boat"], target_date, species)
            r["actual_top"] = actual.get("top_per_angler") if actual else None
            r["actual_total"] = actual.get("total_catch") if actual else None
            r["actual_rows"] = actual.get("n_rows") if actual else 0

    # ── DataFrame ─────────────────────────────────────
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        "予測尾数",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1
    df_sorted.index.name = "順位"

    st.divider()
    header = f"📋 ランキング: {species} × {target_date}"
    if is_past:
        header += "  *(過去日付 / 実績付き)*"
    st.subheader(header)

    # ── サマリ列定義 ─────────────────────────────────
    summary_cols = ["boat", "site", "予測尾数", "tier", "tier_label", "confidence", "signal", "n_trips"]
    if is_past:
        summary_cols.extend(["actual_top", "actual_total", "actual_rows"])

    display_df = df_sorted[summary_cols].copy()

    # tier に絵文字
    display_df["tier"] = display_df["tier"].apply(
        lambda t: f"{tier_emoji(t)} {t}" if pd.notna(t) else "?"
    )
    display_df["confidence"] = display_df["confidence"].apply(confidence_badge)

    # データ薄い船宿に警告アイコン
    def _trip_badge(n) -> str:
        if pd.isna(n):
            return "?"
        n = int(n)
        if n < 5:
            return f"⚠️ {n}"
        if n < 20:
            return f"🟡 {n}"
        return f"🟢 {n}"
    display_df["n_trips"] = display_df["n_trips"].apply(_trip_badge)

    # 過去実績との差分
    if is_past:
        def _delta(row) -> str:
            pred = row["予測尾数"]
            act = row["actual_top"]
            if pd.isna(pred) or pd.isna(act):
                return "-"
            d = pred - act
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1f}"
        display_df["pred-actual"] = df_sorted.apply(_delta, axis=1)

    st.dataframe(display_df, use_container_width=True)

    # ── 棒グラフ ───────────────────────────────────
    try:
        if is_past:
            chart_df = df_sorted[["boat", "予測尾数", "actual_top"]].dropna(subset=["予測尾数"])
            chart_df = chart_df.rename(columns={"actual_top": "実績(top)"})
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("boat"))
        else:
            chart_df = df_sorted[["boat", "予測尾数"]].dropna()
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("boat"))
    except Exception:
        pass

    # ── CSV ダウンロード ──────────────────────────────
    csv_buf = io.StringIO()
    out_cols = ["boat", "site", "予測尾数", "tier", "tier_label", "confidence", "signal", "n_trips", "reasoning"]
    if is_past:
        out_cols.extend(["actual_top", "actual_total", "actual_rows"])
    df_sorted[out_cols].to_csv(csv_buf, index_label="順位")
    st.download_button(
        "📥 ランキング CSV ダウンロード",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"ranking_{species}_{target_date}.csv",
        mime="text/csv",
    )

    # ── 信号混在の解釈ヘルプ ──────────────────────────
    signals = {r["signal"] for r in rows if r.get("signal")}
    if len(signals) > 1:
        st.info(
            "📌 **シグナル混在**: 船宿によりブログ記載スタイルが異なるため "
            f"({', '.join(sorted(signals))})、絶対値の単純比較は注意。"
            " `top_per_angler` (竿頭尾数明示) と `qualitative` (絶好調等の定性表現) は"
            "スケールが違います。tier ラベル比較を優先してください。"
        )

    # ── エラー行 ─────────────────────────────────────
    errors = [r for r in rows if r.get("_error")]
    if errors:
        with st.expander(f"⚠️ エラー {len(errors)} 件"):
            for r in errors:
                st.error(f"**{r['boat']}**: {r['_error']}")

    # ── 各船宿の reasoning（全文 + factors） ─────────
    st.divider()
    st.subheader("💭 各船宿の reasoning")
    for _, r in df_sorted.iterrows():
        if r["tier_label"] == "ERROR":
            continue
        pred_str = f"{r['予測尾数']:.1f} 尾" if pd.notna(r["予測尾数"]) else "—"
        title = (
            f"{r['boat']} ({pred_str} / "
            f"{tier_emoji(r['tier'])} tier {r['tier']} {r['tier_label']} / "
            f"{r['confidence']})"
        )
        with st.expander(title):
            if r["reasoning"]:
                st.markdown(r["reasoning"])
            cols = st.columns(2)
            with cols[0]:
                if r["key_factors"]:
                    st.markdown("**🔑 Key factors**")
                    for f in r["key_factors"]:
                        st.markdown(f"- {f}")
            with cols[1]:
                if r["risk_factors"]:
                    st.markdown("**⚠️ Risk factors**")
                    for f in r["risk_factors"]:
                        st.markdown(f"- {f}")
            if is_past and pd.notna(r.get("actual_top")):
                st.divider()
                st.markdown(
                    f"**📊 実績**: top={r['actual_top']:.0f}尾 "
                    f"/ total={r.get('actual_total', '-')} "
                    f"/ 記録 {r.get('actual_rows', 0)} 行"
                )
