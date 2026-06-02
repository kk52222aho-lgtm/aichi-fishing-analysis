"""精度評価ページ — walk-forward backtest の結果を表示 + 新規実行も可能。

CSV は `data/integrated/backtest_*.csv` に永続化されているものを読む。
無ければ「実行する」ボタンで生成。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    REPO_ROOT,
    available_providers,
    list_species,
)
from src import backtest, config, predictions_log  # noqa: E402

st.set_page_config(page_title="精度評価", page_icon="📈", layout="wide")
st.title("📈 精度評価")
st.caption(
    "**運用ログ** (実際の予測 vs 実績) と **walk-forward backtest** (履歴での検証) "
    "の両方を見る。バックテストは「過去にこう予測してたら」、運用ログは「実際にこう予測した」。"
)

INTEGRATED_DIR: Path = config.INTEGRATED_DIR


# ============================================================
# 運用ログ（実際の /predict と /feedback ペア）
# ============================================================
st.subheader("🎯 運用ログ精度 (実予測 vs 実績)")
st.caption(
    "サーバの `/predict` 呼び出しはすべて `data/predictions_log.csv` に記録され、"
    "出船後の `/feedback` で実績が紐付けられる。下記はその集計。"
)

species_filter = st.selectbox(
    "魚種フィルタ",
    ["（全体）"] + list_species(min_trips=3),
    key="opslog_sp",
)
sp_arg = None if species_filter == "（全体）" else species_filter
log_summary = predictions_log.compute_summary(species=sp_arg)

if log_summary.get("n_total", 0) == 0:
    st.info("まだ予測ログがありません。Page 1 から予測を実行すると記録が始まります。")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("総予測数", log_summary.get("n_total", 0))
    c2.metric("実績紐付け済", log_summary.get("n_with_feedback", 0))
    c3.metric(
        "feedback 率",
        f"{(log_summary.get('feedback_rate') or 0) * 100:.0f}%",
    )
    if "mae" in log_summary:
        c4.metric("運用 MAE", f"{log_summary['mae']} 尾")
    if "bias" in log_summary:
        c5.metric("bias", f"{log_summary['bias']:+.2f}")

    # engine 別内訳
    by_eng = log_summary.get("by_engine") or {}
    if by_eng:
        st.markdown("**Engine 別 (実績紐付け済みのみ)**")
        eng_rows = [
            {"engine": k, "n": v["n"], "mae": v["mae"], "bias": v["bias"]}
            for k, v in by_eng.items()
        ]
        st.dataframe(pd.DataFrame(eng_rows), hide_index=True, use_container_width=True)

# ── フィードバック投稿フォーム ───────────────────────────
with st.expander("➕ 実績を入力 (出船後)", expanded=False):
    st.caption(
        "予測時に発行された **prediction_id** に対して、その日実際に何尾釣れたかを送信。"
        "未紐付けの予測候補から選ぶこともできる。"
    )

    pending = predictions_log.find_recent_predictions(
        only_pending_feedback=True, limit=20,
    )
    if pending:
        labels = [
            f"{p.get('target_date')} {p.get('site')} {p.get('species')} "
            f"@{p.get('boat') or '?'} [{p.get('prediction_id')[:8]}...] "
            f"→ 予測 {p.get('predicted_top_per_angler')} 尾"
            for p in pending
        ]
        choice = st.selectbox("未紐付けの予測から選ぶ", labels, key="fb_pending")
        chosen_pid = pending[labels.index(choice)]["prediction_id"]
        st.code(f"prediction_id = {chosen_pid}", language="text")
    else:
        chosen_pid = st.text_input(
            "prediction_id (16 桁の hex)", key="fb_pid_manual",
        )

    fb_actual = st.number_input(
        "実際の竿頭釣果 (個人最大、尾)",
        min_value=0.0, value=0.0, step=1.0, key="fb_actual",
    )
    fb_qual = st.selectbox(
        "定性 (任意)",
        ["", "大漁", "好調", "普通", "渋い", "厳しい", "ボウズ"],
        key="fb_qual",
    )
    fb_notes = st.text_area("メモ (任意)", key="fb_notes", height=70)

    if st.button("📨 フィードバック送信", key="fb_submit"):
        if not chosen_pid:
            st.warning("prediction_id を選ぶか入力してください。")
        else:
            try:
                r = predictions_log.link_feedback(
                    prediction_id=chosen_pid,
                    actual_top_per_angler=float(fb_actual),
                    actual_qualitative=fb_qual or None,
                    notes=fb_notes or None,
                )
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.success(f"✅ 紐付け完了。実績 {fb_actual} 尾を記録しました。")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"失敗: {e}")

st.divider()


# ============================================================
# Walk-forward backtest（履歴での検証、既存セクション）
# ============================================================
st.subheader("🧪 walk-forward backtest (履歴検証)")
st.caption(
    "trip i の予測には trip[0..i-1] までしか使わない。"
    " 全 trip の集計で **MAE / 相関 / tier 一致率 / baseline 比較** を出す。"
)


def _summarize_csv(csv_path: Path) -> Optional[dict]:
    """既存 CSV から summarize を計算。"""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        sp = csv_path.stem.split("_")[-1]
        return backtest.summarize(df, sp)
    except Exception as e:
        return {"error": str(e), "path": str(csv_path)}


# ── 既存 backtest 結果スキャン ─────────────────────────────
st.subheader("📂 既存 backtest 結果")
existing = sorted(INTEGRATED_DIR.glob("backtest_*.csv"))
if not existing:
    st.info("まだ backtest 結果がありません。下のセクションから実行してください。")
else:
    summaries = []
    for p in existing:
        s = _summarize_csv(p)
        if not s or "error" in (s or {}):
            continue
        # ファイル名から種別判定
        name = p.stem
        if "llm_" in name:
            kind = "LLM"
            # backtest_llm_<provider>[_<model>]_<species>
            tokens = name.split("_")
            provider = tokens[2] if len(tokens) > 2 else "?"
            model_label = provider
        else:
            kind = "統計モデル"
            model_label = "LightGBM/sklearn"
        s["kind"] = kind
        s["model_label"] = model_label
        s["file"] = p.name
        summaries.append(s)

    if summaries:
        # 主要メトリクスだけ表に
        cols = [
            "species", "kind", "model_label", "n_predictions",
            "model_mae", "baseline_mae", "vs_baseline_pct",
            "correlation", "tier_exact_match", "tier_within_1", "bias",
        ]
        df_sum = pd.DataFrame(summaries)[
            [c for c in cols if c in summaries[0]]
        ]
        df_sum = df_sum.sort_values(["species", "kind"]).reset_index(drop=True)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        # ── 改善率の可視化 ─────────────────────────
        impr_df = df_sum[df_sum["vs_baseline_pct"].notna()].copy()
        if not impr_df.empty:
            impr_df["label"] = impr_df["species"] + " (" + impr_df["model_label"] + ")"
            st.bar_chart(
                impr_df.set_index("label")["vs_baseline_pct"],
                height=250,
            )
            st.caption("vs_baseline_pct: + なら baseline より改善、- なら悪化（平均値予測同等）")

# ── 個別結果の詳細閲覧 ─────────────────────────────────────
st.divider()
st.subheader("🔍 個別結果の詳細")
if existing:
    selected = st.selectbox("詳細を見る CSV", [p.name for p in existing])
    csv_path = INTEGRATED_DIR / selected
    df_detail = pd.read_csv(csv_path)
    if "datetime" in df_detail.columns:
        df_detail["datetime"] = pd.to_datetime(df_detail["datetime"], errors="coerce")

    s = _summarize_csv(csv_path)
    if s and "error" not in s:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("N (予測数)", s.get("n_predictions"))
        with c2:
            st.metric("MAE", f"{s.get('model_mae')} 尾")
        with c3:
            improvement = s.get("vs_baseline_pct")
            st.metric(
                "vs baseline",
                f"{improvement * 100:+.1f}%" if improvement is not None else "-",
            )
        with c4:
            st.metric("tier ±1 一致", f"{s.get('tier_within_1', 0) * 100:.0f}%")

    # 時系列グラフ
    if "actual" in df_detail.columns and "predicted" in df_detail.columns:
        chart_df = df_detail[["datetime", "actual", "predicted"]].copy()
        chart_df = chart_df.set_index("datetime")
        st.line_chart(chart_df, height=300)

    # 表
    show_cols = [
        c for c in ["datetime", "actual", "predicted", "residual",
                    "baseline_pred", "tier_pred_label", "tier_actual_label", "train_n"]
        if c in df_detail.columns
    ]
    st.dataframe(df_detail[show_cols], use_container_width=True, hide_index=True)

    # LLM の reasoning も
    if "reasoning" in df_detail.columns:
        with st.expander("💭 各予測の LLM reasoning"):
            for _, row in df_detail.iterrows():
                dt = row.get("datetime")
                act = row.get("actual")
                pred = row.get("predicted")
                st.markdown(
                    f"**{dt}** — 実測 {act} / 予測 {pred}  \n{row.get('reasoning', '')}"
                )

# ── 新規 backtest 実行 ─────────────────────────────────────
st.divider()
st.subheader("🚀 新規 backtest 実行")
st.caption(
    "**注**: 統計モデルは数十秒〜数分。LLM は 1 魚種 100-300 API call で 数 100k tokens 消費 + 10-30 分。"
)

tab_stat, tab_llm = st.tabs(["統計モデル", "LLM"])

with tab_stat:
    species_list = list_species(min_trips=10)
    sp_stat = st.multiselect(
        "対象魚種（複数可）",
        species_list,
        default=[s for s in ["マダイ", "ホウボウ", "ワラサ", "トラフグ", "ブリ"] if s in species_list],
        key="stat_sp",
    )
    min_train_stat = st.number_input("min_train", 3, 50, 10, key="stat_min")
    if st.button("📊 統計モデル backtest 実行", key="run_stat"):
        if not sp_stat:
            st.warning("魚種を1つ以上選択してください。")
        else:
            with st.spinner("学習 + 予測 中..."):
                progress = st.progress(0.0)
                results = []
                for i, sp in enumerate(sp_stat, 1):
                    try:
                        r = backtest.run_for_species(
                            sp, min_train=int(min_train_stat),
                            save_csv=True, save_plots=False,
                        )
                        results.append(r)
                    except Exception as e:
                        results.append({"species": sp, "error": str(e)})
                    progress.progress(i / len(sp_stat), text=f"[{i}/{len(sp_stat)}] {sp}")
                progress.empty()
            st.success(f"✅ 完了: {len(results)} 魚種")
            st.json(results)
            st.cache_data.clear()
            st.button("🔄 ページを再読込", on_click=lambda: st.rerun())

with tab_llm:
    species_list = list_species(min_trips=10)
    sp_llm = st.selectbox(
        "対象魚種（quota の関係で 1 つずつ）",
        species_list,
        index=species_list.index("マダイ") if "マダイ" in species_list else 0,
        key="llm_sp",
    )
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        # 船宿は registry から
        from utils import list_boats
        boats = list_boats(min_rows=20)
        boat_llm = st.selectbox("対象船宿", boats, key="llm_boat")
        # site も自動解決
        from utils import boat_to_site
        site_llm = boat_to_site(boat_llm) or "morozaki"
        st.caption(f"site: **{site_llm}** (registry 由来)")
    with col_l2:
        providers = available_providers()
        provider_llm = st.selectbox("provider", providers, key="llm_prov")
        min_train_llm = st.number_input("min_train", 3, 50, 10, key="llm_min")

    st.warning(
        f"**quota 注意**: {sp_llm} の trip 数 - min_train 個の API call が発生。"
        "本文 2-3k tokens × N 回 = 数百k tokens 消費見込み。"
    )

    if st.button("🤖 LLM backtest 実行", key="run_llm"):
        with st.spinner(f"{provider_llm} で walk-forward 中... (数十分)"):
            try:
                r = backtest.run_llm_for_species(
                    sp_llm,
                    site=site_llm,
                    boat=boat_llm,
                    min_train=int(min_train_llm),
                    provider=provider_llm,
                    save_csv=True,
                    save_plots=False,
                )
                st.success("✅ LLM backtest 完了")
                st.json(r)
                st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ 失敗: {e}")
                import traceback
                with st.expander("traceback"):
                    st.code(traceback.format_exc())
