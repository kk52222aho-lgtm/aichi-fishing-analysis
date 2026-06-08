"""過去の予測 vs 実績の答え合わせページ — 釣り人向けの友達口調な見せ方。

精度評価ページ (5) は技術指標 (MAE, correlation 等) 中心。こっちは:
- ピタリ / ほぼ / ハズレ の分布
- 1 件ごとの 「予測 N 尾 vs 実績 M 尾」 カード
- 時系列の折れ線

を素直に出す。バックテスト CSV (data/integrated/backtest_*.csv) を読むだけ。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.llm_predictor import backtest_blowout_probability

st.set_page_config(page_title="過去の予測履歴", page_icon="📜", layout="wide")
st.title("📜 過去の予測の答え合わせ")
st.caption(
    "過去の予測がどれくらい当たってたかをチェックできます。"
    " 一覧ではバックテスト結果（trip i の予測には trip[0..i-1] までしか使わない、"
    "**カンニングなしの厳しめ評価**）を表示。"
)

INTEGRATED_DIR: Path = config.INTEGRATED_DIR


# ── CSV 一覧から (魚種, モデル種別) を抽出 ─────────────
def _list_backtests() -> list[dict]:
    out = []
    for p in sorted(INTEGRATED_DIR.glob("backtest_*.csv")):
        name = p.stem
        if "llm_" in name:
            tokens = name.split("_")
            # backtest_llm_<provider>[_<model>]_<species>[_step3 ..]
            kind = f"LLM ({tokens[2] if len(tokens) > 2 else '?'})"
            species = tokens[-1].replace("_step3", "").replace("_step3_1", "")
            if species.startswith("step"):
                continue  # _step3_2 等の中間ファイル
        else:
            kind = "統計モデル"
            species = name.replace("backtest_", "")
        # CSV を軽く検査 (rows 数だけ)
        try:
            df_head = pd.read_csv(p, nrows=1)
            if "actual" not in df_head.columns or "predicted" not in df_head.columns:
                continue
        except Exception:
            continue
        try:
            n = sum(1 for _ in p.open(encoding="utf-8")) - 1
        except Exception:
            n = 0
        out.append({"species": species, "kind": kind, "n": n, "path": p})
    return out


backtests = _list_backtests()
if not backtests:
    st.warning("バックテスト結果が見つかりません。")
    st.stop()

# ── 魚種選択 ─────────────────────────────────────────
species_options = sorted({b["species"] for b in backtests})
default_sp = "マダイ" if "マダイ" in species_options else species_options[0]
selected_species = st.selectbox(
    "魚種",
    species_options,
    index=species_options.index(default_sp),
)

# その魚種の CSV をリスト
matching = [b for b in backtests if b["species"] == selected_species]
if len(matching) > 1:
    kind_options = [b["kind"] for b in matching]
    selected_kind = st.radio(
        "モデル",
        kind_options,
        horizontal=True,
        help="統計モデル = LightGBM (数値特徴量だけ) / LLM = 文脈含めて推論",
    )
    chosen = next(b for b in matching if b["kind"] == selected_kind)
else:
    chosen = matching[0]
    st.caption(f"モデル: **{chosen['kind']}**")

# ── CSV 読み込み + verdict 判定 ─────────────────────
df = pd.read_csv(chosen["path"])
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["abs_err"] = (df["predicted"] - df["actual"]).abs()


def _verdict(err: float) -> str:
    if pd.isna(err):
        return "?"
    if err <= 1:
        return "🎯 ピタリ"
    if err <= 3:
        return "👍 ほぼ当たり"
    if err <= 5:
        return "🤔 まあまあ"
    return "😅 ハズレ"


df["verdict"] = df["abs_err"].apply(_verdict)
df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

# ── サマリ ────────────────────────────────────────
st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
total = len(df)
c1.metric("予測した回数", total)
c2.metric("🎯 ピタリ", (df["verdict"] == "🎯 ピタリ").sum())
c3.metric("👍 ほぼ当たり", (df["verdict"] == "👍 ほぼ当たり").sum())
c4.metric("🤔 まあまあ", (df["verdict"] == "🤔 まあまあ").sum())
c5.metric("😅 ハズレ", (df["verdict"] == "😅 ハズレ").sum())

hit = ((df["verdict"] == "🎯 ピタリ") | (df["verdict"] == "👍 ほぼ当たり")).sum()
hit_rate = hit / total * 100 if total else 0
mae = df["abs_err"].mean()
st.markdown(
    f"### {selected_species} は **{hit}/{total} 回 (誤差 3 尾以内で当たり)** "
    f"= **{hit_rate:.0f}%** の当たり率。平均誤差 **{mae:.1f} 尾**。"
)

# ── 時系列グラフ（予測 vs 実績） ───────────────────
st.subheader("📊 時系列で見る")
chart_df = df[["datetime", "actual", "predicted"]].copy().set_index("datetime")
chart_df.columns = ["実績 (尾)", "予測 (尾)"]
st.line_chart(chart_df, height=300)
st.caption("青 = 予測 / 赤系 = 実績。同じ線に乗ってる時期ほど当たってた時期。")

# ── 1 件ずつカード形式（最大 30 件） ───────────────
st.subheader("📋 1 件ずつ見る (新しい順)")
st.caption(
    "**reasoning** がある場合（LLM）は展開すると、その時 LLM が「なぜそう予測したか」を読めます。"
)

show_count = st.slider("表示件数", min_value=5, max_value=min(100, total), value=min(20, total))
for _, row in df.head(show_count).iterrows():
    dt = row["datetime"]
    actual = row["actual"]
    pred = row["predicted"]
    err = row["abs_err"]
    verdict = row["verdict"]

    title = (
        f"{dt.strftime('%Y-%m-%d') if pd.notna(dt) else '?'}  —  "
        f"予測 **{pred:.1f}** 尾 / 実績 **{actual:.0f}** 尾  →  {verdict} (誤差 {err:.1f})"
    )
    with st.expander(title, expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric("予測", f"{pred:.1f} 尾")
            tp = row.get("tier_pred_label", row.get("tier_pred", "?"))
            st.caption(f"予測ランク: {tp}")
        with cc2:
            st.metric("実績", f"{actual:.0f} 尾")
            ta = row.get("tier_actual_label", row.get("tier_actual", "?"))
            st.caption(f"実績ランク: {ta}")

        # LLM の reasoning があれば
        if "reasoning" in row and pd.notna(row["reasoning"]) and str(row["reasoning"]).strip():
            st.markdown("**💭 LLM の予測理由**")
            st.info(str(row["reasoning"]))

        # baseline (= 過去平均だけで予測) との比較
        if "baseline_pred" in row and pd.notna(row["baseline_pred"]):
            baseline = float(row["baseline_pred"])
            baseline_err = abs(baseline - actual)
            cc3, cc4 = st.columns(2)
            with cc3:
                st.metric("単純平均で予測したら", f"{baseline:.1f} 尾",
                          help="過去全 trip の平均をそのまま予測値にした場合")
            with cc4:
                if err < baseline_err:
                    delta = baseline_err - err
                    st.metric("単純平均より", f"-{delta:.1f} 尾 良い",
                              delta_color="normal")
                else:
                    delta = err - baseline_err
                    st.metric("単純平均より", f"+{delta:.1f} 尾 悪い",
                              delta_color="inverse")

st.divider()


# ============================================================
# 🎣 大漁日確率の検証 (新機能)
# ============================================================
st.subheader("🎣 大漁日確率の答え合わせ")
st.caption(
    "「もし過去の各日に、当時のデータだけで大漁日確率を計算してたら？」をシミュレーション。"
    " **未来データは一切使わず**、その時点で見えていた trip だけで予測を再現します。"
)

with st.spinner("大漁日確率を過去 trip で再計算中..."):
    bo_df = backtest_blowout_probability(selected_species)

if bo_df.empty:
    st.warning(f"{selected_species} の検証データが不足しています (n < 5)。")
else:
    n_total = len(bo_df)
    n_blowout = int(bo_df["actually_blowout"].sum())
    mean_prob = float(bo_df["predicted_prob"].mean())

    # トップサマリ
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("検証対象 trip", n_total)
    sc2.metric("うち実際の大漁日", f"{n_blowout} 回",
               help=f"閾値以上だった日数")
    sc3.metric("実際の大漁日率", f"{n_blowout / n_total * 100:.0f}%")
    sc4.metric("予測の平均確率", f"{mean_prob * 100:.0f}%",
               help="全 trip の予測確率平均。実際の大漁日率と近いほどキャリブレーションが良い")

    # Calibration: 予測確率帯ごとの実際の大漁日率
    st.markdown("**🎯 キャリブレーション**: 予測確率が当てになるか")
    st.caption(
        "「予測 30-40%」と言った日のうち、実際に大漁だったのが 30-40% なら完璧にキャリブレートされてる。"
        " 大きくズレてたら確率が信用できない。"
    )
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
              "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    bo_df["bin"] = pd.cut(bo_df["predicted_prob"], bins=bins, labels=labels, include_lowest=True)
    cal = bo_df.groupby("bin", observed=True).agg(
        件数=("actually_blowout", "size"),
        実際の大漁回数=("actually_blowout", "sum"),
    ).reset_index()
    cal["実際の大漁率"] = (cal["実際の大漁回数"] / cal["件数"] * 100).round(0).astype(int).astype(str) + "%"
    cal.columns = ["予測確率帯", "件数", "実際の大漁回数", "実際の大漁率"]
    st.dataframe(cal, hide_index=True, use_container_width=True)

    # precision/recall at thresholds
    st.markdown("**🚨 アラート発令基準ごとの的中率**")
    st.caption(
        "**precision**: 「大漁日アラート」を出した日のうち、実際に大漁だった割合。"
        "**recall**: 全ての実際の大漁日のうち、アラートを出せた割合。"
    )
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    pr_rows = []
    for t in thresholds:
        alert = bo_df["predicted_prob"] >= t
        tp = int((alert & bo_df["actually_blowout"]).sum())
        fp = int((alert & ~bo_df["actually_blowout"]).sum())
        fn = int((~alert & bo_df["actually_blowout"]).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        pr_rows.append({
            "アラート閾値": f"≥ {int(t * 100)}%",
            "アラート発令回数": tp + fp,
            "うち大漁的中 (TP)": tp,
            "空振り (FP)": fp,
            "見逃し (FN)": fn,
            "precision": f"{precision * 100:.0f}%" if precision is not None else "—",
            "recall": f"{recall * 100:.0f}%" if recall is not None else "—",
        })
    st.dataframe(pd.DataFrame(pr_rows), hide_index=True, use_container_width=True)

    # 個別 trip 一覧 (折りたたみ)
    with st.expander("📋 1 件ずつ見る (新しい順)"):
        bo_show = bo_df.sort_values("datetime", ascending=False).copy()
        bo_show["日付"] = bo_show["datetime"].dt.strftime("%Y-%m-%d")
        bo_show["予測確率"] = (bo_show["predicted_prob"] * 100).round(0).astype(int).astype(str) + "%"
        bo_show["実際の竿頭"] = bo_show["actual"].round(0).astype(int)
        bo_show["閾値"] = bo_show["threshold"].round(0).astype(int)
        bo_show["大漁?"] = bo_show["actually_blowout"].map({True: "🎯 YES", False: "—"})
        st.dataframe(
            bo_show[["日付", "予測確率", "実際の竿頭", "閾値", "大漁?", "n_similar"]],
            hide_index=True, use_container_width=True,
        )

st.divider()
st.caption(
    "📈 もっと細かい指標 (MAE / RMSE / 相関 / ランク一致率) や、自分で新規バックテスト"
    "を回したい場合は **精度評価** ページへ。"
)
