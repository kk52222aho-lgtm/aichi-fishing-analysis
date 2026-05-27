"""単発予測ページ — 船宿×魚種×日付×仕掛け×人数 → LLM 予測。"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from utils import (
    available_providers,
    boat_to_site,
    confidence_badge,
    default_provider,
    list_boats,
    list_sites,
    list_species,
    safe_predict,
    tier_emoji,
)

st.set_page_config(page_title="単発予測", page_icon="🎣", layout="wide")
st.title("🎣 単発予測")
st.caption("船宿 × 魚種 × 日付 × 仕掛け × 人数 を選ぶと、LLM が過去ログ + 気象 + 潮汐 + 月齢を読んで予測します。")

# ── 入力フォーム ─────────────────────────────────────────
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        boats = list_boats(min_rows=1)
        if not boats:
            st.error("catches.csv に船宿データがありません。先に取り込みを実行してください。")
            st.stop()
        boat = st.selectbox("船宿", boats, index=0)
        # site は船宿から逆引き、無ければ手動選択にフォールバック
        derived_site = boat_to_site(boat)
        if derived_site:
            site = derived_site
            sites_dict = dict(list_sites())
            st.caption(f"site: **{sites_dict.get(site, site)}** (registry 由来)")
        else:
            site = st.selectbox(
                "site (registry 未登録)",
                [code for code, _ in list_sites()],
                format_func=lambda c: dict(list_sites()).get(c, c),
            )

    with col2:
        species_list = list_species(min_trips=3)
        if not species_list:
            st.error("catches.csv に魚種データがありません。")
            st.stop()
        default_idx = species_list.index("マダイ") if "マダイ" in species_list else 0
        species = st.selectbox("魚種", species_list, index=default_idx)
        target_date = st.date_input("対象日", value=date.today() + timedelta(days=1))

    with col3:
        hour = st.number_input("出船時刻 (hour)", min_value=0, max_value=23, value=6)
        anglers = st.number_input("乗船人数", min_value=1, max_value=20, value=6)
        tackle = st.text_input("仕掛け (任意)", value="タイラバ" if species == "マダイ" else "")

    st.divider()
    col_prov, col_btn = st.columns([2, 1])
    with col_prov:
        providers = available_providers()
        provider = st.selectbox(
            "LLM provider",
            providers,
            index=0,
            help="先頭が最も TPM が広いプロバイダ。rate limit 時は自動で次の候補に切替されます。",
        )
    with col_btn:
        st.write("")  # spacer
        st.write("")
        submitted = st.form_submit_button("🔮 予測する", use_container_width=True)

# ── 予測実行 ─────────────────────────────────────────────
if submitted:
    with st.spinner(f"{provider} に問い合わせ中... (5-15秒)"):
        try:
            result = safe_predict(
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
        except Exception as e:
            st.error(f"予測失敗: {e}")
            st.stop()

    pred = result.get("prediction", {})
    ctx = result.get("boat_context", {})
    cond = result.get("conditions", {})
    model_info = result.get("model", {})

    # ── メイン結果 ─────────────────────────────────────
    st.divider()
    st.subheader(f"📋 予測結果: {boat} × {species} × {target_date}")

    # backtest 知見: tier が本命、絶対値は参考
    tier = pred.get("tier", "?")
    tier_label = pred.get("tier_label", "?")
    st.markdown(
        f"## {tier_emoji(tier)} tier {tier} — **{tier_label}**",
    )
    st.caption(
        "★ backtest 知見: tier ±1 一致は **55-65%** で実用線。"
        " 絶対値（尾数）は参考値（マダイ MAE 16 尾、本質的にばらつき大）。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "予測尾数（参考）",
            f"{pred.get('predicted_top_per_angler', '?')} 尾",
        )
    with c2:
        st.metric("confidence", confidence_badge(pred.get("confidence", "?")))
    with c3:
        signal = pred.get("signal_used") or ctx.get("primary_signal") or "?"
        st.metric("signal_used", signal)

    # ── reasoning ─────────────────────────────────────
    st.markdown("### 💭 reasoning")
    st.info(pred.get("reasoning", "(reasoning なし)"))

    # ── key/risk factors ──────────────────────────────
    col_k, col_r = st.columns(2)
    with col_k:
        st.markdown("### ✅ key factors")
        kf = pred.get("key_factors", [])
        if kf:
            for f in kf:
                st.markdown(f"- {f}")
        else:
            st.caption("(なし)")
    with col_r:
        st.markdown("### ⚠️ risk factors")
        rf = pred.get("risk_factors", [])
        if rf:
            for f in rf:
                st.markdown(f"- {f}")
        else:
            st.caption("(なし)")

    # ── 相対指標 ───────────────────────────────────────
    vs_avg = pred.get("vs_boat_avg")
    vs_med = pred.get("vs_boat_median")
    if vs_avg is not None or vs_med is not None:
        st.markdown("### 📊 自船比")
        cc1, cc2 = st.columns(2)
        with cc1:
            if vs_avg is not None:
                st.metric("vs 自船平均", f"{vs_avg:.2f}x")
        with cc2:
            if vs_med is not None:
                st.metric("vs 自船中央値", f"{vs_med:.2f}x")

    # ── 詳細情報（expander） ──────────────────────────
    with st.expander("🔍 boat_context (過去統計)"):
        st.json(ctx)
    with st.expander("☁️ 当日コンディション"):
        st.json(cond)
    with st.expander("🤖 モデル情報"):
        st.json(model_info)
