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
    signal_label_ja,
    tier_emoji,
)
from src import predictions_log

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
            "LLM プロバイダ",
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

    # ── 予測ログに記録 (出船後にフィードバック紐付け用)
    # 直接 predict_with_llm を呼んでるので engine 情報を補ってから渡す
    log_result = dict(result)
    log_result.setdefault("engine", "llm")
    try:
        pid = predictions_log.log_prediction(log_result)
    except Exception as e:
        pid = None
        st.caption(f"⚠️ 予測ログ保存失敗 (機能は影響なし): {e}")

    # ── メイン結果 ─────────────────────────────────────
    st.divider()
    head_col, cta_col = st.columns([3, 1])
    with head_col:
        st.subheader(f"📋 予測結果: {boat} × {species} × {target_date}")
    with cta_col:
        # 「他の船と比べる」導線（同条件で 4-5 船宿を一気に並べる）
        if st.button(
            "🆚 他の船と比べる",
            help="同じ魚種・日付で他の船宿を一気に予測してランキング表示します",
            use_container_width=True,
        ):
            st.session_state["_compare_species"] = species
            st.session_state["_compare_date"] = target_date
            try:
                st.switch_page("pages/2_🆚_船宿比較.py")
            except Exception:
                st.info("左サイドバーの「船宿比較」ページを開いてください。")

    # backtest 知見: ランク (tier) が本命、絶対値は参考
    tier = pred.get("tier", "?")
    tier_label = pred.get("tier_label", "?")
    st.markdown(
        f"## {tier_emoji(tier)} ランク {tier} — **{tier_label}**",
    )
    st.caption(
        "★ **ランクの見方**: 1 厳しい / 2 やや渋い / 3 普通 / 4 好調 / 5 大漁。"
        " バックテスト実測でランク ±1 以内一致 **55-65%** で実用線。"
        " 絶対値（尾数）は参考（マダイで平均誤差 16 尾、本質的にばらつき大）。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "予測尾数（参考）",
            f"{pred.get('predicted_top_per_angler', '?')} 尾",
        )
    with c2:
        st.metric("信頼度", confidence_badge(pred.get("confidence", "?")))
    with c3:
        signal = pred.get("signal_used") or ctx.get("primary_signal") or "?"
        st.metric("使用シグナル", signal_label_ja(signal))

    # ── 予測根拠 ─────────────────────────────────────
    st.markdown("### 💭 予測の根拠")
    st.info(pred.get("reasoning", "(根拠なし)"))

    # ── プラス要因 / リスク要因 ────────────────────────
    col_k, col_r = st.columns(2)
    with col_k:
        st.markdown("### ✅ プラス要因")
        kf = pred.get("key_factors", [])
        if kf:
            for f in kf:
                st.markdown(f"- {f}")
        else:
            st.caption("(なし)")
    with col_r:
        st.markdown("### ⚠️ リスク要因")
        rf = pred.get("risk_factors", [])
        if rf:
            for f in rf:
                st.markdown(f"- {f}")
        else:
            st.caption("(なし)")

    # ── この船の普段との比較 ─────────────────────────────
    vs_avg = pred.get("vs_boat_avg")
    vs_med = pred.get("vs_boat_median")
    if vs_avg is not None or vs_med is not None:
        st.markdown("### 📊 この船の普段と比べると")
        cc1, cc2 = st.columns(2)
        with cc1:
            if vs_avg is not None:
                st.metric(
                    "普段の平均と比べて",
                    f"{vs_avg:.2f} 倍",
                    help="1.0 が普段並み、1.5 なら 5割増の予測、0.5 なら半分",
                )
        with cc2:
            if vs_med is not None:
                st.metric(
                    "普段の中央値と比べて",
                    f"{vs_med:.2f} 倍",
                )

    # ── 実績フィードバック誘導 ──────────────────────────
    if pid:
        st.divider()
        st.markdown("### 🎯 出船後の結果を教えてください")
        st.caption(
            "実際に何尾釣れたかを教えてくれると、このアプリの精度が改善されます。"
        )

        # その場で素早く実績登録できるショートカット（最初から開いておく）
        with st.expander("⚡ 釣果を入力する", expanded=False):
            quick_actual = st.number_input(
                "実際の竿頭釣果 (個人最大、尾)",
                min_value=0.0, value=0.0, step=1.0,
                key=f"quick_{pid}",
            )
            quick_qual = st.selectbox(
                "定性 (任意)",
                ["", "大漁", "好調", "普通", "渋い", "厳しい", "ボウズ"],
                key=f"quick_q_{pid}",
            )
            quick_notes = st.text_input(
                "メモ (任意)", key=f"quick_n_{pid}",
            )
            if st.button("✅ 実績を記録", key=f"quick_btn_{pid}"):
                try:
                    r = predictions_log.link_feedback(
                        prediction_id=pid,
                        actual_top_per_angler=float(quick_actual),
                        actual_qualitative=quick_qual or None,
                        notes=quick_notes or None,
                    )
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        st.success(
                            f"記録完了。予測 {pred.get('predicted_top_per_angler')} 尾 / "
                            f"実績 {quick_actual} 尾 (誤差 "
                            f"{abs(float(pred.get('predicted_top_per_angler', 0)) - quick_actual):.1f} 尾)"
                        )
                except Exception as e:
                    st.error(f"失敗: {e}")

    # ── 詳細情報（人が読める要約に置換） ──────────────────
    st.divider()
    st.markdown("### 📊 この予測の根拠データ")

    # 過去統計を bullet で
    n_trips_total = ctx.get("n_trips_total") or 0   # 全 trip
    n_trips_w_label = ctx.get("n_trips") or 0       # 竿頭の数値が記録されてる trip
    p_median = ctx.get("median")
    p_max = ctx.get("max")
    p_mean = ctx.get("mean")
    if n_trips_total:
        st.markdown(
            f"**{boat} で {species} を狙ったのは過去 {n_trips_total} 回**:"
        )
        if n_trips_w_label and n_trips_w_label < n_trips_total:
            st.caption(
                f"（うち竿頭の数字が記録されているのは {n_trips_w_label} 回。"
                "過去最大が小さく見える場合、ブログに数字が書かれてない日が多い "
                "船宿である可能性があります）"
            )
        bullets = []
        if p_median is not None:
            bullets.append(f"竿頭の中央値: **{p_median} 尾**（半分の日はこの値以下）")
        if p_mean is not None:
            bullets.append(f"竿頭の平均: **{round(p_mean, 1)} 尾**")
        if p_max is not None:
            bullets.append(
                f"記録された範囲での最大: **{int(p_max)} 尾**"
                + ("" if n_trips_w_label >= 20 else "（データ少なめ、要注意）")
            )
        recent = ctx.get("recent_5_trips") or []
        if recent:
            # NaN / None の top_per_angler は除外
            valid = []
            for r in recent[::-1]:  # 新しい順
                v = r.get("top_per_angler")
                try:
                    fv = float(v)
                    if fv == fv:  # NaN チェック
                        valid.append((r.get("datetime", "")[:10], fv))
                except (TypeError, ValueError):
                    continue
                if len(valid) >= 3:
                    break
            if valid:
                recent_str = ", ".join(
                    f"{d} → {int(v) if v.is_integer() else v} 尾"
                    for d, v in valid
                )
                bullets.append(f"直近の竿頭記録: {recent_str}")
        for b in bullets:
            st.markdown(f"- {b}")
    else:
        st.caption("過去データなし")

    # 当日コンディションを bullet で
    st.markdown("**今日のコンディション**:")
    if cond:
        cond_lines = []
        if "sea_surface_temperature" in cond:
            cond_lines.append(f"🌊 海面水温 **{cond['sea_surface_temperature']}℃**")
        if "wave_height" in cond:
            cond_lines.append(f"🌊 波高 **{cond['wave_height']} m**")
        if "wind_speed_10m" in cond:
            # _readable_conditions が km/h -> m/s 換算済みで返す
            ws_ms = float(cond["wind_speed_10m"])
            cond_lines.append(f"💨 風速 **{ws_ms:.1f} m/s**")
        if "tide_phase" in cond:
            cond_lines.append(f"🌙 潮回り **{cond['tide_phase']}**")
        if "tide_cm" in cond:
            cond_lines.append(f"🌊 潮位 **{cond['tide_cm']} cm**")
        if "moon_phase" in cond:
            cond_lines.append(f"🌙 月齢 **{cond['moon_phase']}**")
        if cond_lines:
            for c in cond_lines:
                st.markdown(f"- {c}")
        else:
            st.caption("コンディション情報なし")
    else:
        st.caption("コンディション情報なし")

    # 開発者向け（普段は折りたたみ）
    with st.expander("🔧 技術情報 (開発者向け)", expanded=False):
        if pid:
            st.code(f"prediction_id = {pid}", language="text")
        st.caption("**過去統計の生データ**")
        st.json(ctx)
        st.caption("**当日コンディションの生データ**")
        st.json(cond)
        st.caption("**使用モデル**")
        st.json(model_info)
