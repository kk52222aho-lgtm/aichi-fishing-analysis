"""最新エントリ取り込みトリガー — build_seed_dataset を UI から起動。

注意:
- LLM extract は quota を消費する。short months_back（1-2 ヶ月）推奨
- 同期実行なので Streamlit セッションがブロックされる（プログレスは print 由来でこまめに出る）
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import streamlit as st

from utils import (
    available_providers,
    boat_to_blog_id,
    list_boats,
    load_registry,
)
from src.build_seed_dataset import build  # noqa: E402

st.set_page_config(page_title="最新取り込み", page_icon="🔄", layout="wide")
st.title("🔄 最新エントリ取り込み")
st.caption(
    "船宿のブログから直近のエントリを取り込んで catches.csv を更新します。"
    " 既に scrape 済みのエントリは飛ばすので、初回より高速。"
)

# ── 船宿選択 ─────────────────────────────────────────────
reg = load_registry()
boats_in_reg = [v.get("boat") for v in reg.values()]
boats_with_data = list_boats(min_rows=1)
# registry と catches.csv の両方をマージ
all_boats = sorted(set(boats_in_reg) | set(boats_with_data))
if not all_boats:
    st.error("registry / catches.csv に船宿がありません。")
    st.stop()

with st.form("ingest_form"):
    col1, col2 = st.columns(2)

    with col1:
        boat = st.selectbox("船宿", all_boats, index=0)
        blog_id = boat_to_blog_id(boat)
        if not blog_id:
            st.error(f"{boat} の blog_id が registry に無いため取り込み不可。")
            st.stop()
        st.caption(f"blog_id: **{blog_id}**")

        # registry から site / primary_signal も自動解決
        reg_entry = next(
            (v for v in reg.values() if v.get("boat") == boat), {}
        )
        site = reg_entry.get("site", "morozaki")
        primary_signal = reg_entry.get("primary_signal", "qualitative")
        st.caption(f"site: **{site}** / primary_signal: **{primary_signal}**")

    with col2:
        months_back = st.number_input(
            "何ヶ月分遡るか", min_value=1, max_value=12, value=1,
            help="初回は 6、日常更新は 1 推奨。skip_existing=True で既処理は飛ばす。",
        )
        run_yolo = st.checkbox("YOLO 画像検出も実行", value=True)
        use_llm = st.checkbox("LLM で本文抽出 (quota 消費)", value=True)

    st.divider()
    col_p, col_b = st.columns([2, 1])
    with col_p:
        providers = available_providers()
        provider = st.selectbox("LLM provider", providers, index=0)
        fallback = st.selectbox(
            "fallback provider",
            ["(none)"] + [p for p in providers if p != provider],
            index=1 if len(providers) > 1 else 0,
        )
    with col_b:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("🚀 取り込み開始", use_container_width=True)

# ── 取り込み実行 ─────────────────────────────────────────
if submitted:
    st.divider()
    st.subheader(f"🔄 {boat} ({blog_id}) 取り込み中...")

    log_area = st.empty()
    log_text = ""

    class _StreamlitLogger(io.StringIO):
        """build の print 出力を Streamlit に逐次フラッシュする。"""
        def write(self, s):
            super().write(s)
            nonlocal_holder["log"] += s
            # 末尾 200 行だけ表示（ログ肥大対策）
            tail = "\n".join(nonlocal_holder["log"].splitlines()[-200:])
            log_area.code(tail, language="text")
            return len(s)

    nonlocal_holder = {"log": ""}
    logger = _StreamlitLogger()

    try:
        with redirect_stdout(logger):
            df = build(
                blog_id=blog_id,
                site=site,
                boat=boat,
                months_back=int(months_back),
                use_llm_extract=use_llm,
                llm_provider=provider,
                llm_fallback_provider=None if fallback == "(none)" else fallback,
                primary_signal=primary_signal,
                run_yolo=run_yolo,
            )
        st.success(f"✅ 完了: {boat} = {len(df)} 行")
        st.balloons()
    except Exception as e:
        st.error(f"❌ 失敗: {e}")
        with st.expander("traceback"):
            import traceback
            st.code(traceback.format_exc())

    # 取り込み後はキャッシュをクリアして他ページにも反映
    st.cache_data.clear()
    st.info("ℹ️ load_catches キャッシュをクリアしました。他ページで最新データが見えます。")
