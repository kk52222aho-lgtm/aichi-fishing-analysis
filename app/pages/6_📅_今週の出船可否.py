"""今週の出船可否カレンダー — 次の 7 日間、地点ごとに風速・波高から判定。

LLM 予測は呼ばない（高速・低コスト）。気象データから判定するだけ。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from utils import boat_status, list_sites
from src import config, weather_fetcher

st.set_page_config(page_title="今週の出船可否", page_icon="📅", layout="wide")
st.title("📅 今週の出船可否 — どの日に出れる？")
st.caption(
    "次の 7 日間、地点ごとに **風速・波高・突風・降水** から「出船できそうか」を一目で判定。"
    " LLM 予測は呼びません（コスト 0、ロード 2-3 秒）。"
)


@st.cache_data(ttl=1800)  # 30 分キャッシュ
def get_weekly_outlook(
    site_code: str,
    morning_hours: tuple[int, ...] = (5, 6, 7, 8),
) -> list[dict[str, Any]]:
    """site の次の 7 日間、朝の時間帯平均から boat_status を返す。"""
    site_obj = config.SITES.get(site_code)
    if not site_obj:
        return []
    today = date.today()
    end = today + timedelta(days=7)
    try:
        df = weather_fetcher.fetch_and_cache(site_obj, today, end)
    except Exception as e:
        st.error(f"気象データ取得失敗 ({site_code}): {e}")
        return []
    if df.empty:
        return []

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    df["_date"] = df["time"].dt.date
    df["_hour"] = df["time"].dt.hour

    out: list[dict[str, Any]] = []
    for i in range(7):
        d = today + timedelta(days=i)
        day_df = df[(df["_date"] == d) & (df["_hour"].isin(morning_hours))]
        if day_df.empty:
            continue
        # 風速は m/s 換算（Open-Meteo は km/h）
        cond: dict[str, Any] = {}
        if "wind_speed_10m" in day_df.columns:
            cond["wind_speed_10m"] = round(
                day_df["wind_speed_10m"].mean() / 3.6, 1
            )
        if "wind_gusts_10m" in day_df.columns:
            cond["wind_gusts_10m"] = round(
                day_df["wind_gusts_10m"].mean() / 3.6, 1
            )
        for src_col in ("wave_height", "swell_wave_height"):
            if src_col in day_df.columns and day_df[src_col].notna().any():
                cond[src_col] = round(float(day_df[src_col].mean()), 2)
        if "precipitation" in day_df.columns and day_df["precipitation"].notna().any():
            cond["precipitation"] = round(float(day_df["precipitation"].max()), 1)

        status = boat_status(cond)
        out.append({
            "date": d,
            "weekday": ["月", "火", "水", "木", "金", "土", "日"][d.weekday()],
            "status": status,
            "cond": cond,
        })
    return out


# ── 地点選択 ─────────────────────────────────────────
sites = list_sites()
site_codes = [c for c, _ in sites]
site_labels = {c: name for c, name in sites}

selected_sites = st.multiselect(
    "地点",
    site_codes,
    default=site_codes[:3] if len(site_codes) >= 3 else site_codes,
    format_func=lambda c: site_labels.get(c, c),
    help="地点ごとに 7 日間の出船可否を判定します。複数選択可。",
)

if not selected_sites:
    st.info("地点を 1 つ以上選んでください。")
    st.stop()

st.divider()

# ── 地点ごとに table 表示 ────────────────────────────
for site_code in selected_sites:
    site_name = site_labels.get(site_code, site_code)
    st.subheader(f"📍 {site_name}")

    with st.spinner(f"{site_name} の天気を取得中..."):
        outlook = get_weekly_outlook(site_code)

    if not outlook:
        st.warning("気象データが取得できませんでした。")
        continue

    # テーブル形式で表示
    rows = []
    for day in outlook:
        d = day["date"]
        c = day["cond"]
        s = day["status"]
        rows.append({
            "日付": d.strftime("%m/%d") + f" ({day['weekday']})",
            "判定": f"{s['emoji']} {s['label']}",
            "風速 (m/s)": c.get("wind_speed_10m"),
            "突風 (m/s)": c.get("wind_gusts_10m"),
            "波高 (m)": c.get("wave_height"),
            "降水 (mm)": c.get("precipitation"),
            "理由": " / ".join(s["reasons"]) if s["reasons"] else "—",
        })
    df_outlook = pd.DataFrame(rows)
    st.dataframe(
        df_outlook,
        use_container_width=True,
        hide_index=True,
        column_config={
            "風速 (m/s)": st.column_config.NumberColumn(format="%.1f"),
            "突風 (m/s)": st.column_config.NumberColumn(format="%.1f"),
            "波高 (m)": st.column_config.NumberColumn(format="%.1f"),
            "降水 (mm)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # 推奨日のハイライト
    ok_days = [d for d in outlook if d["status"]["level"] in ("ok", "caution")]
    no_go_days = [d for d in outlook if d["status"]["level"] == "no_go"]
    if ok_days:
        ok_str = "、".join(
            f"{d['date'].strftime('%m/%d')}({d['weekday']})" for d in ok_days[:3]
        )
        st.success(f"☀️ **おすすめ**: {ok_str}")
    if no_go_days:
        no_go_str = "、".join(
            f"{d['date'].strftime('%m/%d')}({d['weekday']})" for d in no_go_days
        )
        st.error(f"⛔ **欠航見込み**: {no_go_str}")

    st.divider()


st.caption(
    "**判定基準 (一般的な釣り船の目安)**:  \n"
    "⛔ 出船困難 = 風速 ≥ 12 m/s / 波高 ≥ 2.5 m / 突風 ≥ 18 m/s  \n"
    "⚠️ 厳しい海況 = 風速 8-12 m/s / 波高 1.5-2.5 m  \n"
    "✅ 出船可能 = 風速 4-8 m/s / 波高 < 1.5 m  \n"
    "☀️ ベタ凪 = 風速 < 4 m/s  \n"
    "**注**: 各船宿の判断基準は異なるため、最終決定は船宿へ要確認。"
)
