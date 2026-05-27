"""統合データ・予測結果の可視化。日本語ラベル対応。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, font_setup

font_setup.apply()


def plot_monthly_catch(df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    s = (
        df.assign(month=pd.to_datetime(df["datetime"]).dt.to_period("M").astype(str))
        .groupby("month")["count"]
        .sum()
    )
    s.plot(kind="bar", ax=ax, color="#3a7bd5")
    ax.set_title("月別 釣果数（合計匹数）")
    ax.set_xlabel("月")
    ax.set_ylabel("匹数")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_weather_vs_catch(df: pd.DataFrame, weather_col: str = "wave_height", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    sub = df.dropna(subset=[weather_col, "count"])
    label_map = {
        "wave_height": "波高 (m)",
        "wind_speed_10m": "風速 (m/s)",
        "temperature_2m": "気温 (℃)",
        "pressure_msl": "気圧 (hPa)",
        "sea_surface_temperature": "海水面温度 (℃)",
    }
    ax.scatter(sub[weather_col], sub["count"], alpha=0.6, color="#e07a3a")
    ax.set_xlabel(label_map.get(weather_col, weather_col))
    ax.set_ylabel("釣果数（匹）")
    ax.set_title(f"{label_map.get(weather_col, weather_col)} と釣果数の関係")
    ax.grid(alpha=0.3)
    return ax


def plot_species_breakdown(df: pd.DataFrame, top_n: int = 10, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    s = (
        df.dropna(subset=["species"])
        .groupby("species")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    s.iloc[::-1].plot(kind="barh", ax=ax, color="#3aa14b")
    ax.set_title(f"魚種別 釣果数（上位{top_n}）")
    ax.set_xlabel("匹数")
    ax.set_ylabel("魚種")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_prediction_vs_actual(actual: np.ndarray, predicted: np.ndarray, species: str, ax=None):
    """予測 vs 実績 散布図 + y=x の参照線。モデルの妥当性チェック用。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.6, color="#a050d0")
    lo = float(min(np.min(actual), np.min(predicted), 0))
    hi = float(max(np.max(actual), np.max(predicted)))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x")
    mae = float(np.mean(np.abs(predicted - actual)))
    ax.set_xlabel("実績 釣果数（匹）")
    ax.set_ylabel("予測 釣果数（匹）")
    ax.set_title(f"予測 vs 実績 — {species}（MAE={mae:.2f}）")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


_TIER_COLORS = {
    1: "#c0392b",   # 厳しい — 赤
    2: "#e67e22",   # やや渋い — 橙
    3: "#7f8c8d",   # 普通 — 灰
    4: "#2980b9",   # 好調 — 青
    5: "#27ae60",   # 大漁 — 緑
}


def plot_backtest_timeseries(result: pd.DataFrame, species: str,
                             out: Path | str | None = None):
    """walk-forward backtest の予測 vs 実績 を時系列で描画。

    上段: actual と predicted のライン
    下段: 残差（予測-実績）。色は予測 tier。
    """
    if result.empty:
        return None
    dt = pd.to_datetime(result["datetime"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(dt, result["actual"], "o-", color="#c0392b", label="実績", markersize=6, linewidth=1.5)
    ax1.plot(dt, result["predicted"], "s--", color="#2980b9", label="予測", markersize=5, linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("竿頭（個人最大 尾）")
    mae = float(np.mean(np.abs(result["residual"])))
    corr = float(result[["actual", "predicted"]].corr().iloc[0, 1])
    tier_within1 = float((np.abs(result["tier_pred"] - result["tier_actual"]) <= 1).mean())
    ax1.set_title(
        f"{species} — Walk-Forward Backtest "
        f"(N={len(result)}, MAE={mae:.2f}, R={corr:.2f}, tier±1一致={tier_within1:.0%})"
    )
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # 残差バー
    colors = [_TIER_COLORS.get(int(t), "#7f8c8d") for t in result["tier_pred"]]
    ax2.bar(dt, result["residual"], color=colors, alpha=0.7, width=2.0)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("予測 − 実績")
    ax2.set_xlabel("出船日")
    ax2.grid(alpha=0.3, axis="y")

    fig.autofmt_xdate()
    fig.tight_layout()
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out
    return fig


def plot_predicted_vs_actual(result: pd.DataFrame, species: str,
                             out: Path | str | None = None):
    """backtest 結果を散布図で。色は予測 tier、対角線が完全予測。"""
    if result.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 7))

    for tier, color in _TIER_COLORS.items():
        sub = result[result["tier_pred"] == tier]
        if not sub.empty:
            label = f"tier {tier} ({sub['tier_pred_label'].iloc[0]})"
            ax.scatter(sub["actual"], sub["predicted"], color=color,
                       alpha=0.75, s=60, label=label, edgecolors="white", linewidth=0.5)

    lo = 0.0
    hi = float(max(result["actual"].max(), result["predicted"].max())) + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x")
    mae = float(np.mean(np.abs(result["residual"])))
    bias = float(result["residual"].mean())
    ax.set_xlabel("実績 竿頭（尾）")
    ax.set_ylabel("予測 竿頭（尾）")
    ax.set_title(f"{species} — 予測 vs 実績（MAE={mae:.2f}, bias={bias:+.2f}）")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")

    fig.tight_layout()
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out
    return fig


def render_report(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    plot_monthly_catch(df, ax=axes[0])
    plot_species_breakdown(df, ax=axes[1])
    plot_weather_vs_catch(df, "wave_height", ax=axes[2])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="統合データから簡易レポートPNGを生成")
    parser.add_argument("--input", type=Path, default=config.INTEGRATED_DIR / "integrated.parquet")
    parser.add_argument("--out", type=Path, default=config.INTEGRATED_DIR / "report.png")
    args = parser.parse_args()
    df = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    saved = render_report(df, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    _cli()
