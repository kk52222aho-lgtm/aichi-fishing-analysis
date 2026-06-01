"""過去 trip データで walk-forward backtest を実行する。

「次の予測がどれだけ当たるか」を時系列順に検証:
    trip i の予測には trip[0..i-1] までしか使わない（未来のデータを覗かない）
    各 trip で予測値・実績値・tier の予測/実績を記録
    全 trip 終わったら MAE / correlation / tier 一致率 を集計

予測ラベルは `top_per_angler`（個人最大釣果）。

使い方:
    from src.backtest import walk_forward, summarize
    res = walk_forward("イサキ", min_train=5)
    print(summarize(res, "イサキ"))

    # 可視化
    from src.visualizer import plot_backtest_timeseries, plot_predicted_vs_actual
    plot_backtest_timeseries(res, "イサキ", out="data/integrated/イサキ_backtest_ts.png")
    plot_predicted_vs_actual(res, "イサキ", out="data/integrated/イサキ_backtest_scatter.png")

CLI:
    python -m src.backtest --species イサキ --min-train 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config, features, predictor


def _quantile_cutoffs(values: np.ndarray) -> list[float]:
    if len(values) < 5:
        return []
    return [float(x) for x in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]


def walk_forward(
    species: str,
    label_column: str = predictor.DEFAULT_LABEL_COLUMN,
    min_train: int = 5,
    integrated_path: Path | str | None = None,
) -> pd.DataFrame:
    """trip 単位の walk-forward 予測を実行。

    Args:
        species: 対象魚種
        label_column: 予測対象列（default top_per_angler）
        min_train: 最小学習サイズ。これより少ない trip では予測しない
        integrated_path: 統合データ parquet/csv

    Returns:
        DataFrame: datetime, actual, predicted, residual, tier_pred, tier_actual,
                   tier_pred_label, tier_actual_label, train_n
    """
    integrated_path = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    if not integrated_path.exists():
        raise FileNotFoundError(f"統合データが見つかりません: {integrated_path}")

    df = pd.read_parquet(integrated_path) if integrated_path.suffix == ".parquet" else pd.read_csv(integrated_path)
    sub = df[df["species"] == species].copy()
    if label_column not in sub.columns:
        raise ValueError(f"label_column '{label_column}' が無い")
    sub = sub.dropna(subset=[label_column]).copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"])
    sub = sub.sort_values("datetime").reset_index(drop=True)

    if len(sub) < min_train + 1:
        raise ValueError(f"trip 数が足りません: {len(sub)} < {min_train + 1}")

    from . import derived_features as _df_mod

    rows: list[dict] = []
    for i in range(min_train, len(sub)):
        train_df = sub.iloc[:i]
        test_df = sub.iloc[[i]]

        # Step 4: 同魚種 × 同月±1 の過去 trip 統計を特徴量に追加
        # train 用: train_df 自身を history（各 row は自分より前の row だけ参照）
        # test  用: train_df を history（test 行は train 全体を参照可、未来漏洩なし）
        train_with = _df_mod.add_similar_past_features(
            train_df, history_df=train_df, label_col=label_column,
        )
        test_with = _df_mod.add_similar_past_features(
            test_df, history_df=train_df, label_col=label_column,
        )

        X_train = features.build_features(train_with)
        y_train = train_df[label_column].astype(float).values
        X_test = features.build_features(test_with)

        # カラム合わせ（test 側に無い列は 0 埋め）
        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0.0
        X_test = X_test[X_train.columns]

        model = predictor._make_estimator(n_train=len(X_train))
        model.fit(X_train, y_train)
        yhat = float(model.predict(X_test)[0])
        yhat = max(0.0, yhat)
        actual = float(test_df[label_column].iloc[0])

        cutoffs = _quantile_cutoffs(y_train)
        tier_pred, tier_pred_label = predictor._value_to_tier(yhat, cutoffs)
        tier_actual, tier_actual_label = predictor._value_to_tier(actual, cutoffs)

        # Naive baseline: 学習データの平均をそのまま予測値とする
        baseline_pred = float(np.mean(y_train))

        rows.append({
            "datetime": test_df["datetime"].iloc[0],
            "actual": actual,
            "predicted": round(yhat, 2),
            "residual": round(yhat - actual, 2),
            "baseline_pred": round(baseline_pred, 2),
            "baseline_residual": round(baseline_pred - actual, 2),
            "tier_pred": tier_pred,
            "tier_pred_label": tier_pred_label,
            "tier_actual": tier_actual,
            "tier_actual_label": tier_actual_label,
            "train_n": i,
        })

    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame, species: str) -> dict:
    """MAE / correlation / tier 一致率 を集計。"""
    if result.empty:
        return {"species": species, "n_predictions": 0}

    abs_err = np.abs(result["residual"])
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((result["residual"] ** 2).mean()))
    bias = float(result["residual"].mean())  # 系統誤差（+ なら過大、- なら過小）
    corr = float(result[["actual", "predicted"]].corr().iloc[0, 1])

    tier_exact = float((result["tier_pred"] == result["tier_actual"]).mean())
    tier_within1 = float(
        (np.abs(result["tier_pred"] - result["tier_actual"]) <= 1).mean()
    )

    # 平均値だけ予測する naive baseline との比較
    baseline_mae = float(np.abs(result["baseline_residual"]).mean()) if "baseline_residual" in result.columns else None
    improvement = (
        round((baseline_mae - mae) / baseline_mae, 3)
        if baseline_mae and baseline_mae > 0 else None
    )

    return {
        "species": species,
        "n_predictions": int(len(result)),
        "actual_mean": float(result["actual"].mean()),
        "actual_std": float(result["actual"].std()),
        "model_mae": round(mae, 2),
        "baseline_mae": round(baseline_mae, 2) if baseline_mae else None,
        "vs_baseline_pct": improvement,  # + なら model 改善、- なら model 悪化
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "correlation": round(corr, 3),
        "tier_exact_match": round(tier_exact, 3),
        "tier_within_1": round(tier_within1, 3),
    }


def run_for_species(
    species: str,
    min_train: int = 5,
    save_csv: bool = True,
    save_plots: bool = True,
) -> dict:
    """1 魚種の backtest + 集計 + プロット出力 をまとめて。"""
    result = walk_forward(species, min_train=min_train)
    summary = summarize(result, species)

    if save_csv:
        out_csv = config.INTEGRATED_DIR / f"backtest_{species}.csv"
        result.to_csv(out_csv, index=False)
        summary["csv_path"] = str(out_csv)

    if save_plots:
        try:
            from . import visualizer
            ts_out = config.INTEGRATED_DIR / f"backtest_{species}_timeseries.png"
            sc_out = config.INTEGRATED_DIR / f"backtest_{species}_scatter.png"
            visualizer.plot_backtest_timeseries(result, species, out=ts_out)
            visualizer.plot_predicted_vs_actual(result, species, out=sc_out)
            summary["plots"] = [str(ts_out), str(sc_out)]
        except Exception as e:
            summary["plot_error"] = str(e)

    return summary


def walk_forward_llm(
    species: str,
    site: str = "irago",
    boat: Optional[str] = "maruman2010",
    label_column: str = predictor.DEFAULT_LABEL_COLUMN,
    min_train: int = 5,
    integrated_path: Path | str | None = None,
    catches_path: Path | str | None = None,
    provider: str = "gemini",
    model: Optional[str] = None,
    sleep_sec: float = 0.5,
    use_cache: bool = True,
) -> pd.DataFrame:
    """LLM 推論で walk-forward backtest。

    各 trip i について:
      - その日より前の catches.csv 行だけを LLM への過去統計 input にする
      - LLM が予測 → 実績と比較

    遅延 import: 通常の backtest 実行時に LLM の依存を要求しない。
    """
    from . import llm_predictor

    integrated_path = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    catches_path = Path(catches_path) if catches_path else config.FISHING_DIR / "catches.csv"

    def _to_naive(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, errors="coerce")
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
        return s

    df = pd.read_parquet(integrated_path) if integrated_path.suffix == ".parquet" else pd.read_csv(integrated_path)
    sub = df[df["species"] == species].copy()
    if label_column not in sub.columns:
        raise ValueError(f"label_column '{label_column}' が無い")
    sub = sub.dropna(subset=[label_column]).copy()
    sub["datetime"] = _to_naive(sub["datetime"])
    sub = sub.sort_values("datetime").reset_index(drop=True)

    if len(sub) < min_train + 1:
        raise ValueError(f"trip 数が足りません: {len(sub)} < {min_train + 1}")

    # catches.csv も読んで時系列にソート（LLM への過去統計用）
    # catches.csv は +09:00 付きなので tz を剥がして integrated 側と揃える
    full_catches = pd.read_csv(catches_path)
    full_catches["datetime"] = _to_naive(full_catches["datetime"])

    import time
    rows: list[dict] = []
    for i in range(min_train, len(sub)):
        test_row = sub.iloc[i]
        target_dt = pd.Timestamp(test_row["datetime"])
        actual = float(test_row[label_column])

        # 過去 catches を時系列で打ち切り、一時 CSV を作って LLM に渡す
        past = full_catches[full_catches["datetime"] < target_dt]
        tmp_csv = config.DATA_DIR / "predictions" / f"_tmp_past_{species}_{i}.csv"
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)
        past.to_csv(tmp_csv, index=False)

        # 429 / queue_exceeded / rate limit に対して exponential backoff + 別 provider fallback
        # 順序: provider → groq (primary が groq 以外なら) → cerebras (primary が cerebras 以外なら)
        primary = provider
        fallback_chain = [primary]
        for alt in ("groq", "cerebras"):
            if alt != primary and llm_predictor._get_api_key(alt):
                fallback_chain.append(alt)

        last_err = None
        res = None
        used_provider = primary
        for prov in fallback_chain:
            backoff = 5
            for attempt in range(4):  # 4 attempts: 5s, 10s, 20s, 40s waits
                try:
                    res = llm_predictor.predict_with_llm(
                        site=site,
                        species=species,
                        target_date=target_dt.date(),
                        hour=int(test_row.get("departure_hour") or 5),
                        boat=boat,
                        anglers=int(test_row["anglers"]) if pd.notna(test_row.get("anglers")) else None,
                        target_species=test_row.get("target_species"),
                        tackle=test_row.get("tackle") if pd.notna(test_row.get("tackle")) else None,
                        provider=prov,
                        model=model if prov == primary else None,
                        use_cache=use_cache,
                        catches_path=tmp_csv,
                    )
                    used_provider = prov
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    is_429 = "429" in msg or "queue" in msg or "rate" in msg or "resource_exhausted" in msg
                    if not is_429:
                        break  # 非 429 はリトライしない
                    if attempt < 3:
                        print(f"  ⏳ trip {i} {prov} 429, retry in {backoff}s "
                              f"(attempt {attempt+1}/4)")
                        time.sleep(backoff)
                        backoff *= 2
            if res is not None:
                if used_provider != primary:
                    print(f"  ↪ trip {i}: {primary} → {used_provider} にフォールバック成功")
                break

        if res is not None:
            yhat = float(res["prediction"].get("predicted_top_per_angler", 0.0))
            tier_pred = int(res["prediction"].get("tier", 3))
            tier_pred_label = res["prediction"].get("tier_label", "")
            reasoning = res["prediction"].get("reasoning", "")
        else:
            e = last_err
            used_provider = "fallback"
            print(f"  ⚠️ LLM error trip {i} (全プロバイダ失敗): {e}")
            # fallback: 過去全 trip mean ではなく直近 3 trip の median（変化に追従）
            past_sp = past[past["species"] == species]["top_per_angler"].dropna().astype(float)
            if len(past_sp) >= 3:
                yhat = float(past_sp.tail(3).median())
            elif len(past_sp) > 0:
                yhat = float(past_sp.median())
            else:
                yhat = 0.0
            tier_pred = 3
            tier_pred_label = "普通"
            reasoning = f"(LLM error: {e}; fallback=recent3 median)"

        # tier_actual は過去 trip 分布で判定
        past_vals = past[past["species"] == species]["top_per_angler"].dropna().astype(float).values
        cutoffs = _quantile_cutoffs(past_vals)
        tier_actual, tier_actual_label = predictor._value_to_tier(actual, cutoffs)

        baseline_pred = float(np.mean(past_vals)) if len(past_vals) else 0.0

        rows.append({
            "datetime": target_dt,
            "actual": actual,
            "predicted": round(yhat, 2),
            "residual": round(yhat - actual, 2),
            "baseline_pred": round(baseline_pred, 2),
            "baseline_residual": round(baseline_pred - actual, 2),
            "tier_pred": tier_pred,
            "tier_pred_label": tier_pred_label,
            "tier_actual": tier_actual,
            "tier_actual_label": tier_actual_label,
            "reasoning": reasoning,
            "used_provider": used_provider,
            "train_n": i,
        })

        try:
            tmp_csv.unlink()
        except Exception:
            pass
        time.sleep(sleep_sec)

    return pd.DataFrame(rows)


def run_llm_for_species(
    species: str,
    site: str = "irago",
    boat: Optional[str] = "maruman2010",
    min_train: int = 5,
    provider: str = "gemini",
    model: Optional[str] = None,
    save_csv: bool = True,
    save_plots: bool = True,
    use_cache: bool = True,
) -> dict:
    """LLM backtest + 集計 + プロット出力 をまとめて。"""
    result = walk_forward_llm(
        species, site=site, boat=boat, min_train=min_train,
        provider=provider, model=model, use_cache=use_cache,
    )
    summary = summarize(result, species)
    summary["predictor"] = "llm"
    summary["provider"] = provider
    summary["model"] = model

    suffix = provider if model is None else f"{provider}_{model}"
    if save_csv:
        out_csv = config.INTEGRATED_DIR / f"backtest_llm_{suffix}_{species}.csv"
        result.to_csv(out_csv, index=False)
        summary["csv_path"] = str(out_csv)

    if save_plots:
        try:
            from . import visualizer
            ts_out = config.INTEGRATED_DIR / f"backtest_llm_{suffix}_{species}_timeseries.png"
            sc_out = config.INTEGRATED_DIR / f"backtest_llm_{suffix}_{species}_scatter.png"
            visualizer.plot_backtest_timeseries(result, f"{species} ({suffix})", out=ts_out)
            visualizer.plot_predicted_vs_actual(result, f"{species} ({suffix})", out=sc_out)
            summary["plots"] = [str(ts_out), str(sc_out)]
        except Exception as e:
            summary["plot_error"] = str(e)

    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="walk-forward backtest")
    parser.add_argument("--species", required=True, action="append",
                        help="対象魚種（複数指定可）")
    parser.add_argument("--min-train", type=int, default=5)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results = []
    for sp in args.species:
        try:
            r = run_for_species(
                sp, min_train=args.min_train,
                save_csv=not args.no_csv, save_plots=not args.no_plots,
            )
            results.append(r)
        except Exception as e:
            results.append({"species": sp, "error": str(e)})

    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()
