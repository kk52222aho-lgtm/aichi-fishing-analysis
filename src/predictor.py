"""釣果予測モデル — 学習と推論。

学習ラベル: `top_per_angler`（本文「竿頭 N尾」抽出値 = 個人最大釣果）
  理由:
   - 釣り客視点の指標（自分が行ったら何匹釣れるかの上限）
   - 本文抽出で正確に取れる（YOLO ノイズ無し）
   - シマアジなど YOLO クラスに無い魚種でも学習可能
   - anglers の値が無くても比較可能

予測出力は 5 段階の自船相対評価も含む（その船・その魚種の過去データから quintile）:
    1 厳しい / 2 やや渋い / 3 普通 / 4 好調 / 5 大漁

魚種ごとに 1 モデルを学習。LightGBM 優先、無ければ scikit-learn フォールバック。
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from . import config, data_integrator, features

DEFAULT_LABEL_COLUMN = "top_per_angler"

# 5 段階のラベル
TIER_LABELS: tuple[str, ...] = ("厳しい", "やや渋い", "普通", "好調", "大漁")


def _make_estimator():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, random_state=42, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=42,
        )


def _model_path(species: str) -> Path:
    safe = species.replace("/", "_").replace("\\", "_")
    return config.MODELS_DIR / f"{safe}.joblib"


def _compute_boat_stats(values: np.ndarray) -> dict[str, Any]:
    """学習データのラベル値分布から自船相対評価用の統計を計算。"""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}
    q = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.5)),
        "p75": float(np.quantile(values, 0.75)),
        "tier_cutoffs": [float(x) for x in q],  # [p20, p40, p60, p80]
    }


def _value_to_tier(value: float, cutoffs: list[float]) -> tuple[int, str]:
    """値を 1-5 の tier と日本語ラベルに変換。"""
    if not cutoffs or len(cutoffs) != 4:
        return 3, TIER_LABELS[2]
    # right=True: v <= c1 → bin 0、c1 < v <= c2 → bin 1、…、v > c4 → bin 4
    idx = int(np.digitize(value, cutoffs, right=True))
    idx = max(0, min(idx, 4))
    return idx + 1, TIER_LABELS[idx]


def train(
    species: str,
    integrated_path: Path | str | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    min_samples: int = 10,
) -> dict[str, Any]:
    """指定魚種のモデルを学習して保存。

    Args:
        species: 対象魚種（catches.csv の species 列の値）
        integrated_path: 統合データ parquet/csv のパス
        label_column: 学習ラベル列。default "top_per_angler"
        min_samples: 学習に必要な最小行数
    """
    integrated_path = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    if not integrated_path.exists():
        raise FileNotFoundError(f"統合データが見つかりません: {integrated_path}")

    df = pd.read_parquet(integrated_path) if integrated_path.suffix == ".parquet" else pd.read_csv(integrated_path)
    sub = df[df["species"] == species].copy()
    if label_column not in sub.columns:
        raise ValueError(f"label_column '{label_column}' が統合データに存在しません。columns={list(sub.columns)}")
    sub = sub.dropna(subset=[label_column])  # ラベル欠損行は学習に使わない
    if len(sub) < min_samples:
        raise ValueError(
            f"学習データ不足: species={species}, label={label_column} で {len(sub)} 行 "
            f"(最低 {min_samples} 行必要)"
        )

    X = features.build_features(sub)
    y = sub[label_column].astype(float).values

    n = len(sub)
    n_test = max(1, n // 5)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    model = _make_estimator()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))

    boat_stats = _compute_boat_stats(y)  # 全学習データから（テスト除外しない方が安定）

    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "species": species,
        "label_column": label_column,
        "trained_at": datetime.now().isoformat(),
        "n_samples": int(n),
        "boat_stats": boat_stats,
        "metrics": {"mae_holdout": mae},
    }
    out = _model_path(species)
    joblib.dump(bundle, out)
    return {
        "species": species,
        "label_column": label_column,
        "n_samples": n,
        "mae_holdout": mae,
        "boat_stats": boat_stats,
        "model_path": str(out),
    }


def _load(species: str) -> dict[str, Any]:
    p = _model_path(species)
    if not p.exists():
        raise FileNotFoundError(f"モデル未学習: {species} ({p})")
    return joblib.load(p)


def predict(
    site: str,
    species: str,
    target_date: str | date,
    hour: int = 6,
    boat: Optional[str] = None,
    anglers: Optional[int] = None,
    target_species: Optional[str] = None,
    tackle: Optional[str] = None,
    departure_hour: Optional[int] = None,
) -> dict[str, Any]:
    """出船前情報のみから釣果を予測。

    返り値は 3軸構造:
        ① 絶対値（top_per_angler の生予測）
        ② 自船評価（mean/median 比 + 5段階 tier）
        ③ 他船相対（複数船宿揃ったら）
    """
    bundle = _load(species)

    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    target_dt = pd.Timestamp(datetime.combine(target_date, datetime.min.time()).replace(hour=hour))

    row = data_integrator.build_inference_row(
        site=site,
        target_dt=target_dt,
        species=species,
        boat=boat,
        anglers=anglers,
        target_species=target_species,
        tackle=tackle,
        departure_hour=departure_hour if departure_hour is not None else hour,
    )

    X = features.build_features(row)
    cols = bundle["feature_columns"]
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols]
    yhat = float(bundle["model"].predict(X)[0])
    yhat = max(0.0, yhat)

    bstats = bundle.get("boat_stats") or {}
    cutoffs = bstats.get("tier_cutoffs", [])
    tier_num, tier_label = _value_to_tier(yhat, cutoffs)

    vs_avg = (yhat / bstats["mean"]) if bstats.get("mean") else None
    vs_median = (yhat / bstats["median"]) if bstats.get("median") else None
    label_column = bundle.get("label_column", DEFAULT_LABEL_COLUMN)

    return {
        "site": site,
        "site_name_ja": config.SITES[site].name_ja if site in config.SITES else site,
        "species": species,
        "date": str(target_date),
        "hour": hour,
        "boat": boat,
        "anglers": anglers,
        "tackle": tackle,
        "prediction": {
            # ① 絶対値
            label_column: round(yhat, 2),
            # ② 自船相対
            "vs_boat_avg": round(vs_avg, 3) if vs_avg is not None else None,
            "vs_boat_median": round(vs_median, 3) if vs_median is not None else None,
            "tier": tier_num,
            "tier_label": tier_label,
            # ③ 他船相対（将来用）
            "vs_other_boats": None,
        },
        "boat_context": bstats,
        "model": {
            "label_column": label_column,
            "mae_holdout": bundle.get("metrics", {}).get("mae_holdout"),
            "n_samples_train": bundle.get("n_samples"),
            "trained_at": bundle.get("trained_at"),
        },
    }


def list_models() -> list[dict[str, Any]]:
    out = []
    for p in sorted(config.MODELS_DIR.glob("*.joblib")):
        try:
            b = joblib.load(p)
            bstats = b.get("boat_stats") or {}
            out.append({
                "species": b.get("species", p.stem),
                "label_column": b.get("label_column"),
                "trained_at": b.get("trained_at"),
                "n_samples": b.get("n_samples"),
                "metrics": b.get("metrics"),
                "boat_stats_summary": {
                    "n": bstats.get("n"),
                    "median": bstats.get("median"),
                    "tier_cutoffs": bstats.get("tier_cutoffs"),
                },
                "path": str(p),
            })
        except Exception as e:
            out.append({"species": p.stem, "error": str(e), "path": str(p)})
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="釣果予測モデル（ラベル= top_per_angler、5段階出力）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--species", required=True)
    p_train.add_argument("--data", type=Path, default=None)
    p_train.add_argument("--label", default=DEFAULT_LABEL_COLUMN,
                         help=f"学習ラベル列 (default {DEFAULT_LABEL_COLUMN})")
    p_train.add_argument("--min-samples", type=int, default=10)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--site", required=True, choices=list(config.SITES))
    p_pred.add_argument("--species", required=True)
    p_pred.add_argument("--date", required=True)
    p_pred.add_argument("--hour", type=int, default=6)
    p_pred.add_argument("--boat", default=None)
    p_pred.add_argument("--anglers", type=int, default=None)
    p_pred.add_argument("--tackle", default=None)
    p_pred.add_argument("--target-species", default=None)

    sub.add_parser("list")

    args = parser.parse_args()
    if args.cmd == "train":
        result = train(
            args.species, args.data,
            label_column=args.label, min_samples=args.min_samples,
        )
    elif args.cmd == "predict":
        result = predict(
            args.site, args.species, args.date,
            hour=args.hour, boat=args.boat, anglers=args.anglers,
            tackle=args.tackle, target_species=args.target_species,
        )
    else:
        result = list_models()

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()
