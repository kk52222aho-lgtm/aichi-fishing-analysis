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
import warnings
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

# 本文の定性的評価 → 5 段階 tier。数値ラベル(top_per_angler)が無い行を
# ティア分類モードの学習に取り込むための写像。表記ゆれも吸収する。
QUALITATIVE_TO_TIER: dict[str, int] = {
    "絶好調": 5,
    "大漁": 5,
    "爆釣": 5,
    "好調": 4,
    "普通": 3,
    "まずまず": 3,
    "イマイチ": 2,
    "やや渋い": 2,
    "渋い": 2,
    "厳しい": 1,
    "難しい": 1,
    "ボウズ": 1,
    "ボーズ": 1,
}


def _make_estimator(n_train: Optional[int] = None):
    """学習サンプル数に応じてモデル容量を調整。

    n_train が None または >= 20 ならフル設定。極小データ (<20) のみ num_leaves と
    n_estimators を縮め、reg_alpha/lambda を加えて過学習を抑制。
    閾値 20 の根拠: 11 魚種 backtest で n_train>=20 はマダイなど分布が広い魚種が
    多く、容量不足が悪化要因になるため。
    """
    small = n_train is not None and n_train < 20
    try:
        from lightgbm import LGBMRegressor
        if small:
            return LGBMRegressor(
                n_estimators=120, learning_rate=0.05, num_leaves=8,
                min_child_samples=3, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1,
            )
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, random_state=42, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        if small:
            return HistGradientBoostingRegressor(
                max_iter=120, learning_rate=0.05, max_depth=4,
                min_samples_leaf=3, l2_regularization=0.1, random_state=42,
            )
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=42,
        )


def _make_classifier(n_train: Optional[int] = None):
    """ティア分類用の推定器。学習数が少ないうちは容量を絞り正則化を効かせる。

    閾値 60: tier 5 クラス × 推奨 10件/クラス を一応の目安に、それ未満は小データ扱い。
    class_weight="balanced" で偏った tier 分布（好調が多い）を補正する。
    """
    small = n_train is not None and n_train < 60
    try:
        from lightgbm import LGBMClassifier
        if small:
            return LGBMClassifier(
                n_estimators=150, learning_rate=0.05, num_leaves=8,
                min_child_samples=3, reg_alpha=0.1, reg_lambda=0.1,
                class_weight="balanced", random_state=42, verbose=-1,
            )
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, class_weight="balanced",
            random_state=42, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        if small:
            return HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4,
                min_samples_leaf=3, l2_regularization=0.1, random_state=42,
            )
        return HistGradientBoostingClassifier(
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


def _build_tier_labels(
    sub: pd.DataFrame, label_column: str = DEFAULT_LABEL_COLUMN,
) -> tuple[pd.Series, list[float], dict[str, int]]:
    """各行を 1-5 tier に変換する。

    優先順位:
      1. 数値ラベル(top_per_angler)があれば、その魚種内の分位 [p20,p40,p60,p80]
         で 5 段階に割り当て（客観的）。
      2. 数値が無く質的評価(qualitative)があれば QUALITATIVE_TO_TIER で割り当て。
      3. どちらも無い行は NaN（学習から除外）。

    Returns:
        (tier: 1-5 の float Series。未確定は NaN, cutoffs: 数値用カットオフ,
         src: {"from_numeric": n, "from_qualitative": n})
    """
    tier = pd.Series(np.nan, index=sub.index, dtype=float)

    cutoffs: list[float] = []
    if label_column in sub.columns:
        num = pd.to_numeric(sub[label_column], errors="coerce")
        num_valid = num.dropna()
        if len(num_valid) >= 5:  # 分位を安定して取れる最低数
            cutoffs = _compute_boat_stats(num_valid.values).get("tier_cutoffs", [])
            if len(cutoffs) == 4:
                idx = np.clip(np.digitize(num_valid.values, cutoffs, right=True), 0, 4) + 1
                tier.loc[num_valid.index] = idx.astype(float)
    n_from_num = int(tier.notna().sum())

    if "qualitative" in sub.columns:
        mask = tier.isna()
        mapped = sub.loc[mask, "qualitative"].map(QUALITATIVE_TO_TIER)
        tier.loc[mask] = mapped.astype(float)
    n_from_qual = int(tier.notna().sum()) - n_from_num

    return tier, cutoffs, {"from_numeric": n_from_num, "from_qualitative": n_from_qual}


def _train_tier(
    species: str, sub: pd.DataFrame, label_column: str, min_samples: int,
) -> dict[str, Any]:
    """ティア分類モデル(5段階)を学習して保存。

    数値ラベルが無く質的評価しかない行も取り込むため、回帰モードより
    学習データが大幅に増える。指標は accuracy と ±1 tier 以内の正解率。
    """
    sub = sub.copy()
    tier, cutoffs, src = _build_tier_labels(sub, label_column)
    sub["_tier"] = tier
    sub = sub.dropna(subset=["_tier"])
    sub["_tier"] = sub["_tier"].astype(int)
    n = len(sub)
    if n < min_samples:
        raise ValueError(
            f"学習データ不足(tier): species={species} で {n} 行 "
            f"(数値 {src['from_numeric']} + 質的 {src['from_qualitative']}, 最低 {min_samples} 行必要)"
        )
    if sub["_tier"].nunique() < 2:
        raise ValueError(
            f"tier が 1 種類しか無いため分類不可: species={species}, tier={sorted(sub['_tier'].unique())}"
        )
    if n < 50:
        warnings.warn(
            f"データ数が少ないため精度が不安定になる可能性があります: "
            f"species={species}, n={n} (推奨 50 行以上)",
            stacklevel=2,
        )

    if "datetime" in sub.columns:
        sub["datetime"] = pd.to_datetime(sub["datetime"], errors="coerce", utc=True)
        sub = sub.sort_values("datetime", kind="stable").reset_index(drop=True)

    y = sub["_tier"].astype(int).values
    n_test = max(1, n // 5)
    sub_train = sub.iloc[:-n_test].reset_index(drop=True)
    sub_test = sub.iloc[-n_test:].reset_index(drop=True)
    X_train = features.build_features(sub_train)
    X_test = features.build_features(sub_test)
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test[X_train.columns]
    y_train, y_test = y[:-n_test], y[-n_test:]

    model = _make_classifier(n_train=len(X_train))
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(int)
    acc = float(np.mean(pred == y_test))
    within1 = float(np.mean(np.abs(pred - y_test) <= 1))

    vals, counts = np.unique(y, return_counts=True)
    tier_dist = {int(v): int(c) for v, c in zip(vals, counts)}
    num_for_stats = pd.to_numeric(sub.get(label_column), errors="coerce").dropna().values \
        if label_column in sub.columns else np.array([])
    boat_stats = _compute_boat_stats(num_for_stats) if len(num_for_stats) else {}

    bundle = {
        "task": "tier_clf",
        "model": model,
        "feature_columns": list(X_train.columns),
        "species": species,
        "label_column": label_column,
        "tier_cutoffs": cutoffs,
        "classes_": [int(c) for c in model.classes_],
        "tier_dist": tier_dist,
        "label_source": src,
        "trained_at": datetime.now().isoformat(),
        "n_samples": int(n),
        "boat_stats": boat_stats,
        "metrics": {"accuracy": acc, "within1_acc": within1},
    }
    joblib.dump(bundle, _model_path(species))
    return {
        "species": species,
        "task": "tier_clf",
        "n_samples": n,
        "label_source": src,
        "tier_dist": tier_dist,
        "accuracy": acc,
        "within1_acc": within1,
        "model_path": str(_model_path(species)),
    }


def train(
    species: str,
    integrated_path: Path | str | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    min_samples: int = 10,
    mode: str = "reg",
) -> dict[str, Any]:
    """指定魚種のモデルを学習して保存。

    Args:
        species: 対象魚種（catches.csv の species 列の値）
        integrated_path: 統合データ parquet/csv のパス
        label_column: 学習ラベル列。default "top_per_angler"
        min_samples: 学習に必要な最小行数
        mode: "reg"=top_per_angler 回帰（従来）/ "tier"=5段階ティア分類
              （質的評価行も取り込むのでデータ量が増える）
    """
    integrated_path = Path(integrated_path) if integrated_path else config.INTEGRATED_DIR / "integrated.parquet"
    if not integrated_path.exists():
        raise FileNotFoundError(f"統合データが見つかりません: {integrated_path}")

    df = pd.read_parquet(integrated_path) if integrated_path.suffix == ".parquet" else pd.read_csv(integrated_path)
    sub = df[df["species"] == species].copy()

    if mode == "tier":
        return _train_tier(species, sub, label_column, min_samples)

    if label_column not in sub.columns:
        raise ValueError(f"label_column '{label_column}' が統合データに存在しません。columns={list(sub.columns)}")
    sub = sub.dropna(subset=[label_column])  # ラベル欠損行は学習に使わない
    if len(sub) < min_samples:
        raise ValueError(
            f"学習データ不足: species={species}, label={label_column} で {len(sub)} 行 "
            f"(最低 {min_samples} 行必要)"
        )
    if len(sub) < 50:
        warnings.warn(
            f"データ数が少ないため精度が不安定になる可能性があります: "
            f"species={species}, n={len(sub)} (推奨 50 行以上)",
            stacklevel=2,
        )

    # 時系列順にソート（古い→新しい）— validation を未来側に固定するため
    if "datetime" in sub.columns:
        sub["datetime"] = pd.to_datetime(sub["datetime"], errors="coerce", utc=True)
        sub = sub.sort_values("datetime", kind="stable").reset_index(drop=True)

    y = sub[label_column].astype(float).values
    n = len(sub)
    n_test = max(1, n // 5)

    # split を先に取って features.build_features() を train/test 別個に呼ぶ
    # （build_features 内の fillna(median) で test 行の値が train に漏れるのを防ぐ）
    sub_train = sub.iloc[:-n_test].reset_index(drop=True)
    sub_test = sub.iloc[-n_test:].reset_index(drop=True)
    X_train = features.build_features(sub_train)
    X_test = features.build_features(sub_test)
    # test に train 側の列が無ければ 0、余分な列は捨てる
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test[X_train.columns]
    y_train, y_test = y[:-n_test], y[-n_test:]

    model = _make_estimator(n_train=len(X_train))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))

    boat_stats = _compute_boat_stats(y)  # 全学習データから（テスト除外しない方が安定）

    bundle = {
        "model": model,
        "feature_columns": list(X_train.columns),
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

    if bundle.get("task") == "tier_clf":
        model = bundle["model"]
        classes = [int(c) for c in getattr(model, "classes_", bundle.get("classes_", []))]
        proba = np.asarray(model.predict_proba(X)[0], dtype=float)
        top = int(np.argmax(proba))
        tier_num = int(classes[top])
        tier_label = TIER_LABELS[tier_num - 1] if 1 <= tier_num <= 5 else str(tier_num)
        tier_probs = {int(c): round(float(p), 3) for c, p in zip(classes, proba)}
        bstats = bundle.get("boat_stats") or {}
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
                "tier": tier_num,
                "tier_label": tier_label,
                "tier_probabilities": tier_probs,
                "tier_confidence": round(float(proba[top]), 3),
                "vs_boat_avg": None,
                "vs_boat_median": None,
                "vs_other_boats": None,
            },
            "boat_context": bstats,
            "model": {
                "task": "tier_clf",
                "accuracy": bundle.get("metrics", {}).get("accuracy"),
                "within1_acc": bundle.get("metrics", {}).get("within1_acc"),
                "n_samples_train": bundle.get("n_samples"),
                "trained_at": bundle.get("trained_at"),
            },
        }

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
    p_train.add_argument("--mode", choices=["reg", "tier"], default="tier",
                         help="reg=top_per_angler 回帰 / tier=5段階ティア分類(質的行も活用, default)")

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
            mode=args.mode,
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
