"""YOLOによる魚種判別と釣果カウント抽出。

学習データ収集パイプライン:
  釣果写真 → YOLO 検出 → 種別 count → 釣果ログの空 count を補完
  → predictor.train() で気象/海象/潮汐などから count を予測する回帰モデルを学習

GenesisEngine-v6 で学習した 21種統合モデル (v6.3.3 / finetune_v6, mAP50=0.611) をロード。
英語クラス名で出力されるので JA_NAMES_MAP で日本語化する。

  cls 0-16  : 初期17種 (タイ類/青物/底物/小型/軟体)
  cls 17    : カサゴ        (add_rockfish.py)
  cls 18    : ホウボウ      (add_chita_local.py)
  cls 19    : オコゼ        (add_chita_local.py)
  cls 20    : サワラ        (add_chita_local.py)
  cls 11 (Puffer) はトラフグ特化に書き換え済み
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


# ─────────────────────────────────────────────────────────────
# GenesisEngine-v6 FISH_TARGETS の英→日マッピング (cls 順、独立コピー)
# config.FISH_TARGETS のキーと「ja」値を貼り付け固定化することで
# GenesisEngine-v6 への import 依存を避ける (別 Drive ディレクトリ)
# ─────────────────────────────────────────────────────────────
JA_NAMES_MAP: dict[str, str] = {
    "Red Sea Bream":           "マダイ",
    "Black Sea Bream":         "クロダイ",
    "Crimson Sea Bream":       "チダイ",
    "Japanese Amberjack":      "ブリ",
    "Striped Bonito":          "カツオ",
    "Largehead Hairtail":      "タチウオ",
    "Japanese Flounder":       "ヒラメ",
    "Japanese Sea Bass":       "スズキ",
    "Flathead":                "マゴチ",
    "Filefish":                "カワハギ",
    "Japanese Whiting":        "シロギス",
    "Puffer":                  "フグ",
    "Japanese Horse Mackerel": "アジ",
    "Chub Mackerel":           "サバ",
    "Isaki":                   "イサキ",
    "Squid":                   "イカ",
    "Common Octopus":          "マダコ",
    "Rockfish":                "カサゴ",
    "Red Gurnard":             "ホウボウ",
    "Devil Stinger":           "オコゼ",
    "Japanese Spanish Mackerel": "サワラ",
}


def to_ja(species_en_or_ja: str) -> str:
    """英語クラス名なら日本語に変換、既に日本語ならそのまま。未登録はそのまま返す。"""
    return JA_NAMES_MAP.get(species_en_or_ja, species_en_or_ja)


# ─────────────────────────────────────────────────────────────
# 単一検出
# ─────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    """1 検出 = 1 bbox = 1 個体."""
    image_path: Path
    species: str                      # 日本語 (マダイなど)
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2


@dataclass
class ImageDetections:
    """1 画像の全 bbox + 種別 count."""
    image_path: Path
    detections: list[Prediction] = field(default_factory=list)

    @property
    def count_per_species(self) -> dict[str, int]:
        """魚種(ja) → 検出個数."""
        return dict(Counter(d.species for d in self.detections))

    @property
    def total(self) -> int:
        return len(self.detections)


# ─────────────────────────────────────────────────────────────
# 推論ラッパー
# ─────────────────────────────────────────────────────────────
class FishSpeciesPredictor:
    """学習済み YOLO 統合モデルのラッパー (遅延ロード)."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
    ) -> None:
        self.weights_path = (
            Path(weights_path) if weights_path else config.YOLO_DEFAULT_WEIGHTS
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    # ── ロード ────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.weights_path.exists():
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
        except ImportError:
            self._model = None

    @property
    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    # ── 単一最良検出 (1枚 → 1検出、既存 API 互換) ─────────
    def predict(self, image_path: str | Path) -> Prediction:
        self._ensure_loaded()
        if self._model is None:
            return Prediction(image_path=Path(image_path), species="未判定", confidence=0.0)

        results = self._model.predict(
            str(image_path), conf=self.conf_threshold,
            iou=self.iou_threshold, verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return Prediction(image_path=Path(image_path), species="不明", confidence=0.0)
        best = max(results[0].boxes, key=lambda b: float(b.conf))
        conf = float(best.conf)
        if conf < self.conf_threshold:
            return Prediction(image_path=Path(image_path), species="不明", confidence=conf)
        cls_id = int(best.cls)
        species_en = self._model.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        return Prediction(
            image_path=Path(image_path),
            species=to_ja(species_en),
            confidence=conf,
            bbox=(x1, y1, x2, y2),
        )

    def predict_batch(self, image_paths: Iterable[str | Path]) -> list[Prediction]:
        return [self.predict(p) for p in image_paths]

    # ── 複数 bbox 検出 (1枚 → 全検出、count 用) ────────────
    def detect_all(self, image_path: str | Path) -> ImageDetections:
        """1 枚の画像から全 bbox を返す (釣果数えに使う)."""
        self._ensure_loaded()
        img_path = Path(image_path)
        det = ImageDetections(image_path=img_path)
        if self._model is None:
            return det

        results = self._model.predict(
            str(image_path), conf=self.conf_threshold,
            iou=self.iou_threshold, verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return det

        for box in results[0].boxes:
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls)
            species_en = self._model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            det.detections.append(Prediction(
                image_path=img_path,
                species=to_ja(species_en),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))
        return det

    def detect_batch(self, image_paths: Iterable[str | Path]) -> list[ImageDetections]:
        return [self.detect_all(p) for p in image_paths]


# ─────────────────────────────────────────────────────────────
# 釣果ログ補完
# ─────────────────────────────────────────────────────────────
def fill_missing_species(
    catches: pd.DataFrame,
    predictor: FishSpeciesPredictor,
) -> pd.DataFrame:
    """`species` が空の行について `image_path` から推論して埋める (既存 API)."""
    df = catches.copy()
    if "species" not in df.columns or "image_path" not in df.columns:
        return df
    mask = df["species"].isna() | (df["species"].astype(str).str.strip() == "")
    targets = df.loc[mask & df["image_path"].notna(), "image_path"].tolist()
    if not targets:
        return df
    preds = predictor.predict_batch(targets)
    pred_map = {str(p.image_path): p for p in preds}
    for idx in df.index[mask]:
        ip = df.at[idx, "image_path"]
        if pd.isna(ip):
            continue
        p = pred_map.get(str(ip))
        if p and p.confidence >= predictor.conf_threshold:
            df.at[idx, "species"] = p.species
            note = df.at[idx, "notes"] if "notes" in df.columns else ""
            df.at[idx, "notes"] = (
                (str(note) if pd.notna(note) else "") + f" [YOLO推定 conf={p.confidence:.2f}]"
            ).strip()
    return df


def fill_missing_count(
    catches: pd.DataFrame,
    predictor: FishSpeciesPredictor,
    explode_per_species: bool = True,
) -> pd.DataFrame:
    """`count` が空の行について `image_path` から検出個数を埋める。

    explode_per_species=True (default):
        1 写真に複数魚種が写っている場合、種別ごとに行を分割する。
        (species 列が「マダイ」「チダイ」など特定種なら、その種だけカウント)
    explode_per_species=False:
        species 列に対応する種だけを数えて count を埋める。新しい行は追加しない。

    入力 DataFrame に必須の列: image_path, species(空可), count(空可)
    任意列: trip_id, date, site, boat, ...などはそのままコピーされる
    """
    df = catches.copy()
    if "image_path" not in df.columns:
        raise ValueError("catches に image_path 列が必要です")
    if "count" not in df.columns:
        df["count"] = pd.NA

    # count が空 & image_path がある行を対象に
    mask = df["count"].isna() & df["image_path"].notna() & \
        (df["image_path"].astype(str).str.strip() != "")
    target_rows = df[mask].copy()
    if target_rows.empty:
        return df

    # 各画像で1回だけ検出 (重複呼び出し回避)
    unique_paths = target_rows["image_path"].astype(str).unique().tolist()
    det_map: dict[str, ImageDetections] = {}
    for p in unique_paths:
        det_map[p] = predictor.detect_all(p)

    out_rows = []
    for idx, row in target_rows.iterrows():
        img = str(row["image_path"])
        det = det_map.get(img)
        if det is None or det.total == 0:
            # 検出 0 → count=0 で埋める (魚が写っていない or 信頼度不足)
            new = row.copy()
            new["count"] = 0
            note_col = row.get("notes", "")
            new["notes"] = (str(note_col) if pd.notna(note_col) else "") \
                + " [YOLO検出0]"
            new["notes"] = new["notes"].strip()
            out_rows.append(new)
            continue

        per_sp = det.count_per_species  # {マダイ: 3, チダイ: 1, ...}
        row_species = row.get("species")
        row_species = str(row_species).strip() if pd.notna(row_species) else ""

        if row_species:
            # species が既に書かれている → その種だけ数える
            new = row.copy()
            new["count"] = per_sp.get(row_species, 0)
            note_col = row.get("notes", "")
            new["notes"] = (
                (str(note_col) if pd.notna(note_col) else "")
                + f" [YOLO count={new['count']}]"
            ).strip()
            out_rows.append(new)
        elif explode_per_species and per_sp:
            # species 空 → 検出された種ごとに行を複製
            for sp, c in per_sp.items():
                new = row.copy()
                new["species"] = sp
                new["count"] = c
                note_col = row.get("notes", "")
                new["notes"] = (
                    (str(note_col) if pd.notna(note_col) else "")
                    + f" [YOLO 種推定+count={c}]"
                ).strip()
                out_rows.append(new)
        else:
            # species 空 & explode しない → 最頻種だけ採用
            top_sp = max(per_sp, key=per_sp.get)
            new = row.copy()
            new["species"] = top_sp
            new["count"] = per_sp[top_sp]
            out_rows.append(new)

    # 結合: 対象外の行 + 新しい行
    out_df = pd.concat([df[~mask], pd.DataFrame(out_rows)], ignore_index=True)
    return out_df


# ─────────────────────────────────────────────────────────────
# CLI (動作確認 + 釣果ログ補完)
# ─────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO 推論 + 釣果ログ補完")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("predict", help="1 枚の画像を推論")
    p_one.add_argument("image", help="画像パス")
    p_one.add_argument("--weights", default=None, help="重み (default: config.YOLO_DEFAULT_WEIGHTS)")
    p_one.add_argument("--conf", type=float, default=0.4)

    p_det = sub.add_parser("detect_all", help="1 枚の画像から全 bbox を出す")
    p_det.add_argument("image")
    p_det.add_argument("--weights", default=None)
    p_det.add_argument("--conf", type=float, default=0.4)

    p_fill = sub.add_parser("fill_count", help="釣果 CSV の count 列を埋める")
    p_fill.add_argument("csv_in",  help="入力 CSV (列: image_path, species(空可), count(空可), ...)")
    p_fill.add_argument("csv_out", help="出力 CSV")
    p_fill.add_argument("--weights", default=None)
    p_fill.add_argument("--conf", type=float, default=0.4)
    p_fill.add_argument("--no_explode", action="store_true",
                        help="species 空でも行分割しない (最頻種を採用)")

    args = parser.parse_args()
    pred = FishSpeciesPredictor(
        weights_path=getattr(args, "weights", None) or config.YOLO_DEFAULT_WEIGHTS,
        conf_threshold=getattr(args, "conf", 0.4),
    )
    if not pred.is_ready:
        print(f"❌ モデル未ロード: weights={pred.weights_path}")
        raise SystemExit(1)

    if args.cmd == "predict":
        p = pred.predict(args.image)
        print(json.dumps({
            "image": str(p.image_path), "species": p.species,
            "confidence": p.confidence, "bbox": p.bbox,
        }, ensure_ascii=False, indent=2))
    elif args.cmd == "detect_all":
        det = pred.detect_all(args.image)
        print(json.dumps({
            "image": str(det.image_path),
            "total": det.total,
            "count_per_species": det.count_per_species,
            "detections": [
                {"species": d.species, "confidence": d.confidence, "bbox": d.bbox}
                for d in det.detections
            ],
        }, ensure_ascii=False, indent=2))
    elif args.cmd == "fill_count":
        df_in = pd.read_csv(args.csv_in)
        df_out = fill_missing_count(
            df_in, pred, explode_per_species=not args.no_explode,
        )
        df_out.to_csv(args.csv_out, index=False)
        print(f"✅ {len(df_in)} → {len(df_out)} 行: {args.csv_out}")


if __name__ == "__main__":
    _cli()
