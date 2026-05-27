"""matplotlib で日本語ラベルを文字化けなく表示するためのフォント設定。

優先順:
  1. japanize-matplotlib があれば即適用（最も確実）
  2. システムにある日本語フォントを順に探索（Windows / macOS / Linux 共通）
  3. 見つからなければ警告のみ（描画は継続）

`from src.font_setup import apply` を最初に呼ぶだけで OK。
"""
from __future__ import annotations

import warnings
from typing import Iterable

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_CANDIDATE_FONTS: tuple[str, ...] = (
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "Hiragino Sans",
    "Hiragino Maru Gothic Pro",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
)


def _installed_font_names() -> set[str]:
    return {f.name for f in fm.fontManager.ttflist}


def _first_available(candidates: Iterable[str]) -> str | None:
    installed = _installed_font_names()
    for name in candidates:
        if name in installed:
            return name
    return None


def apply(font: str | None = None) -> str | None:
    """日本語フォントを matplotlib のデフォルトに設定する。

    Args:
        font: 明示的にフォント名を指定したい場合に渡す。

    Returns:
        実際に設定されたフォント名。見つからなければ None。
    """
    try:
        import japanize_matplotlib  # noqa: F401
        plt.rcParams["axes.unicode_minus"] = False
        return plt.rcParams["font.family"][0] if plt.rcParams["font.family"] else None
    except ImportError:
        pass

    chosen = font or _first_available(_CANDIDATE_FONTS)
    if chosen is None:
        warnings.warn(
            "日本語フォントが見つかりませんでした。"
            "`pip install japanize-matplotlib` を推奨します。",
            stacklevel=2,
        )
        return None

    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


if __name__ == "__main__":
    name = apply()
    print(f"使用フォント: {name}")
