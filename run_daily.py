"""常駐の本体。1日1回これを回せば釣果ログが貯まり続ける。

    python run_daily.py              # 更新 → integrated 再構築 → commit & push
    python run_daily.py --no-push    # commit まで（push せん）
    python run_daily.py --no-git     # ファイルを更新するだけ
    python run_daily.py --status     # 今どうなっとるか（通信もAPIも使わん）

設計:
  1. **窓は広めに取る**。既存エントリは skip_existing で飛ばすので、
     窓を広げても増えるのは列挙のコストだけ。PC が数日落ちとっても取り返せる。
  2. **追記しかせん**。data/scraped/ は .gitignore 対象で、手元のスクレイプ
     キャッシュは Colab 版の一部でしかない。全面再構築すると必ず行が減るので
     update_all は新規キーの追記だけを行う（src/update_all.py の docstring 参照）。
  3. **commit & push まで含める**。この repo は Colab から clone して使うので、
     push しとらん更新は次の再 clone で巻き戻る。
  4. LLM の無料枠は1日ぶん（13隻で 15〜25 エントリ程度）なら十分収まる。
     枠に当たってもカスケードが別プロバイダに逃がすし、取れんかったエントリは
     extraction_method="none" のまま残るので翌日以降に拾い直せる。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402  (.env を読ませるため先に import)
from src.update_all import rebuild_integrated, update_all  # noqa: E402

DEFAULT_MONTHS = 2   # PC が長めに落ちとっても取り返せる窓


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def status() -> None:
    """通信もAPIも使わずに、今どこまで貯まっとるかだけ出す。"""
    path = config.FISHING_DIR / "catches.csv"
    if not path.exists():
        print("catches.csv が無い")
        return
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    print(f"catches.csv: {len(df)} 行 / 最新 {dt.max()} / 最古 {dt.min()}")
    lag = (pd.Timestamp.now(tz="UTC") - dt.max()).days
    print(f"最新からの遅れ: {lag} 日" + ("  ⚠️ 止まっとる可能性" if lag > 3 else ""))
    g = df.assign(dt=dt).groupby("boat")["dt"].max().sort_values()
    print("\n船宿別の最新:")
    for boat, d in g.items():
        print(f"   {boat:14s} {d}")
    ipath = config.INTEGRATED_DIR / "integrated.parquet"
    if ipath.exists():
        i = pd.read_parquet(ipath)
        print(f"\nintegrated.parquet: {len(i)} 行 / {len(i.columns)} 列")


def main() -> int:
    p = argparse.ArgumentParser(description="釣果ログの日次更新")
    p.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-git", action="store_true")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()

    if a.status:
        status()
        return 0

    print(f"===== 日次更新 {datetime.now():%Y-%m-%d %H:%M:%S} =====", flush=True)
    path = config.FISHING_DIR / "catches.csv"
    before = len(pd.read_csv(path)) if path.exists() else 0

    update_all(months_back=a.months)

    after = len(pd.read_csv(path)) if path.exists() else 0
    if after == before:
        print("新規行なし。integrated の再構築も commit も行わん。")
        return 0

    rebuild_integrated()

    if a.no_git:
        return 0

    _git("add", "data/fishing_logs/catches.csv", "data/integrated/integrated.parquet")
    if not _git("diff", "--cached", "--quiet").returncode:
        print("staged 差分なし。commit せん。")
        return 0
    msg = f"data: 日次更新 {datetime.now():%Y-%m-%d} ({before} -> {after} 行, +{after - before})"
    r = _git("commit", "-m", msg)
    print(r.stdout.strip() or r.stderr.strip())
    if a.no_push:
        return 0

    # push は競合しうる（Colab 側からも push される）。落ちても手元の更新は残る。
    r = _git("push", "origin", "HEAD")
    if r.returncode:
        print(f"⚠️ push 失敗（手元の commit は残っとる。次回か手動で解消すること）:\n{r.stderr.strip()[:500]}")
        return 0
    print("✅ push 済み")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
