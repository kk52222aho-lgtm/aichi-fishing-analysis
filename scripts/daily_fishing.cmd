@echo off
rem 釣り船ブログの日次収集。ブログは流れる源で、遡れる範囲にも限りがある。
rem   1. 全13隻の新規エントリを取得 → 本文を LLM 抽出 → catches.csv に追記
rem   2. integrated.parquet を再構築（アプリと予測が読む本体）
rem   3. commit & push（push しとらん更新は Colab の再 clone で巻き戻るため）
rem タスクスケジューラ: aichi-fishing-daily  毎日 06:30
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python run_daily.py >> logs\task.log 2>&1
exit /b %ERRORLEVEL%
