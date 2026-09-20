@echo off
rem Daily collection from the charter-boat blogs. The blogs are a flowing source
rem and only reach back so far, so a missed day is not always recoverable.
rem   1. Fetch new entries for all 13 boats -> LLM extract -> append catches.csv
rem   2. Rebuild integrated.parquet (what the app and the predictor read)
rem   3. commit and push (unpushed updates are rolled back by Colab's re-clone)
rem Scheduled task: aichi-fishing-daily  daily 06:30
rem
rem KEEP THIS FILE ASCII-ONLY. cmd.exe reads .cmd as CP932; UTF-8 Japanese in a
rem rem line desynchronises the parser and the tail gets run as a command. It
rem exits WITHOUT writing a log line, so nothing looks wrong from the outside.
rem See memory feedback_silent_cron_death (29 days lost that way).
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python run_daily.py >> logs\task.log 2>&1
exit /b %ERRORLEVEL%
