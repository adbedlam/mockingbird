@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
uv run --no-sync python -m mockingbird.dashboard --open
if errorlevel 1 pause
