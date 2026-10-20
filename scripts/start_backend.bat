@echo off
REM 启动后端（Windows）
cd /d "%~dp0..\backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
