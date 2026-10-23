@echo off
title Propjini Face Verification Service
echo ========================================================
echo Starting Propjini Face Verification Service on Port 8000
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
