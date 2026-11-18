@echo off
chcp 65001 >nul
title COVID-19 Chest X-Ray Medical Diagnosis
cd /d "%~dp0"

echo ==================================================
echo   COVID-19 Chest X-Ray Medical Diagnosis App
echo ==================================================
echo Starting server... Your browser will open shortly.
echo Press Ctrl+C in this window to stop the server.
echo.

python app.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] The app exited with an error.
    pause
)
