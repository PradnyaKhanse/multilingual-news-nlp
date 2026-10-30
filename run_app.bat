@echo off
title Multilingual News NLP Project Launcher
color 0A
cd /d "%~dp0"

echo ======================================================================
echo           Starting Multilingual News NLP Project (Streamlit)
echo ======================================================================
echo.

REM 1. Set UTF-8 encoding for Devanagari (Hindi/Marathi) and English
set PYTHONIOENCODING=utf-8

REM 2. Check if local virtual environment exists
if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please ensure Python is installed or run: python -m venv .venv ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 3. Launch Streamlit app
echo Launching Streamlit web application...
echo App URL: http://localhost:8501
echo.
echo NOTE: Keep this command window open/minimized during your presentation!
echo ======================================================================
echo.

".venv\Scripts\streamlit.exe" run app.py

pause
