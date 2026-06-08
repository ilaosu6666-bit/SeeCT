@echo off
setlocal enabledelayedexpansion
title SeeCT - Launcher
cd /d "%~dp0"

echo ========================================
echo   SeeCT - Launcher
echo ========================================
echo.

:: --- Create venv if missing ---
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found. Creating...
    set "PYCMD="
    python --version >nul 2>&1 && set "PYCMD=python"
    if "!PYCMD!"=="" python3 --version >nul 2>&1 && set "PYCMD=python3"
    if "!PYCMD!"=="" py --version >nul 2>&1 && set "PYCMD=py"
    if "!PYCMD!"=="" (
        echo [ERROR] Python not found. Install Python 3.10+ first.
        echo https://www.python.org/downloads/
        pause
        exit /b 1
    )
    !PYCMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Using existing virtual environment.
)

:: --- Activate venv ---
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: --- Install dependencies if needed ---
python -c "import streamlit, torch, cv2, numpy, PIL, matplotlib" >nul 2>&1
if !errorlevel! neq 0 (
    echo [SETUP] Installing dependencies (may take a few minutes)...
    pip install -r requirements.txt -q
    if !errorlevel! neq 0 (
        echo [WARN] Some packages may have failed. Trying to continue...
    )
    echo [OK] Dependencies installed.
) else (
    echo [OK] All dependencies ready.
)

:: --- Start server ---
echo.
echo ========================================
echo   Starting at http://localhost:8501
echo   Press Ctrl+C to stop.
echo ========================================
start "" http://localhost:8501
streamlit run app.py --server.port 8501

pause
