@echo off
setlocal enabledelayedexpansion
title SeeCT - Launcher
cd /d "%~dp0.."

echo ========================================
echo   SeeCT - Launcher
echo ========================================
echo.

:: --- Step 1: venv ---
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
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
    echo     Using: !PYCMD!
    !PYCMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [1/4] Virtual environment ready.
)

:: --- Step 2: Activate ---
echo [2/4] Activating environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Activation failed.
    pause
    exit /b 1
)
echo [OK] Activated.

:: --- Step 3: Dependencies ---
echo [3/4] Checking dependencies...
.venv\Scripts\python.exe -c "import streamlit, torch, cv2, numpy, PIL, matplotlib" >nul 2>&1
if !errorlevel! neq 0 (
    echo     Installing packages (this may take a few minutes)...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [WARN] Some packages may have failed. Trying to continue...
    ) else (
        echo [OK] Dependencies installed.
    )
) else (
    echo [OK] All dependencies ready.
)

:: --- Step 4: Start ---
echo.
echo [4/4] Starting server...
echo.
echo ========================================
echo   Open http://localhost:8501
echo   Press Ctrl+C to stop.
echo ========================================
start "" http://localhost:8501
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

pause
