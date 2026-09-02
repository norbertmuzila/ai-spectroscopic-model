@echo off
REM ===================================================================
REM   USB4000 Spectral Console - one-click launcher
REM   Double-click this file. It installs what is missing, trains the
REM   model the first time, then opens the dashboard in your browser.
REM ===================================================================
title USB4000 Spectral Console
cd /d "%~dp0"

echo.
echo  ==================================================================
echo    USB4000 Spectral Console
echo  ==================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo  [X] Python is not on your PATH.
  echo      Install Python 3.11 from python.org and tick "Add to PATH".
  pause
  exit /b 1
)

python -c "import fastapi, sklearn, seabreeze, libusb_package" >nul 2>&1
if errorlevel 1 (
  echo  Installing dependencies, this happens once and takes a few minutes...
  python -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo  [X] Dependency install failed. Scroll up for the reason.
    pause
    exit /b 1
  )
)

if not exist "data\models\ensemble.joblib" (
  echo.
  echo  Training the identification model. This happens once, ~4 minutes.
  python scripts\train.py
)

echo.
echo  Checking the spectrometer...
python scripts\diagnose.py
echo.
echo  ------------------------------------------------------------------
echo   Opening http://127.0.0.1:8000
echo   Leave this window open. Press Ctrl+C here to stop the server.
echo  ------------------------------------------------------------------
echo.
python run.py
pause
