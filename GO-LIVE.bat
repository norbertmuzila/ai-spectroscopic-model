@echo off
REM ===================================================================
REM   Publish the Spectral Console at a public HTTPS URL.
REM
REM   The spectrometer stays plugged into THIS machine - that is the
REM   only way a public link can read real hardware, because a cloud
REM   server has no USB bus. Leave this window open; closing it takes
REM   the URL down.
REM ===================================================================
title Spectral Console - LIVE
cd /d "%~dp0"

if not exist "tools\cloudflared.exe" (
  echo Downloading the tunnel client, one time only...
  python scripts\fetch_cloudflared.py
)

echo.
echo  Starting. The public URL and password appear below in a few seconds.
echo.
python run.py --tunnel --no-browser
pause
