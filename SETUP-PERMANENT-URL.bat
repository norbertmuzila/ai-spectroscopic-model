@echo off
REM ===================================================================
REM   One-time setup: permanent public address + auto-start.
REM
REM   After this the console runs by itself whenever you sign in to
REM   Windows, with no terminal window, at an address that never
REM   changes. The spectrometer stays plugged into this machine.
REM ===================================================================
title Spectral Console - permanent setup
cd /d "%~dp0"

echo.
echo  [1/3] Signing in to Tailscale (a browser window may open)
"C:\Program Files\Tailscale\tailscale.exe" up --hostname=spectral-console --accept-dns=false

echo.
echo  [2/3] Publishing a permanent HTTPS address
python scripts\setup_funnel.py

echo.
echo  [3/3] Installing auto-start (no terminal, runs at sign-in)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_autostart.ps1"

echo.
echo  Done. Your address is in data\public_url.txt and in the dashboard header.
pause
