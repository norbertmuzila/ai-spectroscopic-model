@echo off
REM Double-click this to install the USB4000 driver.
REM It will ask for administrator rights - that is required to install any
REM Windows driver, and is why the dashboard cannot do it for you.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL-DRIVER.ps1"
