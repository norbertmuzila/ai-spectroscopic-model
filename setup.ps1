# Setup for the AI Spectroscopic Model (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AI Spectroscopic Model - setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Python" -ForegroundColor Yellow
python --version

Write-Host "`n[2/4] Installing dependencies" -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "`n[3/4] Checking for the spectrometer" -ForegroundColor Yellow
$devs = python -c "try:`n import seabreeze.spectrometers as s`n print(len(s.list_devices()))`nexcept Exception as e:`n print('ERR', e)" 2>&1
Write-Host "  seabreeze reports: $devs"
Write-Host "  If this is 0 or ERR, the USB4000 needs a libusb driver."
Write-Host "  Install Zadig from https://zadig.akeo.ie/, select the USB4000,"
Write-Host "  choose libusb-win32 or WinUSB, and click Install Driver."
Write-Host "  Close SpectraSuite and OceanView first - they hold the device."

Write-Host "`n[4/4] Training the model (this takes 5-20 minutes)" -ForegroundColor Yellow
python scripts/train.py

Write-Host "`nSetup complete. Start the dashboard with:" -ForegroundColor Green
Write-Host "  python run.py" -ForegroundColor Green
Write-Host "`nOptional but recommended - install the measured USGS library:" -ForegroundColor Green
Write-Host "  python scripts/fetch_usgs_library.py" -ForegroundColor Green
Write-Host "  python scripts/train.py" -ForegroundColor Green
