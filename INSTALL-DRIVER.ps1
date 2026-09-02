<#
  ============================================================================
   USB4000 driver installer

   Right-click this file and choose "Run with PowerShell", or double-click
   INSTALL-DRIVER.bat next to it. Accept the Windows security prompt.

   What it does and why
   --------------------
   Your USB4000 currently reports Windows problem code 28,
   CM_PROB_FAILED_INSTALL: "the drivers for this device are not installed".
   Windows enumerates the device but has bound no driver to it, so no library
   can open it - not python-seabreeze, not SpectraSuite, nothing.

   This installs OOI_USB4000.inf, which matches USB\VID_2457&PID_1022 exactly
   and binds Microsoft's in-box WinUSB driver. WinUSB is what python-seabreeze
   talks through. The driver files ship with python-seabreeze and are already
   downloaded into drivers\win next to this script, so this runs offline.

   Installing a driver modifies the system driver store, which is why Windows
   requires Administrator and why this cannot be done from the dashboard.
  ============================================================================
#>

$ErrorActionPreference = "Stop"

# ---- elevate ---------------------------------------------------------------
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Host "Requesting administrator rights (driver installation requires it)..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-File", "`"$PSCommandPath`""
    )
    exit
}

$root = Split-Path -Parent $PSCommandPath
Set-Location $root

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  USB4000 driver installation" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1. is the device plugged in? -----------------------------------------
Write-Host "[1/5] Looking for the spectrometer..."
$dev = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_2457&PID_1022*" }
if (-not $dev) {
    Write-Host "  The USB4000 is not attached." -ForegroundColor Red
    Write-Host "  Plug it into a USB port directly on the computer (not a hub),"
    Write-Host "  using a DATA cable, then run this again."
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}
foreach ($d in $dev) {
    Write-Host ("  found: {0}" -f $d.InstanceId)
    Write-Host ("  status: {0}   problem: {1}" -f $d.Status, $d.ProblemDescription)
}

# ---- 2. anything holding the device? --------------------------------------
Write-Host ""
Write-Host "[2/5] Checking for software that would hold the device open..."
$holders = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match "SpectraSuite|OceanView|OmniDriver|javaw" }
if ($holders) {
    Write-Host "  These are running and can claim the spectrometer:" -ForegroundColor Yellow
    $holders | ForEach-Object { Write-Host ("    {0} (pid {1})" -f $_.ProcessName, $_.Id) }
    Write-Host "  Close them before measuring - only one process can own the device."
} else {
    Write-Host "  nothing conflicting is running."
}

# ---- 3. install the INF ----------------------------------------------------
$inf = Join-Path $root "drivers\win\OOI_USB4000.inf"
Write-Host ""
Write-Host "[3/5] Installing $inf"
if (-not (Test-Path $inf)) {
    Write-Host "  Driver files are missing. Re-download them with:" -ForegroundColor Red
    Write-Host "     python scripts\fetch_driver.py"
    Read-Host "Press Enter to close"
    exit 1
}

& pnputil.exe /add-driver "$inf" /install
$rc = $LASTEXITCODE
Write-Host ("  pnputil exit code: {0}" -f $rc)

# 259 = ERROR_NO_MORE_ITEMS, returned when the package is already present.
if ($rc -ne 0 -and $rc -ne 259) {
    Write-Host ""
    Write-Host "  pnputil refused the package." -ForegroundColor Yellow
    Write-Host "  The catalog is from 2010 and signed with SHA-1, which Windows 11"
    Write-Host "  may reject. Use Zadig instead - it generates a fresh signed"
    Write-Host "  catalog and does the same job:"
    Write-Host "     1. Download https://zadig.akeo.ie/"
    Write-Host "     2. Options -> List All Devices"
    Write-Host "     3. Select 'Ocean Optics USB4000' in the dropdown"
    Write-Host "     4. Choose WinUSB on the right, click Replace Driver"
}

# ---- 4. re-scan so Windows binds it to the attached device ----------------
Write-Host ""
Write-Host "[4/5] Rescanning the USB bus..."
& pnputil.exe /scan-devices | Out-Null
Start-Sleep -Seconds 3

$dev2 = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_2457&PID_1022*" }
foreach ($d in $dev2) {
    $svc = ""
    try { $svc = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_Service').Data } catch {}
    Write-Host ("  status: {0}   driver service: {1}" -f $d.Status, $(if ($svc) { $svc } else { "<still none>" }))
}

# ---- 5. verify from Python -------------------------------------------------
Write-Host ""
Write-Host "[5/5] Verifying that seabreeze can open it..."
Write-Host ""
& python "$root\scripts\diagnose.py"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  If the verdict above says the spectrometer is reading, start the" -ForegroundColor Cyan
Write-Host "  dashboard with START.bat and press Connect." -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close"
