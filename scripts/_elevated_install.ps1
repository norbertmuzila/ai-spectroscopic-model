<#
  Elevated half of the USB4000 driver installation.

  Launched by INSTALL-DRIVER.ps1 (or directly with -Verb RunAs). Everything it
  does is written to drivers\install-log.txt as well as the console, so the
  result can be verified afterwards from an unelevated shell.
#>
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$log  = Join-Path $root "drivers\install-log.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Set-Content -Path $log -Value "" -Encoding utf8

function Say($msg) {
    Write-Host $msg
    Add-Content -Path $log -Value $msg -Encoding utf8
}

Say "=== USB4000 driver install  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
Say "elevated: $isAdmin"
if (-not $isAdmin) {
    Say "RESULT: NOT_ELEVATED"
    exit 1
}

# --- close anything holding the device ------------------------------------
$holders = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match "SpectraSuite|OceanView|OmniDriver" }
if ($holders) {
    foreach ($h in $holders) { Say ("holder running: {0} (pid {1})" -f $h.ProcessName, $h.Id) }
} else {
    Say "no conflicting Ocean Optics software running"
}

# --- before ----------------------------------------------------------------
$before = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_2457&PID_1022*" }
foreach ($d in $before) { Say ("before: {0}  status={1}  problem={2}" -f $d.InstanceId, $d.Status, $d.Problem) }

# --- install ---------------------------------------------------------------
$inf = Join-Path $root "drivers\win\OOI_USB4000.inf"
Say "installing: $inf"
$out = & pnputil.exe /add-driver "$inf" /install 2>&1
$rc = $LASTEXITCODE
foreach ($line in $out) { Say ("  pnputil: " + $line) }
Say "pnputil exit: $rc"

# --- if that failed, try the whole Ocean Optics driver set ------------------
if ($rc -ne 0 -and $rc -ne 259) {
    Say "single INF rejected; trying every Ocean Optics INF in the package"
    $out2 = & pnputil.exe /add-driver (Join-Path $root "drivers\win\OOI_*.inf") /install 2>&1
    $rc2 = $LASTEXITCODE
    foreach ($line in $out2) { Say ("  pnputil(all): " + $line) }
    Say "pnputil(all) exit: $rc2"
    if ($rc2 -eq 0) { $rc = 0 }
}

# --- force Windows to reconsider the attached device ------------------------
Say "rescanning device tree"
& pnputil.exe /scan-devices 2>&1 | ForEach-Object { Say ("  scan: " + $_) }

# If the device is still code 28, restarting it makes Windows redo driver
# selection now that the package is in the store.
foreach ($d in (Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_2457&PID_1022*" })) {
    if ($d.Status -ne "OK") {
        Say ("restarting device {0}" -f $d.InstanceId)
        try {
            Disable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false -ErrorAction Stop
            Start-Sleep -Seconds 2
            Enable-PnpDevice  -InstanceId $d.InstanceId -Confirm:$false -ErrorAction Stop
            Say "  restarted"
        } catch {
            Say ("  restart failed: " + $_.Exception.Message)
        }
    }
}
Start-Sleep -Seconds 3

# --- after -----------------------------------------------------------------
$ok = $false
foreach ($d in (Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_2457&PID_1022*" })) {
    $svc = ""
    try { $svc = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_Service').Data } catch {}
    Say ("after: {0}  status={1}  problem={2}  service={3}" -f $d.InstanceId, $d.Status, $d.Problem, $(if ($svc) { $svc } else { "<none>" }))
    if ($d.Status -eq "OK" -and $svc -match "WinUSB|libusb") { $ok = $true }
}

Say ("RESULT: " + $(if ($ok) { "DRIVER_BOUND" } else { "STILL_UNBOUND" }))
Say "=== done ==="

if (-not $ok) {
    Write-Host ""
    Write-Host "The driver did not bind. Use Zadig instead:" -ForegroundColor Yellow
    Write-Host "  1. https://zadig.akeo.ie/  (no install needed, just run it)"
    Write-Host "  2. Options -> List All Devices"
    Write-Host "  3. Pick 'Ocean Optics USB4000' in the dropdown"
    Write-Host "  4. Choose WinUSB on the right, click Replace Driver"
}

Write-Host ""
Write-Host "Log written to: $log"
Write-Host "You can close this window."
Start-Sleep -Seconds 20
