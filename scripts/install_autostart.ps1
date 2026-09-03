<#
  ============================================================================
   Make the Spectral Console start by itself, with no terminal window.

   Registers a Scheduled Task that launches the console when you sign in to
   Windows. It runs under pythonw.exe, which is the console-less build of the
   interpreter - so there is no black window to leave open and nothing to close
   by accident.

   The task runs as you rather than as SYSTEM. That is deliberate: the console
   drives a USB device, and a process in your own session has the most
   predictable access to it.

   Usage:
       powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
       powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
  ============================================================================
#>
param(
    [switch]$Remove,
    [string]$TaskName = "Spectral Console",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    } catch {
        Write-Host "No task named '$TaskName' was registered." -ForegroundColor Yellow
    }
    exit 0
}

# ---- locate pythonw.exe ----------------------------------------------------
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Host "python is not on PATH." -ForegroundColor Red
    exit 1
}
$pythonw = Join-Path (Split-Path -Parent $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) {
    Write-Host "pythonw.exe not found next to $python - falling back to python.exe" -ForegroundColor Yellow
    $pythonw = $python
}

$runpy = Join-Path $root "run.py"
if (-not (Test-Path $runpy)) {
    Write-Host "run.py not found at $runpy" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Installing auto-start" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  interpreter : $pythonw"
Write-Host "  script      : $runpy"
Write-Host "  working dir : $root"
Write-Host "  task name   : $TaskName"

$action = New-ScheduledTaskAction -Execute $pythonw `
    -Argument "`"$runpy`" --no-browser --port $Port" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Delay the start a little: at sign-in the USB stack and the network are still
# settling, and a console that comes up before them just fails its first
# hardware probe.
$trigger.Delay = "PT20S"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Runs the USB4000 Spectral Console in the background at sign-in." | Out-Null

Write-Host ""
Write-Host "  Registered." -ForegroundColor Green

# ---- start it now so there is nothing to do but use it ---------------------
Write-Host "  Starting it now..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 8

$up = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}

Write-Host ""
if ($up) {
    Write-Host "  The console is running in the background on port $Port." -ForegroundColor Green
} else {
    Write-Host "  It did not answer on port $Port yet. First start also loads the" -ForegroundColor Yellow
    Write-Host "  52 MB model, which can take a minute. Check again with:" -ForegroundColor Yellow
    Write-Host "     Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
}
Write-Host ""
Write-Host "  It will now start automatically every time you sign in."
Write-Host "  To stop that:  powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove"
Write-Host ""
