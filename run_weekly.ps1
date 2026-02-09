# run_weekly.ps1 - one-click runner for semi-cycle-lab
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$ScriptPy = Join-Path $Root "scripts\build_report.py"

if (-not (Test-Path $ScriptPy)) {
    Write-Host "[error] Cannot find: $ScriptPy" -ForegroundColor Red
    Write-Host "Make sure you are running this from the repo root: $Root"
    exit 2
}

if (-not (Test-Path $VenvPy)) {
    Write-Host "[error] Cannot find venv python: $VenvPy" -ForegroundColor Red
    Write-Host "Fix: create venv in repo root and install deps:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\activate"
    Write-Host "  python -m pip install -r requirements.txt"
    exit 3
}

# Optional: log every run
$LogsDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogPath = Join-Path $LogsDir "run_$Stamp.log"

Write-Host "[info] Repo root : $Root"
Write-Host "[info] Python    : $VenvPy"
Write-Host "[info] Running   : $ScriptPy"
Write-Host "[info] Log file  : $LogPath"
Write-Host ""

# Run report builder
& $VenvPy $ScriptPy 2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[error] build_report failed. See log: $LogPath" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[ok] Done. Open:" -ForegroundColor Green
Write-Host "  reports\index.md"
Write-Host "  reports\$(Get-Date -Format 'yyyy-MM-dd').md"
