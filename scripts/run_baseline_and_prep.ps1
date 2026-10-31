param(
    [switch]$Cpu
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
& (Join-Path $ProjectRoot "venv\Scripts\Activate.ps1")

Write-Host ""
Write-Host "=== Step 1: Start federation windows ===" -ForegroundColor Cyan
if ($Cpu) {
    & "$PSScriptRoot\start_federation.ps1" -Cpu
} else {
    & "$PSScriptRoot\start_federation.ps1"
}

Write-Host ""
Write-Host "=== Waiting 15 seconds for SuperNodes to be ready ===" -ForegroundColor Yellow
Write-Host "Watch the three SuperNode windows and wait until each shows it is connected."
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "=== Step 2: Run baseline seed 42 - 10 rounds ===" -ForegroundColor Cyan
& "$PSScriptRoot\run_experiments.ps1" -Mode baseline -Seeds @(42)

Write-Host ""
Write-Host "=== Step 3: Regenerate all result files ===" -ForegroundColor Cyan
$env:MEDIFED_ROOT = $ProjectRoot
$env:PYTHONUTF8 = "1"
& (Join-Path $ProjectRoot "venv\Scripts\python.exe") scripts\demo_prep.py

Write-Host ""
Write-Host "=== Done. Run scripts\run_app.ps1 to start the dashboard ===" -ForegroundColor Green
