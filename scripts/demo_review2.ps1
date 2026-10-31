param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    throw "Virtual environment not found at $venvActivate"
}
& $venvActivate

$start = Get-Date
$steps = @(
    "python scripts/run_tests.py",
    "python scripts/evaluate_federated.py baseline_seed42",
    "python scripts/evaluate_federated.py personalized_seed42",
    "python scripts/summarize_runs.py",
    "python scripts/demo_predict.py"
)

foreach ($step in $steps) {
    Write-Host "`n=== $step ==="
    Read-Host "Press Enter to continue"
    Invoke-Expression $step
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $step"
        exit $LASTEXITCODE
    }
}

Write-Host "`n=== opening charts ==="
& python -c "import os; print('results/chart_methods_comparison.png' if os.path.exists('results/chart_methods_comparison.png') else 'MISSING')"
& python -c "import os; print('results/chart_rounds.png' if os.path.exists('results/chart_rounds.png') else 'MISSING')"

$elapsed = (Get-Date) - $start
Write-Host ("Elapsed: {0}" -f $elapsed)
