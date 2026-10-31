param(
    [int]$Port = 8000
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:MEDIFED_ROOT = $projectRoot
$activateScript = Join-Path $projectRoot "venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    throw "Virtual environment not found at $activateScript"
}

. $activateScript
Set-Location $projectRoot
Write-Host "MediFed is available at http://127.0.0.1:$Port"
python -m uvicorn app.server:app --host 127.0.0.1 --port $Port