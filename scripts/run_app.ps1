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

# Free the port if something is already bound to it.
$existing = netstat -ano | Select-String ":$Port\s" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique
foreach ($pid in $existing) {
    if ($pid -match '^\d+$') {
        Write-Host "Killing PID $pid on port $Port"
        taskkill /PID $pid /F 2>$null
    }
}

Write-Host "MediFed is available at http://127.0.0.1:$Port"
python -m uvicorn app.server:app --host 127.0.0.1 --port $Port