param(
    [ValidateSet("baseline", "personalized")]
    [string]$Mode = "baseline",
    [int[]]$Seeds = @(42, 43, 44)
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$env:MEDIFED_ROOT = $ProjectRoot

$oldConsoleEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

try {
    & (Join-Path $ProjectRoot "venv\Scripts\Activate.ps1")

    $superLink = Get-NetTCPConnection -LocalPort 9093 -State Listen -ErrorAction SilentlyContinue
    if (-not $superLink) {
        throw "Start the federation first: scripts\start_federation.ps1"
    }

    $pyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
    $rounds = 20
    if (Test-Path $pyprojectPath) {
        $parsed = Get-Content $pyprojectPath -Raw
        if ($parsed -match 'num-server-rounds\s*=\s*(\d+)') {
            $rounds = [int]$Matches[1]
        }
    }

    foreach ($seed in $Seeds) {
        $runName = "{0}_seed{1}" -f $Mode, $seed
        $runDirectory = Join-Path $ProjectRoot "results\runs\$runName"
        $configPath = Join-Path $runDirectory "run_config.toml"
        $logPath = Join-Path $runDirectory "server_log.txt"
        $checkpointPath = Join-Path $ProjectRoot "checkpoints\${runName}_global.pt"
        $bnDir = Join-Path $ProjectRoot "checkpoints\$runName"

        if (Test-Path $runDirectory) {
            Get-ChildItem -Path $runDirectory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        }
        if (Test-Path $bnDir) {
            Get-ChildItem -Path $bnDir -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        }
        if (Test-Path $checkpointPath) {
            Remove-Item -Path $checkpointPath -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

        $configLines = @(
            "seed = $seed",
            "mode = `"$Mode`"",
            "run-name = `"$runName`"",
            "num-server-rounds = $rounds"
        )
        [System.IO.File]::WriteAllLines(
            $configPath,
            $configLines,
            [System.Text.UTF8Encoding]::new($false)
        )

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host "Running: $runName"
        & cmd.exe /c "flwr run . local-deployment --stream --run-config `"$configPath`" 2>&1" | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "Federation run failed for $runName"
        }
        $timer.Stop()

        $complete = $false
        if (Test-Path $checkpointPath) {
            $homeRows = 0
            foreach ($name in @("hospital1_rounds.csv", "hospital2_rounds.csv", "hospital3_rounds.csv")) {
                $csvPath = Join-Path $runDirectory $name
                if (Test-Path $csvPath) {
                    $rows = Get-Content $csvPath | Measure-Object -Line
                    $homeRows += [int]$rows.Lines - 1
                }
            }
            if ($homeRows -eq (3 * 2 * $rounds)) {
                $complete = $true
            }
        }

        if (-not $complete) {
            throw "RUN FAILED: $runName is incomplete. Missing checkpoint or round rows."
        }

        Write-Host "RUN OK: $runName"
        Write-Host ("Elapsed: {0}" -f $timer.Elapsed)
    }
}
finally {
    [Console]::OutputEncoding = $oldConsoleEncoding
    $env:PYTHONUTF8 = ""
    $env:PYTHONIOENCODING = ""
}