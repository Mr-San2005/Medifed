param(
    [ValidateSet("baseline", "personalized")]
    [string]$Mode = "baseline",
    [int[]]$Seeds = @(42, 43, 44)
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$env:MEDIFED_ROOT = $ProjectRoot
& (Join-Path $ProjectRoot "venv\Scripts\Activate.ps1")

$superLink = Get-NetTCPConnection -LocalPort 9093 -State Listen -ErrorAction SilentlyContinue
if (-not $superLink) {
    Write-Host "Start the federation first: scripts\start_federation.ps1"
    exit 1
}

foreach ($seed in $Seeds) {
    $runName = "{0}_seed{1}" -f $Mode, $seed
    $runDirectory = Join-Path $ProjectRoot "results\runs\$runName"
    $configPath = Join-Path $runDirectory "run_config.toml"
    $logPath = Join-Path $runDirectory "server_log.txt"
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

    @(
        "seed = $seed"
        "mode = `"$Mode`""
        "run-name = `"$runName`""
    ) | Set-Content -Path $configPath -Encoding ascii

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    flwr run . local-deployment --stream --run-config $configPath 2>&1 |
        Tee-Object -FilePath $logPath
    $runExitCode = $LASTEXITCODE
    $timer.Stop()
    Write-Host ("Elapsed: {0}" -f $timer.Elapsed)

    if ($runExitCode -ne 0) {
        exit $runExitCode
    }
}