# Starts the Flower deployment runtime for the MediFed FedAvg baseline.
#
# Opens four new PowerShell windows:
#   1 SuperLink  (aggregation server)
#   3 SuperNodes (hospital clients, partition-id 0/1/2)
#
# The project root is taken from this script's own location, so the script
# works from any current folder. Each window sets MEDIFED_ROOT so the client
# and server code can find data/, results/ and checkpoints/ even though Flower
# copies the app under the user profile.
#
# SuperLink is started first, then an 8-second wait, then the three SuperNodes.
# Close each window manually with Ctrl+C when the run is finished.
# This script does not execute "flwr run".

param(
    [switch]$Cpu
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ActivateScript = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"

function Get-PortsInUse {
    $portsInUse = @()
    foreach ($port in 9092..9096) {
        if (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) {
            $portsInUse += $port
        }
    }
    return $portsInUse
}

$occupiedPorts = @(Get-PortsInUse)
if ($occupiedPorts.Count -gt 0) {
    Write-Host "Ports in use: $($occupiedPorts -join ', ')"
    Write-Host "Run scripts\stop_federation.ps1 first"
    exit 1
}

function Start-FederationWindow {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandLine,
        [switch]$SetCpu
    )
    $deviceSetting = ""
    if ($SetCpu) {
        $deviceSetting = "`$env:MEDIFED_DEVICE = 'cpu'`n"
    }
    $inner = @"
`$env:MEDIFED_ROOT = '$ProjectRoot'
$deviceSetting& '$ActivateScript'
Set-Location '$ProjectRoot'
$CommandLine
"@
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($inner)
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-EncodedCommand",
        $encodedCommand
    )
}

Start-FederationWindow "flower-superlink --insecure --disable-runtime-dependency-installation"
Start-Sleep -Seconds 8
Start-FederationWindow -SetCpu:$Cpu "flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config `"partition-id=0 num-partitions=3`" --port 9094"
Start-FederationWindow -SetCpu:$Cpu "flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config `"partition-id=1 num-partitions=3`" --port 9095"
Start-FederationWindow -SetCpu:$Cpu "flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config `"partition-id=2 num-partitions=3`" --port 9096"

Write-Host "Wait until all three SuperNode windows show 'Starting Flower SuperExec', then run:  flwr run . local-deployment --stream"
