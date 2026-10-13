# Stops MediFed Flower runtime processes and reports any ports that remain occupied.

$processPattern = "flower-superlink|flower-supernode|flwr-serverapp|flwr-clientapp|flwr-superexec"
$flowerProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $processPattern }
)

if ($flowerProcesses.Count -eq 0) {
    Write-Host "No Flower federation processes found."
} else {
    $flowerProcesses |
        Select-Object ProcessId, Name, CommandLine |
        Format-Table -AutoSize
    foreach ($process in $flowerProcesses) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 2
$portsStillInUse = @(
    Get-NetTCPConnection -LocalPort 9092, 9093, 9094, 9095, 9096 -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty LocalPort -Unique
)
if ($portsStillInUse.Count -eq 0) {
    Write-Host "Ports 9092-9096 are free."
} else {
    Write-Host "Ports still in use: $($portsStillInUse -join ', ')"
}