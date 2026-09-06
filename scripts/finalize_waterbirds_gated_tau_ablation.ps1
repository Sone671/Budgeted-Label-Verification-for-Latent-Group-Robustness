$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = Join-Path $projectRoot 'outputs\waterbirds_gated_tau_ablation_seed40_59_frozen'
$completeMarker = Join-Path $outputRoot 'ABLATION_COMPLETE.json'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

while ($true) {
    $ready = $false
    if (Test-Path -LiteralPath $completeMarker) {
        try {
            $metadata = Get-Content -LiteralPath $completeMarker -Raw | ConvertFrom-Json
            $ready = ($metadata.seeds.Count -eq 20 -and [int]$metadata.rows -ge 80)
        } catch {
            $ready = $false
        }
    }
    if ($ready) {
        break
    }
    Start-Sleep -Seconds 60
}

Set-Location -LiteralPath $projectRoot
& $pythonPath scripts\run_waterbirds_gated_tau_ablation_frozen.py `
    --num-workers 0
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonPath scripts\analyze_waterbirds_gated_tau_ablation.py
exit $LASTEXITCODE
