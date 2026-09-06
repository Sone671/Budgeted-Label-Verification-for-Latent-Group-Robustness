param(
    [string]$Destination = "dist/real_noise_end_to_end_replay.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$root = [System.IO.Path]::GetFullPath($root)
if ([System.IO.Path]::IsPathRooted($Destination) -or $Destination -match "(^|[\\/])\.\.([\\/]|$)") {
    throw "Destination must be a path under the repository root."
}
$destinationPath = Join-Path $root $Destination
$destinationDir = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

$items = @(
    "pyproject.toml",
    "README.md",
    "docs/END_TO_END_REAL_NOISE_RUNBOOK.md",
    "configs/cifar10n_end_to_end_confirmatory.yaml",
    "configs/waterbirds_end_to_end_confirmatory.yaml",
    "configs/waterbirds_end_to_end_stress.yaml",
    "src/robust_verify",
    "tests/test_cifar_n_replay.py",
    "tests/test_end_to_end_smoke.py"
)

$paths = foreach ($item in $items) {
    $path = Join-Path $root $item
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing package item: $item" }
    $path
}

 $staging = Join-Path $env:TEMP ("real_noise_end_to_end_replay_" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    foreach ($item in $items) {
        $source = Join-Path $root $item
        $target = Join-Path $staging $item
        $targetParent = Split-Path -Parent $target
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Recurse
    }
    Get-ChildItem -Path $staging -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Get-ChildItem -Path $staging -Recurse -File -Filter "*.pyc" |
        Remove-Item -Force
    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Force
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $destinationPath -CompressionLevel Optimal
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
Write-Output "Created $destinationPath"
