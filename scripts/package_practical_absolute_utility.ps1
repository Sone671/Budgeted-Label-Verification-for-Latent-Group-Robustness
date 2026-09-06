param(
    [string]$Destination = "dist/practical_absolute_utility_audit.zip"
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
    "P0_PROTOCOL.md",
    "docs/PRACTICAL_ABSOLUTE_UTILITY.md",
    "src/robust_verify/__init__.py",
    "src/robust_verify/absolute_utility.py",
    "scripts/p0_absolute_utility.py",
    "scripts/package_practical_absolute_utility.ps1",
    "tests/test_absolute_utility.py"
)

foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $item))) {
        throw "Missing package item: $item"
    }
}

$staging = Join-Path $env:TEMP ("practical_absolute_utility_" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    $manifest = @()
    foreach ($item in $items) {
        $source = Join-Path $root $item
        $target = Join-Path $staging $item
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Recurse
        $manifest += [PSCustomObject]@{
            path = $item.Replace("\", "/")
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $source).Length
        }
    }
    $manifestPath = Join-Path $staging "PRACTICAL_ABSOLUTE_UTILITY_PACKAGE_MANIFEST.json"
    [PSCustomObject]@{
        package = "practical_absolute_utility_audit"
        files = $manifest
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8
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
