param(
    [string]$Destination = "dist/celeba_repairvalue_seed20_29_server.zip"
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([System.IO.Path]::IsPathRooted($Destination) -or $Destination -match "(^|[\\/])\.\.([\\/]|$)") {
    throw "Destination must be a relative path below the repository root."
}
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $root $Destination))
if (-not $destinationPath.StartsWith($root + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Destination resolved outside the repository root."
}
$destinationDirectory = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

$items = @(
    "pyproject.toml",
    "requirements.txt",
    "LICENSE",
    "README.md",
    "CELEBA_REPAIRVALUE_SERVER_README.md",
    "configs/celeba_e2e_repairvalue_seed20_29.yaml",
    "src/robust_verify",
    "scripts/run_celeba_repairvalue_server.py",
    "scripts/run_celeba_repairvalue_server.sh",
    "tests/test_celeba_repairvalue_confirmatory.py",
    "tests/test_end_to_end_smoke.py",
    "tests/test_modern_baselines.py",
    "tests/test_absolute_utility.py"
)

$staging = Join-Path ([System.IO.Path]::GetTempPath()) (
    "celeba_repairvalue_server_" + [guid]::NewGuid().ToString("N")
)
try {
    New-Item -ItemType Directory -Path $staging | Out-Null
    foreach ($item in $items) {
        $source = Join-Path $root $item
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Missing package item: $item"
        }
        $target = Join-Path $staging $item
        $targetParent = Split-Path -Parent $target
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Recurse
    }

    Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $staging -Recurse -File -Filter "*.pyc" |
        Remove-Item -Force

    $manifestLines = Get-ChildItem -LiteralPath $staging -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            $relative = [System.IO.Path]::GetRelativePath($staging, $_.FullName).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relative"
        }
    [System.IO.File]::WriteAllLines(
        (Join-Path $staging "PACKAGE_MANIFEST.sha256"),
        $manifestLines,
        [System.Text.UTF8Encoding]::new($false)
    )

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

$zipHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
$hashPath = $destinationPath + ".sha256"
[System.IO.File]::WriteAllText(
    $hashPath,
    "$zipHash  $([System.IO.Path]::GetFileName($destinationPath))`n",
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output "Created $destinationPath"
Write-Output "SHA256 $zipHash"
