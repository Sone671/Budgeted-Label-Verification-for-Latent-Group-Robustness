param(
    [string]$DestinationRoot = "G:\latent_group_verification_mvp\data\dopanim_source"
)

$ErrorActionPreference = "Stop"
$url = "https://zenodo.org/api/records/14016659/files/train.zip/content"
$partial = Join-Path $DestinationRoot "train.zip.partial"
$final = Join-Path $DestinationRoot "train.zip"
$expectedMd5 = "95e628ed85927c1ac82d7680072d317b"

New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
while (-not (Test-Path -LiteralPath $final)) {
    & curl.exe --fail --location --continue-at - --retry 2 --retry-delay 10 --output $partial $url
    if ((Test-Path -LiteralPath $partial)) {
        $hashLines = certutil.exe -hashfile $partial MD5 2>$null
        $actualMd5 = (($hashLines | Select-String -Pattern '^[0-9A-Fa-f]{32}$' | Select-Object -First 1).Line).ToLower()
        if ($actualMd5 -eq $expectedMd5) {
            Move-Item -LiteralPath $partial -Destination $final
            Write-Output "[dopanim-download-complete] $final"
            break
        }
    }
    Start-Sleep -Seconds 10
}
