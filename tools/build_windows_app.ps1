param(
    [switch]$InstallPyInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $RepoRoot "dist"
$AppDir = Join-Path $DistDir "sleepy"
$ZipPath = Join-Path $RepoRoot "sleepy-Windows.zip"
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    python -m pip install pyinstaller
}

python -m PyInstaller --clean --noconfirm sleepy.spec

if (-not (Test-Path $AppDir)) {
    throw "Expected build output was not found: $AppDir"
}

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

$zipCreated = $false
for ($attempt = 1; $attempt -le 10; $attempt++) {
    try {
        Compress-Archive -Path $AppDir -DestinationPath $ZipPath -Force
        $zipCreated = $true
        break
    }
    catch {
        if ($attempt -eq 10) {
            throw
        }
        Start-Sleep -Seconds 2
    }
}

if (-not $zipCreated) {
    throw "Failed to create distribution zip: $ZipPath"
}

Write-Host ""
Write-Host "Built app:"
Write-Host "  $AppDir\sleepy.exe"
Write-Host ""
Write-Host "Folder for local testing:"
Write-Host "  $AppDir"
Write-Host ""
Write-Host "Zip for distribution:"
Write-Host "  $ZipPath"
