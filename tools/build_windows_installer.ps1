param(
    [switch]$InstallPyInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$InstallerSpecDir = Join-Path $RepoRoot "build\installer_spec"
$IconPath = Join-Path $RepoRoot "logo\logo.ico"
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    powershell -ExecutionPolicy Bypass -File .\tools\build_windows_app.ps1 -InstallPyInstaller
} else {
    powershell -ExecutionPolicy Bypass -File .\tools\build_windows_app.ps1
}

$zipPath = Join-Path $RepoRoot "sleepy-Windows.zip"
$installerSource = Join-Path $RepoRoot "dist\sleepy-Setup.exe"
$installerTarget = Join-Path $RepoRoot "sleepy-Setup.exe"

New-Item -ItemType Directory -Path $InstallerSpecDir -Force | Out-Null

python -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --windowed `
    --specpath $InstallerSpecDir `
    --name sleepy-Setup `
    --icon $IconPath `
    --add-data "$zipPath;." `
    tools\install_sleepy.py

Copy-Item -LiteralPath $installerSource -Destination $installerTarget -Force

Write-Host ""
Write-Host "Built installer:"
Write-Host "  $installerTarget"
Write-Host ""
Write-Host "Running this installer will install sleepy to:"
Write-Host "  %LOCALAPPDATA%\Programs\sleepy"
Write-Host "and create:"
Write-Host "  Desktop\sleepy.lnk"
