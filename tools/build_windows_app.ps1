param(
    [switch]$InstallPyInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    python -m pip install pyinstaller
}

python -m PyInstaller --clean --noconfirm OnlineSleepScore.spec

Write-Host ""
Write-Host "Built app:"
Write-Host "  $RepoRoot\dist\OnlineSleepScore\OnlineSleepScore.exe"
Write-Host ""
Write-Host "To distribute, copy the whole folder:"
Write-Host "  $RepoRoot\dist\OnlineSleepScore"
