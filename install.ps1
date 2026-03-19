# Unsloth Studio Installer for Windows PowerShell
# Usage: irm https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

$VenvName = "unsloth_studio"
$PythonVersion = "3.13"

Write-Host "==> Installing Unsloth Studio"

# 1. Install Python 3.13 if not present
if (-not (Get-Command python -ErrorAction SilentlyContinue) -or
    -not ((python --version 2>&1) -match "3\.13")) {
    Write-Host "==> Installing Python ${PythonVersion}..."
    winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# 2. Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing uv..."
    winget install --id=astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# 3. Create venv
Write-Host "==> Creating Python ${PythonVersion} virtual environment (${VenvName})..."
uv venv $VenvName --python $PythonVersion

# 4. Activate venv
$ActivateScript = Join-Path $VenvName "Scripts\activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Error "Error: Could not find activation script at $ActivateScript"
    exit 1
}

# 5. Install unsloth
Write-Host "==> Installing unsloth..."
uv pip install unsloth --torch-backend=auto

# 6. Run studio setup
Write-Host "==> Running unsloth studio setup..."
unsloth studio setup

Write-Host ""
Write-Host "==> Unsloth Studio is ready!"
Write-Host "    To launch, run:"
Write-Host "      .\${VenvName}\Scripts\activate"
Write-Host "      unsloth studio -H 0.0.0.0 -p 8888"
