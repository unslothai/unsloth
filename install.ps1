# Unsloth Studio Installer for Windows PowerShell
# Usage:  irm https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1 | iex
# Local:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\install.ps1

function Install-UnslothStudio {
    $ErrorActionPreference = "Stop"

    $VenvName = "unsloth_studio"
    $PythonVersion = "3.13"

    Write-Host ""
    Write-Host "========================================="
    Write-Host "   Unsloth Studio Installer (Windows)"
    Write-Host "========================================="
    Write-Host ""

    # ── Helper: refresh PATH from registry (preserving current session entries) ──
    function Refresh-SessionPath {
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$machine;$user;$env:Path"
    }

    # ── Check winget ──
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Error: winget is not available." -ForegroundColor Red
        Write-Host "       Install it from https://aka.ms/getwinget" -ForegroundColor Yellow
        Write-Host "       or install Python $PythonVersion and uv manually, then re-run." -ForegroundColor Yellow
        return
    }

    # ── Install Python 3.13 if no compatible Python found ──
    # setup.ps1 accepts Python 3.11-3.13; we install 3.13 if nothing compatible is found
    $NeedPython = $true
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyVer = python --version 2>&1
        if ($pyVer -match "Python 3\.1[1-3]\.") {
            Write-Host "==> Python already installed: $pyVer"
            $NeedPython = $false
        }
    }
    if ($NeedPython) {
        Write-Host "==> Installing Python ${PythonVersion}..."
        winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
        Refresh-SessionPath
    }

    # ── Install uv if not present ──
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "==> Installing uv package manager..."
        winget install --id=astral-sh.uv -e --accept-package-agreements --accept-source-agreements
        Refresh-SessionPath
        # Fallback: if winget didn't put uv on PATH, try the PowerShell installer
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "    Trying alternative uv installer..."
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            Refresh-SessionPath
        }
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Error: uv could not be installed." -ForegroundColor Red
        Write-Host "       Install it from https://docs.astral.sh/uv/" -ForegroundColor Yellow
        return
    }

    # ── Create venv (skip if it already exists) ──
    $VenvPython = Join-Path $VenvName "Scripts\python.exe"
    if (-not (Test-Path $VenvName)) {
        Write-Host "==> Creating Python ${PythonVersion} virtual environment (${VenvName})..."
        uv venv $VenvName --python $PythonVersion
    } else {
        Write-Host "==> Virtual environment ${VenvName} already exists, skipping creation."
    }

    # ── Install unsloth directly into the venv (no activation needed) ──
    Write-Host "==> Installing unsloth (this may take a few minutes)..."
    uv pip install --python $VenvPython unsloth --torch-backend=auto

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, Node.js, and other dependencies automatically via winget.
    Write-Host "==> Running unsloth studio setup..."
    $UnslothExe = Join-Path $VenvName "Scripts\unsloth.exe"
    & $UnslothExe studio setup

    Write-Host ""
    Write-Host "========================================="
    Write-Host "   Unsloth Studio installed!"
    Write-Host "========================================="
    Write-Host ""
    Write-Host "  To launch, run:"
    Write-Host ""
    Write-Host "    .\${VenvName}\Scripts\activate"
    Write-Host "    unsloth studio -H 0.0.0.0 -p 8888"
    Write-Host ""
}

Install-UnslothStudio
