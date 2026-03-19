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

    # ── Install Python if no compatible version (3.11-3.13) found ──
    $DetectedPythonVersion = ""
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyVer = python --version 2>&1
        if ($pyVer -match "Python (3\.1[1-3])\.\d+") {
            Write-Host "==> Python already installed: $pyVer"
            $DetectedPythonVersion = $Matches[1]
        }
    }
    if (-not $DetectedPythonVersion) {
        Write-Host "==> Installing Python ${PythonVersion}..."
        winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
        Refresh-SessionPath
        if ($LASTEXITCODE -ne 0) {
            # winget returns non-zero for "already installed" -- only fail if python is truly missing
            if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
                Write-Host "[ERROR] Python installation failed (exit code $LASTEXITCODE)" -ForegroundColor Red
                return
            }
        }
        $DetectedPythonVersion = $PythonVersion
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

    # ── Create venv (skip if it already exists and has a valid interpreter) ──
    $VenvPython = Join-Path $VenvName "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        if (Test-Path $VenvName) { Remove-Item -Recurse -Force $VenvName }
        Write-Host "==> Creating Python ${DetectedPythonVersion} virtual environment (${VenvName})..."
        uv venv $VenvName --python $DetectedPythonVersion
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to create virtual environment (exit code $LASTEXITCODE)" -ForegroundColor Red
            return
        }
    } else {
        Write-Host "==> Virtual environment ${VenvName} already exists, skipping creation."
    }

    # ── Install unsloth directly into the venv (no activation needed) ──
    Write-Host "==> Installing unsloth (this may take a few minutes)..."
    uv pip install --python $VenvPython unsloth --torch-backend=auto
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install unsloth (exit code $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, Node.js, and other dependencies automatically via winget.
    Write-Host "==> Running unsloth studio setup..."
    $UnslothExe = Join-Path $VenvName "Scripts\unsloth.exe"
    & $UnslothExe studio setup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] unsloth studio setup failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "========================================="
    Write-Host "   Unsloth Studio installed!"
    Write-Host "   Launching Unsloth Studio..."
    Write-Host "========================================="
    Write-Host ""
    $UnslothExe = Join-Path $VenvName "Scripts\unsloth.exe"
    & $UnslothExe studio -H 0.0.0.0 -p 8888
}

Install-UnslothStudio
