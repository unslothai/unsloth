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

    # ── Helper: refresh PATH from registry (deduplicating entries) ──
    function Refresh-SessionPath {
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $merged  = "$machine;$user;$env:Path"
        $seen    = @{}
        $unique  = @()
        foreach ($p in $merged -split ";") {
            $key = $p.TrimEnd("\").ToLowerInvariant()
            if ($key -and -not $seen.ContainsKey($key)) {
                $seen[$key] = $true
                $unique += $p
            }
        }
        $env:Path = $unique -join ";"
    }

    # ── Check winget ──
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Error: winget is not available." -ForegroundColor Red
        Write-Host "       Install it from https://aka.ms/getwinget" -ForegroundColor Yellow
        Write-Host "       or install Python $PythonVersion and uv manually, then re-run." -ForegroundColor Yellow
        return
    }

    # ── Helper: detect a working Python 3.11-3.13 on the system ──
    # Returns the version string (e.g. "3.13") or "" if none found.
    # Uses try-catch + stderr redirection so that App Execution Alias stubs
    # (WindowsApps) and other non-functional executables are probed safely
    # without triggering $ErrorActionPreference = "Stop".
    function Find-CompatiblePython {
        # Try the Python Launcher first (most reliable on Windows)
        $pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            foreach ($minor in @("3.13", "3.12", "3.11")) {
                try {
                    $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") { return $Matches[1] }
                } catch {}
            }
        }
        # Try python3 / python via Get-Command -All to look past stubs that
        # might shadow a real Python further down PATH.
        # Skip WindowsApps entries: the App Execution Alias stubs live there
        # and can open the Microsoft Store as a side effect. Legitimate Store
        # Python is already detected via the py launcher above (Store packages
        # include py since Python 3.11).
        foreach ($name in @("python3", "python")) {
            foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
                if (-not $cmd.Source) { continue }
                if ($cmd.Source -like "*\WindowsApps\*") { continue }
                try {
                    $out = & $cmd.Source --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") { return $Matches[1] }
                } catch {}
            }
        }
        return ""
    }

    # ── Install Python if no compatible version (3.11-3.13) found ──
    $DetectedPythonVersion = Find-CompatiblePython
    if ($DetectedPythonVersion) {
        Write-Host "==> Python already installed: Python $DetectedPythonVersion"
    }
    if (-not $DetectedPythonVersion) {
        Write-Host "==> Installing Python ${PythonVersion}..."
        $pythonPackageId = "Python.Python.$PythonVersion"
        # Temporarily lower ErrorActionPreference so that winget stderr
        # (progress bars, warnings) does not become a terminating error
        # on PowerShell 5.1 where native-command stderr is ErrorRecord.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            winget install -e --id $pythonPackageId --accept-package-agreements --accept-source-agreements
            $wingetExit = $LASTEXITCODE
        } catch { $wingetExit = 1 }
        $ErrorActionPreference = $prevEAP
        Refresh-SessionPath

        # Re-detect after install (PATH may have changed)
        $DetectedPythonVersion = Find-CompatiblePython

        if (-not $DetectedPythonVersion) {
            # Python still not functional after winget -- force reinstall.
            # This handles both real failures AND "already installed" codes where
            # winget thinks Python is present but it's not actually on PATH
            # (e.g. user partially uninstalled, or installed via a different method).
            Write-Host "    Python not found on PATH after winget. Retrying with --force..."
            $ErrorActionPreference = "Continue"
            try {
                winget install -e --id $pythonPackageId --accept-package-agreements --accept-source-agreements --force
                $wingetExit = $LASTEXITCODE
            } catch { $wingetExit = 1 }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath
            $DetectedPythonVersion = Find-CompatiblePython
        }

        if (-not $DetectedPythonVersion) {
            Write-Host "[ERROR] Python installation failed (exit code $wingetExit)" -ForegroundColor Red
            Write-Host "        Please install Python $PythonVersion manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-Host "        Then re-run this installer." -ForegroundColor Yellow
            return
        }
    }

    # ── Install uv if not present ──
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "==> Installing uv package manager..."
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { winget install --id=astral-sh.uv -e --accept-package-agreements --accept-source-agreements } catch {}
        $ErrorActionPreference = $prevEAP
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
    if (-not (Test-Path $UnslothExe)) {
        Write-Host "[ERROR] unsloth CLI was not installed correctly." -ForegroundColor Red
        Write-Host "        Expected: $UnslothExe" -ForegroundColor Yellow
        Write-Host "        This usually means an older unsloth version was installed that does not include the Studio CLI." -ForegroundColor Yellow
        Write-Host "        Try re-running the installer or see: https://github.com/unslothai/unsloth?tab=readme-ov-file#-quickstart" -ForegroundColor Yellow
        return
    }
    & $UnslothExe studio setup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] unsloth studio setup failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "========================================="
    Write-Host "   Unsloth Studio installed!"
    Write-Host "========================================="
    Write-Host ""

    # Launch studio automatically in interactive terminals;
    # in non-interactive environments (CI, Docker) just print instructions.
    $LaunchCmd = ".\${VenvName}\Scripts\unsloth.exe studio -H 0.0.0.0 -p 8888"
    $IsInteractive = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        Write-Host "==> Launching Unsloth Studio..."
        Write-Host "    To relaunch later, run:"
        Write-Host ""
        Write-Host "    $LaunchCmd"
        Write-Host ""
        & (Join-Path $VenvName "Scripts\unsloth.exe") studio -H 0.0.0.0 -p 8888
    } else {
        Write-Host "  To launch, run:"
        Write-Host ""
        Write-Host "    $LaunchCmd"
        Write-Host ""
    }
}

Install-UnslothStudio
