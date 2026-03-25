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

    # -- Helper: refresh PATH from registry (deduplicating entries) --
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

    # -- Check winget --
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Error: winget is not available." -ForegroundColor Red
        Write-Host "       Install it from https://aka.ms/getwinget" -ForegroundColor Yellow
        Write-Host "       or install Python $PythonVersion and uv manually, then re-run." -ForegroundColor Yellow
        return
    }

    # -- Helper: detect a working Python 3.11-3.13 on the system --
    # Returns the version string (e.g. "3.13") or "" if none found.
    # Uses try-catch + stderr redirection so that App Execution Alias stubs
    # (WindowsApps) and other non-functional executables are probed safely
    # without triggering $ErrorActionPreference = "Stop".
    #
    # Skips Anaconda/Miniconda Python: conda-bundled CPython ships modified
    # DLL search paths that break torch's c10.dll loading on Windows.
    # Standalone CPython (python.org, winget, uv) does not have this issue.
    #
    # NOTE: A venv created from conda Python inherits conda's base_prefix
    # even if the venv path does not contain "conda". We check both the
    # executable path AND sys.base_prefix to catch this.
    $script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'

    function Test-IsCondaPython {
        param([string]$Exe)
        if ($Exe -match $script:CondaSkipPattern) { return $true }
        try {
            $basePrefix = (& $Exe -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
            if ($basePrefix -match $script:CondaSkipPattern) { return $true }
        } catch { }
        return $false
    }

    # Returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    # The resolved Path is passed to `uv venv --python` to prevent uv from
    # re-resolving the version string back to a conda interpreter.
    function Find-CompatiblePython {
        # Try the Python Launcher first (most reliable on Windows)
        # py.exe resolves to the standard CPython install, not conda.
        $pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if ($pyLauncher -and $pyLauncher.Source -notmatch $script:CondaSkipPattern) {
            foreach ($minor in @("3.13", "3.12", "3.11")) {
                try {
                    $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") {
                        $ver = $Matches[1]
                        # Resolve the actual executable path and verify it is not conda-based
                        $resolvedExe = (& $pyLauncher.Source "-$minor" -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
                        if ($resolvedExe -and (Test-Path $resolvedExe) -and -not (Test-IsCondaPython $resolvedExe)) {
                            return @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # Try python3 / python via Get-Command -All to look past stubs that
        # might shadow a real Python further down PATH.
        # Skip WindowsApps entries: the App Execution Alias stubs live there
        # and can open the Microsoft Store as a side effect. Legitimate Store
        # Python is already detected via the py launcher above (Store packages
        # include py since Python 3.11).
        # Skip Anaconda/Miniconda: check both path and sys.base_prefix.
        foreach ($name in @("python3", "python")) {
            foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
                if (-not $cmd.Source) { continue }
                if ($cmd.Source -like "*\WindowsApps\*") { continue }
                if (Test-IsCondaPython $cmd.Source) { continue }
                try {
                    $out = & $cmd.Source --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") {
                        return @{ Version = $Matches[1]; Path = $cmd.Source }
                    }
                } catch {}
            }
        }
        return $null
    }

    # -- Install Python if no compatible version (3.11-3.13) found --
    # Find-CompatiblePython returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    $DetectedPython = Find-CompatiblePython
    if ($DetectedPython) {
        Write-Host "==> Python already installed: Python $($DetectedPython.Version)"
    }
    if (-not $DetectedPython) {
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
        $DetectedPython = Find-CompatiblePython

        if (-not $DetectedPython) {
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
            $DetectedPython = Find-CompatiblePython
        }

        if (-not $DetectedPython) {
            Write-Host "[ERROR] Python installation failed (exit code $wingetExit)" -ForegroundColor Red
            Write-Host "        Please install Python $PythonVersion manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-Host "        Then re-run this installer." -ForegroundColor Yellow
            return
        }
    }

    # -- Install uv if not present --
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

    # -- Create venv (skip if it already exists and has a valid interpreter) --
    # Pass the resolved executable path to uv so it does not re-resolve
    # a version string back to a conda interpreter.
    $VenvPython = Join-Path $VenvName "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        if (Test-Path $VenvName) { Remove-Item -Recurse -Force $VenvName }
        Write-Host "==> Creating Python $($DetectedPython.Version) virtual environment (${VenvName})..."
        uv venv $VenvName --python "$($DetectedPython.Path)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to create virtual environment (exit code $LASTEXITCODE)" -ForegroundColor Red
            return
        }
    } else {
        Write-Host "==> Virtual environment ${VenvName} already exists, skipping creation."
    }

    # -- Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) --
    $HasNvidiaSmi = $false
    $NvidiaSmiExe = $null
    try {
        $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvSmiCmd) {
            & $nvSmiCmd.Source 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $HasNvidiaSmi = $true; $NvidiaSmiExe = $nvSmiCmd.Source }
        }
    } catch {}
    if (-not $HasNvidiaSmi) {
        foreach ($p in @(
            "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "$env:SystemRoot\System32\nvidia-smi.exe"
        )) {
            if (Test-Path $p) {
                try {
                    & $p 2>&1 | Out-Null
                    if ($LASTEXITCODE -eq 0) { $HasNvidiaSmi = $true; $NvidiaSmiExe = $p; break }
                } catch {}
            }
        }
    }
    if ($HasNvidiaSmi) {
        Write-Host "[OK] NVIDIA GPU detected" -ForegroundColor Green
    } else {
        Write-Host "[WARN] No NVIDIA GPU detected. Studio will run in chat-only (GGUF) mode." -ForegroundColor Yellow
        Write-Host "       Training and GPU inference require an NVIDIA GPU with drivers installed." -ForegroundColor Yellow
        Write-Host "       https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    }

    # -- Choose the correct PyTorch index URL based on driver CUDA version --
    # Mirrors Get-PytorchCudaTag in setup.ps1.
    function Get-TorchIndexUrl {
        $baseUrl = "https://download.pytorch.org/whl"
        if (-not $NvidiaSmiExe) { return "$baseUrl/cpu" }
        try {
            $output = & $NvidiaSmiExe 2>&1 | Out-String
            if ($output -match 'CUDA Version:\s+(\d+)\.(\d+)') {
                $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                if ($major -ge 13)                    { return "$baseUrl/cu130" }
                if ($major -eq 12 -and $minor -ge 8)  { return "$baseUrl/cu128" }
                if ($major -eq 12 -and $minor -ge 6)  { return "$baseUrl/cu126" }
                if ($major -ge 12) { return "$baseUrl/cu124" }
                if ($major -ge 11) { return "$baseUrl/cu118" }
                return "$baseUrl/cpu"
            }
        } catch {}
        Write-Host "[WARN] Could not determine CUDA version from nvidia-smi, defaulting to cu126" -ForegroundColor Yellow
        return "$baseUrl/cu126"
    }
    $TorchIndexUrl = Get-TorchIndexUrl

    # -- Install PyTorch first, then unsloth separately --
    #
    # Why two steps?
    #   `uv pip install unsloth --torch-backend=cpu` on Windows resolves to
    #   unsloth==2024.8 (a pre-CLI release with no unsloth.exe) because the
    #   cpu-only solver cannot satisfy newer unsloth's dependencies.
    #   Installing torch first from the explicit CUDA index, then upgrading
    #   unsloth in a second step, avoids this solver dead-end.
    #
    # Why --upgrade-package instead of --upgrade?
    #   `--upgrade unsloth` re-resolves ALL dependencies including torch,
    #   pulling torch from default PyPI and stripping the +cuXXX suffix
    #   that step 1 installed (e.g. torch 2.5.1+cu124 -> 2.10.0 with no
    #   CUDA suffix).  `--upgrade-package unsloth` upgrades ONLY unsloth
    #   to the latest version while preserving the already-pinned torch
    #   CUDA wheels.  Missing dependencies (transformers, trl, peft, etc.)
    #   are still pulled in because they are new, not upgrades.
    #
    Write-Host "==> Installing PyTorch ($TorchIndexUrl)..."
    uv pip install --python $VenvPython torch torchvision torchaudio --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install PyTorch (exit code $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    Write-Host "==> Installing unsloth (this may take a few minutes)..."
    uv pip install --python $VenvPython --upgrade-package unsloth "unsloth>=2026.3.11"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install unsloth (exit code $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    # -- Run studio setup --
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
    $IsInteractive = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        Write-Host "==> Launching Unsloth Studio..."
        Write-Host ""
        $UnslothExe = Join-Path $VenvName "Scripts\unsloth.exe"
        & $UnslothExe studio -H 0.0.0.0 -p 8888
    } else {
        Write-Host "  To launch, run:"
        Write-Host ""
        Write-Host "    .\${VenvName}\Scripts\activate"
        Write-Host "    unsloth studio -H 0.0.0.0 -p 8888"
        Write-Host ""
    }
}

Install-UnslothStudio
