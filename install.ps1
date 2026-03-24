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

    function New-StudioShortcuts {
        param(
            [Parameter(Mandatory = $true)][string]$UnslothExePath
        )

        if (-not (Test-Path $UnslothExePath)) {
            Write-Host "[WARN] Cannot create shortcuts: unsloth.exe not found at $UnslothExePath" -ForegroundColor Yellow
            return
        }
        # Persist an absolute path in launcher scripts so shortcut working
        # directory changes do not break process startup.
        $UnslothExePath = (Resolve-Path $UnslothExePath).Path

        $appDir = Join-Path $env:LOCALAPPDATA "Unsloth Studio"
        $launcherPs1 = Join-Path $appDir "launch-studio.ps1"
        $launcherVbs = Join-Path $appDir "launch-studio.vbs"
        $desktopLink = Join-Path ([Environment]::GetFolderPath("Desktop")) "Unsloth Studio.lnk"
        $startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
        $startMenuLink = Join-Path $startMenuDir "Unsloth Studio.lnk"
        $iconPath = Join-Path $appDir "unsloth.ico"
        $bundledIcon = $null
        if ($PSScriptRoot -and $PSScriptRoot.Trim()) {
            $bundledIcon = Join-Path $PSScriptRoot "studio\frontend\public\unsloth.ico"
        }
        $iconUrl = "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth.ico"

        if (-not (Test-Path $appDir)) {
            New-Item -ItemType Directory -Path $appDir -Force | Out-Null
        }

        $launcherContent = @"
`$ErrorActionPreference = 'Stop'
`$basePort = 8888
`$maxPortOffset = 19
`$timeoutSec = 60
`$pollIntervalMs = 1000

function Test-StudioHealth {
    param([Parameter(Mandatory = `$true)][int]`$Port)
    try {
        `$url = "http://127.0.0.1:`$Port/api/health"
        `$resp = Invoke-RestMethod -Uri `$url -TimeoutSec 1 -Method Get
        return (`$resp -and `$resp.status -eq 'healthy')
    } catch {
        return `$false
    }
}

function Get-CandidatePorts {
    # Fast path: only probe base port + currently listening ports in range.
    `$ports = @(`$basePort)
    try {
        `$maxPort = `$basePort + `$maxPortOffset
        `$listening = Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { `$_.LocalPort -ge `$basePort -and `$_.LocalPort -le `$maxPort } |
            Select-Object -ExpandProperty LocalPort -Unique |
            Sort-Object
        foreach (`$p in `$listening) {
            if (`$ports -notcontains `$p) { `$ports += `$p }
        }
    } catch {
        # Fallback when Get-NetTCPConnection is unavailable/restricted.
        for (`$offset = 1; `$offset -le `$maxPortOffset; `$offset++) {
            `$ports += (`$basePort + `$offset)
        }
    }
    return `$ports
}

function Find-HealthyStudioPort {
    foreach (`$candidate in (Get-CandidatePorts)) {
        if (Test-StudioHealth -Port `$candidate) {
            return `$candidate
        }
    }
    return `$null
}

# If Studio is already healthy on any expected port, just open it and exit.
`$existingPort = Find-HealthyStudioPort
if (`$existingPort) {
    Start-Process "http://localhost:`$existingPort"
    exit 0
}

`$powershellExe = Join-Path `$env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
`$studioCommand = "& `"$UnslothExePath`" studio -H 0.0.0.0 -p `$basePort"
`$launchArgs = @(
    '-NoExit',
    '-ExecutionPolicy',
    'Bypass',
    '-Command',
    `$studioCommand
)

try {
    `$proc = Start-Process -FilePath `$powershellExe -ArgumentList `$launchArgs -WorkingDirectory `$env:USERPROFILE -PassThru
} catch {
    `$msg = "Could not launch Unsloth Studio terminal.`n`nError: `$(`$_.Exception.Message)"
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [System.Windows.Forms.MessageBox]::Show(`$msg, 'Unsloth Studio') | Out-Null
    } catch {}
    exit 1
}

`$browserOpened = `$false
`$deadline = (Get-Date).AddSeconds(`$timeoutSec)
while ((Get-Date) -lt `$deadline) {
    `$healthyPort = Find-HealthyStudioPort
    if (`$healthyPort) {
        Start-Process "http://localhost:`$healthyPort"
        `$browserOpened = `$true
        break
    }
    if (`$proc.HasExited) { break }
    Start-Sleep -Milliseconds `$pollIntervalMs
}
if (-not `$browserOpened -and `$proc.HasExited) {
    `$msg = "Unsloth Studio exited before becoming healthy. Check terminal output for errors."
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [System.Windows.Forms.MessageBox]::Show(`$msg, 'Unsloth Studio') | Out-Null
    } catch {}
}
exit 0
"@

        # Use Unicode-safe encoding: launcher paths can contain non-ASCII characters.
        Set-Content -Path $launcherPs1 -Value $launcherContent -Encoding UTF8 -Force
        $vbsContent = @"
Set shell = CreateObject("WScript.Shell")
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""$launcherPs1"""
shell.Run cmd, 0, False
"@
        # WSH handles UTF-16LE reliably for .vbs files with non-ASCII paths.
        Set-Content -Path $launcherVbs -Value $vbsContent -Encoding Unicode -Force

        # Prefer bundled icon from local clone/dev installs.
        # If not available, best-effort download from raw GitHub.
        # We only attach the icon if the resulting file has a valid ICO header.
        $hasValidIcon = $false
        if ($bundledIcon -and (Test-Path $bundledIcon)) {
            try {
                Copy-Item -Path $bundledIcon -Destination $iconPath -Force
            } catch { }
        } elseif (-not (Test-Path $iconPath)) {
            try {
                Invoke-WebRequest -Uri $iconUrl -OutFile $iconPath -UseBasicParsing
            } catch { }
        }

        if (Test-Path $iconPath) {
            try {
                $bytes = [System.IO.File]::ReadAllBytes($iconPath)
                if (
                    $bytes.Length -ge 4 -and
                    $bytes[0] -eq 0 -and
                    $bytes[1] -eq 0 -and
                    $bytes[2] -eq 1 -and
                    $bytes[3] -eq 0
                ) {
                    $hasValidIcon = $true
                } else {
                    Remove-Item $iconPath -Force -ErrorAction SilentlyContinue
                }
            } catch {
                Remove-Item $iconPath -Force -ErrorAction SilentlyContinue
            }
        }

        $wscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
        $shortcutArgs = "//B //Nologo `"$launcherVbs`""

        try {
            $wshell = New-Object -ComObject WScript.Shell
            foreach ($linkPath in @($desktopLink, $startMenuLink)) {
                if (-not $linkPath -or [string]::IsNullOrWhiteSpace($linkPath)) { continue }
                try {
                    $shortcut = $wshell.CreateShortcut($linkPath)
                    $shortcut.TargetPath = $wscriptExe
                    $shortcut.Arguments = $shortcutArgs
                    $shortcut.WorkingDirectory = $appDir
                    $shortcut.Description = "Launch Unsloth Studio"
                    if ($hasValidIcon) {
                        $shortcut.IconLocation = "$iconPath,0"
                    }
                    $shortcut.Save()
                } catch {
                    Write-Host "[WARN] Could not create shortcut at ${linkPath}: $($_.Exception.Message)" -ForegroundColor Yellow
                }
            }
            Write-Host "[OK] Created Unsloth Studio desktop and Start menu shortcuts" -ForegroundColor Green
        } catch {
            Write-Host "[WARN] Shortcut creation unavailable: $($_.Exception.Message)" -ForegroundColor Yellow
        }
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

    # ── Install Python if no compatible version (3.11-3.13) found ──
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

    # ── Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) ──
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

    # ── Choose the correct PyTorch index URL based on driver CUDA version ──
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

    # ── Install PyTorch first, then unsloth separately ──
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
    uv pip install --python $VenvPython --upgrade-package unsloth unsloth
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

    New-StudioShortcuts -UnslothExePath $UnslothExe

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
