#Requires -Version 5.1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
<#
.SYNOPSIS
    Full environment setup for Unsloth Studio on Windows (bundled version).
.DESCRIPTION
    Always installs Node.js if needed. When running from pip install:
    skips frontend build (already bundled). When running from git repo:
    full setup including frontend build.
    Supports NVIDIA GPU (full training + inference) and CPU-only (GGUF chat mode).
.NOTES
    Usage: powershell -ExecutionPolicy Bypass -File setup.ps1
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $ScriptDir

# Detect if running from pip install (no frontend/ dir in studio)
$FrontendDir = Join-Path $ScriptDir "frontend"
$OxcValidatorDir = Join-Path $ScriptDir "backend\core\data_recipe\oxc-validator"
$IsPipInstall = -not (Test-Path $FrontendDir)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

# Reload ALL environment variables from registry.
# Picks up changes made by installers (winget, msi, etc.) including
# Path, CUDA_PATH, CUDA_PATH_V*, and any other vars they set.
function Refresh-Environment {
    foreach ($level in @('Machine', 'User')) {
        $vars = [System.Environment]::GetEnvironmentVariables($level)
        foreach ($key in $vars.Keys) {
            if ($key -eq 'Path') { continue }
            Set-Item -Path "Env:$key" -Value $vars[$key] -ErrorAction SilentlyContinue
        }
    }
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"
}

# Find nvcc on PATH, CUDA_PATH, or standard toolkit dirs.
# Returns the path to nvcc.exe, or $null if not found.
function Find-Nvcc {
    param([string]$MaxVersion = "")

    # If MaxVersion is set, we need to find a toolkit <= that version.
    # CUDA toolkits install side-by-side under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\

    $toolkitBase = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'

    if ($MaxVersion -and (Test-Path $toolkitBase)) {
        $drMajor = [int]$MaxVersion.Split('.')[0]
        $drMinor = [int]$MaxVersion.Split('.')[1]

        # Get all installed CUDA dirs, sorted descending (highest first)
        $cudaDirs = Get-ChildItem -Directory $toolkitBase | Where-Object {
            $_.Name -match '^v(\d+)\.(\d+)'
        } | Sort-Object { [version]($_.Name -replace '^v','') } -Descending

        foreach ($dir in $cudaDirs) {
            if ($dir.Name -match '^v(\d+)\.(\d+)') {
                $tkMajor = [int]$Matches[1]; $tkMinor = [int]$Matches[2]
                $compatible = ($tkMajor -lt $drMajor) -or ($tkMajor -eq $drMajor -and $tkMinor -le $drMinor)
                if ($compatible) {
                    $nvcc = Join-Path $dir.FullName 'bin\nvcc.exe'
                    if (Test-Path $nvcc) {
                        return $nvcc
                    }
                }
            }
        }

        # No compatible side-by-side version found
        return $null
    }

    # Fallback: no version constraint — pick latest or whatever is available

    # 1. Check nvcc on PATH
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    # 2. Check CUDA_PATH env var
    $cudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Process')
    if (-not $cudaRoot) { $cudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine') }
    if (-not $cudaRoot) { $cudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'User') }
    if ($cudaRoot -and (Test-Path (Join-Path $cudaRoot 'bin\nvcc.exe'))) {
        return (Join-Path $cudaRoot 'bin\nvcc.exe')
    }

    # 3. Scan standard toolkit directory
    if (Test-Path $toolkitBase) {
        $latest = Get-ChildItem -Directory $toolkitBase | Sort-Object Name | Select-Object -Last 1
        if ($latest -and (Test-Path (Join-Path $latest.FullName 'bin\nvcc.exe'))) {
            return (Join-Path $latest.FullName 'bin\nvcc.exe')
        }
    }

    return $null
}

# Detect CUDA Compute Capability via nvidia-smi.
# Returns e.g. "80" for A100 (8.0), "89" for RTX 4090 (8.9), etc.
# Returns $null if detection fails.
function Get-CudaComputeCapability {
    # Use the resolved absolute path ($NvidiaSmiExe) to survive Refresh-Environment
    $smiExe = if ($script:NvidiaSmiExe) { $script:NvidiaSmiExe } else {
        $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($cmd) { $cmd.Source } else { $null }
    }
    if (-not $smiExe) { return $null }

    try {
        $raw = & $smiExe --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }

        # nvidia-smi may return multiple GPUs; take the first one
        $cap = ($raw -split "`n")[0].Trim()
        if ($cap -match '^(\d+)\.(\d+)$') {
            $major = $Matches[1]
            $minor = $Matches[2]
            return "$major$minor"
        }
    } catch { }

    return $null
}

# Check if an nvcc binary supports a given sm_ architecture.
# Uses `nvcc --list-gpu-code` which outputs sm_* tokens (--list-gpu-arch
# outputs compute_* tokens instead).  Available since CUDA 11.6.
# Returns $false if the flag isn't supported (old toolkit) — safer to reject
# and fall back to scanning/PTX than to assume support and fail later.
function Test-NvccArchSupport {
    param([string]$NvccExe, [string]$Arch)
    try {
        $listCode = & $NvccExe --list-gpu-code 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { return $false }
        return ($listCode -match "sm_$Arch")
    } catch {
        return $false
    }
}

# Given an nvcc binary, return the highest sm_ architecture it supports.
# Returns e.g. "90" for CUDA 12.4. Returns $null if detection fails.
function Get-NvccMaxArch {
    param([string]$NvccExe)
    try {
        $listCode = & $NvccExe --list-gpu-code 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { return $null }
        $arches = @()
        foreach ($line in $listCode -split "`n") {
            if ($line.Trim() -match '^sm_(\d+)') {
                $arches += [int]$Matches[1]
            }
        }
        if ($arches.Count -gt 0) {
            return ($arches | Sort-Object | Select-Object -Last 1).ToString()
        }
    } catch { }
    return $null
}

# Detect driver's max CUDA version from nvidia-smi and return the highest
# compatible PyTorch CUDA index tag (e.g. "cu128").
# PyTorch on Windows ships CPU-only by default from PyPI; CUDA wheels live at
# https://download.pytorch.org/whl/<tag>. The tag must not exceed the driver's
# capability: e.g. driver "CUDA Version: 12.9" → cu128 (not cu130).
function Get-PytorchCudaTag {
    $smiExe = if ($script:NvidiaSmiExe) { $script:NvidiaSmiExe } else {
        $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($cmd) { $cmd.Source } else { $null }
    }
    if (-not $smiExe) { return "cu124" }

    try {
        # 2>&1 | Out-String merges stderr into stdout then converts to a single
        # string.  Plain 2>$null doesn't fully suppress stderr in PS 5.1 --
        # ErrorRecord objects leak into $output and break the -match.
        $output = & $smiExe 2>&1 | Out-String
        if ($output -match 'CUDA Version:\s+(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            # PyTorch 2.10 offers: cu124, cu126, cu128, cu130
            if ($major -ge 13) { return "cu130" }
            if ($major -eq 12 -and $minor -ge 8) { return "cu128" }
            if ($major -eq 12 -and $minor -ge 6) { return "cu126" }
            return "cu124"
        }
    } catch { }

    return "cu124"
}

# Find Visual Studio Build Tools for cmake -G flag.
# Strategy: (1) vswhere, (2) scan filesystem (handles broken vswhere registration).
# Returns @{ Generator = "Visual Studio 17 2022"; InstallPath = "C:\..."; Source = "..." } or $null.
function Find-VsBuildTools {
    $map = @{ '2022' = '17'; '2019' = '16'; '2017' = '15' }

    # --- Try vswhere first (works when VS is properly registered) ---
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $info = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion 2>$null
        $path = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($info -and $path) {
            $y = $info.Trim()
            $n = $map[$y]
            if ($n) {
                return @{ Generator = "Visual Studio $n $y"; InstallPath = $path.Trim(); Source = 'vswhere' }
            }
        }
    }

    # --- Scan filesystem (handles broken vswhere registration after winget cycles) ---
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    $editions = @('BuildTools', 'Community', 'Professional', 'Enterprise')
    $years = @('2022', '2019', '2017')

    foreach ($y in $years) {
        foreach ($r in $roots) {
            foreach ($ed in $editions) {
                $candidate = Join-Path $r "Microsoft Visual Studio\$y\$ed"
                if (Test-Path $candidate) {
                    $vcDir = Join-Path $candidate "VC\Tools\MSVC"
                    if (Test-Path $vcDir) {
                        $cl = Get-ChildItem -Path $vcDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
                        if ($cl) {
                            $n = $map[$y]
                            if ($n) {
                                return @{ Generator = "Visual Studio $n $y"; InstallPath = $candidate; Source = "filesystem ($ed)"; ClExe = $cl.FullName }
                            }
                        }
                    }
                }
            }
        }
    }

    return $null
}

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
Write-Host "+==============================================+" -ForegroundColor Green
Write-Host "|       Unsloth Studio Setup (Windows)         |" -ForegroundColor Green
Write-Host "+==============================================+" -ForegroundColor Green

# ==========================================================================
#  PHASE 1: System-level prerequisites (winget installs, env vars)
#  All heavy system tool installs happen here BEFORE touching Python.
# ==========================================================================

# ============================================
# 1a. GPU detection
# ============================================
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null  # Absolute path -- survives Refresh-Environment
try {
    $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvSmiCmd) {
        & $nvSmiCmd.Source 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $HasNvidiaSmi = $true
            $NvidiaSmiExe = $nvSmiCmd.Source
        }
    }
} catch {}
# Fallback: nvidia-smi may not be on PATH even though a GPU + driver exist.
# Check the default install location and the Windows driver store.
if (-not $HasNvidiaSmi) {
    $nvSmiDefaults = @(
        "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "$env:SystemRoot\System32\nvidia-smi.exe"
    )
    foreach ($p in $nvSmiDefaults) {
        if (Test-Path $p) {
            try {
                & $p 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $HasNvidiaSmi = $true
                    $NvidiaSmiExe = $p
                    Write-Host "   Found nvidia-smi at $(Split-Path $p -Parent)" -ForegroundColor Gray
                    break
                }
            } catch {}
        }
    }
}
if (-not $HasNvidiaSmi) {
    Write-Host ""
    Write-Host "[WARN] No NVIDIA GPU detected. Studio will run in chat-only (GGUF) mode." -ForegroundColor Yellow
    Write-Host "       Training and GPU inference require an NVIDIA GPU with drivers installed." -ForegroundColor Yellow
    Write-Host "       https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "[OK] NVIDIA GPU detected" -ForegroundColor Green
}

# ============================================
# 1a.5. Windows Long Paths (required for deep node_modules / Python paths)
# ============================================
$LongPathsEnabled = $false
try {
    $regVal = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
    if ($regVal -and $regVal.LongPathsEnabled -eq 1) {
        $LongPathsEnabled = $true
    }
} catch {}

if ($LongPathsEnabled) {
    Write-Host "[OK] Windows Long Paths enabled" -ForegroundColor Green
} else {
    Write-Host "Windows Long Paths not enabled (required for Triton compilation and deep dependency paths)." -ForegroundColor Yellow
    Write-Host "   Requesting admin access to fix..." -ForegroundColor Yellow
    try {
        # Spawn an elevated process to set the registry key (triggers UAC prompt)
        $proc = Start-Process -FilePath "reg.exe" `
            -ArgumentList 'add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f' `
            -Verb RunAs -Wait -PassThru -ErrorAction Stop
        if ($proc.ExitCode -eq 0) {
            $LongPathsEnabled = $true
            Write-Host "[OK] Windows Long Paths enabled (via UAC)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Failed to enable Long Paths (exit code: $($proc.ExitCode))" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARN] Could not enable Long Paths (UAC was declined or not available)" -ForegroundColor Yellow
        Write-Host "       Run this manually in an Admin terminal:" -ForegroundColor Yellow
        Write-Host '       reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f' -ForegroundColor Cyan
    }
}

# ============================================
# 1b. Git (required by pip for git+https:// deps and by npm)
# ============================================
$HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
if (-not $HasGit) {
    Write-Host "Git not found -- installing via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        try {
            winget install Git.Git --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Environment
            $HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
        } catch { }
    }
    if (-not $HasGit) {
        Write-Host "[ERROR] Git is required but could not be installed automatically." -ForegroundColor Red
        Write-Host "        Install Git from https://git-scm.com/download/win and re-run." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Git installed: $(git --version)" -ForegroundColor Green
} else {
    Write-Host "[OK] Git found: $(git --version)" -ForegroundColor Green
}

# ============================================
# 1c. CMake (required for llama.cpp build)
# ============================================
$HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
if (-not $HasCmake) {
    Write-Host "CMake not found -- installing via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        try {
            winget install Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Environment
            $HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
        } catch { }
    }
    # winget may succeed but cmake isn't on PATH yet (MSI PATH changes need a
    # new shell). Try the default install location as a fallback.
    if (-not $HasCmake) {
        $cmakeDefaults = @(
            "$env:ProgramFiles\CMake\bin",
            "${env:ProgramFiles(x86)}\CMake\bin",
            "$env:LOCALAPPDATA\CMake\bin"
        )
        foreach ($d in $cmakeDefaults) {
            if (Test-Path (Join-Path $d "cmake.exe")) {
                $env:Path = "$d;$env:Path"
                # Persist to user PATH so Refresh-Environment does not drop it later
                $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
                if (-not $userPath -or $userPath -notlike "*$d*") {
                    [Environment]::SetEnvironmentVariable('Path', "$d;$userPath", 'User')
                }
                $HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
                if ($HasCmake) {
                    Write-Host "   Found cmake at $d (added to PATH)" -ForegroundColor Gray
                    break
                }
            }
        }
    }
    if ($HasCmake) {
        Write-Host "[OK] CMake installed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] CMake is required but could not be installed." -ForegroundColor Red
        Write-Host "        Install CMake from https://cmake.org/download/ and re-run." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[OK] CMake found: $(cmake --version | Select-Object -First 1)" -ForegroundColor Green
}

# ============================================
# 1d. Visual Studio Build Tools (C++ compiler for llama.cpp)
# ============================================
$CmakeGenerator = $null
$VsInstallPath = $null
$vsResult = Find-VsBuildTools

if (-not $vsResult) {
    Write-Host "Visual Studio Build Tools not found -- installing via winget..." -ForegroundColor Yellow
    Write-Host "   (This is a one-time install, may take several minutes)" -ForegroundColor Gray
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        $prevEAPTemp = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        winget install Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
        $ErrorActionPreference = $prevEAPTemp
        # Re-scan after install (don't trust vswhere catalog)
        $vsResult = Find-VsBuildTools
    }
}

if ($vsResult) {
    $CmakeGenerator = $vsResult.Generator
    $VsInstallPath = $vsResult.InstallPath
    Write-Host "[OK] $CmakeGenerator detected via $($vsResult.Source)" -ForegroundColor Green
    if ($vsResult.ClExe) { Write-Host "   cl.exe: $($vsResult.ClExe)" -ForegroundColor Gray }
} else {
    Write-Host "[ERROR] Visual Studio Build Tools could not be found or installed." -ForegroundColor Red
    Write-Host "        Manual install:" -ForegroundColor Red
    Write-Host '        1. winget install Microsoft.VisualStudio.2022.BuildTools --source winget' -ForegroundColor Yellow
    Write-Host '        2. Open Visual Studio Installer -> Modify -> check "Desktop development with C++"' -ForegroundColor Yellow
    exit 1
}

# ============================================
# 1e. CUDA Toolkit (nvcc for llama.cpp build + env vars)
# ============================================
if ($HasNvidiaSmi) {
# IMPORTANT: The CUDA Toolkit version must be <= the max CUDA version the
# NVIDIA driver supports.  nvidia-smi reports this as "CUDA Version: X.Y".
# If we install a toolkit newer than the driver supports, llama-server will
# fail at runtime with "ggml_cuda_init: failed to initialize CUDA: (null)".

# -- Detect max CUDA version the driver supports --
$DriverMaxCuda = $null
try {
    $smiOut = & $NvidiaSmiExe 2>&1 | Out-String
    if ($smiOut -match "CUDA Version:\s+([\d]+)\.([\d]+)") {
        $DriverMaxCuda = "$($Matches[1]).$($Matches[2])"
        Write-Host "   Driver supports up to CUDA $DriverMaxCuda" -ForegroundColor Gray
    }
} catch {}

# Detect compute capability early so we can validate toolkit support
$CudaArch = Get-CudaComputeCapability
if ($CudaArch) {
    Write-Host "   GPU Compute Capability = $($CudaArch.Insert($CudaArch.Length-1, '.')) (sm_$CudaArch)" -ForegroundColor Gray
}

# -- Find a toolkit that's compatible with the driver AND the GPU --
# Strategy: prefer the toolkit at CUDA_PATH (user's existing setup) if it's
# compatible with the driver AND supports the GPU architecture.  Only fall back
# to scanning side-by-side installs if CUDA_PATH is missing, points to an
# incompatible version, or can't compile for the GPU.  This avoids
# header/binary mismatches when multiple toolkits are installed.
$IncompatibleToolkit = $null
$NvccPath = $null

if ($DriverMaxCuda) {
    $drMajorCuda = [int]$DriverMaxCuda.Split('.')[0]
    $drMinorCuda = [int]$DriverMaxCuda.Split('.')[1]

    # --- Step 1: Check existing CUDA_PATH first ---
    $existingCudaPath = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
    if (-not $existingCudaPath) {
        $existingCudaPath = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'User')
    }
    if ($existingCudaPath -and (Test-Path (Join-Path $existingCudaPath 'bin\nvcc.exe'))) {
        $candidateNvcc = Join-Path $existingCudaPath 'bin\nvcc.exe'
        $verOut = & $candidateNvcc --version 2>&1 | Out-String
        if ($verOut -match 'release\s+(\d+)\.(\d+)') {
            $tkMaj = [int]$Matches[1]; $tkMin = [int]$Matches[2]
            $isCompat = ($tkMaj -lt $drMajorCuda) -or ($tkMaj -eq $drMajorCuda -and $tkMin -le $drMinorCuda)
            if ($isCompat) {
                # Also verify the toolkit supports our GPU architecture
                Write-Host "   [DEBUG] Checking CUDA compatibility: toolkit=$tkMaj.$tkMin arch=sm_$CudaArch" -ForegroundColor Magenta
                $archOk = $true
                if ($CudaArch) {
                    $archOk = Test-NvccArchSupport -NvccExe $candidateNvcc -Arch $CudaArch
                    if (-not $archOk) {
                        Write-Host "   [INFO] CUDA_PATH toolkit (CUDA $tkMaj.$tkMin) does not support GPU arch sm_$CudaArch" -ForegroundColor Yellow
                        Write-Host "          Looking for a newer toolkit..." -ForegroundColor Yellow
                    }
                }
                if ($archOk) {
                    $NvccPath = $candidateNvcc
                    Write-Host "   [OK] Using existing CUDA Toolkit at CUDA_PATH (nvcc: $NvccPath)" -ForegroundColor Green
                }
            } else {
                Write-Host "   [INFO] CUDA_PATH ($existingCudaPath) has CUDA $tkMaj.$tkMin which exceeds driver max $DriverMaxCuda" -ForegroundColor Yellow
            }
        }
    }

    # --- Step 2: Fall back to scanning side-by-side installs ---
    if (-not $NvccPath) {
        $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
        if ($NvccPath) {
            Write-Host "   [OK] Found compatible CUDA Toolkit (nvcc: $NvccPath)" -ForegroundColor Green
            if ($existingCudaPath) {
                $selectedRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
                if ($existingCudaPath.TrimEnd('\') -ne $selectedRoot.TrimEnd('\')) {
                    Write-Host "   [INFO] Overriding CUDA_PATH from $existingCudaPath to $selectedRoot" -ForegroundColor Yellow
                }
            }
        } else {
            # Check if there's an incompatible (too new) toolkit installed
            $AnyNvcc = Find-Nvcc
            if ($AnyNvcc) {
                $NvccOut = & $AnyNvcc --version 2>&1 | Out-String
                if ($NvccOut -match "release\s+([\d]+\.[\d]+)") {
                    $IncompatibleToolkit = $Matches[1]
                }
            }
        }
    }
} else {
    $NvccPath = Find-Nvcc
}

# -- If incompatible toolkit is blocking, tell user to uninstall it --
if (-not $NvccPath -and $IncompatibleToolkit) {
    Write-Host "" -ForegroundColor Red
    Write-Host "========================================================================" -ForegroundColor Red
    Write-Host "[ERROR] CUDA Toolkit $IncompatibleToolkit is installed but INCOMPATIBLE" -ForegroundColor Red
    Write-Host "        with your NVIDIA driver (which supports up to CUDA $DriverMaxCuda)." -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "  This will cause 'failed to initialize CUDA' errors at runtime." -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "  To fix:" -ForegroundColor Yellow
    Write-Host "    1. Open Control Panel -> Programs -> Uninstall a program" -ForegroundColor Yellow
    Write-Host "    2. Uninstall 'NVIDIA CUDA Toolkit $IncompatibleToolkit'" -ForegroundColor Yellow
    Write-Host "    3. Re-run setup.bat (it will install CUDA $DriverMaxCuda automatically)" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "  Alternatively, update your NVIDIA driver to one that supports CUDA $IncompatibleToolkit." -ForegroundColor Gray
    Write-Host "========================================================================" -ForegroundColor Red
    exit 1
}

# -- No toolkit at all: install via winget --
if (-not $NvccPath) {
    Write-Host "CUDA toolkit (nvcc) not found -- installing via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        if ($DriverMaxCuda) {
            # Query winget for available CUDA Toolkit versions
            $drMajor = [int]$DriverMaxCuda.Split('.')[0]
            $drMinor = [int]$DriverMaxCuda.Split('.')[1]
            $AvailableVersions = @()
            try {
                $rawOutput = winget show Nvidia.CUDA --versions --accept-source-agreements 2>&1 | Out-String
                # Parse version lines (e.g. "12.6", "12.5", "11.8")
                foreach ($line in $rawOutput -split "`n") {
                    $line = $line.Trim()
                    if ($line -match '^\d+\.\d+') {
                        $AvailableVersions += $line
                    }
                }
            } catch {}

            # Filter to compatible versions (<= driver max) and pick the highest
            $BestVersion = $null
            foreach ($ver in $AvailableVersions) {
                $parts = $ver.Split('.')
                $vMajor = [int]$parts[0]
                $vMinor = [int]$parts[1]
                if ($vMajor -lt $drMajor -or ($vMajor -eq $drMajor -and $vMinor -le $drMinor)) {
                    $BestVersion = $ver
                    break  # list is descending, first match is highest compatible
                }
            }

            if ($BestVersion) {
                Write-Host "   Installing CUDA Toolkit $BestVersion via winget...  " -ForegroundColor Cyan
                $prevEAPCuda = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                winget install --id=Nvidia.CUDA --version=$BestVersion -e --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
                $ErrorActionPreference = $prevEAPCuda
                Refresh-Environment
                $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
                if ($NvccPath) {
                    Write-Host "   [OK] CUDA Toolkit $BestVersion installed (nvcc: $NvccPath)" -ForegroundColor Green
                }
            } else {
                Write-Host "   [WARN] No compatible CUDA Toolkit version found in winget (need <= $DriverMaxCuda)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "   Installing CUDA Toolkit (latest) via winget..." -ForegroundColor Cyan
            winget install --id=Nvidia.CUDA -e --source winget --accept-package-agreements --accept-source-agreements
            Refresh-Environment
            $NvccPath = Find-Nvcc
            if ($NvccPath) {
                Write-Host "   [OK] CUDA Toolkit installed (nvcc: $NvccPath)" -ForegroundColor Green
            }
        }
    }
}

if (-not $NvccPath) {
    Write-Host "[ERROR] CUDA Toolkit (nvcc) is required but could not be found or installed." -ForegroundColor Red
    if ($DriverMaxCuda) {
        Write-Host "        Install CUDA Toolkit $DriverMaxCuda from https://developer.nvidia.com/cuda-toolkit-archive" -ForegroundColor Yellow
    } else {
        Write-Host "        Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
    exit 1
}

# -- Set CUDA env vars so cmake AND MSBuild can find the toolkit --
$CudaToolkitRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
# CUDA_PATH: used by cmake's find_package(CUDAToolkit)
[Environment]::SetEnvironmentVariable('CUDA_PATH', $CudaToolkitRoot, 'Process')
# CudaToolkitDir: the MSBuild property that CUDA .targets checks directly
# Trailing backslash required -- the .targets file appends subpaths to it
[Environment]::SetEnvironmentVariable('CudaToolkitDir', "$CudaToolkitRoot\", 'Process')
# Always persist CUDA_PATH to User registry so the compatible toolkit is used
# in future sessions (overwrites any existing value pointing to a newer, incompatible version)
[Environment]::SetEnvironmentVariable('CUDA_PATH', $CudaToolkitRoot, 'User')
Write-Host "   Persisted CUDA_PATH=$CudaToolkitRoot to user environment" -ForegroundColor Gray
# Clear all versioned CUDA_PATH_V* env vars in this process to prevent
# cmake/MSBuild from discovering a conflicting CUDA installation.
$cudaPathVars = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object { $_ -match '^CUDA_PATH_V' })
foreach ($v in $cudaPathVars) {
    [Environment]::SetEnvironmentVariable($v, $null, 'Process')
}
# Set only the versioned var matching the selected toolkit (e.g. CUDA_PATH_V13_0)
$tkDirName = Split-Path $CudaToolkitRoot -Leaf
if ($tkDirName -match '^v(\d+)\.(\d+)') {
    $cudaPathVerVar = "CUDA_PATH_V$($Matches[1])_$($Matches[2])"
    [Environment]::SetEnvironmentVariable($cudaPathVerVar, $CudaToolkitRoot, 'Process')
    Write-Host "   Set $cudaPathVerVar (cleared other CUDA_PATH_V* vars)" -ForegroundColor Gray
}
# Ensure nvcc's bin dir is on PATH for this process
$nvccBinDir = Split-Path $NvccPath -Parent
if ($env:PATH -notlike "*$nvccBinDir*") {
    [Environment]::SetEnvironmentVariable('PATH', "$nvccBinDir;$env:PATH", 'Process')
}
# Persist nvcc bin dir to User PATH so it works in new terminals
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath -or $userPath -notlike "*$nvccBinDir*") {
    if ($userPath) {
        [Environment]::SetEnvironmentVariable('Path', "$nvccBinDir;$userPath", 'User')
    } else {
        [Environment]::SetEnvironmentVariable('Path', "$nvccBinDir", 'User')
    }
    Write-Host "   Persisted CUDA bin dir to user PATH" -ForegroundColor Gray
}

# -- Ensure CUDA ↔ Visual Studio integration files exist --
# When CUDA is installed before VS Build Tools (or VS is reinstalled after CUDA),
# the MSBuild .targets/.props files that let VS compile .cu files are missing.
# cmake fails with "No CUDA toolset found". Fix: copy from CUDA extras dir.
if ($VsInstallPath -and $CudaToolkitRoot) {
    $vsCustomizations = Join-Path $VsInstallPath "MSBuild\Microsoft\VC\v170\BuildCustomizations"
    $cudaExtras = Join-Path $CudaToolkitRoot "extras\visual_studio_integration\MSBuildExtensions"
    if ((Test-Path $cudaExtras) -and (Test-Path $vsCustomizations)) {
        $hasTargets = Get-ChildItem $vsCustomizations -Filter "CUDA *.targets" -ErrorAction SilentlyContinue
        if (-not $hasTargets) {
            Write-Host "   [INFO] CUDA VS integration missing -- copying .targets files..." -ForegroundColor Yellow
            try {
                Copy-Item "$cudaExtras\*" $vsCustomizations -Force -ErrorAction Stop
                Write-Host "   [OK] CUDA VS integration files installed" -ForegroundColor Green
            } catch {
                # Direct copy failed (needs admin). Try elevated copy via Start-Process.
                try {
                    $copyCmd = "Copy-Item '$cudaExtras\*' '$vsCustomizations' -Force"
                    Start-Process powershell -ArgumentList "-NoProfile -Command $copyCmd" -Verb RunAs -Wait -ErrorAction Stop
                    $hasTargetsRetry = Get-ChildItem $vsCustomizations -Filter "CUDA *.targets" -ErrorAction SilentlyContinue
                    if ($hasTargetsRetry) {
                        Write-Host "   [OK] CUDA VS integration files installed (elevated)" -ForegroundColor Green
                    } else {
                        throw "Copy did not produce .targets files"
                    }
                } catch {
                    Write-Host "   [WARN] Could not copy CUDA VS integration files" -ForegroundColor Yellow
                    Write-Host "          The llama.cpp build may fail with 'No CUDA toolset found'." -ForegroundColor Yellow
                    Write-Host "          Manual fix: copy contents of" -ForegroundColor Yellow
                    Write-Host "            $cudaExtras" -ForegroundColor Cyan
                    Write-Host "          into:" -ForegroundColor Yellow
                    Write-Host "            $vsCustomizations" -ForegroundColor Cyan
                }
            }
        }
    }
}

Write-Host "[OK] CUDA Toolkit: $NvccPath" -ForegroundColor Green
Write-Host "   CUDA_PATH      = $CudaToolkitRoot" -ForegroundColor Gray
Write-Host "   CudaToolkitDir = $CudaToolkitRoot\" -ForegroundColor Gray

# $CudaArch was detected earlier (before toolkit selection) so it could
# influence which toolkit we picked.  Just log the final state here.
if (-not $CudaArch) {
    Write-Host "   [WARN] Could not detect compute capability -- cmake will use defaults" -ForegroundColor Yellow
}
} else {
    Write-Host "[SKIP] CUDA Toolkit -- no NVIDIA GPU detected" -ForegroundColor Yellow
}

# ============================================
# 1f. Node.js / npm (skip if pip-installed -- only needed for frontend build)
# ============================================
if ($IsPipInstall) {
    Write-Host "[OK] Running from pip install - frontend already bundled, skipping Node/npm check" -ForegroundColor Green
} else {
    # setup.sh installs Node LTS (v22) via nvm. We enforce the same range here:
    # Node >= 20, npm >= 11.
    $NeedNode = $true
    try {
        $NodeVersion = (node -v 2>$null)
        $NpmVersion = (npm -v 2>$null)
        if ($NodeVersion -and $NpmVersion) {
            $NodeMajor = [int]($NodeVersion -replace 'v','').Split('.')[0]
            $NpmMajor = [int]$NpmVersion.Split('.')[0]

            if ($NodeMajor -ge 20 -and $NpmMajor -ge 11) {
                Write-Host "[OK] Node $NodeVersion and npm $NpmVersion already meet requirements." -ForegroundColor Green
                $NeedNode = $false
            } else {
                Write-Host "[WARN] Node $NodeVersion / npm $NpmVersion too old." -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "[WARN] Node/npm not found." -ForegroundColor Yellow
    }

    if ($NeedNode) {
        Write-Host "Installing Node.js LTS via winget..." -ForegroundColor Cyan
        try {
            winget install OpenJS.NodeJS.LTS --source winget --accept-package-agreements --accept-source-agreements
            Refresh-Environment
        } catch {
            Write-Host "[ERROR] Could not install Node.js automatically." -ForegroundColor Red
            Write-Host "Please install Node.js >= 20 from https://nodejs.org/" -ForegroundColor Red
            exit 1
        }
    }

    Write-Host "[OK] Node $(node -v) | npm $(npm -v)" -ForegroundColor Green
}

# ============================================
# 1g. Python (>= 3.11 and < 3.14, matching setup.sh)
# ============================================
$HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$PythonOk = $false

if ($HasPython) {
    $PyVer = python --version 2>&1
    if ($PyVer -match "(\d+)\.(\d+)") {
        $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
        if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
            Write-Host "[OK] Python $PyVer" -ForegroundColor Green
            $PythonOk = $true
        } else {
            Write-Host "[ERROR] Python $PyVer is outside supported range (need >= 3.11 and < 3.14)." -ForegroundColor Red
            Write-Host "        Install Python 3.12 from https://python.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
    }
} else {
    # No Python at all -- install 3.12
    Write-Host "Python not found -- installing Python 3.12 via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        winget install -e --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
        Refresh-Environment
    }
    $HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $HasPython) {
        Write-Host "[ERROR] Python could not be installed automatically." -ForegroundColor Red
        Write-Host "        Install Python 3.12 from https://python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[OK] Python $(python --version)" -ForegroundColor Green
    $PythonOk = $true
}

# Ensure Python Scripts dir is on PATH (so 'unsloth' command works in new terminals)
$ScriptsDir = python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user') if __import__('os').path.exists(sysconfig.get_path('scripts', 'nt_user')) else sysconfig.get_path('scripts'))"
if ($LASTEXITCODE -eq 0 -and $ScriptsDir -and (Test-Path $ScriptsDir)) {
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $UserPathEntries = if ($UserPath) { $UserPath.Split(';') } else { @() }
    if (-not ($UserPathEntries | Where-Object { $_.TrimEnd('\') -eq $ScriptsDir })) {
        $newUserPath = if ($UserPath) { "$ScriptsDir;$UserPath" } else { $ScriptsDir }
        [Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')

        # Also add to current process so it's available immediately
        $ProcessPathEntries = $env:PATH.Split(';')
        if (-not ($ProcessPathEntries | Where-Object { $_.TrimEnd('\') -eq $ScriptsDir })) {
            $env:PATH = "$ScriptsDir;$env:PATH"
        }
        Write-Host "   Persisted Python Scripts dir to user PATH: $ScriptsDir" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "--- System prerequisites ready ---" -ForegroundColor Green
Write-Host ""

# ==========================================================================
#  PHASE 2: Frontend build (skip if pip-installed -- already bundled)
# ==========================================================================
$DistDir = Join-Path $FrontendDir "dist"
# Skip build if dist/ exists and no tracked input is newer than dist/.
# Checks src/, public/, package.json, config files -- not just src/.
$NeedFrontendBuild = $true
if ($IsPipInstall) {
    $NeedFrontendBuild = $false
    Write-Host "[OK] Running from pip install - frontend already bundled, skipping build" -ForegroundColor Green
} elseif (Test-Path $DistDir) {
    $DistTime = (Get-Item $DistDir).LastWriteTime
    $NewerFile = $null
    # Check src/ and public/ recursively (probe paths directly, not via -Include)
    foreach ($subDir in @("src", "public")) {
        $subPath = Join-Path $FrontendDir $subDir
        if (Test-Path $subPath) {
            $NewerFile = Get-ChildItem -Path $subPath -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt $DistTime } | Select-Object -First 1
            if ($NewerFile) { break }
        }
    }
    # Also check all top-level files (package.json, bun.lock, vite.config.ts, index.html, etc.)
    if (-not $NewerFile) {
        $NewerFile = Get-ChildItem -Path $FrontendDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -gt $DistTime } |
            Select-Object -First 1
    }
    if (-not $NewerFile) {
        $NeedFrontendBuild = $false
        Write-Host "[OK] Frontend already built and up to date -- skipping build" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Frontend source changed since last build -- rebuilding..." -ForegroundColor Yellow
    }
}
$NeedFrontendBuild = $true
if ($NeedFrontendBuild -and -not $IsPipInstall) {
    Write-Host ""
    Write-Host "Building frontend..." -ForegroundColor Cyan
    # npm writes warnings to stderr; lower ErrorActionPreference so PS doesn't
    # treat them as terminating errors (same pattern as the pip section below).
    $prevEAP_npm = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $FrontendDir
    npm install 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        Write-Host "[ERROR] npm install failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "   Try running 'npm install' manually in frontend/ to see errors" -ForegroundColor Yellow
        exit 1
    }
    npm run build 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        Write-Host "[ERROR] npm run build failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_npm
    Write-Host "[OK] Frontend built to frontend/dist" -ForegroundColor Green
}

if (Test-Path $OxcValidatorDir) {
    Write-Host "Installing OXC validator runtime..." -ForegroundColor Cyan
    $prevEAP_oxc = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $OxcValidatorDir
    npm install 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_oxc
        Write-Host "[ERROR] OXC validator npm install failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_oxc
    Write-Host "[OK] OXC validator runtime installed" -ForegroundColor Green
}

# ==========================================================================
#  PHASE 3: Python environment + dependencies
# ==========================================================================
Write-Host ""
Write-Host "Setting up Python environment..." -ForegroundColor Cyan

# Find Python
$PythonCmd = $null
foreach ($candidate in @("python3.13", "python3.12", "python3.11", "python3", "python")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match 'Python 3\.(\d+)') {
            $minor = [int]$Matches[1]
            if ($minor -ge 11 -and $minor -le 13) {
                $PythonCmd = $candidate
                break
            }
        }
    } catch { }
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] No Python 3.11-3.13 found." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Using $PythonCmd ($(& $PythonCmd --version 2>&1))" -ForegroundColor Green

# Always create a .venv for isolation -- even for pip installs.
# Created in the repo root (parent of studio/).
$VenvDir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "   Creating virtual environment at $VenvDir..." -ForegroundColor Cyan
    & $PythonCmd -m venv $VenvDir
} else {
    Write-Host "   Reusing existing virtual environment at $VenvDir" -ForegroundColor Green
}

# pip and python write to stderr even on success (progress bars, warnings).
# With $ErrorActionPreference = "Stop" (set at top of script), PS 5.1
# converts stderr lines into terminating ErrorRecords, breaking output.
# Lower to "Continue" for the pip/python section.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
. $ActivateScript

# Try to use uv (much faster than pip), fall back to pip if unavailable
$UseUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $UseUv = $true
} else {
    Write-Host "   Installing uv package manager..." -ForegroundColor Cyan
    try {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null
        Refresh-Environment
        # Re-activate venv since Refresh-Environment rebuilds PATH from
        # registry and drops the venv's Scripts directory
        . $ActivateScript
        if (Get-Command uv -ErrorAction SilentlyContinue) { $UseUv = $true }
    } catch { }
}

# Helper: install a package, preferring uv with pip fallback
function Fast-Install {
    param([Parameter(ValueFromRemainingArguments=$true)]$Args_)
    if ($UseUv) {
        $VenvPy = (Get-Command python).Source
        $result = & uv pip install --python $VenvPy @Args_ 2>&1
        if ($LASTEXITCODE -eq 0) { return }
    }
    & python -m pip install @Args_ 2>&1
}

Fast-Install --upgrade pip | Out-Null

# if (-not $IsPipInstall) {
#     # Running from repo: copy requirements and do editable install
#     $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
#     $ReqsSrc = Join-Path $RepoRoot "backend\requirements"
#     $ReqsDst = Join-Path $PackageDir "requirements"
#     if (-not (Test-Path $ReqsDst)) { New-Item -ItemType Directory -Path $ReqsDst | Out-Null }
#     Copy-Item (Join-Path $ReqsSrc "*.txt") $ReqsDst -Force

#     Write-Host "   Installing CLI entry point..." -ForegroundColor Cyan
#     pip install -e $RepoRoot 2>&1 | Out-Null
# } else {
#     # Running from pip install: the package is in system Python but not in
#     # the fresh .venv. Install it so run_install() can find its modules
#     # and bundled requirements files.
#     Write-Host "   Installing package into venv..." -ForegroundColor Cyan
#     pip install unsloth-roland-test 2>&1 | Out-Null
# }

# Pre-install PyTorch with CUDA support.
# On Windows, the default PyPI torch wheel is CPU-only.
# We need PyTorch's CUDA index to get GPU-enabled wheels.
# PyTorch bundles its own CUDA runtime, so this works regardless
# of whether the CUDA Toolkit is installed yet.
# The CUDA tag is chosen based on the driver's max supported CUDA version.

# Windows MAX_PATH (260 chars) causes Triton kernel compilation to fail because
# the auto-generated filenames are extremely long. Use a short cache directory.
$TorchCacheDir = "C:\tc"
if (-not (Test-Path $TorchCacheDir)) { New-Item -ItemType Directory -Path $TorchCacheDir -Force | Out-Null }
$env:TORCHINDUCTOR_CACHE_DIR = $TorchCacheDir
[Environment]::SetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', $TorchCacheDir, 'User')
Write-Host "[OK] TORCHINDUCTOR_CACHE_DIR set to $TorchCacheDir (avoids MAX_PATH issues)" -ForegroundColor Green

if ($HasNvidiaSmi) {
    $CuTag = Get-PytorchCudaTag
    Write-Host "   Installing PyTorch with CUDA support ($CuTag)..." -ForegroundColor Cyan
    Write-Host "   (This download is ~2.8 GB -- may take a few minutes)" -ForegroundColor Gray
    $output = Fast-Install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CuTag" | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] PyTorch CUDA install failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        exit 1
    }

    # Install Triton for Windows (enables torch.compile -- without it training can hang)
    Write-Host "   Installing Triton for Windows..." -ForegroundColor Cyan
    $output = Fast-Install "triton-windows<3.7" | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Triton install failed -- torch.compile may not work" -ForegroundColor Yellow
        Write-Host $output -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Triton for Windows installed (enables torch.compile)" -ForegroundColor Green
    }
} else {
    Write-Host "   Installing PyTorch (CPU-only)..." -ForegroundColor Cyan
    $output = Fast-Install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu" | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] PyTorch install failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        exit 1
    }
}

# Ordered heavy dependency installation -- shared cross-platform script
Write-Host "   Running ordered dependency installation..." -ForegroundColor Cyan
python "$PSScriptRoot\install_python_stack.py"
# Restore ErrorActionPreference after pip/python work
$ErrorActionPreference = $prevEAP

# ── Pre-install transformers 5.x into .venv_t5/ ──
# Models like GLM-4.7-Flash need transformers>=5.3.0. Instead of pip-installing
# at runtime (slow, ~10-15s), we pre-install into a separate directory.
# The training subprocess just prepends .venv_t5/ to sys.path -- instant switch.
Write-Host ""
Write-Host "   Pre-installing transformers 5.x for newer model support..." -ForegroundColor Cyan
$VenvT5Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5"
if (Test-Path $VenvT5Dir) { Remove-Item -Recurse -Force $VenvT5Dir }
New-Item -ItemType Directory -Path $VenvT5Dir -Force | Out-Null
$prevEAP_t5 = $ErrorActionPreference
$ErrorActionPreference = "Continue"
foreach ($pkg in @("transformers==5.3.0", "huggingface_hub==1.7.1", "hf_xet==1.4.2")) {
    $output = Fast-Install --target $VenvT5Dir --no-deps $pkg | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5/" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
# tiktoken is needed by Qwen-family tokenizers -- install with deps since
# regex/requests may be missing on Windows
$output = Fast-Install --target $VenvT5Dir tiktoken | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Could not install tiktoken into .venv_t5/ -- Qwen tokenizers may fail" -ForegroundColor Yellow
}
$ErrorActionPreference = $prevEAP_t5
Write-Host "[OK] Transformers 5.x pre-installed to .venv_t5/" -ForegroundColor Green

# ==========================================================================
#  PHASE 3.5: Install OpenSSL dev (for HTTPS support in llama-server)
# ==========================================================================
# llama-server needs OpenSSL to download models from HuggingFace via -hf.
# ShiningLight.OpenSSL.Dev includes headers + libs that cmake can find.
$OpenSslAvailable = $false

# Check if OpenSSL dev is already installed (look for include dir)
$OpenSslRoots = @(
    'C:\Program Files\OpenSSL-Win64',
    'C:\Program Files\OpenSSL',
    'C:\OpenSSL-Win64'
)
$OpenSslRoot = $null
foreach ($root in $OpenSslRoots) {
    if (Test-Path (Join-Path $root 'include\openssl\ssl.h')) {
        $OpenSslRoot = $root
        break
    }
}

if ($OpenSslRoot) {
    $OpenSslAvailable = $true
    Write-Host "[OK] OpenSSL dev found at $OpenSslRoot" -ForegroundColor Green
} else {
    Write-Host "" 
    Write-Host "Installing OpenSSL dev (for HTTPS in llama-server)..." -ForegroundColor Cyan
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        winget install -e --id ShiningLight.OpenSSL.Dev --accept-package-agreements --accept-source-agreements
        # Re-check after install
        foreach ($root in $OpenSslRoots) {
            if (Test-Path (Join-Path $root 'include\openssl\ssl.h')) {
                $OpenSslRoot = $root
                $OpenSslAvailable = $true
                Write-Host "[OK] OpenSSL dev installed at $OpenSslRoot" -ForegroundColor Green
                break
            }
        }
    }
    if (-not $OpenSslAvailable) {
        Write-Host "[WARN] OpenSSL dev not available -- llama-server will be built without HTTPS" -ForegroundColor Yellow
    }
}

# ==========================================================================
#  PHASE 4: Build llama.cpp with CUDA for GGUF inference + export
# ==========================================================================
# Builds at ~/.unsloth/llama.cpp — a single shared location under the user's
# home directory. This is used by both the inference server and the GGUF
# export pipeline (unsloth-zoo).
# We build:
#   - llama-server:   for GGUF model inference (with HTTPS if OpenSSL available)
#   - llama-quantize: for GGUF export quantization
# Prerequisites (git, cmake, VS Build Tools, CUDA Toolkit) already installed in Phase 1.
$UnslothHome = Join-Path $env:USERPROFILE ".unsloth"
if (-not (Test-Path $UnslothHome)) { New-Item -ItemType Directory -Force $UnslothHome | Out-Null }
$LlamaCppDir = Join-Path $UnslothHome "llama.cpp"
$BuildDir = Join-Path $LlamaCppDir "build"
$LlamaServerBin = Join-Path $BuildDir "bin\Release\llama-server.exe"

$HasCmakeForBuild = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)

# Check if existing llama-server matches current GPU mode. A CUDA-built binary
# on a now-CPU-only machine (or vice versa) needs to be rebuilt.
$NeedRebuild = $false
if (Test-Path $LlamaServerBin) {
    $CmakeCacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $CmakeCacheFile) {
        $cachedCuda = Select-String -Path $CmakeCacheFile -Pattern 'GGML_CUDA:BOOL=ON' -Quiet
        if ($HasNvidiaSmi -and -not $cachedCuda) {
            Write-Host "   Existing llama-server is CPU-only but GPU is available -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        } elseif (-not $HasNvidiaSmi -and $cachedCuda) {
            Write-Host "   Existing llama-server was built with CUDA but no GPU detected -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        }
    }
}

if ((Test-Path $LlamaServerBin) -and -not $NeedRebuild) {
    Write-Host ""
    Write-Host "[OK] llama-server already exists at $LlamaServerBin" -ForegroundColor Green
} elseif (-not $HasCmakeForBuild) {
    Write-Host ""
    if (-not $HasNvidiaSmi) {
        # CPU-only machines depend entirely on llama-server for GGUF chat -- cmake is required
        Write-Host "[ERROR] CMake is required to build llama-server for GGUF chat mode." -ForegroundColor Red
        Write-Host "        Install CMake from https://cmake.org/download/ and re-run setup." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[SKIP] llama-server build -- cmake not available" -ForegroundColor Yellow
    Write-Host "       GGUF inference and export will not be available." -ForegroundColor Yellow
    Write-Host "       Install CMake from https://cmake.org/download/ and re-run setup." -ForegroundColor Yellow
} else {
    Write-Host ""
    if ($HasNvidiaSmi) {
        Write-Host "Building llama.cpp with CUDA support..." -ForegroundColor Cyan
    } else {
        Write-Host "Building llama.cpp (CPU-only, no NVIDIA GPU detected)..." -ForegroundColor Cyan
    }
    Write-Host "   This typically takes 5-10 minutes on first build." -ForegroundColor Gray
    Write-Host ""

    # Start total build timer
    $totalSw = [System.Diagnostics.Stopwatch]::StartNew()

    # Native commands (git, cmake) write to stderr even on success.
    # With $ErrorActionPreference = "Stop" (set at top of script), PS 5.1
    # converts stderr lines into terminating ErrorRecords, breaking output.
    # Lower to "Continue" for the build section.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $BuildOk = $true
    $FailedStep = ""

    # Re-sanitize CUDA_PATH_V* vars — Refresh-Environment (called during
    # Node/Python installs above) may have repopulated conflicting versioned
    # vars from the Machine registry.
    if ($HasNvidiaSmi -and $CudaToolkitRoot) {
        $cudaPathVars2 = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object { $_ -match '^CUDA_PATH_V' })
        foreach ($v2 in $cudaPathVars2) {
            [Environment]::SetEnvironmentVariable($v2, $null, 'Process')
        }
        $tkDirName2 = Split-Path $CudaToolkitRoot -Leaf
        if ($tkDirName2 -match '^v(\d+)\.(\d+)') {
            [Environment]::SetEnvironmentVariable("CUDA_PATH_V$($Matches[1])_$($Matches[2])", $CudaToolkitRoot, 'Process')
        }
        # Also re-assert CUDA_PATH and CudaToolkitDir in case they were overwritten
        [Environment]::SetEnvironmentVariable('CUDA_PATH', $CudaToolkitRoot, 'Process')
        [Environment]::SetEnvironmentVariable('CudaToolkitDir', "$CudaToolkitRoot\", 'Process')
    }

    # -- Step A: Clone or pull llama.cpp --

    if (Test-Path (Join-Path $LlamaCppDir ".git")) {
        Write-Host "   llama.cpp repo already cloned, pulling latest..." -ForegroundColor Gray
        git -C $LlamaCppDir pull 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   [WARN] git pull failed -- using existing source" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   Cloning llama.cpp..." -ForegroundColor Gray
        if (Test-Path $LlamaCppDir) { Remove-Item -Recurse -Force $LlamaCppDir }
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $LlamaCppDir 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $BuildOk = $false
            $FailedStep = "git clone"
        }
    }

    # -- Step B: cmake configure --
    # Clean stale CMake cache to prevent previous CUDA settings from leaking
    # into a CPU-only rebuild (or vice versa).
    $CmakeCacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $CmakeCacheFile) {
        Remove-Item -Recurse -Force $BuildDir
    }

    if ($BuildOk) {
        Write-Host ""
        Write-Host "--- cmake configure ---" -ForegroundColor Cyan

        $CmakeArgs = @(
            '-S', $LlamaCppDir,
            '-B', $BuildDir,
            '-G', $CmakeGenerator,
            '-Wno-dev'
        )
        # Tell cmake exactly where VS is (bypasses registry lookup)
        if ($VsInstallPath) {
            $CmakeArgs += "-DCMAKE_GENERATOR_INSTANCE=$VsInstallPath"
        }
        # Common flags
        $CmakeArgs += '-DBUILD_SHARED_LIBS=OFF'
        $CmakeArgs += '-DLLAMA_BUILD_TESTS=OFF'
        $CmakeArgs += '-DLLAMA_BUILD_EXAMPLES=OFF'
        $CmakeArgs += '-DLLAMA_BUILD_SERVER=ON'
        $CmakeArgs += '-DGGML_NATIVE=ON'
        # HTTPS support via OpenSSL
        if ($OpenSslAvailable -and $OpenSslRoot) {
            $CmakeArgs += "-DOPENSSL_ROOT_DIR=$OpenSslRoot"
            $CmakeArgs += '-DLLAMA_OPENSSL=ON'
        } else {
            $CmakeArgs += '-DLLAMA_CURL=OFF'
        }
        $CmakeArgs += '-DCMAKE_EXE_LINKER_FLAGS=/NODEFAULTLIB:LIBCMT'
        # CUDA flags -- only if GPU available, otherwise explicitly disable
        if ($HasNvidiaSmi -and $NvccPath) {
            $CmakeArgs += '-DGGML_CUDA=ON'
            $CmakeArgs += "-DCUDAToolkit_ROOT=$CudaToolkitRoot"
            $CmakeArgs += "-DCUDA_TOOLKIT_ROOT_DIR=$CudaToolkitRoot"
            $CmakeArgs += "-DCMAKE_CUDA_COMPILER=$NvccPath"
            if ($CudaArch) {
                # Validate nvcc actually supports this architecture
                if (Test-NvccArchSupport -NvccExe $NvccPath -Arch $CudaArch) {
                    $CmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
                } else {
                    # GPU arch too new for this toolkit -- fall back to highest supported.
                    # PTX forward-compatibility will JIT-compile for the actual GPU at runtime.
                    $maxArch = Get-NvccMaxArch -NvccExe $NvccPath
                    if ($maxArch) {
                        $CmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$maxArch"
                        Write-Host "   [WARN] GPU is sm_$CudaArch but nvcc only supports up to sm_$maxArch" -ForegroundColor Yellow
                        Write-Host "          Building with sm_$maxArch (PTX will JIT for your GPU at runtime)" -ForegroundColor Yellow
                    }
                    # else: omit flag entirely, let cmake pick defaults
                }
            }
        } else {
            $CmakeArgs += '-DGGML_CUDA=OFF'
        }

        $cmakeOutput = cmake @CmakeArgs 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            $BuildOk = $false
            $FailedStep = "cmake configure"
            Write-Host $cmakeOutput -ForegroundColor Red
            if ($cmakeOutput -match 'No CUDA toolset found|CUDA_TOOLKIT_ROOT_DIR|nvcc') {
                Write-Host ""
                Write-Host "   Hint: CUDA VS integration may be missing. Try running as admin:" -ForegroundColor Yellow
                Write-Host "   Copy contents of:" -ForegroundColor Yellow
                Write-Host "     <CUDA_PATH>\extras\visual_studio_integration\MSBuildExtensions" -ForegroundColor Yellow
                Write-Host "   into:" -ForegroundColor Yellow
                Write-Host "     <VS_PATH>\MSBuild\Microsoft\VC\v170\BuildCustomizations" -ForegroundColor Yellow
            }
        }
    }

    # -- Step C: Build llama-server --
    $NumCpu = [Environment]::ProcessorCount
    if ($NumCpu -lt 1) { $NumCpu = 4 }

    if ($BuildOk) {
        Write-Host ""
        Write-Host "--- cmake build (llama-server) ---" -ForegroundColor Cyan
        Write-Host "   Parallel jobs: $NumCpu" -ForegroundColor Gray
        Write-Host ""

        $output = cmake --build $BuildDir --config Release --target llama-server -j $NumCpu 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            $BuildOk = $false
            $FailedStep = "cmake build (llama-server)"
            Write-Host $output -ForegroundColor Red
        }
    }

    # -- Step D: Build llama-quantize (optional, best-effort) --
    if ($BuildOk) {
        Write-Host ""
        Write-Host "--- cmake build (llama-quantize) ---" -ForegroundColor Cyan
        $output = cmake --build $BuildDir --config Release --target llama-quantize -j $NumCpu 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   [WARN] llama-quantize build failed (GGUF export may be unavailable)" -ForegroundColor Yellow
            Write-Host $output -ForegroundColor Yellow
        }
    }

    # Restore ErrorActionPreference
    $ErrorActionPreference = $prevEAP

    # Stop timer
    $totalSw.Stop()
    $totalMin = [math]::Floor($totalSw.Elapsed.TotalMinutes)
    $totalSec = [math]::Round($totalSw.Elapsed.TotalSeconds % 60, 1)

    # -- Summary --
    Write-Host ""
    if ($BuildOk -and (Test-Path $LlamaServerBin)) {
        Write-Host "[OK] llama-server built at $LlamaServerBin" -ForegroundColor Green
        $QuantizeBin = Join-Path $BuildDir "bin\Release\llama-quantize.exe"
        if (Test-Path $QuantizeBin) {
            Write-Host "[OK] llama-quantize available for GGUF export" -ForegroundColor Green
        }
        Write-Host "   Build time: ${totalMin}m ${totalSec}s" -ForegroundColor Cyan
    } else {
        # Check alternate paths (some cmake generators don't use Release subdir)
        $altBin = Join-Path $BuildDir "bin\llama-server.exe"
        if ($BuildOk -and (Test-Path $altBin)) {
            Write-Host "[OK] llama-server built at $altBin" -ForegroundColor Green
            Write-Host "   Build time: ${totalMin}m ${totalSec}s" -ForegroundColor Cyan
        } else {
            Write-Host "[FAILED] llama.cpp build failed at step: $FailedStep (${totalMin}m ${totalSec}s)" -ForegroundColor Red
            Write-Host "         To retry: delete $LlamaCppDir and re-run setup." -ForegroundColor Yellow
            exit 1
        }
    }
}

# ============================================
# Done
# ============================================
Write-Host ""
Write-Host "+===============================================+" -ForegroundColor Green
Write-Host "|           Setup Complete!                     |" -ForegroundColor Green
Write-Host "|                                               |" -ForegroundColor Green
Write-Host "|  Launch with:                                 |" -ForegroundColor Green
Write-Host "|    unsloth studio -H 0.0.0.0 -p 8000          |" -ForegroundColor Green
Write-Host "|                                               |" -ForegroundColor Green
Write-Host "+===============================================+" -ForegroundColor Green
