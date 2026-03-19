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
    Default output is minimal (step/substep), aligned with studio/setup.sh.

    FULL / LEGACY LOGGING (defensible audit trail, multi-line [OK]/[WARN]/paths):
      unsloth studio setup --verbose
      (sets UNSLOTH_VERBOSE=1; same as install_python_stack.py)
      Or:  $env:UNSLOTH_VERBOSE='1'; powershell -File .\studio\setup.ps1
      Or:  .\setup.ps1 --verbose

    Why WSL/bash can look richer than PowerShell 5.1:
      - setup.sh uses printf with 256-color ANSI (e.g. 38;5;150).
      - Plain Write-Host -ForegroundColor uses ~16 ConsoleColors unless VT is on.
      - With VT, step/substep use the same C_DIM/C_OK/C_WARN/C_ERR codes as setup.sh.
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $ScriptDir

# Same as: unsloth studio setup --verbose  (see unsloth_cli/commands/studio.py)
foreach ($a in $args) {
    if ($a -eq '--verbose' -or $a -eq '-v') {
        $env:UNSLOTH_VERBOSE = '1'
        break
    }
}
$script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq '1')

# Detect if running from pip install (no frontend/ dir in studio)
$FrontendDir = Join-Path $ScriptDir "frontend"
$OxcValidatorDir = Join-Path $ScriptDir "backend\core\data_recipe\oxc-validator"
$IsPipInstall = -not (Test-Path $FrontendDir)

# ---------------------------------------------
# Helper functions
# ---------------------------------------------

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

    # Fallback: no version constraint - pick latest or whatever is available

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
# Returns $false if the flag isn't supported (old toolkit) - safer to reject
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
# capability: e.g. driver "CUDA Version: 12.9" -> cu128 (not cu130).
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

# ---------------------------------------------
# Output style (aligned with studio/setup.sh: step / substep)
# UNSLOTH_VERBOSE=1 restores pre-minimal multi-line diagnostics (see .NOTES).
# ---------------------------------------------
# U+2500 x52 (same count as setup.sh RULE); codepoint form avoids UTF-8 literal mojibake in PS 5.1
$Rule = [string]::new([char]0x2500, 52)

# Enable ANSI (256-color) on Windows 10+ console when NO_COLOR unset -- matches bash look more closely.
function Enable-StudioVirtualTerminal {
    if ($env:NO_COLOR) { return $false }
    try {
        Add-Type -Namespace StudioVT -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
'@ -ErrorAction Stop
        $h = [StudioVT.Native]::GetStdHandle(-11)
        [uint32]$mode = 0
        if (-not [StudioVT.Native]::GetConsoleMode($h, [ref]$mode)) { return $false }
        $mode = $mode -bor 0x0004
        return [StudioVT.Native]::SetConsoleMode($h, $mode)
    } catch {
        return $false
    }
}
$script:StudioVtOk = Enable-StudioVirtualTerminal

# Same 256-color codes as studio/setup.sh: C_TITLE / C_DIM / C_OK / C_WARN / C_ERR
function Get-StudioAnsi {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('Title', 'Dim', 'Ok', 'Warn', 'Err', 'Reset')]
        [string]$Kind
    )
    $e = [char]27
    switch ($Kind) {
        'Title' { return "${e}[38;5;150m" }
        'Dim'   { return "${e}[38;5;245m" }
        'Ok'    { return "${e}[38;5;108m" }
        'Warn'  { return "${e}[38;5;136m" }
        'Err'   { return "${e}[91m" }
        'Reset' { return "${e}[0m" }
    }
}

function Write-SetupVerboseDetail {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Color = "Gray"
    )
    if (-not $script:UnslothVerbose) { return }
    if ($script:StudioVtOk -and -not $env:NO_COLOR) {
        $ansi = switch ($Color) {
            'Green' { (Get-StudioAnsi Ok) }
            'Gray' { (Get-StudioAnsi Dim) }
            'DarkGray' { (Get-StudioAnsi Dim) }
            'Yellow' { (Get-StudioAnsi Warn) }
            'Cyan' { (Get-StudioAnsi Title) }
            'Red' { (Get-StudioAnsi Err) }
            default { (Get-StudioAnsi Dim) }
        }
        Write-Host ($ansi + $Message + (Get-StudioAnsi Reset))
    } else {
        $fc = switch ($Color) {
            'Green' { 'DarkGreen' }
            'Gray' { 'DarkGray' }
            'Cyan' { 'Green' }
            default { $Color }
        }
        Write-Host $Message -ForegroundColor $fc
    }
}

function step {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Value,
        [string]$Color = "Green"
    )
    if ($script:StudioVtOk -and -not $env:NO_COLOR) {
        $dim = Get-StudioAnsi Dim
        $rst = Get-StudioAnsi Reset
        $val = switch ($Color) {
            'Green' { Get-StudioAnsi Ok }
            'Yellow' { Get-StudioAnsi Warn }
            'Red' { Get-StudioAnsi Err }
            'DarkGray' { Get-StudioAnsi Dim }
            default { Get-StudioAnsi Ok }
        }
        $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
        Write-Host ("  {0}{1}{2}{3}{4}{2}" -f $dim, $padded, $rst, $val, $Value)
    } else {
        $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
        Write-Host ("  {0}" -f $padded) -NoNewline -ForegroundColor DarkGray
        $fc = switch ($Color) {
            'Green' { 'DarkGreen' }
            'Yellow' { 'Yellow' }
            'Red' { 'Red' }
            'DarkGray' { 'DarkGray' }
            default { 'DarkGreen' }
        }
        Write-Host $Value -ForegroundColor $fc
    }
}

function substep {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Color = "DarkGray"
    )
    if ($script:StudioVtOk -and -not $env:NO_COLOR) {
        $msgCol = switch ($Color) {
            'Yellow' { (Get-StudioAnsi Warn) }
            default { (Get-StudioAnsi Dim) }
        }
        $pad = "".PadRight(15)
        Write-Host ("  {0}{1}{2}{3}" -f $msgCol, $pad, $Message, (Get-StudioAnsi Reset))
    } else {
        $fc = switch ($Color) {
            'Yellow' { 'Yellow' }
            default { 'DarkGray' }
        }
        Write-Host ("  {0,-15}{1}" -f "", $Message) -ForegroundColor $fc
    }
}

Write-Host ""
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    # Sloth U+1F9A5 (supplementary plane): ConvertFromUtf32; `u{1F9A5} inside "-f" breaks {1} parsing
    Write-Host ("  " + (Get-StudioAnsi Title) + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup" + (Get-StudioAnsi Reset))
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    # No VT: approximate C_TITLE (150) as green; C_OK-style text as dark green
    Write-Host ("  " + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup") -ForegroundColor Green
    Write-Host "  $Rule" -ForegroundColor DarkGray
}

# --- Phase 1: system prerequisites (winget, env) ---
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
                    if ($script:UnslothVerbose) {
                        substep "nvidia-smi: $(Split-Path $p -Parent)"
                    }
                    break
                }
            } catch {}
        }
    }
}
if (-not $HasNvidiaSmi) {
    step "gpu" "none (chat-only / GGUF)" "Yellow"
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[WARN] No NVIDIA GPU detected. Studio will run in chat-only (GGUF) mode." "Yellow"
        Write-SetupVerboseDetail "       Training and GPU inference require an NVIDIA GPU with drivers installed." "Yellow"
        Write-SetupVerboseDetail "       https://www.nvidia.com/Download/index.aspx" "Yellow"
    }
} else {
    step "gpu" "NVIDIA detected"
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[OK] NVIDIA GPU detected" "Green"
    }
}

# --- Windows long paths (deep node_modules / Python) ---
$LongPathsEnabled = $false
try {
    $regVal = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
    if ($regVal -and $regVal.LongPathsEnabled -eq 1) {
        $LongPathsEnabled = $true
    }
} catch {}

if ($LongPathsEnabled) {
    step "long paths" "enabled"
} else {
    substep "enabling long paths (UAC may prompt)..."
    try {
        $proc = Start-Process -FilePath "reg.exe" `
            -ArgumentList 'add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f' `
            -Verb RunAs -Wait -PassThru -ErrorAction Stop
        if ($proc.ExitCode -eq 0) {
            $LongPathsEnabled = $true
            step "long paths" "enabled"
        } else {
            step "long paths" "failed (exit $($proc.ExitCode))" "Yellow"
        }
    } catch {
        step "long paths" "skipped (run reg add LongPathsEnabled as Admin)" "Yellow"
        if ($script:UnslothVerbose) {
            Write-SetupVerboseDetail "       Run this manually in an Admin terminal:" "Yellow"
            Write-SetupVerboseDetail '       reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f' "Cyan"
        }
    }
}

# --- Git ---
$HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
if (-not $HasGit) {
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail "Git not found -- installing via winget..." "Yellow" }
    substep "installing git..."
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        try {
            winget install Git.Git --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Environment
            $HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
        } catch { }
    }
    if (-not $HasGit) {
        step "error" "git required -- https://git-scm.com/download/win" "Red"
        exit 1
    }
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail "[OK] Git installed: $(git --version)" "Green" }
}
step "git" "$(git --version)"

# --- CMake ---
$HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
if (-not $HasCmake) {
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail "CMake not found -- installing via winget..." "Yellow" }
    substep "installing cmake..."
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
                    if ($script:UnslothVerbose) { substep "cmake at $d (PATH)" }
                    break
                }
            }
        }
    }
    if (-not $HasCmake) {
        step "error" "cmake required -- https://cmake.org/download/" "Red"
        exit 1
    }
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail "[OK] CMake installed" "Green" }
}
step "cmake" "$(cmake --version | Select-Object -First 1)"

# --- Visual Studio Build Tools ---
$CmakeGenerator = $null
$VsInstallPath = $null
$vsResult = Find-VsBuildTools

if (-not $vsResult) {
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "Visual Studio Build Tools not found -- installing via winget..." "Yellow"
        Write-SetupVerboseDetail "   (This is a one-time install, may take several minutes)" "Gray"
    }
    substep "installing VS Build Tools (one-time, may take a while)..."
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
    step "vs" "$CmakeGenerator ($($vsResult.Source))"
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[OK] $CmakeGenerator detected via $($vsResult.Source)" "Green"
        if ($vsResult.ClExe) { Write-SetupVerboseDetail "   cl.exe: $($vsResult.ClExe)" "Gray" }
    }
} else {
    step "error" "VS Build Tools missing -- winget install Microsoft.VisualStudio.2022.BuildTools" "Red"
    exit 1
}

# --- CUDA toolkit ---
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
        if ($script:UnslothVerbose) { substep "driver CUDA max: $DriverMaxCuda" }
    }
} catch {}

$CudaArch = Get-CudaComputeCapability
if ($CudaArch -and $script:UnslothVerbose) {
    substep "compute capability: $($CudaArch.Insert($CudaArch.Length-1, '.')) (sm_$CudaArch)"
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
                if ($script:UnslothVerbose) {
                    substep "CUDA toolkit check: $tkMaj.$tkMin vs sm_$CudaArch"
                }
                $archOk = $true
                if ($CudaArch) {
                    $archOk = Test-NvccArchSupport -NvccExe $candidateNvcc -Arch $CudaArch
                    if (-not $archOk -and $script:UnslothVerbose) {
                        substep "CUDA_PATH toolkit $tkMaj.$tkMin lacks sm_$CudaArch; scanning for newer..." "Yellow"
                    }
                }
                if ($archOk) {
                    $NvccPath = $candidateNvcc
                }
            } elseif ($script:UnslothVerbose) {
                substep "CUDA_PATH exceeds driver max ($DriverMaxCuda)" "Yellow"
            }
        }
    }

    # --- Step 2: Fall back to scanning side-by-side installs ---
    if (-not $NvccPath) {
        $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
        if ($NvccPath -and $existingCudaPath) {
            $selectedRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
            if ($existingCudaPath.TrimEnd('\') -ne $selectedRoot.TrimEnd('\') -and $script:UnslothVerbose) {
                substep "CUDA_PATH -> $selectedRoot"
            }
        }
        if (-not $NvccPath) {
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
    substep "installing CUDA toolkit (winget)..."
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
                $prevEAPCuda = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                winget install --id=Nvidia.CUDA --version=$BestVersion -e --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
                $ErrorActionPreference = $prevEAPCuda
                Refresh-Environment
                $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
            } else {
                step "cuda" "no winget package <= $DriverMaxCuda" "Yellow"
            }
        } else {
            winget install --id=Nvidia.CUDA -e --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Environment
            $NvccPath = Find-Nvcc
        }
    }
}

if (-not $NvccPath) {
    $cudaHint = if ($DriverMaxCuda) { "<= $DriverMaxCuda" } else { "(driver-matched)" }
    step "error" "nvcc not found -- install CUDA $cudaHint" "Red"
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
if ($script:UnslothVerbose) { substep "CUDA_PATH -> $CudaToolkitRoot" }
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
    if ($script:UnslothVerbose) { substep "$cudaPathVerVar set" }
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
    if ($script:UnslothVerbose) { substep "CUDA bin on user PATH" }
}

# -- Ensure CUDA <-> Visual Studio integration files exist --
# When CUDA is installed before VS Build Tools (or VS is reinstalled after CUDA),
# the MSBuild .targets/.props files that let VS compile .cu files are missing.
# cmake fails with "No CUDA toolset found". Fix: copy from CUDA extras dir.
if ($VsInstallPath -and $CudaToolkitRoot) {
    $vsCustomizations = Join-Path $VsInstallPath "MSBuild\Microsoft\VC\v170\BuildCustomizations"
    $cudaExtras = Join-Path $CudaToolkitRoot "extras\visual_studio_integration\MSBuildExtensions"
    if ((Test-Path $cudaExtras) -and (Test-Path $vsCustomizations)) {
        $hasTargets = Get-ChildItem $vsCustomizations -Filter "CUDA *.targets" -ErrorAction SilentlyContinue
        if (-not $hasTargets) {
            substep "CUDA/VS MSBuild integration (copying .targets)..."
            try {
                Copy-Item "$cudaExtras\*" $vsCustomizations -Force -ErrorAction Stop
            } catch {
                try {
                    $copyCmd = "Copy-Item '$cudaExtras\*' '$vsCustomizations' -Force"
                    Start-Process powershell -ArgumentList "-NoProfile -Command $copyCmd" -Verb RunAs -Wait -ErrorAction Stop
                    $hasTargetsRetry = Get-ChildItem $vsCustomizations -Filter "CUDA *.targets" -ErrorAction SilentlyContinue
                    if (-not $hasTargetsRetry) { throw "Copy did not produce .targets files" }
                } catch {
                    step "cuda/vs" "targets not installed (llama build may fail)" "Yellow"
                    substep "copy $cudaExtras -> $vsCustomizations" "Yellow"
                }
            }
        }
    }
}

step "cuda" $NvccPath
if (-not $CudaArch) { step "gpu arch" "unknown (cmake defaults)" "Yellow" }
if ($script:UnslothVerbose) {
    Write-SetupVerboseDetail "[OK] CUDA Toolkit: $NvccPath" "Green"
    Write-SetupVerboseDetail "   CUDA_PATH      = $CudaToolkitRoot" "Gray"
    Write-SetupVerboseDetail "   CudaToolkitDir = $CudaToolkitRoot\" "Gray"
    if (-not $CudaArch) {
        Write-SetupVerboseDetail "   [WARN] Could not detect compute capability -- cmake will use defaults" "Yellow"
    }
}
} else {
    step "cuda" "skipped (no NVIDIA GPU)" "Yellow"
}

# --- Node.js / npm (frontend build only) ---
if ($IsPipInstall) {
    step "frontend" "bundled (pip install)"
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
                $NeedNode = $false
            } else {
                substep "node/npm below v20 / npm11; upgrading..." "Yellow"
            }
        }
    } catch {
        substep "node/npm not on PATH"
    }

    if ($NeedNode) {
        if ($script:UnslothVerbose) {
            Write-SetupVerboseDetail "Node.js not on PATH or below Node 20 / npm 11 -- installing LTS via winget (OpenJS.NodeJS.LTS)." "Yellow"
        }
        substep "installing Node LTS (winget)..."
        try {
            winget install OpenJS.NodeJS.LTS --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Environment
        } catch {
            step "error" "Node install failed -- https://nodejs.org/" "Red"
            exit 1
        }
    }

    step "node" "$(node -v) | npm $(npm -v)"
}

# --- Python (3.11 - 3.13) ---
$HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$PythonOk = $false

if ($HasPython) {
    $PyVer = python --version 2>&1
    if ($PyVer -match "(\d+)\.(\d+)") {
        $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
        if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
            $PythonOk = $true
        } else {
            step "error" "Python $PyVer not in 3.11-3.13" "Red"
            exit 1
        }
    }
} else {
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail "Python not found -- installing Python 3.12 via winget..." "Yellow" }
    substep "installing Python 3.12 (winget)..."
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        winget install -e --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
        Refresh-Environment
    }
    $HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $HasPython) {
        step "error" "Python required -- https://python.org/downloads/" "Red"
        exit 1
    }
    $PythonOk = $true
}
if ($PythonOk) { step "python" "$(python --version 2>&1)" }
if ($script:UnslothVerbose -and $PythonOk) {
    Write-SetupVerboseDetail "[OK] Python $(python --version 2>&1)" "Green"
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
        if ($script:UnslothVerbose) {
            Write-SetupVerboseDetail "   Persisted Python Scripts dir to user PATH: $ScriptsDir" "Gray"
        }
    }
}

Write-Host ""
if ($script:UnslothVerbose) {
    Write-SetupVerboseDetail "--- System prerequisites ready ---" "Green"
    Write-Host ""
}

# --- Phase 2: frontend ---
$DistDir = Join-Path $FrontendDir "dist"
# Skip build if dist/ exists and no tracked input is newer than dist/.
# Checks src/, public/, package.json, config files -- not just src/.
$NeedFrontendBuild = $true
if ($IsPipInstall) {
    $NeedFrontendBuild = $false
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
        step "frontend" "up to date"
    } else {
        substep "frontend sources changed; rebuilding..."
    }
}
if ($NeedFrontendBuild -and -not $IsPipInstall) {
    substep "building frontend..."
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "--- Phase 2: frontend (npm) ---" "Green"
        Write-SetupVerboseDetail "Tailwind v4: venv .gitignore with '*' blocks scanning; we temporarily rename matching .gitignore up the tree." "Gray"
    }

    # -- Tailwind v4 .gitignore workaround --
    # Tailwind v4's oxide scanner respects .gitignore in parent directories.
    # Python venvs create a .gitignore with "*" (ignore everything), which
    # prevents Tailwind from scanning .tsx source files for class names.
    # Temporarily hide any such .gitignore during the build, then restore it.
    $HiddenGitignores = @()
    $WalkDir = (Get-Item $FrontendDir).Parent.FullName
    while ($WalkDir -and $WalkDir -ne [System.IO.Path]::GetPathRoot($WalkDir)) {
        $gi = Join-Path $WalkDir ".gitignore"
        if (Test-Path $gi) {
            $content = Get-Content $gi -Raw -ErrorAction SilentlyContinue
            if ($content -and ($content.Trim() -match '^\*$')) {
                $hidden = "$gi._twbuild"
                Rename-Item -Path $gi -NewName (Split-Path $hidden -Leaf) -Force
                $HiddenGitignores += $gi
                if ($script:UnslothVerbose) { substep "hide $gi for Tailwind" }
            }
        }
        $WalkDir = Split-Path $WalkDir -Parent
    }

    # npm writes warnings to stderr; lower ErrorActionPreference so PS doesn't
    # treat them as terminating errors (same pattern as the pip section below).
    $prevEAP_npm = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $FrontendDir
    $npmInstallLog = npm install 2>&1 | Out-String
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail $npmInstallLog "DarkGray" }
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
        step "error" "npm install failed ($LASTEXITCODE); run npm install in frontend/" "Red"
        Write-Host $npmInstallLog -ForegroundColor Red
        exit 1
    }
    $npmBuildLog = npm run build 2>&1 | Out-String
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail $npmBuildLog "DarkGray" }
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
        step "error" "npm run build failed ($LASTEXITCODE)" "Red"
        Write-Host $npmBuildLog -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_npm

    # -- Restore hidden .gitignore files --
    foreach ($gi in $HiddenGitignores) {
        Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue
    }

    # -- Validate CSS output --
    $CssFiles = Get-ChildItem (Join-Path $DistDir "assets") -Filter "*.css" -ErrorAction SilentlyContinue
    $MaxCssSize = ($CssFiles | Measure-Object -Property Length -Maximum).Maximum
    if ($MaxCssSize -lt 100000) {
        step "frontend" "built (CSS small; check .gitignore / Tailwind)" "Yellow"
    } else {
        step "frontend" "built"
    }
}

if (Test-Path $OxcValidatorDir) {
    substep "oxc-validator runtime..."
    $prevEAP_oxc = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $OxcValidatorDir
    $oxcNpmLog = npm install 2>&1 | Out-String
    if ($script:UnslothVerbose) { Write-SetupVerboseDetail $oxcNpmLog "DarkGray" }
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_oxc
        step "error" "oxc-validator npm install failed ($LASTEXITCODE)" "Red"
        Write-Host $oxcNpmLog -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_oxc
    step "oxc-validator" "ready"
}

# --- Phase 3: Python venv + deps ---
Write-Host ""

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
    step "error" "no Python 3.11-3.13 on PATH" "Red"
    exit 1
}

# Always create a .venv for isolation -- even for pip installs.
# Created in the repo root (parent of studio/).
$VenvDir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv"
if (-not (Test-Path $VenvDir)) {
    substep "creating venv ($PythonCmd)..."
    & $PythonCmd -m venv $VenvDir
} else {
    substep "reusing existing venv"
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
    substep "installing uv..."
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

if ($script:UnslothVerbose) {
    Write-SetupVerboseDetail "--- Phase 3: PyTorch + Python stack ---" "Green"
    Write-Host ""
}

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
if ($script:UnslothVerbose) { substep "TORCHINDUCTOR_CACHE_DIR=$TorchCacheDir" }

if ($HasNvidiaSmi) {
    $CuTag = Get-PytorchCudaTag
    substep "PyTorch + CUDA ($CuTag) ~2.8GB..."
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "   PyPI default torch is CPU-only on Windows; using CUDA wheels:" "Gray"
        Write-SetupVerboseDetail "   https://download.pytorch.org/whl/$CuTag" "Cyan"
    }
    $output = Fast-Install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CuTag" | Out-String
    if ($LASTEXITCODE -ne 0) {
        step "error" "PyTorch CUDA install failed ($LASTEXITCODE)" "Red"
        Write-Host $output -ForegroundColor Red
        exit 1
    }
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[OK] torch / torchvision / torchaudio (CUDA $CuTag)" "Green"
        Write-SetupVerboseDetail ($output.TrimEnd()) "DarkGray"
    }
    step "pytorch" "CUDA $CuTag"

    substep "triton-windows (torch.compile)..."
    $output = Fast-Install "triton-windows<3.7" | Out-String
    if ($LASTEXITCODE -ne 0) {
        step "triton" "install failed (torch.compile may not work)" "Yellow"
        if ($script:UnslothVerbose) { Write-Host $output -ForegroundColor Yellow }
    } else {
        step "triton" "ok"
        if ($script:UnslothVerbose) { Write-SetupVerboseDetail ($output.TrimEnd()) "DarkGray" }
    }
} else {
    substep "PyTorch (CPU)..."
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "   No NVIDIA GPU: installing CPU wheels from https://download.pytorch.org/whl/cpu" "Gray"
    }
    $output = Fast-Install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu" | Out-String
    if ($LASTEXITCODE -ne 0) {
        step "error" "PyTorch CPU install failed ($LASTEXITCODE)" "Red"
        Write-Host $output -ForegroundColor Red
        exit 1
    }
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[OK] torch / torchvision / torchaudio (CPU)" "Green"
        Write-SetupVerboseDetail ($output.TrimEnd()) "DarkGray"
    }
    step "pytorch" "CPU"
}

# Ordered heavy dependency installation -- shared cross-platform script
substep "install_python_stack.py..."
python "$PSScriptRoot\install_python_stack.py"
# Restore ErrorActionPreference after pip/python work
$ErrorActionPreference = $prevEAP

# -- Pre-install transformers 5.x into .venv_t5/ --
# Models like GLM-4.7-Flash need transformers>=5.3.0. Instead of pip-installing
# at runtime (slow, ~10-15s), we pre-install into a separate directory.
# The training subprocess just prepends .venv_t5/ to sys.path -- instant switch.
substep "transformers 5.x -> .venv_t5..."
$VenvT5Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5"
if (Test-Path $VenvT5Dir) { Remove-Item -Recurse -Force $VenvT5Dir }
New-Item -ItemType Directory -Path $VenvT5Dir -Force | Out-Null
$prevEAP_t5 = $ErrorActionPreference
$ErrorActionPreference = "Continue"
foreach ($pkg in @("transformers==5.3.0", "huggingface_hub==1.7.1", "hf_xet==1.4.2")) {
    $output = Fast-Install --target $VenvT5Dir --no-deps $pkg | Out-String
    if ($LASTEXITCODE -ne 0) {
        step "error" ".venv_t5 $pkg failed" "Red"
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
# tiktoken is needed by Qwen-family tokenizers -- install with deps since
# regex/requests may be missing on Windows
$output = Fast-Install --target $VenvT5Dir tiktoken | Out-String
if ($LASTEXITCODE -ne 0) {
    step "tiktoken" "optional install failed (Qwen may warn)" "Yellow"
}
$ErrorActionPreference = $prevEAP_t5
step "transformers" "5.x pre-installed"

# --- OpenSSL dev (llama-server HTTPS) ---
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
    step "openssl" "found"
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "[OK] OpenSSL headers/libs: $OpenSslRoot" "Green"
        Write-SetupVerboseDetail "   (llama-server -hf uses HTTPS; cmake: OPENSSL_ROOT_DIR)" "Gray"
    }
} else {
    substep "OpenSSL dev (winget)..."
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        winget install -e --id ShiningLight.OpenSSL.Dev --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
        foreach ($root in $OpenSslRoots) {
            if (Test-Path (Join-Path $root 'include\openssl\ssl.h')) {
                $OpenSslRoot = $root
                $OpenSslAvailable = $true
                break
            }
        }
    }
    if (-not $OpenSslAvailable) {
        step "openssl" "missing (llama HTTPS off)" "Yellow"
        if ($script:UnslothVerbose) {
            Write-SetupVerboseDetail "   GGUF chat works; -hf model download needs OpenSSL dev (ShiningLight.OpenSSL.Dev)" "Yellow"
        }
    } else {
        step "openssl" "installed"
        if ($script:UnslothVerbose -and $OpenSslRoot) {
            Write-SetupVerboseDetail "[OK] OpenSSL headers/libs: $OpenSslRoot" "Green"
        }
    }
}

# --- Phase 4: llama.cpp (GGUF) ---
# Builds at ~/.unsloth/llama.cpp - a single shared location under the user's
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
            substep "rebuild llama-server (GPU now available)" "Yellow"
            $NeedRebuild = $true
        } elseif (-not $HasNvidiaSmi -and $cachedCuda) {
            substep "rebuild llama-server (no GPU)" "Yellow"
            $NeedRebuild = $true
        }
    }
}

if ((Test-Path $LlamaServerBin) -and -not $NeedRebuild) {
    step "llama.cpp" "cached"
} elseif (-not $HasCmakeForBuild) {
    if (-not $HasNvidiaSmi) {
        step "error" "cmake required for llama-server (GGUF chat)" "Red"
        exit 1
    }
    step "llama.cpp" "skipped (no cmake)" "Yellow"
} else {
    if ($script:UnslothVerbose) {
        Write-SetupVerboseDetail "--- Phase 4: llama.cpp (cmake + MSBuild) ---" "Green"
        Write-SetupVerboseDetail "   Repo: $LlamaCppDir | Build: $BuildDir" "Gray"
        Write-Host ""
    }
    if ($HasNvidiaSmi) {
        substep "building llama.cpp (CUDA) ~5-10 min..."
    } else {
        substep "building llama.cpp (CPU) ~5-10 min..."
    }

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

    # Re-sanitize CUDA_PATH_V* vars - Refresh-Environment (called during
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
        substep "git pull llama.cpp..."
        git -C $LlamaCppDir pull 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -and $script:UnslothVerbose) {
            substep "git pull failed; using tree on disk" "Yellow"
        }
    } else {
        substep "git clone llama.cpp..."
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
        substep "cmake configure..."

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
                        step "cuda arch" "sm_$maxArch (PTX JIT for sm_$CudaArch)" "Yellow"
                    }
                    # else: omit flag entirely, let cmake pick defaults
                }
            }
        } else {
            $CmakeArgs += '-DGGML_CUDA=OFF'
        }

        if ($script:UnslothVerbose) {
            Write-SetupVerboseDetail ("   cmake " + ($CmakeArgs -join ' ')) "Gray"
        }

        $cmakeOutput = cmake @CmakeArgs 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0 -and $script:UnslothVerbose) {
            Write-SetupVerboseDetail $cmakeOutput.TrimEnd() "DarkGray"
        }
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
        substep "cmake build: llama-server (-j $NumCpu)..."

        $output = cmake --build $BuildDir --config Release --target llama-server -j $NumCpu 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            $BuildOk = $false
            $FailedStep = "cmake build (llama-server)"
            Write-Host $output -ForegroundColor Red
        }
    }

    # -- Step D: Build llama-quantize (optional, best-effort) --
    if ($BuildOk) {
        substep "cmake build: llama-quantize..."
        $output = cmake --build $BuildDir --config Release --target llama-quantize -j $NumCpu 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            step "llama-quantize" "skipped (export may lack quant)" "Yellow"
            if ($script:UnslothVerbose) { Write-Host $output -ForegroundColor Yellow }
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
        step "llama.cpp" "built"
        $QuantizeBin = Join-Path $BuildDir "bin\Release\llama-quantize.exe"
        if (Test-Path $QuantizeBin) {
            step "llama-quantize" "built"
        }
        step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
    } else {
        # Check alternate paths (some cmake generators don't use Release subdir)
        $altBin = Join-Path $BuildDir "bin\llama-server.exe"
        if ($BuildOk -and (Test-Path $altBin)) {
            step "llama.cpp" "built"
            step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
        } else {
            step "llama.cpp" "failed: $FailedStep (${totalMin}m ${totalSec}s)" "Red"
            substep "retry: remove $LlamaCppDir and re-run setup" "Yellow"
            exit 1
        }
    }
}

# --- Footer (matches setup.sh) ---
Write-Host ""
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
    Write-Host ("  {0}Unsloth Studio Installed{1}" -f (Get-StudioAnsi Title), (Get-StudioAnsi Reset))
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    Write-Host "  $Rule" -ForegroundColor DarkGray
    Write-Host "  Unsloth Studio Installed" -ForegroundColor Green
    Write-Host "  $Rule" -ForegroundColor DarkGray
}
step "launch" "unsloth studio -H 0.0.0.0 -p 8888"
Write-Host ""
