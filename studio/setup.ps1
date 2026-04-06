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

    FULL / LEGACY LOGGING (defensible audit trail, detailed multi-line output):
      unsloth studio setup --verbose
      Or:  $env:UNSLOTH_VERBOSE='1'; powershell -File .\studio\setup.ps1
      Or:  .\setup.ps1 --verbose
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $ScriptDir

# --------------------------------------------------------------------------
#  Maintainer-editable defaults
#  Change these in the GitHub-hosted script so users get updated defaults.
#  User env vars always override these baked-in values.
# --------------------------------------------------------------------------
# Prefer "latest" over "master" -- "master" bypasses the prebuilt resolver
# (no matching GitHub release), forces a source build, and causes HTTP 422
# errors. Only use "master" temporarily when the latest release is missing
# support for a new model architecture.
$DefaultLlamaPrForce = ""
$DefaultLlamaSource = "https://github.com/ggml-org/llama.cpp"
$DefaultLlamaTag = "latest"
$DefaultLlamaForceCompileRef = "master"

# Verbose can be enabled either by CLI flag or by UNSLOTH_VERBOSE=1.
$script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq '1')
foreach ($a in $args) {
    if ($a -eq '--verbose' -or $a -eq '-v') {
        $script:UnslothVerbose = $true
        break
    }
}
# Propagate to child processes (e.g. install_python_stack.py) so they
# also respect verbose mode. Process-scoped -- does not persist.
if ($script:UnslothVerbose) {
    $env:UNSLOTH_VERBOSE = '1'
}
$script:LlamaCppDegraded = $false

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
    # Merge: venv Scripts (if active) > Machine > User > current $env:Path. Dedup raw+expanded.
    $venvScripts = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts' } else { $null }
    $sources = @()
    if ($venvScripts) { $sources += $venvScripts }
    $sources += @($machinePath, $userPath, $env:Path)
    $merged = ($sources | Where-Object { $_ }) -join ';'
    $seen = @{}
    $unique = New-Object System.Collections.Generic.List[string]
    foreach ($p in $merged -split ";") {
        $rawKey = $p.Trim().Trim('"').TrimEnd("\").ToLowerInvariant()
        $expKey = [Environment]::ExpandEnvironmentVariables($p).Trim().Trim('"').TrimEnd("\").ToLowerInvariant()
        if ($rawKey -and -not $seen.ContainsKey($rawKey) -and -not $seen.ContainsKey($expKey)) {
            $seen[$rawKey] = $true
            if ($expKey -and $expKey -ne $rawKey) { $seen[$expKey] = $true }
            $unique.Add($p)
        }
    }
    $env:Path = $unique -join ";"
}

# ── Helper: safely add a directory to the persistent User PATH ──
# Direct registry access preserves REG_EXPAND_SZ (avoids dotnet/runtime#1442).
# Append (default) keeps existing tools first; Prepend for must-win entries.
function Add-ToUserPath {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [ValidateSet('Append','Prepend')]
        [string]$Position = 'Append'
    )
    try {
        $regKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
        try {
            $rawPath = $regKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            [string[]]$entries = if ($rawPath) { $rawPath -split ';' } else { @() } # string[] prevents scalar collapse
            $normalDir = $Directory.Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
            $expNormalDir = [Environment]::ExpandEnvironmentVariables($Directory).Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
            $kept = New-Object System.Collections.Generic.List[string]
            $matchIndices = New-Object System.Collections.Generic.List[int]
            for ($i = 0; $i -lt $entries.Count; $i++) {
                $stripped = $entries[$i].Trim().Trim('"')
                $rawNorm = $stripped.TrimEnd('\').ToLowerInvariant()
                $expNorm = [Environment]::ExpandEnvironmentVariables($stripped).TrimEnd('\').ToLowerInvariant()
                $isMatch = ($rawNorm -and ($rawNorm -eq $normalDir -or $rawNorm -eq $expNormalDir)) -or
                           ($expNorm -and ($expNorm -eq $normalDir -or $expNorm -eq $expNormalDir))
                if ($isMatch) {
                    $matchIndices.Add($i)
                    continue
                }
                $kept.Add($entries[$i])
            }
            $alreadyPresent = $matchIndices.Count -gt 0
            if ($alreadyPresent -and $Position -eq 'Append') { # Append: idempotent no-op
                return $false
            }
            if ($alreadyPresent -and $Position -eq 'Prepend' -and # Prepend: no-op if already at front
                $matchIndices.Count -eq 1 -and $matchIndices[0] -eq 0) {
                return $false
            }
            # One-time backup under HKCU\Software\Unsloth\PathBackup
            if ($rawPath) {
                try {
                    $backupKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Unsloth')
                    try {
                        $existingBackup = $backupKey.GetValue('PathBackup', $null)
                        if (-not $existingBackup) {
                            $backupKey.SetValue('PathBackup', $rawPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                        }
                    } finally {
                        $backupKey.Close()
                    }
                } catch { }
            }
            if (-not $rawPath) {
                Write-Host "[WARN] User PATH is empty - initializing with $Directory" -ForegroundColor Yellow
            }
            $newPath = if ($rawPath) {
                if ($Position -eq 'Prepend') {
                    (@($Directory) + $kept) -join ';'
                } else {
                    ($kept + @($Directory)) -join ';'
                }
            } else {
                $Directory
            }
            if ($newPath -ceq $rawPath) { # no actual change
                return $false
            }
            $regKey.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
            # Broadcast WM_SETTINGCHANGE via dummy env-var roundtrip.
            # [NullString]::Value avoids PS 7.5+/.NET 9 $null-to-"" coercion.
            try {
                $d = "UnslothPathRefresh_$([guid]::NewGuid().ToString('N').Substring(0,8))"
                [Environment]::SetEnvironmentVariable($d, '1', 'User')
                [Environment]::SetEnvironmentVariable($d, [NullString]::Value, 'User')
            } catch { }
            return $true
        } finally {
            $regKey.Close()
        }
    } catch {
        Write-Host "[WARN] Could not update User PATH: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

# PowerShell 5.1 compatibility helper: avoid relying on New-TemporaryFile.
function New-UnslothTemporaryFile {
    $tempPath = [System.IO.Path]::GetTempFileName()
    return Get-Item -LiteralPath $tempPath
}

function Get-InstalledLlamaPrebuiltRelease {
    param([string]$InstallDir)

    $metadataPath = Join-Path $InstallDir "UNSLOTH_PREBUILT_INFO.json"
    if (-not (Test-Path $metadataPath)) {
        return $null
    }

    try {
        $payload = Get-Content $metadataPath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }

    if (-not $payload.published_repo -or -not $payload.release_tag) {
        return $null
    }

    $message = "installed release: $($payload.published_repo)@$($payload.release_tag)"
    if ($payload.tag -and $payload.tag -ne $payload.release_tag) {
        $message += " (tag $($payload.tag))"
    }
    return $message
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
        $latest = Get-ChildItem -Directory $toolkitBase | Where-Object {
            $_.Name -match '^v(\d+)\.(\d+)'
        } | Sort-Object { [version]($_.Name -replace '^v','') } -Descending | Select-Object -First 1
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
    if (-not $smiExe) { return "cu126" }

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
            if ($major -ge 12) { return "cu124" }
            if ($major -ge 11) { return "cu118" }
            return "cpu"
        }
    } catch { }

    return "cu126"
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
# Output style (aligned with studio/setup.sh: step / substep)
# ─────────────────────────────────────────────
$Rule = [string]::new([char]0x2500, 52)

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

function Invoke-SetupCommand {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [switch]$AlwaysQuiet
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Reset to avoid stale values from prior native commands.
        $global:LASTEXITCODE = 0
        if ($script:UnslothVerbose -and -not $AlwaysQuiet) {
            # Merge stderr into stdout so progress/warning output stays visible
            # without flipping $? on successful native commands (PS 5.1 treats
            # stderr records as errors that set $? = $false even on exit code 0).
            & $Command 2>&1 | Out-Host
        } else {
            $output = & $Command 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                Write-Host $output -ForegroundColor Red
            }
        }
        return [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Write-LlamaFailureLog {
    param(
        [string]$Output,
        [int]$MaxLines = 120
    )
    if (-not $Output) { return }
    $lines = @(
        ($Output -split "`r?`n") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($lines.Count -eq 0) { return }
    if ($lines.Count -gt $MaxLines) {
        Write-Host "   Showing last $MaxLines lines:" -ForegroundColor DarkGray
        $lines = $lines | Select-Object -Last $MaxLines
    }
    foreach ($line in $lines) {
        Write-Host "   | $line" -ForegroundColor DarkGray
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

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
Write-Host ""
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    Write-Host ("  " + (Get-StudioAnsi Title) + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup" + (Get-StudioAnsi Reset))
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    Write-Host ("  " + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup") -ForegroundColor Green
    Write-Host "  $Rule" -ForegroundColor DarkGray
}

# Back up User PATH under HKCU\Software\Unsloth before any modifications.
try {
    $envKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $false)
    if ($envKey) {
        try {
            $rawPath = $envKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        } finally {
            $envKey.Close()
        }
        if ($rawPath) {
            $backupKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Unsloth')
            try {
                $existingBackup = $backupKey.GetValue('PathBackup', $null)
                if (-not $existingBackup) {
                    $backupKey.SetValue('PathBackup', $rawPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                }
            } finally {
                $backupKey.Close()
            }
        }
    }
} catch {
    Write-Host "[DEBUG] Could not back up User PATH: $($_.Exception.Message)" -ForegroundColor DarkGray
}

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
        & $nvSmiCmd.Source *> $null
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
                & $p *> $null
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
    step "gpu" "none (chat-only / GGUF)" "Yellow"
    substep "Training and GPU inference require an NVIDIA GPU with drivers installed." "Yellow"
    Write-Host ""
} else {
    step "gpu" "NVIDIA GPU detected"
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
    step "long paths" "enabled"
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
            step "long paths" "enabled (via UAC)"
        } else {
            step "long paths" "failed to enable (exit code: $($proc.ExitCode))" "Yellow"
        }
    } catch {
        step "long paths" "could not enable (UAC declined/unavailable)" "Yellow"
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
            Invoke-SetupCommand { winget install Git.Git --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
            Refresh-Environment
            $HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
        } catch { }
    }
    if (-not $HasGit) {
        Write-Host "[ERROR] Git is required but could not be installed automatically." -ForegroundColor Red
        Write-Host "        Install Git from https://git-scm.com/download/win and re-run." -ForegroundColor Red
        exit 1
    }
    step "git" "$(git --version)"
} else {
    step "git" "$(git --version)"
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
            Invoke-SetupCommand { winget install Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
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
                # Persist to user PATH (Prepend so this cmake wins over older ones).
                Add-ToUserPath -Directory $d -Position 'Prepend' | Out-Null
                $HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
                if ($HasCmake) {
                    Write-Host "   Found cmake at $d (added to PATH)" -ForegroundColor Gray
                    break
                }
            }
        }
    }
    if ($HasCmake) {
        step "cmake" "installed"
    } else {
        Write-Host "[ERROR] CMake is required but could not be installed." -ForegroundColor Red
        Write-Host "        Install CMake from https://cmake.org/download/ and re-run." -ForegroundColor Red
        exit 1
    }
} else {
    step "cmake" "$(cmake --version | Select-Object -First 1)"
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
    step "vs" "$CmakeGenerator ($($vsResult.Source))"
    if ($vsResult.ClExe) { substep "cl.exe: $($vsResult.ClExe)" }
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
        substep "driver supports up to CUDA $DriverMaxCuda"
    }
} catch {}

# Detect compute capability early so we can validate toolkit support
$CudaArch = Get-CudaComputeCapability
if ($CudaArch) {
    substep "GPU Compute Capability = $($CudaArch.Insert($CudaArch.Length-1, '.')) (sm_$CudaArch)"
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
                $archOk = $true
                if ($CudaArch) {
                    $archOk = Test-NvccArchSupport -NvccExe $candidateNvcc -Arch $CudaArch
                    if (-not $archOk) {
                        substep "CUDA_PATH toolkit (CUDA $tkMaj.$tkMin) does not support GPU arch sm_$CudaArch" "Yellow"
                        substep "Looking for a newer toolkit..." "Yellow"
                    }
                }
                if ($archOk) {
                    $NvccPath = $candidateNvcc
                    substep "using existing CUDA Toolkit at CUDA_PATH (nvcc: $NvccPath)"
                }
            } else {
                substep "CUDA_PATH ($existingCudaPath) has CUDA $tkMaj.$tkMin which exceeds driver max $DriverMaxCuda" "Yellow"
            }
        }
    }

    # --- Step 2: Fall back to scanning side-by-side installs ---
    if (-not $NvccPath) {
        $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
        if ($NvccPath) {
            substep "found compatible CUDA Toolkit (nvcc: $NvccPath)"
            if ($existingCudaPath) {
                $selectedRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
                if ($existingCudaPath.TrimEnd('\') -ne $selectedRoot.TrimEnd('\')) {
                    substep "overriding CUDA_PATH from $existingCudaPath to $selectedRoot" "Yellow"
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
                substep "Installing CUDA Toolkit $BestVersion via winget..."
                $prevEAPCuda = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                Invoke-SetupCommand { winget install --id=Nvidia.CUDA --version=$BestVersion -e --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                $ErrorActionPreference = $prevEAPCuda
                Refresh-Environment
                $NvccPath = Find-Nvcc -MaxVersion $DriverMaxCuda
                if ($NvccPath) {
                    substep "CUDA Toolkit $BestVersion installed (nvcc: $NvccPath)"
                }
            } else {
                substep "no compatible CUDA Toolkit version found in winget (need <= $DriverMaxCuda)" "Yellow"
            }
        } else {
            substep "Installing CUDA Toolkit (latest) via winget..."
            winget install --id=Nvidia.CUDA -e --source winget --accept-package-agreements --accept-source-agreements
            Refresh-Environment
            $NvccPath = Find-Nvcc
            if ($NvccPath) {
                substep "CUDA Toolkit installed (nvcc: $NvccPath)"
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
substep "Persisted CUDA_PATH=$CudaToolkitRoot to user environment"
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
    substep "Set $cudaPathVerVar (cleared other CUDA_PATH_V* vars)"
}
# Ensure nvcc's bin dir is on PATH for this process
$nvccBinDir = Split-Path $NvccPath -Parent
if ($env:PATH -notlike "*$nvccBinDir*") {
    [Environment]::SetEnvironmentVariable('PATH', "$nvccBinDir;$env:PATH", 'Process')
}
# Persist nvcc bin dir (Prepend so the driver-compatible toolkit wins).
if (Add-ToUserPath -Directory $nvccBinDir -Position 'Prepend') {
    substep "Persisted CUDA bin dir to user PATH"
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
            substep "CUDA VS integration missing -- copying .targets files..." "Yellow"
            try {
                Copy-Item "$cudaExtras\*" $vsCustomizations -Force -ErrorAction Stop
                substep "CUDA VS integration files installed"
            } catch {
                # Direct copy failed (needs admin). Try elevated copy via Start-Process.
                try {
                    $copyCmd = "Copy-Item '$cudaExtras\*' '$vsCustomizations' -Force"
                    Start-Process powershell -ArgumentList "-NoProfile -Command $copyCmd" -Verb RunAs -Wait -ErrorAction Stop
                    $hasTargetsRetry = Get-ChildItem $vsCustomizations -Filter "CUDA *.targets" -ErrorAction SilentlyContinue
                    if ($hasTargetsRetry) {
                        substep "CUDA VS integration files installed (elevated)"
                    } else {
                        throw "Copy did not produce .targets files"
                    }
                } catch {
                    substep "could not copy CUDA VS integration files" "Yellow"
                    substep "The llama.cpp build may fail with 'No CUDA toolset found'." "Yellow"
                    substep "Manual fix: copy contents of" "Yellow"
                    substep "$cudaExtras"
                    substep "into:" "Yellow"
                    substep "$vsCustomizations"
                }
            }
        }
    }
}

step "cuda" $NvccPath
substep "CUDA_PATH      = $CudaToolkitRoot"
substep "CudaToolkitDir = $CudaToolkitRoot\"

# $CudaArch was detected earlier (before toolkit selection) so it could
# influence which toolkit we picked.  Just log the final state here.
if (-not $CudaArch) {
    substep "could not detect compute capability -- cmake will use defaults" "Yellow"
}
} else {
    step "cuda" "skipped (no NVIDIA GPU detected)" "Yellow"
}

# ============================================
# 1f. Node.js / npm (skip if pip-installed or Tauri -- only needed for frontend build)
# ============================================
$SkipFrontend = ($env:SKIP_STUDIO_FRONTEND -eq "1")
if ($IsPipInstall) {
    step "frontend" "bundled (pip install)"
} elseif ($SkipFrontend) {
    step "frontend" "bundled (Tauri)"
} else {
    # setup.sh installs Node LTS (v22) via nvm. We enforce the same range here:
    # Vite 8 requires Node ^20.19.0 || >=22.12.0, npm >= 11.
    $NeedNode = $true
    try {
        $NodeVersion = (node -v 2>$null)
        $NpmVersion = (npm -v 2>$null)
        if ($NodeVersion -and $NpmVersion) {
            $NodeParts = ($NodeVersion -replace 'v','').Split('.')
            $NodeMajor = [int]$NodeParts[0]
            $NodeMinor = [int]$NodeParts[1]
            $NpmMajor = [int]$NpmVersion.Split('.')[0]

            # Vite 8: ^20.19.0 || >=22.12.0
            $NodeOk = ($NodeMajor -eq 20 -and $NodeMinor -ge 19) -or
                      ($NodeMajor -eq 22 -and $NodeMinor -ge 12) -or
                      ($NodeMajor -ge 23)
            if ($NodeOk -and $NpmMajor -ge 11) {
                substep "Node $NodeVersion and npm $NpmVersion already meet requirements."
                $NeedNode = $false
            } else {
                substep "Node $NodeVersion / npm $NpmVersion too old." "Yellow"
            }
        }
    } catch {
        substep "Node/npm not found." "Yellow"
    }

    if ($NeedNode) {
        substep "installing Node.js LTS via winget..."
        try {
            winget install OpenJS.NodeJS.LTS --source winget --accept-package-agreements --accept-source-agreements
            Refresh-Environment
        } catch {
            Write-Host "[ERROR] Could not install Node.js automatically." -ForegroundColor Red
            Write-Host "Please install Node.js >= 20 from https://nodejs.org/" -ForegroundColor Red
            exit 1
        }
    }

    step "node" "$(node -v) | npm $(npm -v)"

    # ── bun (optional, faster package installs) ──
    # Installed via npm — Node is already guaranteed above. Works on all platforms.
    if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
        substep "installing bun (faster frontend package installs)..."
        $prevEAP_bun = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        Invoke-SetupCommand { npm install -g bun } | Out-Null
        $ErrorActionPreference = $prevEAP_bun
        Refresh-Environment
        if (Get-Command bun -ErrorAction SilentlyContinue) {
            substep "bun installed ($(bun --version))"
        } else {
            substep "bun install skipped (npm will be used instead)"
        }
    } else {
        substep "bun already installed ($(bun --version))"
    }
}

# 1g. Python (>= 3.11 and < 3.14). Prefer py.exe so a 3.14 ahead of 3.13 on PATH does not trip the gate.
$HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$PyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
$PythonOk = $false
$DetectedPyVer = $null

if ($PyLauncher) {
    foreach ($minor in @("3.13", "3.12", "3.11")) {
        try {
            $out = & $PyLauncher.Source "-$minor" --version 2>&1 | Out-String
            if ($out -match 'Python (3\.\d+\.\d+)') {
                $DetectedPyVer = $Matches[1]
                # Make `python` resolvable for the rest of setup. Without this,
                # py-launcher-only installs (no python.exe on PATH) pass the gate
                # and then crash on the first bare `python` call below.
                try {
                    $resolvedExe = (& $PyLauncher.Source "-$minor" -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
                    if ($resolvedExe -and (Test-Path $resolvedExe)) {
                        $resolvedDir = Split-Path -Parent $resolvedExe
                        $alreadyOnPath = ($env:PATH -split ';' | Where-Object { $_.TrimEnd('\') -ieq $resolvedDir.TrimEnd('\') }).Count -gt 0
                        if (-not $alreadyOnPath) {
                            $env:PATH = "$resolvedDir;$env:PATH"
                        }
                        $HasPython = $true
                    }
                } catch { }
                $PythonOk = $true
                break
            }
        } catch { }
    }
}

if (-not $PythonOk -and $HasPython) {
    $PyVer = python --version 2>&1
    if ($PyVer -match "(\d+)\.(\d+)") {
        $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
        if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
            $DetectedPyVer = "$PyMajor.$PyMinor"
            $PythonOk = $true
        }
    }
}

if ($PythonOk) {
    substep "Python $DetectedPyVer"
} elseif (-not $HasPython) {
    # No `python` on PATH (and py.exe either absent or only had unsupported
    # minors). Try winget as before -- gating on $HasPython alone, not also
    # on $PyLauncher, so a launcher-only install with just 3.14 still gets
    # an automatic 3.12 install instead of a hard error.
    Write-Host "Python 3.11-3.13 not found -- installing Python 3.12 via winget..." -ForegroundColor Yellow
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
    step "python" "$(python --version 2>&1)"
    $PythonOk = $true
} else {
    # python.exe is on PATH but its version is unsupported, and py.exe (if
    # present) had no supported minor either.
    Write-Host "[ERROR] No supported Python (3.11-3.13) found on this system." -ForegroundColor Red
    Write-Host "        py.exe could not locate -3.11/-3.12/-3.13 and `python` on PATH is unsupported." -ForegroundColor Yellow
    Write-Host "        Install Python 3.12 from https://python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Add user-scheme Python Scripts dir to PATH (nt_user only, no venv fallback).
$ScriptsDir = python -c "import os, sysconfig; p = sysconfig.get_path('scripts', 'nt_user'); print(p if os.path.exists(p) else '')"
if ($LASTEXITCODE -eq 0 -and $ScriptsDir -and (Test-Path $ScriptsDir)) {
    # Append (not Prepend) -- this dir has other pip scripts; shim handles unsloth.
    if (Add-ToUserPath -Directory $ScriptsDir) {
        # Also add to current process so it's available immediately
        $ProcessPathEntries = $env:PATH.Split(';')
        if (-not ($ProcessPathEntries | Where-Object { $_.TrimEnd('\') -eq $ScriptsDir })) {
            $env:PATH = "$ScriptsDir;$env:PATH"
        }
        substep "Persisted Python Scripts dir to user PATH: $ScriptsDir"
    }
}

Write-Host ""
step "system" "prerequisites ready"
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
    step "frontend" "bundled (pip install)"
} elseif ($SkipFrontend) {
    $NeedFrontendBuild = $false
    step "frontend" "bundled (Tauri)"
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
    # Also check all top-level files (package.json, vite.config.ts, index.html, etc.)
    if (-not $NewerFile) {
        $NewerFile = Get-ChildItem -Path $FrontendDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "bun.lock" -and $_.LastWriteTime -gt $DistTime } |
            Select-Object -First 1
    }
    if (-not $NewerFile) {
        $NeedFrontendBuild = $false
        step "frontend" "up to date"
    } else {
        substep "Frontend source changed since last build -- rebuilding..." "Yellow"
    }
}
if ($NeedFrontendBuild -and -not $IsPipInstall) {
    Write-Host ""
    substep "building frontend..."

    # ── Tailwind v4 .gitignore workaround ──
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
                substep "Temporarily hiding $gi (venv .gitignore blocks Tailwind scanner)"
            }
        }
        $WalkDir = Split-Path $WalkDir -Parent
    }

    # Use bun if available (faster install), fall back to npm.
    # Bun is used only as package manager; Node runs the actual build (Vite 8).
    $prevEAP_npm = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $FrontendDir

    $UseBun = $null -ne (Get-Command bun -ErrorAction SilentlyContinue)

    # bun's package cache can become corrupt -- packages get stored with only
    # metadata but no actual content (bin/, lib/). When this happens bun install
    # exits 0 but leaves binaries missing. We validate after install and clear
    # the cache + retry once before falling back to npm.
    if ($UseBun) {
        Write-Host "   Using bun for package install (faster)" -ForegroundColor DarkGray
        $bunExit = Invoke-SetupCommand { bun install }
        # On Windows, .bin/ entries vary by package manager:
        #   npm  → tsc, tsc.cmd, tsc.ps1
        #   bun  → tsc.exe, tsc.bunx
        $hasTsc = (Test-Path "node_modules\.bin\tsc") -or (Test-Path "node_modules\.bin\tsc.cmd") -or (Test-Path "node_modules\.bin\tsc.exe") -or (Test-Path "node_modules\.bin\tsc.bunx")
        $hasVite = (Test-Path "node_modules\.bin\vite") -or (Test-Path "node_modules\.bin\vite.cmd") -or (Test-Path "node_modules\.bin\vite.exe") -or (Test-Path "node_modules\.bin\vite.bunx")
        if ($bunExit -eq 0 -and $hasTsc -and $hasVite) {
            # bun install succeeded and critical binaries are present
        } elseif ($bunExit -eq 0) {
            Write-Host "   bun install exited 0 but critical binaries are missing, clearing cache and retrying..." -ForegroundColor Yellow
            if (Test-Path "node_modules") {
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
            }
            Invoke-SetupCommand { bun pm cache rm } | Out-Null
            $bunExit = Invoke-SetupCommand { bun install }
            $hasTsc = (Test-Path "node_modules\.bin\tsc") -or (Test-Path "node_modules\.bin\tsc.cmd") -or (Test-Path "node_modules\.bin\tsc.exe") -or (Test-Path "node_modules\.bin\tsc.bunx")
            $hasVite = (Test-Path "node_modules\.bin\vite") -or (Test-Path "node_modules\.bin\vite.cmd") -or (Test-Path "node_modules\.bin\vite.exe") -or (Test-Path "node_modules\.bin\vite.bunx")
            if ($bunExit -ne 0 -or -not $hasTsc -or -not $hasVite) {
                Write-Host "   bun retry failed, falling back to npm" -ForegroundColor Yellow
                if (Test-Path "node_modules") {
                    Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
                }
                $UseBun = $false
            }
        } else {
            substep "bun install failed (exit $bunExit), falling back to npm" "Yellow"
            if (Test-Path "node_modules") {
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
            }
            $UseBun = $false
        }
    }
    if (-not $UseBun) {
        $npmExit = Invoke-SetupCommand { npm install }
        if ($npmExit -ne 0) {
            Pop-Location
            $ErrorActionPreference = $prevEAP_npm
            foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
            Write-Host "[ERROR] npm install failed (exit code $npmExit)" -ForegroundColor Red
            Write-Host "   Try running 'npm install' manually in frontend/ to see errors" -ForegroundColor Yellow
            exit 1
        }
    }

    # Always use npm to run the build (Node runtime — avoids bun Windows runtime issues)
    $buildExit = Invoke-SetupCommand { npm run build }
    if ($buildExit -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
        Write-Host "[ERROR] npm run build failed (exit code $buildExit)" -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_npm

    # ── Restore hidden .gitignore files ──
    foreach ($gi in $HiddenGitignores) {
        Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue
    }

    # ── Validate CSS output ──
    $CssFiles = Get-ChildItem (Join-Path $DistDir "assets") -Filter "*.css" -ErrorAction SilentlyContinue
    $MaxCssSize = ($CssFiles | Measure-Object -Property Length -Maximum).Maximum
    if ($MaxCssSize -lt 100000) {
        step "frontend" "built (warning: CSS may be truncated)" "Yellow"
    } else {
        step "frontend" "built"
    }
}

if (Test-Path $OxcValidatorDir) {
    substep "installing OXC validator runtime..."
    $prevEAP_oxc = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $OxcValidatorDir
    $oxcInstallExit = Invoke-SetupCommand { npm install }
    if ($oxcInstallExit -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_oxc
        Write-Host "[ERROR] OXC validator npm install failed (exit code $oxcInstallExit)" -ForegroundColor Red
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_oxc
    step "oxc runtime" "installed"
}

# ==========================================================================
#  PHASE 3: Python environment + dependencies
# ==========================================================================
Write-Host ""
substep "setting up Python environment..."

# Find Python -- skip Anaconda/Miniconda distributions.
# Conda-bundled CPython ships modified DLL search paths that break
# torch's c10.dll loading on Windows. Standalone CPython (python.org,
# winget, uv) does not have this issue.
# Uses Get-Command -All to look past conda entries that shadow a valid
# standalone Python further down PATH, and probes py.exe (the Python
# Launcher) which reliably finds python.org installs.
#
# NOTE: A venv created from conda Python inherits conda's base_prefix
# even though the venv path itself does not contain "conda". We check
# both the executable path AND sys.base_prefix to catch this case.
$CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'
$PythonCmd = $null

# Helper: check if a Python executable is conda-based by inspecting
# both the path and sys.base_prefix (catches venvs created from conda).
function Test-IsConda {
    param([string]$Exe)
    if ($Exe -match $CondaSkipPattern) { return $true }
    try {
        $basePrefix = (& $Exe -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
        if ($basePrefix -match $CondaSkipPattern) { return $true }
    } catch { }
    return $false
}

# 1. Try the Python Launcher (py.exe) first -- most reliable on Windows.
#    py.exe is installed by python.org and resolves to standalone CPython.
$pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
if ($pyLauncher -and $pyLauncher.Source -notmatch $CondaSkipPattern) {
    foreach ($minor in @("3.13", "3.12", "3.11")) {
        try {
            $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
            if ($out -match 'Python 3\.(\d+)') {
                $pyMinor = [int]$Matches[1]
                if ($pyMinor -ge 11 -and $pyMinor -le 13) {
                    # Resolve the actual executable path so venv creation
                    # does not re-resolve back to a conda interpreter.
                    $resolvedExe = (& $pyLauncher.Source "-$minor" -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
                    if ($resolvedExe -and (Test-Path $resolvedExe) -and -not (Test-IsConda $resolvedExe)) {
                        $PythonCmd = $resolvedExe
                        break
                    }
                }
            }
        } catch { }
    }
}

# 2. Fall back to scanning python3.x / python3 / python on PATH.
#    Use Get-Command -All to look past conda entries.
if (-not $PythonCmd) {
    foreach ($candidate in @("python3.13", "python3.12", "python3.11", "python3", "python")) {
        foreach ($cmdInfo in @(Get-Command $candidate -All -ErrorAction SilentlyContinue)) {
            try {
                if (-not $cmdInfo.Source) { continue }
                if ($cmdInfo.Source -like "*\WindowsApps\*") { continue }
                if (Test-IsConda $cmdInfo.Source) {
                    substep "skipping $($cmdInfo.Source) (conda Python breaks torch DLL loading)" "Yellow"
                    continue
                }
                $ver = & $cmdInfo.Source --version 2>&1
                if ($ver -match 'Python 3\.(\d+)') {
                    $minor = [int]$Matches[1]
                    if ($minor -ge 11 -and $minor -le 13) {
                        $PythonCmd = $cmdInfo.Source
                        break
                    }
                }
            } catch { }
        }
        if ($PythonCmd) { break }
    }
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] No standalone Python 3.11-3.13 found (conda Python is not supported)." -ForegroundColor Red
    Write-Host "        Install Python from https://python.org/downloads/ or via:" -ForegroundColor Yellow
    Write-Host "        winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}

substep "Using $PythonCmd ($(& $PythonCmd --version 2>&1))"

# The venv must already exist (created by install.ps1).
# This script (setup.ps1 / "unsloth studio update") only updates packages.
$VenvDir = Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio"

# Stale-venv detection: if the venv exists but its torch flavor no longer
# matches the current machine, repair according to invocation context.
# - install.ps1 sets UNSLOTH_INSTALL_ROLLBACK_MANAGED=1 so setup can delegate
#   to the installer-level rollback that restores the previous environment.
# - direct `unsloth studio update` keeps the pre-existing self-repair behavior.
# In no-torch mode, a missing torch package is expected.
$NoTorchMode = $env:UNSLOTH_NO_TORCH -match '^(?i:true|1|yes)$'
$InstallerManagedSetup = $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -match '^(?i:true|1|yes)$'
if ((Test-Path $VenvDir -PathType Container) -and -not $NoTorchMode) {
    $VenvPyExe = Join-Path $VenvDir "Scripts\python.exe"
    $installedTorchTag = $null
    $shouldRebuild = $false

    if (Test-Path $VenvPyExe) {
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $VenvPyExe
            $psi.Arguments = '-c "import torch; print(torch.__version__)"'
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $torchVer = $proc.StandardOutput.ReadToEnd().Trim()
            $finished = $proc.WaitForExit(30000)
            if ($finished -and $proc.ExitCode -eq 0 -and $torchVer) {
                if ($torchVer -match '\+(cu\d+)') {
                    $installedTorchTag = $Matches[1]
                } elseif ($torchVer -match '\+cpu') {
                    $installedTorchTag = "cpu"
                } else {
                    # Untagged wheel (plain "2.x.y" from PyPI) -- treat as cpu
                    $installedTorchTag = "cpu"
                }
            } else {
                if (-not $finished) { try { $proc.Kill() } catch {} }
                $shouldRebuild = $true
            }
        } catch {
            $shouldRebuild = $true
        }
    } else {
        # Missing python.exe means the venv is incomplete -- rebuild it.
        $shouldRebuild = $true
    }

    if (-not $shouldRebuild) {
        $expectedTorchTag = if ($HasNvidiaSmi) { Get-PytorchCudaTag } else { "cpu" }
        if ($installedTorchTag -and $installedTorchTag -ne $expectedTorchTag) {
            $shouldRebuild = $true
        }
    }

    if ($shouldRebuild) {
        $reason = if ($installedTorchTag) { "torch $installedTorchTag != required $expectedTorchTag" } else { "torch could not be imported" }
        if ($InstallerManagedSetup) {
            substep "Stale venv detected ($reason)." "Yellow"
            Write-Host "   [ERROR] The existing Studio environment needs repair." -ForegroundColor Red
            Write-Host "           Re-run install.ps1 so it can replace the environment safely with rollback." -ForegroundColor Yellow
            exit 1
        }
        substep "Stale venv detected ($reason) -- rebuilding..." "Yellow"
        try {
            Remove-Item $VenvDir -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Host "   [ERROR] Could not remove stale venv: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "           Close any running Studio/Python processes and re-run setup." -ForegroundColor Red
            exit 1
        }
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvDir" -ForegroundColor Red
    Write-Host "        Run install.ps1 first to create the environment:" -ForegroundColor Yellow
    Write-Host "        irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
} else {
    substep "reusing existing virtual environment at $VenvDir"
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
    substep "installing uv package manager..."
    try {
        Invoke-SetupCommand { Invoke-Expression (Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1") } | Out-Null
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

# ── Check if Python deps need updating ──
# Compare installed package version against PyPI latest.
# Skip all Python dependency work if versions match (fast update path).
$_PkgName = if ($env:STUDIO_PACKAGE_NAME) { $env:STUDIO_PACKAGE_NAME } else { "unsloth" }
$SkipPythonDeps = $false

if ($env:SKIP_STUDIO_BASE -ne "1" -and $env:STUDIO_LOCAL_INSTALL -ne "1") {
    # Only check when NOT called from install.ps1 (which just installed the package)
    $InstalledVer = try { (& python -c "from importlib.metadata import version; print(version('$_PkgName'))" 2>$null | Out-String).Trim() } catch { "" }
    $LatestVer = ""
    try {
        $pypiJson = Invoke-RestMethod -Uri "https://pypi.org/pypi/$_PkgName/json" -TimeoutSec 5 -ErrorAction Stop
        $LatestVer = "$($pypiJson.info.version)".Trim()
    } catch { }

    if ($InstalledVer -and $LatestVer -and ($InstalledVer -eq $LatestVer)) {
        step "python" "$_PkgName $InstalledVer is up to date"
        $SkipPythonDeps = $true
    } elseif ($InstalledVer -and $LatestVer) {
        substep "$_PkgName $InstalledVer -> $LatestVer available, updating..."
    } elseif (-not $LatestVer) {
        substep "could not reach PyPI, updating to be safe..."
    }
}

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

if (-not $SkipPythonDeps) {

if ($script:UnslothVerbose) {
    Fast-Install --upgrade pip
} else {
    Fast-Install --upgrade pip | Out-Null
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
substep "TORCHINDUCTOR_CACHE_DIR set to $TorchCacheDir (avoids MAX_PATH issues)"

if ($HasNvidiaSmi) {
    $CuTag = Get-PytorchCudaTag
} else {
    $CuTag = "cpu"
}

$PyTorchWhlBase = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }

if ($CuTag -eq "cpu") {
    substep "installing PyTorch (CPU-only)..."
    if ($script:UnslothVerbose) {
        Fast-Install torch torchvision torchaudio --index-url "$PyTorchWhlBase/cpu"
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install torch torchvision torchaudio --index-url "$PyTorchWhlBase/cpu" | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-Host "[FAILED] PyTorch install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        exit 1
    }
} else {
    substep "installing PyTorch with CUDA support ($CuTag)..."
    substep "(This download is ~2.8 GB -- may take a few minutes)"
    if ($script:UnslothVerbose) {
        Fast-Install torch torchvision torchaudio --index-url "$PyTorchWhlBase/$CuTag"
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install torch torchvision torchaudio --index-url "$PyTorchWhlBase/$CuTag" | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-Host "[FAILED] PyTorch CUDA install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        exit 1
    }

    # Install Triton for Windows (enables torch.compile -- without it training can hang)
    substep "installing Triton for Windows..."
    if ($script:UnslothVerbose) {
        Fast-Install "triton-windows<3.7"
        $tritonInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install "triton-windows<3.7" | Out-String
        $tritonInstallExit = $LASTEXITCODE
    }
    if ($tritonInstallExit -ne 0) {
        substep "Triton install failed -- torch.compile may not work" "Yellow"
        Write-Host $output -ForegroundColor Yellow
    } else {
        substep "Triton for Windows installed (enables torch.compile)"
    }
}

# Ordered heavy dependency installation -- shared cross-platform script
substep "running ordered dependency installation..."
python "$PSScriptRoot\install_python_stack.py"
$stackExit = $LASTEXITCODE
# Restore ErrorActionPreference after pip/python work
$ErrorActionPreference = $prevEAP
if ($stackExit -ne 0) {
    Write-Host "[FAILED] Python dependency installation failed (exit code $stackExit)" -ForegroundColor Red
    Write-Host "   Re-run the installer or check the error above for details." -ForegroundColor Red
    exit 1
}

# ── Pre-install transformers 5.x into .venv_t5_530/ and .venv_t5_550/ ──
# Models like GLM-4.7-Flash, Qwen3 MoE need transformers>=5.3.0.
# Gemma 4 models need transformers>=5.5.0.
# Pre-install into separate directories to avoid runtime pip overhead.
# The training subprocess prepends the appropriate dir to sys.path.
Write-Host ""

# Clean up legacy single .venv_t5 directory
$VenvT5Legacy = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5"
if (Test-Path $VenvT5Legacy) { Remove-Item -Recurse -Force $VenvT5Legacy }

$prevEAP_t5 = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# --- .venv_t5_530 (transformers 5.3.0) ---
substep "pre-installing transformers 5.3.0 for newer model support..."
$VenvT5_530Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5_530"
if (Test-Path $VenvT5_530Dir) { Remove-Item -Recurse -Force $VenvT5_530Dir }
New-Item -ItemType Directory -Path $VenvT5_530Dir -Force | Out-Null
foreach ($pkg in @("transformers==5.3.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2")) {
    if ($script:UnslothVerbose) {
        Fast-Install --target $VenvT5_530Dir --no-deps $pkg
        $t5PkgExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install --target $VenvT5_530Dir --no-deps $pkg | Out-String
        $t5PkgExit = $LASTEXITCODE
    }
    if ($t5PkgExit -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5_530/" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_530Dir tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_530Dir tiktoken | Out-String
    $tiktokenInstallExit = $LASTEXITCODE
}
if ($tiktokenInstallExit -ne 0) {
    substep "Could not install tiktoken into .venv_t5_530/ -- Qwen tokenizers may fail" "Yellow"
}
step "transformers" "5.3.0 pre-installed"

# --- .venv_t5_550 (transformers 5.5.0) ---
substep "pre-installing transformers 5.5.0 for Gemma 4 support..."
$VenvT5_550Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5_550"
if (Test-Path $VenvT5_550Dir) { Remove-Item -Recurse -Force $VenvT5_550Dir }
New-Item -ItemType Directory -Path $VenvT5_550Dir -Force | Out-Null
foreach ($pkg in @("transformers==5.5.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2")) {
    if ($script:UnslothVerbose) {
        Fast-Install --target $VenvT5_550Dir --no-deps $pkg
        $t5PkgExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install --target $VenvT5_550Dir --no-deps $pkg | Out-String
        $t5PkgExit = $LASTEXITCODE
    }
    if ($t5PkgExit -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5_550/" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_550Dir tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_550Dir tiktoken | Out-String
    $tiktokenInstallExit = $LASTEXITCODE
}
if ($tiktokenInstallExit -ne 0) {
    substep "Could not install tiktoken into .venv_t5_550/ -- Qwen tokenizers may fail" "Yellow"
}
$ErrorActionPreference = $prevEAP_t5
step "transformers" "5.5.0 pre-installed"

} else {
    step "python" "dependencies up to date"
    # Restore ErrorActionPreference (was lowered for pip/python section)
    $ErrorActionPreference = $prevEAP
}

# ── Pre-install transformers 5.x into .venv_t5_530/ and .venv_t5_550/ ──
# Runs outside the deps fast-path gate so that upgrades from the legacy
# single .venv_t5 are always migrated to the tiered layout.
$VenvT5_530Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5_530"
$VenvT5_550Dir = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5_550"
$VenvT5Legacy = Join-Path $env:USERPROFILE ".unsloth\studio\.venv_t5"

$_NeedT5Install = $false
if (Test-Path $VenvT5Legacy) {
    Remove-Item -Recurse -Force $VenvT5Legacy
    $_NeedT5Install = $true
}
if (-not (Test-Path $VenvT5_530Dir)) { $_NeedT5Install = $true }
if (-not (Test-Path $VenvT5_550Dir)) { $_NeedT5Install = $true }
# Also reinstall when python deps were updated
if (-not $SkipPythonDeps) { $_NeedT5Install = $true }

if ($_NeedT5Install) {
Write-Host ""

$prevEAP_t5 = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# --- .venv_t5_530 (transformers 5.3.0) ---
substep "pre-installing transformers 5.3.0 for newer model support..."
if (Test-Path $VenvT5_530Dir) { Remove-Item -Recurse -Force $VenvT5_530Dir }
New-Item -ItemType Directory -Path $VenvT5_530Dir -Force | Out-Null
foreach ($pkg in @("transformers==5.3.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2")) {
    if ($script:UnslothVerbose) {
        Fast-Install --target $VenvT5_530Dir --no-deps $pkg
        $t5PkgExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install --target $VenvT5_530Dir --no-deps $pkg | Out-String
        $t5PkgExit = $LASTEXITCODE
    }
    if ($t5PkgExit -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5_530/" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_530Dir tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_530Dir tiktoken | Out-String
    $tiktokenInstallExit = $LASTEXITCODE
}
if ($tiktokenInstallExit -ne 0) {
    substep "Could not install tiktoken into .venv_t5_530/ -- Qwen tokenizers may fail" "Yellow"
}
step "transformers" "5.3.0 pre-installed"

# --- .venv_t5_550 (transformers 5.5.0) ---
substep "pre-installing transformers 5.5.0 for Gemma 4 support..."
if (Test-Path $VenvT5_550Dir) { Remove-Item -Recurse -Force $VenvT5_550Dir }
New-Item -ItemType Directory -Path $VenvT5_550Dir -Force | Out-Null
foreach ($pkg in @("transformers==5.5.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2")) {
    if ($script:UnslothVerbose) {
        Fast-Install --target $VenvT5_550Dir --no-deps $pkg
        $t5PkgExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install --target $VenvT5_550Dir --no-deps $pkg | Out-String
        $t5PkgExit = $LASTEXITCODE
    }
    if ($t5PkgExit -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5_550/" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_550Dir tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_550Dir tiktoken | Out-String
    $tiktokenInstallExit = $LASTEXITCODE
}
if ($tiktokenInstallExit -ne 0) {
    substep "Could not install tiktoken into .venv_t5_550/ -- Qwen tokenizers may fail" "Yellow"
}
$ErrorActionPreference = $prevEAP_t5
step "transformers" "5.5.0 pre-installed"

} # end $_NeedT5Install

# ==========================================================================
#  PHASE 3.4: Prefer prebuilt llama.cpp bundles before source build
# ==========================================================================
$UnslothHome = Join-Path $env:USERPROFILE ".unsloth"
if (-not (Test-Path $UnslothHome)) { New-Item -ItemType Directory -Force $UnslothHome | Out-Null }
$LlamaCppDir = Join-Path $UnslothHome "llama.cpp"
$NeedLlamaSourceBuild = $false
$SkipPrebuiltInstall = $false
$RequestedLlamaTag = if ($env:UNSLOTH_LLAMA_TAG) { $env:UNSLOTH_LLAMA_TAG } else { $DefaultLlamaTag }
$HelperReleaseRepo = "ggml-org/llama.cpp"
$LlamaPr = if ($env:UNSLOTH_LLAMA_PR) { $env:UNSLOTH_LLAMA_PR.Trim() } else { "" }

$LlamaPrForce = if ($env:UNSLOTH_LLAMA_PR_FORCE) { $env:UNSLOTH_LLAMA_PR_FORCE.Trim() } else { $DefaultLlamaPrForce }
$LlamaSource = $DefaultLlamaSource
if ($LlamaSource.EndsWith('.git')) { $LlamaSource = $LlamaSource.Substring(0, $LlamaSource.Length - 4) }
$ResolvedSourceUrl = $LlamaSource
$ResolvedSourceRef = $RequestedLlamaTag
$ResolvedSourceRefKind = "tag"
$ResolvedLlamaTag = $RequestedLlamaTag

if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {
    $NeedLlamaSourceBuild = $true
    $SkipPrebuiltInstall = $true
}

function Invoke-LlamaHelper {
    param(
        [string[]]$Arguments,
        [string]$StderrPath = $null
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $previousNativeErrorPreference = $null
    $restoreNativeErrorPreference = $false
    $ErrorActionPreference = "Continue"
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
        $restoreNativeErrorPreference = $true
    }

    try {
        # Capture all output (stdout + stderr) so that PowerShell does not
        # convert stderr lines into visible ErrorRecord objects.  Separate
        # stdout from stderr afterwards.
        $allOutput = & python "$PSScriptRoot\install_llama_prebuilt.py" @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $stdoutLines = @()
        $stderrLines = @()
        foreach ($line in $allOutput) {
            if ($line -is [System.Management.Automation.ErrorRecord]) {
                $stderrLines += $line.ToString()
            } else {
                $stdoutLines += $line
            }
        }
        if ($StderrPath -and $stderrLines.Count -gt 0) {
            $stderrLines | Out-File -FilePath $StderrPath -Encoding utf8
        }
        return [pscustomobject]@{
            Output = $stdoutLines
            ExitCode = $exitCode
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($restoreNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
    }
}

if ($LlamaSource -ne "https://github.com/ggml-org/llama.cpp") {
    step "llama.cpp" "custom source: $LlamaSource -- forcing source build" "Yellow"
    $NeedLlamaSourceBuild = $true
    $SkipPrebuiltInstall = $true
}

if (-not $LlamaPr -and $LlamaPrForce -and $LlamaPrForce -match '^\d+$' -and [int]$LlamaPrForce -gt 0) {
    $LlamaPr = $LlamaPrForce
    step "llama.cpp" "baked-in PR_FORCE=$LlamaPrForce" "Yellow"
}

if ($LlamaPr) {
    if ($LlamaPr -notmatch '^\d+$' -or [int]$LlamaPr -le 0) {
        Write-Host "[ERROR] UNSLOTH_LLAMA_PR=$LlamaPr is not a valid PR number" -ForegroundColor Red
        exit 1
    }
    step "llama.cpp" "UNSLOTH_LLAMA_PR=$LlamaPr -- will build from PR head" "Yellow"
    $ResolvedLlamaTag = "pr-$LlamaPr"
    $ResolvedSourceUrl = $LlamaSource
    $ResolvedSourceRef = "pr-$LlamaPr"
    $ResolvedSourceRefKind = "pull"
    $NeedLlamaSourceBuild = $true
    $SkipPrebuiltInstall = $true
}

if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {
    Write-Host ""
    substep "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt llama.cpp install" "Yellow"
    $NeedLlamaSourceBuild = $true
} elseif ($SkipPrebuiltInstall) {
    Write-Host ""
    substep "Skipping prebuilt install -- falling back to source build" "Yellow"
} else {
    Write-Host ""
    substep "installing prebuilt llama.cpp bundle (preferred path)..."
    if (Test-Path $LlamaCppDir) {
        substep "Existing llama.cpp install detected -- validating staged prebuilt update before replacement"
    }
    $prebuiltArgs = @(
            "$PSScriptRoot\install_llama_prebuilt.py",
            "--install-dir", $LlamaCppDir,
            "--llama-tag", $RequestedLlamaTag,
            "--published-repo", $HelperReleaseRepo,
            "--simple-policy"
        )
        if ($env:UNSLOTH_LLAMA_RELEASE_TAG) {
            $prebuiltArgs += @("--published-release-tag", $env:UNSLOTH_LLAMA_RELEASE_TAG)
        }
        $prevEAPPrebuilt = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $previousNativeErrorPreference = $null
        $restoreNativeErrorPreference = $false
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
            $restoreNativeErrorPreference = $true
        }
        try {
            if ($script:UnslothVerbose) {
                # Show live output in verbose mode while still capturing for error log
                $prebuiltLog = Join-Path $env:TEMP "unsloth-prebuilt-$PID.log"
                & python @prebuiltArgs 2>&1 | Tee-Object -FilePath $prebuiltLog | Out-Host
                $prebuiltExit = $LASTEXITCODE
                $prebuiltOutput = if (Test-Path $prebuiltLog) { Get-Content $prebuiltLog -Raw } else { "" }
                Remove-Item $prebuiltLog -ErrorAction SilentlyContinue
            } else {
                $prebuiltOutput = & python @prebuiltArgs 2>&1 | Out-String
                $prebuiltExit = $LASTEXITCODE
            }
        } finally {
            if ($restoreNativeErrorPreference) {
                $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
            }
        }
        $ErrorActionPreference = $prevEAPPrebuilt

        if ($prebuiltExit -eq 0) {
            if ($prebuiltOutput -match "already matches") {
                step "llama.cpp" "prebuilt up to date and validated"
            } else {
                step "llama.cpp" "prebuilt installed and validated"
            }
            $installedRelease = Get-InstalledLlamaPrebuiltRelease -InstallDir $LlamaCppDir
            if ($installedRelease) {
                substep $installedRelease
            }
        } elseif ($prebuiltExit -eq 3) {
            step "llama.cpp" "install blocked by active llama.cpp process" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path $LlamaCppDir) {
                substep "Existing install was restored" "Yellow"
            }
            substep "Close Studio or other llama.cpp users and retry" "Yellow"
            exit 3
        } else {
            step "llama.cpp" "prebuilt install failed (continuing)" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path $LlamaCppDir) {
                substep "Prebuilt update failed; existing install was restored or cleaned before source build fallback" "Yellow"
            }
            substep "Prebuilt llama.cpp path unavailable or failed validation -- falling back to source build" "Yellow"
            $NeedLlamaSourceBuild = $true
        }
}

# ==========================================================================
#  PHASE 3.5: Install OpenSSL dev (for HTTPS support in llama-server)
# ==========================================================================
# llama-server needs OpenSSL to download models from HuggingFace via -hf.
# ShiningLight.OpenSSL.Dev includes headers + libs that cmake can find.
$OpenSslAvailable = $false

if ($NeedLlamaSourceBuild) {
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
        substep "OpenSSL dev found at $OpenSslRoot"
    } else {
        Write-Host ""
        substep "installing OpenSSL dev (for HTTPS in llama-server)..."
        $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
        if ($HasWinget) {
            winget install -e --id ShiningLight.OpenSSL.Dev --accept-package-agreements --accept-source-agreements
            # Re-check after install
            foreach ($root in $OpenSslRoots) {
                if (Test-Path (Join-Path $root 'include\openssl\ssl.h')) {
                    $OpenSslRoot = $root
                    $OpenSslAvailable = $true
                    substep "OpenSSL dev installed at $OpenSslRoot"
                    break
                }
            }
        }
        if (-not $OpenSslAvailable) {
            substep "OpenSSL dev not available -- llama-server will be built without HTTPS" "Yellow"
        }
    }
} else {
    substep "OpenSSL dev install skipped -- prebuilt llama.cpp already validated" "Yellow"
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
$OriginalLlamaCppDir = $LlamaCppDir
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

if (-not $NeedLlamaSourceBuild) {
    Write-Host ""
    step "llama.cpp" "prebuilt (validated)"
} elseif ((Test-Path $LlamaServerBin) -and -not $NeedRebuild -and $RequestedLlamaTag -ne "master") {
    # Skip rebuild only for pinned tags (e.g. b8635).  When the requested
    # tag is "master" (a moving target), always rebuild so the binary picks
    # up new model architecture support (e.g. Gemma 4).
    Write-Host ""
    step "llama.cpp" "already built"
} elseif (-not $HasCmakeForBuild) {
    Write-Host ""
    if (-not $HasNvidiaSmi) {
        # CPU-only machines depend entirely on llama-server for GGUF chat -- cmake is required
        substep "CMake is required to build llama-server for GGUF chat mode." "Yellow"
        substep "Continuing setup without llama.cpp build." "Yellow"
        substep "Install CMake from https://cmake.org/download/ and re-run setup." "Yellow"
    }
    step "llama.cpp" "build skipped (cmake not available)" "Yellow"
    substep "GGUF inference and export will not be available." "Yellow"
    substep "Install CMake from https://cmake.org/download/ and re-run setup." "Yellow"
    $script:LlamaCppDegraded = $true
} else {
    Write-Host ""
    if ($HasNvidiaSmi) {
        substep "building llama.cpp with CUDA support..."
    } else {
        substep "building llama.cpp (CPU-only, no NVIDIA GPU detected)..."
    }
    substep "This typically takes 5-10 minutes on first build."
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

    if (-not $LlamaPr) {
        $ResolvedSourceUrl = $LlamaSource
        if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {
            if ($RequestedLlamaTag -eq "latest") {
                $ResolvedSourceRef = if ($env:UNSLOTH_LLAMA_FORCE_COMPILE_REF) {
                    $env:UNSLOTH_LLAMA_FORCE_COMPILE_REF
                } else {
                    $DefaultLlamaForceCompileRef
                }
                $ResolvedSourceRefKind = "branch"
            } else {
                $ResolvedSourceRef = $RequestedLlamaTag
                $ResolvedSourceRefKind = "tag"
            }
        } elseif ($RequestedLlamaTag -eq "latest") {
            $resolveTagArgs = @("--resolve-llama-tag", "latest", "--published-repo", "ggml-org/llama.cpp", "--output-format", "json")
            $resolveTagResult = Invoke-LlamaHelper -Arguments $resolveTagArgs
            $resolveTagOutput = $resolveTagResult.Output
            $resolveTagExit = $resolveTagResult.ExitCode
            if ($resolveTagExit -eq 0 -and $resolveTagOutput) {
                try {
                    $ResolvedSourceRef = (($resolveTagOutput | Out-String) | ConvertFrom-Json).llama_tag
                } catch {
                    $ResolvedSourceRef = ""
                }
            } else {
                $ResolvedSourceRef = ""
            }
            if ([string]::IsNullOrWhiteSpace($ResolvedSourceRef)) {
                $ResolvedSourceRef = "latest"
            }
            $ResolvedSourceRefKind = "tag"
        } else {
            $ResolvedSourceRef = $RequestedLlamaTag
            $ResolvedSourceRefKind = "tag"
        }
        if ([string]::IsNullOrWhiteSpace($ResolvedSourceUrl)) { $ResolvedSourceUrl = $LlamaSource }
        if ([string]::IsNullOrWhiteSpace($ResolvedSourceRef)) { $ResolvedSourceRef = $RequestedLlamaTag }
    }

    # -- Step A: Clone or pull llama.cpp --

    $UseConcreteRef = ($ResolvedSourceRef -ne "latest" -and -not [string]::IsNullOrWhiteSpace($ResolvedSourceRef))

    if (Test-Path (Join-Path $LlamaCppDir ".git")) {
        Write-Host "   Syncing llama.cpp to $ResolvedSourceRef..." -ForegroundColor Gray
        # Always sync the remote URL so switching between default/fork sources works
        Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir remote set-url origin "$ResolvedSourceUrl.git" } | Out-Null
        if ($LlamaPr) {
            $gitFetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir fetch --depth 1 origin "pull/$LlamaPr/head" }
            if ($gitFetchExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git fetch PR #$LlamaPr"
            } else {
                $gitCheckoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir checkout -B "pr-$LlamaPr" FETCH_HEAD }
                if ($gitCheckoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout PR #$LlamaPr"
                } else {
                    Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir clean -fdx } | Out-Null
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "pull") {
            $gitFetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir fetch --depth 1 origin $ResolvedSourceRef }
            if ($gitFetchExit -ne 0) {
                substep "git fetch failed -- using existing source" "Yellow"
            } else {
                $gitCheckoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir checkout -B unsloth-llama-build FETCH_HEAD }
                if ($gitCheckoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout"
                } else {
                    Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir clean -fdx } | Out-Null
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "commit") {
            $gitFetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir fetch --depth 1 origin $ResolvedSourceRef }
            if ($gitFetchExit -ne 0) {
                substep "git fetch failed -- using existing source" "Yellow"
            } else {
                $gitCheckoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir checkout -B unsloth-llama-build FETCH_HEAD }
                if ($gitCheckoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout"
                } else {
                    Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir clean -fdx } | Out-Null
                }
            }
        } elseif ($UseConcreteRef) {
            $gitFetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir fetch --depth 1 origin $ResolvedSourceRef }
            if ($gitFetchExit -ne 0) {
                substep "git fetch failed -- using existing source" "Yellow"
            } else {
                $gitCheckoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir checkout -B unsloth-llama-build FETCH_HEAD }
                if ($gitCheckoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout"
                } else {
                    Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir clean -fdx } | Out-Null
                }
            }
        } else {
            $gitFetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir fetch --depth 1 origin }
            if ($gitFetchExit -ne 0) {
                substep "git fetch failed -- using existing source" "Yellow"
            } else {
                $gitCheckoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir checkout -B unsloth-llama-build FETCH_HEAD }
                if ($gitCheckoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout"
                } else {
                    Invoke-SetupCommand -AlwaysQuiet { git -C $LlamaCppDir clean -fdx } | Out-Null
                }
            }
        }
    } else {
        Write-Host "   Cloning llama.cpp @ $ResolvedSourceRef..." -ForegroundColor Gray
        $buildTmp = "$LlamaCppDir.build.$PID"
        $null = New-Item -ItemType Directory -Force -Path (Split-Path $LlamaCppDir -Parent)
        if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
        if ($LlamaPr) {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$LlamaSource.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin "pull/$LlamaPr/head:pr-$LlamaPr" }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch PR #$LlamaPr"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout "pr-$LlamaPr" }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout PR #$LlamaPr"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "pull") {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$ResolvedSourceUrl.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin $ResolvedSourceRef }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch source PR ref"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout -B unsloth-llama-build FETCH_HEAD }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout source PR ref"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "commit") {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$ResolvedSourceUrl.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin $ResolvedSourceRef }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch source commit"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout -B unsloth-llama-build FETCH_HEAD }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout source commit"
                    if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
                }
            }
        } else {
            $cloneArgs = @("clone", "--depth", "1")
            if ($UseConcreteRef) {
                $cloneArgs += @("--branch", $ResolvedSourceRef)
            }
            $cloneArgs += @("$ResolvedSourceUrl.git", $buildTmp)
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git @cloneArgs }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path $buildTmp) { Remove-Item -Recurse -Force $buildTmp }
            }
        }
        # Use temp dir for build; swap into $LlamaCppDir only after build succeeds
        if ($BuildOk) {
            $LlamaCppDir = $buildTmp
            $BuildDir = Join-Path $LlamaCppDir "build"
        }
    }

    # -- Step B: cmake configure --

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
                        substep "GPU is sm_$CudaArch but nvcc only supports up to sm_$maxArch" "Yellow"
                        substep "Building with sm_$maxArch (PTX will JIT for your GPU at runtime)" "Yellow"
                    }
                    # else: omit flag entirely, let cmake pick defaults
                }
            }
        } else {
            $CmakeArgs += '-DGGML_CUDA=OFF'
        }

        $cmakeOutput = cmake @CmakeArgs 2>&1 | Out-String
        $cmakeConfigureExit = $LASTEXITCODE
        if ($cmakeConfigureExit -ne 0) {
            $BuildOk = $false
            $FailedStep = "cmake configure"
            Write-LlamaFailureLog -Output $cmakeOutput
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
        $cmakeBuildServerExit = $LASTEXITCODE
        if ($cmakeBuildServerExit -ne 0) {
            $BuildOk = $false
            $FailedStep = "cmake build (llama-server)"
            Write-LlamaFailureLog -Output $output
        }
    }

    # -- Step D: Build llama-quantize (optional, best-effort) --
    if ($BuildOk) {
        Write-Host ""
        Write-Host "--- cmake build (llama-quantize) ---" -ForegroundColor Cyan
        $output = cmake --build $BuildDir --config Release --target llama-quantize -j $NumCpu 2>&1 | Out-String
        $cmakeBuildQuantizeExit = $LASTEXITCODE
        if ($cmakeBuildQuantizeExit -ne 0) {
            substep "llama-quantize build failed (GGUF export may be unavailable)" "Yellow"
            Write-LlamaFailureLog -Output $output
        }
    }

    # Swap temp build dir into final location (only if we built in a temp dir)
    if ($BuildOk -and $LlamaCppDir -ne $OriginalLlamaCppDir) {
        if (Test-Path $OriginalLlamaCppDir) { Remove-Item -Recurse -Force $OriginalLlamaCppDir }
        Move-Item $LlamaCppDir $OriginalLlamaCppDir
        $LlamaCppDir = $OriginalLlamaCppDir
        $BuildDir = Join-Path $LlamaCppDir "build"
        $LlamaServerBin = Join-Path $BuildDir "bin\Release\llama-server.exe"
    } elseif (-not $BuildOk -and $LlamaCppDir -ne $OriginalLlamaCppDir) {
        # Build failed -- clean up temp dir, preserve existing install
        if (Test-Path $LlamaCppDir) { Remove-Item -Recurse -Force $LlamaCppDir }
        $LlamaCppDir = $OriginalLlamaCppDir
        $BuildDir = Join-Path $LlamaCppDir "build"
        $LlamaServerBin = Join-Path $BuildDir "bin\Release\llama-server.exe"
    }

    # Restore ErrorActionPreference
    $ErrorActionPreference = $prevEAP

    # Stop timer
    $totalSw.Stop()
    $totalMin = [math]::Floor($totalSw.Elapsed.TotalMinutes)
    $totalSec = [math]::Round($totalSw.Elapsed.TotalSeconds % 60, 1)

    # -- Summary --
    if ($BuildOk -and (Test-Path $LlamaServerBin)) {
        step "llama.cpp" "built"
        $QuantizeBin = Join-Path $BuildDir "bin\Release\llama-quantize.exe"
        if (Test-Path $QuantizeBin) {
            step "llama-quantize" "built"
        }
        step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
    } else {
        $altBin = Join-Path $BuildDir "bin\llama-server.exe"
        if ($BuildOk -and (Test-Path $altBin)) {
            step "llama.cpp" "built"
            step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
        } else {
            step "llama.cpp" "build failed at: $FailedStep (${totalMin}m ${totalSec}s); continuing" "Yellow"
            substep "To retry: delete $LlamaCppDir and re-run setup." "Yellow"
            $script:LlamaCppDegraded = $true
        }
    }
}

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
$DoneLabel = if ($env:SKIP_STUDIO_BASE -eq "1") { "Unsloth Studio Setup Complete" } else { "Unsloth Studio Updated" }
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
    if ($script:LlamaCppDegraded) {
        Write-Host ("  " + (Get-StudioAnsi Warn) + "$DoneLabel (limited: llama.cpp unavailable)" + (Get-StudioAnsi Reset))
    } else {
        Write-Host ("  " + (Get-StudioAnsi Title) + $DoneLabel + (Get-StudioAnsi Reset))
    }
    Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    Write-Host "  $Rule" -ForegroundColor DarkGray
    if ($script:LlamaCppDegraded) {
        Write-Host "  $DoneLabel (limited: llama.cpp unavailable)" -ForegroundColor Yellow
    } else {
        Write-Host "  $DoneLabel" -ForegroundColor Green
    }
    Write-Host "  $Rule" -ForegroundColor DarkGray
}
step "launch" "unsloth studio -H 0.0.0.0 -p 8888"
Write-Host ""

# Match studio/setup.sh: exit non-zero for degraded llama.cpp when called
# from install.ps1 (SKIP_STUDIO_BASE=1) so the installer can detect the
# failure. Direct 'unsloth studio update' does not set SKIP_STUDIO_BASE,
# so it keeps degraded installs successful.
if ($script:LlamaCppDegraded -and $env:SKIP_STUDIO_BASE -eq "1") {
    exit 1
}
