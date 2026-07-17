#Requires -Version 5.1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
<#
.SYNOPSIS
    Full environment setup for Unsloth Studio on Windows (bundled version).
.DESCRIPTION
    Uses an isolated, Unsloth-managed Node.js for the frontend build when the
    system Node/npm do not meet requirements (never modifies the system Node).
    When running from pip install: skips frontend build (already bundled). When
    running from git repo: full setup including frontend build.
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

# Corporate-mirror / proxy escape hatch for the frontend npm/bun install (#6491).
# studio/frontend/.npmrc pins registry=https://registry.npmjs.org/ as a supply-chain
# lock, which overrides a corporate user's ~/.npmrc proxy and causes 403s behind a
# firewall. UNSLOTH_NPM_REGISTRY is a deliberate opt-in: when set we splat it as
# `--registry <url>` into every npm/bun install. `--registry` is the highest-precedence
# override for BOTH tools and leaves min-release-age / save-exact in force. Empty array
# (the default) splats to nothing, so normal installs are unchanged.
$NpmRegistryArgs = @()
if ($env:UNSLOTH_NPM_REGISTRY) {
    $NpmRegistryArgs = @('--registry', $env:UNSLOTH_NPM_REGISTRY)
}

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
# CUDA toolkit state, published by Resolve-CudaToolkit. Only the Phase 4 source
# build consumes these; the prebuilt path leaves them at these defaults.
$script:CudaToolkitReady = $false
$script:NvccPath = $null
$script:CudaToolkitRoot = $null
$script:CudaArch = $null

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

    $toolkitBase = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'

    if ($MaxVersion -and (Test-Path $toolkitBase)) {
        $drMajor = [int]$MaxVersion.Split('.')[0]

        # Get all installed CUDA dirs, sorted descending (highest first)
        $cudaDirs = Get-ChildItem -Directory $toolkitBase | Where-Object {
            $_.Name -match '^v(\d+)\.(\d+)'
        } | Sort-Object { [version]($_.Name -replace '^v','') } -Descending

        foreach ($dir in $cudaDirs) {
            if ($dir.Name -match '^v(\d+)\.(\d+)') {
                $tkMajor = [int]$Matches[1]
                $compatible = ($tkMajor -le $drMajor)
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

function Write-CudaDriverToolkitMismatch {
    param(
        [Parameter(Mandatory = $true)][string]$ToolkitVersion,
        [Parameter(Mandatory = $true)][string]$DriverMaxCuda,
        [string]$Color = "Yellow"
    )
    $toolkitMajor = $ToolkitVersion.Split('.')[0]
    $driverMajor = $DriverMaxCuda.Split('.')[0]
    substep "CUDA Toolkit $ToolkitVersion is a major-version mismatch: toolkit major $toolkitMajor exceeds driver CUDA major $driverMajor ($DriverMaxCuda)." $Color
    substep "Update the NVIDIA GPU driver to run CUDA Toolkit $ToolkitVersion, or install a CUDA $driverMajor.x toolkit." $Color
    substep "Or let Studio use the prebuilt CUDA bundle; it does not need the local toolkit." $Color
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
        # Bounded: a wedged nvidia-smi must not hang setup after the initial
        # -L probe succeeded (the helper merges stderr after stdout, so the
        # first line is still the compute_cap value).
        $raw = Invoke-NvidiaSmiBounded $smiExe @('--query-gpu=compute_cap', '--format=csv,noheader')
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
        # Bounded: a wedged nvidia-smi must not hang setup. The helper merges
        # stderr into the returned string, matching the old 2>&1 | Out-String
        # shape (plain 2>$null leaks ErrorRecord objects in PS 5.1).
        $output = Invoke-NvidiaSmiBounded $smiExe
        # Newer NVIDIA drivers (e.g. 610.x on Windows) print
        # "CUDA UMD Version: X.Y" instead of the legacy "CUDA Version: X.Y".
        # Accept both spellings so we don't fall through to the cu126 default.
        if ($output -match 'CUDA(?: UMD)? Version:\s+(\d+)\.(\d+)') {
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

# Trim trailing slashes from the URL PATH only, preserving ?query / #fragment. A whole-URL
# TrimEnd('/') corrupts a token that ends in "/" (base64 ...abc/); a single strip leaves
# .../cu128// as an empty leaf. Mirrors _trim_index_path_slashes (py) / _trim_index_path_slashes
# (install.sh) / Trim-IndexPathSlashes (install.ps1).
function Trim-IndexPathSlashes {
    param([string]$Url)
    $value = $Url.Trim()
    $idx = $value.IndexOfAny([char[]]@('?', '#'))
    if ($idx -lt 0) {
        return $value.TrimEnd('/')
    }
    return $value.Substring(0, $idx).TrimEnd('/') + $value.Substring($idx)
}

# Explicit torch-index pin (UNSLOTH_TORCH_INDEX_URL / _FAMILY), shared by the stale-venv
# check and install selection so a pinned index wins over GPU probing (matches install.sh
# / install.ps1 / install_python_stack.py). URL is verbatim; _FAMILY is the leaf joined
# to the mirror base so UNSLOTH_PYTORCH_MIRROR is honoured.
function Get-PinnedTorchIndexUrl {
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
        return (Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY)) {
        $base = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        return "$base/$($env:UNSLOTH_TORCH_INDEX_FAMILY.Trim().Trim('/'))"
    }
    return $null
}

# Last path segment of a wheel index URL, query/fragment dropped first so a
# token-authenticated pin (.../cu128?token=x) classifies as cu128 (else a raw leaf never
# equals the installed tag and reinstalls every update). Classification only. Mirrors
# _torch_index_leaf (py) / _torch_index_url_leaf (install.sh).
function Get-TorchIndexLeaf {
    param([string]$Url)
    if ([string]::IsNullOrWhiteSpace($Url)) { return $null }
    $path = ($Url -split '[?#]', 2)[0]
    if ([string]::IsNullOrWhiteSpace($path)) { return $null }
    return ($path.TrimEnd('/') -split '/')[-1].ToLowerInvariant()
}

# Redact index-URL credentials (userinfo + ?query= values) from captured installer output
# before printing on failure. uv/pip failure text embeds the failing --index-url verbatim,
# which can carry a user:token@ or ?token= secret. Mirrors _redact_install_output (py) /
# _redact_install_output (install.sh) / Redact-InstallOutput (install.ps1). Verbose mode
# streams live output uncaptured, so it is intentionally not redacted there (developer opt-in).
function Redact-InstallOutput {
    param([string]$Text)
    if (-not $Text) { return $Text }
    $Text = $Text -replace '(https?://)[^/@\s`]+@', '$1<redacted>@'
    return $Text -replace '([?&][^=\s&`]+)=[^&#\s`]+', '$1=<redacted>'
}

# AMD per-arch leaves needing the torch 2.11 floor (the _grouped_mm <2.11 bug). MUST
# match the install-spec path below and install.ps1 / install_python_stack.py; other
# leaves publish <2.11 wheels and stay on default specs.
function Test-RocmGfx211Leaf {
    param([string]$Leaf)
    return @('gfx120x-all', 'gfx1151', 'gfx1150') -contains $Leaf
}

# rocmX.Y versions KNOWN to ship torch 2.11: rocm7.2 is the only stable one today.
# Do NOT floor an unknown newer rocm (rocm7.3, ...) speculatively. MUST match
# _ROCM_KNOWN_TORCH211_VERSIONS (Python) and the rocm7.2 leaf in install.sh / install.ps1.
function Test-RocmKnown211Version {
    param([int]$Major, [int]$Minor)
    return ($Major -eq 7 -and $Minor -eq 2)
}

# True only for a real CUDA family leaf: "cu" + digits (cu118, cu128, ...). Mirrors
# _is_cuda_family_leaf. A bare -like 'cu*' would match "custom"/"current" and rebuild
# the venv every run.
function Test-CudaFamilyLeaf {
    param([string]$Leaf)
    if ([string]::IsNullOrWhiteSpace($Leaf)) { return $false }
    # EXACT cu+digits: cu128-private is NOT a family leaf, so it routes through the
    # unknown-leaf path.
    return $Leaf -match '^cu[0-9]+$'
}

# True only for a real pip ROCm family leaf: EXACT rocm<digits>[.<digits>] or a gfx leaf.
# A leaf that merely STARTS with rocm (rocm-rel-7.2.1, rocm7.2-private) is a custom pin
# the verbatim path owns, so anchor the match. Mirrors _is_pip_rocm_family_leaf / install.sh.
function Test-PipRocmFamilyLeaf {
    param([string]$Leaf)
    if ([string]::IsNullOrWhiteSpace($Leaf)) { return $false }
    return ($Leaf -like 'gfx*') -or ($Leaf -match '^rocm[0-9]+(\.[0-9]+)?$')
}

# Stale-venv ROCm comparison for a pinned gfx*/rocm* index. Returns @{ Expected; Installed }
# so the caller rebuilds when they differ. Mirrors _rocm_pin_family_mismatch (same rocmX.Y
# / gfx cases). An untagged (no +rocm) wheel never satisfies a ROCm pin -> stale.
function Get-RocmPinStaleTags {
    param([string]$PinLeaf, [string]$TorchVersion)
    $_pinRocm = [regex]::Match($PinLeaf, '^rocm(\d+)\.(\d+)')
    $_pinVer = if ($_pinRocm.Success) { "$($_pinRocm.Groups[1].Value).$($_pinRocm.Groups[2].Value)" } else { $null }
    # Installed rocm version and whether the wheel is a per-arch (three-part) build.
    $_instRocm = [regex]::Match($TorchVersion, '\+rocm(\d+)\.(\d+)')
    $_instVer = if ($_instRocm.Success) { "$($_instRocm.Groups[1].Value).$($_instRocm.Groups[2].Value)" } else { $null }
    $_instPerArch = [regex]::IsMatch($TorchVersion, '\+rocm\d+\.\d+\.\d+')
    # A ROCm build MUST carry a +rocm tag; an untagged wheel can't satisfy any ROCm pin.
    $_instHasRocm = [regex]::IsMatch($TorchVersion, '\+rocm')
    $_instRel = [regex]::Match($TorchVersion, '^(\d+)\.(\d+)')
    $_instIs211 = $false
    if ($_instRel.Success) {
        $_instIs211 = ([int]$_instRel.Groups[1].Value -gt 2) -or ([int]$_instRel.Groups[1].Value -eq 2 -and [int]$_instRel.Groups[2].Value -ge 11)
    }

    if ($PinLeaf -like 'gfx*') {
        if (Test-RocmGfx211Leaf $PinLeaf) {
            # Expect the AMD per-arch (three-part) 2.11 wheel: satisfied only when BOTH
            # a 2.11 release AND a three-part rocm tag are installed.
            $installed = if ($_instIs211 -and $_instPerArch) { "rocm-perarch(torch>=2.11)" } else { "rocm-generic-or-old" }
            return @{ Expected = "rocm-perarch(torch>=2.11)"; Installed = $installed }
        }
        # Non-2.11 gfx leaf (<2.11 spec): stale on an untagged wheel or a 2.11+ build.
        $installed = if (-not $_instHasRocm) { "not-rocm" } elseif ($_instIs211) { "rocm(torch>=2.11)" } else { "rocm(torch<2.11)" }
        return @{
            Expected  = "rocm(torch<2.11)"
            Installed = $installed
        }
    }

    # rocmX.Y pin.
    if ($_pinVer -and $_instVer) {
        # Both readable: exact compare. When they match AND the pin is KNOWN-2.11, the
        # installed release must also be 2.11 (a +rocm7.2 wheel drifted to 2.12 shares the
        # tag but violates the spec), so fold the release into the tag. Mirrors _rocm_pin_family_mismatch.
        $_pinKnown211 = Test-RocmKnown211Version -Major ([int]$_pinRocm.Groups[1].Value) -Minor ([int]$_pinRocm.Groups[2].Value)
        $_instOn211 = $_instRel.Success -and [int]$_instRel.Groups[1].Value -eq 2 -and [int]$_instRel.Groups[2].Value -eq 11
        if ($_pinKnown211 -and -not $_instOn211) {
            return @{ Expected = "rocm$_pinVer(torch2.11)"; Installed = "rocm$_instVer(torch-off-2.11)" }
        }
        return @{ Expected = "rocm$_pinVer"; Installed = "rocm$_instVer" }
    }
    $_pinNeeds211 = $false
    if ($_pinRocm.Success) {
        # Only KNOWN-2.11 rocm (rocm7.2) is on the 2.11 line (no speculative floor).
        # Matches _ROCM_KNOWN_TORCH211_VERSIONS.
        $_pinNeeds211 = Test-RocmKnown211Version -Major ([int]$_pinRocm.Groups[1].Value) -Minor ([int]$_pinRocm.Groups[2].Value)
    }
    # Fallback (installed rocm version unreadable): compare on the 2.11 line; an untagged
    # wheel never satisfies a rocmX.Y pin -> stale.
    $installed = if (-not $_instHasRocm) { "not-rocm" } elseif ($_instIs211) { "rocm(torch>=2.11)" } else { "rocm(torch<2.11)" }
    return @{
        Expected  = if ($_pinNeeds211) { "rocm(torch>=2.11)" } else { "rocm(torch<2.11)" }
        Installed = $installed
    }
}

# VS generator -> MSBuild BuildCustomizations dir; toolset tracks the VS major
# (18->v180, 17->v170), defaulting to v170 when unparseable.
function Get-VcBuildCustomizationsDir {
    param(
        [Parameter(Mandatory)][string]$VsInstallPath,
        [string]$Generator
    )
    $toolset = 'v170'
    if ($Generator -and ($Generator -match 'Visual Studio (\d+)\b')) {
        $toolset = "v$($Matches[1])0"
    }
    return (Join-Path $VsInstallPath "MSBuild\Microsoft\VC\$toolset\BuildCustomizations")
}

# Installed cmake version, or $null if absent/unparseable.
function Get-CmakeVersion {
    $raw = & cmake --version 2>$null | Select-Object -First 1
    if ($raw -and ($raw -match '(\d+)\.(\d+)(?:\.(\d+))?')) {
        $patch = if ($Matches[3]) { $Matches[3] } else { '0' }
        return [version]"$($Matches[1]).$($Matches[2]).$patch"
    }
    return $null
}

# VS 18 2026 generator needs cmake >= 4.2 (added there); true for older VS generators.
function Test-CmakeSupportsGenerator {
    param(
        [Parameter(Mandatory)][string]$CmakeVersion,
        [Parameter(Mandatory)][string]$Generator
    )
    if ($Generator -match 'Visual Studio 18\b') {
        $clean = ($CmakeVersion -replace '[^0-9.].*$', '').TrimEnd('.')
        try { $v = [version]$clean } catch { return $false }
        return ($v -ge [version]'4.2')
    }
    return $true
}

function Test-CmakeListsGenerator {
    # Does `cmake --help` actually list the generator? A VS-bundled cmake can drive
    # VS 2026 below the 4.2 floor, so probe rather than trust the version. (#6473)
    param([Parameter(Mandatory)][string]$Generator)
    $help = & cmake --help 2>$null | Out-String
    if (-not $help) { return $false }
    $haystack = ($help -replace '\s+', ' ')
    $needle = ($Generator -replace '\s+', ' ')
    return $haystack.Contains($needle)
}

function Test-CmakeCanDriveGenerator {
    # cmake can drive $Generator if it lists it (VS-bundled below 4.2) or meets the floor.
    param([Parameter(Mandatory)][string]$Generator)
    if (Test-CmakeListsGenerator -Generator $Generator) { return $true }
    $verObj = Get-CmakeVersion
    $verStr = if ($verObj) { $verObj.ToString() } else { '0.0' }
    return (Test-CmakeSupportsGenerator -CmakeVersion $verStr -Generator $Generator)
}

function Add-DefaultCmakeToPath {
    # Prepend the default CMake dir so a freshly winget-installed cmake wins over an
    # older one already on PATH. $true if found. (#6473)
    $cmakeDefaults = @(
        "$env:ProgramFiles\CMake\bin",
        "${env:ProgramFiles(x86)}\CMake\bin",
        "$env:LOCALAPPDATA\CMake\bin"
    )
    foreach ($d in $cmakeDefaults) {
        if (Test-Path (Join-Path $d "cmake.exe")) {
            $env:Path = "$d;$env:Path"
            Add-ToUserPath -Directory $d -Position 'Prepend' | Out-Null
            return $true
        }
    }
    return $false
}

function Get-FallbackVsGenerator {
    # Newest pre-2026 VS whose generator the current cmake can drive, for when the
    # VS 2026 generator is unusable (old/offline cmake) but an older toolchain exists.
    # vswhere first (catches non-default roots like D:\), then Program Files; matches
    # Find-VsBuildTools. Returns @{ Generator; InstallPath } or $null. (#6473)
    $knownEditions = @('BuildTools', 'Community', 'Professional', 'Enterprise', 'Preview')

    # install path if it holds a usable cl.exe, else $null
    $tryCandidate = {
        param($gen, $installPath)
        if (-not $installPath) { return $null }
        $vcDir = Join-Path $installPath "VC\Tools\MSVC"
        if (-not (Test-Path $vcDir)) { return $null }
        $cl = Get-ChildItem -Path $vcDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cl) { return @{ Generator = $gen; InstallPath = $installPath } }
        return $null
    }

    # vswhere (non-default roots)
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $json = & $vsw -all -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json 2>$null | Out-String
        if ($json) {
            try { $instances = @($json | ConvertFrom-Json) } catch { $instances = @() }
            $ranked = $instances | ForEach-Object {
                $label = if ($_.catalog -and $_.catalog.productLineVersion) { [string]$_.catalog.productLineVersion } else { '' }
                [pscustomobject]@{ Gen = (Resolve-VsGeneratorFromLabel $label); Path = [string]$_.installationPath }
            } | Where-Object { $_.Gen -and ($_.Gen -notmatch 'Visual Studio 18\b') }
            # newest first: 2022 > 2019 > 2017
            $ranked = $ranked | Sort-Object { switch -regex ($_.Gen) { '17 2022' {0} '16 2019' {1} '15 2017' {2} default {9} } }
            foreach ($cand in $ranked) {
                if (-not (Test-CmakeListsGenerator -Generator $cand.Gen)) { continue }
                $res = & $tryCandidate $cand.Gen $cand.Path
                if ($res) { return $res }
            }
        }
    }

    # Program Files scan
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    $older = @(
        @{ Dir = '2022'; Generator = 'Visual Studio 17 2022' },
        @{ Dir = '2019'; Generator = 'Visual Studio 16 2019' },
        @{ Dir = '2017'; Generator = 'Visual Studio 15 2017' }
    )
    foreach ($entry in $older) {
        if (-not (Test-CmakeListsGenerator -Generator $entry.Generator)) { continue }
        foreach ($r in $roots) {
            $vsBase = Join-Path $r "Microsoft Visual Studio\$($entry.Dir)"
            if (-not (Test-Path $vsBase)) { continue }
            foreach ($ed in $knownEditions) {
                $candidate = Join-Path $vsBase $ed
                if (-not (Test-Path $candidate)) { continue }
                $res = & $tryCandidate $entry.Generator $candidate
                if ($res) { return $res }
            }
        }
    }
    return $null
}

# VS version label -> cmake generator. vswhere's productLineVersion is the year for
# VS <= 2022 but the internal major "18" for VS 2026, and dir names use either form,
# so accept both. (VS 2026 detection adapted from @LeoBorcherding's #6038.)
function Resolve-VsGeneratorFromLabel {
    param([string]$Label)
    if (-not $Label) { return $null }
    $map = @{
        '2026' = 'Visual Studio 18 2026'; '18' = 'Visual Studio 18 2026'
        '2022' = 'Visual Studio 17 2022'; '17' = 'Visual Studio 17 2022'
        '2019' = 'Visual Studio 16 2019'; '16' = 'Visual Studio 16 2019'
        '2017' = 'Visual Studio 15 2017'; '15' = 'Visual Studio 15 2017'
    }
    return $map[$Label.Trim()]
}

# Find VS Build Tools for cmake -G: vswhere, then a filesystem scan (handles broken
# vswhere registration). Returns @{ Generator; InstallPath; Source } or $null.
function Find-VsBuildTools {
    # vswhere first (works when VS is properly registered)
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $info = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion 2>$null
        $path = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($info -and $path) {
            $gen = Resolve-VsGeneratorFromLabel $info
            if ($gen) {
                return @{ Generator = $gen; InstallPath = $path.Trim(); Source = 'vswhere' }
            }
        }
    }

    # filesystem scan (handles broken vswhere registration); VS 2026+ dir is "18"
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    $knownEditions = @('BuildTools', 'Community', 'Professional', 'Enterprise', 'Preview')
    $dirs = @('18', '2026', '2022', '2019', '2017')

    foreach ($d in $dirs) {
        $gen = Resolve-VsGeneratorFromLabel $d
        if (-not $gen) { continue }
        foreach ($r in $roots) {
            $vsBase = Join-Path $r "Microsoft Visual Studio\$d"
            if (-not (Test-Path $vsBase)) { continue }
            # VS 2026 (dir "18") may use non-standard edition names, so scan every subdir
            if ($d -eq '18' -or $d -eq '2026') {
                $editionCandidates = Get-ChildItem -Path $vsBase -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
            } else {
                $editionCandidates = $knownEditions | ForEach-Object { Join-Path $vsBase $_ }
            }
            foreach ($candidate in $editionCandidates) {
                if (-not (Test-Path $candidate)) { continue }
                $vcDir = Join-Path $candidate "VC\Tools\MSVC"
                if (Test-Path $vcDir) {
                    $cl = Get-ChildItem -Path $vcDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
                    if ($cl) {
                        $ed = Split-Path $candidate -Leaf
                        return @{ Generator = $gen; InstallPath = $candidate; Source = "filesystem ($ed)"; ClExe = $cl.FullName }
                    }
                }
            }
        }
    }

    return $null
}

# Install CMake + VS Build Tools, deferred here from Phase 1 so the prebuilt path
# never pays for a multi-GB install. Called only when a source build is committed.
# CMake is best-effort (build skips downstream if absent); VS Build Tools are
# required, so exit 1 with guidance if missing. No-ops for VS when already detected.
function Ensure-BuildToolsForLlamaSourceBuild {
    # CMake
    if ($null -eq (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Host "CMake not found -- installing via winget (needed for the llama.cpp source build)..." -ForegroundColor Yellow
        if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
            try {
                Invoke-SetupCommand { winget install Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                Refresh-Environment
            } catch { }
        }
        # winget may install cmake but not put it on PATH yet; try the default dir
        if ($null -eq (Get-Command cmake -ErrorAction SilentlyContinue)) {
            $cmakeDefaults = @(
                "$env:ProgramFiles\CMake\bin",
                "${env:ProgramFiles(x86)}\CMake\bin",
                "$env:LOCALAPPDATA\CMake\bin"
            )
            foreach ($d in $cmakeDefaults) {
                if (Test-Path (Join-Path $d "cmake.exe")) {
                    $env:Path = "$d;$env:Path"
                    Add-ToUserPath -Directory $d -Position 'Prepend' | Out-Null
                    break
                }
            }
        }
        if ($null -ne (Get-Command cmake -ErrorAction SilentlyContinue)) { step "cmake" "installed" }
    }

    # VS Build Tools
    if ($script:VsInstallPath) { return }   # already detected by the early probe
    $vsResult = Find-VsBuildTools
    if (-not $vsResult) {
        Write-Host "Visual Studio Build Tools not found -- installing via winget..." -ForegroundColor Yellow
        Write-Host "   (Needed only for the llama.cpp source build; may take several minutes)" -ForegroundColor Gray
        if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
            $prevEAPTemp = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            winget install Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
            $ErrorActionPreference = $prevEAPTemp
            # Re-scan after install (don't trust vswhere catalog)
            $vsResult = Find-VsBuildTools
        }
    }
    if ($vsResult) {
        $script:CmakeGenerator = $vsResult.Generator
        $script:VsInstallPath = $vsResult.InstallPath
        step "vs" "$($vsResult.Generator) ($($vsResult.Source))"
        if ($vsResult.ClExe) { substep "cl.exe: $($vsResult.ClExe)" }
    } else {
        Write-Host "[ERROR] Visual Studio Build Tools are required for the llama.cpp source build but could not be found or installed." -ForegroundColor Red
        Write-Host "        Manual install:" -ForegroundColor Red
        Write-Host '        1. winget install Microsoft.VisualStudio.2022.BuildTools --source winget' -ForegroundColor Yellow
        Write-Host '        2. Open Visual Studio Installer -> Modify -> check "Desktop development with C++"' -ForegroundColor Yellow
        exit 1
    }
}

# Detect the VC++ 2015-2022 Redistributable that the prebuilt llama-server and
# PyTorch need (they link VCRUNTIME140_1.dll etc., which the Universal CRT lacks).
# Signal is System32\vcruntime140_1.dll (VS 2019+), registry as fallback.
function Test-VCRedistInstalled {
    $sys = $env:SystemRoot
    if ($sys -and (Test-Path (Join-Path $sys 'System32\vcruntime140_1.dll'))) { return $true }
    foreach ($k in @(
        'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    )) {
        try {
            $r = Get-ItemProperty -Path $k -ErrorAction Stop
            if ($r.Installed -eq 1 -and [int]$r.Major -ge 14 -and [int]$r.Minor -ge 20) { return $true }
        } catch { }
    }
    return $false
}

# Install the VC++ 2015-2022 runtime if missing (non-fatal; usually a no-op).
function Ensure-VCRedist {
    if (Test-VCRedistInstalled) { step "vcredist" "present"; return }
    Write-Host "Microsoft Visual C++ Redistributable (2015-2022) is missing; the prebuilt llama.cpp and PyTorch need it. Installing the runtime..." -ForegroundColor Yellow
    if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
        try {
            Invoke-SetupCommand { winget install --id Microsoft.VCRedist.2015+.x64 --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
            Refresh-Environment
        } catch { substep "VCRedist install failed: $($_.Exception.Message)" "Yellow" }
    }
    if (Test-VCRedistInstalled) { step "vcredist" "installed" }
    else {
        substep "Could not install the VC++ Redistributable automatically." "Yellow"
        substep "If llama-server or torch reports a missing VCRUNTIME140.dll, install:" "Yellow"
        substep "https://aka.ms/vs/17/release/vc_redist.x64.exe" "Yellow"
    }
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
                Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
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
# Mirror the plain (no ANSI) form of step/substep messages to the
# OS-level stdout handle when a parent is consuming our stdout via
# a pipe (CI `tee`, Python subprocess.PIPE, CREATE_NO_WINDOW grandchild).
# Write-Host on PS 5.1 routes through $Host.UI / the Information
# stream, neither of which propagates reliably across the
# install.ps1 -> unsloth.exe -> python -> powershell.exe ->
# setup.ps1 process chain. [Console]::Out always lands on the OS
# stdout file handle. Gated on IsOutputRedirected so the
# interactive-console path keeps the colorized Write-Host output
# only (no double-print).
function Write-StudioStdoutMirror {
    param([Parameter(Mandatory = $true)][string]$Line)
    try {
        if ([Console]::IsOutputRedirected) {
            [Console]::Out.WriteLine($Line)
            [Console]::Out.Flush()
        }
    } catch {}
}

function step {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Value,
        [string]$Color = "Green"
    )
    $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
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
        Write-Host ("  {0}{1}{2}{3}{4}{2}" -f $dim, $padded, $rst, $val, $Value)
    } else {
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
    Write-StudioStdoutMirror ("  {0}{1}" -f $padded, $Value)
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
    Write-StudioStdoutMirror ("  {0,-15}{1}" -f "", $Message)
}

function Show-NpmRegistryHint {
    # Print actionable guidance when a frontend/OXC npm/bun install fails and the
    # registry lock is the likely cause (corporate firewall/proxy). No-op once the
    # user has opted in via UNSLOTH_NPM_REGISTRY. We never switch registries
    # automatically -- we only guide.
    if ($env:UNSLOTH_NPM_REGISTRY) { return }
    $mirror = $env:NPM_CONFIG_REGISTRY
    if (-not $mirror) {
        # Read npm config from a dir with no project .npmrc so the frontend's pinned
        # registry= does not mask the user's ~/.npmrc / global mirror.
        $pushed = $false
        try {
            Push-Location ([System.IO.Path]::GetTempPath()) -ErrorAction Stop
            $pushed = $true
            $mirror = (& npm config get registry 2>$null | Out-String).Trim()
        } catch { $mirror = "" } finally { if ($pushed) { Pop-Location } }
    }
    if ($mirror -in @("", "undefined", "null", "https://registry.npmjs.org", "https://registry.npmjs.org/")) {
        $mirror = ""
    }
    Write-Host ""
    step "frontend" "registry.npmjs.org looks blocked (corporate firewall/proxy?)" "Yellow"
    if ($mirror) {
        substep "Studio pins the public npm registry; your mirror is being ignored."
        substep "Detected a registry in your npm config:"
        substep "  $mirror"
        substep "Re-run pointing Studio at it:"
        substep "  `$env:UNSLOTH_NPM_REGISTRY='$mirror'; .\install.ps1 --local"
    } else {
        substep "If you use a private mirror/proxy, point Studio at it and re-run:"
        substep "  `$env:UNSLOTH_NPM_REGISTRY='https://your-mirror.example/api/npm/'; .\install.ps1 --local"
    }
    substep "(min-release-age and save-exact stay enforced.)"
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
# ── Helper: run nvidia-smi under a timeout ──
# A wedged NVIDIA driver can make nvidia-smi block during init or after a reset;
# WaitForExit bounds it (mirrors Invoke-AmdSmiNoElevate below) so detection
# cannot hang setup. No RunAsInvoker compat layer: nvidia-smi does not
# auto-elevate. Returns combined stdout+stderr; "" on timeout/failure.
function Invoke-NvidiaSmiBounded {
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
        [Parameter(Position = 1)][string[]]$SmiArgs = @(),
        [int]$TimeoutSec = 10
    )
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = ($SmiArgs -join ' ')
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $errTask = $proc.StandardError.ReadToEndAsync()
        if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
            try { $proc.Kill() } catch {}
            $global:LASTEXITCODE = 124
            return ""
        }
        $global:LASTEXITCODE = $proc.ExitCode
        return ($outTask.Result + "`n" + $errTask.Result)
    } catch {
        $global:LASTEXITCODE = 1
        return ""
    }
}

# ── Helper: nvidia-smi -L lists at least one real GPU ──
# Exit code 0 alone is not enough: a stale/driverless nvidia-smi can exit 0
# while listing no GPU, which would mark an AMD host NVIDIA and suppress ROCm
# detection. Require a "GPU <n>:" data row.
function Test-NvidiaSmiHasGpu {
    param([Parameter(Mandatory = $true)][string]$Exe)
    $out = Invoke-NvidiaSmiBounded $Exe @('-L')
    return ($LASTEXITCODE -eq 0 -and $out -match '(?m)^GPU\s+\d+:')
}

$HasNvidiaSmi = $false
$NvidiaSmiExe = $null  # Absolute path -- survives Refresh-Environment
try {
    $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvSmiCmd -and (Test-NvidiaSmiHasGpu $nvSmiCmd.Source)) {
        $HasNvidiaSmi = $true
        $NvidiaSmiExe = $nvSmiCmd.Source
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
                if (Test-NvidiaSmiHasGpu $p) {
                    $HasNvidiaSmi = $true
                    $NvidiaSmiExe = $p
                    Write-Host "   Found nvidia-smi at $(Split-Path $p -Parent)" -ForegroundColor Gray
                    break
                }
            } catch {}
        }
    }
}
# ── Helper: run amd-smi without triggering a UAC elevation prompt ──
# amd-smi on Windows auto-elevates to read GPU/APU memory, surfacing a confusing
# DiskPart UAC prompt mid-install (Studio backend amd.py hits the same). RunAsInvoker
# forces it (and helpers it spawns) to run un-elevated; on failure the WMI name ->
# gfx fallback still resolves the arch.
function Invoke-AmdSmiNoElevate {
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
        [Parameter(Position = 1)][string[]]$SmiArgs = @(),
        [int]$TimeoutSec = 30
    )
    # RunAsInvoker blocks the auto-elevation/UAC prompt; the timeout bounds a flaky
    # amd-smi that can otherwise spin for minutes (30s mirrors the backend amd.py).
    $prevCompat = [Environment]::GetEnvironmentVariable('__COMPAT_LAYER', 'Process')
    $env:__COMPAT_LAYER = 'RunAsInvoker'
    try {
        # [Process]::Start, NOT Start-Process -PassThru: the latter leaves .ExitCode
        # $null after WaitForExit on PS 5.1, so $LASTEXITCODE (checked by callers)
        # reads non-zero and kills detection. Async reads drain the pipes (no
        # deadlock); amd-smi args have no spaces so a plain join is safe.
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = ($SmiArgs -join ' ')
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $errTask = $proc.StandardError.ReadToEndAsync()
        if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
            try { $proc.Kill() } catch {}
            $global:LASTEXITCODE = 124
            return ""
        }
        $global:LASTEXITCODE = $proc.ExitCode
        return ($outTask.Result + "`n" + $errTask.Result)
    } catch {
        $global:LASTEXITCODE = 1
        return ""
    } finally {
        if ($null -eq $prevCompat) {
            Remove-Item Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
        } else {
            $env:__COMPAT_LAYER = $prevCompat
        }
    }
}

# ── AMD ROCm detection (Windows): probe hipinfo/amd-smi for actual GPU ──
$HasROCm = $false
$HipSdkInstalled = $false   # HIP SDK binary found (independent of device accessibility)
$ROCmGpuLabel = $null
$script:ROCmGfxArch = $null
if (-not $HasNvidiaSmi) {
    # hipinfo: PATH first, then HIP_PATH/ROCM_PATH bin fallback (mirrors NVIDIA smi path resolution).
    # AMD HIP SDK sets HIP_PATH but may not add the bin dir to PATH depending on install type.
    # Ignore the venv hipInfo.exe (AMD wheel, on PATH): not a HIP SDK, so amd-smi
    # would still auto-elevate. Cf. _path_inside_venv().
    function Test-HipinfoIsVenvInternal {
        param([AllowNull()][string]$HipinfoPath)
        if ([string]::IsNullOrWhiteSpace($HipinfoPath)) { return $false }
        # VenvDir/VIRTUAL_ENV can be unset this early (the update flow probes before
        # VenvDir is set), so also derive the venv from the setup python + default
        # Studio home, else the venv hipInfo isn't caught.
        $venvRoots = @()
        if ($env:VIRTUAL_ENV) { $venvRoots += $env:VIRTUAL_ENV }
        $vd = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue
        if ($vd) { $venvRoots += $vd }
        if ($env:UNSLOTH_SETUP_PYTHON) {
            try { $venvRoots += (Split-Path -Parent (Split-Path -Parent $env:UNSLOTH_SETUP_PYTHON)) } catch {}
        }
        if ($env:USERPROFILE) { $venvRoots += (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio") }
        # A custom Studio home (UNSLOTH_STUDIO_HOME / STUDIO_HOME alias) moves the
        # venv off the default path; seed it too or its hipInfo escapes the filter.
        $studioHomeEnv = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() } else { $null }
        if ($studioHomeEnv) {
            # Expand a leading ~ like the canonical resolver below; else GetFullPath
            # keeps the literal ~ (cwd-relative) and the hipInfo escapes the filter.
            if (($studioHomeEnv -eq "~" -or $studioHomeEnv -like "~/*" -or $studioHomeEnv -like "~\*") -and -not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                # A bare "~" leaves an empty child path; Join-Path rejects that on
                # PS 5.1, so use USERPROFILE directly and only join a real remainder.
                $studioHomeRest = $studioHomeEnv.Substring(1).TrimStart('/', '\')
                $studioHomeEnv = if ($studioHomeRest) { Join-Path $env:USERPROFILE $studioHomeRest } else { $env:USERPROFILE }
            }
            $venvRoots += (Join-Path $studioHomeEnv "unsloth_studio")
        }
        try { $hip = [System.IO.Path]::GetFullPath($HipinfoPath).TrimEnd('\', '/') } catch { return $false }
        foreach ($root in $venvRoots) {
            if ([string]::IsNullOrWhiteSpace($root)) { continue }
            try { $r = [System.IO.Path]::GetFullPath($root).TrimEnd('\', '/') } catch { continue }
            # Skip a bare drive root (e.g. a non-venv UNSLOTH_SETUP_PYTHON like
            # C:\Python311\python.exe yields C:) -- it would match every path on that drive.
            if ($r -match '^[a-zA-Z]:$') { continue }
            if ($hip.Equals($r, [System.StringComparison]::OrdinalIgnoreCase) -or
                $hip.StartsWith($r + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
        return $false
    }
    # Scan all hipinfo and keep the first non-venv one (the venv copy from the
    # bnb fix could shadow a real HIP SDK's). -CommandType Application matches
    # only real executables, not a user alias/function named hipinfo.
    $hipinfoExe = Get-Command hipinfo -CommandType Application -All -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-HipinfoIsVenvInternal $_.Source) } |
        Select-Object -First 1
    if (-not $hipinfoExe) {
        # Iterate the env roots (mirrors the Python list) and take the first non-venv
        # bin\hipinfo.exe, so a venv-internal HIP_PATH can't mask a real SDK in ROCM_PATH.
        $hipMissingLabel = $null; $hipMissingRoot = $null; $hipMissingCandidate = $null
        foreach ($hipEnvLabel in @("HIP_PATH", "HIP_PATH_57", "ROCM_PATH")) {
            $hipRoot = [Environment]::GetEnvironmentVariable($hipEnvLabel)
            if ([string]::IsNullOrWhiteSpace($hipRoot)) { continue }
            $hipinfoCandidate = Join-Path $hipRoot "bin\hipinfo.exe"
            if (-not (Test-Path $hipinfoCandidate)) {
                if (-not $hipMissingLabel) { $hipMissingLabel = $hipEnvLabel; $hipMissingRoot = $hipRoot; $hipMissingCandidate = $hipinfoCandidate }
                continue
            }
            if (Test-HipinfoIsVenvInternal $hipinfoCandidate) { continue }   # venv copy (AMD wheel): not a HIP SDK
            substep "[WARN] hipinfo not on PATH -- located via ${hipEnvLabel}: $hipinfoCandidate" "Yellow"
            substep "       Add '$(Join-Path $hipRoot 'bin')' to your PATH to suppress this warning" "Yellow"
            substep "       Quick fix: [Environment]::SetEnvironmentVariable('PATH',`$env:PATH+';$(Join-Path $hipRoot 'bin')','User')" "Yellow"
            $hipinfoExe = [PSCustomObject]@{ Source = $hipinfoCandidate }
            break
        }
        if ((-not $hipinfoExe) -and $hipMissingLabel) {
            substep "[WARN] ${hipMissingLabel}=$hipMissingRoot is set but hipinfo.exe not found at $hipMissingCandidate" "Yellow"
            substep "       HIP SDK install may be incomplete -- re-install from:" "Yellow"
            substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
        }
    }
    if ($hipinfoExe) {
        $HipSdkInstalled = $true   # binary found → SDK is installed regardless of device state
        try {
            $hipOut = & $hipinfoExe.Source 2>&1 | Out-String
            if ($hipOut -match "(?i)gcnArchName") {
                # hipinfo can crash after printing gcnArchName (#6043).
                # Once the arch is printed, keep the ROCm wheel path.
                $HasROCm = $true
                $_hipAllArches = @([regex]::Matches($hipOut, "(?im)^\s*gcnArchName\s*:\s*(\S+)") | ForEach-Object { ($_.Groups[1].Value -split ':')[0].Trim().ToLower() })
                $_hipVisIdx = if ($env:HIP_VISIBLE_DEVICES -match '^\d') { [int]($env:HIP_VISIBLE_DEVICES -split ',')[0] } elseif ($env:ROCR_VISIBLE_DEVICES -match '^\d') { [int]($env:ROCR_VISIBLE_DEVICES -split ',')[0] } else { 0 }
                if ($_hipAllArches.Count -gt 0) {
                    $script:ROCmGfxArch = if ($_hipVisIdx -lt $_hipAllArches.Count) { $_hipAllArches[$_hipVisIdx] } else { $_hipAllArches[0] }
                    $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                } else {
                    $ROCmGpuLabel = "AMD ROCm"
                }
                if ($LASTEXITCODE -ne 0) {
                    substep "[INFO] hipinfo exited with code $LASTEXITCODE but reported gcnArchName -- treating as ROCm-capable (see #6043)" "Cyan"
                }
            } elseif ($LASTEXITCODE -ne 0) {
                # hipinfo ran but returned a HIP runtime error without any gcnArchName
                # output (e.g. "no ROCm-capable device detected"), or crashed before
                # printing device info.
                $firstLine = ($hipOut -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -First 1)
                substep "[WARN] hipinfo returned a HIP runtime error (exit $LASTEXITCODE)" "Yellow"
                substep "       $firstLine" "Yellow"
                substep "       Ensure ROCm drivers are installed: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
            }
        } catch {}
    }
    # amd-smi fallback: HIP runtime present but hipinfo unavailable (no full HIP SDK).
    # 'list' confirms GPU visibility, 'static --asic' extracts the gfx arch hipinfo
    # would give. Critical for Strix Halo (gfx1151) and other HIP-runtime-only iGPUs.
    #
    # BUT on hosts without a working HIP runtime amd-smi elevates a child at runtime,
    # popping a UAC/DiskPart prompt RunAsInvoker can't suppress (its manifest is
    # asInvoker; even 'amd-smi version' hangs). So only probe when a HIP SDK is present
    # (hipinfo found -> un-elevated) or the user opts in; else fall through to WMI name
    # inference (enough to pick ROCm wheels + the ROCm llama.cpp prebuilt).
    # An explicit opt-out (UNSLOTH_ENABLE_AMD_SMI=0/false/no/off) wins over the HIP-SDK
    # heuristic: a HIP SDK binary with a broken runtime can still pop the prompt, so
    # $HipSdkInstalled must NOT silently re-enable it.
    $amdSmiOptOut = $env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(0|false|no|off)$'
    $amdSmiAllowed = (-not $amdSmiOptOut) -and ($HipSdkInstalled -or ($env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(1|true|yes|on)$'))
    if (-not $HasROCm -and $amdSmiAllowed) {
        $amdSmiExe = Get-Command "amd-smi" -ErrorAction SilentlyContinue
        if ($amdSmiExe) {
            try {
                $smiOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('list')
                if ($LASTEXITCODE -eq 0 -and $smiOut -match "(?im)^GPU\s*[:\[]\s*\d") {
                    $HasROCm = $true
                    # Attempt 1: newer amd-smi versions embed the gfx arch in list output.
                    # Collect ALL gfx tokens in output order so that on mixed-arch systems
                    # we can honour HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES and pick the
                    # arch for the *runtime-visible* GPU rather than always the first one.
                    # Do NOT deduplicate: a dual same-arch system (e.g. two gfx1151 APUs)
                    # must produce a 2-element array so HIP_VISIBLE_DEVICES=1 selects the
                    # second GPU rather than triggering a false out-of-range warning.
                    # Note: this mapping assumes amd-smi lists GPUs in the same order as
                    # HIP enumerates them (both follow PCI bus order in practice); it may
                    # give the wrong arch when GPU indices are non-contiguous (very rare).
                    $allGfxArches = @([regex]::Matches($smiOut, '(?i)\b(gfx\d+[a-z]?)\b') |
                        ForEach-Object { $_.Groups[1].Value.ToLower() })
                    if ($allGfxArches.Count -gt 0) {
                        # Resolve which GPU index is runtime-visible.  When a single
                        # integer index is set, use it; fall back to index 0 otherwise
                        # (comma-separated lists or unset → first GPU, same as before).
                        $visGpu = if ($env:HIP_VISIBLE_DEVICES) { $env:HIP_VISIBLE_DEVICES }
                                  elseif ($env:ROCR_VISIBLE_DEVICES) { $env:ROCR_VISIBLE_DEVICES }
                                  else { $null }
                        $gpuIdx = 0
                        if ($visGpu -match '^\s*(\d+)\s*$') { $gpuIdx = [int]$Matches[1] }
                        if ($gpuIdx -ge $allGfxArches.Count) {
                            substep "[WARN] HIP/ROCR_VISIBLE_DEVICES index $gpuIdx is out of range ($($allGfxArches.Count) GPU(s) detected); defaulting to GPU 0 for arch selection" "Yellow"
                            $gpuIdx = 0
                        }
                        $script:ROCmGfxArch = $allGfxArches[$gpuIdx]
                        $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                    } else {
                        # Attempt 2: 'static --asic' exposes ASIC details on ROCm 6+,
                        # including the GFX target needed for wheel index selection.
                        $smiAsicOut = ""
                        try { $smiAsicOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('static','--asic') } catch {}
                        if ($smiAsicOut -match "(?i)\b(gfx\d+[a-z]?)\b") {
                            $script:ROCmGfxArch = $Matches[1].ToLower()
                            $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                        } elseif ($smiAsicOut -match "(?im)Market.?Name\s*[:\|]\s*([^\r\n]+)") {
                            $ROCmGpuLabel = "AMD ROCm ($($Matches[1].Trim()))"
                        } else {
                            $ROCmGpuLabel = "AMD ROCm"
                        }
                    }
                }
            } catch {}
        }
    }
    # WMI fallback: AMD GPU in device list but no HIP SDK → guide the user.
    # WMI gives a marketing name (e.g. "AMD Radeon 890M") but never a gfx arch.
    # $HasROCm is intentionally NOT set here — we cannot confirm ROCm runtime
    # support without hipinfo or amd-smi.  The name is saved to $ROCmGpuLabel
    # so the name-based inference below can still attempt an arch lookup.
    if (-not $HasROCm) {
        try {
            $wmiGpu = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "AMD|Radeon" } |
                Select-Object -First 1
            if ($wmiGpu) { $ROCmGpuLabel = $wmiGpu.Name }
        } catch {}
    }
    # ── Arch resolution: env-var override → name inference ──────────────────
    # Runs after all probes, even when none confirmed a ROCm runtime ($HasROCm false):
    # the Adrenalin driver alone runs the per-gfx ROCm llama.cpp prebuilt (bundles its
    # own runtime), and all it needs is the gfx arch, inferable from the WMI GPU name.
    # Resolving it here lets setup.ps1 forward --rocm-gfx so a GPU llama.cpp is pulled
    # instead of CPU. (PyTorch ROCm wheels still require a HIP SDK -- gated on $HasROCm
    # below -- so this only affects llama.cpp / inference.)
    if (-not $script:ROCmGfxArch) {
        # 1. Manual override: set UNSLOTH_ROCM_GFX_ARCH=gfx1151 before running.
        if ($env:UNSLOTH_ROCM_GFX_ARCH) {
            $script:ROCmGfxArch = $env:UNSLOTH_ROCM_GFX_ARCH.Trim().ToLower()
            $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
            substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $script:ROCmGfxArch" "Cyan"
        }
        # 2. Best-effort name → arch lookup (amd-smi / WMI). Most-specific first,
        #    first match wins. Covers only arches the ROCm prebuilts support
        #    (gfx120X/110X/1151/1150/103X); unknown names fall back cleanly to CPU.
        elseif ($ROCmGpuLabel) {
            $nameArchTable = @(
                @{ P = "9070 XT|9080";                                        A = "gfx1201" }  # RDNA 4 (Radeon RX 9070 XT / 9080)
                @{ P = "9070|9060";                                           A = "gfx1200" }  # RDNA 4 (Radeon RX 9070 / 9060)
                @{ P = "8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max"; A = "gfx1151" }  # RDNA 3.5 (Strix Halo: Radeon 8060S/8050S/8040S iGPU, Ryzen AI Max+)
                @{ P = "890M|880M|860M|840M|Strix Point|Krackan|HX 37[05]|AI 9 HX|AI 9 36[05]|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33"; A = "gfx1150" }  # RDNA 3.5 (Strix/Krackan Point: Radeon 890M/880M iGPU, Ryzen AI 9 HX 370/375)
                @{ P = "RX 7900|RX 7800|RX 7700(?!S)|PRO W7900|PRO W7800|PRO W7700"; A = "gfx1100" }  # RDNA 3 desktop / workstation (Navi 31)
                @{ P = "RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500|PRO V710"; A = "gfx1102" }  # RDNA 3 (Navi 33)
                @{ P = "780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme"; A = "gfx1103" }  # RDNA 3 iGPU (Phoenix / Hawk Point)
                @{ P = "RX 6900|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900";  A = "gfx1030" }  # RDNA 2 (Navi 21) -- gfx103X family
                @{ P = "RX 6650|RX 6600|PRO W6600|PRO W6650";                  A = "gfx1032" }  # RDNA 2 (Navi 23) -- gfx103X family
                @{ P = "RX 6500|RX 6400|RX 6300|PRO W6400|PRO W6500";          A = "gfx1034" }  # RDNA 2 (Navi 24) -- gfx103X family
            )
            foreach ($row in $nameArchTable) {
                if ($ROCmGpuLabel -match $row.P) {
                    $script:ROCmGfxArch = $row.A
                    $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                    substep "gfx arch inferred from GPU name: $script:ROCmGfxArch" "Cyan"
                    substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$script:ROCmGfxArch to skip inference next time" "Cyan"
                    break
                }
            }
        }
    }
    # Capture ROCm version early for display and wheel selection.
    # Run whenever the HIP SDK binary is present, not just when the device is accessible --
    # hipconfig --version works even when hipinfo reports no ROCm device (driver issue).
    if ($HasROCm -or $HipSdkInstalled) {
        $script:ROCmVersion = $null
        $hipConfigExe = Get-Command hipconfig -ErrorAction SilentlyContinue
        if (-not $hipConfigExe) {
            $hipRoot = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { $null }
            if ($hipRoot) {
                $hipConfigCandidate = Join-Path $hipRoot "bin\hipconfig.exe"
                if (Test-Path $hipConfigCandidate) {
                    $hipConfigEnvLabel = if ($env:HIP_PATH) { "HIP_PATH" } else { "ROCM_PATH" }
                    substep "[WARN] hipconfig not on PATH -- located via ${hipConfigEnvLabel}: $hipConfigCandidate" "Yellow"
                    $hipConfigExe = [PSCustomObject]@{ Source = $hipConfigCandidate }
                }
            }
        }
        if ($hipConfigExe) {
            try {
                $hipVerOut = & $hipConfigExe.Source --version 2>&1 | Out-String
                if ($LASTEXITCODE -eq 0) {
                    $hipVerLine = ($hipVerOut -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
                    if ($hipVerLine -match '(\d+\.\d+)') {
                        $script:ROCmVersion     = $Matches[1]
                        $script:ROCmVersionFull = $hipVerLine
                    }
                }
            } catch {}
        }
        if (-not $script:ROCmVersion -and $amdSmiAllowed) {
            $amdSmiVer = Get-Command "amd-smi" -ErrorAction SilentlyContinue
            if ($amdSmiVer) {
                try {
                    $smiVerOut = Invoke-AmdSmiNoElevate $amdSmiVer.Source @('version')
                    if ($LASTEXITCODE -eq 0 -and $smiVerOut -match 'ROCm version:\s*(\d+\.\d+)') { $script:ROCmVersion = $Matches[1] }
                } catch {}
            }
        }
    }
}

if ($HasNvidiaSmi) {
    step "gpu" "NVIDIA GPU detected"
} elseif ($HasROCm) {
    step "gpu" $ROCmGpuLabel
    $hipSdkPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "on system PATH" }
    substep "HIP SDK: $hipSdkPath"
    if ($script:ROCmVersionFull) { substep "hipconfig: $script:ROCmVersionFull" }
} elseif ($HipSdkInstalled -and $ROCmGpuLabel) {
    # HIP SDK is installed but ROCm can't see the device (driver issue, not SDK issue)
    $sdkVer = if ($script:ROCmVersionFull) { " (HIP $script:ROCmVersionFull)" } else { "" }
    Write-Host ""
    step "gpu" "AMD GPU detected -- not ROCm-accessible$sdkVer" "Yellow"
    substep "Detected: $ROCmGpuLabel" "Yellow"
    substep "[WARN] HIP SDK is installed but hipinfo reports no ROCm-capable device." "Yellow"
    substep "       This is a driver issue, not an SDK issue." "Yellow"
    substep "       Ensure the ROCm compute driver is installed alongside the display driver:" "Yellow"
    substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
} elseif ($script:ROCmGfxArch) {
    # Known arch: PyTorch comes from AMD's bundled-runtime ROCm wheels (repo.amd.com),
    # which ship their own runtime -- HIP SDK optional (only adds the system toolchain).
    Write-Host ""
    step "gpu" "AMD ROCm ($script:ROCmGfxArch)" "Cyan"
    substep "Detected: $ROCmGpuLabel" "Cyan"
    substep "GPU PyTorch uses AMD's bundled-runtime ROCm wheels -- HIP SDK not required (optional)." "Cyan"
    Write-Host ""
} elseif ($ROCmGpuLabel) {
    Write-Host ""
    step "gpu" "AMD GPU detected -- arch unknown" "Yellow"
    substep "Detected: $ROCmGpuLabel" "Yellow"
    substep "Could not determine the GPU arch (gfx...). Install the HIP SDK or set" "Yellow"
    substep "UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm PyTorch:" "Yellow"
    substep "https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    Write-Host ""
} else {
    Write-Host ""
    step "gpu" "none (chat-only / GGUF)" "Yellow"
    substep "Training and GPU inference require an NVIDIA or AMD ROCm GPU." "Yellow"
    Write-Host ""
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
# 1b.5. Visual C++ Redistributable (runtime for the prebuilt llama.cpp + PyTorch)
# ============================================
# Runtime dep, not a build tool: the prebuilt llama-server and PyTorch load it.
Ensure-VCRedist

# ============================================
# 1c. CMake (only needed for a llama.cpp SOURCE build -- detection only)
# ============================================
# Detection only: the prebuilt path needs no compiler, so do not install or exit
# here. Ensure-BuildToolsForLlamaSourceBuild installs CMake if a source build runs.
$HasCmake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
if ($HasCmake) {
    step "cmake" "$(cmake --version | Select-Object -First 1)"
} else {
    step "cmake" "not detected (only needed if a llama.cpp source build is required)" "Yellow"
}

# ============================================
# 1d. Visual Studio Build Tools (only needed for a llama.cpp SOURCE build -- detection only)
# ============================================
# Detection only: detect VS for a possible source build, but never install or exit
# here. Install is deferred to Ensure-BuildToolsForLlamaSourceBuild.
$CmakeGenerator = $null
$VsInstallPath = $null
$vsResult = Find-VsBuildTools

if ($vsResult) {
    $CmakeGenerator = $vsResult.Generator
    $VsInstallPath = $vsResult.InstallPath
    step "vs" "$CmakeGenerator ($($vsResult.Source)) (only used if a source build is needed)"
    if ($vsResult.ClExe) { substep "cl.exe: $($vsResult.ClExe)" }
} else {
    step "vs" "not detected (only needed if a llama.cpp source build is required)" "Yellow"
}

# ============================================
# 1e. CUDA Toolkit (nvcc for llama.cpp build + env vars)
# ============================================
# Defined here but invoked lazily right before a Phase 4 source build; the
# prebuilt llama.cpp path needs no local toolkit. With -RequireOrExit a source
# build is committed, so hard-fail if no driver-compatible toolkit can be found
# or installed. Without it, detection is best-effort and only sets the flag.
function Resolve-CudaToolkit {
    param([switch]$RequireOrExit)
# Toolkit major must be <= the driver's max CUDA major (nvidia-smi "CUDA Version: X.Y");
# a newer-major toolkit fails at runtime ("ggml_cuda_init: failed to initialize CUDA").

$DriverMaxCuda = $null
try {
    # Bounded: source-build toolkit resolution must not hang on a wedged smi.
    # test_resolve_cuda_toolkit.ps1 extracts this function alone into a child
    # pwsh (no Invoke-NvidiaSmiBounded in scope) and stubs nvidia-smi with a
    # .ps1 script, so fall back to direct invocation when the bounded runner
    # is unavailable; production setup.ps1 always has it defined.
    $smiOut = if (Get-Command Invoke-NvidiaSmiBounded -ErrorAction SilentlyContinue) {
        Invoke-NvidiaSmiBounded $NvidiaSmiExe
    } else {
        & $NvidiaSmiExe 2>&1 | Out-String
    }
    # Newer drivers report "CUDA UMD Version: X.Y" instead of "CUDA Version: X.Y"; accept both.
    if ($smiOut -match "CUDA(?: UMD)? Version:\s+([\d]+)\.([\d]+)") {
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
            $isCompat = ($tkMaj -le $drMajorCuda)
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
                substep "CUDA_PATH ($existingCudaPath) has CUDA $tkMaj.$tkMin with major $tkMaj, which exceeds driver CUDA major $drMajorCuda ($DriverMaxCuda)" "Yellow"
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
            # No side-by-side match: a major-compatible toolkit may still be on
            # PATH/CUDA_PATH/a custom dir; use it, else record it as too-new.
            $AnyNvcc = Find-Nvcc
            if ($AnyNvcc) {
                $NvccOut = & $AnyNvcc --version 2>&1 | Out-String
                if ($NvccOut -match "release\s+(\d+)\.(\d+)") {
                    $tkMaj = [int]$Matches[1]; $tkMin = [int]$Matches[2]
                    if ($tkMaj -le $drMajorCuda) {
                        $NvccPath = $AnyNvcc
                        substep "found compatible CUDA Toolkit (nvcc: $NvccPath)"
                    } else {
                        $IncompatibleToolkit = "$tkMaj.$tkMin"
                    }
                }
            }
        }
    }
} else {
    $NvccPath = Find-Nvcc
}

# A newer-major toolkit blocked by the driver: explain the mismatch.
if (-not $NvccPath -and $IncompatibleToolkit) {
    Write-CudaDriverToolkitMismatch -ToolkitVersion $IncompatibleToolkit -DriverMaxCuda $DriverMaxCuda
    if (-not $RequireOrExit) {
        $script:CudaToolkitReady = $false
        return
    }
    # Reached only by a source build (forced, or after a prebuilt-install failure);
    # with no compatible toolkit it must fail (setup.sh degrades to CPU instead).
    Write-Host "" -ForegroundColor Red
    Write-Host "========================================================================" -ForegroundColor Red
    Write-Host "[ERROR] CUDA source build cannot use the installed toolkit with this driver." -ForegroundColor Red
    Write-Host "========================================================================" -ForegroundColor Red
    exit 1
}

# -- No toolkit at all: install via winget (only when a source build needs it) --
if (-not $NvccPath -and $RequireOrExit) {
    Write-Host "CUDA toolkit (nvcc) not found -- installing via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        if ($DriverMaxCuda) {
            # Query winget for available CUDA Toolkit versions
            $drMajor = [int]$DriverMaxCuda.Split('.')[0]
            $AvailableVersions = @()
            try {
                $rawOutput = winget show Nvidia.CUDA --versions --source winget --accept-source-agreements 2>&1 | Out-String
                # Parse version lines (e.g. "12.6", "12.5", "11.8")
                foreach ($line in $rawOutput -split "`n") {
                    $line = $line.Trim()
                    if ($line -match '^\d+\.\d+') {
                        $AvailableVersions += $line
                    }
                }
            } catch {}

            # Filter to compatible major versions and pick the highest
            $BestVersion = $null
            foreach ($ver in $AvailableVersions) {
                $parts = $ver.Split('.')
                $vMajor = [int]$parts[0]
                if ($vMajor -le $drMajor) {
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
                substep "no compatible CUDA Toolkit version found in winget (need CUDA major <= $drMajor)" "Yellow"
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
    if (-not $RequireOrExit) {
        substep "no driver-compatible CUDA Toolkit found -- skipping; prebuilt llama.cpp needs no local toolkit" "Yellow"
        $script:CudaToolkitReady = $false
        return
    }
    Write-Host "[ERROR] CUDA Toolkit (nvcc) is required but could not be found or installed." -ForegroundColor Red
    if ($DriverMaxCuda) {
        Write-Host "        Install a CUDA Toolkit with major version $($DriverMaxCuda.Split('.')[0]) from https://developer.nvidia.com/cuda-toolkit-archive" -ForegroundColor Yellow
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
    $vsCustomizations = Get-VcBuildCustomizationsDir -VsInstallPath $VsInstallPath -Generator $CmakeGenerator
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
# Publish the resolved toolkit to script scope for the Phase 4 build.
$script:NvccPath = $NvccPath
$script:CudaToolkitRoot = $CudaToolkitRoot
$script:CudaArch = $CudaArch
$script:CudaToolkitReady = $true
}

if ($HasROCm) {
    $rocmVerLabel = if ($script:ROCmVersionFull) { "ROCm $script:ROCmVersionFull" } elseif ($script:ROCmVersion) { "ROCm $script:ROCmVersion" } else { "ROCm (version unknown)" }
    step "rocm" $rocmVerLabel
} elseif ($script:ROCmGfxArch) {
    # GPU training/inference works via AMD's bundled-runtime ROCm PyTorch wheels;
    # the HIP SDK is optional (only the system ROCm toolchain).
    step "rocm" "GPU via bundled ROCm wheels ($script:ROCmGfxArch) -- HIP SDK optional" "Cyan"
} elseif ($ROCmGpuLabel) {
    step "rocm" "AMD GPU detected -- arch unknown; HIP SDK not found" "Yellow"
}

# ============================================
# 1f. Node.js / npm (skip if pip-installed or Tauri -- only needed for frontend build)
# ============================================
# Frontend and OXC share this Node floor. The helper returns:
# system | bundled | skip.
function Get-NodeDecision {
    param(
        [string]$NodeVersion,    # `node -v` output, e.g. v22.17.1 (or empty)
        [string]$NpmVersion,     # `npm -v`  output, e.g. 10.9.2  (or empty)
        [string]$SkipInstall     # "1" => never auto-install
    )
    $node = ($NodeVersion -replace '^v', '').Trim()
    $npm = "$NpmVersion".Trim()
    if ($node -match '^\d+\.\d+' -and $npm -match '^\d+') {
        $nodeMajor = [int]($node.Split('.')[0])
        $nodeMinor = [int]($node.Split('.')[1])
        $npmMajor = [int]($npm.Split('.')[0])
        $nodeOk = ($nodeMajor -eq 20 -and $nodeMinor -ge 19) -or
                  ($nodeMajor -eq 22 -and $nodeMinor -ge 12) -or
                  ($nodeMajor -ge 23)
        if ($nodeOk -and $npmMajor -ge 11) { return "system" }
    }
    if ($SkipInstall -eq "1") { return "skip" }
    return "bundled"
}

$SkipFrontend = ($env:SKIP_STUDIO_FRONTEND -eq "1")
$NodeOverride = $null
$NodeParent = $null
$NodeDir = $null
$SysNodeVersion = ""
$SysNpmVersion = ""
$NodeSource = $null

if (-not $IsPipInstall) {
    # Put Node beside the Studio root. OXC can still need npm when the
    # frontend build is skipped.
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $NodeOverride = $env:UNSLOTH_STUDIO_HOME.Trim() }
    elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $NodeOverride = $env:STUDIO_HOME.Trim() }
    if ($NodeOverride) {
        if ($NodeOverride -eq "~") {
            $NodeOverride = $env:USERPROFILE
        } elseif ($NodeOverride -like "~/*" -or $NodeOverride -like "~\*") {
            $NodeOverride = (Join-Path $env:USERPROFILE $NodeOverride.Substring(1).TrimStart('/', '\'))
        }
        if (-not (Test-Path -LiteralPath $NodeOverride -PathType Container)) {
            Write-Host "ERROR: UNSLOTH_STUDIO_HOME/STUDIO_HOME=$NodeOverride does not exist." -ForegroundColor Red
            Write-Host "       Run install.ps1 to create the install root before 'unsloth studio update'." -ForegroundColor Red
            exit 1
        }
        $NodeParent = (Resolve-Path -LiteralPath $NodeOverride).Path
        # An override pointing at the legacy default maps to the legacy sibling
        # ~/.unsloth/node (what the runtime resolver and setup.sh use), not <root>/node.
        $_legacyStudio = Join-Path $env:USERPROFILE ".unsloth\studio"
        if (Test-Path -LiteralPath $_legacyStudio -PathType Container) {
            $_legacyStudio = (Resolve-Path -LiteralPath $_legacyStudio).Path
        }
        if ($NodeParent -eq $_legacyStudio) {
            $NodeParent = Join-Path $env:USERPROFILE ".unsloth"
            $NodeOverride = $null
        }
    } else {
        $NodeParent = Join-Path $env:USERPROFILE ".unsloth"
    }
    $NodeDir = Join-Path $NodeParent "node"

    # Probe system node/npm without letting a missing/broken command abort setup.
    # Under $ErrorActionPreference = "Stop" a bare `node -v` for an absent node
    # throws a terminating error `2>$null` cannot swallow, and a present-but-broken
    # shim throws too. Guard with Get-Command (node/npm independently) + try/catch;
    # empty version => Get-NodeDecision returns "bundled".
    $SysNodeVersion = try { if (Get-Command node -ErrorAction SilentlyContinue) { (node -v 2>$null) } else { "" } } catch { "" }
    $SysNpmVersion = try { if (Get-Command npm -ErrorAction SilentlyContinue) { (npm -v 2>$null) } else { "" } } catch { "" }
    $NodeSource = Get-NodeDecision -NodeVersion "$SysNodeVersion" -NpmVersion "$SysNpmVersion" -SkipInstall "$($env:UNSLOTH_SKIP_NODE_INSTALL)"
}

if ($IsPipInstall) {
    step "frontend" "bundled (pip install)"
} elseif ($SkipFrontend) {
    step "frontend" "bundled (Tauri)"
} else {
    # Stale npm used to trigger system Node changes. Keep this process-local
    # and provision only when the build or OXC needs Node.
    if ($NodeSource -eq "system") {
        substep "Node $SysNodeVersion and npm $SysNpmVersion already meet requirements (system)."
    } elseif ($NodeSource -eq "bundled") {
        substep "Node='$SysNodeVersion' npm='$SysNpmVersion' unsuitable; will use an isolated Node (system left untouched)."
    } else {
        substep "Node='$SysNodeVersion' npm='$SysNpmVersion' unsuitable and UNSLOTH_SKIP_NODE_INSTALL set; frontend build will be skipped." "Yellow"
    }
}

# Conda CPython ships modified DLL search paths that break torch's c10.dll
# loading on Windows; a venv made from conda Python inherits its base_prefix,
# so check the executable path AND sys.base_prefix.
$CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'
function Test-IsConda {
    param([string]$Exe)
    if ($Exe -match $CondaSkipPattern) { return $true }
    try {
        $basePrefix = (& $Exe -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
        if ($basePrefix -match $CondaSkipPattern) { return $true }
    } catch { }
    return $false
}

# 1g. Python (>= 3.11 and < 3.14). Prefer the interpreter install.ps1 already
# resolved and built the venv with (UNSLOTH_SETUP_PYTHON), or the existing
# venv python, before re-probing a system where a 3.14 or a WindowsApps stub
# ahead on PATH would trip the gate. setup.ps1 only updates packages in that
# venv, so the handoff is safe to reuse once validated.
function Resolve-ReusedSetupPython {
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_SETUP_PYTHON) -and
        (Test-Path -LiteralPath $env:UNSLOTH_SETUP_PYTHON)) {
        return $env:UNSLOTH_SETUP_PYTHON
    }
    # Standalone `unsloth studio setup/update` (install.ps1 did not run): derive
    # the venv python from the studio root, mirroring the resolver below.
    $root = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() }
            elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() }
            else { Join-Path $env:USERPROFILE ".unsloth\studio" }
    if ($root -eq "~") {
        # Join-Path with an empty child throws on Windows PowerShell 5.1.
        $root = $env:USERPROFILE
    } elseif ($root -like "~/*" -or $root -like "~\*") {
        $root = Join-Path $env:USERPROFILE $root.Substring(1).TrimStart('/', '\')
    }
    $venvPy = Join-Path $root "unsloth_studio\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { return $venvPy }
    return $null
}
$ReusedSetupPython = Resolve-ReusedSetupPython

$HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$PythonOk = $false
$DetectedPyVer = $null

function Get-CompatiblePythonVersion {
    param([string]$PythonExe)
    try {
        $out = & $PythonExe --version 2>&1 | Out-String
        if ($out -match 'Python (3\.(11|12|13)(\.\d+)?)') {
            return $Matches[1]
        }
    } catch { }
    return $null
}

function Add-PythonDirToProcessPath {
    param([string]$PythonExe)
    try {
        if ($PythonExe -and (Test-Path -LiteralPath $PythonExe)) {
            $resolvedDir = Split-Path -Parent $PythonExe
            $alreadyOnPath = ($env:PATH -split ';' | Where-Object { $_.TrimEnd('\') -ieq $resolvedDir.TrimEnd('\') }).Count -gt 0
            if (-not $alreadyOnPath) {
                $env:PATH = "$resolvedDir;$env:PATH"
            }
            $script:HasPython = $true
        }
    } catch { }
}

# Reuse the install.ps1 / venv interpreter before any system probe.
$ValidatedSetupPython = $null
if ($ReusedSetupPython) {
    $_reusedVer = Get-CompatiblePythonVersion $ReusedSetupPython
    if ($_reusedVer -and -not (Test-IsConda $ReusedSetupPython)) {
        $DetectedPyVer = $_reusedVer
        Add-PythonDirToProcessPath $ReusedSetupPython
        $PythonOk = $true
        $ValidatedSetupPython = $ReusedSetupPython
    }
}

# Fall back to every py.exe on PATH (all-users and per-user launchers can both
# register). -All is required: Windows PowerShell 5.1 returns only the first
# launcher without it, and the PowerShell 7 multi-match array breaks the call
# operator if used directly.
$PyLaunchers = if ($PythonOk) { @() } else { @(Get-Command py -All -CommandType Application -ErrorAction SilentlyContinue) }

foreach ($PyLauncher in $PyLaunchers) {
    if ($PyLauncher.Source -match $CondaSkipPattern) { continue }
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
                        Add-PythonDirToProcessPath $resolvedExe
                    }
                } catch { }
                $PythonOk = $true
                break
            }
        } catch { }
    }
    if ($PythonOk) { break }
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

# Provision Node when the frontend build OR the OXC runtime install needs it (the
# OXC `npm install` runs whenever its dir exists, regardless of dist staleness);
# never eagerly. System Node is used read-only; the isolated one is ours.
$NeedNodeForSetup = (-not $IsPipInstall) -and ($NeedFrontendBuild -or (Test-Path $OxcValidatorDir))
if ($NeedNodeForSetup) {
    if ($NodeSource -eq "skip") {
        if ($NeedFrontendBuild) {
            step "frontend" "skipped (no suitable Node; system left untouched)" "Yellow"
        }
        $NeedFrontendBuild = $false
        substep "found Node='$SysNodeVersion' npm='$SysNpmVersion'; Studio needs Node >=20.19/22.12/23 and npm >= 11" "Yellow"
        substep "install a suitable Node + npm, or unset UNSLOTH_SKIP_NODE_INSTALL to let Unsloth manage an isolated Node" "Yellow"
    } elseif ($NodeSource -eq "bundled") {
        New-Item -ItemType Directory -Force -Path $NodeParent -ErrorAction SilentlyContinue | Out-Null
        # Minimal ownership guard for a custom-home dir (the full Studio-owned
        # helpers are defined later); never os.replace over a user-owned dir.
        if ($NodeOverride -and (Test-Path -LiteralPath $NodeDir -PathType Container)) {
            $nodeOwnedMarker = Join-Path $NodeDir ".unsloth-studio-owned"
            $nodeMeta = Join-Path $NodeDir "UNSLOTH_NODE_PREBUILT_INFO.json"
            if (-not (Test-Path -LiteralPath $nodeOwnedMarker) -and -not (Test-Path -LiteralPath $nodeMeta)) {
                Write-Host "[ERROR] $NodeDir already exists and is not a Studio-owned Node install." -ForegroundColor Red
                Write-Host "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
                exit 1
            }
        }
        substep "installing isolated Node (system Node/npm left untouched)..."
        # The main Python resolver runs later; bare `python` may be a Store stub or
        # absent this early, so prefer the validated handed-off/venv Python.
        $NodeInstallPython = if ($ValidatedSetupPython) { $ValidatedSetupPython } else { "python" }
        $nodeOut = & $NodeInstallPython "$PSScriptRoot\install_node_prebuilt.py" --install-dir $NodeDir 2>&1 | Out-String
        $nodeExit = $LASTEXITCODE
        if ($nodeExit -eq 3) {
            Write-Host $nodeOut -ForegroundColor DarkGray
            step "node" "install blocked by another active Studio install" "Red"
            exit 3
        } elseif ($nodeExit -ne 0) {
            Write-Host $nodeOut -ForegroundColor DarkGray
            Write-Host "[ERROR] Could not install an isolated Node automatically." -ForegroundColor Red
            Write-Host "        Install Node >= 20.19 (with npm >= 11) from https://nodejs.org/ and re-run, or check your network." -ForegroundColor Yellow
            exit 1
        }
        if ($NodeOverride -and (Test-Path -LiteralPath $NodeDir -PathType Container)) {
            New-Item -ItemType File -Force -Path (Join-Path $NodeDir ".unsloth-studio-owned") -ErrorAction SilentlyContinue | Out-Null
        }
        # Windows Node zip ships node.exe + npm.cmd at the root; prepend it (this
        # process only) so node/npm/bun resolve here for the build.
        $env:PATH = "$NodeDir;" + $env:PATH
        # Keep npm and module resolution inside the isolated Node.
        $env:NPM_CONFIG_PREFIX = $NodeDir
        $env:npm_config_prefix = $NodeDir
        Remove-Item Env:NODE_PATH -ErrorAction SilentlyContinue
        step "node" "$(node -v) | npm $(npm -v) (isolated)"

        # bun (optional, faster installs); npm -g stays in the isolated prefix.
        if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
            substep "installing bun (faster frontend package installs)..."
            $prevEAP_bun = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            Invoke-SetupCommand { npm install -g bun --allow-scripts=bun @NpmRegistryArgs } | Out-Null
            $ErrorActionPreference = $prevEAP_bun
            Refresh-Environment
            # Refresh-Environment rebuilds PATH (Machine;User;current), demoting the
            # isolated-Node prepend; re-prepend so it wins for the build and OXC step.
            $env:PATH = "$NodeDir;" + $env:PATH
            $env:NPM_CONFIG_PREFIX = $NodeDir
            $env:npm_config_prefix = $NodeDir
            Remove-Item Env:NODE_PATH -ErrorAction SilentlyContinue
            if (Get-Command bun -ErrorAction SilentlyContinue) {
                substep "bun installed ($(bun --version))"
            } else {
                substep "bun install skipped (npm will be used instead)"
            }
        }
    } else {
        # system Node already satisfies requirements; use it as-is. We do NOT
        # install global packages (bun) here -- the build falls back to npm.
        step "node" "$SysNodeVersion | npm $SysNpmVersion (system)"
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
        $bunExit = Invoke-SetupCommand { bun install @NpmRegistryArgs }
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
            $bunExit = Invoke-SetupCommand { bun install @NpmRegistryArgs }
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
        $npmExit = Invoke-SetupCommand { npm install @NpmRegistryArgs }
        if ($npmExit -ne 0) {
            Pop-Location
            $ErrorActionPreference = $prevEAP_npm
            foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
            Write-Host "[ERROR] npm install failed (exit code $npmExit)" -ForegroundColor Red
            Write-Host "   Try running 'npm install' manually in frontend/ to see errors" -ForegroundColor Yellow
            Show-NpmRegistryHint
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

if ((Test-Path $OxcValidatorDir) -and $NodeSource -ne "skip" -and (Get-Command npm -ErrorAction SilentlyContinue)) {
    substep "installing OXC validator runtime..."
    $prevEAP_oxc = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $OxcValidatorDir
    $oxcInstallExit = Invoke-SetupCommand { npm install @NpmRegistryArgs }
    if ($oxcInstallExit -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_oxc
        Write-Host "[ERROR] OXC validator npm install failed (exit code $oxcInstallExit)" -ForegroundColor Red
        Show-NpmRegistryHint
        exit 1
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_oxc
    step "oxc runtime" "installed"
} elseif ((Test-Path $OxcValidatorDir) -and $NodeSource -ne "skip") {
    # No npm on PATH (e.g. a pip install with no system Node and no isolated Node
    # provisioned). Skip rather than abort; the runtime resolver degrades. Mirrors setup.sh.
    substep "OXC validator runtime skipped (no npm found); code validation degrades until Node is available" "Yellow"
}

# ==========================================================================
#  PHASE 3: Python environment + dependencies
# ==========================================================================
Write-Host ""
substep "setting up Python environment..."

# Find Python -- skip Anaconda/Miniconda distributions ($CondaSkipPattern and
# Test-IsConda are defined above the 1g gate). Standalone CPython (python.org,
# winget, uv) does not have conda's torch c10.dll loading issue.
$PythonCmd = $null

# 0. Reuse the interpreter install.ps1 already resolved and built the venv with
#    (UNSLOTH_SETUP_PYTHON, or the existing venv python) before probing the
#    system -- it is already validated as supported and non-conda.
if ($ReusedSetupPython) {
    try {
        $out = & $ReusedSetupPython --version 2>&1 | Out-String
        if ($out -match 'Python 3\.(\d+)') {
            $pyMinor = [int]$Matches[1]
            if ($pyMinor -ge 11 -and $pyMinor -le 13 -and -not (Test-IsConda $ReusedSetupPython)) {
                $PythonCmd = $ReusedSetupPython
            }
        }
    } catch { }
}

# 1. Try the Python Launcher (py.exe) first -- most reliable on Windows.
#    Enumerate every launcher with -All (Windows PowerShell 5.1 returns only
#    the first match without it) and search each for a supported, non-conda
#    interpreter.
$PyLaunchersResolve = if ($PythonCmd) { @() } else { @(Get-Command py -All -CommandType Application -ErrorAction SilentlyContinue) }
foreach ($pyLauncher in $PyLaunchersResolve) {
    if ($pyLauncher.Source -match $CondaSkipPattern) { continue }
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
    if ($PythonCmd) { break }
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

substep "Python found: $PythonCmd"

# The venv must already exist (created by install.ps1); this script only
# updates packages. UNSLOTH_STUDIO_HOME (or STUDIO_HOME alias) overrides the
# root. UNSLOTH_STUDIO_HOME wins when both are set. Whitespace-only values
# are treated as unset to match Python .strip() semantics.
$_studioOverrideVar = $null
$_studioOverride = $null
if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
    $_studioOverrideVar = "UNSLOTH_STUDIO_HOME"
    $_studioOverride = $env:UNSLOTH_STUDIO_HOME.Trim()
} elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
    $_studioOverrideVar = "STUDIO_HOME"
    $_studioOverride = $env:STUDIO_HOME.Trim()
}
if ($_studioOverride) {
    if ($_studioOverride -eq "~" -or $_studioOverride -like "~/*" -or $_studioOverride -like "~\*") {
        $_studioOverride = (Join-Path $env:USERPROFILE $_studioOverride.Substring(1).TrimStart('/','\'))
    }
    if (Test-Path -LiteralPath $_studioOverride -PathType Container) {
        $StudioHome = (Resolve-Path -LiteralPath $_studioOverride).Path
        # why: mirror setup.sh:417 and install.ps1:130 -- fail fast when the
        # custom root is read-only instead of erroring later while creating
        # sidecar venvs / installing packages.
        $_setupWriteProbe = Join-Path $StudioHome (".unsloth-write-probe-" + [guid]::NewGuid())
        try {
            [System.IO.File]::WriteAllText($_setupWriteProbe, "")
            Remove-Item -LiteralPath $_setupWriteProbe -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "ERROR: $_studioOverrideVar=$StudioHome is not writable." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "ERROR: $_studioOverrideVar=$_studioOverride does not exist." -ForegroundColor Red
        Write-Host "       Run install.ps1 to create the install root before 'unsloth studio update'." -ForegroundColor Red
        exit 1
    }
} else {
    $StudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
}
$VenvDir = Join-Path $StudioHome "unsloth_studio"

# why: in env-override mode $StudioHome is user-chosen; require the
# ownership marker before Remove-Item so unrelated dirs survive. Gated on
# the canonical comparison so an override pointing at the legacy default
# still behaves like a default install.
$StudioOwnedMarker = ".unsloth-studio-owned"
$LegacyStudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
$_studioHomeCanon = $StudioHome
if (Test-Path -LiteralPath $_studioHomeCanon -PathType Container) {
    $_studioHomeCanon = (Resolve-Path -LiteralPath $_studioHomeCanon).Path
}
if (Test-Path -LiteralPath $LegacyStudioHome -PathType Container) {
    $LegacyStudioHome = (Resolve-Path -LiteralPath $LegacyStudioHome).Path
}
$StudioHomeIsCustom = ($_studioHomeCanon -ne $LegacyStudioHome)
# Directory-local evidence that Studio created $Path, used to adopt a custom-home
# llama.cpp predating the .unsloth-studio-owned marker (see setup.sh). Only the
# prebuilt UNSLOTH_PREBUILT_INFO.json counts; source builds are indistinguishable
# from a user clone on Windows and stay under the strict guard.
function Test-StudioOwnedAdoptable {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath (Join-Path $Path "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf) { return $true }
    return $false
}
function Assert-StudioOwnedOrAbsent {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    if ($StudioHomeIsCustom -and -not (Test-Path -LiteralPath (Join-Path $Path $StudioOwnedMarker) -PathType Leaf)) {
        if (Test-StudioOwnedAdoptable $Path) {
            Mark-StudioOwned $Path
            return
        }
        Write-Host "[ERROR] $Path already exists and is not marked as a Studio-owned $Label." -ForegroundColor Red
        Write-Host "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
        exit 1
    }
}
function Mark-StudioOwned {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    try {
        [System.IO.File]::WriteAllText((Join-Path $Path $StudioOwnedMarker), "")
    } catch {}
}

# Stale-venv detection: if the venv exists but its torch flavor no longer
# matches the current machine, repair according to invocation context.
# - install.ps1 sets UNSLOTH_INSTALL_ROLLBACK_MANAGED=1 so setup can delegate
#   to the installer-level rollback that restores the previous environment.
# - direct `unsloth studio update` keeps the pre-existing self-repair behavior.
# In no-torch mode, a missing torch package is expected.
$NoTorchMode = $env:UNSLOTH_NO_TORCH -match '^(?i:true|1|yes)$'
$InstallerManagedSetup = $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -match '^(?i:true|1|yes)$'
if ((Test-Path -LiteralPath $VenvDir -PathType Container) -and -not $NoTorchMode) {
    $VenvPyExe = Join-Path $VenvDir "Scripts\python.exe"
    $installedTorchTag = $null
    $shouldRebuild = $false
    # Set when a stale venv under a pin is repaired in place (force-reinstall) not wiped.
    $script:PinChangedForceReinstall = $false

    if (Test-Path -LiteralPath $VenvPyExe) {
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
                } elseif ($torchVer -match '\+rocm') {
                    # Any +rocm / gfx wheel -> generic "rocm" flavor (the exact version is
                    # repaired later by install_python_stack.py; here we only need the flavor).
                    $installedTorchTag = "rocm"
                } elseif ($torchVer -match '\+cpu') {
                    $installedTorchTag = "cpu"
                } else {
                    # Untagged wheel (plain "2.x.y" from PyPI) -> cpu.
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
        $_pinnedIdx = Get-PinnedTorchIndexUrl
        $_expectedKnown = $true
        if ($_pinnedIdx) {
            $_pinLeaf = Get-TorchIndexLeaf $_pinnedIdx
            # Digit-gated like the install selection: a custom rocm-* leaf (rocm-current /
            # rocm-rel-7.2.1) is NOT a ROCm family and must not be stale-compared.
            if (Test-PipRocmFamilyLeaf $_pinLeaf) {
                # Don't collapse a pinned ROCm/gfx leaf to a generic "rocm" (would mask a
                # family change, rocm6.4 -> gfx1151). Get-RocmPinStaleTags uses the SAME 2.11
                # allowlist as the install path, so a gfx110X-all/gfx90a/gfx908 pin on a
                # valid <2.11 wheel is NOT stale.
                $_rocmTags = Get-RocmPinStaleTags -PinLeaf $_pinLeaf -TorchVersion $torchVer
                $expectedTorchTag  = $_rocmTags.Expected
                $installedTorchTag = $_rocmTags.Installed
            } elseif ((Test-CudaFamilyLeaf $_pinLeaf) -or $_pinLeaf -eq 'cpu') {
                # cu*/cpu leaves stay specific so a cu126-vs-cu128 mismatch rebuilds;
                # /custom and /current fall through to the unknown-index branch below.
                $expectedTorchTag = $_pinLeaf
            } else {
                # Custom index whose leaf is not a torch flavor (a /simple mirror): the
                # flavor can't be inferred, so never treat the venv as stale over it.
                $_expectedKnown = $false
                $expectedTorchTag = $installedTorchTag
            }
        } elseif ($HasNvidiaSmi) {
            $expectedTorchTag = Get-PytorchCudaTag
        } elseif ($HasROCm -or $script:ROCmGfxArch) {
            # AMD/ROCm host with no explicit pin: an existing +rocm wheel is correct (gfx
            # arch counts even when $HasROCm is false). But only the arches the install path
            # maps to a repo.amd.com index get ROCm torch; an unmapped arch falls back to
            # CPU there, so expect "cpu" for those or a correct CPU venv rebuilds every update.
            $_rocmWheelArches = @(
                "gfx1201", "gfx1200",           # RDNA 4
                "gfx1151", "gfx1150",           # RDNA 3.5 (Strix Halo/Point)
                "gfx1103", "gfx1102", "gfx1101", "gfx1100",  # RDNA 3
                "gfx90a", "gfx908"              # MI200 / MI100
            )
            if ($script:ROCmGfxArch -and ($_rocmWheelArches -contains $script:ROCmGfxArch)) {
                # A correct +rocm wheel is not stale. A CPU wheel on a supported AMD arch is
                # NOT wiped either (the AMD Windows ROCm override below upgrades it in place);
                # expect "cpu" for that case. A wrong CUDA wheel still rebuilds.
                if ($installedTorchTag -eq "cpu") {
                    $expectedTorchTag = "cpu"
                } else {
                    $expectedTorchTag = "rocm"
                }
            } else {
                $expectedTorchTag = "cpu"
            }
        } else {
            $expectedTorchTag = "cpu"
        }
        if ($_expectedKnown -and $installedTorchTag -and $installedTorchTag -ne $expectedTorchTag) {
            $shouldRebuild = $true
        }
    }

    # A stale venv under a pin whose torch still imports is repaired IN PLACE (the
    # dependency pass force-reinstalls from the pin). The rebuild path wipes the venv and
    # would strand a direct `studio update` at "Virtual environment not found"; only a
    # broken venv or an unpinned drift wipes.
    if ($shouldRebuild -and $_pinnedIdx -and $installedTorchTag) {
        substep "Torch-index pin changed ($installedTorchTag) -- reinstalling torch from the pin in place." "Cyan"
        $script:PinChangedForceReinstall = $true
        $shouldRebuild = $false
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
        # why: mirror install.ps1 env-mode guard so an update against a custom
        # UNSLOTH_STUDIO_HOME never wipes an unrelated unsloth_studio venv;
        # -PathType Leaf rejects a directory masquerading as the sentinel.
        if (
            $StudioHomeIsCustom -and
            -not (Test-Path -LiteralPath (Join-Path $VenvDir $StudioOwnedMarker) -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "share\studio.conf") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe") -PathType Leaf)
        ) {
            Write-Host "[ERROR] $VenvDir already exists but does not look like an Unsloth Studio install." -ForegroundColor Red
            Write-Host "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
            exit 1
        }
        try {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Host "   [ERROR] Could not remove stale venv: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "           Close any running Studio/Python processes and re-run setup." -ForegroundColor Red
            exit 1
        }
    }
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvDir" -ForegroundColor Red
    Write-Host "        Run install.ps1 first to create the environment:" -ForegroundColor Yellow
    Write-Host "        irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
} else {
    substep "reusing existing virtual environment at $VenvDir"
    $_venvPyExe = Join-Path $VenvDir "Scripts\python.exe"
    if (Test-Path -LiteralPath $_venvPyExe) {
        try {
            $_venvPyVer = (& $_venvPyExe --version 2>&1 | Out-String).Trim()
            if ($_venvPyVer) { substep $_venvPyVer }
        } catch {}
    }
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
    # An explicit --index-url must win: inherited uv index vars otherwise pull CPU torch
    # over the CUDA/ROCm build (#6898), so drop them for pinned installs (the scrub covers
    # the whole function since the pip fallback honours PIP_* too). UV_TORCH_BACKEND /
    # UV_FIND_LINKS also reroute a pinned resolve; UV_NO_CONFIG=1 (+ dropping UV_CONFIG_FILE)
    # stops a uv.toml/pyproject index outranking the pin (uv 0.10). PIP_NO_INDEX=1 makes the
    # pip fallback ignore ALL indexes (including the pinned --index-url), and PIP_INDEX_URL
    # replaces the primary index the pin sets, so both must go for a pinned install too.
    $saved = @{}
    $pinned = @($Args_) -contains '--index-url'
    if ($pinned) {
        foreach ($n in 'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX', 'UV_EXTRA_INDEX_URL',
                       'UV_TORCH_BACKEND', 'UV_FIND_LINKS', 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS',
                       'PIP_NO_INDEX', 'PIP_INDEX_URL',
                       'UV_CONFIG_FILE', 'UV_NO_CONFIG', 'PIP_CONFIG_FILE') {
            $saved[$n] = [Environment]::GetEnvironmentVariable($n)
            Remove-Item "Env:$n" -ErrorAction SilentlyContinue
        }
        $env:UV_NO_CONFIG = '1'
        # A `pip config` global.extra-index-url still adds indexes to the pip FALLBACK;
        # PIP_CONFIG_FILE = 'nul' (Windows devnull) loads NO config (uv ignores pip config).
        $env:PIP_CONFIG_FILE = 'nul'
    }
    try {
        if ($UseUv) {
            $VenvPy = (Get-Command python).Source
            $result = & uv pip install --python $VenvPy @Args_ 2>&1
            if ($LASTEXITCODE -eq 0) { return }
        }
        & python -m pip install @Args_ 2>&1
    }
    finally {
        if ($pinned) {
            Remove-Item "Env:UV_NO_CONFIG" -ErrorAction SilentlyContinue
            Remove-Item "Env:PIP_CONFIG_FILE" -ErrorAction SilentlyContinue
        }
        foreach ($n in $saved.Keys) { if ($null -ne $saved[$n]) { Set-Item "Env:$n" $saved[$n] } }
    }
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
        # A pre-#6483-fix install can be stuck on anyio>=4.14 even though
        # $_PkgName itself is current; the fast path above would otherwise
        # never reach install_python_stack's anyio repair (#6797).
        $_anyioBad = $false
        try {
            & python -c "
import re, sys
from importlib.metadata import version, PackageNotFoundError
try:
    parts = version('anyio').split('.')
    major = int(parts[0])
    minor = int(re.sub(r'[^0-9].*', '', parts[1])) if len(parts) > 1 else 0
except (PackageNotFoundError, ValueError, IndexError):
    sys.exit(1)
sys.exit(0 if (major, minor) >= (4, 14) else 1)
" 2>$null
            if ($LASTEXITCODE -eq 0) { $_anyioBad = $true }
        } catch {}
        if ($_anyioBad) {
            substep "anyio >=4.14 found (#6483) -- forcing dependency pass to repair..." "Cyan"
            $SkipPythonDeps = $false
        }
        # ...but not if an AMD GPU is present and installed PyTorch is CPU-only
        # (host predates ROCm-wheel support, or GPU added later): the fast "up to
        # date" path would leave the user on CPU torch with Train/Export disabled.
        # Force the dependency pass so the ROCm wheels get installed.
        if ($script:ROCmGfxArch) {
            $_torchIsCpu = $true
            try {
                & python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) { $_torchIsCpu = $false }
            } catch {}
            if ($_torchIsCpu) {
                substep "AMD GPU ($script:ROCmGfxArch) detected but installed PyTorch is CPU-only -- reinstalling ROCm PyTorch" "Cyan"
                $SkipPythonDeps = $false
            }
        }
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
#     pip install unsloth 2>&1 | Out-Null
# }

# A torch-index pin change repairs in place: force the dependency pass so the torch install
# below force-reinstalls from the new pin (else the fast path keeps the old wheel).
if ($script:PinChangedForceReinstall) { $SkipPythonDeps = $false }

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

# Triton/inductor filenames are long and can hit Windows MAX_PATH (260). With long
# paths on, cache under Studio home; else use a short drive-root dir for headroom.
if ($LongPathsEnabled) {
    $TorchCacheDir = Join-Path $StudioHome "TORCHINDUCTOR_CACHE_DIR"
} else {
    $TorchCacheDir = "C:\tc"
}
if (-not (Test-Path -LiteralPath $TorchCacheDir)) { [System.IO.Directory]::CreateDirectory($TorchCacheDir) | Out-Null }
$env:TORCHINDUCTOR_CACHE_DIR = $TorchCacheDir
[Environment]::SetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', $TorchCacheDir, 'User')
substep "TORCHINDUCTOR_CACHE_DIR set to $TorchCacheDir (avoids MAX_PATH issues)"

# Explicit pin (URL or family) wins over GPU probing and suppresses the AMD reroute below;
# matches install.sh / install.ps1 / install_python_stack.py.
$PinnedTorchIndexUrl = Get-PinnedTorchIndexUrl
$TorchIndexPinned = [bool]$PinnedTorchIndexUrl
if ($PinnedTorchIndexUrl) {
    $CuTag = Get-TorchIndexLeaf $PinnedTorchIndexUrl
} elseif ($HasNvidiaSmi) {
    $CuTag = Get-PytorchCudaTag
} else {
    $CuTag = "cpu"
}

# ── GPU arch → newest compatible Windows ROCm wheel release ──
# Wheels bundle their own ROCm runtime; the installed HIP SDK version does
# not constrain which release to use.  Always picks the newest release that
# supports the GPU architecture.
# ── AMD Windows ROCm torch override ──────────────────────────────────────────
# Uses AMD's arch-specific pip index (repo.amd.com/rocm/whl/{arch}/).
# Wheels bundle their own ROCm runtime; HIP SDK version is irrelevant.
$ROCmGfxArch = $script:ROCmGfxArch
$ROCmIndexUrl = $null
# Install AMD ROCm PyTorch wheels when ROCm is confirmed OR a gfx arch is known
# (name-inferred on Adrenalin-only hosts). The per-arch wheels bundle the runtime
# (rocm-sdk-libraries-<gfx>), so torch.cuda.is_available() is True without a HIP
# SDK -- which flips Studio out of chat-only (CHAT_ONLY) and enables Train/Export.
# Gating on $HasROCm alone left Strix Halo / Radeon 8060S on CPU torch; a failed
# ROCm install still falls back to CPU below, so this is safe.
if (-not $TorchIndexPinned -and ($HasROCm -or $ROCmGfxArch) -and $CuTag -eq "cpu") {
    $amdIndexBase = if ($env:UNSLOTH_ROCM_WINDOWS_MIRROR) { $env:UNSLOTH_ROCM_WINDOWS_MIRROR.TrimEnd('/') } else { "https://repo.amd.com/rocm/whl" }
    $archFamilyMap = @{
        "gfx1201" = "gfx120X-all"; "gfx1200" = "gfx120X-all"  # RDNA 4
        "gfx1151" = "gfx1151";     "gfx1150" = "gfx1150"      # RDNA 3.5 (Strix Halo/Point)
        "gfx1103" = "gfx110X-all"; "gfx1102" = "gfx110X-all"  # RDNA 3
        "gfx1101" = "gfx110X-all"; "gfx1100" = "gfx110X-all"
        "gfx90a"  = "gfx90a";      "gfx908"  = "gfx908"       # MI200/MI100
    }
    # gfx120X and Strix have a null _grouped_mm kernel on torch <2.11.0.
    # Mirrors the $torchFloorMap in install.ps1 so both installers enforce
    # the same floor and ceiling when pulling from AMD's per-arch index.
    $torchFloorMap = @{
        "gfx1201" = "torch>=2.11.0,<2.12.0"; "gfx1200" = "torch>=2.11.0,<2.12.0"
        "gfx1151" = "torch>=2.11.0,<2.12.0"; "gfx1150" = "torch>=2.11.0,<2.12.0"
    }
    # Companion ranges for torchvision/torchaudio -- must stay in sync with the
    # torch ceiling so pip can always find a consistent trio on AMD's per-arch
    # index.  AMD publishes each package independently and may add a newer
    # torchvision (e.g. 0.27 for torch 2.12) before removing 0.26, which would
    # cause pip to resolve an ABI-incompatible set if these are left bare.
    # Matches _ROCM_TORCH_PKG_SPECS["rocm7.2"] in install_python_stack.py.
    # Bump all three ceilings together when torch 2.12.x is validated.
    $torchvisionFloorMap = @{
        "gfx1201" = "torchvision>=0.26.0,<0.27.0"; "gfx1200" = "torchvision>=0.26.0,<0.27.0"
        "gfx1151" = "torchvision>=0.26.0,<0.27.0"; "gfx1150" = "torchvision>=0.26.0,<0.27.0"
    }
    $torchaudioFloorMap = @{
        "gfx1201" = "torchaudio>=2.11.0,<2.12.0"; "gfx1200" = "torchaudio>=2.11.0,<2.12.0"
        "gfx1151" = "torchaudio>=2.11.0,<2.12.0"; "gfx1150" = "torchaudio>=2.11.0,<2.12.0"
    }
    $archFamily = if ($ROCmGfxArch -and $archFamilyMap.ContainsKey($ROCmGfxArch)) { $archFamilyMap[$ROCmGfxArch] } else { $null }
    $ROCmTorchSpec  = if ($ROCmGfxArch -and $torchFloorMap.ContainsKey($ROCmGfxArch))        { $torchFloorMap[$ROCmGfxArch]        } else { "torch" }
    $ROCmVisionSpec = if ($ROCmGfxArch -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch))  { $torchvisionFloorMap[$ROCmGfxArch]  } else { "torchvision" }
    $ROCmAudioSpec  = if ($ROCmGfxArch -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch))   { $torchaudioFloorMap[$ROCmGfxArch]   } else { "torchaudio" }
    if ($archFamily) {
        $ROCmIndexUrl = "$amdIndexBase/$archFamily/"
    } elseif ($ROCmGfxArch) {
        # GPU arch detected but not in the supported wheel map — warn explicitly
        # so the user knows why they are getting CPU PyTorch instead of ROCm.
        substep "[WARN] AMD GPU ($ROCmGfxArch) not in supported arch list -- falling back to CPU-only PyTorch" "Yellow"
        substep "       Supported: gfx1200/1201 (RDNA 4), gfx1150/1151 (RDNA 3.5), gfx1100-1103 (RDNA 3), gfx90a, gfx908" "Yellow"
    } else {
        # HIP SDK present ($HasROCm=true via amd-smi) but gcnArchName was not
        # readable — warn rather than silently falling back to CPU PyTorch.
        substep "[WARN] AMD GPU detected (HIP SDK present) but GPU arch could not be read -- falling back to CPU-only PyTorch" "Yellow"
        substep "       Arch detection requires hipinfo to report gcnArchName. Re-install the HIP SDK if this is unexpected." "Yellow"
    }
}

# A pinned gfx*/rocm index skips the auto-reroute above; route it through the ROCm install
# path with the same floor/companions the unpinned AMD path uses (mirrors install.ps1),
# else the CUDA branch installs bare torch and resolves a known-bad <2.11 or ABI-mismatched
# wheel for gfx115x/gfx120x/rocm>=7.2.
if ($TorchIndexPinned -and -not $ROCmIndexUrl -and $PinnedTorchIndexUrl) {
    $_pinLeaf = Get-TorchIndexLeaf $PinnedTorchIndexUrl
    $_pinRocm211 = $false
    # Anchor the match ($) so a suffixed custom leaf (rocm7.2-private) falls through to the
    # verbatim install instead of being floored by its rocm7.2 prefix.
    if ($_pinLeaf -match '^rocm(\d+)\.(\d+)$') {
        # Only KNOWN-2.11 rocm (rocm7.2) gets the floor (no speculative floor). Matches
        # Test-RocmKnown211Version / _ROCM_KNOWN_TORCH211_VERSIONS.
        $_pinRocm211 = Test-RocmKnown211Version -Major ([int]$Matches[1]) -Minor ([int]$Matches[2])
    }
    # Only the 2.11 gfx arches need the floor; others publish <2.11 and stay bare. Reuse
    # Test-RocmGfx211Leaf so this allowlist and the stale-venv check never diverge.
    $_pinGfx211 = Test-RocmGfx211Leaf $_pinLeaf
    if ($_pinGfx211 -or $_pinRocm211) {
        $ROCmIndexUrl   = $PinnedTorchIndexUrl
        $ROCmTorchSpec  = "torch>=2.11.0,<2.12.0"
        $ROCmVisionSpec = "torchvision>=0.26.0,<0.27.0"
        $ROCmAudioSpec  = "torchaudio>=2.11.0,<2.12.0"
        substep "pinned ROCm index ($_pinLeaf) -- enforcing $ROCmTorchSpec" "Cyan"
    } elseif (Test-PipRocmFamilyLeaf $_pinLeaf) {
        # Other gfx / older rocm (<=7.1) ship torch <2.11; route via the ROCm path with
        # bare specs. Only EXACT rocm<digits> and gfx* are --index-url families; a suffixed
        # leaf stays on the verbatim path. Mirrors install.ps1 / _is_pip_rocm_family_leaf.
        $ROCmIndexUrl   = $PinnedTorchIndexUrl
        $ROCmTorchSpec  = "torch"
        $ROCmVisionSpec = "torchvision"
        $ROCmAudioSpec  = "torchaudio"
    }
}

$PyTorchWhlBase = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }

# A full URL pin is used verbatim; a family pin already set $CuTag. A pinned ROCm install
# goes through $ROCmIndexUrl; on failure the fallback uses the CPU index, not the ROCm pin.
$TorchInstallIndexUrl = if ($ROCmIndexUrl) { "$PyTorchWhlBase/cpu" } elseif ($PinnedTorchIndexUrl) { $PinnedTorchIndexUrl } else { "$PyTorchWhlBase/$CuTag" }

$ROCmCpuFallback = $false
if ($ROCmIndexUrl) {
    substep "installing PyTorch (AMD ROCm, $ROCmGfxArch)..."
    if ($ROCmTorchSpec -ne "torch") {
        substep "  enforcing $ROCmTorchSpec $ROCmVisionSpec $ROCmAudioSpec (known _grouped_mm bug in older wheels)" "Cyan"
    }
    if ($script:UnslothVerbose) {
        Fast-Install $ROCmTorchSpec $ROCmVisionSpec $ROCmAudioSpec --force-reinstall --index-url $ROCmIndexUrl
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install $ROCmTorchSpec $ROCmVisionSpec $ROCmAudioSpec --force-reinstall --index-url $ROCmIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-Host "[WARN] AMD ROCm PyTorch install failed -- falling back to CPU" -ForegroundColor Yellow
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Yellow
        $ROCmIndexUrl = $null
        $ROCmCpuFallback = $true
    } else {
        # Tell install_python_stack.py to skip probe + suppress manual-install warning.
        $env:UNSLOTH_ROCM_TORCH_INSTALLED = "1"
        substep "GPU ROCm PyTorch installed ($ROCmGfxArch) -- training and GPU inference will use the GPU" "Cyan"
    }
}

if (-not $ROCmIndexUrl -and ($CuTag -eq "cpu" -or $ROCmCpuFallback)) {
    substep "installing PyTorch (CPU-only)..."
    # After an AMD ROCm fallback, force-reinstall so a partial ROCm torch (which satisfies
    # the CPU torch>= range) is replaced by the CPU build; skip on a genuine CPU host to
    # stay fast. $ROCmCpuFallback matters when a PINNED ROCm index failed ($CuTag is still
    # the rocm leaf), else the partial ROCm torch survives.
    # Build the array directly: an if-expression collapses @("x") to a scalar string that
    # @splat would enumerate char-by-char.
    $cpuForce = @()
    if ($ROCmCpuFallback) { $cpuForce = @("--force-reinstall") }
    # --force-reinstall on a pin change: a stale +cu / +rocm wheel still satisfies the CPU
    # torch>= range, so uv would keep it and only swap companions.
    if ($script:PinChangedForceReinstall) { $cpuForce = @("--force-reinstall") }
    if ($script:UnslothVerbose) {
        Fast-Install torch torchvision torchaudio @cpuForce --index-url $TorchInstallIndexUrl
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install torch torchvision torchaudio @cpuForce --index-url $TorchInstallIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-Host "[FAILED] PyTorch install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
        exit 1
    }
} elseif (-not $ROCmIndexUrl) {
    substep "installing PyTorch with CUDA support ($CuTag)..."
    substep "(This download is ~2.8 GB -- may take a few minutes)"
    # --force-reinstall on a pin change: an installed cuXXX wheel satisfies the bare torch
    # requirement (PEP 440 ignores the +cuXXX tag), so without it a changed CUDA pin (cu126
    # -> cu128) never applies.
    $cudaForce = @()
    if ($script:PinChangedForceReinstall) { $cudaForce = @("--force-reinstall") }
    # An unknown-leaf custom pin (/simple, /current) routes here with $CuTag as that leaf.
    # Bound the trio like the fresh custom-pin install paths so a mirror can't
    # pull an ABI-newer companion against the capped torch. Known cu* leaves keep bare specs.
    $cudaTorchSpec = "torch"
    $cudaVisionSpec = "torchvision"
    $cudaAudioSpec = "torchaudio"
    if ($TorchIndexPinned -and -not (Test-CudaFamilyLeaf $CuTag)) {
        $cudaTorchSpec = "torch>=2.4,<2.11.0"
        $cudaVisionSpec = "torchvision>=0.19,<0.26.0"
        $cudaAudioSpec = "torchaudio>=2.4,<2.11.0"
    }
    if ($script:UnslothVerbose) {
        Fast-Install $cudaTorchSpec $cudaVisionSpec $cudaAudioSpec @cudaForce --index-url $TorchInstallIndexUrl
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install $cudaTorchSpec $cudaVisionSpec $cudaAudioSpec @cudaForce --index-url $TorchInstallIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-Host "[FAILED] PyTorch CUDA install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
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
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Yellow
    } else {
        substep "Triton for Windows installed (enables torch.compile)"
    }
}

# No unsloth.exe rename needed. setup.ps1 runs *via* unsloth.exe, so renaming the
# running launcher only ever failed (WinError 32) and printed a scary warning. It's
# also unnecessary: install.ps1 sets SKIP_STUDIO_BASE=1 (base never reinstalled) and
# 'studio update' goes through uv (--upgrade-package), whose pip fallback no-ops on
# the already-satisfied bare unsloth/unsloth-zoo. Either way unsloth.exe stays.

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

} else {
    step "python" "dependencies up to date"
    # Restore ErrorActionPreference (was lowered for pip/python section)
    $ErrorActionPreference = $prevEAP
}

# ── Pre-install transformers 5.x into .venv_t5_530/, .venv_t5_550/, and .venv_t5_510/ ──
# Runs outside the deps fast-path gate so that upgrades from the legacy
# single .venv_t5 are always migrated to the tiered layout.
# T5 sidecar venvs live under the resolved $StudioHome so custom installs are self-contained.
$VenvT5_530Dir = Join-Path $StudioHome ".venv_t5_530"
$VenvT5_550Dir = Join-Path $StudioHome ".venv_t5_550"
$VenvT5_510Dir = Join-Path $StudioHome ".venv_t5_510"
$VenvT5Legacy = Join-Path $StudioHome ".venv_t5"

function Test-TargetPackageVersion {
    param(
        [Parameter(Mandatory = $true)][string]$TargetDir,
        [Parameter(Mandatory = $true)][string]$PackageName,
        [Parameter(Mandatory = $true)][string]$ExpectedVersion
    )
    if (-not (Test-Path -LiteralPath $TargetDir -PathType Container)) { return $false }
    $packageNorm = $PackageName.Replace("-", "_")
    foreach ($pattern in @("$packageNorm-*.dist-info", "$PackageName-*.dist-info")) {
        foreach ($distInfo in @(Get-ChildItem -LiteralPath $TargetDir -Directory -Filter $pattern -ErrorAction SilentlyContinue)) {
            $metadata = Join-Path $distInfo.FullName "METADATA"
            if (-not (Test-Path -LiteralPath $metadata -PathType Leaf)) { continue }
            foreach ($line in (Get-Content -LiteralPath $metadata -ErrorAction SilentlyContinue)) {
                if ($line -eq "Version: $ExpectedVersion") { return $true }
            }
        }
    }
    return $false
}

$_NeedT5Install = $false
if (Test-Path -LiteralPath $VenvT5Legacy) {
    Assert-StudioOwnedOrAbsent -Path $VenvT5Legacy -Label "legacy transformers sidecar venv"
    Remove-Item -LiteralPath $VenvT5Legacy -Recurse -Force
    $_NeedT5Install = $true
}
if (-not (Test-Path -LiteralPath $VenvT5_530Dir)) { $_NeedT5Install = $true }
if (-not (Test-Path -LiteralPath $VenvT5_550Dir)) { $_NeedT5Install = $true }
if (-not (Test-Path -LiteralPath $VenvT5_510Dir)) { $_NeedT5Install = $true }
if (-not (Test-TargetPackageVersion -TargetDir $VenvT5_530Dir -PackageName "transformers" -ExpectedVersion "5.3.0")) { $_NeedT5Install = $true }
if (-not (Test-TargetPackageVersion -TargetDir $VenvT5_550Dir -PackageName "transformers" -ExpectedVersion "5.5.0")) { $_NeedT5Install = $true }
if (-not (Test-TargetPackageVersion -TargetDir $VenvT5_510Dir -PackageName "transformers" -ExpectedVersion "5.10.2")) { $_NeedT5Install = $true }
# Also reinstall when python deps were updated
if (-not $SkipPythonDeps) { $_NeedT5Install = $true }

if ($_NeedT5Install) {
Write-Host ""

$prevEAP_t5 = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# --- .venv_t5_530 (transformers 5.3.0) ---
substep "pre-installing transformers 5.3.0 for newer model support..."
Assert-StudioOwnedOrAbsent -Path $VenvT5_530Dir -Label "transformers 5.3 sidecar venv"
if (Test-Path -LiteralPath $VenvT5_530Dir) { Remove-Item -LiteralPath $VenvT5_530Dir -Recurse -Force }
[System.IO.Directory]::CreateDirectory($VenvT5_530Dir) | Out-Null
Mark-StudioOwned -Path $VenvT5_530Dir
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
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
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
Assert-StudioOwnedOrAbsent -Path $VenvT5_550Dir -Label "transformers 5.5 sidecar venv"
if (Test-Path -LiteralPath $VenvT5_550Dir) { Remove-Item -LiteralPath $VenvT5_550Dir -Recurse -Force }
[System.IO.Directory]::CreateDirectory($VenvT5_550Dir) | Out-Null
Mark-StudioOwned -Path $VenvT5_550Dir
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
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
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
step "transformers" "5.5.0 pre-installed"

# --- .venv_t5_510 (transformers 5.10.2) ---
substep "pre-installing transformers 5.10.2 for Gemma 4 Unified support..."
Assert-StudioOwnedOrAbsent -Path $VenvT5_510Dir -Label "transformers 5.10 sidecar venv"
if (Test-Path -LiteralPath $VenvT5_510Dir) { Remove-Item -LiteralPath $VenvT5_510Dir -Recurse -Force }
[System.IO.Directory]::CreateDirectory($VenvT5_510Dir) | Out-Null
Mark-StudioOwned -Path $VenvT5_510Dir
foreach ($pkg in @("transformers==5.10.2", "huggingface_hub==1.8.0", "hf_xet==1.4.2")) {
    if ($script:UnslothVerbose) {
        Fast-Install --target $VenvT5_510Dir --no-deps $pkg
        $t5PkgExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install --target $VenvT5_510Dir --no-deps $pkg | Out-String
        $t5PkgExit = $LASTEXITCODE
    }
    if ($t5PkgExit -ne 0) {
        Write-Host "[FAIL] Could not install $pkg into .venv_t5_510/" -ForegroundColor Red
        Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        exit 1
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_510Dir tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_510Dir tiktoken | Out-String
    $tiktokenInstallExit = $LASTEXITCODE
}
if ($tiktokenInstallExit -ne 0) {
    substep "Could not install tiktoken into .venv_t5_510/ -- Qwen tokenizers may fail" "Yellow"
}
$ErrorActionPreference = $prevEAP_t5
step "transformers" "5.10.2 pre-installed"

} # end $_NeedT5Install

# ==========================================================================
#  PHASE 3.4: Prefer prebuilt llama.cpp bundles before source build
# ==========================================================================
# Nest llama.cpp under $StudioHome only for real env-overrides, never the
# legacy default. Reuses $StudioHomeIsCustom from the canonical comparison
# computed above so the llama.cpp nest matches ownership-guard semantics.
if ($StudioHomeIsCustom) {
    $UnslothHome = $StudioHome
} else {
    $UnslothHome = Join-Path $env:USERPROFILE ".unsloth"
}
if (-not (Test-Path -LiteralPath $UnslothHome)) { [System.IO.Directory]::CreateDirectory($UnslothHome) | Out-Null }
$LlamaCppDir = Join-Path $UnslothHome "llama.cpp"
$NeedLlamaSourceBuild = $false
$SkipPrebuiltInstall = $false
$RequestedLlamaTag = if ($env:UNSLOTH_LLAMA_TAG) { $env:UNSLOTH_LLAMA_TAG } else { $DefaultLlamaTag }
# Every host installs the fork's app-* prebuilts now: GPU Windows (CUDA / ROCm)
# already did, and the fork now also ships the CPU bundles for Windows x64 and
# arm64 (windows-cpu / windows-arm64). ggml-org artifacts are no longer used by
# default. Mirrors setup.sh's routing.
$HelperReleaseRepo = "unslothai/llama.cpp"
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

$LocalLlamaCppLinked = $false
$LocalLlamaCppSrc = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR
if ($LocalLlamaCppSrc) {
    if (-not (Test-Path -LiteralPath $LocalLlamaCppSrc -PathType Container)) {
        step "llama.cpp" "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $LocalLlamaCppSrc" "Red"
        exit 1
    }
    $ResolvedLocal = (Resolve-Path -LiteralPath $LocalLlamaCppSrc).Path
    # Reusing a local dir disables both the prebuilt download and the source
    # build, so a runnable llama-server.exe must already be present. Accept any
    # layout LlamaCppBackend._layout_candidates() resolves (root-level, build\bin,
    # or build\bin\Release) so the flag never rejects a tree Studio could run.
    $LocalLlamaServerFound = $false
    foreach ($_cand in @(
            (Join-Path $ResolvedLocal "llama-server.exe"),
            (Join-Path $ResolvedLocal "build\bin\llama-server.exe"),
            (Join-Path $ResolvedLocal "build\bin\Release\llama-server.exe"))) {
        if (Test-Path -LiteralPath $_cand) { $LocalLlamaServerFound = $true; break }
    }
    if ($ResolvedLocal -eq $LlamaCppDir) {
        # Points at the canonical install location itself: never delete-then-link
        # onto itself. Reuse an existing build here (skip prebuilt + source) so the
        # staged prebuilt installer can't replace a build the user asked to reuse;
        # if nothing is built yet, fall through to the normal install.
        if ($LocalLlamaServerFound) {
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR is the canonical install location and already holds a build; reusing it" "Yellow"
            $LocalLlamaCppLinked = $true
            $NeedLlamaSourceBuild = $false
        } else {
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR points to the canonical install location with nothing built there yet; running the normal install" "Yellow"
        }
    } else {
        # Fail clearly rather than junction an unbuilt or wrong-platform checkout
        # and leave Studio with no usable binary.
        if (-not $LocalLlamaServerFound) {
            step "llama.cpp" "no llama-server.exe under $ResolvedLocal (looked for .\llama-server.exe, .\build\bin and .\build\bin\Release) -- build llama.cpp there first, or drop --with-llama-cpp-dir" "Red"
            exit 1
        }
        # If the target is already a junction/symlink (e.g. a previous
        # --with-llama-cpp-dir run), delete only the link via DirectoryInfo.Delete().
        # Remove-Item -Recurse -Force on a reparse point can traverse the link and
        # wipe the user's real llama.cpp directory on PowerShell 5.1. Dropping the
        # stale link here also keeps the custom-home ownership check below idempotent.
        # Use Get-Item -Force (not Test-Path): a *broken* junction whose target was
        # moved/deleted makes Test-Path return false, which would leave the dangling
        # link in place and make mklink below fail; Get-Item still resolves it so we
        # can remove it and relink to a new valid directory.
        $existing = Get-Item -LiteralPath $LlamaCppDir -Force -ErrorAction SilentlyContinue
        if ($existing -and ($existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            $existing.Delete()
        }
        if ($StudioHomeIsCustom) {
            Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"
        }
        if (Test-Path -LiteralPath $LlamaCppDir) {
            Remove-Item -Recurse -Force -LiteralPath $LlamaCppDir -ErrorAction SilentlyContinue
            # A locked/in-use tree can silently survive removal (SilentlyContinue
            # masks it). Don't then junction/copy over a half-present dir; mirror the
            # prebuilt path's active-process handling and stop with a clear message.
            if (Test-Path -LiteralPath $LlamaCppDir) {
                step "llama.cpp" "install blocked by active llama.cpp process" "Yellow"
                substep "Close Studio or other llama.cpp users and retry" "Yellow"
                exit 3
            }
        }
        cmd /c "mklink /J `"$LlamaCppDir`" `"$ResolvedLocal`"" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            substep "Could not create directory junction; copying instead..." "Yellow"
            Copy-Item -Recurse -LiteralPath $ResolvedLocal -Destination $LlamaCppDir
        }
        Write-Host ""
        step "llama.cpp" "linked local directory: $ResolvedLocal"
        $LocalLlamaCppLinked = $true
        $NeedLlamaSourceBuild = $false
    }
}

if ($LocalLlamaCppLinked) {
    # local directory linked above; skip prebuilt install
} elseif ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {
    Write-Host ""
    substep "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt llama.cpp install" "Yellow"
    $NeedLlamaSourceBuild = $true
} elseif ($SkipPrebuiltInstall) {
    Write-Host ""
    substep "Skipping prebuilt install -- falling back to source build" "Yellow"
} else {
    Write-Host ""
    if (Test-Path -LiteralPath $LlamaCppDir) {
        substep "Existing llama.cpp install detected -- validating staged prebuilt update before replacement"
        # If the existing install is the wrong kind (e.g. windows-cpu on a ROCm
        # machine that should have windows-rocm), remove it so the installer is
        # forced to download the correct variant rather than skipping on tag match.
        $existingMetaPath = Join-Path $LlamaCppDir "UNSLOTH_PREBUILT_INFO.json"
        if (Test-Path $existingMetaPath) {
            try {
                $existingMeta = Get-Content $existingMetaPath -Raw | ConvertFrom-Json
                $existingKind = $existingMeta.install_kind
                # A ROCm host may legitimately carry the fork's windows-rocm bundle
                # or the upstream windows-hip fallback, so accept either and never
                # treat a valid ROCm install as mismatched. A name-inferred gfx
                # arch (Adrenalin-only, no confirmed runtime) still counts as
                # ROCm-capable -- the ROCm prebuilt bundles its own runtime,
                # mirroring the --rocm-gfx forward below. The CPU branch covers both
                # the x64 windows-cpu and arm64 windows-arm64 bundles (Windows arm64
                # has no GPU prebuilt). NOTE: this block is currently inert --
                # write_prebuilt_metadata does not persist an install_kind key, so
                # $existingKind is always null; keep $expectedKinds in sync with the
                # kinds install_llama_prebuilt.py installs before relying on it.
                $expectedKinds = if ($HasROCm -or $script:ROCmGfxArch) { @("windows-rocm", "windows-hip") } elseif ($HasNvidiaSmi) { @("windows-cuda") } else { @("windows-cpu", "windows-arm64") }
                if ($existingKind -and ($existingKind -notin $expectedKinds)) {
                    substep "Removing mismatched llama.cpp install (found '$existingKind', need one of: $($expectedKinds -join ', '))..."
                    Remove-Item -Recurse -Force -LiteralPath $LlamaCppDir -ErrorAction SilentlyContinue
                }
            } catch {
                # unreadable metadata -- let the installer handle it
            }
        }
    }
    substep "installing prebuilt llama.cpp bundle (preferred path)..."
    # why: install_llama_prebuilt.py uses os.replace(), which would displace
    # an unrelated $env:UNSLOTH_STUDIO_HOME\llama.cpp before the source-build
    # ownership check below ever runs.
    if ($StudioHomeIsCustom) {
        Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"
    }
    $prebuiltArgs = @(
            "$PSScriptRoot\install_llama_prebuilt.py",
            "--install-dir", $LlamaCppDir,
            "--llama-tag", $RequestedLlamaTag,
            "--published-repo", $HelperReleaseRepo
        )
        if ($HasROCm) {
            $prebuiltArgs += "--has-rocm"
        }
        # Forward the resolved gfx arch so the per-gfx ROCm prebuilt is picked even
        # when the installer's probe can't confirm the runtime (amd-smi-only /
        # Adrenalin-only, name-inferred arch). --rocm-gfx is authoritative and
        # implies ROCm in install_llama_prebuilt.py, so the GPU prebuilt is selected
        # even with $HasROCm false. Gating on $HasROCm gave Strix Halo / 8060S CPU.
        if ($script:ROCmGfxArch) {
            $prebuiltArgs += @("--rocm-gfx", $script:ROCmGfxArch)
        }
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
            if ($StudioHomeIsCustom -and (Test-Path -LiteralPath $LlamaCppDir -PathType Container)) {
                Mark-StudioOwned -Path $LlamaCppDir
            }
            $installedRelease = Get-InstalledLlamaPrebuiltRelease -InstallDir $LlamaCppDir
            if ($installedRelease) {
                substep $installedRelease
            }
        } elseif ($prebuiltExit -eq 3) {
            step "llama.cpp" "install blocked by active llama.cpp process" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path -LiteralPath $LlamaCppDir) {
                substep "Existing install was restored" "Yellow"
            }
            substep "Close Studio or other llama.cpp users and retry" "Yellow"
            exit 3
        } else {
            step "llama.cpp" "prebuilt install failed (continuing)" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path -LiteralPath $LlamaCppDir) {
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
            winget install -e --id ShiningLight.OpenSSL.Dev --source winget --accept-package-agreements --accept-source-agreements
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
# Prerequisites git, cmake, VS Build Tools were installed in Phase 1; the CUDA
# Toolkit is resolved lazily just below via Resolve-CudaToolkit (source build only).
$OriginalLlamaCppDir = $LlamaCppDir
$BuildDir = Join-Path $LlamaCppDir "build"
$LlamaServerBin = Join-Path $BuildDir "bin\Release\llama-server.exe"

$HasCmakeForBuild = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)

# Check if existing llama-server matches current GPU mode. A CUDA-built binary
# on a now-CPU-only machine (or vice versa) needs to be rebuilt.
$NeedRebuild = $false
if (Test-Path -LiteralPath $LlamaServerBin) {
    $CmakeCacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path -LiteralPath $CmakeCacheFile) {
        $cachedCuda = Select-String -LiteralPath $CmakeCacheFile -Pattern 'GGML_CUDA:BOOL=ON' -Quiet
        if ($HasNvidiaSmi -and -not $cachedCuda) {
            Write-Host "   Existing llama-server is CPU-only but GPU is available -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        } elseif (-not $HasNvidiaSmi -and $cachedCuda) {
            Write-Host "   Existing llama-server was built with CUDA but no GPU detected -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        }
    }
}

# Install build tools now (last resort) rather than eagerly in Phase 1, so the
# prebuilt path stays fast. Same condition as the if/elseif chain below: a source
# build runs only when needed and no usable binary is already present. A linked
# local dir sets $NeedLlamaSourceBuild = $false, so this no-ops for that path.
$WillBuildLlamaFromSource = $NeedLlamaSourceBuild -and `
    -not ((Test-Path -LiteralPath $LlamaServerBin) -and -not $NeedRebuild -and $RequestedLlamaTag -ne "master")
if ($WillBuildLlamaFromSource) {
    Ensure-BuildToolsForLlamaSourceBuild
    # refresh so the chain below sees a newly installed cmake
    $HasCmakeForBuild = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
}

if ($LocalLlamaCppLinked) {
    # Local dir linked above -- honor the flag's contract: skip BOTH the prebuilt
    # download and the source build. Falling through here would run CMake inside
    # the user's checkout (via the junction) when it lacks build\bin\Release\llama-server.exe.
    Write-Host ""
    step "llama.cpp" "linked (skipping build)"
} elseif (-not $NeedLlamaSourceBuild) {
    Write-Host ""
    step "llama.cpp" "prebuilt (validated)"
} elseif ((Test-Path -LiteralPath $LlamaServerBin) -and -not $NeedRebuild -and $RequestedLlamaTag -ne "master") {
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
    # Finalize the VS generator (gate/fallback below) BEFORE Resolve-CudaToolkit,
    # which copies the CUDA .targets into the current generator's dir; a later swap
    # would strand them. The CMake 4.2 gate for VS 2026 is checked only here, in the
    # source-build path, so a VS 2026 + cmake < 4.2 host can still use the prebuilt. (#6473)
    if ($CmakeGenerator -match 'Visual Studio 18\b') {
        if (-not (Test-CmakeCanDriveGenerator -Generator $CmakeGenerator)) {
            $cmakeVerObj = Get-CmakeVersion
            $cmakeVerStr = if ($cmakeVerObj) { $cmakeVerObj.ToString() } else { '0.0' }
            substep "CMake $cmakeVerStr cannot drive the Visual Studio 2026 generator (need 4.2+ or a VS-bundled cmake) -- updating via winget..." "Yellow"
            if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
                # upgrade first (fast if Kitware.CMake is already a winget app), then
                # prepend the default dir so the new cmake wins over an older one on PATH
                try {
                    Invoke-SetupCommand { winget upgrade Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                    Refresh-Environment
                } catch { substep "CMake winget upgrade failed: $($_.Exception.Message)" "Yellow" }
                Add-DefaultCmakeToPath | Out-Null
                # upgrade no-ops if the cmake came from Scoop/Chocolatey/VS, not the
                # Kitware winget package; install it so a 4.2+ cmake is available
                if (-not (Test-CmakeCanDriveGenerator -Generator $CmakeGenerator)) {
                    try {
                        Invoke-SetupCommand { winget install Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                        Refresh-Environment
                    } catch { substep "CMake winget install failed: $($_.Exception.Message)" "Yellow" }
                    Add-DefaultCmakeToPath | Out-Null
                }
            }
            if (-not (Test-CmakeCanDriveGenerator -Generator $CmakeGenerator)) {
                # cmake still cannot drive VS 2026; before failing, fall back to an
                # older installed VS whose generator it can drive (e.g. VS 2022 + old
                # cmake on an offline box keeps building)
                $fallback = Get-FallbackVsGenerator
                if ($fallback) {
                    substep "CMake cannot drive $CmakeGenerator; falling back to $($fallback.Generator)" "Yellow"
                    $CmakeGenerator = $fallback.Generator
                    $VsInstallPath = $fallback.InstallPath
                } else {
                    Write-Host "[ERROR] CMake 4.2+ is required to build llama.cpp with the Visual Studio 2026 generator, and no older Visual Studio toolchain was found to fall back to." -ForegroundColor Red
                    Write-Host "        Upgrade CMake from https://cmake.org/download/ and re-run, or use a prebuilt llama.cpp bundle." -ForegroundColor Red
                    exit 1
                }
            }
        }
        substep "CMake can drive the $CmakeGenerator generator"
    }

    # CUDA resolved here (fail fast if none), after the final VS generator so its
    # .targets land in the toolset cmake actually uses.
    if ($HasNvidiaSmi) { Resolve-CudaToolkit -RequireOrExit }

    Write-Host ""
    if ($HasNvidiaSmi) {
        substep "building llama.cpp with CUDA support..."
    } elseif ($HasROCm -or $script:ROCmGfxArch) {
        # AMD GPU present but in the CPU-only source-build fallback: a HIP source
        # build needs the full HIP SDK + ROCm clang toolchain. AMD GPU acceleration
        # comes from the per-gfx ROCm prebuilt (bundles the runtime, no SDK) -- reaching
        # here means it couldn't be installed. Warn loudly, don't ship a slow CPU build.
        $_amdArch = if ($script:ROCmGfxArch) { $script:ROCmGfxArch } else { "ROCm" }
        substep "[WARN] AMD GPU ($_amdArch) detected, but the GPU-accelerated ROCm" "Yellow"
        substep "       llama.cpp prebuilt could not be installed -- falling back to a CPU build." "Yellow"
        substep "       The prebuilt is the AMD GPU path (no HIP SDK required). To restore GPU" "Yellow"
        substep "       acceleration: re-run the installer (check your network / proxy), or set" "Yellow"
        substep "       UNSLOTH_LLAMA_RELEASE_TAG to a tag with a gfx prebuilt for your GPU." "Yellow"
        substep "building llama.cpp (CPU-only fallback)..." "Yellow"
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

    if (Test-Path -LiteralPath (Join-Path $LlamaCppDir ".git")) {
        # why: in-place git mutation (remote set-url, checkout -B, clean -fdx)
        # rewrites $LlamaCppDir; mirror the prebuilt and temp-dir-swap guards
        # so an unrelated workspace .git tree is never silently overwritten.
        if ($StudioHomeIsCustom) {
            Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"
        }
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
        # why: in-place git-sync (the temp-dir clone path calls Mark-StudioOwned
        # at swap-time) must mark the existing tree so a subsequent prebuilt
        # update path's Assert-StudioOwnedOrAbsent does not exit on the same root.
        if ($BuildOk -and $StudioHomeIsCustom) {
            Mark-StudioOwned -Path $LlamaCppDir
        }
    } else {
        Write-Host "   Cloning llama.cpp @ $ResolvedSourceRef..." -ForegroundColor Gray
        $buildTmp = "$LlamaCppDir.build.$PID"
        $null = [System.IO.Directory]::CreateDirectory((Split-Path -LiteralPath $LlamaCppDir))
        if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
        if ($LlamaPr) {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$LlamaSource.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin "pull/$LlamaPr/head:pr-$LlamaPr" }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch PR #$LlamaPr"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout "pr-$LlamaPr" }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout PR #$LlamaPr"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "pull") {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$ResolvedSourceUrl.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin $ResolvedSourceRef }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch source PR ref"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout -B unsloth-llama-build FETCH_HEAD }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout source PR ref"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
                }
            }
        } elseif ($ResolvedSourceRefKind -eq "commit") {
            $cloneExit = Invoke-SetupCommand -AlwaysQuiet { git clone --depth 1 "$ResolvedSourceUrl.git" $buildTmp }
            if ($cloneExit -ne 0) {
                $BuildOk = $false
                $FailedStep = "git clone"
                if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
            }
            if ($BuildOk) {
                $fetchExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp fetch --depth 1 origin $ResolvedSourceRef }
                if ($fetchExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git fetch source commit"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
                }
            }
            if ($BuildOk) {
                $checkoutExit = Invoke-SetupCommand -AlwaysQuiet { git -C $buildTmp checkout -B unsloth-llama-build FETCH_HEAD }
                if ($checkoutExit -ne 0) {
                    $BuildOk = $false
                    $FailedStep = "git checkout source commit"
                    if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
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
                if (Test-Path -LiteralPath $buildTmp) { Remove-Item -LiteralPath $buildTmp -Recurse -Force }
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
            # UNSLOTH_LLAMA_CUDA_ARCHS (e.g. "120" or "89;86") forces the build
            # arch and wins over detection, matching setup.sh.
            $CudaArchOverride = if ($env:UNSLOTH_LLAMA_CUDA_ARCHS) { ($env:UNSLOTH_LLAMA_CUDA_ARCHS -replace '\s', '') } else { '' }
            if ((-not $CudaArch) -and (-not $CudaArchOverride)) {
                # No detectable compute capability (#5854): -DGGML_CUDA=ON with no
                # arch builds a PTX-only binary, so build CPU instead. Mirrors the
                # Linux fix; set UNSLOTH_LLAMA_CUDA_ARCHS=120 to force a CUDA build.
                substep "could not detect a CUDA compute capability; building CPU llama.cpp instead of a PTX-only binary (set UNSLOTH_LLAMA_CUDA_ARCHS=120 to force a CUDA build)." "Yellow"
                $CmakeArgs += '-DGGML_CUDA=OFF'
            } else {
                $CmakeArgs += '-DGGML_CUDA=ON'
                # Accept a host MSVC newer than nvcc's whitelist; a fresh toolkit
                # (e.g. CUDA 13.3) otherwise aborts with "#error -- unsupported
                # Microsoft Visual Studio version!". Mirrors the Linux fix. Via env
                # (covers the configure probe + build), after Refresh-Environment, idempotent.
                $nvccAllowFlag = '-allow-unsupported-compiler'
                if ([string]::IsNullOrEmpty($env:NVCC_PREPEND_FLAGS)) {
                    $env:NVCC_PREPEND_FLAGS = $nvccAllowFlag
                } elseif ($env:NVCC_PREPEND_FLAGS -notlike "*$nvccAllowFlag*") {
                    $env:NVCC_PREPEND_FLAGS = "$($env:NVCC_PREPEND_FLAGS) $nvccAllowFlag"
                }
                substep "NVCC_PREPEND_FLAGS = $env:NVCC_PREPEND_FLAGS"
                $CmakeArgs += "-DCUDAToolkit_ROOT=$CudaToolkitRoot"
                $CmakeArgs += "-DCUDA_TOOLKIT_ROOT_DIR=$CudaToolkitRoot"
                $CmakeArgs += "-DCMAKE_CUDA_COMPILER=$NvccPath"
                if ($CudaArchOverride) {
                    # Forced arch wins verbatim (no nvcc validation), matching setup.sh.
                    $CmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchOverride"
                } elseif ($CudaArch) {
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
                $hintCustomizations = if ($VsInstallPath) { Get-VcBuildCustomizationsDir -VsInstallPath $VsInstallPath -Generator $CmakeGenerator } else { "<VS_PATH>\MSBuild\Microsoft\VC\v170\BuildCustomizations" }
                Write-Host "     $hintCustomizations" -ForegroundColor Yellow
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

    # -- Step E: Build the DiffusionGemma visual server (optional, best-effort) --
    # An example target present on llama.cpp PR #24423; lets Studio serve
    # DiffusionGemma GGUFs without DG_VISUAL_BIN. No-op when not configured.
    if ($BuildOk) {
        $null = cmake --build $BuildDir --config Release --target llama-diffusion-gemma-visual-server -j $NumCpu 2>&1 | Out-String
    }

    # Swap temp build dir into final location (only if we built in a temp dir)
    if ($BuildOk -and $LlamaCppDir -ne $OriginalLlamaCppDir) {
        Assert-StudioOwnedOrAbsent -Path $OriginalLlamaCppDir -Label "llama.cpp install"
        if (Test-Path -LiteralPath $OriginalLlamaCppDir) { Remove-Item -LiteralPath $OriginalLlamaCppDir -Recurse -Force }
        Move-Item -LiteralPath $LlamaCppDir -Destination $OriginalLlamaCppDir
        $LlamaCppDir = $OriginalLlamaCppDir
        $BuildDir = Join-Path $LlamaCppDir "build"
        $LlamaServerBin = Join-Path $BuildDir "bin\Release\llama-server.exe"
        Mark-StudioOwned -Path $LlamaCppDir
    } elseif (-not $BuildOk -and $LlamaCppDir -ne $OriginalLlamaCppDir) {
        # Build failed -- clean up temp dir, preserve existing install
        if (Test-Path -LiteralPath $LlamaCppDir) { Remove-Item -LiteralPath $LlamaCppDir -Recurse -Force }
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
    if ($BuildOk -and (Test-Path -LiteralPath $LlamaServerBin)) {
        step "llama.cpp" "built"
        $QuantizeBin = Join-Path $BuildDir "bin\Release\llama-quantize.exe"
        if (Test-Path -LiteralPath $QuantizeBin) {
            step "llama-quantize" "built"
        }
        step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
    } else {
        $altBin = Join-Path $BuildDir "bin\llama-server.exe"
        if ($BuildOk -and (Test-Path -LiteralPath $altBin)) {
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
