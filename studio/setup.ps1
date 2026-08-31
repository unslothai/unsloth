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

# This script is spawned as powershell.exe -- Windows PowerShell 5.1 (see the PSModulePath note
# below) -- where the Invoke-WebRequest progress bar is redrawn on every read and sets the rate
# instead of the link: the VC++ runtime (24.4 MB, Ensure-VCRedist) took 38.18s with the bar on
# against 0.29s with it off on a windows-latest runner. -UseBasicParsing does not help; only this
# preference does. Script scope, in a separate short-lived process, so nothing outlives it.
$ProgressPreference = 'SilentlyContinue'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $ScriptDir

# `unsloth studio update` spawns powershell.exe, which is Windows PowerShell 5.1,
# and the child inherits the caller's PSModulePath. Launched from a PowerShell 7
# prompt that path leads with PowerShell 7's module directories, which ship their
# own Microsoft.PowerShell.Security. 5.1 finds that copy first and cannot load it:
#
#   The 'Get-ExecutionPolicy' command was found in the module
#   'Microsoft.PowerShell.Security', but the module could not be loaded.
#
# Any Security cmdlet reached during the run ends it there with exit 1 and no
# further output -- Get-AuthenticodeSignature, which verifies the VC++ runtime
# download, sits on this path. A try/catch does not help, because the failure is
# module loading in this process rather than an error the caller can catch.
#
# Prepended, not appended: the problem is precedence, not absence. Clearing the
# variable so 5.1 rebuilds its default does not help either, because the
# machine-level value on the windows-latest image also leads with PS7.
#
# PowerShell rewrites PSModulePath only for a direct pwsh -> powershell.exe hop,
# so any intermediate process defeats it (PowerShell/PowerShell#18681 is this
# exact chain through Python). install.ps1 carries the same block for the same
# reason; scripts/uninstall.ps1 needs none, as it loads no Security cmdlet.
#
# Not restored afterwards, deliberately. $env: is the process environment, so
# running this script in an interactive console leaves the reordering in place
# for that session. Narrowing the trigger to detect the broken chain would risk
# skipping the fix on a chain this list does not anticipate, and the cost of
# that is the install failing outright, against a session-lived module
# precedence change here. See the matching note in install.ps1.
if ($PSVersionTable.PSEdition -ne 'Core' -and $env:SystemRoot) {
    $_UnslothSystemModules = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules'
    if (Test-Path $_UnslothSystemModules) {
        $_UnslothKept = @(
            $env:PSModulePath -split ';' |
                Where-Object { $_ -and ($_ -ne $_UnslothSystemModules) }
        )
        $env:PSModulePath = (@($_UnslothSystemModules) + $_UnslothKept) -join ';'
    }
}

# UTF-8 output invariant. 5.1 encodes redirected output with the OEM code page,
# but the desktop app decodes the pipe as UTF-8 (from_utf8_lossy, install.rs):
# U+1F9A5 prints as '??', U+2500 becomes a bare 0xC4 and arrives as U+FFFD.
# Must precede the first write, since this rebuilds [Console]::Out. ASCII-only,
# because 5.1 parses these BOM-less files as ANSI.
$_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false
try {
    [Console]::OutputEncoding = $_UnslothUtf8NoBom
} catch {
    # No console (CREATE_NO_WINDOW). The setter P/Invokes SetConsoleOutputCP and
    # drops the cached writer BEFORE throwing, assigning OutputEncoding only
    # after, so [Console]::Out would rebuild on the OLD code page. Bind UTF-8
    # writers instead: redirected step/substep use Out as their only sink, and
    # Tauri decodes stderr the same way to build its failure text.
    try {
        $_UnslothStdout = New-Object System.IO.StreamWriter -ArgumentList ([Console]::OpenStandardOutput()), $_UnslothUtf8NoBom
        $_UnslothStdout.AutoFlush = $true
        [Console]::SetOut($_UnslothStdout)
        $_UnslothStderr = New-Object System.IO.StreamWriter -ArgumentList ([Console]::OpenStandardError()), $_UnslothUtf8NoBom
        $_UnslothStderr.AutoFlush = $true
        [Console]::SetError($_UnslothStderr)
    } catch { }
}
$OutputEncoding = $_UnslothUtf8NoBom
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Resolved once: it picks the output sink in step/substep and must not change
# mid-run. See Write-StudioStdoutMirror.
$script:StudioStdoutRedirected = $false
try { $script:StudioStdoutRedirected = [Console]::IsOutputRedirected } catch { }

# Every other line in this script reached the pipe through Write-Host, which
# 5.1's console host writes with its own writer on the OEM code page, not the
# UTF-8 [Console]::Out rebound above. Under CREATE_NO_WINDOW that writer is
# still what the desktop app reads, so the banner emoji, the U+2500 rule and
# every warning arrived as U+FFFD out of from_utf8_lossy. One sink instead: the
# console handle when redirected, Write-Host when interactive, since it is the
# only one that colorizes. Defined above the first write, for the same ordering
# reason as the encoding block.
function Write-StudioLine {
    param([string]$Message = "", [string]$ForegroundColor)
    if ($script:StudioStdoutRedirected) {
        try { [Console]::Out.WriteLine($Message); [Console]::Out.Flush() } catch {}
        return
    }
    if ($PSBoundParameters.ContainsKey('ForegroundColor')) {
        Write-Host $Message -ForegroundColor $ForegroundColor
    } else {
        Write-Host $Message
    }
}

# --------------------------------------------------------------------------
#  Maintainer-editable defaults
#  Change these in the GitHub-hosted script so users get updated defaults.
#  User env vars always override these baked-in values.
# --------------------------------------------------------------------------
# Prefer "latest" over "master" -- "master" bypasses the prebuilt resolver
# (no matching GitHub release), forces a source build, and causes HTTP 422
# errors. Only use "master" temporarily when the latest release is missing
# support for a new model architecture.
#
# UNSLOTH_LLAMA_CPP_BACKEND : "auto" (default), "cpu", "cuda", "vulkan",
# "hip", or "rocm". Concrete values select and persist a backend across updates;
# "auto" restores detection. Overrides Unsloth's Settings > System selection.
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

function Exit-SetupFailure {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$Code = 1
    )
    if ($Code -eq 0) { $Code = 1 }
    if ((@("1", "true") -contains $env:UNSLOTH_TAURI_MODE) -or
        (@("1", "true") -contains $env:UNSLOTH_TAURI_UPDATE)) {
        $singleLine = ($Message -replace '[\r\n]+', ' ').Trim()
        [Console]::Out.WriteLine("[TAURI:ERROR] $singleLine")
        [Console]::Out.Flush()
    }
    exit $Code
}

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
            # PSModulePath joins Path as an exception. Reloading it from the
            # registry would undo the normalization at the top of this file,
            # and several callers of this function run before the uv installer
            # (which loads Microsoft.PowerShell.Security), so the module path
            # would be broken again exactly where it has to be right.
            if ($key -eq 'Path' -or $key -eq 'PSModulePath') { continue }
            # Same exception, for the UTF-8 invariant at the top. This runs
            # repeatedly, so a registry PYTHONUTF8=0 would otherwise reload over
            # ours and every later Python child would go back to mojibake.
            if ($key -eq 'PYTHONUTF8' -or $key -eq 'PYTHONIOENCODING') { continue }
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
    if (Get-Variable -Name StageRoot -ValueOnly -ErrorAction SilentlyContinue) { return $false }
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
                Write-StudioLine "[WARN] User PATH is empty - initializing with $Directory" -ForegroundColor Yellow
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
        Write-StudioLine "[WARN] Could not update User PATH: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

# PowerShell 5.1 compatibility helper: avoid relying on New-TemporaryFile.
function New-UnslothTemporaryFile {
    $tempPath = [System.IO.Path]::GetTempFileName()
    return Get-Item -LiteralPath $tempPath
}

function Remove-AgentInstructionFiles {
    param([string[]]$Roots)

    foreach ($root in $Roots) {
        if (-not $root) { continue }
        $item = Get-Item -LiteralPath $root -Force -ErrorAction SilentlyContinue
        if (-not $item -or -not $item.PSIsContainer) { continue }
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
        $pending = New-Object System.Collections.Stack
        $pending.Push($item)
        while ($pending.Count -gt 0) {
            $current = $pending.Pop()
            foreach ($child in @(Get-ChildItem -LiteralPath $current.FullName -Force -ErrorAction SilentlyContinue)) {
                if ($child.PSIsContainer) {
                    if (-not ($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                        $pending.Push($child)
                    }
                } elseif ($child.Name -in @("AGENTS.md", "CLAUDE.md")) {
                    Remove-Item -LiteralPath $child.FullName -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
}

# Recognize ERROR_ACCESS_DENIED through PowerShell's wrapper exceptions.
function Test-AccessDeniedError {
    param($ErrorRecord)

    $ex = if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) { $ErrorRecord.Exception } else { $ErrorRecord }
    while ($ex) {
        if ($ex -is [System.UnauthorizedAccessException]) { return $true }
        # IOException uses HRESULT; Win32Exception uses NativeErrorCode.
        if ($ex.HResult -eq -2147024891) { return $true }
        if ($ex -is [System.ComponentModel.Win32Exception] -and $ex.NativeErrorCode -eq 5) { return $true }
        $ex = $ex.InnerException
    }
    if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) {
        return ($ErrorRecord.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::PermissionDenied)
    }
    return $false
}

# Keep denied ACLs distinct from absent paths instead of letting Test-Path throw.
function Get-PathState {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path,
        [ValidateSet("Any", "Leaf", "Container")][string]$PathType = "Any"
    )

    if ([string]::IsNullOrWhiteSpace($Path)) { return "Absent" }
    try {
        if (Test-Path -LiteralPath $Path -PathType $PathType -ErrorAction Stop) { return "Present" }
        return "Absent"
    } catch {
        if (Test-AccessDeniedError $_) { return "Denied" }
        # Malformed path, offline drive, dangling link: nothing usable there.
        return "Absent"
    }
}

# Non-throwing Test-Path for paths inside install trees we do not control.
# Callers that must react to a denial use Get-PathState instead.
function Test-PathQuiet {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path,
        [ValidateSet("Any", "Leaf", "Container")][string]$PathType = "Any"
    )

    return ((Get-PathState -Path $Path -PathType $PathType) -eq "Present")
}

# install.ps1 carries a verbatim copy because it cannot dot-source this file;
# test_denied_llama_cpp_preflight.py enforces parity.
# Dir, marker and listing: no single probe separates readable/absent/denied.
function Get-LlamaCppInstallReadState {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

    $dirState = Get-PathState -Path $Path -PathType Container
    if ($dirState -eq "Denied") { return "Denied" }
    if ($dirState -ne "Present") { return "Absent" }
    switch (Get-PathState -Path (Join-Path $Path "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf) {
        "Denied"  { return "Denied" }
    }
    try { $null = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1) }
    catch {
        if (Test-AccessDeniedError $_) { return "Denied" }
        # Nonfatal here: nothing has been installed yet.
    }
    return "Readable"
}

# Describe a denied link target without risking another reporting failure.
function Get-PathDenialDetail {
    param([Parameter(Mandatory = $true)][AllowNull()][AllowEmptyString()][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if (-not $item) { return "" }
    # Non-filesystem providers do not expose FileSystemInfo attributes.
    if ($item -isnot [System.IO.FileSystemInfo]) { return "" }
    if (-not ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { return "" }
    $target = $null
    try { $target = $item.Target } catch { $target = $null }
    # PS 5.1 exposes .Target as a collection; PS 7 as a string.
    if ($target) { return " (it is a link to $(@($target) -join ', '))" }
    return " (it is a link)"
}

# Print guidance; returns the failure reason as its only pipeline output.
function Write-PathAccessDenied {
    param(
        [Parameter(Mandatory = $true)][AllowNull()][AllowEmptyString()][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Label,
        # Never tell users to delete a build they supplied.
        [switch]$UserSupplied,
        # Unreadable custom homes cannot be identified as managed caches.
        [switch]$OwnershipUnverified
    )

    step "permissions" "$Label at $Path cannot be read: access is denied$(Get-PathDenialDetail -Path $Path)" "Red"
    if ($UserSupplied) {
        substep "Unsloth will not touch a directory you pointed it at, so this has to be fixed at the source" "Yellow"
        substep "Restore access with these two in an elevated PowerShell, or point UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build:" "Yellow"
    } elseif ($OwnershipUnverified) {
        substep "Unsloth cannot confirm this folder is its own install while it is unreadable, so it will not tell you to remove it" "Yellow"
        substep "Restore access with these two in an elevated PowerShell, or move the folder aside and re-run setup:" "Yellow"
    } else {
        substep "This folder lives outside the app, so reinstalling Unsloth Studio, to any drive, reuses it and fails the same way" "Yellow"
        substep "Simplest fix: close Unsloth, delete or rename $Path, then re-run setup (it is a managed cache and gets reinstalled)" "Yellow"
        substep "If deleting is also denied, run these two in an elevated PowerShell, then re-run setup:" "Yellow"
    }
    substep "takeown /F `"$Path`" /R /D Y" "Yellow"
    substep "icacls `"$Path`" /reset /T" "Yellow"
    substep "Antivirus or Controlled folder access can deny this path too; allow or exclude it, then retry" "Yellow"
    if ($UserSupplied) {
        return "Access denied reading $Label at $Path. Restore access with takeown/icacls, or point UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build, then re-run setup."
    }
    if ($OwnershipUnverified) {
        return "Access denied reading $Label at $Path. Unsloth cannot confirm that folder is its own install while it is unreadable: restore access with takeown/icacls, or move it aside, then re-run setup."
    }
    return "Access denied reading the existing $Label at $Path. Delete or rename that folder (Unsloth reinstalls it) or restore access with takeown/icacls, then re-run setup. Reinstalling the app does not reset it."
}

# Canonicalize a directory for comparison; unresolvable ones are only normalized.
function Get-CanonicalDir {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

    $trimmedPath = $Path.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmedPath)) { return $trimmedPath }
    $resolvedPath = $null
    if ((Get-PathState -Path $trimmedPath -PathType Container) -eq "Present") {
        try { $resolvedPath = (Resolve-Path -LiteralPath $trimmedPath).Path } catch {}
    }
    if (-not $resolvedPath) {
        try {
            $resolvedPath =
                $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($trimmedPath)
        } catch {
            $resolvedPath = [System.Environment]::ExpandEnvironmentVariables($trimmedPath)
        }
        try { $resolvedPath = [System.IO.Path]::GetFullPath($resolvedPath) } catch {}
    }
    # Resolve-Path keeps a trailing separator, so trim after both branches (never a root).
    try {
        $root = [System.IO.Path]::GetPathRoot($resolvedPath)
        if ($root -and $resolvedPath.Length -gt $root.Length) { return $resolvedPath.TrimEnd('\', '/') }
    } catch {}
    return $resolvedPath
}

# Compare canonical homes so path spelling does not change ownership policy.
function Test-StudioHomeIsCustom {
    return ((Get-CanonicalDir -Path $StudioHome) -ne
        (Get-CanonicalDir -Path (Join-Path $env:USERPROFILE ".unsloth\studio")))
}

# Is this bin\unsloth.cmd one install.ps1 wrote? Only then does it prove the root is
# ours. The name alone is a plausible wrapper for anything built on unsloth, and the
# caller uses the answer to decide whether a user-chosen venv may be deleted. The
# trampoline is the marker; no other file carries it. Mirrored in install.ps1
# (Test-UnslothCmdShimFile) and scripts/uninstall.ps1 (_IsUnslothCmdShim).
function Test-UnslothCmdShimFile {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        if ((Get-Item -LiteralPath $Path -ErrorAction Stop).Length -gt 8192) { return $false }
        $text = [System.IO.File]::ReadAllText($Path)
    } catch {
        # Unreadable proves nothing, and "proves nothing" must not mean "deletable".
        return $false
    }
    return ($text -like "*unsloth-studio-managed-launcher*" -and $text -like "*from unsloth_cli import app*")
}

# Shared default cache, or the custom Unsloth home's llama.cpp tree.
function Get-ManagedLlamaCppDir {
    if ($StageRoot) {
        return (Join-Path $StageRoot "llama.cpp")
    }
    if (-not (Test-StudioHomeIsCustom)) {
        return (Join-Path $env:USERPROFILE ".unsloth\llama.cpp")
    }
    return (Join-Path (Get-CanonicalDir -Path $StudioHome) "llama.cpp")
}

# Failure reason when the managed tree is denied; never touches its ACLs.
function Invoke-ManagedLlamaCppPreflight {
    # Let the existing profile validation handle a missing USERPROFILE later.
    if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }
    $dir = Get-ManagedLlamaCppDir
    if ((Get-LlamaCppInstallReadState -Path $dir) -ne "Denied") { return $null }
    Write-StudioLine ""
    # A denied custom home cannot be claimed as an Unsloth-managed cache.
    $homeIsCustom = Test-StudioHomeIsCustom
    # Preserve user-supplied wording when either override names this tree.
    $suppliedDir = if ($WithLlamaCppDir) { $WithLlamaCppDir } else { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR }
    $userSupplied = (-not [string]::IsNullOrWhiteSpace($suppliedDir)) -and
        ((Get-CanonicalDir -Path $suppliedDir) -eq (Get-CanonicalDir -Path $dir))
    $reason = Write-PathAccessDenied -Path $dir -Label "llama.cpp install" `
        -UserSupplied:$userSupplied -OwnershipUnverified:$homeIsCustom
    substep "Stopping here, before phase 1: nothing has been downloaded or installed" "Yellow"
    substep "Fix access, then run the same install, setup, or update command again" "Yellow"
    Write-StudioLine ""
    return "$reason Nothing was installed."
}

# Stop every install path consistently when its destination is unreadable.
function Exit-PathAccessDenied {
    param(
        [Parameter(Mandatory = $true)][AllowNull()][AllowEmptyString()][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Label,
        [switch]$UserSupplied,
        [switch]$OwnershipUnverified
    )

    Exit-SetupFailure (Write-PathAccessDenied -Path $Path -Label $Label `
        -UserSupplied:$UserSupplied -OwnershipUnverified:$OwnershipUnverified)
}

function Get-InstalledLlamaPrebuiltRelease {
    param([string]$InstallDir)

    $metadataPath = Join-Path $InstallDir "UNSLOTH_PREBUILT_INFO.json"
    if (-not (Test-PathQuiet $metadataPath)) {
        return $null
    }

    try {
        $payload = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
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
    substep "Or let Unsloth use the prebuilt CUDA bundle; it does not need the local toolkit." $Color
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

# Reserved for the OS, and budgeted per compile job, both in MB. The budget sits
# above the measured per-translation-unit peak on purpose; see the derivation on
# _LLAMA_BUILD_* in setup.sh, which these must match and which the shell test
# pins. MSVC and Windows, where the freeze was reported, need that headroom more.
$LlamaBuildReserveMb = 2048
$LlamaBuildMbPerJob = 2048

# The cmake -j count. A negative $TotalMb means RAM was unreadable and keeps the
# old core-count behaviour. Zero is a reading, not a failure: a box with no
# memory left is the last one that should get its full core count, so it falls
# through and floors at 1, as _llama_jobs_for does for a numeric 0.
# UNSLOTH_LLAMA_BUILD_JOBS wins. Pure, so the tests can drive it.
function Get-LlamaJobsFor {
    param([int]$Cores, [long]$TotalMb)

    $override = 0
    if ($env:UNSLOTH_LLAMA_BUILD_JOBS -and
        [int]::TryParse($env:UNSLOTH_LLAMA_BUILD_JOBS.Trim(), [ref]$override) -and $override -ge 1) {
        return $override
    }
    if ($Cores -lt 1) { $Cores = 4 }
    if ($TotalMb -lt 0) { return $Cores }
    $jobs = [int][Math]::Floor(($TotalMb - $LlamaBuildReserveMb) / $LlamaBuildMbPerJob)
    if ($jobs -lt 1) { $jobs = 1 }
    if ($jobs -gt $Cores) { $jobs = $Cores }
    return $jobs
}

# Usable RAM in MB; -1 when it cannot be read at all. Available, not installed:
# a box with 8 GB already resident cannot host a 14 GB compile just because
# 16 GB is fitted. AvailableMBytes counts the standby list, which the Free
# counters do not, and the raw perf class is not localized the way Get-Counter
# paths are; installed RAM stays the fallback. A reading of 0 is returned as 0
# rather than treated as unreadable, since falling back to installed RAM under
# real pressure would hand the machine its full core count at the worst moment.
function Get-UsableMemoryMb {
    try {
        $avail = (Get-CimInstance Win32_PerfRawData_PerfOS_Memory -ErrorAction Stop).AvailableMBytes
        if ($null -ne $avail) { return [long]$avail }
    } catch { }
    try {
        $bytes = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
        if ($bytes -gt 0) { return [long]($bytes / 1MB) }
    } catch { }
    try {
        # TotalVisibleMemorySize is in KB and excludes hardware-reserved RAM.
        $kb = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).TotalVisibleMemorySize
        if ($kb -gt 0) { return [long]($kb / 1KB) }
    } catch { }
    return -1
}

function Get-LlamaBuildJobs {
    return (Get-LlamaJobsFor -Cores ([Environment]::ProcessorCount) -TotalMb (Get-UsableMemoryMb))
}

# Classify the physical NVIDIA inventory for a cu126 fallback: "cu126" when it covers
# every GPU, "uncovered" for an incompatible mix, empty when no fallback is needed or the
# inventory is unreadable. CUDA_VISIBLE_DEVICES is ignored because the wheel must support
# the host. Mirrors _nvidia_cu126_verdict in install.sh.
function Get-NvidiaCu126Verdict {
    # Floor is per-release, not fixed: only 2.11 dropped sm_70 from cu128.
    param([string]$SmiExe, [int]$LegacyFloorSm = 75)
    if (-not $SmiExe) { return '' }
    $raw = Invoke-NvidiaSmiBounded $SmiExe @('--query-gpu=compute_cap', '--format=csv,noheader,nounits') -StdoutOnly
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return '' }
    $legacy = $false
    $outsideCu126 = $false
    $seen = $false
    foreach ($line in ($raw -split "`n")) {
        $value = $line.Trim()
        if (-not $value) { continue }
        if ($value -notmatch '^(\d+)\.(\d+)$') { return '' }
        $sm = ([int]$Matches[1] * 10) + [int]$Matches[2]
        if ($sm -lt $LegacyFloorSm) { $legacy = $true }
        if ($sm -lt 50 -or $sm -gt 90) { $outsideCu126 = $true }
        $seen = $true
    }
    if (-not $seen -or -not $legacy) { return '' }
    if ($outsideCu126) { return 'uncovered' }
    return 'cu126'
}

function Get-CudaFamilyCappedForPreTuring {
    param([string]$Family, [string]$SmiExe)
    if ($Family -notin @('cu128', 'cu130')) { return $Family }
    # Windows pins torch<2.11, whose cu128 still ships sm_70, so only cu130
    # strands a Volta here. Raise to 75 when that pin reaches 2.11.
    $legacyFloorSm = if ($Family -eq 'cu128') { 70 } else { 75 }
    $verdict = Get-NvidiaCu126Verdict $SmiExe $legacyFloorSm
    if (-not $verdict) { return $Family }
    # This runs twice per setup; announce once without polluting pipeline output.
    $announce = -not $script:PreTuringCapAnnounced
    $script:PreTuringCapAnnounced = $true
    if ($verdict -eq 'cu126') {
        if ($announce) {
            substep "pre-Turing NVIDIA GPUs (sm_<75) are present -- selecting cu126, because PyTorch 2.11's $Family wheels start at sm_75" "Yellow"
        }
        return 'cu126'
    }
    if ($announce) {
        substep "this host mixes pre-Turing NVIDIA GPUs with GPUs that cu126 cannot serve; no PyTorch 2.11 CUDA family covers both" "Yellow"
        substep "keeping $Family, so the pre-Turing GPUs will be unusable; set UNSLOTH_TORCH_INDEX_FAMILY=cu126 to choose the other way" "Yellow"
    }
    return $Family
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
            if ($major -ge 13)                        { $family = "cu130" }
            elseif ($major -eq 12 -and $minor -ge 8)  { $family = "cu128" }
            elseif ($major -eq 12 -and $minor -ge 6)  { $family = "cu126" }
            elseif ($major -ge 12) { $family = "cu124" }
            elseif ($major -ge 11) { $family = "cu118" }
            else { return "cpu" }
            return (Get-CudaFamilyCappedForPreTuring $family $smiExe)
        }
    } catch { }

    return "cu126"
}

# Trim trailing slashes from the URL PATH only, preserving ?query / #fragment: a whole-URL
# TrimEnd corrupts a token ending in "/", a single strip leaves .../cu128// empty. Shared.
function Trim-IndexPathSlashes {
    param([string]$Url)
    $value = $Url.Trim()
    $idx = $value.IndexOfAny([char[]]@('?', '#'))
    if ($idx -lt 0) {
        return $value.TrimEnd('/')
    }
    return $value.Substring(0, $idx).TrimEnd('/') + $value.Substring($idx)
}

# Explicit torch-index pin (UNSLOTH_TORCH_INDEX_URL / _FAMILY), shared by the stale-venv check
# and install selection so a pinned index wins over GPU probing (parity with the other
# installers). URL is verbatim; _FAMILY is the leaf joined to the mirror base.
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

# Last path segment of a wheel index URL, query/fragment dropped first so a token-authenticated
# pin (.../cu128?token=x) classifies as cu128 (else it reinstalls every update). Classification
# only. Shared with the py / install.sh leaf extractors.
function Get-TorchIndexLeaf {
    param([string]$Url)
    if ([string]::IsNullOrWhiteSpace($Url)) { return $null }
    $path = ($Url -split '[?#]', 2)[0]
    if ([string]::IsNullOrWhiteSpace($path)) { return $null }
    return ($path.TrimEnd('/') -split '/')[-1].ToLowerInvariant()
}

# Redact index-URL credentials (userinfo + ?query= + #fragment) from captured installer
# output before printing on failure; uv/pip errors echo the failing --index-url verbatim.
# Mirrors the other installers. Verbose mode streams uncaptured, so it isn't redacted.
function Redact-InstallOutput {
    param([string]$Text)
    if (-not $Text) { return $Text }
    $Text = $Text -replace '(https?://)[^/@\s`]+@', '$1<redacted>@'
    $Text = $Text -replace '([?&][^=\s&`]+)=[^&#\s`]+', '$1=<redacted>'
    # A #token=... fragment is as sensitive as a query; URL-anchored.
    return $Text -replace '(https?://[^\s`#]+)#[^\s`]+', '$1#<redacted>'
}

# AMD per-arch leaves needing the torch 2.11 floor (the _grouped_mm <2.11 bug). MUST match
# the install-spec path below and the other installers; other leaves ship <2.11 and stay default.
function Test-RocmGfx211Leaf {
    param([string]$Leaf)
    return @('gfx120x-all', 'gfx1151', 'gfx1150', 'gfx1152') -contains $Leaf
}

# rocmX.Y versions KNOWN to ship torch 2.11: rocm7.2 only today. Do NOT floor an unknown newer
# rocm speculatively. MUST match _ROCM_KNOWN_TORCH211_VERSIONS and the rocm7.2 leaf elsewhere.
function Test-RocmKnown211Version {
    param([int]$Major, [int]$Minor)
    return ($Major -eq 7 -and $Minor -eq 2)
}

# True only for a real CUDA family leaf: "cu" + digits (cu118, cu128, ...). A bare -like 'cu*'
# would match "custom"/"current" and rebuild the venv every run. Mirrors _is_cuda_family_leaf.
function Test-CudaFamilyLeaf {
    param([string]$Leaf)
    if ([string]::IsNullOrWhiteSpace($Leaf)) { return $false }
    # EXACT cu+digits: cu128-private routes through the unknown-leaf path instead.
    return $Leaf -match '^cu[0-9]+$'
}

# True only for a real pip ROCm family leaf: EXACT rocm<digits>[.<digits>] or a gfx leaf. A leaf
# that merely STARTS with rocm (rocm-rel-7.2.1, rocm7.2-private) is a custom pin the verbatim
# path owns, so anchor the match. Mirrors _is_pip_rocm_family_leaf / install.sh.
function Test-PipRocmFamilyLeaf {
    param([string]$Leaf)
    if ([string]::IsNullOrWhiteSpace($Leaf)) { return $false }
    # gfx must be followed by a digit (an architecture leaf); gfx-private is custom.
    return ($Leaf -match '^gfx[0-9]') -or ($Leaf -match '^rocm[0-9]+(\.[0-9]+)?$')
}

# Stale-venv ROCm comparison for a pinned gfx*/rocm* index. Returns @{ Expected; Installed } so
# the caller rebuilds when they differ. Mirrors _rocm_pin_family_mismatch (same rocmX.Y / gfx
# cases). An untagged (no +rocm) wheel never satisfies a ROCm pin -> stale.
function Get-RocmPinStaleTags {
    param([string]$PinLeaf, [string]$TorchVersion)
    $_pinRocm = [regex]::Match($PinLeaf, '^rocm(\d+)\.(\d+)')
    $_pinVer = if ($_pinRocm.Success) { "$($_pinRocm.Groups[1].Value).$($_pinRocm.Groups[2].Value)" } else { $null }
    # The family classifier accepts a major-only rocm<d> leaf too (rocm7).
    $_pinMajorOnly = [regex]::Match($PinLeaf, '^rocm(\d+)$')
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

    # Major-only rocm pin (rocm7): compare majors only -- a +rocm6.4 wheel under a rocm7
    # pin is stale, any +rocm7.x wheel satisfies it (no pinned minor to compare, and the
    # 2.11-line fallback below would invert both verdicts). Mirrors _rocm_pin_family_mismatch.
    if ($_pinMajorOnly.Success) {
        $_pinMaj = [int]$_pinMajorOnly.Groups[1].Value
        if ($_instVer) {
            $_instMaj = [int]$_instRocm.Groups[1].Value
            $expected = if ($_instMaj -eq $_pinMaj) { "rocm$_instVer" } else { "rocm$_pinMaj.x" }
            return @{ Expected = $expected; Installed = "rocm$_instVer" }
        }
        # Untagged wheel never satisfies a ROCm pin; a +rocm tag with an unreadable
        # version is accepted (matches the lenient unreadable fallback below).
        $installed = if ($_instHasRocm) { "rocm" } else { "not-rocm" }
        return @{ Expected = "rocm"; Installed = $installed }
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

# Bounded "ask python a question" probe, shared by every torch probe below: a wedged torch import
# or a hanging Intel driver init would block a bare `& python -c ...` forever. ProcessStartInfo,
# not &, so stderr cannot trip $ErrorActionPreference; BOTH streams drain async so a noisy import
# cannot deadlock on a full pipe; WaitForExit bounds the wait and kills the child. Every failure
# reads as .Ok = $false, and .Error carries WHICH failure: stderr used to be drained and thrown
# away, so "the HIP DLLs will not load", "torch is not installed" and "the import never came back"
# all reached the caller as one silent False -- and one such caller deletes the environment over
# that answer. Mirrors install.ps1's copy.
function Invoke-BoundedPythonProbe {
    param([string]$PythonExe, [string]$Code, [int]$TimeoutSec = 30)
    # TimedOut separates "never answered" from "answered with a failure": both leave Ok
    # false, only the second says anything about the installation.
    $result = [pscustomobject]@{ Ok = $false; Output = ""; Error = ""; TimedOut = $false }
    if (-not $PythonExe -or -not $Code) { return $result }
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $PythonExe
        $psi.Arguments = "-c `"$Code`""
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $errTask = $proc.StandardError.ReadToEndAsync()
        if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
            try { $proc.Kill() } catch {}
            # Synthesised, not read back: waiting on the reader tasks of a wedged child would
            # reintroduce the hang this helper exists to bound.
            $result.Error = "python did not answer within $TimeoutSec seconds"
            $result.TimedOut = $true
            return $result
        }
        $result.Output = $outTask.GetAwaiter().GetResult()
        # Kept, not discarded: the only place a failed probe's OSError / WinError text exists, and
        # the caller decides what to do with the venv based on it.
        $result.Error = $errTask.GetAwaiter().GetResult()
        $result.Ok = ($proc.ExitCode -eq 0)
        return $result
    } catch {
        $result.Error = $_.Exception.Message
        return $result
    }
}

# True when $PythonExe's torch can actually drive an Intel GPU. Quiet on purpose: the three
# callers read a False differently. Only an XPU build can answer True, so a CPU build never
# vetoes a migration.
function Test-TorchXpuAvailable {
    param([string]$PythonExe)
    if (-not $PythonExe) { return $false }
    # Line-anchored so a stdout banner cannot hide the answer; a timeout reads as "no XPU".
    $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.xpu.is_available())'
    return ($probe.Ok -and $probe.Output -match '(?m)^\s*True\s*$')
}

# Post-install XPU runtime check. A WMI name match says the part is XPU-capable, not that the
# compute runtime works: on an old Intel driver the wheel installs fine, never initializes, and
# unsloth/device_type.py raises NotImplementedError at import -- a hard crash, not a chat-only
# downgrade. Warn only: the wheel is correct and a driver update fixes it.
function Assert-XpuRuntimeReady {
    param([string]$PythonExe)
    if (Test-TorchXpuAvailable -PythonExe $PythonExe) { return $true }
    substep "[WARN] PyTorch XPU is installed but torch.xpu.is_available() is False." "Yellow"
    substep "       The Intel GPU driver is most likely too old -- PyTorch XPU on Windows" "Yellow"
    substep "       needs Intel Graphics Driver 32.0.101.6739 (WHQL) or newer." "Yellow"
    substep "       Update the driver, then re-run. See:" "Yellow"
    substep "       https://unsloth.ai/docs/get-started/install/intel" "Yellow"
    return $false
}

# Bounded Win32_VideoController scan: the query can block forever on a degraded WMI repository,
# -ErrorAction only suppresses reported errors, and -OperationTimeoutSec is not enforced for the
# local COM session this uses, so out of process with a wall-clock kill is the only bound that
# holds. Ok = $false on an empty answer too, since a Windows host always has an adapter.
# Mirrors install.ps1's copy.
function Invoke-BoundedVideoControllerScan {
    param([int]$TimeoutSec = 15)
    $result = [pscustomobject]@{ Ok = $false; Names = @() }
    $job = $null
    try {
        $job = Start-Job -ScriptBlock {
            Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Name
        }
        if (Wait-Job -Job $job -Timeout $TimeoutSec) {
            $names = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
            $result.Names = @($names | Where-Object { $_ })
            $result.Ok = ($result.Names.Count -gt 0)
        } else {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
        }
    } catch {
    } finally {
        if ($job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue }
    }
    return $result
}

# Registry fallback for the scan above, mirroring install_llama_prebuilt.py's
# windows_intel_gpu_in_registry(): the display-adapter class key, one NNNN subkey per driver
# config. Weaker than WMI (a config can outlive removed hardware), so it is the fallback here
# while that function is registry-first -- there a false positive only picks a different
# llama.cpp bundle, here it would install XPU torch on a host with no Arc. Mirrors install.ps1.
function Get-IntelRegistryAdapterNames {
    $names = @()
    $classKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try {
        $subs = @(Get-ChildItem -LiteralPath $classKey -ErrorAction SilentlyContinue)
    } catch { return @() }
    foreach ($sub in $subs) {
        # Guarded per subkey, not around the loop: one unreadable entry must not discard the
        # adapters found after it. Matches windows_intel_gpu_in_registry()'s per-key skip.
        try {
            # Numeric subkeys only: "Properties" is ACL-restricted and not an adapter.
            if ("$($sub.PSChildName)" -match '^\d+$') {
                $props = Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue
                if ($props) {
                    $desc = "$($props.DriverDesc)"
                    # Callers re-filter on "Intel", so a vendor-ID hit with a localized or
                    # OEM-branded DriverDesc would be found here and dropped there. Tag it
                    # instead; still only XPU-capable if the name says Arc / Data Center GPU.
                    if ("$($props.MatchingDeviceId)" -match '(?i)ven_8086') {
                        $names += if ($desc -match '(?i)intel') { $desc }
                                  elseif ($desc) { "Intel $desc" }
                                  else { "Intel Graphics" }
                    } elseif ($desc -match '(?i)intel') {
                        $names += $desc
                    }
                }
            }
        } catch { }
    }
    return $names
}

# The studio venv directory, guessed from the environment alone. The canonical $VenvDir is
# resolved ~1100 lines below, far past the hardware report, so the Intel scan needs its own
# read-only guess. Returns $null when there is nothing to look at; never creates or validates.
function Get-ProbableStudioVenvDir {
    $root = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() }
            elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() }
            else { $null }
    # Expand a leading ~ like the canonical resolver; without it the path stays cwd-relative.
    if ($root -and ($root -eq "~" -or $root -like "~/*" -or $root -like "~\*")) {
        if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }
        $rest = $root.Substring(1).TrimStart('/', '\')
        $root = if ($rest) { Join-Path $env:USERPROFILE $rest } else { $env:USERPROFILE }
    }
    if (-not $root) {
        if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }
        $root = Join-Path $env:USERPROFILE ".unsloth\studio"
    }
    $venv = Join-Path $root "unsloth_studio"
    if (Test-Path -LiteralPath $venv -PathType Container) { return $venv }
    return $null
}

# Is this venv's torch an XPU wheel? Read off disk, not through the interpreter: version.py
# carries the full local label ("2.9.1+xpu") and costs nothing, so a CPU-only host never pays for
# an `import torch`. The dist-info name is NOT usable -- pip normalises the local label out of it.
function Test-VenvTorchIsXpu {
    param([string]$VenvPath)
    if (-not $VenvPath) { return $false }
    try {
        $verPy = Join-Path $VenvPath "Lib\site-packages\torch\version.py"
        if (-not (Test-Path -LiteralPath $verPy)) { return $false }
        return [bool]((Get-Content -LiteralPath $verPy -TotalCount 40 -ErrorAction Stop) -match "__version__\s*=\s*'[^']*\+xpu")
    } catch { return $false }
}

# Is this venv's torch a ROCm wheel? The AMD counterpart of the check above, for the same reason:
# on an AMD box whose HIP runtime faulted the DLL load raises OSError (or hangs) while version.py
# on disk still names a perfectly good wheel, so answering by launching the interpreter that just
# died is how a working environment got deleted. Read off disk, never imported.
#
# The label is always "+rocm", never "+gfx", on every index that publishes wheels:
# repo.amd.com/rocm/whl/<arch>/torch/ (the Windows wheels, arch in the URL only) ships
# torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl, and download.pytorch.org/whl/rocm6.4 ships
# the two-component 2.8.0+rocm6.4 (Linux only). "gfx" is accepted anyway, cheaply, so a future
# per-arch local label cannot make this read a ROCm venv as unknown and delete it. The dist-info
# name is NOT usable either -- pip normalises the local label out of it.
function Test-VenvTorchIsRocm {
    param([string]$VenvPath)
    if (-not $VenvPath) { return $false }
    try {
        $verPy = Join-Path $VenvPath "Lib\site-packages\torch\version.py"
        if (-not (Test-Path -LiteralPath $verPy)) { return $false }
        return [bool]((Get-Content -LiteralPath $verPy -TotalCount 40 -ErrorAction Stop) -match "__version__\s*=\s*'[^']*\+(rocm|gfx)")
    } catch { return $false }
}

# The NVIDIA counterpart of the XPU and ROCm on-disk rescues: a wedged display driver hangs or
# faults `import torch`, and the chain then fell through to a rebuild with a null tag, deleting a
# healthy CUDA venv with no rollback copy. Returns the FAMILY, since the stale check needs it.
function Get-VenvTorchCudaTag {
    param([string]$VenvPath)
    if (-not $VenvPath) { return $null }
    try {
        $verPy = Join-Path $VenvPath "Lib\site-packages\torch\version.py"
        if (-not (Test-Path -LiteralPath $verPy)) { return $null }
        $line = (Get-Content -LiteralPath $verPy -TotalCount 40 -ErrorAction Stop |
                 Where-Object { $_ -match "__version__\s*=\s*'[^']*\+(cu[0-9]+)" } |
                 Select-Object -First 1)
        if (-not $line) { return $null }
        $m = [regex]::Match($line, "__version__\s*=\s*'[^']*\+(cu[0-9]+)")
        if ($m.Success) { return $m.Groups[1].Value.ToLowerInvariant() }
        return $null
    } catch { return $null }
}

function Test-VenvTorchIsCuda {
    param([string]$VenvPath)
    return [bool](Get-VenvTorchCudaTag -VenvPath $VenvPath)
}

# Same free disk read, plus the supported range: unsloth/models/_utils.py raises at import for an
# XPU device on torch < 2.6, so flavour alone would call a 2.5+xpu venv fine; 2.11 is the trio's
# ceiling. Flavour and range ONLY -- whether the runtime reaches the GPU is a driver question.
function Test-VenvTorchIsXpuSupported {
    param([string]$VenvPath)
    if (-not $VenvPath) { return $false }
    try {
        $verPy = Join-Path $VenvPath "Lib\site-packages\torch\version.py"
        if (-not (Test-Path -LiteralPath $verPy)) { return $false }
        $line = @(Get-Content -LiteralPath $verPy -TotalCount 40 -ErrorAction Stop |
            Where-Object { $_ -match "^__version__\s*=\s*'[^']*'" })[0]
        if (-not $line -or $line -notmatch "^__version__\s*=\s*'([^']*)'") { return $false }
        $label = $Matches[1].ToLowerInvariant()
        if ($label -notmatch '\+xpu') { return $false }
        if ($label -notmatch '^(\d+)\.(\d+)\b') { return $false }
        return ([int]$Matches[1] -eq 2 -and [int]$Matches[2] -ge 6 -and [int]$Matches[2] -lt 11)
    } catch { return $false }
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
        Write-StudioLine "CMake not found -- installing via winget (needed for the llama.cpp source build)..." -ForegroundColor Yellow
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
        Write-StudioLine "Visual Studio Build Tools not found -- installing via winget..." -ForegroundColor Yellow
        Write-StudioLine "   (Needed only for the llama.cpp source build; may take several minutes)" -ForegroundColor Gray
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
        Write-StudioLine "[ERROR] Visual Studio Build Tools are required for the llama.cpp source build but could not be found or installed." -ForegroundColor Red
        Write-StudioLine "        Manual install:" -ForegroundColor Red
        Write-StudioLine '        1. winget install Microsoft.VisualStudio.2022.BuildTools --source winget' -ForegroundColor Yellow
        Write-StudioLine '        2. Open Visual Studio Installer -> Modify -> check "Desktop development with C++"' -ForegroundColor Yellow
        Exit-SetupFailure "Visual Studio Build Tools are required for the llama.cpp source build"
    }
}

# Machine arch: PROCESSOR_ARCHITECTURE describes this PROCESS, so an emulated x64 shell on
# ARM64 reports AMD64; PROCESSOR_ARCHITEW6432 is ARM64 in exactly that case.
function Get-HostMachineArch {
    $osArch = ""
    try { $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() } catch { }
    foreach ($s in @([string]$env:PROCESSOR_ARCHITEW6432, [string]$env:PROCESSOR_ARCHITECTURE, $osArch)) {
        if ($s.ToLowerInvariant() -eq "arm64") { return "arm64" }
    }
    return "other"
}

# Detect the VC++ 2015-2022 Redistributable prebuilt llama-server and PyTorch need (they
# link VCRUNTIME140_1.dll, absent from the Universal CRT). Registry first: Runtimes\x64 is
# the only x64-specific proof; System32\vcruntime140_1.dll is arch-blind and on ARM64 may
# be the ARM64-only package, unloadable under x64 emulation.
function Test-VCRedistInstalled {
    foreach ($k in @(
        'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    )) {
        try {
            $r = Get-ItemProperty -Path $k -ErrorAction Stop
            if ($r.Installed -eq 1 -and [int]$r.Major -ge 14 -and [int]$r.Minor -ge 20) { return $true }
        } catch { }
    }
    if ((Get-HostMachineArch) -eq "arm64") { return $false }
    $sys = $env:SystemRoot
    if ($sys -and (Test-Path (Join-Path $sys 'System32\vcruntime140_1.dll'))) { return $true }
    return $false
}

# Install the VC++ 2015-2022 runtime if missing (non-fatal; usually a no-op). Unlike CMake
# and Build Tools torch cannot import without it, and winget is absent on LTSC/Server images.
function Ensure-VCRedist {
    if (Test-VCRedistInstalled) { step "vcredist" "present"; return }
    if ($StageRoot) { step "vcredist" "missing; unchanged during staging" "Yellow"; return }
    Write-StudioLine "Microsoft Visual C++ Redistributable (2015-2022) is missing; the prebuilt llama.cpp and PyTorch need it. Installing the runtime..." -ForegroundColor Yellow
    if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
        try {
            Invoke-SetupCommand { winget install --id Microsoft.VCRedist.2015+.x64 --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
            Refresh-Environment
        } catch { substep "VCRedist install failed: $($_.Exception.Message)" "Yellow" }
    }
    if (-not (Test-VCRedistInstalled)) {
        # Evergreen link; /quiet /norestart so it never blocks or reboots an unattended run.
        # Always the x64 package, deliberately: Microsoft ships it as the Arm64X superset of
        # both ARM64 and X64 binaries and documents it as the one for ARM64 devices, while
        # the arm64 package is ARM64-only (learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
        # PROCESSOR_ARCHITECTURE is wrong twice here: it reports the process, and the runtime
        # must match the interpreter loading the DLLs, an emulated x64 Python not yet created.
        $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $dst = Join-Path ([System.IO.Path]::GetTempPath()) "vc_redist.x64.exe"
        substep "winget unavailable or failed; downloading the runtime directly..."
        # Windows PowerShell 5.1 on an old image can carry a .NET default protocol set that
        # predates TLS 1.2, which aka.ms refuses -- exactly the no-winget host this fallback
        # exists for. SystemDefault (0) means "let the OS choose" and already covers TLS 1.2+,
        # so only an explicit legacy set is upgraded, and it is restored afterwards.
        $_prevProtocol = $null
        try {
            $_cur = [System.Net.ServicePointManager]::SecurityProtocol
            if ([int]$_cur -ne 0 -and ([int]$_cur -band [int][System.Net.SecurityProtocolType]::Tls12) -eq 0) {
                [System.Net.ServicePointManager]::SecurityProtocol = $_cur -bor [System.Net.SecurityProtocolType]::Tls12
                $_prevProtocol = $_cur
            }
        } catch { $_prevProtocol = $null }
        try {
            Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 300
            # HTTPS secures the transfer, not the payload, and this runs with the setup
            # process's privileges. The evergreen URL rules out a SHA-256 pin (the bytes
            # change with every VS servicing update), so check the publisher. Status alone
            # is not enough: any trusted CA's code-signing cert passes it.
            $sig = Get-AuthenticodeSignature -LiteralPath $dst
            if ($sig.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
                $null -eq $sig.SignerCertificate -or
                $sig.SignerCertificate.Subject -notmatch '(^|,\s*)O="?Microsoft Corporation"?(,|$)') {
                throw "the downloaded VC++ runtime is not validly signed by Microsoft (signature status: $($sig.Status))"
            }
            $p = Start-Process -FilePath $dst -ArgumentList '/quiet', '/norestart' -Wait -PassThru
            # 3010 = success, reboot required; usable either way.
            if ($p.ExitCode -notin @(0, 3010)) {
                substep "VC++ runtime installer exited $($p.ExitCode)" "Yellow"
            }
            Refresh-Environment
        } catch {
            substep "Direct VC++ runtime download failed: $($_.Exception.Message)" "Yellow"
        } finally {
            if ($null -ne $_prevProtocol) {
                try { [System.Net.ServicePointManager]::SecurityProtocol = $_prevProtocol } catch { }
            }
            Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
        }
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
    # A redirected stdout is not a console and GetConsoleMode fails on a non-console handle, so the
    # block below could only return $false anyway. Answer without Add-Type, which runs csc.exe and
    # drops source in %TEMP%. The CLI and the desktop app both pipe us, so this is the path the
    # compile was on.
    if ($script:StudioStdoutRedirected) { return $false }
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
        Write-StudioLine ($ansi + $Message + (Get-StudioAnsi Reset))
    } else {
        $fc = switch ($Color) {
            'Green' { 'DarkGreen' }
            'Gray' { 'DarkGray' }
            'Cyan' { 'Green' }
            default { $Color }
        }
        Write-StudioLine $Message -ForegroundColor $fc
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
            # Redact per record: uv/pip echo index URLs (credentials and all) in
            # their errors, and verbose mode must not bypass the quiet path's
            # redaction. ForEach-Object/Out-Host leave $LASTEXITCODE untouched.
            & $Command 2>&1 | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        } else {
            $output = & $Command 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
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
        Write-StudioLine "   Showing last $MaxLines lines:" -ForegroundColor DarkGray
        $lines = $lines | Select-Object -Last $MaxLines
    }
    foreach ($line in $lines) {
        Write-StudioLine "   | $line" -ForegroundColor DarkGray
    }
}
# Plain (no ANSI) form of a step/substep message on the OS stdout handle.
# Write-Host on 5.1 goes through the Information stream, which does not survive
# every install.ps1 -> python -> powershell.exe chain.
#
# One half of an either/or: when redirected this is the ONLY sink. It used to
# run in ADDITION to Write-Host, assuming that never reached the pipe. It does,
# because the CLI spawns us as `-Command "& 'setup.ps1' *>&1"`, so every step
# printed twice.
function Write-StudioStdoutMirror {
    param([Parameter(Mandatory = $true)][string]$Line)
    try {
        if ($script:StudioStdoutRedirected) {
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
    # Exactly one sink: the console handle when redirected, Write-Host (the only
    # one that colorizes) when interactive.
    if ($script:StudioStdoutRedirected) {
        Write-StudioStdoutMirror ("  {0}{1}" -f $padded, $Value)
        return
    }
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
        $fc = switch ($Color) {
            'Green' { 'DarkGreen' }
            'Yellow' { 'Yellow' }
            'Red' { 'Red' }
            'DarkGray' { 'DarkGray' }
            default { 'DarkGreen' }
        }
        # One composed record, not `-NoNewline` label + value: two Write-Host
        # calls are two Information records, and a redirected consumer turns each
        # boundary into a line break. Costs the dimmed label on no-VT consoles.
        Write-Host ("  {0}{1}" -f $padded, $Value) -ForegroundColor $fc
    }
}

function substep {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Color = "DarkGray"
    )
    # Exactly one sink, as in `step` above.
    if ($script:StudioStdoutRedirected) {
        Write-StudioStdoutMirror ("  {0,-15}{1}" -f "", $Message)
        return
    }
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
    Write-StudioLine ""
    step "frontend" "registry.npmjs.org looks blocked (corporate firewall/proxy?)" "Yellow"
    if ($mirror) {
        substep "Unsloth pins the public npm registry; your mirror is being ignored."
        substep "Detected a registry in your npm config:"
        substep "  $mirror"
        substep "Re-run pointing Unsloth at it:"
        substep "  `$env:UNSLOTH_NPM_REGISTRY='$mirror'; .\install.ps1 --local"
    } else {
        substep "If you use a private mirror/proxy, point Unsloth at it and re-run:"
        substep "  `$env:UNSLOTH_NPM_REGISTRY='https://your-mirror.example/api/npm/'; .\install.ps1 --local"
    }
    substep "(min-release-age and save-exact stay enforced.)"
}

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
Write-StudioLine ""
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    Write-StudioLine ("  " + (Get-StudioAnsi Title) + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup" + (Get-StudioAnsi Reset))
    Write-StudioLine ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    Write-StudioLine ("  " + [char]::ConvertFromUtf32(0x1F9A5) + " Unsloth Studio Setup") -ForegroundColor Green
    Write-StudioLine "  $Rule" -ForegroundColor DarkGray
}

# WebView2 caches keyed by the bundle id can keep serving the previous frontend
# after an update. Cache-only: storage, cookies, settings, models and the studio
# database are untouched. Called only once the UNSLOTH_STUDIO_HOME / STUDIO_HOME
# override is validated, so a mistyped override cannot wipe the cache and then abort.
function Clear-WebViewCaches {
    if (-not $env:LOCALAPPDATA) { return }
    $wvDefault = Join-Path $env:LOCALAPPDATA "ai.unsloth.studio\EBWebView\Default"
    # Drop the version stamp first. The old WebView still holds these files, which is
    # why the removals below can fail; the app's own clear is the retry, and it is
    # skipped while the stamp matches the running version. Unconditional, since a
    # repair or a local rebuild leaves the version unchanged and a redundant clear on
    # the next launch is the cheap side of the trade.
    Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA "ai.unsloth.studio\.webview-cache-cleared") `
        -Force -ErrorAction SilentlyContinue
    $wvCleared = $false
    foreach ($wvSub in @("Cache", "Code Cache", "GPUCache", "Service Worker")) {
        $wvPath = Join-Path $wvDefault $wvSub
        # Get-Item -Force, not Test-Path: the probe throws on an ACL denial under
        # "Stop", and a reparse point must be unlinked, not recursed into.
        $wvItem = Get-Item -LiteralPath $wvPath -Force -ErrorAction SilentlyContinue
        if (-not $wvItem) { continue }
        try {
            if ($wvItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { $wvItem.Delete() }
            else { Remove-Item -LiteralPath $wvPath -Recurse -Force -ErrorAction Stop }
            $wvCleared = $true
        } catch { }
    }
    if ($wvCleared) { substep "cleared stale WebView caches (ai.unsloth.studio); settings and data kept" }
}

# Resolve and preflight the install root before phase 1, with the same override
# precedence as the other resolvers. Everything below joins onto USERPROFILE, so a
# blank one must fail here instead of throwing a raw binding error under "Stop".
# Null or empty only: Join-Path accepts whitespace, and those runs used to complete.
if ([string]::IsNullOrEmpty($env:USERPROFILE)) {
    Write-StudioLine "ERROR: USERPROFILE is not set." -ForegroundColor Red
    Exit-SetupFailure "USERPROFILE is not set"
}
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
        # Mirror setup.sh and install.ps1: fail before install work if it is read-only.
        $_setupWriteProbe = Join-Path $StudioHome (".unsloth-write-probe-" + [guid]::NewGuid())
        try {
            [System.IO.File]::WriteAllText($_setupWriteProbe, "")
            Remove-Item -LiteralPath $_setupWriteProbe -Force -ErrorAction SilentlyContinue
        } catch {
            Write-StudioLine "ERROR: $_studioOverrideVar=$StudioHome is not writable." -ForegroundColor Red
            Exit-SetupFailure "$_studioOverrideVar=$StudioHome is not writable"
        }
    } else {
        Write-StudioLine "ERROR: $_studioOverrideVar=$_studioOverride does not exist." -ForegroundColor Red
        Write-StudioLine "       Run install.ps1 to create the install root before 'unsloth studio update'." -ForegroundColor Red
        Exit-SetupFailure "$_studioOverrideVar=$_studioOverride does not exist"
    }
} else {
    $StudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
}
$StageRoot = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_STAGE_ROOT)) { $env:UNSLOTH_STUDIO_STAGE_ROOT.Trim() } else { $null }
$RuntimeRoot = if ($StageRoot) { $StageRoot } else { $StudioHome }
$VenvDir = Join-Path $RuntimeRoot "unsloth_studio"
$StudioOwnedMarker = ".unsloth-studio-owned"
# Mirrors install_manifest.NO_TORCH_MARKER; keep the two in step.
$NoTorchMarker = ".unsloth-no-torch"
$LegacyStudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
$StudioHomeIsCustom = Test-StudioHomeIsCustom
$LlamaCppDir = Get-ManagedLlamaCppDir
$UnslothHome = Split-Path -Parent $LlamaCppDir

$WithLlamaCppDir = $null
$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight
if ($llamaPreflightFailure) {
    Exit-SetupFailure $llamaPreflightFailure
}

# Back up User PATH under HKCU\Software\Unsloth before any modifications.
if (-not $StageRoot) {
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
        Write-StudioLine "[DEBUG] Could not back up User PATH: $($_.Exception.Message)" -ForegroundColor DarkGray
    }
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
        [int]$TimeoutSec = 10,
        # Driver warnings on stderr would corrupt machine-readable --query-gpu
        # output; the human-readable probes keep the default merge.
        [switch]$StdoutOnly
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
        if ($StdoutOnly) { return $outTask.Result }
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
                    Write-StudioLine "   Found nvidia-smi at $(Split-Path $p -Parent)" -ForegroundColor Gray
                    break
                }
            } catch {}
        }
    }
}
# ── Helper: run amd-smi without triggering a UAC elevation prompt ──
# amd-smi on Windows auto-elevates to read GPU/APU memory, surfacing a confusing
# DiskPart UAC prompt mid-install (Unsloth backend amd.py hits the same). RunAsInvoker
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
$script:ROCmGpuLabels = @()   # every AMD adapter name WMI reported (shadowing-aware inference)
$script:ROCmGfxArch = $null
# Beside ROCmGfxArch, NOT inside the `-not $HasNvidiaSmi` block below: the ROCm summary
# reads this unconditionally, so an NVIDIA host would hit an undefined variable under a
# caller's Set-StrictMode. Assigning here also clears a stale value on a re-invocation.
$script:ROCmUnsupportedGfxArch = $null
# APU gfx arches whose board commonly also carries a discrete Radeon. HIP often
# enumerates the APU first, so an index-0 pick reads the iGPU's arch and the dGPU never
# gets its wheels (#7776: gfx1036 Raphael shadowing a gfx1200 RX 9060 XT). In sync with
# _SHADOWING_INTEGRATED_GFX in studio/install_python_stack.py. The Strix arches
# (gfx1150/1151/1152) are deliberately absent: first-class training targets, untouched.
$script:ShadowingIntegratedGfx = @(
    "gfx90c",   # Renoir / Cezanne
    "gfx1013",  # Cyan Skillfish
    "gfx1033",  # Van Gogh
    "gfx1035",  # Rembrandt
    "gfx1036",  # Raphael / Mendocino
    "gfx1103",  # Phoenix / Hawk Point
    "gfx1153"   # Krackan Point 2
)

# gfx arch -> AMD per-arch wheel index family. Defined here, not beside $ROCmIndexUrl,
# because Resolve-ShadowingGfxPick runs during detection, long before the install block,
# and must know which arches AMD ships Windows wheels for. In sync with
# _GFX_TO_AMD_INDEX_ARCH (test_rocm_arch_table_parity.py).
$archFamilyMap = @{
    "gfx1201" = "gfx120X-all"; "gfx1200" = "gfx120X-all"  # RDNA 4
    "gfx1151" = "gfx1151";     "gfx1150" = "gfx1150"      # RDNA 3.5 (Strix Halo/Point)
    "gfx1152" = "gfx1152"                                 # RDNA 3.5 (Krackan Point)
    "gfx1103" = "gfx110X-all"; "gfx1102" = "gfx110X-all"  # RDNA 3
    "gfx1101" = "gfx110X-all"; "gfx1100" = "gfx110X-all"
    "gfx1036" = "gfx103X-all"; "gfx1035" = "gfx103X-all"  # RDNA 2 (RX 6000)
    "gfx1034" = "gfx103X-all"; "gfx1033" = "gfx103X-all"
    "gfx1032" = "gfx103X-all"; "gfx1031" = "gfx103X-all"
    "gfx1030" = "gfx103X-all"
    "gfx90a"  = "gfx90a";      "gfx908"  = "gfx908"       # MI200/MI100
}


# True when any of the three masks is set. Mirrors _visible_devices_pinned(): ANY value
# is a deliberate selection, "" and "-1" included -- those select NO GPU rather than
# meaning "unset".
function Test-VisibleDevicesPinned {
    foreach ($visEnv in @($env:HIP_VISIBLE_DEVICES, $env:ROCR_VISIBLE_DEVICES, $env:CUDA_VISIBLE_DEVICES)) {
        if ($null -ne $visEnv) { return $true }
    }
    return $false
}

# The one mask -> index resolver for every pick site below; the sites used to inline
# their own expressions and disagreed (hipinfo rejected " 1 ", amd-smi rejected "1,0"),
# and a mask Resolve-ShadowingGfxPick honours but the index ignores lands on GPU 0, the
# very iGPU this preference exists to skip. Mirrors _pick_visible_index(): first-set-wins
# (the runtime lets an empty HIP mask shadow CUDA_VISIBLE_DEVICES rather than defer to
# it), and an unparseable or out-of-range value falls back to GPU 0.
function Resolve-VisibleGpuIndex {
    param([int]$Count)
    foreach ($visEnv in @($env:HIP_VISIBLE_DEVICES, $env:ROCR_VISIBLE_DEVICES, $env:CUDA_VISIBLE_DEVICES)) {
        if ($null -eq $visEnv) { continue }
        $val = $visEnv.Trim()
        if ($val -eq "" -or $val -eq "-1") { return 0 }
        $first = ($val -split ',')[0].Trim()
        # TryParse, not [int]: '2147483648' overflows and .NET's \d also matches full-width
        # digits, and either cast throws a TERMINATING error under this script's
        # $ErrorActionPreference = "Stop", aborting the install where nothing catches it.
        [int]$parsed = 0
        if ([int]::TryParse($first, [ref]$parsed)) {
            if ($parsed -ge 0 -and $parsed -lt $Count) { return $parsed }
            substep "[WARN] HIP/ROCR/CUDA_VISIBLE_DEVICES index $first is out of range ($Count GPU(s) detected); defaulting to GPU 0 for arch selection" "Yellow"
        }
        return 0
    }
    return 0
}

# Mirrors _dedup_pick(). setup.ps1 resolves the arch and builds $ROCmIndexUrl from it
# before ever invoking the Python stack installer, so the shadowing-iGPU skip has to
# happen here too or a fresh install still pulls the iGPU's wheel family. Returns $Picked
# unchanged when pinned, when only one distinct arch exists, or when no discrete arch is.
function Resolve-ShadowingGfxPick {
    param([AllowNull()][string]$Picked, [AllowNull()][string[]]$AllArches)
    if (-not $Picked) { return $Picked }
    # Only arches in $archFamilyMap have an AMD Windows wheel index.
    function Test-GfxHasWheels { param([AllowNull()][string]$Arch) return [bool]($Arch -and $archFamilyMap.ContainsKey($Arch)) }
    # A selected device is honoured verbatim; never repick over the user.
    if (Test-VisibleDevicesPinned) { return $Picked }
    if ($script:ShadowingIntegratedGfx -notcontains $Picked) { return $Picked }
    $distinctArches = @($AllArches | Select-Object -Unique)
    if ($distinctArches.Count -lt 2) { return $Picked }
    # Deposing a supported APU for a discrete card with no Windows wheels (gfx1036 + an
    # older gfx1010) resolves to no index and drops the host to CPU, worse than the
    # shadowing itself. So prefer a wheel-backed discrete card, and fall back to an
    # unsupported one only when the pick has no wheels either: taking the first
    # non-integrated arch instead sent gfx90c,gfx1010,gfx1200 to CPU torch despite the
    # supported gfx1200. Mirrors _dedup_pick()'s `_withWheels or (...)`.
    $pickedHasWheels = Test-GfxHasWheels $Picked
    $others = @($AllArches | Where-Object { $script:ShadowingIntegratedGfx -notcontains $_ })
    $withWheels = @($others | Where-Object { Test-GfxHasWheels $_ })
    $candidates = if ($withWheels.Count -gt 0) { $withWheels }
                  elseif (-not $pickedHasWheels) { $others }
                  else { @() }
    $discreteArch = $candidates | Select-Object -First 1
    if (-not $discreteArch) { return $Picked }
    # Not always device 1: on gfx1036,gfx1010,gfx1200 the pick is device 2, and
    # naming 1 would expose the gfx1010 the wheels do not target.
    $discreteIdx = [array]::IndexOf(@($AllArches), $discreteArch)
    if ($discreteIdx -lt 0) { $discreteIdx = 1 }
    substep "multiple AMD GPUs detected ($($distinctArches -join ', ')); installing for the discrete $discreteArch instead of the integrated $Picked" "Cyan"
    # HIP still enumerates the iGPU as device 0 at runtime, so the wheels alone do not
    # steer training to the dGPU; setx makes the mask persist across sessions.
    substep "Run 'setx HIP_VISIBLE_DEVICES $discreteIdx' and reopen your terminal so Unsloth uses $discreteArch at runtime too, not just at install time" "Cyan"
    return $discreteArch
}
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
        # Unsloth home, else the venv hipInfo isn't caught.
        $venvRoots = @()
        if ($env:VIRTUAL_ENV) { $venvRoots += $env:VIRTUAL_ENV }
        $vd = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue
        if ($vd) { $venvRoots += $vd }
        if ($env:UNSLOTH_SETUP_PYTHON) {
            try { $venvRoots += (Split-Path -Parent (Split-Path -Parent $env:UNSLOTH_SETUP_PYTHON)) } catch {}
        }
        if ($env:USERPROFILE) { $venvRoots += (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio") }
        # A custom Unsloth home (UNSLOTH_STUDIO_HOME / STUDIO_HOME alias) moves the
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
                if ($_hipAllArches.Count -gt 0) {
                    # hipinfo is itself a HIP application, so under a mask it already
                    # enumerated only the visible devices, renumbered from 0; indexing it
                    # again applies the mask twice and lands on the wrong card. amd-smi and
                    # the WMI path below list every GPU, so they still index.
                    $script:ROCmGfxArch = $_hipAllArches[0]
                    $script:ROCmGfxArch = Resolve-ShadowingGfxPick $script:ROCmGfxArch $_hipAllArches
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
                        # amd-smi lists every GPU regardless of the masks, so resolve the
                        # index here, via the shared helper so a comma list or a padded
                        # value cannot select a different GPU than elsewhere.
                        $script:ROCmGfxArch = $allGfxArches[(Resolve-VisibleGpuIndex $allGfxArches.Count)]
                        $script:ROCmGfxArch = Resolve-ShadowingGfxPick $script:ROCmGfxArch $allGfxArches
                        $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                    } else {
                        # Attempt 2: 'static --asic' exposes ASIC details on ROCm 6+,
                        # including the GFX target needed for wheel index selection.
                        $smiAsicOut = ""
                        try { $smiAsicOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('static','--asic') } catch {}
                        # Every token, not just the first: this branch is reached precisely
                        # when 'list' saw multiple GPUs but printed no arches, so a leading
                        # iGPU here reintroduces #7776.
                        $asicGfxArches = @([regex]::Matches($smiAsicOut, '(?i)\b(gfx\d+[a-z]?)\b') |
                            ForEach-Object { $_.Groups[1].Value.ToLower() })
                        if ($asicGfxArches.Count -gt 0) {
                            $script:ROCmGfxArch = $asicGfxArches[(Resolve-VisibleGpuIndex $asicGfxArches.Count)]
                            $script:ROCmGfxArch = Resolve-ShadowingGfxPick $script:ROCmGfxArch $asicGfxArches
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
            # Keep every AMD adapter, not just the first: WMI orders controllers as the
            # driver stack enumerated them, so a shadowing iGPU can lead here exactly as
            # under HIP (#7776), and the inference below runs the same preference over the
            # whole list. ConfigManagerErrorCode 0 is "working properly": a disabled or
            # driver-errored Radeon must not depose a working iGPU.
            $amdGpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "AMD|Radeon" })
            $healthyGpus = @($amdGpus | Where-Object {
                ($null -eq $_.ConfigManagerErrorCode) -or ($_.ConfigManagerErrorCode -eq 0) })
            # If that leaves none, the filter alone made the host look GPU-less and the
            # inference forwards nothing: code 45 ("not connected") is routine on a muxless
            # laptop with a parked dGPU, and with no healthy peer there is nothing to
            # depose. Mirrors the same fallback in install_python_stack.py's WMI path.
            # @() wraps the WHOLE if, not each branch: a one-element array unrolls on its way
            # out of an if-expression, and a scalar's .Count is $null under Windows PowerShell
            # 5.1, so the single Radeon in the machine read as no AMD GPU at all -- no
            # $script:ROCmGpuLabels, no inferred gfx arch, "gpu none" in the report, and an
            # installed +rocm venv judged stale against a required "cpu" (#8335). Same idiom as
            # the Intel scan below and in install.ps1.
            $wmiGpus = @(if ($healthyGpus.Count -gt 0) { $healthyGpus } else { $amdGpus })
            if ($wmiGpus.Count -gt 0) {
                $script:ROCmGpuLabels = @($wmiGpus | ForEach-Object { $_.Name })
                $ROCmGpuLabel = $script:ROCmGpuLabels[0]
            }
        } catch {}
    }
    # Peer names for the REPORT ONLY. amd-smi can confirm a runtime with no gfx token and
    # only the first card's name, and the scan above is skipped there, so the verdict below
    # would speak for a host it has seen one adapter of. Kept out of $script:ROCmGpuLabels
    # deliberately: that feeds the inference, and widening it turned a CPU install into ROCm.
    $script:ROCmPeerLabels = @()
    if ($HasROCm -and -not $script:ROCmGfxArch) {
        try {
            $peerGpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "AMD|Radeon" })
            $healthyPeers = @($peerGpus | Where-Object {
                ($null -eq $_.ConfigManagerErrorCode) -or ($_.ConfigManagerErrorCode -eq 0) })
            $usePeers = @(if ($healthyPeers.Count -gt 0) { $healthyPeers } else { $peerGpus })
            $script:ROCmPeerLabels = @($usePeers | ForEach-Object { $_.Name })
        } catch {}
    }

    # GPU name -> gfx arch for AMD generations Unsloth's ROCm wheels do NOT cover: RDNA 1
    # and Polaris 10/20/30 (unslothai#8529). Kept apart from $nameArchTable on purpose: it
    # only WORDS a message, never selects a wheel index or a prebuilt. AMD's TheRock ships
    # RDNA 1 wheels, but not on the repo.amd.com indexes routed here, and never gfx803.
    # The (?!0) guards stop "RX 570" swallowing an "RX 5700". Names from LLVM's AMDGPU
    # tables plus libdrm amdgpu.ids/pci.ids for the Navi 10/14 professional parts LLVM
    # omits; nothing is guessed, so Polaris 11/12 (RX 460/550/560, a different die) is
    # left out.
    $unsupportedNameArchTable = @(
        @{ P = "Radeon Pro V520|Radeon Pro 5600M";        A = "gfx1011" }  # RDNA 1
        @{ P = "RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700";     A = "gfx1010" }  # RDNA 1 (Navi 10)
        @{ P = "RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300";        A = "gfx1012" }  # RDNA 1 (Navi 14)
        @{ P = "RX 4[78]0(?!0)|RX 5[789]0(?!0)|Radeon Pro WX 7100|Radeon Pro WX 5100"; A = "gfx803"  }  # Polaris 10/20/30
    )
    $script:ROCmUnsupportedGfxArch = $null
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
                @{ P = "9070|9080|R9700";                                     A = "gfx1201" }  # RDNA 4 (Navi 48: Radeon RX 9070 XT / 9070 GRE / 9070 / 9080, Radeon AI PRO R9700)
                @{ P = "9060";                                                A = "gfx1200" }  # RDNA 4 (Navi 44: Radeon RX 9060 XT / 9060)
                @{ P = "8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max"; A = "gfx1151" }  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
                @{ P = "890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]"; A = "gfx1150" }  # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
                @{ P = "860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33"; A = "gfx1152" }  # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
                @{ P = "RX 7900|PRO W7900|PRO W7800";                         A = "gfx1100" }  # RDNA 3 desktop / workstation (Navi 31)
                @{ P = "RX 7800|RX 7700(?!S)|PRO W7700|PRO V710";             A = "gfx1101" }  # RDNA 3 (Navi 32)
                @{ P = "RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500";        A = "gfx1102" }  # RDNA 3 (Navi 33)
                @{ P = "780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme"; A = "gfx1103" }  # RDNA 3 iGPU (Phoenix / Hawk Point)
                @{ P = "RX 6900|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900";  A = "gfx1030" }  # RDNA 2 (Navi 21) -- gfx103X family
                @{ P = "RX 6650|RX 6600|PRO W6600|PRO W6650";                  A = "gfx1032" }  # RDNA 2 (Navi 23) -- gfx103X family
                @{ P = "RX 6500|RX 6400|RX 6300|PRO W6400|PRO W6500";          A = "gfx1034" }  # RDNA 2 (Navi 24) -- gfx103X family
            )
            function Get-GfxArchFromGpuName {
                param([AllowNull()][string]$Name, [object[]]$Table)
                if (-not $Name) { return $null }
                foreach ($row in $Table) { if ($Name -match $row.P) { return $row.A } }
                return $null
            }
            # Only the WMI path carries more than one name; other callers left a single
            # synthesized label, so default to it.
            # @() wraps the WHOLE if, not each branch: a one-element branch unrolls on its way
            # out, leaving a bare String on a single-adapter host, and $gpuNames[$nameIdx] then
            # indexes the NAME and yields "A". Nothing maps, and the $nameArches[0] rescue below
            # is skipped under a visible-device mask, so a pinned single-GPU host inferred no
            # arch at all -- the same "gpu none" the WMI scan above used to report.
            $gpuNames = @(if ($script:ROCmGpuLabels) { @($script:ROCmGpuLabels) } else { @($ROCmGpuLabel) })
            # Index over the ADAPTER list, not the inferred arches: an unrecognised name
            # drops out below, and indexing the shortened list would name the wrong card.
            $nameIdx = Resolve-VisibleGpuIndex $gpuNames.Count
            $nameArches = @()
            foreach ($gpuName in $gpuNames) {
                $inferred = Get-GfxArchFromGpuName -Name $gpuName -Table $nameArchTable
                if ($inferred) { $nameArches += $inferred }
            }
            $pickedName = Get-GfxArchFromGpuName -Name $gpuNames[$nameIdx] -Table $nameArchTable
            # Borrow another adapter's arch only when unpinned: an unmappable leading
            # adapter is exactly the #7776 iGPU and the named discrete card should decide,
            # but under a mask substituting installs wheels for a GPU they masked away.
            if (-not $pickedName -and -not (Test-VisibleDevicesPinned) -and $nameArches.Count -gt 0) {
                $pickedName = $nameArches[0]
            }
            if ($pickedName) {
                # Repick only when every adapter mapped: an unknown name may BE the
                # discrete card, so skipping the iGPU could pick the wrong one.
                $script:ROCmGfxArch = if ($nameArches.Count -eq $gpuNames.Count) {
                    Resolve-ShadowingGfxPick -Picked $pickedName -AllArches $nameArches
                } else { $pickedName }
                $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
                substep "gfx arch inferred from GPU name: $script:ROCmGfxArch" "Cyan"
                substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$script:ROCmGfxArch to skip inference next time" "Cyan"
            } else {
                # Nothing mapped: the card may be a generation ROCm never covered rather
                # than one we failed to recognise (unslothai#8529). $script:ROCmGfxArch
                # stays null either way, so only the wording of the report below moves.
                # Stay quiet when a PEER is covered: "no override can help" is false beside a
                # card that has wheels. Read-only; never reaches $gpuNames or the inference.
                # HIP/ROCR only, unlike Test-VisibleDevicesPinned: CUDA_VISIBLE_DEVICES says
                # nothing about which Radeon was chosen, and counting it would fire the verdict
                # beside a covered card.
                $unsupMasked = @($env:HIP_VISIBLE_DEVICES, $env:ROCR_VISIBLE_DEVICES) |
                    Where-Object { $null -ne $_ }
                $unsupPeerCovered = $false
                if ($unsupMasked) {
                    # Under a mask a peer cannot answer for the named card, as the
                    # arch-borrowing rule above. But $gpuNames is one synthesized label on the
                    # amd-smi path, and $nameIdx cannot index a card it never saw, so a mask
                    # onto a supported peer would be blamed on adapter 0. Unseen peer, no
                    # verdict: Win32_VideoController does not promise HIP's order to guess with.
                    $unsupPeerCovered = ($script:ROCmPeerLabels.Count -gt $gpuNames.Count)
                } else {
                    foreach ($peerName in $script:ROCmPeerLabels) {
                        if (Get-GfxArchFromGpuName -Name $peerName -Table $nameArchTable) {
                            $unsupPeerCovered = $true
                            break
                        }
                    }
                }
                if (-not $unsupPeerCovered) {
                    $script:ROCmUnsupportedGfxArch = Get-GfxArchFromGpuName -Name $gpuNames[$nameIdx] -Table $unsupportedNameArchTable
                }
            }
        }
    }
    # 3. Last resort: the arch install.ps1 resolved a second ago. Last because everything above
    #    is mask- and shadowing-aware and the installer's scan is not, so this fills a gap rather
    #    than deposing a better answer. Without it, a scan that answers there but not here expects
    #    cpu torch against the ROCm wheels just placed, calls the venv stale, and loops forever.
    #    Private, never UNSLOTH_ROCM_GFX_ARCH: nested installers read that as an operator
    #    override, and this value is inferred, not chosen by anyone.
    #    Skipped under a visible-device mask, matching the inference above, which leaves
    #    $pickedName unset rather than borrowing a peer's arch when pinned. The installer's scan
    #    ignores the masks, so its answer is the FIRST adapter, not the selected one, and taking
    #    it would install for a GPU the mask hides from the runtime entirely. A host that wants
    #    an arch named under a mask sets UNSLOTH_ROCM_GFX_ARCH, which still wins above.
    if (-not $script:ROCmGfxArch -and $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF -and
        -not (Test-VisibleDevicesPinned)) {
        $script:ROCmGfxArch = $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF.Trim().ToLower()
        $ROCmGpuLabel = "AMD ROCm ($script:ROCmGfxArch)"
        substep "gfx arch forwarded by the installer: $script:ROCmGfxArch" "Cyan"
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

# gfx arches AMD publishes Windows ROCm wheels for (repo.amd.com/rocm/whl/<family>). Hoisted
# above the Intel scan, which needs it: an unlisted arch gets CPU torch, so it must not outrank a
# usable Arc card. test_rocm_arch_table_parity.py keeps it in sync with $archFamilyMap below.
$_rocmWheelArches = @(
    "gfx1201", "gfx1200",           # RDNA 4
    "gfx1151", "gfx1150", "gfx1152",  # RDNA 3.5 (Strix Halo/Point, Krackan Point)
    "gfx1103", "gfx1102", "gfx1101", "gfx1100",  # RDNA 3
    "gfx1036", "gfx1035", "gfx1034", "gfx1033", "gfx1032", "gfx1031", "gfx1030",  # RDNA 2 (RX 6000)
    "gfx90a", "gfx908"              # MI200 / MI100
)
# "AMD gets GPU wheels here", NOT "an AMD GPU is present": $HasROCm / $ROCmGfxArch are true on
# unmapped arches (Vega, RDNA1) too, and those install CPU torch.
$AmdHasGpuWheels = [bool]($script:ROCmGfxArch -and ($_rocmWheelArches -contains $script:ROCmGfxArch))

# Mirrors the Intel scan in install.ps1 so setup does not report "none (chat-only)" right after
# install.ps1 reported a usable Arc GPU. Self-contained, because `studio update` runs setup.ps1
# standalone. Same $AmdHasGpuWheels gate (an Arc card wins over an AMD host heading for CPU torch
# anyway) and the same Arc / Data Center match.
$script:IsIntelXpu = $false
# Set when the stale check below keeps an installed +xpu venv instead of wiping it. Separate from
# $script:IsIntelXpu on purpose: it steers the INSTALL, not the hardware report, which must stay
# honest about the NVIDIA GPU that is also in the machine.
$script:PreservedXpuVenv = $false
# Flavour tag ("rocm" / "cu128" / "xpu") of a GPU wheel the stale check below kept. Declared here,
# like the flag above it, because the index selection reads it on every run while a fresh install
# never reaches the assignment -- so under a caller's Set-StrictMode the read would be fatal.
$script:PreservedInstallerTorchTag = $null
$IntelGpuLabel = $null
if (-not $HasNvidiaSmi -and -not $AmdHasGpuWheels) {
    try {
        # Bounded, registry as the fallback when WMI does not answer: an unbounded query hangs
        # `studio update`, a swallowed one silently reports no GPU.
        $_gpuScan = Invoke-BoundedVideoControllerScan
        # @() wraps the WHOLE if, not each branch: a one-element array unrolls on its way out,
        # making $_gpuNames a String on a single-adapter host and turning the += below into
        # string concatenation.
        $_gpuNames = @(if ($_gpuScan.Ok) { $_gpuScan.Names } else { Get-IntelRegistryAdapterNames })
        # One definition for both the reconciliation gate and the classification below.
        $_xpuNameRe = "(?i)Intel.*(Arc|Data Center GPU)"
        # On non-English Windows the WMI name carries no ASCII "Intel" for the classification
        # below. The registry helper resolves the PCI vendor id, so use it to RE-LABEL an adapter
        # WMI already reported, never to add one (an unmatched entry is a driver record outliving
        # its card). Gated on the absence of an XPU match, not of any Intel name: a hybrid laptop
        # reports "Intel UHD" next to a localized Arc.
        if ($_gpuScan.Ok -and -not ($_gpuNames | Where-Object { $_ -match $_xpuNameRe })) {
            foreach ($_reg in @(Get-IntelRegistryAdapterNames)) {
                foreach ($_wmiName in $_gpuNames) {
                    if ($_wmiName -and $_reg.Contains($_wmiName)) { $_gpuNames += $_reg; break }
                }
            }
        }
        $xpuGpu = $_gpuNames | Where-Object { $_ -match $_xpuNameRe } | Select-Object -First 1
        if ($xpuGpu) { $script:IsIntelXpu = $true; $IntelGpuLabel = $xpuGpu }
    } catch {}
    # Neither WMI nor the registry recognised an adapter, so ask the environment: an installed
    # +xpu wheel whose runtime initialises proves the host better than a marketing name. The same
    # check runs ~1300 lines below, but AFTER the report just below, so without this an Arc user
    # on a wedged WMI is told no training GPU exists and then watches setup keep the XPU
    # environment. Own try, so a throwing scan or a junk UNSLOTH_STUDIO_HOME cannot abort setup.
    if (-not $script:IsIntelXpu) {
        try {
            $_probeVenv = Get-ProbableStudioVenvDir
            if (Test-VenvTorchIsXpu $_probeVenv) {
                $_probePy = Join-Path $_probeVenv "Scripts\python.exe"
                if (Test-TorchXpuAvailable -PythonExe $_probePy) {
                    $script:IsIntelXpu = $true
                    $IntelGpuLabel = "Intel XPU runtime (reported by PyTorch)"
                }
            }
        } catch {}
    }
}

if ($HasNvidiaSmi) {
    step "gpu" "NVIDIA GPU detected"
} elseif ($script:IsIntelXpu) {
    # Ranks above every AMD branch: only true when AMD gets no GPU wheel ($AmdHasGpuWheels gates
    # the scan above), so those branches would all end on CPU torch.
    Write-StudioLine ""
    step "gpu" "Intel GPU detected" "Green"
    substep "$IntelGpuLabel"
    substep "PyTorch XPU (SYCL) wheels provide training and GPU inference on this GPU." "Cyan"
    Write-StudioLine ""
} elseif ($HasROCm -and -not $script:ROCmUnsupportedGfxArch) {
    # Guarded like the HIP SDK arm below: amd-smi can report a GPU with no gfx token
    # and only a market name, which sets $HasROCm without an arch.
    step "gpu" $ROCmGpuLabel
    $hipSdkPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "on system PATH" }
    substep "HIP SDK: $hipSdkPath"
    if ($script:ROCmVersionFull) { substep "hipconfig: $script:ROCmVersionFull" }
} elseif ($HipSdkInstalled -and $ROCmGpuLabel -and -not $script:ROCmUnsupportedGfxArch) {
    # HIP SDK installed but ROCm can't see the device (driver issue, not SDK issue).
    # Excludes cards already known to be out of scope: the #8529 reporters installed the
    # HIP SDK BECAUSE this arm said to, so unguarded it hides the arm below from them.
    $sdkVer = if ($script:ROCmVersionFull) { " (HIP $script:ROCmVersionFull)" } else { "" }
    Write-StudioLine ""
    step "gpu" "AMD GPU detected -- not ROCm-accessible$sdkVer" "Yellow"
    substep "Detected: $ROCmGpuLabel" "Yellow"
    substep "[WARN] HIP SDK is installed but hipinfo reports no ROCm-capable device." "Yellow"
    substep "       This is a driver issue, not an SDK issue." "Yellow"
    substep "       Ensure the ROCm compute driver is installed alongside the display driver:" "Yellow"
    substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
} elseif ($script:ROCmGfxArch) {
    # Known arch: PyTorch comes from AMD's bundled-runtime ROCm wheels (repo.amd.com),
    # which ship their own runtime -- HIP SDK optional (only adds the system toolchain).
    Write-StudioLine ""
    step "gpu" "AMD ROCm ($script:ROCmGfxArch)" "Cyan"
    substep "Detected: $ROCmGpuLabel" "Cyan"
    substep "GPU PyTorch uses AMD's bundled-runtime ROCm wheels -- HIP SDK not required (optional)." "Cyan"
    Write-StudioLine ""
} elseif ($script:ROCmUnsupportedGfxArch) {
    # Detected, identified, out of scope for ROCm PyTorch. Ranks above the "arch
    # unknown" arm below: the arch is known here, and that arm's advice cannot succeed
    # on this GPU (unslothai#8529).
    Write-StudioLine ""
    step "gpu" "AMD GPU detected ($script:ROCmUnsupportedGfxArch) -- no ROCm PyTorch wheels Unsloth installs" "Yellow"
    substep "Detected: $ROCmGpuLabel" "Yellow"
    # Not "training runs on CPU": with no CUDA/XPU visible, unsloth raises
    # NotImplementedError at import (unsloth/device_type.py). The Vulkan setter is
    # single-quoted so PowerShell prints $env:... rather than expanding it; a pasted
    # VAR=value resolves as a command name here and sets nothing.
    # Both claims are conditional: an explicit index pin is honoured for any arch, and
    # this same script THROWS on the Vulkan variable on Windows ARM64, where no bundle
    # is published, so the usual advice would abort the next update instead of helping.
    $unsupPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) -or `
                   (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY))
    if ($unsupPinned) {
        substep "Unsloth ships no ROCm PyTorch wheels for $script:ROCmUnsupportedGfxArch, but the torch" "Yellow"
        substep "index you pinned is used as given, so torch is whatever it publishes." "Yellow"
    } else {
        substep "Unsloth installs no ROCm PyTorch wheels for $script:ROCmUnsupportedGfxArch, so torch stays" "Yellow"
        substep "CPU-only: Unsloth training and GPU inference are unavailable. Installing the" "Yellow"
        substep "HIP SDK or setting UNSLOTH_ROCM_GFX_ARCH will not change that for it." "Yellow"
    }
    if ((Get-HostMachineArch) -eq "arm64") {
        substep "GGUF chat would need Vulkan on this GPU, and no Windows ARM64 Vulkan bundle is published: build llama.cpp from source, or run this on x64." "Yellow"
    } else {
        substep "GGUF chat can still use this GPU through Vulkan: set" "Yellow"
        substep '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan" and re-run the installer. It selects' "Yellow"
        substep "the llama.cpp bundle at install time, so setting it afterwards has no effect" "Yellow"
        substep "until you install or update again." "Yellow"
    }
    Write-StudioLine ""
} elseif ($ROCmGpuLabel) {
    Write-StudioLine ""
    step "gpu" "AMD GPU detected -- arch unknown" "Yellow"
    substep "Detected: $ROCmGpuLabel" "Yellow"
    substep "Could not determine the GPU arch (gfx...). Install the HIP SDK or set" "Yellow"
    substep "UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm PyTorch:" "Yellow"
    substep "https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    Write-StudioLine ""
} else {
    Write-StudioLine ""
    step "gpu" "none (chat-only / GGUF)" "Yellow"
    substep "Training and GPU inference require an NVIDIA, AMD ROCm, or Intel Arc GPU." "Yellow"
    Write-StudioLine ""
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
} elseif ($StageRoot) {
    step "long paths" "disabled; unchanged during staging" "Yellow"
} else {
    Write-StudioLine "Windows Long Paths not enabled (required for Triton compilation and deep dependency paths)." -ForegroundColor Yellow
    Write-StudioLine "   Requesting admin access to fix..." -ForegroundColor Yellow
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
        Write-StudioLine "       Run this manually in an Admin terminal:" -ForegroundColor Yellow
        Write-StudioLine '       reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f' -ForegroundColor Cyan
    }
}

# ============================================
# 1b. Git (only required for --local / source installs)
# ============================================
# Was fatal as "required by pip and npm", but the consumer path uses neither: the
# unsloth-zoo git+https URL is STUDIO_LOCAL_INSTALL only, node is a pinned prebuilt, and the
# frontend lockfile has no VCS deps. Being fatal blocked clean no-winget Windows boxes.
$HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
if (-not $HasGit) {
    # Fatal only where git is used: --local and the opt-in llama.cpp source build. A local
    # llama.cpp dir overrides those opt-ins, but only once it holds a reusable binary:
    # pointing at the canonical install location with nothing built there falls through to
    # the normal install, so an explicit source build still needs git. The automatic
    # fallback after a failed prebuilt download is not knowable here; Phase 4 handles it.
    $gitNeeded = ($env:STUDIO_LOCAL_INSTALL -eq '1')
    $_localLlamaDir = if ($env:UNSLOTH_LOCAL_LLAMA_CPP_DIR) { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR.Trim() } else { "" }
    $_localLlamaBuilt = $false
    if ($_localLlamaDir) {
        # Same layout candidates as the reuse check in Phase 4.
        foreach ($_c in @("llama-server.exe", "build\bin\llama-server.exe", "build\bin\Release\llama-server.exe")) {
            # Denied here terminated the run under "Stop" long before Phase 4's
            # guarded probes, so this scan needs the same three-state handling.
            $_cState = Get-PathState -Path (Join-Path $_localLlamaDir $_c)
            if ($_cState -eq "Denied") {
                Exit-PathAccessDenied -Path $_localLlamaDir -Label "the UNSLOTH_LOCAL_LLAMA_CPP_DIR build" -UserSupplied
            }
            if ($_cState -eq "Present") { $_localLlamaBuilt = $true; break }
        }
    }
    if (-not $_localLlamaBuilt) {
        $_prForce = if ($env:UNSLOTH_LLAMA_PR_FORCE) { $env:UNSLOTH_LLAMA_PR_FORCE.Trim() } else { $DefaultLlamaPrForce }
        $_llamaSrc = $DefaultLlamaSource -replace '\.git$', ''
        # Same tag resolution as Phase 4. "master" is a branch, never a release, so the
        # prebuilt lookup always misses and Phase 4 rebuilds it from source.
        $_llamaTag = if ($env:UNSLOTH_LLAMA_TAG) { $env:UNSLOTH_LLAMA_TAG } else { $DefaultLlamaTag }
        if ($_llamaTag -eq "master") { $gitNeeded = $true }
        if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq '1') { $gitNeeded = $true }
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_LLAMA_PR)) { $gitNeeded = $true }
        # Same positive-integer predicate as the PR_FORCE promotion below: 0 or non-numeric
        # never forces a source build, so it must not demand git.
        if ($_prForce -match '^\d+$' -and [int]$_prForce -gt 0) { $gitNeeded = $true }
        if ($_llamaSrc -ne "https://github.com/ggml-org/llama.cpp") { $gitNeeded = $true }
    }
    if ($gitNeeded -and $StageRoot) {
        Exit-SetupFailure "Background staging cannot install Git; retry with the foreground updater."
    }
    if ($gitNeeded -or -not $StageRoot) {
        Write-StudioLine "Git not found -- attempting install via winget..." -ForegroundColor Yellow
        $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
        if ($HasWinget) {
            try {
                Invoke-SetupCommand { winget install Git.Git --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                Refresh-Environment
                $HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
            } catch { }
        }
    }
    if (-not $HasGit) {
        if ($gitNeeded) {
            Write-StudioLine "[ERROR] Git is required for --local and llama.cpp source-build installs but could not be installed." -ForegroundColor Red
            Write-StudioLine "        --local clones unsloth-zoo, and a source build clones llama.cpp." -ForegroundColor Red
            Write-StudioLine "        Install Git from https://git-scm.com/download/win and re-run." -ForegroundColor Red
            Exit-SetupFailure "Git is required for --local / source-build installs but could not be installed"
        }
        step "git" "not found (not required)" "Yellow"
        substep "Unsloth installs prebuilt binaries and wheels, so git is not needed."
        substep "Install it only for --local/source installs: https://git-scm.com/download/win"
    } else {
        step "git" "$(git --version)"
    }
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
    Write-StudioLine "" -ForegroundColor Red
    Write-StudioLine "========================================================================" -ForegroundColor Red
    Write-StudioLine "[ERROR] CUDA source build cannot use the installed toolkit with this driver." -ForegroundColor Red
    Write-StudioLine "========================================================================" -ForegroundColor Red
    Exit-SetupFailure "The installed CUDA toolkit is incompatible with the current driver"
}

# -- No toolkit at all: install via winget (only when a source build needs it) --
if (-not $NvccPath -and $RequireOrExit) {
    Write-StudioLine "CUDA toolkit (nvcc) not found -- installing via winget..." -ForegroundColor Yellow
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
    Write-StudioLine "[ERROR] CUDA Toolkit (nvcc) is required but could not be found or installed." -ForegroundColor Red
    if ($DriverMaxCuda) {
        Write-StudioLine "        Install a CUDA Toolkit with major version $($DriverMaxCuda.Split('.')[0]) from https://developer.nvidia.com/cuda-toolkit-archive" -ForegroundColor Yellow
    } else {
        Write-StudioLine "        Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
    Exit-SetupFailure "A compatible CUDA Toolkit could not be found or installed"
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

if ($HasROCm -and -not $script:ROCmUnsupportedGfxArch) {
    $rocmVerLabel = if ($script:ROCmVersionFull) { "ROCm $script:ROCmVersionFull" } elseif ($script:ROCmVersion) { "ROCm $script:ROCmVersion" } else { "ROCm (version unknown)" }
    step "rocm" $rocmVerLabel
} elseif ($script:ROCmGfxArch) {
    # GPU training/inference works via AMD's bundled-runtime ROCm PyTorch wheels;
    # the HIP SDK is optional (only the system ROCm toolchain).
    step "rocm" "GPU via bundled ROCm wheels ($script:ROCmGfxArch) -- HIP SDK optional" "Cyan"
} elseif ($script:ROCmUnsupportedGfxArch) {
    # Naming the HIP SDK here would read as "install it and this resolves"; on a
    # generation with no ROCm PyTorch wheels it never does (unslothai#8529).
    step "rocm" "AMD GPU detected ($script:ROCmUnsupportedGfxArch) -- no ROCm PyTorch wheels Unsloth installs" "Yellow"
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


function Test-PackagedFrontend {
    param([string]$LocalInstall, [string]$IndexPath, [string]$ProjectFilePath)
    # install.ps1 and `unsloth studio update` explicitly pass 0 for PyPI
    # installs. Wheel extraction mtimes do not preserve build ordering, so use
    # the release-built dist whenever its entry point is present.
    #
    # $ProjectFilePath is the pyproject.toml beside studio/. The mode records
    # where the Python package came from, not which tree this script runs out
    # of, and an editable overlay separates the two: it leaves the mode at 0
    # while $ScriptDir is a checkout, whose dist is a stale build artifact
    # rather than a release one. A wheel ships no top-level files, so that file
    # existing means source tree -- keep the mtime rebuild there.
    if ($LocalInstall -ne "0") { return $false }
    if ($ProjectFilePath -and (Test-Path -LiteralPath $ProjectFilePath -PathType Leaf)) { return $false }
    return (Test-Path -LiteralPath $IndexPath -PathType Leaf)
}

$SkipFrontend = ($env:SKIP_STUDIO_FRONTEND -eq "1")
$NodeOverride = $null
$NodeParent = $null
$NodeDir = $null
$SysNodeVersion = ""
$SysNpmVersion = ""
$NodeSource = $null

if (-not $IsPipInstall) {
    # Put Node beside the Unsloth root. OXC can still need npm when the
    # frontend build is skipped.
    if ($StageRoot) {
        $NodeParent = $StageRoot
    } else {
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $NodeOverride = $env:UNSLOTH_STUDIO_HOME.Trim() }
        elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $NodeOverride = $env:STUDIO_HOME.Trim() }
        if ($NodeOverride) {
            if ($NodeOverride -eq "~") {
                $NodeOverride = $env:USERPROFILE
            } elseif ($NodeOverride -like "~/*" -or $NodeOverride -like "~\*") {
                $NodeOverride = (Join-Path $env:USERPROFILE $NodeOverride.Substring(1).TrimStart('/', '\'))
            }
            if (-not (Test-Path -LiteralPath $NodeOverride -PathType Container)) {
                Write-StudioLine "ERROR: UNSLOTH_STUDIO_HOME/STUDIO_HOME=$NodeOverride does not exist." -ForegroundColor Red
                Write-StudioLine "       Run install.ps1 to create the install root before 'unsloth studio update'." -ForegroundColor Red
                Exit-SetupFailure "UNSLOTH_STUDIO_HOME/STUDIO_HOME=$NodeOverride does not exist"
            }
            $NodeParent = (Resolve-Path -LiteralPath $NodeOverride).Path
            # legacy default overrides map to ~/.unsloth/node, matching runtime resolution.
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
    Write-StudioLine "Python 3.11-3.13 not found -- installing Python 3.12 via winget..." -ForegroundColor Yellow
    $HasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
    if ($HasWinget) {
        winget install -e --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
        Refresh-Environment
    }
    $HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $HasPython) {
        Write-StudioLine "[ERROR] Python could not be installed automatically." -ForegroundColor Red
        Write-StudioLine "        Install Python 3.12 from https://python.org/downloads/" -ForegroundColor Yellow
        Exit-SetupFailure "Python could not be installed automatically"
    }
    step "python" "$(python --version 2>&1)"
    $PythonOk = $true
} else {
    # python.exe is on PATH but its version is unsupported, and py.exe (if
    # present) had no supported minor either.
    Write-StudioLine "[ERROR] No supported Python (3.11-3.13) found on this system." -ForegroundColor Red
    Write-StudioLine "        py.exe could not locate -3.11/-3.12/-3.13 and `python` on PATH is unsupported." -ForegroundColor Yellow
    Write-StudioLine "        Install Python 3.12 from https://python.org/downloads/" -ForegroundColor Yellow
    Exit-SetupFailure "No supported Python 3.11-3.13 was found"
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

Write-StudioLine ""
step "system" "prerequisites ready"
Write-StudioLine ""

# ==========================================================================
#  PHASE 2: Frontend build (skip if pip-installed -- already bundled)
# ==========================================================================
$DistDir = Join-Path $FrontendDir "dist"
$PackagedFrontend = Test-PackagedFrontend `
    -LocalInstall "$($env:STUDIO_LOCAL_INSTALL)" `
    -IndexPath (Join-Path $DistDir "index.html") `
    -ProjectFilePath (Join-Path $PackageDir "pyproject.toml")
# Wheel extraction mtimes are not a source-freshness signal. Standard PyPI
# installs use the release-built dist; local/source installs retain mtime checks.
# Tauri is checked first so the reported reason matches setup.sh on a desktop
# update, where both this and the packaged branch would otherwise apply.
$NeedFrontendBuild = $true
if ($SkipFrontend) {
    $NeedFrontendBuild = $false
    step "frontend" "bundled (Tauri)"
} elseif ($IsPipInstall -or $PackagedFrontend) {
    $NeedFrontendBuild = $false
    step "frontend" "bundled (pip install)"
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
        substep "found Node='$SysNodeVersion' npm='$SysNpmVersion'; Unsloth needs Node >=20.19/22.12/23 and npm >= 11" "Yellow"
        substep "install a suitable Node + npm, or unset UNSLOTH_SKIP_NODE_INSTALL to let Unsloth manage an isolated Node" "Yellow"
    } elseif ($NodeSource -eq "bundled") {
        New-Item -ItemType Directory -Force -Path $NodeParent -ErrorAction SilentlyContinue | Out-Null
        # Minimal ownership guard for a custom-home dir (the full Unsloth-owned
        # helpers are defined later); never os.replace over a user-owned dir.
        if ($NodeOverride -and (Test-Path -LiteralPath $NodeDir -PathType Container)) {
            $nodeOwnedMarker = Join-Path $NodeDir ".unsloth-studio-owned"
            $nodeMeta = Join-Path $NodeDir "UNSLOTH_NODE_PREBUILT_INFO.json"
            if (-not (Test-Path -LiteralPath $nodeOwnedMarker) -and -not (Test-Path -LiteralPath $nodeMeta)) {
                Write-StudioLine "[ERROR] $NodeDir already exists and is not an Unsloth-owned Node install." -ForegroundColor Red
                Write-StudioLine "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
                Exit-SetupFailure "$NodeDir is not an Unsloth-owned Node install"
            }
        }
        substep "installing isolated Node (system Node/npm left untouched)..."
        # The main Python resolver runs later; bare `python` may be a Store stub or
        # absent this early, so prefer the validated handed-off/venv Python.
        $NodeInstallPython = if ($ValidatedSetupPython) { $ValidatedSetupPython } else { "python" }
        $nodeOut = & $NodeInstallPython "$PSScriptRoot\install_node_prebuilt.py" --install-dir $NodeDir 2>&1 | Out-String
        $nodeExit = $LASTEXITCODE
        if ($nodeExit -eq 3) {
            Write-StudioLine $nodeOut -ForegroundColor DarkGray
            step "node" "install blocked by another active Unsloth install" "Red"
            Exit-SetupFailure "Node install is blocked by another active Unsloth install" 3
        } elseif ($nodeExit -ne 0) {
            Write-StudioLine $nodeOut -ForegroundColor DarkGray
            Write-StudioLine "[ERROR] Could not install an isolated Node automatically." -ForegroundColor Red
            Write-StudioLine "        Install Node >= 20.19 (with npm >= 11) from https://nodejs.org/ and re-run, or check your network." -ForegroundColor Yellow
            Exit-SetupFailure "Could not install an isolated Node runtime"
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
    Write-StudioLine ""
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
        Write-StudioLine "   Using bun for package install (faster)" -ForegroundColor DarkGray
        $bunExit = Invoke-SetupCommand { bun install @NpmRegistryArgs }
        # On Windows, .bin/ entries vary by package manager:
        #   npm  → tsc, tsc.cmd, tsc.ps1
        #   bun  → tsc.exe, tsc.bunx
        $hasTsc = (Test-Path "node_modules\.bin\tsc") -or (Test-Path "node_modules\.bin\tsc.cmd") -or (Test-Path "node_modules\.bin\tsc.exe") -or (Test-Path "node_modules\.bin\tsc.bunx")
        $hasVite = (Test-Path "node_modules\.bin\vite") -or (Test-Path "node_modules\.bin\vite.cmd") -or (Test-Path "node_modules\.bin\vite.exe") -or (Test-Path "node_modules\.bin\vite.bunx")
        if ($bunExit -eq 0 -and $hasTsc -and $hasVite) {
            # bun install succeeded and critical binaries are present
        } elseif ($bunExit -eq 0) {
            Write-StudioLine "   bun install exited 0 but critical binaries are missing, clearing cache and retrying..." -ForegroundColor Yellow
            if (Test-Path "node_modules") {
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
            }
            Invoke-SetupCommand { bun pm cache rm } | Out-Null
            $bunExit = Invoke-SetupCommand { bun install @NpmRegistryArgs }
            $hasTsc = (Test-Path "node_modules\.bin\tsc") -or (Test-Path "node_modules\.bin\tsc.cmd") -or (Test-Path "node_modules\.bin\tsc.exe") -or (Test-Path "node_modules\.bin\tsc.bunx")
            $hasVite = (Test-Path "node_modules\.bin\vite") -or (Test-Path "node_modules\.bin\vite.cmd") -or (Test-Path "node_modules\.bin\vite.exe") -or (Test-Path "node_modules\.bin\vite.bunx")
            if ($bunExit -ne 0 -or -not $hasTsc -or -not $hasVite) {
                Write-StudioLine "   bun retry failed, falling back to npm" -ForegroundColor Yellow
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
            Write-StudioLine "[ERROR] npm install failed (exit code $npmExit)" -ForegroundColor Red
            Write-StudioLine "   Try running 'npm install' manually in frontend/ to see errors" -ForegroundColor Yellow
            Show-NpmRegistryHint
            Exit-SetupFailure "Frontend dependency installation failed (exit code $npmExit)"
        }
    }

    # Always use npm to run the build (Node runtime — avoids bun Windows runtime issues)
    $buildExit = Invoke-SetupCommand { npm run build }
    if ($buildExit -ne 0) {
        Pop-Location
        $ErrorActionPreference = $prevEAP_npm
        foreach ($gi in $HiddenGitignores) { Rename-Item -Path "$gi._twbuild" -NewName (Split-Path $gi -Leaf) -Force -ErrorAction SilentlyContinue }
        Write-StudioLine "[ERROR] npm run build failed (exit code $buildExit)" -ForegroundColor Red
        Exit-SetupFailure "Frontend build failed (exit code $buildExit)"
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
        Write-StudioLine "[ERROR] OXC validator npm install failed (exit code $oxcInstallExit)" -ForegroundColor Red
        Show-NpmRegistryHint
        Exit-SetupFailure "OXC validator dependency installation failed (exit code $oxcInstallExit)"
    }
    Pop-Location
    $ErrorActionPreference = $prevEAP_oxc
    step "oxc runtime" "installed"
} elseif ((Test-Path $OxcValidatorDir) -and $NodeSource -ne "skip") {
    # No npm on PATH (e.g. a pip install with no system Node and no isolated Node
    # provisioned). Skip rather than abort; the runtime resolver degrades. Mirrors setup.sh.
    substep "OXC validator runtime skipped (no npm found); code validation degrades until Node is available" "Yellow"
}

Remove-AgentInstructionFiles -Roots @(
    (Join-Path $FrontendDir "node_modules"),
    (Join-Path $OxcValidatorDir "node_modules")
)

# ==========================================================================
#  PHASE 3: Python environment + dependencies
# ==========================================================================
Write-StudioLine ""
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
    Write-StudioLine "[ERROR] No standalone Python 3.11-3.13 found (conda Python is not supported)." -ForegroundColor Red
    Write-StudioLine "        Install Python from https://python.org/downloads/ or via:" -ForegroundColor Yellow
    Write-StudioLine "        winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    Exit-SetupFailure "No standalone Python 3.11-3.13 was found"
}

substep "Python found: $PythonCmd"

# $StudioHome / $VenvDir are resolved and preflighted before phase 1, so only the
# cache clear stays here. Venv-gated: a writable-but-empty override still fails the
# venv check below, and clearing first would cost the cache for a run that then does
# nothing. Still before any install work.
if (-not $StageRoot -and (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe") -PathType Leaf)) {
    Clear-WebViewCaches
}

# why: in env-override mode $StudioHome is user-chosen; require the
# ownership marker before Remove-Item so unrelated dirs survive. Gated on
# the canonical comparison so an override pointing at the legacy default
# still behaves like a default install.
# Directory-local evidence that Unsloth created $Path, used to adopt a custom-home
# llama.cpp or whisper.cpp predating the .unsloth-studio-owned marker (see
# setup.sh). Only Unsloth prebuilt markers count; source builds are
# indistinguishable from a user clone on Windows and stay under the strict guard.
# "Yes" / "No" / "Denied". A denied marker is not evidence of absence: reading it
# as "No" makes the guard below call an Unsloth tree an unrelated directory and
# tell the user to move it aside, when the real problem is permissions.
function Get-StudioAdoptableState {
    param([Parameter(Mandatory = $true)][string]$Path)
    $denied = $false
    foreach ($marker in @("UNSLOTH_PREBUILT_INFO.json", "UNSLOTH_WHISPER_PREBUILT_INFO.json")) {
        switch (Get-PathState -Path (Join-Path $Path $marker) -PathType Leaf) {
            "Present" { return "Yes" }
            "Denied"  { $denied = $true }
        }
    }
    if ($denied) { return "Denied" }
    # Windows reports a MISSING child of an unreadable directory as absent, so the
    # probes above cannot tell "no markers here" from "cannot look"; listing the
    # directory itself can. Without this a denied tree reads as unowned.
    try { $null = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1) }
    catch {
        if (Test-AccessDeniedError $_) { return "Denied" }
        # Anything else was not adoptable before either; this must not throw.
    }
    return "No"
}
# Boolean view for callers that only gate a cosmetic cleanup on adoption.
function Test-StudioOwnedAdoptable {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ((Get-StudioAdoptableState -Path $Path) -eq "Yes")
}
function Assert-StudioOwnedOrAbsent {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label,
        # whisper.cpp is non-fatal by contract, so it needs the denial handed back
        # rather than exited on. Only this mode returns a value.
        [switch]$NonFatal
    )
    # Denied is not Absent: a root we cannot read cannot be proven ours, and
    # returning here would let the caller replace it. Both stops stay gated on
    # $StudioHomeIsCustom, as before; a default-home denial is reported by the
    # phase that owns the path.
    $pathState = Get-PathState -Path $Path -PathType Container
    if ($pathState -ne "Present") {
        if ($StudioHomeIsCustom -and $pathState -eq "Denied") {
            if ($NonFatal) { return "Denied" }
            Exit-PathAccessDenied -Path $Path -Label $Label -OwnershipUnverified
        }
        return
    }
    $markerState = Get-PathState -Path (Join-Path $Path $StudioOwnedMarker) -PathType Leaf
    if ($StudioHomeIsCustom -and $markerState -eq "Denied") {
        if ($NonFatal) { return "Denied" }
        Exit-PathAccessDenied -Path $Path -Label $Label -OwnershipUnverified
    }
    if ($StudioHomeIsCustom -and $markerState -ne "Present") {
        $adoptState = Get-StudioAdoptableState -Path $Path
        if ($adoptState -eq "Denied") {
            if ($NonFatal) { return "Denied" }
            Exit-PathAccessDenied -Path $Path -Label $Label -OwnershipUnverified
        }
        if ($adoptState -eq "Yes") {
            Mark-StudioOwned $Path
            return
        }
        Write-StudioLine "[ERROR] $Path already exists and is not marked as an Unsloth-owned $Label." -ForegroundColor Red
        Write-StudioLine "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
        Exit-SetupFailure "$Label path is not an Unsloth-owned install: $Path"
    }
}
function Mark-StudioOwned {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    try {
        [System.IO.File]::WriteAllText((Join-Path $Path $StudioOwnedMarker), "")
    } catch {}
}

# The mode this venv was installed with. install.ps1 exports UNSLOTH_NO_TORCH for
# its own run only, so a later `unsloth studio update` (which exports nothing) has
# no other way to know. Two sources, because the completion manifest is dropped
# before every dependency pass and so cannot answer for a run killed mid-pass:
# the manifest key first, then .unsloth-no-torch, which outlives the pass. Neither
# present reads as "install torch" -- the pre-existing behavior.
function Get-PersistedNoTorch {
    param([Parameter(Mandatory = $true)][string]$VenvPath)
    $manifestPath = Join-Path $VenvPath "unsloth_install_manifest.json"
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        $payload = $null
        try {
            $payload = Get-Content -LiteralPath $manifestPath -Raw -ErrorAction Stop | ConvertFrom-Json
        } catch {
            $payload = $null
        }
        if ($null -ne $payload -and $null -ne $payload.no_torch) {
            return ("$($payload.no_torch)" -match '^\s*(?i:true|1|yes|on)\s*$')
        }
    }
    return (Test-Path -LiteralPath (Join-Path $VenvPath $NoTorchMarker) -PathType Leaf)
}

# Written before anything that could be interrupted, and cleared when torch is
# wanted so migrating out of no-torch leaves nothing stale behind.
function Set-PersistedNoTorch {
    param(
        [Parameter(Mandatory = $true)][string]$VenvPath,
        [Parameter(Mandatory = $true)][bool]$NoTorch
    )
    if (-not (Test-Path -LiteralPath $VenvPath -PathType Container)) { return }
    $markerPath = Join-Path $VenvPath $NoTorchMarker
    try {
        if ($NoTorch) {
            [System.IO.File]::WriteAllText($markerPath, "")
        } elseif (Test-Path -LiteralPath $markerPath -PathType Leaf) {
            Remove-Item -LiteralPath $markerPath -Force -ErrorAction Stop
        }
    } catch {}
}

# Stale-venv detection: if the venv exists but its torch flavor no longer
# matches the current machine, repair according to invocation context.
# - install.ps1 sets UNSLOTH_INSTALL_ROLLBACK_MANAGED=1 so setup can delegate
#   to the installer-level rollback that restores the previous environment.
# - direct `unsloth studio update` keeps the pre-existing self-repair behavior.
# In no-torch mode, a missing torch package is expected.
$NoTorchMode = $env:UNSLOTH_NO_TORCH -match '^\s*(?i:true|1|yes|on)\s*$'
# No env var at all means `unsloth studio update` / `studio setup` / setup.bat,
# none of which export one. Without the manifest fallback the check below reads a
# GGUF-only venv's missing torch as a stale venv and tries to delete the venv this
# script is itself running out of, which fails on a locked python.exe.
if (-not $NoTorchMode -and [string]::IsNullOrWhiteSpace($env:UNSLOTH_NO_TORCH)) {
    $NoTorchMode = Get-PersistedNoTorch -VenvPath $VenvDir
    if ($NoTorchMode) {
        substep "no-torch install detected -- keeping this environment GGUF-only." "Yellow"
    }
}
# Persist before the torch install and the dependency pass below, either of which
# can be interrupted; install_python_stack.py refreshes the same marker.
Set-PersistedNoTorch -VenvPath $VenvDir -NoTorch $NoTorchMode
# install_python_stack.py drops the manifest before its dependency pass, so it
# cannot repeat the lookup above; hand it the resolved answer. This also collapses
# every accepted spelling to one value both sides parse identically.
$env:UNSLOTH_NO_TORCH = if ($NoTorchMode) { "true" } else { "false" }
$InstallerManagedSetup = $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -match '^(?i:true|1|yes)$'
# The torch family install.ps1 settled on, in the vocabulary the probe below produces (cu<digits>
# / rocm / xpu / cpu). $null means it did not say: an installer predating the variable (setup.ps1
# ships in the pip package, install.ps1 is fetched from unsloth.ai, so the two can be different
# ages), --no-torch, or a custom index whose leaf names no flavor. Absent is unknown rather than a
# mismatch, which keeps an older cached installer on its pre-variable behaviour. IsNullOrWhiteSpace
# rather than a presence test: 7.5+ keeps an empty assignment as a present blank value, 5.1 and
# 7.0-7.4 remove the variable.
$InstallerTorchTag = if ([string]::IsNullOrWhiteSpace($env:UNSLOTH_INSTALLER_TORCH_TAG)) { $null }
                     else { $env:UNSLOTH_INSTALLER_TORCH_TAG.Trim().ToLowerInvariant() }
# Only the stale-venv block below assigns this, but the XPU install reads it to decide whether to
# force-reinstall and a fresh install never enters that block. Declaring it keeps a caller's
# Set-StrictMode from making the read fatal.
$installedTorchTag = $null
# Hoisted for the same reason: four install arms read it to decide --force-reinstall and the
# installer-managed repair below raises it, yet only the block a fresh install never enters
# assigns it.
$script:PinChangedForceReinstall = $false
$script:TorchImportDefinitivelyFailed = $false
if ((Test-Path -LiteralPath $VenvDir -PathType Container) -and -not $NoTorchMode) {
    $VenvPyExe = Join-Path $VenvDir "Scripts\python.exe"
    $installedTorchTag = $null
    # Declared before the branch that assigns it: the failure message below reads its .Error, and
    # a venv with no python.exe never runs the probe at all.
    $_verProbe = $null
    $shouldRebuild = $false
    # Set when a stale venv under a pin is repaired in place (force-reinstall) not wiped.
    $script:PinChangedForceReinstall = $false

    if (Test-Path -LiteralPath $VenvPyExe) {
        # Bounded like every other torch probe: reading stdout to EOF before waiting would hang
        # setup forever on a wedged `import torch`, and an undrained stderr deadlocks on a noisy
        # import. An unreadable flavor -> rebuild.
        $_verProbe = Invoke-BoundedPythonProbe -PythonExe $VenvPyExe -Code 'import torch; print(torch.__version__)'
        $torchVer = $_verProbe.Output.Trim()
        if ($_verProbe.Ok -and $torchVer) {
            if ($torchVer -match '\+(cu\d+)') {
                $installedTorchTag = $Matches[1]
            } elseif ($torchVer -match '\+rocm') {
                # Any +rocm / gfx wheel -> generic "rocm" flavor (the exact version is
                # repaired later by install_python_stack.py; here we only need the flavor).
                $installedTorchTag = "rocm"
            } elseif ($torchVer -match '\+xpu') {
                # Without this arm a +xpu wheel reads as "cpu" and an Arc venv looks correct
                # while being CPU-only. Matches ConvertTo-TorchFlavorTag.
                $installedTorchTag = "xpu"
            } elseif ($torchVer -match '\+cpu') {
                $installedTorchTag = "cpu"
            } else {
                # Untagged wheel (plain "2.x.y" from PyPI) -> cpu.
                $installedTorchTag = "cpu"
            }
        } elseif (Test-VenvTorchIsXpu -VenvPath $VenvDir) {
            # Bounding this probe made a timeout mean "rebuild", and the host most likely to
            # time out inside `import torch` is an Arc box with a stalled driver -- where
            # version.py still names a good +xpu wheel, and the stale path would DELETE $VenvDir.
            # Trust the disk and warn about the driver; other families keep rebuilding.
            $installedTorchTag = "xpu"
            substep "PyTorch did not respond in time but this venv holds an XPU build -- keeping it." "Yellow"
            substep "If training fails, update the Intel GPU compute driver." "Yellow"
        } elseif (Test-VenvTorchIsRocm -VenvPath $VenvDir) {
            # Same rescue on the AMD side, and the one that has cost users whole installs: a
            # faulted Adrenalin or HIP runtime makes `import torch` raise at the DLL load or never
            # return, so a fine ROCm environment read as "torch could not be imported" and was
            # thrown away, which does not fix a driver. version.py on disk still names the wheel,
            # so trust it and point at the driver instead (#8335, #7275).
            $installedTorchTag = "rocm"
            substep "PyTorch did not respond but this venv holds a ROCm build -- keeping it." "Yellow"
            substep "If training fails, reboot and update the AMD Adrenalin / HIP SDK driver." "Yellow"
        } elseif (Test-VenvTorchIsCuda -VenvPath $VenvDir) {
            # Without this the chain fell through with a NULL tag, so the no-wipe escape
            # below saw no cu* wheel to preserve. Keep the FAMILY, not a generic "cuda".
            $installedTorchTag = Get-VenvTorchCudaTag -VenvPath $VenvDir
            substep "PyTorch did not respond but this venv holds a $installedTorchTag build -- keeping it." "Yellow"
            substep "If training fails, reboot and update the NVIDIA driver." "Yellow"
            # A half-written torch also leaves a +cu* version.py behind, and the matched
            # install below would write a completion manifest over it. Force the reinstall.
            if ($_verProbe -and -not $_verProbe.TimedOut) {
                $script:TorchImportDefinitivelyFailed = $true
                substep "PyTorch failed to import rather than timing out -- reinstalling the same family in place." "Yellow"
            }
        } else {
            $shouldRebuild = $true
        }
    } else {
        # Missing python.exe means the venv is incomplete -- rebuild it.
        $shouldRebuild = $true
    }

    if (-not $shouldRebuild) {
        # The scan above only matches a WMI marketing name, so a throwing Get-CimInstance -- or
        # an Intel part outside the Arc|Data Center regex -- leaves $script:IsIntelXpu false, and
        # the chain below would expect "cpu", see "xpu" and WIPE a working XPU venv.
        # torch.xpu.is_available() on a +xpu wheel proves the host, so re-evaluate before the
        # stale decision. This repeats the scan's own gate, so NVIDIA / AMD keep winning.
        if ($installedTorchTag -eq "xpu" -and -not $script:IsIntelXpu -and
            -not $HasNvidiaSmi -and -not $AmdHasGpuWheels -and
            (Test-TorchXpuAvailable -PythonExe $VenvPyExe)) {
            $script:IsIntelXpu = $true
            substep "Intel XPU runtime confirmed by PyTorch -- keeping this XPU environment." "Cyan"
        }
        $_pinnedIdx = Get-PinnedTorchIndexUrl
        $_expectedKnown = $true
        if ($_pinnedIdx) {
            $_pinLeaf = Get-TorchIndexLeaf $_pinnedIdx
            # Digit-gated like the install selection: a custom rocm-* leaf (rocm-current /
            # rocm-rel-7.2.1) is NOT a ROCm family and must not be stale-compared.
            if (Test-PipRocmFamilyLeaf $_pinLeaf) {
                # Don't collapse a pinned ROCm/gfx leaf to a generic "rocm" (would mask a family
                # change, rocm6.4 -> gfx1151). Get-RocmPinStaleTags uses the SAME 2.11 allowlist
                # as the install path, so a gfx110X-all/gfx90a/gfx908 pin on a <2.11 wheel is NOT stale.
                $_rocmTags = Get-RocmPinStaleTags -PinLeaf $_pinLeaf -TorchVersion $torchVer
                $expectedTorchTag  = $_rocmTags.Expected
                $installedTorchTag = $_rocmTags.Installed
            } elseif ((Test-CudaFamilyLeaf $_pinLeaf) -or $_pinLeaf -eq 'cpu' -or $_pinLeaf -eq 'xpu') {
                # cu*/cpu/xpu leaves stay specific so a cu126-vs-cu128 (or cpu-vs-xpu) mismatch
                # is caught; /custom and /current fall through to the unknown-index branch below.
                $expectedTorchTag = $_pinLeaf
            } else {
                # Custom index whose leaf is not a torch flavor (a /simple mirror): the
                # flavor can't be inferred, so never treat the venv as stale over it.
                $_expectedKnown = $false
                $expectedTorchTag = $installedTorchTag
            }
        } elseif ($HasNvidiaSmi) {
            $expectedTorchTag = Get-PytorchCudaTag
        } elseif ($script:IsIntelXpu) {
            # Arc / Data Center host: the install below selects the xpu index. BEFORE the AMD
            # arm -- both can be true on an unmapped-arch AMD box, where xpu is what gets
            # installed. A CPU wheel is not wiped (the xpu install force-reinstalls over it in
            # place), so expect "cpu" there; a cu*/rocm wheel still rebuilds.
            $expectedTorchTag = if ($installedTorchTag -eq "cpu") { "cpu" } else { "xpu" }
        } elseif ($HasROCm -or $script:ROCmGfxArch) {
            # AMD/ROCm host with no explicit pin: an existing +rocm wheel is correct (gfx arch
            # counts even when $HasROCm is false). But only the arches the install path maps to a
            # repo.amd.com index get ROCm torch; an unmapped arch installs CPU, so expect "cpu"
            # for those or a correct CPU venv rebuilds every update.
            # $_rocmWheelArches is defined above the Intel scan (that gate needs it too).
            if ($AmdHasGpuWheels) {
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

    # A stale venv under a pin whose torch still imports is repaired IN PLACE (the dependency
    # pass force-reinstalls from the pin). The rebuild path wipes the venv and would strand a
    # direct `studio update`; only a broken venv or an unpinned drift wipes.
    if ($shouldRebuild -and $_pinnedIdx -and $installedTorchTag) {
        substep "Torch-index pin changed ($installedTorchTag) -- reinstalling torch from the pin in place." "Cyan"
        $script:PinChangedForceReinstall = $true
        $shouldRebuild = $false
    }
    # Same for an unpinned cu* -> cu* move: the cap can change the expected family on a
    # healthy venv, and only install.ps1 creates venvs, so wiping here strands a direct
    # `studio update`. CPU/ROCm/XPU drift still rebuilds.
    if ($shouldRebuild -and -not $_pinnedIdx -and $installedTorchTag -and
        (Test-CudaFamilyLeaf $installedTorchTag) -and (Test-CudaFamilyLeaf $expectedTorchTag)) {
        substep "CUDA family $installedTorchTag does not cover this host -- reinstalling $expectedTorchTag in place." "Cyan"
        $script:PinChangedForceReinstall = $true
        $shouldRebuild = $false
    }
    # A +xpu venv is never wiped by a DIRECT update: on a hybrid NVIDIA+Arc box the promotion
    # above is gated on -not $HasNvidiaSmi, so a later pinless update expects a cu* tag, calls the
    # working Arc venv stale and deletes it -- then exits, because only install.ps1 creates venvs.
    # Under install.ps1 the in-place repair below covers every flavour, so this escape stays out.
    if ($shouldRebuild -and -not $InstallerManagedSetup -and $installedTorchTag -eq "xpu") {
        substep "Keeping the installed Intel XPU environment (this host expects $expectedTorchTag)." "Yellow"
        substep "Re-run install.ps1 to replace it -- that path rebuilds with a rollback copy." "Yellow"
        $shouldRebuild = $false
        # Keeping the venv is only half the job: the index selection below prefers NVIDIA, and
        # the CUDA arm does NOT --reinstall-package torch, so uv would keep the +xpu wheel as
        # satisfied while installing triton-windows over torch's XPU triton, with $XpuIndexUrl
        # null and nothing to swap it back. So the whole pass stays on the xpu index.
        $script:PreservedXpuVenv = $true
    }

    # A GPU wheel install.ps1 SELECTED is never force-reinstalled onto a different family on the
    # strength of a rescan. Only a wheel matching the family install.ps1 reported in
    # $env:UNSLOTH_INSTALLER_TORCH_TAG counts as its answer, because "there is a GPU wheel in the
    # venv" does not by itself mean this run put it there: the migrated-venv arm (install.ps1's
    # `if ($_Migrated)`) installs unsloth alone and never touches torch, and install.ps1's flavor
    # repair no-ops whenever its expected tag is 'cpu' or unrecognised. So an ordinary upgrade off
    # the legacy ~/.unsloth/studio/.venv layout can hand setup a +cu118 wheel from a previous
    # install on different hardware; preserving THAT would leave the environment permanently wrong
    # and, on a mapped AMD host, the kept cu* tag also blocks the ROCm reroute below (it needs
    # $CuTag -eq "cpu"). Those repair in place, as they did before this guard existed.
    #
    # $InstallerTorchTag $null means the installer did not say (an older cached install.ps1, or
    # --no-torch), and unknown falls back to preserving, the behaviour that shipped before the
    # variable existed. Otherwise the disagreement is between setup's second probe and the first,
    # not with the hardware -- a Get-CimInstance that threw, an nvidia-smi that did not answer, the
    # single-Radeon unroll of #8335 -- and setup has no better claim than the installer that just
    # ran: the in-place repair below would --force-reinstall the other family over a working GPU
    # environment and exit 0, at which point install.ps1 discards the rollback copy and the damage
    # is permanent. A loud failure traded for a silently wrong install is worse than the loop this
    # block exists to end.
    #
    # Covers cpu rescans (+rocm -> cpu) and family-to-family ones (+rocm -> cu128, +cu128 -> rocm,
    # +xpu -> cu128) alike. Nothing legitimate is suppressed: a venv whose torch does not import at
    # all leaves $installedTorchTag $null and still repairs, a CPU wheel that has to become a GPU
    # one still repairs, cu* -> cu* already repaired in place above, and an explicit index pin
    # escaped before that. A genuine GPU swap is install.ps1's job, done before calling here.
    #
    # $expectedTorchTag is read in the MESSAGE, never in the condition: it is assigned only inside
    # the `if (-not $shouldRebuild)` block above, so on a venv whose torch would not import it was
    # never created and reading it under a caller's Set-StrictMode is fatal. The body is reached
    # only once $installedTorchTag has answered, which is only true on the path that assigned it.
    if ($shouldRebuild -and $InstallerManagedSetup -and
        $installedTorchTag -and $installedTorchTag -ne "cpu" -and
        ((-not $InstallerTorchTag) -or $installedTorchTag -eq $InstallerTorchTag)) {
        substep "This host rescanned as $expectedTorchTag but the installer just placed a $installedTorchTag build here -- keeping it." "Yellow"
        substep "Set UNSLOTH_TORCH_INDEX_URL to move this environment onto another PyTorch build on purpose." "DarkGray"
        $shouldRebuild = $false
        # Keeping the wheel is only half the job: the index selection below re-runs the same
        # rescan, and two of its arms force-reinstall torch regardless of
        # $script:PinChangedForceReinstall (the AMD arm always, the XPU arm whenever the installed
        # tag is not xpu), undoing this guard from a thousand lines further down.
        $script:PreservedInstallerTorchTag = $installedTorchTag
        # Same reason as the direct-update escape above, and it needs its own flag: stay on the
        # xpu index, or triton-windows lands over torch's XPU triton with nothing to swap it back.
        if ($installedTorchTag -eq "xpu") { $script:PreservedXpuVenv = $true }
    }

    $reason = $null
    if ($shouldRebuild) {
        $reason = if ($installedTorchTag) { "torch $installedTorchTag != required $expectedTorchTag" } else { "torch could not be imported" }
        # "torch could not be imported" covers a dead GPU driver, a half-written wheel and no torch
        # at all. Print what python actually said (the last stderr line is the exception), so a
        # WinError 126 reads as a driver problem instead of a broken install.
        if ($_verProbe -and -not $_verProbe.Ok -and $_verProbe.Error) {
            # No @(...)[0] around this: the guard above passes on a whitespace-only stderr,
            # Where-Object then drops every line, and [0] into the empty array that leaves is fatal
            # under a caller's Set-StrictMode. -Last 1 already yields one string or nothing.
            $_probeErrLine = $_verProbe.Error -split "`r?`n" |
                Where-Object { $_.Trim() } | Select-Object -Last 1
            if ($_probeErrLine) { substep "PyTorch reported: $($_probeErrLine.Trim())" "DarkGray" }
        }
    }

    # The abort that used to live here was an unrecoverable fixed point, reported from four
    # separate triggers (#5942, #7275, #8335, plus a driver crash). It told the user to re-run
    # install.ps1 for a safe rollback replace -- but install.ps1 IS the caller under
    # $InstallerManagedSetup, it had already done exactly that earlier in the same run, and its
    # failure path moves the previous environment straight back, so every attempt ended on the
    # byte-identical state it started from and the install could never converge.
    #
    # Repair in place instead of wiping: the dependency pass force-reinstalls the torch trio from
    # the resolved index over this venv, the route an index-pin change and a cu* family change
    # already take above. Deleting is not an option here anyway -- install.ps1 invokes setup
    # through the venv's own python.exe, which is therefore locked by the process running this.
    if ($shouldRebuild -and $InstallerManagedSetup) {
        substep "Environment does not match this host ($reason) -- reinstalling PyTorch in place." "Yellow"
        substep "install.ps1 keeps a rollback copy of the previous environment until this run succeeds." "DarkGray"
        $script:PinChangedForceReinstall = $true
        $shouldRebuild = $false
    }

    # A cu* venv is never wiped by a DIRECT update just because nvidia-smi did not answer:
    # every way that bounded probe comes back empty on a working NVIDIA host collapses
    # $expectedTorchTag to "cpu", no escape above catches it, and the wipe has no rollback
    # copy. Narrow on purpose: unpinned only, cu* installed only, and only when the collapse
    # was for want of an NVIDIA answer. The -and order also keeps the reads legal under
    # Set-StrictMode, since both are assigned inside `if (-not $shouldRebuild)`.
    if ($shouldRebuild -and -not $InstallerManagedSetup -and
        $installedTorchTag -and (Test-CudaFamilyLeaf $installedTorchTag) -and
        -not $_pinnedIdx -and -not $HasNvidiaSmi -and $expectedTorchTag -eq "cpu") {
        substep "nvidia-smi did not answer, but this venv holds a $installedTorchTag build -- keeping it." "Yellow"
        substep "If training runs on CPU, re-run install.ps1: irm https://unsloth.ai/install.ps1 | iex" "Yellow"
        $shouldRebuild = $false
        # Keeping the wheel is half the job: the index selection below rescans, sees no
        # NVIDIA either, and would route the install to the /cpu arm.
        $script:PreservedInstallerTorchTag = $installedTorchTag
    }

    if ($shouldRebuild) {
        substep "Stale venv detected ($reason) -- rebuilding..." "Yellow"
        # why: mirror install.ps1 env-mode guard so an update against a custom
        # UNSLOTH_STUDIO_HOME never wipes an unrelated unsloth_studio venv;
        # -PathType Leaf rejects a directory masquerading as the sentinel.
        # The .cmd counts too, and for the same reason the uninstaller accepts it: a
        # policy's quarantine can take the unsigned .exe and leave a root that is still
        # ours. Content-checked, never by name -- this guard gates a recursive delete.
        if (
            $StudioHomeIsCustom -and
            -not (Test-Path -LiteralPath (Join-Path $VenvDir $StudioOwnedMarker) -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "share\studio.conf") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe") -PathType Leaf) -and
            -not (Test-UnslothCmdShimFile (Join-Path $StudioHome "bin\unsloth.cmd"))
        ) {
            Write-StudioLine "[ERROR] $VenvDir already exists but does not look like an Unsloth Studio install." -ForegroundColor Red
            Write-StudioLine "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." -ForegroundColor Yellow
            Exit-SetupFailure "$VenvDir is not an Unsloth Studio environment"
        }
        try {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction Stop
        } catch {
            Write-StudioLine "   [ERROR] Could not remove stale venv: $($_.Exception.Message)" -ForegroundColor Red
            Write-StudioLine "           Close any running Unsloth/Python processes and re-run setup." -ForegroundColor Red
            Exit-SetupFailure "Could not remove the stale environment at $VenvDir"
        }
    }
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-StudioLine "[ERROR] Virtual environment not found at $VenvDir" -ForegroundColor Red
    Write-StudioLine "        Run install.ps1 first to create the environment:" -ForegroundColor Yellow
    Write-StudioLine "        irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
    Exit-SetupFailure "Virtual environment not found at $VenvDir"
} else {
    substep "reusing existing virtual environment at $VenvDir"
    $_venvPyExe = Join-Path $VenvDir "Scripts\python.exe"
    $_venvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path -LiteralPath $_venvPyExe) {
        # The interpreter is not the only file the rest of this script needs. Everything below
        # reaches the venv through the dot-sourced Activate.ps1 and a bare `python` (Fast-Install
        # resolves its target with (Get-Command python).Source), and install.ps1 deliberately
        # leaves the venv's Scripts directory off PATH. So a venv that kept python.exe but lost
        # Activate.ps1 fails the dot-source non-terminatingly at the "Continue" the pip section
        # runs at, installs the whole stack into whatever interpreter is on PATH, and exits 0.
        # Newly reachable now that an installer-managed stale verdict repairs instead of aborting.
        if (-not (Test-Path -LiteralPath $_venvActivate)) {
            Write-StudioLine "[ERROR] $VenvDir has no activation script at Scripts\Activate.ps1." -ForegroundColor Red
            Write-StudioLine "        The environment is incomplete rather than out of date. Re-run the installer" -ForegroundColor Yellow
            Write-StudioLine "        to rebuild it: irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
            Exit-SetupFailure "No activation script at $_venvActivate"
        }
        try {
            $_venvPyVer = (& $_venvPyExe --version 2>&1 | Out-String).Trim()
            if ($_venvPyVer) { substep $_venvPyVer }
        } catch {}
    } else {
        # Stop here, because nothing downstream would: the activation below is a dot-source, and a
        # MISSING script is a NON-terminating error at the "Continue" the pip section runs at, so
        # setup would print one red line and carry on, resolving every `python` and `uv pip` that
        # follows against whatever interpreter is on PATH, installing the whole stack outside the
        # environment it was asked to build, and exiting 0 over a venv with not one package in it.
        #
        # An interpreter-less venv is incomplete, not out of date, so none of the decisions above
        # apply: the in-place torch repair has no interpreter to repair through, and the rebuild
        # path deletes the directory only to hit "Virtual environment not found" a few lines up.
        # Unlike the abort this replaced upstream, re-creating the environment genuinely changes
        # this state -- install.ps1 still holds the rollback copy of the previous one.
        Write-StudioLine "[ERROR] $VenvDir has no interpreter at Scripts\python.exe." -ForegroundColor Red
        Write-StudioLine "        The environment is incomplete rather than out of date. Re-run the installer" -ForegroundColor Yellow
        Write-StudioLine "        to rebuild it: irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
        Exit-SetupFailure "No interpreter at $_venvPyExe"
    }
}

# pip and python write to stderr even on success (progress bars, warnings).
# With $ErrorActionPreference = "Stop" (set at top of script), PS 5.1
# converts stderr lines into terminating ErrorRecords, breaking output.
# Lower to "Continue" for the pip/python section.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# Existence is not activation. The two refusals above check that a file is THERE; neither can tell
# whether dot-sourcing it took effect, and the ways it silently does not are ordinary damage on a
# half-written venv. Activate.ps1 prepends the venv to PATH in its LAST statement, 28 lines after
# it sets $env:VIRTUAL_ENV, so a copy truncated by an interrupted or out-of-disk `python -m venv`
# parses, runs to its last complete statement and returns without printing anything at all, and an
# unparseable one is a ParserError, non-terminating at the "Continue" set just above. Either way
# the dot-source "succeeds" with the ambient interpreter still first on PATH -- the system or
# Microsoft Store python, since install.ps1 keeps the venv's Scripts directory off PATH on purpose.
# Fast-Install hands exactly that to `uv pip install --python`, the whole stack lands outside the
# venv, every Exit-SetupFailure below keys off that wrong interpreter's exit code, and setup exits
# 0 -- at which point install.ps1 commits over the rollback copy and the previous working
# environment is gone for good.
#
# So assert the post-condition instead of a third pre-condition: the `python` now in effect has to
# live under $VenvDir. $env:VIRTUAL_ENV would not do, precisely because Activate.ps1 sets it
# before the line that matters.
function Assert-VenvActivated {
    param([Parameter(Mandatory = $true)][string]$VenvDir)

    # Both sides normalised through Get-Item, which is what Activate.ps1 itself uses to build the
    # PATH entry ($VenvExecDir = Get-Item -Path $VenvExecPath): agreeing on the normaliser keeps a
    # short 8.3 path, a substituted drive, a junction or a differently cased drive letter from
    # reading as "outside". A guard that false-positives here would break every install, so every
    # branch that cannot PROVE the interpreter is wrong returns.
    $venvRoot = $null
    try { $venvRoot = (Get-Item -LiteralPath $VenvDir -Force -ErrorAction Stop).FullName.TrimEnd('\', '/') } catch { $venvRoot = $null }
    if (-not $venvRoot) { return }

    $_pyCmd = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
    # A function or alias named python carries no Source to judge, and judging it wrong would
    # refuse an install that works. Only an application resolved off PATH is decidable, and that
    # is also the only shape Fast-Install's -- python hand-off can be aimed at.
    if ($_pyCmd -and $_pyCmd.CommandType -ne 'Application') { return }
    $_pyPath = $null
    if ($_pyCmd -and $_pyCmd.Source) {
        try { $_pyPath = (Get-Item -LiteralPath $_pyCmd.Source -Force -ErrorAction Stop).FullName } catch { $_pyPath = $_pyCmd.Source }
    }

    if ($_pyPath) {
        foreach ($_sep in @('\', '/')) {
            if ($_pyPath.StartsWith($venvRoot + $_sep, [System.StringComparison]::OrdinalIgnoreCase)) { return }
        }
    }

    $_where = if ($_pyPath) { $_pyPath } else { "nothing on PATH" }
    Write-StudioLine "[ERROR] Activating $VenvDir did not take effect: python resolves to $_where." -ForegroundColor Red
    Write-StudioLine "        The activation script is present but did not put the environment on PATH," -ForegroundColor Yellow
    Write-StudioLine "        so the environment is incomplete rather than out of date. Re-run the installer" -ForegroundColor Yellow
    Write-StudioLine "        to rebuild it: irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Yellow
    Exit-SetupFailure "Activating $VenvDir did not put its interpreter on PATH (python resolves to $_where)"
}

# Mirrors install.ps1's Install-UvFromRelease: same archive, destination priority and user-PATH
# prepend as astral's installer, but it fetches a data file with a pinned SHA-256 instead of
# running remote script text in-process, which is what AMSI scores hardest. Bumping the version
# means bumping all 3 hashes:
#   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/uv-<arch>-pc-windows-msvc.zip.sha256
$UvPinnedVersion = "0.12.1"
$UvPinnedAssets = @{
    "x86_64" = @{ Asset = "uv-x86_64-pc-windows-msvc.zip";  Sha256 = "8FCB0CB46E1229065E344758980924E569BEF5882EF45F46FADA8FB24E06B74A" }
    "arm64"  = @{ Asset = "uv-aarch64-pc-windows-msvc.zip"; Sha256 = "9BC7C18E616230FA2DC6FB24BC3AFDE18A95C2B5C9433DE747E9502C66041568" }
    "x86"    = @{ Asset = "uv-i686-pc-windows-msvc.zip";    Sha256 = "9B51C33D307A8AB9E9DFD88D4AE1491761F63DE0BFFA3CEC96BEC536491C9B97" }
}

# Not Get-HostMachineArch: it answers arm64/other for the VC++ and prebuilt probes, and "other"
# cannot pick between the x86_64 and i686 archives. install.ps1's resolution order.
function Get-UvHostArch {
    $osArch = ""
    try { $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() } catch { $osArch = "" }
    $signals = @([string]$env:PROCESSOR_ARCHITEW6432, [string]$env:PROCESSOR_ARCHITECTURE, $osArch)
    foreach ($s in $signals) {
        if ($s.ToLowerInvariant() -eq "arm64") { return "arm64" }
    }
    foreach ($s in $signals) {
        if ([string]::IsNullOrWhiteSpace($s)) { continue }
        switch ($s.ToLowerInvariant()) {
            "amd64" { return "x86_64" }
            "x64" { return "x86_64" }
            "x86" { return "x86" }
        }
    }
    return "unknown"
}

# Writes to the pipeline, not the console: under Invoke-SetupCommand a quiet run swallows this
# exactly as it swallowed astral's output, and a verbose run shows it. The console lines around
# the call site are unchanged.
function Get-SetupUvExecutableVerdict {
    # Mirrors Get-UvExecutableVerdict in install.ps1: "ok", "failed" or "unknown". Only the
    # binary answering non-zero is "failed"; a launch that throws or a wait that times out got
    # no verdict, and the digest already proved the bytes are astral's pinned release.
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return "failed" }
    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath $Path -ArgumentList "--version" -NoNewWindow -PassThru `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile -ErrorAction Stop
        if (-not $proc.WaitForExit(20000)) {
            try { $proc.Kill() } catch {}
            Write-Output "uv did not answer --version within 20s; installing it unprobed."
            return "unknown"
        }
        # The timed overload can return before the exit code is cached, which is how
        # arm64 and the Windows containers reported an EMPTY code and had a working uv
        # read as broken. The parameterless wait settles it and returns at once, since
        # the process has already exited. No code at all is still no verdict.
        try { $proc.WaitForExit() } catch {}
        $code = $null
        try { $code = $proc.ExitCode } catch {}
        if ($null -eq $code -or "$code" -eq "") {
            Write-Output "uv --version gave no exit code; installing it unprobed."
            return "unknown"
        }
        if ($code -eq 0) { return "ok" }
        $detail = ""
        try {
            $detail = Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue
        } catch {}
        if ($detail) { $detail = " " + (($detail.Trim()) -replace '\s+', ' ') }
        Write-Output "uv --version exited $code.$detail"
        return "failed"
    } catch {
        Write-Output "could not probe uv: $($_.Exception.Message); installing it unprobed."
        return "unknown"
    } finally {
        Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Install-UvFromPinnedRelease {
    $arch = Get-UvHostArch
    if (-not $UvPinnedAssets.ContainsKey($arch)) {
        Write-Output "No uv build is published for this architecture ($arch)."
        return $false
    }
    $asset  = $UvPinnedAssets[$arch].Asset
    $wanted = $UvPinnedAssets[$arch].Sha256

    # astral's destination priority, so an existing uv is replaced in place and the Get-Command
    # probe after Refresh-Environment still finds it.
    $destDir = $null
    foreach ($candidate in @($env:UV_INSTALL_DIR, $env:UV_UNMANAGED_INSTALL, $env:XDG_BIN_HOME)) {
        if ($candidate) { $destDir = $candidate; break }
    }
    if (-not $destDir -and $env:XDG_DATA_HOME) { $destDir = Join-Path $env:XDG_DATA_HOME "../bin" }
    if (-not $destDir) {
        $userHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
        if (-not $userHome) {
            Write-Output "Could not determine a home directory to install uv into."
            return $false
        }
        $destDir = Join-Path $userHome ".local\bin"
    }

    # astral's sources in astral's order, each exclusive when set. UV_DOWNLOAD_URL (and its older
    # alias INSTALLER_DOWNLOAD_URL) outrank the mirror variables there, and a host that sets one
    # usually cannot reach the public endpoints at all, so trying those first would stall. The pin
    # still applies: a source serving a different build fails the digest and the caller falls back.
    $uvBase = if ($env:UV_DOWNLOAD_URL) {
        @("$($env:UV_DOWNLOAD_URL.TrimEnd('/'))")
    } elseif ($env:INSTALLER_DOWNLOAD_URL) {
        @("$($env:INSTALLER_DOWNLOAD_URL.TrimEnd('/'))")
    } elseif ($env:UV_INSTALLER_GHE_BASE_URL) {
        @("$($env:UV_INSTALLER_GHE_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
    } elseif ($env:UV_INSTALLER_GITHUB_BASE_URL) {
        @("$($env:UV_INSTALLER_GITHUB_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
    } else {
        @("https://releases.astral.sh/github/uv/releases/download/$UvPinnedVersion",
          "https://github.com/astral-sh/uv/releases/download/$UvPinnedVersion")
    }

    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-uv-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    $zip  = Join-Path $work $asset
    try {
        [System.IO.Directory]::CreateDirectory($work) | Out-Null
        # Digest per mirror, as install.ps1 does: a proxy answering 200 with its own body is a
        # successful download by every measure Invoke-WebRequest has.
        $downloaded = $false
        foreach ($base in $uvBase) {
            Write-Output "downloading uv $UvPinnedVersion ($arch) from $base..."
            try {
                Invoke-WebRequest -UseBasicParsing -OutFile $zip -Uri "$base/$asset"
            } catch {
                Write-Output "uv download failed: $($_.Exception.Message)"
                continue
            }
            $actual = ""
            try { $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash } catch {}
            if ($actual -eq $wanted) {
                $downloaded = $true
                break
            }
            Write-Output "uv download failed checksum verification -- discarding it."
            Write-Output "expected $wanted, got $actual"
            Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
        }
        if (-not $downloaded) { return $false }

        # The Windows archives are flat: uv.exe, uvx.exe, uvw.exe at the root.
        Expand-Archive -LiteralPath $zip -DestinationPath $work -Force
        [System.IO.Directory]::CreateDirectory($destDir) | Out-Null
        $stagedUv = Join-Path $work "uv.exe"
        if (-not (Test-Path -LiteralPath $stagedUv)) {
            Write-Output "uv.exe was not present in $asset."
            return $false
        }
        # Run it where it landed, before the destination is touched: a host can have a working
        # older uv while a policy refuses this one, and copying first leaves it with neither.
        if ((Get-SetupUvExecutableVerdict -Path $stagedUv) -eq "failed") {
            Write-Output "the downloaded uv $UvPinnedVersion could not run on this machine."
            return $false
        }

        # uvw.exe is the windowless launcher and has no console to answer a probe on, so the
        # staged uv.exe above stands for the set: it came from the same verified archive.
        # Copy-Item under Stop so a locked or ACL-denied destination fails the install rather
        # than leaving half a set behind quietly.
        $haveUv = $true
        foreach ($exe in @("uv.exe", "uvx.exe", "uvw.exe")) {
            $src = Join-Path $work $exe
            if (-not (Test-Path -LiteralPath $src)) { continue }
            $dst = Join-Path $destDir $exe
            try {
                Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
            } catch {
                $haveUv = $false
                break
            }
            if ($exe -eq "uv.exe") {
                # Invoke-SetupCommand sets ErrorActionPreference to Continue, so compare
                # against the archive we verified: a stale uv.exe must not pass for ours.
                $copied = $false
                try {
                    $copied = (Test-Path -LiteralPath $dst) -and
                        (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash -eq
                        (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
                } catch { $copied = $false }
                if (-not $copied) { $haveUv = $false; break }
            }
        }
        if (-not $haveUv) {
            Write-Output "uv.exe was not present in $asset."
            return $false
        }
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }

    # astral's PATH treatment and opt-outs: an unmanaged install forces no-modify-path there, so
    # it must here too. The user-PATH prepend is what survives the Refresh-Environment below.
    if (-not $env:UV_NO_MODIFY_PATH -and -not $env:UV_UNMANAGED_INSTALL) {
        Add-ToUserPath -Directory $destDir -Position Prepend | Out-Null
    }
    $env:PATH = "$destDir;$env:PATH"
    # Recorded on the script scope as well as returned: the caller runs this through
    # Invoke-SetupCommand, which hands back [int]$LASTEXITCODE rather than the pipeline value,
    # so the return alone cannot tell the fallback whether to run.
    $script:UvPinnedInstalled = $true
    return $true
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
function Enter-StudioVenv {
    if ($StageRoot) {
        $env:VIRTUAL_ENV = $VenvDir
        $env:PATH = (Join-Path $VenvDir "Scripts") + ";" + $env:PATH
        Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
        return
    }
    . $ActivateScript
}
Enter-StudioVenv
Assert-VenvActivated -VenvDir $VenvDir

# Try to use uv (much faster than pip), fall back to pip if unavailable
$UseUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $UseUv = $true
} elseif (-not $StageRoot) {
    substep "installing uv package manager..."
    try {
        $script:UvPinnedInstalled = $false
        Invoke-SetupCommand { Install-UvFromPinnedRelease } | Out-Null
        # The merge base ran astral's installer here, so a failed pinned install needs somewhere
        # to go: with no fallback the setup drops to pip for torch, bitsandbytes and Triton, which
        # is a different resolver rather than a different download. winget, not the remote script,
        # which is the shape this branch removes and is what install.ps1 already tries first.
        if (-not $script:UvPinnedInstalled -and (Get-Command winget -ErrorAction SilentlyContinue)) {
            Invoke-SetupCommand {
                winget install --id astral-sh.uv --source winget --accept-source-agreements `
                    --accept-package-agreements --silent
            } | Out-Null
        }
        Refresh-Environment
        # Re-activate venv since Refresh-Environment rebuilds PATH from
        # registry and drops the venv's Scripts directory
        Enter-StudioVenv
        if (Get-Command uv -ErrorAction SilentlyContinue) { $UseUv = $true }
    } catch { }
}
# Refresh-Environment rebuilt PATH from the registry, and the re-activation meant to put the venv
# back sits inside a catch that swallows everything -- including a dot-source that died after
# Activate.ps1's own `deactivate -nondestructive` had restored the pre-venv PATH. Re-check outside
# the catch: this is the last statement before Fast-Install starts resolving `python`.
Assert-VenvActivated -VenvDir $VenvDir

# Helper: install a package, preferring uv with pip fallback
function Fast-Install {
    param([Parameter(ValueFromRemainingArguments=$true)]$Args_)
    # An explicit --index-url must win: inherited uv index vars otherwise pull CPU torch over
    # the CUDA/ROCm build (#6898), so drop them for pinned installs (scrub covers the whole
    # function since the pip fallback honours PIP_* too). UV_TORCH_BACKEND / UV_FIND_LINKS also
    # reroute; UV_NO_CONFIG=1 (+ dropping UV_CONFIG_FILE) stops a uv.toml index outranking the
    # pin (uv 0.10); PIP_NO_INDEX / PIP_INDEX_URL would defeat the pinned --index-url in pip.
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

# uv first, pip as the fallback -- the shape install.sh uses to prune a dependency-pulled wheel.
# No index scrub: an uninstall reads no index.
function Fast-Uninstall {
    param([Parameter(ValueFromRemainingArguments=$true)]$Args_)
    if ($UseUv) {
        $VenvPy = (Get-Command python).Source
        & uv pip uninstall --python $VenvPy @Args_ 2>&1
        if ($LASTEXITCODE -eq 0) { return }
    }
    & python -m pip uninstall -y @Args_ 2>&1
}

# Fetch a wheel without installing it, so a destructive step can be staged behind a download.
# pip only: uv has no `pip download` (astral-sh/uv#3163). Scrubbed like Fast-Install, minus the
# UV_* pip cannot read: an inherited PIP_INDEX_URL or user pip.conf would outrank --index-url.
function Fast-Download {
    param([Parameter(ValueFromRemainingArguments=$true)]$Args_)
    $saved = @{}
    foreach ($n in 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_INDEX_URL', 'PIP_CONFIG_FILE') {
        $saved[$n] = [Environment]::GetEnvironmentVariable($n)
        Remove-Item "Env:$n" -ErrorAction SilentlyContinue
    }
    $env:PIP_CONFIG_FILE = 'nul'
    try {
        & python -m pip download @Args_ 2>&1
    }
    finally {
        Remove-Item "Env:PIP_CONFIG_FILE" -ErrorAction SilentlyContinue
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
    $_InstalledVersionProbeExit = 1
    $InstalledVer = try {
        $_installedVersionOutput = & python -c "
import sys
sys.path.insert(0, sys.argv[2])
import install_manifest
version, conflict = install_manifest.installed_version_probe(sys.argv[1], ('unsloth-zoo',))
print(version)
sys.exit(2 if conflict else (0 if version else 1))
" $_PkgName $PSScriptRoot 2>$null
        $_InstalledVersionProbeExit = $LASTEXITCODE
        ($_installedVersionOutput | Out-String).Trim()
    } catch { "" }
    $LatestVer = ""
    try {
        $pypiJson = Invoke-RestMethod -Uri "https://pypi.org/pypi/$_PkgName/json" -TimeoutSec 5 -ErrorAction Stop
        $LatestVer = "$($pypiJson.info.version)".Trim()
    } catch { }

    if ($_InstalledVersionProbeExit -eq 2) {
        substep "duplicate metadata found for a core package -- forcing package repair..." "Cyan"
    } elseif ($InstalledVer -and $LatestVer -and ($InstalledVer -eq $LatestVer)) {
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
        # An interrupted install leaves $_PkgName current while studio.txt
        # never finished, so the compare above says "up to date" and update --
        # plus the desktop Repair button -- no-ops on a venv that cannot boot.
        $_studioInstallIncomplete = $false
        try {
            & python -c "
import sys
sys.path.insert(0, sys.argv[1])
try:
    import install_manifest
except Exception:
    sys.exit(0)  # older tree without the manifest helper: leave the fast path alone
sys.exit(0 if install_manifest.verify_install()['ok'] else 1)
" "$PSScriptRoot" 2>$null
            if ($LASTEXITCODE -ne 0) { $_studioInstallIncomplete = $true }
        } catch {}
        if ($_studioInstallIncomplete) {
            substep "studio install incomplete -- forcing dependency pass to repair..." "Cyan"
            $SkipPythonDeps = $false
        }
        # If the desktop app specifies a minimum required backend version and the installed
        # package is older than that requirement, force the dependency pass to upgrade it.
        if ($env:UNSLOTH_DESKTOP_BACKEND_VERSION) {
            $_desktopVerBad = $false
            try {
                & python -c "
import re, sys
try:
    from packaging.version import parse as parse_v
except ImportError:
    def parse_v(v):
        match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', (v or '').strip())
        return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None
installed = parse_v(sys.argv[1])
required = parse_v(sys.argv[2])
sys.exit(0 if installed is not None and required is not None and installed >= required else 1)
" "$InstalledVer" "$env:UNSLOTH_DESKTOP_BACKEND_VERSION" 2>$null
                if ($LASTEXITCODE -ne 0) { $_desktopVerBad = $true }
            } catch {}
            if ($_desktopVerBad) {
                substep "$_PkgName $InstalledVer < $env:UNSLOTH_DESKTOP_BACKEND_VERSION (required by desktop app) -- forcing dependency pass to update..." "Cyan"
                $SkipPythonDeps = $false
            }
        }
        # ...but not if an AMD GPU is present and installed PyTorch is CPU-only
        # (host predates ROCm-wheel support, or GPU added later): the fast "up to
        # date" path would leave the user on CPU torch with Train/Export disabled.
        # Force the dependency pass so the ROCm wheels get installed.
        if ($script:ROCmGfxArch) {
            # Bounded, like every other torch probe here and for the reason the disk-based ROCm
            # rescue above exists: on a faulted HIP runtime `import torch` never comes back, and
            # the rescue now KEEPS that venv rather than deleting it, so this is the first
            # `import torch` such a host reaches and an unbounded call would hang setup forever.
            # A probe that does not answer keeps $_torchIsCpu true, the same safe direction as
            # before: one dependency pass, never a silent skip.
            $_rocmTorchProbe = Invoke-BoundedPythonProbe -PythonExe "python" `
                -Code "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
            $_torchIsCpu = -not $_rocmTorchProbe.Ok
            if ($_torchIsCpu) {
                substep "AMD GPU ($script:ROCmGfxArch) detected but installed PyTorch is CPU-only -- reinstalling ROCm PyTorch" "Cyan"
                $SkipPythonDeps = $false
            }
        }
        # ...and the same for an Intel Arc / Data Center GPU, or an up-to-date package on a CPU
        # wheel stays on CPU torch forever. $SkipPythonDeps is re-tested so an escape taken above
        # does not read twice. Both escapes below exist to reach the XPU install and its two
        # remediations, all three gated on $XpuIndexUrl (set only when the resolved leaf is xpu),
        # so $_xpuIsReachable holds them back where a pin or no-torch mode sends this host
        # elsewhere and clearing the fast path would install nothing and re-fire forever.
        $_pinLeafNow = Get-TorchIndexLeaf (Get-PinnedTorchIndexUrl)
        $_xpuIsReachable = (-not $NoTorchMode) -and ((-not $_pinLeafNow) -or ($_pinLeafNow -eq "xpu"))
        if ($script:IsIntelXpu -and $SkipPythonDeps -and $_xpuIsReachable) {
            # The WHEEL, not the runtime: torch.xpu.is_available() is also false for a supported
            # +xpu wheel on a wedged driver, and the dependency pass cannot repair a driver, so
            # keying on it would force a full resolution every update for nothing.
            if (-not (Test-VenvTorchIsXpuSupported -VenvPath $VenvDir)) {
                substep "Intel GPU detected but installed PyTorch is not a supported XPU build -- reinstalling XPU PyTorch" "Cyan"
                $SkipPythonDeps = $false
            }
        }
        # Keyed off the installed wheel as well as the scan: an explicit xpu pin on a host the
        # scan skips (a mixed NVIDIA + Intel box) still ends up on XPU with $script:IsIntelXpu
        # false. The bitsandbytes floor and the Triton replacement live in the dependency pass
        # below, so a venv that reached +xpu without them would fast-path past them forever.
        if ($SkipPythonDeps -and $_xpuIsReachable -and ($script:IsIntelXpu -or $installedTorchTag -eq "xpu")) {
            $_xpuDepsCode = "import importlib.metadata as m; " +
                "print('BNB=' + next((d.version for d in m.distributions() " +
                "if (d.metadata['Name'] or '').lower() == 'bitsandbytes'), '')); " +
                "print('TRITONWIN=' + next((d.version for d in m.distributions() " +
                "if (d.metadata['Name'] or '').lower().replace('_','-') == 'triton-windows'), ''))"
            $_xpuDeps = Invoke-BoundedPythonProbe -PythonExe "python" -Code $_xpuDepsCode
            if (-not $_xpuDeps.Ok) {
                # A probe that did not answer says nothing about the venv, and reading that as
                # "dependencies are current" would fast-path past both remediations forever.
                # Same direction as an unparseable version below: one extra pass.
                substep "Intel XPU dependencies could not be read -- running the dependency pass" "Cyan"
                $SkipPythonDeps = $false
            } else {
                $_bnbVer = if ($_xpuDeps.Output -match '(?m)^BNB=(\S+)\s*$') { $Matches[1] } else { "" }
                # An unreadable version is treated as stale, the safe direction: one extra pass,
                # never a venv left without 4-bit kernels. Trailing suffixes (0.51.0.dev0) are
                # dropped, not cast.
                $_bnbNum = ($_bnbVer -replace '[^0-9.].*$', '').TrimEnd('.')
                $_bnbStale = $true
                if ($_bnbNum -match '^\d+\.\d+') {
                    try { $_bnbStale = [version]$_bnbNum -lt [version]"0.50.0" } catch {}
                }
                $_tritonWinPresent = $_xpuDeps.Output -match '(?m)^TRITONWIN=\S+\s*$'
                if ($_bnbStale -or $_tritonWinPresent) {
                    substep "Intel XPU dependencies are stale -- running the dependency pass" "Cyan"
                    $SkipPythonDeps = $false
                }
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
# A torch that will not import needs the same pass for the same reason, and it is the only
# thing that can act on it: every --force-reinstall that reads the flag lives INSIDE this
# block, so on a current core package with a valid manifest the fast path announced a
# reinstall it then skipped, and setup exited successfully over an unusable wheel.
if ($script:PinChangedForceReinstall -or $script:TorchImportDefinitivelyFailed) {
    $SkipPythonDeps = $false
}

if (-not $SkipPythonDeps) {

# install_python_stack.py drops the manifest before its own dependency pass, but
# pip, torch and triton are replaced first here. Drop it now so a run killed in
# those leaves the venv marked half-built, not behind a marker that verifies.
$_ManifestDropped = $true
try {
    & python -c "
import sys
sys.path.insert(0, sys.argv[1])
try:
    import install_manifest
except Exception:
    sys.exit(0)  # older tree without the manifest helper
sys.exit(0 if install_manifest.remove_manifest() else 1)
" "$PSScriptRoot" 2>$null
    if ($LASTEXITCODE -ne 0) { $_ManifestDropped = $false }
} catch { $_ManifestDropped = $false }
if (-not $_ManifestDropped) {
    Write-StudioLine "[ERROR] Could not remove the stale unsloth_install_manifest.json." -ForegroundColor Red
    Write-StudioLine "        Refusing to install behind a marker that still reports this venv as complete." -ForegroundColor Red
    Exit-SetupFailure "Could not remove the stale unsloth_install_manifest.json"
}

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
# paths on, cache under Unsloth home; else use a short drive-root dir for headroom.
if ($StageRoot -or $LongPathsEnabled) {
    $TorchCacheDir = Join-Path $RuntimeRoot "TORCHINDUCTOR_CACHE_DIR"
} else {
    $TorchCacheDir = "C:\tc"
}
if (-not (Test-Path -LiteralPath $TorchCacheDir)) { [System.IO.Directory]::CreateDirectory($TorchCacheDir) | Out-Null }
$env:TORCHINDUCTOR_CACHE_DIR = $TorchCacheDir
if (-not $StageRoot) {
    [Environment]::SetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', $TorchCacheDir, 'User')
}
substep "TORCHINDUCTOR_CACHE_DIR set to $TorchCacheDir (avoids MAX_PATH issues)"

# Explicit pin (URL or family) wins over GPU probing and suppresses the AMD reroute below;
# matches install.sh / install.ps1 / install_python_stack.py.
$PinnedTorchIndexUrl = Get-PinnedTorchIndexUrl
$TorchIndexPinned = [bool]$PinnedTorchIndexUrl
if ($PinnedTorchIndexUrl) {
    $CuTag = Get-TorchIndexLeaf $PinnedTorchIndexUrl
} elseif ($script:PreservedXpuVenv) {
    # The stale check kept an installed +xpu venv rather than wiping it, which a hybrid
    # NVIDIA + Arc host reaches. Ahead of the NVIDIA arm deliberately: converting halfway leaves
    # +xpu torch under triton-windows and no XPU index to swap it back. install.ps1 converts.
    $CuTag = "xpu"
} elseif ($script:PreservedInstallerTorchTag) {
    # The stale check kept the GPU wheel install.ps1 placed here minutes ago. That decision has to
    # reach the install arms below or it is undone there: the AMD arm force-reinstalls
    # unconditionally and the XPU arm whenever the installed tag is not xpu, so neither is held off
    # by $script:PinChangedForceReinstall staying false. Selecting the family already installed
    # keeps both out of the way -- a cu* tag skips the AMD reroute below (it needs $CuTag -eq
    # "cpu") and the XPU arm (it needs "xpu") and lands on the CUDA arm, which forces nothing; a
    # kept +rocm venv reads as "cpu" here and lands on the CPU arm, whose bare torch range a +rocm
    # build already satisfies. The +xpu case is the branch above. Behind the pin check, like every
    # other arm: an explicit pin repairs in place before the guard runs anyway.
    $CuTag = if (Test-CudaFamilyLeaf $script:PreservedInstallerTorchTag) { $script:PreservedInstallerTorchTag } else { "cpu" }
} elseif ($HasNvidiaSmi) {
    $CuTag = Get-PytorchCudaTag
} elseif ($script:IsIntelXpu) {
    # XPU (SYCL) wheels ship under the /xpu leaf, so $TorchInstallIndexUrl below resolves to
    # <mirror>/xpu. A pin above still wins; the AMD reroute below needs $CuTag -eq "cpu".
    $CuTag = "xpu"
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
# SDK -- which flips Unsloth out of chat-only (CHAT_ONLY) and enables Train/Export.
# Gating on $HasROCm alone left Strix Halo / Radeon 8060S on CPU torch; a failed
# ROCm install still falls back to CPU below, so this is safe.
if (-not $TorchIndexPinned -and ($HasROCm -or $ROCmGfxArch) -and $CuTag -eq "cpu") {
    $amdIndexBase = if ($env:UNSLOTH_ROCM_WINDOWS_MIRROR) { $env:UNSLOTH_ROCM_WINDOWS_MIRROR.TrimEnd('/') } else { "https://repo.amd.com/rocm/whl" }
    # gfx120X and Strix have a null _grouped_mm kernel on torch <2.11.0.
    # Mirrors the $torchFloorMap in install.ps1 so both installers enforce
    # the same floor and ceiling when pulling from AMD's per-arch index.
    $torchFloorMap = @{
        "gfx1201" = "torch>=2.11.0,<2.12.0"; "gfx1200" = "torch>=2.11.0,<2.12.0"
        "gfx1151" = "torch>=2.11.0,<2.12.0"; "gfx1150" = "torch>=2.11.0,<2.12.0"
        "gfx1152" = "torch>=2.11.0,<2.12.0"
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
        "gfx1152" = "torchvision>=0.26.0,<0.27.0"
    }
    $torchaudioFloorMap = @{
        "gfx1201" = "torchaudio>=2.11.0,<2.12.0"; "gfx1200" = "torchaudio>=2.11.0,<2.12.0"
        "gfx1151" = "torchaudio>=2.11.0,<2.12.0"; "gfx1150" = "torchaudio>=2.11.0,<2.12.0"
        "gfx1152" = "torchaudio>=2.11.0,<2.12.0"
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
        substep "       Supported: gfx1200/1201 (RDNA 4), gfx1150/1151/1152 (RDNA 3.5), gfx1100-1103 (RDNA 3), gfx1030-1036 (RDNA 2), gfx90a, gfx908" "Yellow"
    } else {
        # HIP SDK present ($HasROCm=true via amd-smi) but gcnArchName was not
        # readable — warn rather than silently falling back to CPU PyTorch.
        substep "[WARN] AMD GPU detected (HIP SDK present) but GPU arch could not be read -- falling back to CPU-only PyTorch" "Yellow"
        substep "       Arch detection requires hipinfo to report gcnArchName. Re-install the HIP SDK if this is unexpected." "Yellow"
    }
}

# A pinned gfx*/rocm index skips the auto-reroute above; route it through the ROCm install path
# with the same floor/companions the unpinned AMD path uses (mirrors install.ps1), else the CUDA
# branch installs bare torch and resolves a known-bad wheel for gfx115x/gfx120x/rocm>=7.2.
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

# Declared outside the guard: the bitsandbytes and Triton passes read it from outside too, and
# no-torch mode never reaches the assignment below.
$XpuIndexUrl = $null

if (-not $NoTorchMode) {
# Windows on ARM has win_arm64 torch and torchvision wheels but no torchaudio on any index,
# so every branch below drops it. Ask the interpreter uv resolves for, not
# PROCESSOR_ARCHITECTURE, which describes the host process. Inside the no-torch guard
# because all three uses are, and no-torch installs nothing to skip.
$_setupPlatform = ""
try {
    $_setupPlatform = (& python -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
} catch { $_setupPlatform = "" }
$WinArm64NoAudio = ($_setupPlatform -eq "win-arm64")
if ($WinArm64NoAudio) { substep "windows on arm: skipping torchaudio (no win_arm64 wheel upstream)" }

$ROCmCpuFallback = $false
if ($ROCmIndexUrl) {
    substep "installing PyTorch (AMD ROCm, $ROCmGfxArch)..."
    if ($ROCmTorchSpec -ne "torch") {
        substep "  enforcing $ROCmTorchSpec $ROCmVisionSpec $ROCmAudioSpec (known _grouped_mm bug in older wheels)" "Cyan"
    }
    # Built above the verbose branch: a splat assigned inside it is unset on the other.
    $_rocmTrio = @($ROCmTorchSpec, $ROCmVisionSpec, $ROCmAudioSpec)
    if ($WinArm64NoAudio) { $_rocmTrio = @($ROCmTorchSpec, $ROCmVisionSpec) }
    if ($script:UnslothVerbose) {
        Fast-Install @_rocmTrio --force-reinstall --index-url $ROCmIndexUrl | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install @_rocmTrio --force-reinstall --index-url $ROCmIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-StudioLine "[WARN] AMD ROCm PyTorch install failed -- falling back to CPU" -ForegroundColor Yellow
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Yellow
        $ROCmIndexUrl = $null
        $ROCmCpuFallback = $true
    } else {
        # Tell install_python_stack.py to skip probe + suppress manual-install warning.
        $env:UNSLOTH_ROCM_TORCH_INSTALLED = "1"
        substep "GPU ROCm PyTorch installed ($ROCmGfxArch) -- training and GPU inference will use the GPU" "Cyan"
    }
}

# ── Intel XPU (SYCL) ─────────────────────────────────────────────────────────
# Its own index leaf, so it must not fall into the CUDA branch below ("CUDA support (xpu)").
# Reached on an Arc / Data Center host and on an explicit xpu pin.
$XpuCpuFallback = $false
if (-not $ROCmIndexUrl -and $CuTag -eq "xpu") { $XpuIndexUrl = $TorchInstallIndexUrl }
if ($XpuIndexUrl) {
    substep "installing PyTorch (Intel XPU)..."
    # Bounded like install.ps1's XPU install: the xpu index serves torch past what Unsloth
    # supports. The floor is 2.6, not the usual 2.4 -- unsloth/models/_utils.py raises at import
    # for an XPU device below that, and an older wheel would be kept as satisfying the range.
    $_xpuTrio = @("torch>=2.6,<2.11.0", "torchvision>=0.21,<0.26.0", "torchaudio>=2.6,<2.11.0")
    if ($WinArm64NoAudio) { $_xpuTrio = @("torch>=2.6,<2.11.0", "torchvision>=0.21,<0.26.0") }
    # Gated like $cpuForce below, NOT unconditional: install.ps1 already installed the XPU trio
    # before calling setup, so forcing every pass re-downloads GB there and on every update.
    # Force only on a flavor change -- a CPU (or cu*/rocm) wheel satisfies the range, so uv would
    # keep it and never migrate. An unreadable flavor forces too, the safe direction.
    $xpuForce = @()
    if ($installedTorchTag -ne "xpu") { $xpuForce = @("--force-reinstall") }
    # A changed pin repairs in place, so the existing +xpu wheel must be replaced too.
    if ($script:PinChangedForceReinstall) { $xpuForce = @("--force-reinstall") }
    # And a +xpu wheel that no longer imports: its on-disk tag is still "xpu", so the check
    # above sees no flavour change and the range is satisfied, which leaves the resolver
    # holding the broken wheel and the run writing a completion manifest over it.
    if ($script:TorchImportDefinitivelyFailed) { $xpuForce = @("--force-reinstall") }
    if ($script:UnslothVerbose) {
        Fast-Install @_xpuTrio @xpuForce --index-url $XpuIndexUrl | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install @_xpuTrio @xpuForce --index-url $XpuIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        # Transient XPU-index failure: fall back to a CPU base rather than leaving no torch
        # (same shape as ROCm above).
        Write-StudioLine "[WARN] Intel XPU PyTorch install failed -- falling back to CPU" -ForegroundColor Yellow
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Yellow
        $XpuIndexUrl = $null
        $XpuCpuFallback = $true
        $TorchInstallIndexUrl = "$PyTorchWhlBase/cpu"
    } else {
        substep "GPU XPU PyTorch installed -- training and GPU inference will use the GPU" "Cyan"
        # The wheel being in place does not mean the runtime initializes -- see the helper.
        Assert-XpuRuntimeReady -PythonExe "python" | Out-Null
    }
}

if (-not $ROCmIndexUrl -and -not $XpuIndexUrl -and ($CuTag -eq "cpu" -or $ROCmCpuFallback -or $XpuCpuFallback)) {
    substep "installing PyTorch (CPU-only)..."
    # After an AMD ROCm fallback, force-reinstall so a partial ROCm torch (which satisfies the
    # CPU torch>= range) is replaced by the CPU build; skip on a genuine CPU host to stay fast.
    # $ROCmCpuFallback matters when a PINNED ROCm index failed ($CuTag is still the rocm leaf).
    # Build the array directly: an if-expression collapses @("x") to a scalar @splat would
    # enumerate char-by-char.
    $cpuForce = @()
    if ($ROCmCpuFallback) { $cpuForce = @("--force-reinstall") }
    # Same after an Intel XPU fallback: a partial +xpu torch satisfies the CPU torch>= range.
    if ($XpuCpuFallback) { $cpuForce = @("--force-reinstall") }
    # --force-reinstall on a pin change: a stale +cu / +rocm wheel still satisfies the CPU
    # torch>= range, so uv would keep it and only swap companions.
    if ($script:PinChangedForceReinstall) { $cpuForce = @("--force-reinstall") }
    # Same for a wheel that no longer imports. Nothing else here distinguishes it: the tag is
    # rescued from disk and still reads "cpu", so the range is satisfied and it is kept.
    if ($script:TorchImportDefinitivelyFailed) { $cpuForce = @("--force-reinstall") }
    # A PINNED cpu index installs the bounded trio (parity with _CPU_TORCH_PKG_SPEC): the /cpu
    # index serves newer torch, and _ensure_cpu_torch keeps any CPU build, so a bare trio could
    # land an unsupported version. Unpinned CPU hosts keep the bare trio (pre-pin behavior).
    $cpuTorchSpec = "torch"; $cpuVisionSpec = "torchvision"; $cpuAudioSpec = "torchaudio"
    if ($TorchIndexPinned) {
        $cpuTorchSpec  = "torch>=2.4,<2.12.0"
        $cpuVisionSpec = "torchvision>=0.19,<0.27.0"
        $cpuAudioSpec  = "torchaudio>=2.4,<2.12.0"
    }
    # Bound an XPU fallback too: this is not the plain CPU box the bare trio was preserved for.
    if ($XpuCpuFallback) {
        $cpuTorchSpec  = "torch>=2.4,<2.12.0"
        $cpuVisionSpec = "torchvision>=0.19,<0.27.0"
        $cpuAudioSpec  = "torchaudio>=2.4,<2.12.0"
    }
    $_torchTrio = @($cpuTorchSpec, $cpuVisionSpec, $cpuAudioSpec)
    if ($WinArm64NoAudio) { $_torchTrio = @($cpuTorchSpec, $cpuVisionSpec) }
    if ($script:UnslothVerbose) {
        Fast-Install @_torchTrio @cpuForce --index-url $TorchInstallIndexUrl | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install @_torchTrio @cpuForce --index-url $TorchInstallIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-StudioLine "[FAILED] PyTorch install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
        Exit-SetupFailure "PyTorch installation failed (exit code $torchInstallExit)"
    }
} elseif (-not $ROCmIndexUrl -and -not $XpuIndexUrl) {
    substep "installing PyTorch with CUDA support ($CuTag)..."
    substep "(This download is ~2.8 GB -- may take a few minutes)"
    # --force-reinstall on a pin change: an installed cuXXX wheel satisfies the bare torch
    # requirement (PEP 440 ignores the +cuXXX tag), so without it a changed CUDA pin (cu126
    # -> cu128) never applies.
    $cudaForce = @()
    if ($script:PinChangedForceReinstall -or $script:TorchImportDefinitivelyFailed) {
        $cudaForce = @("--force-reinstall")
    }
    # An unknown-leaf custom pin (/simple, /current) routes here with $CuTag as that leaf. Bound
    # the trio like the fresh custom-pin paths so a mirror can't pull an ABI-newer companion
    # against the capped torch. Known cu* leaves keep bare specs.
    $cudaTorchSpec = "torch"
    $cudaVisionSpec = "torchvision"
    $cudaAudioSpec = "torchaudio"
    if ($TorchIndexPinned -and -not (Test-CudaFamilyLeaf $CuTag)) {
        $cudaTorchSpec = "torch>=2.4,<2.11.0"
        $cudaVisionSpec = "torchvision>=0.19,<0.26.0"
        $cudaAudioSpec = "torchaudio>=2.4,<2.11.0"
    }
    # A custom pin whose leaf is not cpu (a corporate /simple mirror) lands an ARM64 host
    # here, so this branch drops torchaudio too.
    $_cudaTrio = @($cudaTorchSpec, $cudaVisionSpec, $cudaAudioSpec)
    if ($WinArm64NoAudio) { $_cudaTrio = @($cudaTorchSpec, $cudaVisionSpec) }
    if ($script:UnslothVerbose) {
        Fast-Install @_cudaTrio @cudaForce --index-url $TorchInstallIndexUrl | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        $torchInstallExit = $LASTEXITCODE
        $output = ""
    } else {
        $output = Fast-Install @_cudaTrio @cudaForce --index-url $TorchInstallIndexUrl | Out-String
        $torchInstallExit = $LASTEXITCODE
    }
    if ($torchInstallExit -ne 0) {
        Write-StudioLine "[FAILED] PyTorch CUDA install failed (exit code $torchInstallExit)" -ForegroundColor Red
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
        Exit-SetupFailure "PyTorch CUDA installation failed (exit code $torchInstallExit)"
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
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Yellow
    } else {
        substep "Triton for Windows installed (enables torch.compile)"
    }
}
} else {
    substep "skipping direct PyTorch and Triton installation (no-torch mode)." "Yellow"
}

# No unsloth.exe rename needed. install.ps1 no longer starts the generated console
# script at all (it runs the CLI through the venv's python.exe, #8490), so the rename
# would now target a file nothing is holding -- but it was never needed either:
# install.ps1 sets SKIP_STUDIO_BASE=1 (base never reinstalled) and 'studio update'
# goes through uv (--upgrade-package), whose pip fallback no-ops on the
# already-satisfied bare unsloth/unsloth-zoo. Either way unsloth.exe stays.
# Duplicate metadata repair is the one pass that DOES reinstall unsloth under
# SKIP_STUDIO_BASE, so the CLI wraps this script in its launcher transaction.

# ── Publish the torch flavor this run settled on ──
# so install_python_stack.py can enforce it: its dependency steps resolve torch from PyPI, whose
# Windows wheel is 2.11.0+cpu, and only install.ps1 -- never on the updater's path -- repaired
# that. Vocabulary is Get-InstalledTorchTag's; an unknown leaf publishes nothing.
if (-not $NoTorchMode) {
    $_expectedLeaf = Get-TorchIndexLeaf $TorchInstallIndexUrl
    # $ROCmIndexUrl first: on the AMD path $TorchInstallIndexUrl still points at /cpu.
    $_expectedTag = if ($ROCmIndexUrl) { "rocm" }
                    elseif (Test-CudaFamilyLeaf $_expectedLeaf) { $_expectedLeaf }
                    elseif ($_expectedLeaf -eq "cpu" -or $_expectedLeaf -eq "xpu") { $_expectedLeaf }
                    elseif (Test-PipRocmFamilyLeaf $_expectedLeaf) { "rocm" }
                    else { $null }
    # Remove-Item, NOT `= ""`: PowerShell 7.5 took .NET 9's change and KEEPS a name
    # assigned an empty string. Deleting also stops a value inherited from the caller's
    # shell surviving a run that decided it cannot name this host's flavor.
    if ($_expectedTag) {
        $env:UNSLOTH_EXPECTED_TORCH_TAG = $_expectedTag
    } else {
        Remove-Item Env:\UNSLOTH_EXPECTED_TORCH_TAG -ErrorAction SilentlyContinue
    }
    if ($TorchInstallIndexUrl) {
        $env:UNSLOTH_TORCH_INSTALL_INDEX_URL = $TorchInstallIndexUrl
    } else {
        Remove-Item Env:\UNSLOTH_TORCH_INSTALL_INDEX_URL -ErrorAction SilentlyContinue
    }
}

# Ordered heavy dependency installation -- shared cross-platform script
substep "running ordered dependency installation..."
python "$PSScriptRoot\install_python_stack.py"
$stackExit = $LASTEXITCODE

# ── Intel XPU: bitsandbytes must carry XPU kernels ──
# unsloth/bnb_availability.py binds cgemv_4bit_inference_fp16/bf16 for device_type "xpu" and only
# bitsandbytes' XPU library exports those, so a wheel without it turns 4-bit QLoRA off. 0.48.2 is
# the first win_amd64 build carrying that library, but the floor is 0.50.0, the AMD paths' floor:
# <=0.49.2 NaNs at 4-bit decode on AMD, and an Arc card can sit next to a Radeon. unsloth's own
# floor (>=0.45.5) lets a MIGRATED venv keep its old wheel while the stack above upgrades unsloth
# alone, so run after it for the last word. --no-deps (torch and numpy are in), and never the
# curated unsloth[intel-gpu-torch*] extra: it pins torch to one +xpu wheel URL, unpinning the
# trio above, and carries a preview bitsandbytes wheel uv refuses. $XpuIndexUrl is the "ended up
# on XPU" gate. Must stay ABOVE the $ErrorActionPreference restore below: Fast-Install needs
# EAP=Continue or PS 5.1 turns pip's stderr into a terminating error. Best-effort.
if ($stackExit -eq 0 -and $XpuIndexUrl) {
    substep "installing bitsandbytes with Intel XPU kernels..."
    if ($script:UnslothVerbose) {
        Fast-Install --no-deps "bitsandbytes>=0.50.0" | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
        $bnbXpuExit = $LASTEXITCODE
        $bnbOutput = ""
    } else {
        $bnbOutput = Fast-Install --no-deps "bitsandbytes>=0.50.0" | Out-String
        $bnbXpuExit = $LASTEXITCODE
    }
    if ($bnbXpuExit -ne 0) {
        substep "[WARN] could not install an XPU-capable bitsandbytes (exit $bnbXpuExit); 4-bit QLoRA may be unavailable." "Yellow"
        Write-StudioLine (Redact-InstallOutput $bnbOutput) -ForegroundColor Yellow
    }
}

# ── Intel XPU: triton-windows must not shadow torch's XPU triton ──
# triton-windows and torch's XPU triton BOTH own the top-level `triton` package (80 to 160 shared
# paths, __init__.py and _C/libtriton.pyd among them), so a cu*-to-xpu pin repair leaves the CUDA
# build shadowing the XPU one. AFTER the stack, since unsloth declares triton-windows a win32
# core dependency install reinstalls anything removed earlier. Uninstall always paired with a
# reinstall: removing one drops the shared paths the other overwrote. The spec is read from the
# installed torch (renamed pytorch-triton-xpu -> triton-xpu in torch 2.10).
if ($stackExit -eq 0 -and $XpuIndexUrl) {
    # One -c line, so no double quotes (Invoke-BoundedPythonProbe wraps $Code in them).
    $_tritonCode = "import importlib.metadata as m; " +
        "print('TRITONWIN=' + next((d.version for d in m.distributions() " +
        "if (d.metadata['Name'] or '').lower().replace('_','-') == 'triton-windows'), '')); " +
        "print('TRITONXPU=' + next((r.split(';')[0].strip() " +
        "for r in (m.requires('torch') or []) if 'triton' in r.lower()), ''))"
    $_tritonProbe = Invoke-BoundedPythonProbe -PythonExe "python" -Code $_tritonCode
    # Line-anchored like the other probes so a stdout banner ahead of the answer hides nothing.
    $_tritonWinVer = if ($_tritonProbe.Ok -and $_tritonProbe.Output -match '(?m)^TRITONWIN=(\S+)\s*$') { $Matches[1] } else { "" }
    $_tritonXpuSpec = if ($_tritonProbe.Ok -and $_tritonProbe.Output -match '(?m)^TRITONXPU=(\S+)\s*$') { $Matches[1] } else { "" }
    # The spec must itself be an XPU triton (pytorch-triton-xpu / triton-xpu); anything else means
    # torch is not the +xpu wheel this branch assumes.
    if ($_tritonWinVer -and $_tritonXpuSpec -match '(?i)xpu') {
        substep "replacing triton-windows $_tritonWinVer with $_tritonXpuSpec (Intel XPU)..." "Cyan"
        # install_manifest.manifest_path() is venv_root()/MANIFEST_NAME and venv_root() is
        # sys.prefix, which is $VenvDir here -- the same join Get-PersistedNoTorch does. Assembled
        # rather than asked for: a subprocess to learn a constant can hang or fail in a way
        # indistinguishable from "there is no manifest", which reopens the window this closes.
        # test_intel_registry_fallback.ps1 asserts this literal still matches MANIFEST_NAME.
        $_manifestPath = Join-Path $VenvDir "unsloth_install_manifest.json"
        # Fetch, THEN uninstall, THEN install the file. The uninstall cannot go last -- it drops
        # the paths in triton-windows' OWN record, which are the shared ones -- so pre-fetching
        # keeps a dead mirror from stranding the venv between the two steps. uv has no
        # `pip download`, hence Fast-Download.
        $_tritonTmp = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth_triton_xpu_$([guid]::NewGuid().ToString('N').Substring(0,8))"
        try {
            New-Item -ItemType Directory -Force -Path $_tritonTmp -ErrorAction SilentlyContinue | Out-Null
            if ($script:UnslothVerbose) {
                Fast-Download --no-deps --only-binary=:all: -d $_tritonTmp $_tritonXpuSpec --index-url $XpuIndexUrl | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
                $tritonDlExit = $LASTEXITCODE
                $tritonDlOutput = ""
            } else {
                $tritonDlOutput = Fast-Download --no-deps --only-binary=:all: -d $_tritonTmp $_tritonXpuSpec --index-url $XpuIndexUrl | Out-String
                $tritonDlExit = $LASTEXITCODE
            }
            # The exit code alone is not enough: no wheel on disk means nothing to install from.
            $_tritonWheel = @(Get-ChildItem -LiteralPath $_tritonTmp -Filter "*.whl" -ErrorAction SilentlyContinue) | Select-Object -First 1 -ExpandProperty FullName
            if ($tritonDlExit -ne 0 -or -not $_tritonWheel) {
                substep "[WARN] could not fetch $_tritonXpuSpec (exit $tritonDlExit); triton-windows $_tritonWinVer left in place -- it still shadows torch XPU triton, so torch.compile will not use the XPU." "Yellow"
                Write-StudioLine (Redact-InstallOutput $tritonDlOutput) -ForegroundColor Yellow
            } else {
                # From here to the reinstall below is a window where a kill leaves a venv with
                # no triton that the next update still reads as complete and fast-paths past.
                # MOVE the manifest into the wheel's temp dir for the swap, never read and
                # rewrite it: PS 5.1's Set-Content defaults to ANSI and its -Encoding utf8 emits
                # a BOM read_manifest's json.load rejects, so either corrupts a manifest with a
                # non-ASCII path. The finally below deletes the dir, so an unrestored manifest
                # stays gone -- which is the truth while the venv has no triton.
                $_manifestHeld = $null
                $_manifestBlocked = $false
                $_uninstallExit = 0
                if ($_manifestPath -and (Test-Path -LiteralPath $_manifestPath -PathType Leaf)) {
                    $_held = Join-Path $_tritonTmp "held_manifest.json"
                    try {
                        Move-Item -LiteralPath $_manifestPath -Destination $_held -Force -ErrorAction Stop
                        # Move-Item across volumes is a copy plus a delete and reports success
                        # even when only the delete fails, leaving the original in place. The
                        # manifest must be GONE for the swap, so confirm rather than trust.
                        if (Test-Path -LiteralPath $_manifestPath -PathType Leaf) {
                            Remove-Item -LiteralPath $_held -Force -ErrorAction SilentlyContinue
                            $_manifestBlocked = $true
                        } else {
                            $_manifestHeld = $_held
                        }
                    } catch { $_manifestBlocked = $true }
                }
                if ($_manifestBlocked) {
                    # A manifest that will not move (locked, read-only) would stay valid right
                    # through the destructive window, so do not open one. Leaving triton-windows
                    # costs torch.compile on the XPU, which the next run can still fix.
                    substep "[WARN] could not set the install manifest aside; triton-windows $_tritonWinVer left in place -- it still shadows torch XPU triton, so torch.compile will not use the XPU." "Yellow"
                } else {
                    Fast-Uninstall "triton-windows" | Out-Null
                    $_uninstallExit = $LASTEXITCODE
                }
                # A triton-windows that would not uninstall (Unsloth running and holding
                # _C/libtriton.pyd open) still shadows the XPU triton, so installing over it
                # achieves nothing and would restore the manifest onto an unchanged venv.
                if (-not $_manifestBlocked -and $_uninstallExit -ne 0) {
                    substep "[WARN] could not remove triton-windows $_tritonWinVer (exit $_uninstallExit); it still shadows torch XPU triton, so torch.compile will not use the XPU." "Yellow"
                    if ($_manifestHeld) {
                        try { Move-Item -LiteralPath $_manifestHeld -Destination $_manifestPath -Force -ErrorAction Stop } catch {}
                    }
                } elseif (-not $_manifestBlocked) {
                    if ($script:UnslothVerbose) {
                        Fast-Install --force-reinstall --no-deps $_tritonWheel | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
                        $tritonXpuExit = $LASTEXITCODE
                        $tritonOutput = ""
                    } else {
                        $tritonOutput = Fast-Install --force-reinstall --no-deps $_tritonWheel | Out-String
                        $tritonXpuExit = $LASTEXITCODE
                    }
                    $_tritonPresent = ($tritonXpuExit -eq 0)
                    if ($tritonXpuExit -ne 0) {
                        # Off the network by now, so this is disk/permissions. triton-windows is
                        # already gone and took the shared paths with it, so put SOME working
                        # triton back rather than leave the venv unable to import one.
                        Fast-Install --force-reinstall --no-deps "triton-windows<3.7" | Out-Null
                        $tritonBackExit = $LASTEXITCODE
                        $_tritonPresent = ($tritonBackExit -eq 0)
                        Write-StudioLine (Redact-InstallOutput $tritonOutput) -ForegroundColor Yellow
                        if ($tritonBackExit -eq 0) {
                            substep "[WARN] could not install $_tritonXpuSpec (exit $tritonXpuExit); restored triton-windows, so triton still imports -- but torch.compile will not use the XPU." "Yellow"
                        } else {
                            # Redacted: a mirror pin can carry a token, and this is the only place
                            # setup.ps1 shows an index URL. A tokenless URL survives verbatim.
                            $_tritonRepairUrl = Redact-InstallOutput $XpuIndexUrl
                            Write-StudioLine "[ERROR] triton-windows was removed and neither triton would reinstall -- torch.compile is broken." -ForegroundColor Red
                            Write-StudioLine "        Repair with: python -m pip install --force-reinstall --no-deps $_tritonXpuSpec --index-url $_tritonRepairUrl" -ForegroundColor Red
                            # Printing alone left $stackExit at 0, so setup reported success and
                            # install.ps1 COMMITTED this venv over its rollback copy. It has no
                            # importable triton at all; the caller must not accept it.
                            $stackExit = $tritonBackExit
                        }
                    }
                    # Moved back, never rewritten, and only once a triton is importable again. If
                    # neither would reinstall the venv really is incomplete, and losing the
                    # manifest with the temp dir is what makes the next update repair it.
                    if ($_manifestHeld -and $_tritonPresent) {
                        try { Move-Item -LiteralPath $_manifestHeld -Destination $_manifestPath -Force -ErrorAction Stop } catch {}
                        # The finally below deletes the held copy either way, so an unreported
                        # failure loses the manifest silently and the next run does a full
                        # dependency pass with nothing on screen explaining why.
                        if (-not (Test-Path -LiteralPath $_manifestPath -PathType Leaf)) {
                            substep "[WARN] could not restore the install manifest; the next update will re-run the dependency pass." "Yellow"
                        }
                    }
                }
            }
        } finally {
            # The wheel is ~300 MB, so it never outlives the install.
            Remove-Item -Recurse -Force -LiteralPath $_tritonTmp -ErrorAction SilentlyContinue
        }
    }
}

# Restore ErrorActionPreference after pip/python work
$ErrorActionPreference = $prevEAP
if ($stackExit -ne 0) {
    Write-StudioLine "[FAILED] Python dependency installation failed (exit code $stackExit)" -ForegroundColor Red
    Write-StudioLine "   Re-run the installer or check the error above for details." -ForegroundColor Red
    Exit-SetupFailure "Python dependency installation failed (exit code $stackExit)"
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
$VenvT5_530Dir = Join-Path $RuntimeRoot ".venv_t5_530"
$VenvT5_550Dir = Join-Path $RuntimeRoot ".venv_t5_550"
$VenvT5_510Dir = Join-Path $RuntimeRoot ".venv_t5_510"
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
    # Legacy layout -- migrate. The tiered venvs a staged run builds land under the
    # stage root and may never be activated, so removing the live legacy one here
    # would strip the running install of its only sidecar. The live update does it.
    if (-not $StageRoot) {
        Assert-StudioOwnedOrAbsent -Path $VenvT5Legacy -Label "legacy transformers sidecar venv"
        Remove-Item -LiteralPath $VenvT5Legacy -Recurse -Force
    }
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
Write-StudioLine ""

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
        Write-StudioLine "[FAIL] Could not install $pkg into .venv_t5_530/" -ForegroundColor Red
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        Exit-SetupFailure "Could not install $pkg into .venv_t5_530"
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_530Dir --no-deps tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_530Dir --no-deps tiktoken | Out-String
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
        Write-StudioLine "[FAIL] Could not install $pkg into .venv_t5_550/" -ForegroundColor Red
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        Exit-SetupFailure "Could not install $pkg into .venv_t5_550"
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_550Dir --no-deps tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_550Dir --no-deps tiktoken | Out-String
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
        Write-StudioLine "[FAIL] Could not install $pkg into .venv_t5_510/" -ForegroundColor Red
        Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
        $ErrorActionPreference = $prevEAP_t5
        Exit-SetupFailure "Could not install $pkg into .venv_t5_510"
    }
}
if ($script:UnslothVerbose) {
    Fast-Install --target $VenvT5_510Dir --no-deps tiktoken
    $tiktokenInstallExit = $LASTEXITCODE
    $output = ""
} else {
    $output = Fast-Install --target $VenvT5_510Dir --no-deps tiktoken | Out-String
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
# Reuse the managed path resolved and preflighted before phase 1.
if (-not (Test-Path -LiteralPath $UnslothHome)) { [System.IO.Directory]::CreateDirectory($UnslothHome) | Out-Null }
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
$sourceLlamaBackend = "$($env:UNSLOTH_LLAMA_CPP_BACKEND)".Trim().ToLowerInvariant()
$sourceLegacyForceVulkan = "$($env:UNSLOTH_FORCE_VULKAN)".Trim().ToLowerInvariant()
$explicitLlamaSourceBackend = $null
if (-not $IsMacOS) {
    if ($sourceLlamaBackend -in @("cpu", "cuda", "vulkan", "hip", "rocm")) {
        $explicitLlamaSourceBackend = if ($sourceLlamaBackend -eq "hip") { "rocm" } else { $sourceLlamaBackend }
    } elseif ($sourceLlamaBackend -ne "auto" -and $sourceLegacyForceVulkan -in @("1", "true", "yes", "on")) {
        $explicitLlamaSourceBackend = "vulkan"
    }
}

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
        Write-StudioLine "[ERROR] UNSLOTH_LLAMA_PR=$LlamaPr is not a valid PR number" -ForegroundColor Red
        Exit-SetupFailure "UNSLOTH_LLAMA_PR=$LlamaPr is not a valid PR number"
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
    # Unreadable is not missing: reporting "does not exist" would send the user
    # looking for the wrong problem.
    $localSrcState = Get-PathState -Path $LocalLlamaCppSrc -PathType Container
    if ($localSrcState -eq "Denied") {
        Exit-PathAccessDenied -Path $LocalLlamaCppSrc -Label "the UNSLOTH_LOCAL_LLAMA_CPP_DIR directory" -UserSupplied
    }
    if ($localSrcState -ne "Present") {
        step "llama.cpp" "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $LocalLlamaCppSrc" "Red"
        Exit-SetupFailure "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $LocalLlamaCppSrc"
    }
    $ResolvedLocal = (Resolve-Path -LiteralPath $LocalLlamaCppSrc).Path
    # Reusing a local dir disables both the prebuilt download and the source
    # build, so a runnable llama-server.exe must already be present. Accept any
    # layout LlamaCppBackend._layout_candidates() resolves (root-level, build\bin,
    # or build\bin\Release) so the flag never rejects a tree Unsloth could run.
    $LocalLlamaServerFound = $false
    $LocalIsCanonical = ($ResolvedLocal -eq $LlamaCppDir)
    foreach ($_cand in @(
            (Join-Path $ResolvedLocal "llama-server.exe"),
            (Join-Path $ResolvedLocal "build\bin\llama-server.exe"),
            (Join-Path $ResolvedLocal "build\bin\Release\llama-server.exe"))) {
        # Denied must not read as "nothing built here": the canonical branch
        # below would then hand the tree to the prebuilt installer, which
        # replaces the very build this flag asked to reuse.
        $candState = Get-PathState -Path $_cand
        if ($candState -eq "Denied") {
            # -UserSupplied even when this is the canonical location: the
            # override says the tree is the user's build, so never advise
            # deleting it, managed path or not.
            Exit-PathAccessDenied -Path $ResolvedLocal -Label "the UNSLOTH_LOCAL_LLAMA_CPP_DIR build" -UserSupplied
        }
        if ($candState -eq "Present") { $LocalLlamaServerFound = $true; break }
    }
    if ($LocalIsCanonical) {
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
        # and leave Unsloth with no usable binary.
        if (-not $LocalLlamaServerFound) {
            step "llama.cpp" "no llama-server.exe under $ResolvedLocal (looked for .\llama-server.exe, .\build\bin and .\build\bin\Release) -- build llama.cpp there first, or drop --with-llama-cpp-dir" "Red"
            Exit-SetupFailure "No llama-server.exe was found under $ResolvedLocal"
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
            # A link reads Present, so the probe below cannot cover a denied
            # unlink; report it here rather than terminate on the raw throw.
            try { $existing.Delete() }
            catch {
                if (Test-AccessDeniedError $_) { Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install" }
                throw
            }
        }
        if ($StudioHomeIsCustom) {
            Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"
        }
        # The destination is about to be deleted and replaced, so a denial here
        # must stop rather than throw raw: under a default home nothing above
        # has probed it three-state.
        $destState = Get-PathState -Path $LlamaCppDir
        if ($destState -eq "Denied") {
            Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install"
        }
        if ($destState -eq "Present") {
            Remove-Item -Recurse -Force -LiteralPath $LlamaCppDir -ErrorAction SilentlyContinue
            # A locked/in-use tree can silently survive removal (SilentlyContinue
            # masks it). Don't then junction/copy over a half-present dir; mirror the
            # prebuilt path's active-process handling and stop with a clear message.
            # Denied counts as surviving: unreadable is not gone.
            if ((Get-PathState -Path $LlamaCppDir) -ne "Absent") {
                step "llama.cpp" "install blocked by active llama.cpp process" "Yellow"
                substep "Close Unsloth or other llama.cpp users and retry" "Yellow"
                Exit-SetupFailure "llama.cpp install is blocked by an active llama.cpp process" 3
            }
        }
        cmd /c "mklink /J `"$LlamaCppDir`" `"$ResolvedLocal`"" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            substep "Could not create directory junction; copying instead..." "Yellow"
            Copy-Item -Recurse -LiteralPath $ResolvedLocal -Destination $LlamaCppDir
            Remove-AgentInstructionFiles -Roots @($LlamaCppDir)
        }
        Write-StudioLine ""
        step "llama.cpp" "linked local directory: $ResolvedLocal"
        $LocalLlamaCppLinked = $true
        $NeedLlamaSourceBuild = $false
    }
}

if ($LocalLlamaCppLinked) {
    # local directory linked above; skip prebuilt install
} elseif ($explicitLlamaSourceBackend -and $NeedLlamaSourceBuild) {
    Write-StudioLine ""
    step "llama.cpp" "$explicitLlamaSourceBackend was explicitly requested, but this installation requires a source build" "Red"
    substep "Explicit backend selection requires a matching prebuilt bundle; allow prebuilts or unset UNSLOTH_LLAMA_CPP_BACKEND" "Yellow"
    Exit-SetupFailure "$explicitLlamaSourceBackend was explicitly requested, but this installation requires a source build. Explicit backend selection requires a matching prebuilt bundle."
} elseif ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {
    Write-StudioLine ""
    substep "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt llama.cpp install" "Yellow"
    $NeedLlamaSourceBuild = $true
} elseif ($SkipPrebuiltInstall) {
    Write-StudioLine ""
    substep "Skipping prebuilt install -- falling back to source build" "Yellow"
} else {
    Write-StudioLine ""
    # Keep this late guard as defense in depth before the prebuilt installer.
    $llamaDirState = Get-LlamaCppInstallReadState -Path $LlamaCppDir
    if ($llamaDirState -eq "Denied") {
        Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install" -OwnershipUnverified:$StudioHomeIsCustom
    }
    if ($llamaDirState -eq "Readable") {
        substep "Existing llama.cpp install detected -- validating staged prebuilt update before replacement"
        # If the existing install is the wrong kind (e.g. windows-cpu on a ROCm
        # machine that should have windows-rocm), remove it so the installer is
        # forced to download the correct variant rather than skipping on tag match.
        $existingMetaPath = Join-Path $LlamaCppDir "UNSLOTH_PREBUILT_INFO.json"
        # Readable state leaves only marker presence to decide here.
        if (Test-PathQuiet -Path $existingMetaPath -PathType Leaf) {
            try {
                $existingMeta = Get-Content -LiteralPath $existingMetaPath -Raw | ConvertFrom-Json
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
        # Reporting only: the installer reads UNSLOTH_LLAMA_CPP_BACKEND itself, and
        # it is also the only side that can see a choice recorded in the install
        # marker, so forwarding a second copy from here could only ever disagree
        # with it. The override does not change the torch backend.
        $llamaBackend = $sourceLlamaBackend
        $windowsArm64 = (
            $env:OS -eq "Windows_NT" -and
            (
                "$($env:PROCESSOR_ARCHITECTURE)".ToUpperInvariant() -eq "ARM64" -or
                "$($env:PROCESSOR_ARCHITEW6432)".ToUpperInvariant() -eq "ARM64"
            )
        )
        if ($llamaBackend -eq "vulkan" -or $explicitLlamaSourceBackend -eq "vulkan") {
            if ($IsMacOS) {
                Write-StudioLine "[WARN] Vulkan has no effect on macOS; the universal build uses Metal" -ForegroundColor Yellow
            } elseif ($windowsArm64) {
                throw "Vulkan was requested, but no Windows ARM64 Vulkan bundle is published. Unset UNSLOTH_LLAMA_CPP_BACKEND / UNSLOTH_FORCE_VULKAN or compile llama.cpp from source."
            } else {
                Write-StudioLine "  llama.cpp      Vulkan selected for GGUF inference; the PyTorch training backend is unchanged" -ForegroundColor Cyan
            }
        } elseif ($llamaBackend -and $llamaBackend -notin @("auto", "cpu", "cuda", "hip", "rocm")) {
            Write-StudioLine "[WARN] Ignoring UNSLOTH_LLAMA_CPP_BACKEND='$llamaBackend' (expected 'auto', 'cpu', 'cuda', 'vulkan', 'hip', or 'rocm')" -ForegroundColor Yellow
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
            substep "Close Unsloth or other llama.cpp users and retry" "Yellow"
            Exit-SetupFailure "llama.cpp install is blocked by an active llama.cpp process" 3
        } elseif ($prebuiltExit -eq 4) {
            step "llama.cpp" "not enough disk space to install llama.cpp" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            substep "Free up disk or move UNSLOTH_STUDIO_HOME/TEMP to a larger volume, then re-run" "Yellow"
            $PreservedLlamaServerFound = $false
            foreach ($_cand in @(
                    (Join-Path $LlamaCppDir "llama-server.exe"),
                    (Join-Path $LlamaCppDir "build\bin\llama-server.exe"),
                    (Join-Path $LlamaCppDir "build\bin\Release\llama-server.exe"))) {
                if (Test-PathQuiet $_cand) { $PreservedLlamaServerFound = $true; break }
            }
            if (-not $PreservedLlamaServerFound) { $script:LlamaCppDegraded = $true }
            # A preserved server may not satisfy an explicit backend request, and
            # it leaves LlamaCppDegraded false. Never report success on an
            # unverified backend after the requested replacement ran out of space.
            if ($explicitLlamaSourceBackend) {
                step "llama.cpp" "$explicitLlamaSourceBackend was explicitly requested, so the installer will not keep an unverified existing backend" "Red"
                Exit-SetupFailure "$explicitLlamaSourceBackend was explicitly requested, so the installer will not keep an unverified existing llama.cpp backend."
            }
        } elseif ($prebuiltExit -eq 5) {
            step "llama.cpp" "selected backend could not be installed" "Red"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path -LiteralPath $LlamaCppDir) {
                substep "Existing install was restored" "Yellow"
            }
            substep "Check the error above, choose another backend, or retry" "Yellow"
            Exit-SetupFailure "The selected llama.cpp backend could not be installed, so the installer will not substitute a different source backend."
        } elseif ($prebuiltExit -eq 2) {
            step "llama.cpp" "prebuilt install failed" "Yellow"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path -LiteralPath $LlamaCppDir) {
                substep "Prebuilt update failed; existing install was restored or cleaned before source build fallback" "Yellow"
            }
            # Exit 2 means no concrete backend was in play: a request the installer
            # could not honour -- named here or recorded in the install marker,
            # which this script cannot see -- exits 5 above instead.
            substep "Prebuilt llama.cpp path unavailable or failed validation -- falling back to source build" "Yellow"
            $NeedLlamaSourceBuild = $true
        } else {
            step "llama.cpp" "prebuilt helper failed unexpectedly" "Red"
            Write-LlamaFailureLog -Output $prebuiltOutput
            if (Test-Path -LiteralPath $LlamaCppDir) {
                substep "Existing install was restored or left unchanged" "Yellow"
            }
            substep "Source build was not started because it cannot repair an unexpected helper or permissions error" "Yellow"
            Exit-SetupFailure "llama.cpp prebuilt helper failed unexpectedly (exit code $prebuiltExit). Check the error above and retry setup."
        }
}

# ==========================================================================
#  PHASE 3.4: Install the whisper.cpp prebuilt (dictation runtime)
# ==========================================================================
# Mirrors the llama.cpp prebuilt install above; current whisper releases are
# slim bundles that reuse the llama install's ggml runtime, so this runs after
# llama. Failure is never fatal: local dictation falls back to Transformers STT.
$WhisperCppDir = Join-Path $UnslothHome "whisper.cpp"
$WhisperInstaller = Join-Path $PSScriptRoot "install_whisper_prebuilt.py"
# Same opt-outs as setup.sh: a user-configured binary/dir or an explicit skip
# disables the managed install entirely.
if ($env:WHISPER_SERVER_PATH -or $env:UNSLOTH_WHISPER_CPP_PATH) {
    substep "whisper.cpp: using a user-configured binary/dir; skipping managed install"
} elseif ($env:UNSLOTH_SKIP_WHISPER_INSTALL -eq "1") {
    substep "whisper.cpp: install skipped (UNSLOTH_SKIP_WHISPER_INSTALL=1)"
} elseif ($StudioHomeIsCustom -and (Test-Path -LiteralPath $WhisperInstaller) -and
        (Assert-StudioOwnedOrAbsent -Path $WhisperCppDir -Label "whisper.cpp install" -NonFatal) -eq "Denied") {
    # Never fatal, per the phase header: the guard below would exit the whole run
    # on an unreadable tree, taking llama.cpp down with it. Only the denial is
    # caught here; an unowned tree still stops.
    step "whisper.cpp" "install directory cannot be read: access is denied; curated whisper.cpp dictation is unavailable; restore access to $WhisperCppDir or move it aside, then re-run setup; browser and Transformers dictation remain available" "Yellow"
} elseif (Test-Path -LiteralPath $WhisperInstaller) {
    # The installer's atomic activation replaces the whole directory, so the
    # custom-home ownership guard must run first (mirrors the llama block).
    if ($StudioHomeIsCustom) {
        Assert-StudioOwnedOrAbsent -Path $WhisperCppDir -Label "whisper.cpp install"
    }
    $whisperArgs = @($WhisperInstaller, "--install-dir", $WhisperCppDir)
    if ($env:UNSLOTH_WHISPER_RELEASE_TAG) {
        $whisperArgs += @("--published-release-tag", $env:UNSLOTH_WHISPER_RELEASE_TAG)
    }
    if ($script:ROCmGfxArch) {
        $whisperArgs += @("--rocm-gfx", $script:ROCmGfxArch)
    } elseif ($HasROCm) {
        $whisperArgs += "--has-rocm"
    }
    $prevEAPWhisper = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $previousNativeErrorPreferenceW = $null
    $restoreNativeErrorPreferenceW = $false
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $previousNativeErrorPreferenceW = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
        $restoreNativeErrorPreferenceW = $true
    }
    try {
        $whisperOutput = & python @whisperArgs 2>&1 | Out-String
        $whisperExit = $LASTEXITCODE
    } finally {
        if ($restoreNativeErrorPreferenceW) {
            $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreferenceW
        }
    }
    $ErrorActionPreference = $prevEAPWhisper
    if ($whisperExit -eq 0) {
        if ($whisperOutput -match "already matches") {
            step "whisper.cpp" "prebuilt up to date"
        } else {
            step "whisper.cpp" "prebuilt installed"
        }
        if ($StudioHomeIsCustom -and (Test-PathQuiet $WhisperCppDir "Container")) {
            Mark-StudioOwned -Path $WhisperCppDir
        }
    } elseif ($whisperExit -eq 3) {
        step "whisper.cpp" "install busy; keeping existing runtime" "Yellow"
    } elseif ($whisperExit -eq 2) {
        $requiredWhisperLlamaTag = "unknown"
        if ($whisperOutput -match "slim bundle requires llama\.cpp ([^;\s]+)") {
            $requiredWhisperLlamaTag = $Matches[1]
        }
        $installedWhisperLlamaTag = "unknown"
        $llamaMarker = Join-Path $LlamaCppDir "UNSLOTH_PREBUILT_INFO.json"
        if (Test-PathQuiet $llamaMarker "Leaf") {
            try {
                $markerPayload = Get-Content -LiteralPath $llamaMarker -Raw | ConvertFrom-Json
                if ($markerPayload.release_tag) { $installedWhisperLlamaTag = $markerPayload.release_tag }
            } catch {}
        }
        step "whisper.cpp" "no compatible prebuilt (installed llama.cpp $installedWhisperLlamaTag; whisper requires $requiredWhisperLlamaTag); curated whisper.cpp dictation is unavailable; publish paired releases in llama.cpp then whisper.cpp order; browser and Transformers dictation remain available" "Yellow"
    } else {
        step "whisper.cpp" "prebuilt install failed; curated whisper.cpp dictation is unavailable; retry setup or inspect verbose output; browser and Transformers dictation remain available" "Yellow"
    }
}

if ($StageRoot -and $NeedLlamaSourceBuild) {
    Exit-SetupFailure "Background staging cannot install system build tools for llama.cpp; retry with the foreground updater."
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
        Write-StudioLine ""
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
$HasGitForBuild = $null -ne (Get-Command git -ErrorAction SilentlyContinue)

# Check if existing llama-server matches current GPU mode. A CUDA-built binary
# on a now-CPU-only machine (or vice versa) needs to be rebuilt.
$NeedRebuild = $false
# A forced compile, a pinned PR or a custom source skips the prebuilt path and its
# phase 3.4 denial guard, so this can be the first probe to read inside the tree.
# A linked local dir is skipped: it reads through the junction into the user's own
# checkout, and nothing here is consumed on that path anyway.
$llamaBinState = if ($LocalLlamaCppLinked) { "Absent" } else { Get-PathState -Path $LlamaServerBin -PathType Leaf }
if ($llamaBinState -eq "Denied") {
    # Nothing proved this tree is ours here, so do not advise deleting it.
    Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install" -OwnershipUnverified:$StudioHomeIsCustom
}
if ($llamaBinState -eq "Present") {
    $CmakeCacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-PathQuiet $CmakeCacheFile "Leaf") {
        # A listed file can still deny the read, which Test-PathQuiet cannot see.
        try {
            $cachedCuda = Select-String -LiteralPath $CmakeCacheFile -Pattern 'GGML_CUDA:BOOL=ON' -Quiet
        } catch {
            if (-not (Test-AccessDeniedError $_)) { throw }
            Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install" -OwnershipUnverified:$StudioHomeIsCustom
        }
        if ($HasNvidiaSmi -and -not $cachedCuda) {
            Write-StudioLine "   Existing llama-server is CPU-only but GPU is available -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        } elseif (-not $HasNvidiaSmi -and $cachedCuda) {
            Write-StudioLine "   Existing llama-server was built with CUDA but no GPU detected -- rebuilding" -ForegroundColor Yellow
            $NeedRebuild = $true
        }
    }
}

# Install build tools now (last resort) rather than eagerly in Phase 1, so the
# prebuilt path stays fast. Same condition as the if/elseif chain below: a source
# build runs only when needed and no usable binary is already present. A linked
# local dir sets $NeedLlamaSourceBuild = $false, so this no-ops for that path.
$WillBuildLlamaFromSource = $NeedLlamaSourceBuild -and `
    -not ((Test-PathQuiet $LlamaServerBin "Leaf") -and -not $NeedRebuild -and $RequestedLlamaTag -ne "master")
if ($WillBuildLlamaFromSource) {
    if (-not $HasGitForBuild) {
        # Phase 1 keeps git optional, so only the automatic fallback after a failed prebuilt
        # download arrives here without it. Last chance to install: Invoke-SetupCommand
        # returns 0 for command-not-found, so a git-less clone misreports as a cmake failure.
        if ($null -ne (Get-Command winget -ErrorAction SilentlyContinue)) {
            try {
                Invoke-SetupCommand { winget install Git.Git --source winget --accept-package-agreements --accept-source-agreements } | Out-Null
                Refresh-Environment
            } catch { }
        }
        $HasGitForBuild = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
    }
    # Git first, then the toolchain: Ensure-BuildToolsForLlamaSourceBuild exits setup when
    # Build Tools cannot be installed, so running it first made the degraded path below
    # unreachable on a no-winget box, and elsewhere spent a multi-GB download on a clone
    # that cannot happen.
    if ($HasGitForBuild) {
        Ensure-BuildToolsForLlamaSourceBuild
        # refresh so the chain below sees a newly installed cmake
        $HasCmakeForBuild = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
    }
}

if ($LocalLlamaCppLinked) {
    # Local dir linked above -- honor the flag's contract: skip BOTH the prebuilt
    # download and the source build. Falling through here would run CMake inside
    # the user's checkout (via the junction) when it lacks build\bin\Release\llama-server.exe.
    Write-StudioLine ""
    step "llama.cpp" "linked (skipping build)"
} elseif (-not $NeedLlamaSourceBuild) {
    Write-StudioLine ""
    step "llama.cpp" "prebuilt (validated)"
} elseif ((Test-PathQuiet $LlamaServerBin "Leaf") -and -not $NeedRebuild -and $RequestedLlamaTag -ne "master") {
    # Skip rebuild only for pinned tags (e.g. b8635).  When the requested
    # tag is "master" (a moving target), always rebuild so the binary picks
    # up new model architecture support (e.g. Gemma 4).
    Write-StudioLine ""
    step "llama.cpp" "already built"
} elseif (-not $HasGitForBuild) {
    # Before cmake: the toolchain install is skipped without git, so cmake may be missing
    # purely as a consequence. Degrade rather than abort; the opt-in source triggers already
    # required git in Phase 1, so only the automatic fallback lands here.
    Write-StudioLine ""
    step "llama.cpp" "build skipped (git not available)" "Yellow"
    substep "The prebuilt download failed and a source build clones llama.cpp." "Yellow"
    substep "GGUF inference and export will not be available." "Yellow"
    substep "Install Git from https://git-scm.com/download/win and re-run setup." "Yellow"
    $script:LlamaCppDegraded = $true
} elseif (-not $HasCmakeForBuild) {
    Write-StudioLine ""
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
                    Write-StudioLine "[ERROR] CMake 4.2+ is required to build llama.cpp with the Visual Studio 2026 generator, and no older Visual Studio toolchain was found to fall back to." -ForegroundColor Red
                    Write-StudioLine "        Upgrade CMake from https://cmake.org/download/ and re-run, or use a prebuilt llama.cpp bundle." -ForegroundColor Red
                    Exit-SetupFailure "CMake cannot drive the Visual Studio 2026 generator"
                }
            }
        }
        substep "CMake can drive the $CmakeGenerator generator"
    }

    # CUDA resolved here (fail fast if none), after the final VS generator so its
    # .targets land in the toolset cmake actually uses.
    if ($HasNvidiaSmi) { Resolve-CudaToolkit -RequireOrExit }

    Write-StudioLine ""
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
    Write-StudioLine ""

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

    # Denied must not read as "no checkout here": the fresh-clone branch ends in
    # a swap that recursively removes this tree and moves the temp one over it,
    # under "Continue" and unchecked, so an unreadable child would leave a
    # half-deleted install behind. Stop while that is still avoidable.
    $llamaGitState = Get-PathState -Path (Join-Path $LlamaCppDir ".git")
    if ($llamaGitState -eq "Denied") {
        Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install"
    }
    if ($llamaGitState -eq "Present") {
        # why: in-place git mutation (remote set-url, checkout -B, clean -fdx)
        # rewrites $LlamaCppDir; mirror the prebuilt and temp-dir-swap guards
        # so an unrelated workspace .git tree is never silently overwritten.
        if ($StudioHomeIsCustom) {
            Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"
        }
        Write-StudioLine "   Syncing llama.cpp to $ResolvedSourceRef..." -ForegroundColor Gray
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
        Write-StudioLine "   Cloning llama.cpp @ $ResolvedSourceRef..." -ForegroundColor Gray
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
        Write-StudioLine ""
        Write-StudioLine "--- cmake configure ---" -ForegroundColor Cyan

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
                Write-StudioLine ""
                Write-StudioLine "   Hint: CUDA VS integration may be missing. Try running as admin:" -ForegroundColor Yellow
                Write-StudioLine "   Copy contents of:" -ForegroundColor Yellow
                Write-StudioLine "     <CUDA_PATH>\extras\visual_studio_integration\MSBuildExtensions" -ForegroundColor Yellow
                Write-StudioLine "   into:" -ForegroundColor Yellow
                $hintCustomizations = if ($VsInstallPath) { Get-VcBuildCustomizationsDir -VsInstallPath $VsInstallPath -Generator $CmakeGenerator } else { "<VS_PATH>\MSBuild\Microsoft\VC\v170\BuildCustomizations" }
                Write-StudioLine "     $hintCustomizations" -ForegroundColor Yellow
            }
        }
    }

    # -- Step C: Build llama-server --
    $NumCpu = Get-LlamaBuildJobs

    if ($BuildOk) {
        Write-StudioLine ""
        Write-StudioLine "--- cmake build (llama-server) ---" -ForegroundColor Cyan
        Write-StudioLine "   Parallel jobs: $NumCpu of $([Environment]::ProcessorCount) cores (RAM-capped; UNSLOTH_LLAMA_BUILD_JOBS overrides)" -ForegroundColor Gray
        Write-StudioLine ""

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
        Write-StudioLine ""
        Write-StudioLine "--- cmake build (llama-quantize) ---" -ForegroundColor Cyan
        $output = cmake --build $BuildDir --config Release --target llama-quantize -j $NumCpu 2>&1 | Out-String
        $cmakeBuildQuantizeExit = $LASTEXITCODE
        if ($cmakeBuildQuantizeExit -ne 0) {
            substep "llama-quantize build failed (GGUF export may be unavailable)" "Yellow"
            Write-LlamaFailureLog -Output $output
        }
    }

    # -- Step E: Build the DiffusionGemma visual server (optional, best-effort) --
    # An example target present on llama.cpp PR #24423; lets Unsloth serve
    # DiffusionGemma GGUFs without DG_VISUAL_BIN. No-op when not configured.
    if ($BuildOk) {
        $null = cmake --build $BuildDir --config Release --target llama-diffusion-gemma-visual-server -j $NumCpu 2>&1 | Out-String
    }

    # Swap temp build dir into final location (only if we built in a temp dir)
    if ($BuildOk -and $LlamaCppDir -ne $OriginalLlamaCppDir) {
        Assert-StudioOwnedOrAbsent -Path $OriginalLlamaCppDir -Label "llama.cpp install"
        if ((Get-PathState -Path $OriginalLlamaCppDir) -ne "Absent") {
            Remove-Item -LiteralPath $OriginalLlamaCppDir -Recurse -Force -ErrorAction SilentlyContinue
            # Any unreadable or locked child survives the removal, and Move-Item
            # then nests the build *inside* the leftovers instead of replacing
            # them. Both are non-terminating here, so check before destroying
            # more: the temp build is still whole at this point.
            $swapState = Get-PathState -Path $OriginalLlamaCppDir
            if ($swapState -eq "Denied") {
                Exit-PathAccessDenied -Path $OriginalLlamaCppDir -Label "llama.cpp install"
            }
            if ($swapState -ne "Absent") {
                step "llama.cpp" "could not replace the existing install at $OriginalLlamaCppDir" "Red"
                substep "Part of it survived removal; the new build is intact at $LlamaCppDir" "Yellow"
                substep "Close Unsloth and other llama.cpp users, or move that folder aside, then re-run setup" "Yellow"
                Exit-SetupFailure "llama.cpp install at $OriginalLlamaCppDir could not be replaced; the new build is at $LlamaCppDir" 3
            }
        }
        Move-Item -LiteralPath $LlamaCppDir -Destination $OriginalLlamaCppDir
        # A failed move is non-terminating too; without this setup would report
        # a build it never installed.
        if ((Get-PathState -Path $LlamaCppDir) -ne "Absent") {
            step "llama.cpp" "could not move the new build into $OriginalLlamaCppDir" "Red"
            Exit-SetupFailure "llama.cpp build at $LlamaCppDir could not be moved into $OriginalLlamaCppDir" 3
        }
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
    if ($BuildOk -and (Test-PathQuiet $LlamaServerBin "Leaf")) {
        step "llama.cpp" "built"
        $QuantizeBin = Join-Path $BuildDir "bin\Release\llama-quantize.exe"
        if (Test-PathQuiet $QuantizeBin "Leaf") {
            step "llama-quantize" "built"
        }
        step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
    } else {
        $altBin = Join-Path $BuildDir "bin\llama-server.exe"
        if ($BuildOk -and (Test-PathQuiet $altBin "Leaf")) {
            step "llama.cpp" "built"
            step "build time" "${totalMin}m ${totalSec}s" "DarkGray"
        } else {
            step "llama.cpp" "build failed at: $FailedStep (${totalMin}m ${totalSec}s); continuing" "Yellow"
            substep "To retry: delete $LlamaCppDir and re-run setup." "Yellow"
            $script:LlamaCppDegraded = $true
        }
    }
}

$llamaCppItem = Get-Item -LiteralPath $LlamaCppDir -Force -ErrorAction SilentlyContinue
$llamaCppIsLink = $llamaCppItem -and ($llamaCppItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
if (-not $llamaCppIsLink -and (
        -not $StudioHomeIsCustom -or
        (Test-PathQuiet (Join-Path $LlamaCppDir $StudioOwnedMarker) "Leaf") -or
        (Test-StudioOwnedAdoptable $LlamaCppDir)
    )) {
    Remove-AgentInstructionFiles -Roots @($LlamaCppDir)
}

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
$DoneLabel = if ($env:SKIP_STUDIO_BASE -eq "1") { "Unsloth Studio Setup Complete" } else { "Unsloth Studio Updated" }
if ($script:StudioVtOk -and -not $env:NO_COLOR) {
    Write-StudioLine ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
    if ($script:LlamaCppDegraded) {
        Write-StudioLine ("  " + (Get-StudioAnsi Warn) + "$DoneLabel (limited: llama.cpp unavailable)" + (Get-StudioAnsi Reset))
    } else {
        Write-StudioLine ("  " + (Get-StudioAnsi Title) + $DoneLabel + (Get-StudioAnsi Reset))
    }
    Write-StudioLine ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
} else {
    Write-StudioLine "  $Rule" -ForegroundColor DarkGray
    if ($script:LlamaCppDegraded) {
        Write-StudioLine "  $DoneLabel (limited: llama.cpp unavailable)" -ForegroundColor Yellow
    } else {
        Write-StudioLine "  $DoneLabel" -ForegroundColor Green
    }
    Write-StudioLine "  $Rule" -ForegroundColor DarkGray
}
step "launch" "unsloth studio -p 8888"
substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
Write-StudioLine ""

# Match studio/setup.sh: exit non-zero for degraded llama.cpp when called
# from install.ps1 (SKIP_STUDIO_BASE=1) so the installer can detect the
# failure. Direct 'unsloth studio update' does not set SKIP_STUDIO_BASE,
# so it keeps degraded installs successful.
if ($script:LlamaCppDegraded -and $env:SKIP_STUDIO_BASE -eq "1") {
    # Tauri mode reports instead of aborting, exactly as setup.sh does. install.ps1
    # turns any non-zero status from here into Exit-InstallFailure, and install.rs
    # turns that into "Installation failed", so on Windows too a single transient
    # prebuilt download failure would throw away a first-launch install whose own
    # footer just said complete. [TAURI:PROGRESS] (not [TAURI:STEP], which would
    # push the frontend step counter past the seven INSTALL_STEPS entries) reaches
    # the user as install-progress-detail text.
    if (@("1", "true") -contains $env:UNSLOTH_TAURI_MODE) {
        [Console]::Out.WriteLine("[TAURI:PROGRESS] llama.cpp unavailable; GGUF inference is disabled until 'unsloth studio update' succeeds")
        [Console]::Out.Flush()
    } else {
        Exit-SetupFailure "llama.cpp setup did not produce a usable server"
    }
}
