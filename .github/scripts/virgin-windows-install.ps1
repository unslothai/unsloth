# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Runs INSIDE a Windows container, after virgin-windows-probe.ps1 has proved the
# environment has no toolchain. Runs install.ps1 the way a real user on a bare
# Windows box would, then asserts the same things the hosted Windows leg asserts.

[CmdletBinding()]
param(
    [string] $Installer = 'C:\ci\install.ps1',
    [string] $LogPath   = 'C:\ci-out\install.log',
    [string] $Overlay   = ''
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Section($t) { Write-Host ""; Write-Host "=== $t ===" }

# ── Environment the installer needs to be non-interactive ─────────────────────
Section 'install environment'
# install.ps1:2885-2888 prompts `Start Unsloth Studio now? [Y/n]` when
# [Environment]::UserInteractive is true and stdin is not redirected. Both hold in a
# `docker exec` session, so without this the installer BLOCKS FOREVER on Read-Host
# and the job dies on timeout with no diagnosis.
$env:UNSLOTH_SKIP_AUTOSTART = '1'
# install.ps1:254/258 joins $env:USERPROFILE with no null guard. Setting the install
# root explicitly also keeps the container's state entirely under one directory.
$env:UNSLOTH_STUDIO_HOME = 'C:\studio-home'
$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
# Without this, uv's output is discarded on success and the nobuild check below can
# only ever report "built: none".
$env:UNSLOTH_VERBOSE = '1'
if ($Overlay) {
    $env:UNSLOTH_CI_SOURCE_OVERLAY = $Overlay
    Write-Host "overlay: $Overlay"
} else {
    Remove-Item Env:\UNSLOTH_CI_SOURCE_OVERLAY -ErrorAction SilentlyContinue
    Write-Host "overlay: (none -- tests the released wheel)"
}
foreach ($v in 'UNSLOTH_STUDIO_HOME', 'UNSLOTH_SKIP_AUTOSTART', 'UNSLOTH_VERBOSE', 'UNSLOTH_CI_SOURCE_OVERLAY') {
    Write-Host ("  {0,-32} {1}" -f $v, [System.Environment]::GetEnvironmentVariable($v))
}

# NOTE: deliberately NOT setting [Net.ServicePointManager]::SecurityProtocol here.
# install.ps1 does not set it either, so setting it in the harness would hide a real
# installer bug. The probe already reported whether the default negotiates TLS 1.2.

# ── Run the installer exactly as the desktop launches it ──────────────────────
Section 'install'
if (-not (Test-Path -LiteralPath $Installer)) {
    Write-Host "::error::installer not found at $Installer"
    exit 1
}
Write-Host "installer: $Installer ($((Get-Content -LiteralPath $Installer).Count) lines)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# A child powershell.exe, not dot-sourcing: same shape as install.rs:325-339, and it
# gives a real process exit code instead of whatever the last statement returned.
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
    -File $Installer *>&1 | Tee-Object -FilePath $LogPath
$rc = $LASTEXITCODE
$sw.Stop()
Write-Host ""
Write-Host "installer exit code: $rc  (after $([int]$sw.Elapsed.TotalSeconds)s)"

# ── Assertions ────────────────────────────────────────────────────────────────
$failures = @()
$venv = Join-Path $env:UNSLOTH_STUDIO_HOME 'unsloth_studio'
$venvPy = Join-Path $venv 'Scripts\python.exe'

Section 'assert: the install produced something usable'
if ($rc -ne 0) {
    $failures += "installer exited $rc"
} else {
    # Mirrors the Linux leg's "Assert the install is actually usable": an installer
    # that exits 0 having done nothing must not pass.
    if (-not (Test-Path -LiteralPath $venvPy)) {
        $failures += "installer exited 0 but left no managed Python at $venvPy"
        Get-ChildItem -Path $env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue | Format-Table | Out-String | Write-Host
    } else {
        Write-Host "managed python: $venvPy"
        & $venvPy -V
    }
    foreach ($cli in (Join-Path $venv 'Scripts\unsloth.exe'), (Join-Path $env:UNSLOTH_STUDIO_HOME 'bin\unsloth.exe')) {
        if (Test-Path -LiteralPath $cli) { Write-Host "unsloth CLI: $cli" }
        else { $failures += "installer exited 0 but left no unsloth CLI at $cli" }
    }
}

Section 'assert: torch imports'
# On the hosted runner this proves less than it looks like: the runner image ships
# the VC++ 2015-2022 runtime in System32, so Test-VCRedistInstalled (setup.ps1:875)
# short-circuits before it needs winget. THIS container is the first environment in
# which that is not true, so a failure here is a genuine finding about bare Windows,
# not a CI artefact.
if (Test-Path -LiteralPath $venvPy) {
    foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') {
        $p = Join-Path $env:WINDIR "System32\$dll"
        Write-Host ("  System32\{0,-20} {1}" -f $dll, $(if (Test-Path $p) { 'PRESENT' } else { 'ABSENT' }))
    }
    & $venvPy -c "import ctypes.util; print('find_library(vcruntime140):', ctypes.util.find_library('vcruntime140'))"
    & $venvPy -c "import torch; print('torch', torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        $failures += "torch failed to import from the managed Python (VC++ runtime missing?)"
    }
    $global:LASTEXITCODE = 0
} else {
    Write-Host "skipped: no managed Python"
}

Section "assert: the installer took the no-winget path"
if (Test-Path -LiteralPath $LogPath) {
    # install.ps1:1098, the no-winget branch. A container has no Microsoft Store and
    # therefore no App Installer, so this is the fallback path (python.org + astral.sh)
    # under test -- the whole reason a container is a good harness.
    $noWinget = 'will require Python + uv to be already installed'
    if (Select-String -Path $LogPath -Pattern $noWinget -SimpleMatch -Quiet) {
        Write-Host "confirmed: installer reported winget as unavailable and used the fallback path"
    } else {
        $failures += "installer never reported winget as unavailable; it did not take the no-winget path"
    }
}

if ($Overlay -and $rc -eq 0) {
    Section 'assert: this ref was really put under test'
    if (Select-String -Path $LogPath -Pattern 'CI: overlaying source checkout' -SimpleMatch -Quiet) {
        Write-Host "overlay applied; this leg exercised this ref's Python"
    } else {
        $failures += "leg is marked overlay but the installer never overlaid the checkout, so it only tested the released package"
    }
}

Section 'assert: no non-allowlisted source build'
# Shared with the hosted Windows legs so the sdist allowlist lives in one place; the
# script prints its own diagnosis, so only the verdict is folded in here.
$nobuild = Join-Path $PSScriptRoot 'assert-nobuild.ps1'
if (-not (Test-Path -LiteralPath $nobuild)) {
    $failures += "assert-nobuild.ps1 is missing next to this script, so the no-build contract went unchecked"
} else {
    & $nobuild -LogPath $LogPath
    if ($LASTEXITCODE -ne 0) { $failures += "a non-allowlisted source build appears in the install log" }
}

# ── Verdict ───────────────────────────────────────────────────────────────────
Section 'verdict'
if ($failures.Count -gt 0) {
    Write-Host "---- last 60 lines of the install log ----"
    Get-Content -LiteralPath $LogPath -Tail 60 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    Write-Host "-----------------------------------------"
    foreach ($f in $failures) { Write-Host "::error::$f" }
    Write-Host "VIRGIN WINDOWS CONTAINER INSTALL FAILED ($($failures.Count) problem(s))"
    exit 1
}
Write-Host "VIRGIN WINDOWS CONTAINER INSTALL PASSED"
exit 0
