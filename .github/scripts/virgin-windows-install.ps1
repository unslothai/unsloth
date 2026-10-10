# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Runs inside a Windows container after virgin-windows-probe.ps1: install.ps1 then the hosted leg's assertions.

[CmdletBinding()]
param(
    [string] $Installer = 'C:\ci\install.ps1',
    [string] $LogPath   = 'C:\ci-out\install.log',
    [string] $Overlay   = ''
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Section($t) { Write-Host ""; Write-Host "=== $t ===" }

Section 'install environment'
# Without this install.ps1 blocks forever on its autostart Read-Host under `docker exec`.
$env:UNSLOTH_SKIP_AUTOSTART = '1'
# install.ps1 joins $env:USERPROFILE with no null guard.
$env:UNSLOTH_STUDIO_HOME = 'C:\studio-home'
$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
# Without this uv's output is discarded and nobuild can only report "none".
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

# SecurityProtocol is deliberately not set: install.ps1 does not set it either.

Section 'install'
if (-not (Test-Path -LiteralPath $Installer)) {
    Write-Host "::error::installer not found at $Installer"
    exit 1
}
Write-Host "installer: $Installer ($((Get-Content -LiteralPath $Installer).Count) lines)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# A child powershell.exe like install.rs uses, for a real process exit code.
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
    -File $Installer *>&1 | Tee-Object -FilePath $LogPath
$rc = $LASTEXITCODE
$sw.Stop()
Write-Host ""
Write-Host "installer exit code: $rc  (after $([int]$sw.Elapsed.TotalSeconds)s)"

$failures = @()
$venv = Join-Path $env:UNSLOTH_STUDIO_HOME 'unsloth_studio'
$venvPy = Join-Path $venv 'Scripts\python.exe'

Section 'assert: the install produced something usable'
if ($rc -ne 0) {
    $failures += "installer exited $rc"
} else {
    if (-not (Test-Path -LiteralPath $venvPy)) {
        $failures += "installer exited 0 but left no managed Python at $venvPy"
        Get-ChildItem -Path $env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue | Format-Table | Out-String | Write-Host
    } else {
        Write-Host "managed python: $venvPy"
        & $venvPy -V
    }
    foreach ($cli in (Join-Path $venv 'Scripts\unsloth.exe'), (Join-Path $env:UNSLOTH_STUDIO_HOME 'bin\unsloth.exe')) {
        if (Test-Path -LiteralPath $cli) {
            Write-Host "unsloth CLI: $cli"
            # The console script imports the whole command tree, so missing deps surface here.
            & $cli --version
            if ($LASTEXITCODE -ne 0) {
                $failures += "unsloth CLI at $cli is on disk but does not run (exit $LASTEXITCODE)"
            }
        }
        else { $failures += "installer exited 0 but left no unsloth CLI at $cli" }
    }
    # The .cmd shim is the fallback when Application Control blocks the unsigned .exe launchers.
    $cmdShim = Join-Path $env:UNSLOTH_STUDIO_HOME 'bin\unsloth.cmd'
    if (Test-Path -LiteralPath $cmdShim) { Write-Host "unsloth CLI shim: $cmdShim" }
    else { $failures += "installer exited 0 but left no policy-safe CLI shim at $cmdShim" }
}

Section 'assert: torch imports'
# The hosted image ships the VC++ runtime; this container is the first real test without it.
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
    # The no-winget branch: containers have no App Installer.
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
$nobuild = Join-Path $PSScriptRoot 'assert-nobuild.ps1'
if (-not (Test-Path -LiteralPath $nobuild)) {
    $failures += "assert-nobuild.ps1 is missing next to this script, so the no-build contract went unchecked"
} else {
    & $nobuild -LogPath $LogPath
    if ($LASTEXITCODE -ne 0) { $failures += "a non-allowlisted source build appears in the install log" }
}

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
