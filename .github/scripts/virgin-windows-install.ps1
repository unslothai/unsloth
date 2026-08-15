# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Runs INSIDE a Windows container once virgin-windows-probe.ps1 has proved there is no
# toolchain: install.ps1 as a user on a bare Windows box runs it, then the same
# assertions the hosted Windows leg makes.

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
# install.ps1:2885-2888 prompts `Start Unsloth Studio now? [Y/n]` when UserInteractive is
# true and stdin is not redirected -- both hold under `docker exec` -- so without this the
# installer blocks forever on Read-Host and the job dies on timeout with no diagnosis.
$env:UNSLOTH_SKIP_AUTOSTART = '1'
# install.ps1:254/258 joins $env:USERPROFILE with no null guard; an explicit root also
# keeps the container's state in one directory.
$env:UNSLOTH_STUDIO_HOME = 'C:\studio-home'
$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
# Without this uv's output is discarded on success and nobuild can only report "none".
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

# Deliberately NOT setting [Net.ServicePointManager]::SecurityProtocol: install.ps1 does
# not either, so setting it here would hide a real installer bug. The probe already
# reported whether the default negotiates TLS 1.2.

# ── Run the installer exactly as the desktop launches it ──────────────────────
Section 'install'
if (-not (Test-Path -LiteralPath $Installer)) {
    Write-Host "::error::installer not found at $Installer"
    exit 1
}
Write-Host "installer: $Installer ($((Get-Content -LiteralPath $Installer).Count) lines)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# A child powershell.exe, not dot-sourcing: the shape install.rs:325-339 uses, and it
# yields a real process exit code rather than the last statement's value.
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
    # As the Linux leg: an installer that exits 0 having done nothing must not pass.
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
    # issue #8490: the generated .exe launchers above are unsigned, so Application
    # Control can deny them. The .cmd shim runs the managed interpreter instead and
    # is the escape hatch a locked-down machine has, so its absence is a failure.
    $cmdShim = Join-Path $env:UNSLOTH_STUDIO_HOME 'bin\unsloth.cmd'
    if (Test-Path -LiteralPath $cmdShim) { Write-Host "unsloth CLI shim: $cmdShim" }
    else { $failures += "installer exited 0 but left no policy-safe CLI shim at $cmdShim" }
}

Section 'assert: torch imports'
# On the hosted runner this proves less than it looks: the image ships the VC++ runtime
# in System32, so Test-VCRedistInstalled (setup.ps1:875) short-circuits before it needs
# winget. This container is the first place that is not true, so a failure here is a
# genuine finding about bare Windows.
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
    # install.ps1:1098, the no-winget branch. A container has no Store and so no App
    # Installer, which puts the python.org + astral.sh fallback under test.
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
# Shared with the hosted Windows legs so the sdist allowlist lives in one place; it
# prints its own diagnosis, so only the verdict is folded in.
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
