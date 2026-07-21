#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for install.ps1's torch release preservation helpers
# (ConvertTo-TorchNumericRelease, Test-TorchReleaseInWindow, Get-PreviousTorchPin),
# the Windows port of install.sh's _previous_torch_pin (PR 7250). Pure helpers,
# AST-extracted and run in-process -- no GPU/venv needed.
# Run: pwsh -NoProfile -File tests/studio/test_previous_torch_pin.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

# --- Parse install.ps1 (also serves as a syntax gate) and extract the helpers ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

foreach ($name in @("ConvertTo-TorchNumericRelease", "Test-TorchReleaseInWindow", "Get-PreviousTorchPin")) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in install.ps1, found $($fn.Count)" }
    # Pure helpers (no exit / external calls) -- safe to define in this scope.
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$savedUpgrade = $env:UNSLOTH_TORCH_UPGRADE
try {
    Remove-Item Env:UNSLOTH_TORCH_UPGRADE -ErrorAction SilentlyContinue
    $win = "torch>=2.4,<2.12.0"

    # --- ConvertTo-TorchNumericRelease: accepted stable versions ---
    foreach ($v in @("2.10.0", "2.10.0+cpu", "2.10.0+cu126", "2.10.1+cu130", "2.9.1+rocm7.2.1", "2.9.0+xpu", "2.10")) {
        $r = ConvertTo-TorchNumericRelease $v
        Check "release accepts $v" ($null -ne $r)
    }
    $r = ConvertTo-TorchNumericRelease "2.10.1+cu128"
    Check "release strips only the +local tag" ($r.PublicBase -eq "2.10.1" -and $r.Minor -eq 10)

    # --- Rejected: nightly/dev/rc/alpha/garbage never pin ---
    foreach ($v in @("", "   ", "2.11.0.dev20260701+cu130", "2.9.0a0+gitabc123", "2.11.0rc1+cu130",
                     ".2.10", "2.10.", "2..10", "2.x.0", "not-a-version",
                     "Traceback (most recent call last):", "99999999999999999999.1")) {
        Check "release rejects '$v'" ($null -eq (ConvertTo-TorchNumericRelease $v))
    }

    # --- Test-TorchReleaseInWindow ---
    $cases = @(
        @{ v = "2.4.0";  c = "torch>=2.4,<2.12.0";    ok = $true;  n = "at the floor" }
        @{ v = "2.3.1";  c = "torch>=2.4,<2.12.0";    ok = $false; n = "below the floor" }
        @{ v = "2.11.0"; c = "torch>=2.4,<2.12.0";    ok = $true;  n = "just below the ceiling" }
        @{ v = "2.12.0"; c = "torch>=2.4,<2.12.0";    ok = $false; n = "at the ceiling" }
        @{ v = "2.13.0"; c = "torch>=2.4,<2.12.0";    ok = $false; n = "above the ceiling" }
        @{ v = "2.10.0"; c = "torch>=2.11.0,<2.12.0"; ok = $false; n = "2.11 floor rejects 2.10" }
        @{ v = "2.11.0"; c = "torch>=2.11.0,<2.12.0"; ok = $true;  n = "2.11 floor accepts 2.11" }
        @{ v = "2.11.0"; c = "torch>=2.4,<2.13.0";    ok = $true;  n = "future 2.13 ceiling keeps 2.11" }
        @{ v = "2.12.1"; c = "torch>=2.4,<2.13.0";    ok = $true;  n = "future 2.13 ceiling keeps 2.12" }
    )
    foreach ($t in $cases) {
        $rel = ConvertTo-TorchNumericRelease $t.v
        Check ("window: " + $t.n) ((Test-TorchReleaseInWindow -Release $rel -Constraint $t.c) -eq $t.ok)
    }
    # Malformed constraints fail closed.
    $rel = ConvertTo-TorchNumericRelease "2.10.0"
    Check "window: malformed constraint fails closed" (-not (Test-TorchReleaseInWindow -Release $rel -Constraint "torch"))
    Check "window: exact-pin constraint fails closed" (-not (Test-TorchReleaseInWindow -Release $rel -Constraint "torch==2.10.0"))

    # --- Get-PreviousTorchPin: exact-release pin, sh parity ---
    $pin = Get-PreviousTorchPin -TorchVersion "2.10.0+cu128" -Constraint $win
    Check "pin keeps 2.10.0 (exact release, sh parity)" ($pin.TorchSpec -eq "torch==2.10.0")
    Check "pin pairs torchvision to the kept minor" ($pin.VisionSpec -eq "torchvision==0.25.*")
    Check "pin pairs torchaudio to the kept minor" ($pin.AudioSpec -eq "torchaudio==2.10.*")
    $pin = Get-PreviousTorchPin -TorchVersion "2.9.1+rocm7.2.1" -Constraint $win
    Check "pin keeps 2.9.1 with 0.24.*/2.9.* companions" (
        $pin.TorchSpec -eq "torch==2.9.1" -and $pin.VisionSpec -eq "torchvision==0.24.*" -and $pin.AudioSpec -eq "torchaudio==2.9.*")
    $pin = Get-PreviousTorchPin -TorchVersion "2.11.0+cpu" -Constraint $win
    Check "pin keeps 2.11.0 under a future-widened window" ($pin.TorchSpec -eq "torch==2.11.0")

    # No previous version / out-of-window / non-stable -> no pin.
    Check "no pin without a previous version" ($null -eq (Get-PreviousTorchPin -TorchVersion "" -Constraint $win))
    Check "no pin for a below-floor release" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.3.1+cpu" -Constraint $win))
    Check "raised ROCm floor rejects keeping 2.10" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.10.0+rocm7.1" -Constraint "torch>=2.11.0,<2.12.0"))
    Check "no pin for a nightly build" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.11.0.dev20260701+cu130" -Constraint $win))
    Check "no pin for an unsupported 2.12 under a <2.12 window" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.12.0+cu130" -Constraint $win))

    # --- UNSLOTH_TORCH_UPGRADE opt-out (exact string '1', sh parity) ---
    $env:UNSLOTH_TORCH_UPGRADE = "1"
    Check "UNSLOTH_TORCH_UPGRADE=1 disables the pin" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.10.0+cpu" -Constraint $win))
    $env:UNSLOTH_TORCH_UPGRADE = "0"
    Check "UNSLOTH_TORCH_UPGRADE=0 keeps the pin" ($null -ne (Get-PreviousTorchPin -TorchVersion "2.10.0+cpu" -Constraint $win))
} finally {
    if ($null -ne $savedUpgrade) { $env:UNSLOTH_TORCH_UPGRADE = $savedUpgrade }
    else { Remove-Item Env:UNSLOTH_TORCH_UPGRADE -ErrorAction SilentlyContinue }
}

# --- Structural wiring (source assertions) ---
$src = Get-Content $installPath -Raw
Check "probe runs before the rollback move" (
    $src.IndexOf('$script:PrevTorchVer') -ge 0 -and
    $src.IndexOf('$script:PrevTorchVer') -lt $src.IndexOf('Start-StudioVenvRollback -ExistingDir'))
Check "pin decision cites the UNSLOTH_TORCH_UPGRADE escape hatch" ($src -match 'UNSLOTH_TORCH_UPGRADE=1 to get the newest')
Check "kept-release fallback clears the pin" ($src -match '\$script:PrevTorchPin\s*=\s*\$null')
Check "kept release exported for setup.ps1" ($src -match 'UNSLOTH_KEPT_TORCH')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
exit 0
