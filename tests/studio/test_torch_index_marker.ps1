#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for studio/setup.ps1's torch-index MARKER helpers
# (Get-NormalizedIndexUrl, Read-TorchIndexMarker, Write-TorchIndexMarker,
# Test-MarkerPinMismatch, Test-RocmKnown211Version). These converge the ROCm/gfx
# pin-change detection across install.sh / install_python_stack.py / setup.ps1 /
# install.ps1. Pure helpers, AST-extracted and run in-process -- no GPU/venv.
# Run: pwsh -NoProfile -File tests/studio/test_torch_index_marker.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

# Parse setup.ps1 (also a syntax gate) and extract the helpers.
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }

# The marker filename constant the helpers close over.
$TorchIndexMarkerName = ".unsloth-torch-index"

foreach ($name in @(
    "Get-NormalizedFamilyLeaf", "Get-NormalizedIndexUrl", "Get-TorchIndexMarkerPath",
    "Read-TorchIndexMarker", "Write-TorchIndexMarker", "Test-MarkerPinMismatch",
    "Test-RocmKnown211Version"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in setup.ps1, found $($fn.Count)" }
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

Write-Host "Get-NormalizedIndexUrl (trim / strip trailing slash / lowercase leaf)"
# -ceq: PowerShell -eq is case-INsensitive for strings, which would make these
# case-normalization checks vacuous (any casing would pass).
Check "trailing slashes + known family leaf lowered" `
    ((Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all///") -ceq "https://repo.amd.com/rocm/whl/gfx120x-all")
Check "whitespace trimmed" `
    ((Get-NormalizedIndexUrl "  https://download.pytorch.org/whl/cu128  ") -ceq "https://download.pytorch.org/whl/cu128")
Check "host case preserved, unknown custom leaf keeps case" `
    ((Get-NormalizedIndexUrl "https://Mirror.Local/Simple/") -ceq "https://Mirror.Local/Simple")
Check "gfx120X-all == gfx120x-all after normalize" `
    ((Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all") -eq (Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120x-all"))
Check "empty -> null" ($null -eq (Get-NormalizedIndexUrl "   "))

Write-Host "Write/Read-TorchIndexMarker (round trip, atomic, blank ignored)"
$venv = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-marker-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $venv | Out-Null
try {
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://repo.amd.com/rocm/whl/gfx1151"
    Check "marker written verbatim" `
        ((Read-TorchIndexMarker -VenvDir $venv) -eq "https://repo.amd.com/rocm/whl/gfx1151")
    # Per-arch switch overwrites the marker.
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all"
    Check "marker overwritten on re-install" `
        ((Read-TorchIndexMarker -VenvDir $venv) -eq "https://repo.amd.com/rocm/whl/gfx120X-all")
    # Blank URL ignored -- prior marker kept.
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "   "
    Check "blank url leaves prior marker intact" `
        ((Read-TorchIndexMarker -VenvDir $venv) -eq "https://repo.amd.com/rocm/whl/gfx120X-all")
    # No stray temp file left behind.
    $tmpLeft = @(Get-ChildItem -LiteralPath $venv -Filter "$TorchIndexMarkerName.*.tmp" -ErrorAction SilentlyContinue).Count
    Check "no stray temp file left" ($tmpLeft -eq 0)

    Write-Host "Test-MarkerPinMismatch (exact compare; null when no marker)"
    # Marker gfx120X-all now recorded. A gfx1151 pin differs -> mismatch (#2543).
    Check "gfx marker vs gfx1151 pin -> mismatch" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://repo.amd.com/rocm/whl/gfx1151") -eq $true)
    # Same pin (trailing slash, case) -> not a mismatch (no reinstall loop).
    Check "same pin (slash/case) -> no mismatch" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://repo.amd.com/rocm/whl/gfx120x-all/") -eq $false)
    # Custom URL change /simple -> /current is a mismatch (#2544).
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://mirror.local/simple"
    Check "custom /simple marker vs /current pin -> mismatch" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://mirror.local/current") -eq $true)

    Write-Host "Read-TorchIndexMarker (missing / empty -> null)"
    Remove-Item -LiteralPath (Get-TorchIndexMarkerPath -VenvDir $venv) -Force
    Check "missing marker -> null" ($null -eq (Read-TorchIndexMarker -VenvDir $venv))
    Set-Content -LiteralPath (Get-TorchIndexMarkerPath -VenvDir $venv) -Value "" -NoNewline
    Check "empty marker -> null" ($null -eq (Read-TorchIndexMarker -VenvDir $venv))
    Check "no marker -> Test-MarkerPinMismatch null" `
        ($null -eq (Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://x/rocm7.2"))
} finally {
    Remove-Item -LiteralPath $venv -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Test-RocmKnown211Version (KNOWN-2.11 set == rocm7.2 only)"
Check "rocm7.2 -> true"  (Test-RocmKnown211Version -Major 7 -Minor 2)
Check "rocm7.1 -> false" (-not (Test-RocmKnown211Version -Major 7 -Minor 1))
Check "rocm7.3 -> false" (-not (Test-RocmKnown211Version -Major 7 -Minor 3))
Check "rocm8.0 -> false" (-not (Test-RocmKnown211Version -Major 8 -Minor 0))

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
