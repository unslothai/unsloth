#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for studio/setup.ps1's torch-index MARKER helpers
# (Get-NormalizedIndexUrl, Read-TorchIndexMarker, Write-TorchIndexMarker,
# Test-MarkerPinMismatch, Test-RocmKnown211Version), which converge the ROCm/gfx
# pin-change detection across install.sh / py / setup.ps1 / install.ps1. AST-extracted,
# run in-process.
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
    "Get-TorchIndexLeaf", "Remove-IndexUrlCredentials",
    "Test-PipRocmFamilyLeaf", "Get-NormalizedFamilyLeaf", "Get-NormalizedIndexUrl",
    "Get-TorchIndexMarkerPath",
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
# -ceq: PowerShell -eq is case-INsensitive, which would make these case checks vacuous.
Check "trailing slashes + known family leaf lowered" `
    ((Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all///") -ceq "https://repo.amd.com/rocm/whl/gfx120x-all")
Check "whitespace trimmed" `
    ((Get-NormalizedIndexUrl "  https://download.pytorch.org/whl/cu128  ") -ceq "https://download.pytorch.org/whl/cu128")
Check "host case preserved, unknown custom leaf keeps case" `
    ((Get-NormalizedIndexUrl "https://Mirror.Local/Simple/") -ceq "https://Mirror.Local/Simple")
Check "gfx120X-all == gfx120x-all after normalize" `
    ((Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all") -eq (Get-NormalizedIndexUrl "https://repo.amd.com/rocm/whl/gfx120x-all"))
Check "empty -> null" ($null -eq (Get-NormalizedIndexUrl "   "))

Write-Host "Get-NormalizedFamilyLeaf (exact rocm/gfx/cpu/cu lowered; custom keeps case)"
# EXACT rocm/gfx/cpu/cu families are lowercased; a suffixed leaf (rocm7.2-Private,
# rocm-Rel-7.2.1) is a custom pin whose case must survive so a case-only change is a
# real mismatch (Test-PipRocmFamilyLeaf gate).
Check "rocm7.2 family lowered"          ((Get-NormalizedFamilyLeaf "ROCm7.2") -ceq "rocm7.2")
Check "gfx120X-all family lowered"      ((Get-NormalizedFamilyLeaf "GFX120X-all") -ceq "gfx120x-all")
Check "cu128 family lowered"            ((Get-NormalizedFamilyLeaf "CU128") -ceq "cu128")
Check "suffixed rocm7.2-Private kept"   ((Get-NormalizedFamilyLeaf "rocm7.2-Private") -ceq "rocm7.2-Private")
Check "rocm7-Current kept"              ((Get-NormalizedFamilyLeaf "rocm7-Current") -ceq "rocm7-Current")
Check "rocm-Rel-7.2.1 kept"             ((Get-NormalizedFamilyLeaf "rocm-Rel-7.2.1") -ceq "rocm-Rel-7.2.1")
Check "cu128-Private kept"              ((Get-NormalizedFamilyLeaf "cu128-Private") -ceq "cu128-Private")

Write-Host "Test-PipRocmFamilyLeaf (exact rocm/gfx family gate)"
Check "rocm7.2 -> family"        (Test-PipRocmFamilyLeaf "rocm7.2")
Check "rocm7 -> family"          (Test-PipRocmFamilyLeaf "rocm7")
Check "gfx1151 -> family"        (Test-PipRocmFamilyLeaf "gfx1151")
Check "rocm7.2-private -> custom" (-not (Test-PipRocmFamilyLeaf "rocm7.2-private"))
Check "rocm7-current -> custom"  (-not (Test-PipRocmFamilyLeaf "rocm7-current"))
Check "rocm-current -> custom"   (-not (Test-PipRocmFamilyLeaf "rocm-current"))
Check "rocm7.2.1 -> custom"      (-not (Test-PipRocmFamilyLeaf "rocm7.2.1"))
Check "cpu -> not rocm"          (-not (Test-PipRocmFamilyLeaf "cpu"))

Write-Host "Remove-IndexUrlCredentials (userinfo never persists or prints)"
Check "user:token@ stripped" `
    ((Remove-IndexUrlCredentials "https://user:tok@mirror.local/simple") -ceq "https://mirror.local/simple")
Check "credential-free url unchanged" `
    ((Remove-IndexUrlCredentials "https://mirror.local/simple") -ceq "https://mirror.local/simple")
Check "@ in path preserved" `
    ((Remove-IndexUrlCredentials "https://u:p@h/pa@th") -ceq "https://h/pa@th")
# A token can also ride in the query string / fragment of a private feed URL;
# both are dropped so the secret never lands in the marker or logged output.
Check "query token stripped" `
    ((Remove-IndexUrlCredentials "https://mirror.local/simple?token=SECRET") -ceq "https://mirror.local/simple")
Check "fragment stripped" `
    ((Remove-IndexUrlCredentials "https://mirror.local/simple#tok") -ceq "https://mirror.local/simple")
Check "userinfo and query both stripped" `
    ((Remove-IndexUrlCredentials "https://u:p@mirror.local/simple?token=SECRET") -ceq "https://mirror.local/simple")
Check "query stripped on host-only url" `
    ((Remove-IndexUrlCredentials "https://mirror.local?token=SECRET") -ceq "https://mirror.local")
# Backward compatibility: an OLD marker that recorded credentials compares equal to the
# same pin with or without them (normalization strips both sides).
Check "normalize: creds on either side compare equal" `
    ((Get-NormalizedIndexUrl "https://user:tok@x/cu128/") -ceq (Get-NormalizedIndexUrl "https://x/cu128"))

Write-Host "Get-TorchIndexLeaf (query/fragment dropped before classification)"
Check "query dropped" ((Get-TorchIndexLeaf "https://m/whl/cu128?token=x") -ceq "cu128")
Check "fragment dropped" ((Get-TorchIndexLeaf "https://m/whl/cu128#frag") -ceq "cu128")
Check "plain leaf" ((Get-TorchIndexLeaf "https://m/whl/gfx120X-all/") -ceq "gfx120x-all")

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
    # Credentials never persist in the marker file.
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://user:sekrit@mirror.local/simple"
    $markerBody = Read-TorchIndexMarker -VenvDir $venv
    Check "marker stores credential-free url" ($markerBody -ceq "https://mirror.local/simple")
    Check "marker body has no secret" (-not ($markerBody -like "*sekrit*"))
    # A query-carried token must not persist in the marker either.
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://mirror.local/simple?token=sekrit2"
    $markerBody2 = Read-TorchIndexMarker -VenvDir $venv
    Check "marker drops query token" ($markerBody2 -ceq "https://mirror.local/simple")
    Check "marker body has no query secret" (-not ($markerBody2 -like "*sekrit2*"))
    # A cred-bearing pin still matches the stripped marker (no reinstall loop).
    Check "cred pin vs stripped marker -> no mismatch" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://user:sekrit@mirror.local/simple") -eq $false)
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all"

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
    # Case-only custom-index change /Simple -> /simple is a mismatch: normalization keeps
    # unknown-leaf case and the compare is -cne (a case-insensitive -ne would skip it).
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://mirror.local/Simple"
    Check "custom /Simple marker vs /simple pin -> mismatch (case-sensitive)" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://mirror.local/simple") -eq $true)
    # A genuine family leaf differing only in case is NOT a mismatch (family leaves are
    # lowercased by normalization, so gfx120X-all == gfx120x-all).
    Write-TorchIndexMarker -VenvDir $venv -IndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all"
    Check "family gfx120X-all marker vs gfx120x-all pin -> no mismatch" `
        ((Test-MarkerPinMismatch -VenvDir $venv -PinUrl "https://repo.amd.com/rocm/whl/gfx120x-all") -eq $false)

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
