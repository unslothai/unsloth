#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for studio/setup.ps1's pinned-torch-index stale-venv helpers
# (Test-RocmGfx211Leaf, Test-CudaFamilyLeaf, Get-RocmPinStaleTags). Pure helpers,
# AST-extracted and run in-process. Mirrors the Python _rocm_pin_family_mismatch /
# _is_cuda_family_leaf tests.
# Run: pwsh -NoProfile -File tests/studio/test_setup_pin_stale.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

# --- Parse setup.ps1 (also serves as a syntax gate) and extract the helpers ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }

foreach ($name in @("Test-RocmGfx211Leaf", "Test-RocmKnown211Version", "Test-CudaFamilyLeaf", "Get-RocmPinStaleTags")) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in setup.ps1, found $($fn.Count)" }
    # Pure helpers (no exit / external calls) -- safe to define in this scope.
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# A pinned gfx/rocm index is stale when Expected != Installed.
function IsStale($leaf, $ver) {
    $t = Get-RocmPinStaleTags -PinLeaf $leaf -TorchVersion $ver
    return $t.Expected -ne $t.Installed
}

Write-Host "Test-RocmGfx211Leaf (the 2.11 gfx allowlist)"
Check "gfx1151 -> true"        (Test-RocmGfx211Leaf "gfx1151")
Check "gfx1150 -> true"        (Test-RocmGfx211Leaf "gfx1150")
Check "gfx120x-all -> true"    (Test-RocmGfx211Leaf "gfx120x-all")
Check "gfx110x-all -> false"   (-not (Test-RocmGfx211Leaf "gfx110x-all"))
Check "gfx90a -> false"        (-not (Test-RocmGfx211Leaf "gfx90a"))
Check "gfx908 -> false"        (-not (Test-RocmGfx211Leaf "gfx908"))

Write-Host "Test-CudaFamilyLeaf (^cu[0-9])"
Check "cu118 -> true"          (Test-CudaFamilyLeaf "cu118")
Check "cu128 -> true"          (Test-CudaFamilyLeaf "cu128")
Check "cu130 -> true"          (Test-CudaFamilyLeaf "cu130")
Check "custom -> false"        (-not (Test-CudaFamilyLeaf "custom"))
Check "current -> false"       (-not (Test-CudaFamilyLeaf "current"))
Check "cpu -> false"           (-not (Test-CudaFamilyLeaf "cpu"))
Check "empty -> false"         (-not (Test-CudaFamilyLeaf ""))

Write-Host "Get-RocmPinStaleTags (mirror of _rocm_pin_family_mismatch)"
# Exact rocm version comparison.
Check "rocm7.2 pin + 2.11.0+rocm7.2 -> not stale"  (-not (IsStale "rocm7.2" "2.11.0+rocm7.2"))
Check "rocm7.2 pin + 2.10.0+rocm6.4 -> stale"      (IsStale "rocm7.2" "2.10.0+rocm6.4")
Check "rocm6.4 pin + 2.10.0+rocm6.4 -> not stale"  (-not (IsStale "rocm6.4" "2.10.0+rocm6.4"))
# rocm7.2 is a KNOWN-2.11 index. A +rocm7.2 wheel whose RELEASE drifted off 2.11 shares
# the tag but violates the spec -> stale (mirror of _rocm_pin_family_mismatch).
Check "rocm7.2 pin + 2.12.0+rocm7.2 -> stale"      (IsStale "rocm7.2" "2.12.0+rocm7.2")
Check "rocm7.2 pin + 2.13.0+rocm7.2 -> stale"      (IsStale "rocm7.2" "2.13.0+rocm7.2")
Check "rocm7.2 pin + 2.11.5+rocm7.2 -> not stale"  (-not (IsStale "rocm7.2" "2.11.5+rocm7.2"))
# An UNKNOWN newer rocm (off the 2.11 allowlist) isn't floored, so a matching version at
# any release line is NOT stale on this exact-compare branch.
Check "rocm8.0 pin + 2.12.0+rocm8.0 -> not stale"  (-not (IsStale "rocm8.0" "2.12.0+rocm8.0"))
# An untagged (no +rocm) wheel never satisfies a ROCm pin -> always stale.
Check "rocm7.2 pin + 2.10.0 (untagged) -> stale"   (IsStale "rocm7.2" "2.10.0")
Check "rocm7.2 pin + 2.11.0 (untagged) -> stale"   (IsStale "rocm7.2" "2.11.0")
Check "rocm6.4 pin + 2.10.0 (untagged) -> stale"   (IsStale "rocm6.4" "2.10.0")
# 2.11-allowlist gfx pin: per-arch (three-part) wheel is satisfied, generic is stale.
Check "gfx1151 pin + 2.11.0+rocm7.13.0 -> not stale"   (-not (IsStale "gfx1151" "2.11.0+rocm7.13.0"))
Check "gfx1150 pin + 2.11.0+rocm7.13.0 -> not stale"   (-not (IsStale "gfx1150" "2.11.0+rocm7.13.0"))
Check "gfx120x-all pin + 2.11.0+rocm7.13.0 -> not stale" (-not (IsStale "gfx120x-all" "2.11.0+rocm7.13.0"))
Check "gfx1151 pin + 2.11.0+rocm7.2 (generic) -> stale" (IsStale "gfx1151" "2.11.0+rocm7.2")
Check "gfx1151 pin + 2.10.0+rocm6.4 -> stale"           (IsStale "gfx1151" "2.10.0+rocm6.4")
# Non-2.11 gfx pin (gfx110X-all/gfx90a/gfx908): a valid <2.11 wheel is NOT stale.
Check "gfx110x-all pin + 2.10.0+rocm6.4 -> not stale"  (-not (IsStale "gfx110x-all" "2.10.0+rocm6.4"))
Check "gfx90a pin + 2.10.0+rocm6.3 -> not stale"       (-not (IsStale "gfx90a" "2.10.0+rocm6.3"))
Check "gfx908 pin + 2.10.0+rocm7.0 -> not stale"       (-not (IsStale "gfx908" "2.10.0+rocm7.0"))
Check "gfx110x-all pin + 2.11.0+rocm7.2 -> stale"      (IsStale "gfx110x-all" "2.11.0+rocm7.2")
# Non-2.11 gfx pin over an untagged wheel: never satisfies the pin -> stale, so the
# explicit ROCm index is applied even when torch is already <2.11.
Check "gfx110x-all pin + 2.10.0 (untagged) -> stale"   (IsStale "gfx110x-all" "2.10.0")
Check "gfx90a pin + 2.10.0 (untagged) -> stale"        (IsStale "gfx90a" "2.10.0")
# Capital gfx120X-all is lowercased by Get-TorchIndexLeaf before this helper, so the
# 2.11-allowlist branch fires and a generic/untagged wheel is stale.
Check "gfx120x-all pin + 2.11.0+rocm7.2 (generic) -> stale" (IsStale "gfx120x-all" "2.11.0+rocm7.2")
Check "gfx120x-all pin + 2.10.0 (untagged) -> stale"        (IsStale "gfx120x-all" "2.10.0")

Write-Host "Test-RocmKnown211Version + KNOWN-2.11 fallback (rocm7.2 only; no speculative rocm7.3)"
Check "rocm7.2 -> known 2.11"  (Test-RocmKnown211Version -Major 7 -Minor 2)
Check "rocm7.1 -> not known"   (-not (Test-RocmKnown211Version -Major 7 -Minor 1))
Check "rocm7.3 -> not known"   (-not (Test-RocmKnown211Version -Major 7 -Minor 3))
Check "rocm8.0 -> not known"   (-not (Test-RocmKnown211Version -Major 8 -Minor 0))
# Unreadable-installed fallback: a rocm7.3 pin (unknown -> <2.11 line) over a <2.11 +rocm
# wheel with an unreadable version is NOT stale; rocm7.2 (KNOWN-2.11) over the same wheel
# IS stale (#2534 alignment).
Check "rocm7.3 pin + 2.10.0+rocm (unreadable ver) -> not stale" (-not (IsStale "rocm7.3" "2.10.0+rocm"))
Check "rocm7.2 pin + 2.10.0+rocm (unreadable ver) -> stale"     (IsStale "rocm7.2" "2.10.0+rocm")

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
