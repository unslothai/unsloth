#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for studio/setup.ps1's pinned-torch-index stale-venv helpers
# (Test-RocmGfx211Leaf, Test-CudaFamilyLeaf, Get-RocmPinStaleTags). Pure helpers,
# AST-extracted and run in-process -- no GPU/venv needed. Mirrors the Python
# _rocm_pin_family_mismatch / _is_cuda_family_leaf tests so both stay in lockstep.
# Run: pwsh -NoProfile -File tests/studio/test_setup_pin_stale.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

# --- Parse setup.ps1 (also serves as a syntax gate) and extract the helpers ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }

foreach ($name in @("Test-RocmGfx211Leaf", "Test-CudaFamilyLeaf", "Get-RocmPinStaleTags")) {
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
# rocm pin vs unreadable installed rocm version -> compare on the 2.11 line.
Check "rocm7.2 pin + 2.10.0 -> stale"              (IsStale "rocm7.2" "2.10.0")
Check "rocm7.2 pin + 2.11.0 -> not stale"          (-not (IsStale "rocm7.2" "2.11.0"))
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

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
