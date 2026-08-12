#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Behavioural test for the RDNA 1 "detected but not covered by ROCm" wording in
# install.ps1 and studio/setup.ps1 (issue #8529).
#
# The reporter's card is an RX 5700 XT: Navi 10, gfx1010, RDNA 1. AMD publishes
# Windows torch indexes for gfx103X, gfx110X, gfx1150, gfx1151 and gfx120X only,
# so CPU torch is the correct outcome and stays. What was wrong is that the
# installer told them to install the HIP SDK or set UNSLOTH_ROCM_GFX_ARCH, and
# neither can succeed on gfx1010.
#
# The Python suite evaluates the table with Python's `re`; this runs it under the
# .NET regex engine that ships it, which is the only place -match semantics are
# real. Fixtures are the raw WMI adapter names, since that is what the table is
# matched against.
#
# Run: pwsh -NoProfile -File tests/studio/test_rdna1_unsupported_message_8529.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Returns the source text of a variable assignment. The caller Invoke-Expression's
# it at script scope; doing that inside a function would lose the value on return.
function Get-AssignmentSource($path, $varName) {
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $hits = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $n.Left.Extent.Text -eq $varName
    }, $true)
    if ($hits.Count -lt 1) { throw "expected $varName in $path, found none" }
    return $hits[0].Extent.Text
}

foreach ($file in @("install.ps1", "studio/setup.ps1")) {
    $path = Join-Path $root $file
    Write-Host ""
    Write-Host "=== $file ==="

    Invoke-Expression (Get-AssignmentSource $path '$unsupportedNameArchTable')

    # The same first-match-wins loop both installers run over the table.
    function Resolve-Unsupported($name) {
        foreach ($row in $unsupportedNameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }

    # --- the reporter's card, and the rest of RDNA 1 ---------------------------
    Check "RX 5700 XT -> gfx1010"          ((Resolve-Unsupported "AMD Radeon RX 5700 XT") -eq 'gfx1010')
    Check "RX 5700 -> gfx1010"             ((Resolve-Unsupported "AMD Radeon RX 5700") -eq 'gfx1010')
    Check "RX 5600 XT -> gfx1010"          ((Resolve-Unsupported "AMD Radeon RX 5600 XT") -eq 'gfx1010')
    Check "Radeon Pro 5600 XT -> gfx1010"  ((Resolve-Unsupported "AMD Radeon Pro 5600 XT") -eq 'gfx1010')
    Check "Radeon Pro V520 -> gfx1011"     ((Resolve-Unsupported "AMD Radeon Pro V520") -eq 'gfx1011')
    Check "Radeon Pro 5600M -> gfx1011"    ((Resolve-Unsupported "AMD Radeon Pro 5600M") -eq 'gfx1011')
    Check "RX 5500 XT -> gfx1012"          ((Resolve-Unsupported "AMD Radeon RX 5500 XT") -eq 'gfx1012')

    # --- and nothing else -----------------------------------------------------
    # A hit here would print "ROCm does not cover this" at a card that has wheels.
    Check "RX 9070 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9070 XT"))
    Check "RX 9060 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9060 XT"))
    Check "RX 7900 XTX unclaimed"          ($null -eq (Resolve-Unsupported "AMD Radeon RX 7900 XTX"))
    Check "RX 6800 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 6800 XT"))
    Check "8060S Graphics unclaimed"       ($null -eq (Resolve-Unsupported "AMD Radeon 8060S Graphics"))
    Check "RTX 4090 unclaimed"             ($null -eq (Resolve-Unsupported "NVIDIA GeForce RTX 4090"))

    # --- the supported table must still miss RDNA 1 ---------------------------
    # The behavioural half: CPU fallback is correct here and must not move.
    Invoke-Expression (Get-AssignmentSource $path '$nameArchTable')
    function Resolve-Supported($name) {
        foreach ($row in $nameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }
    Check "RX 5700 XT gets no supported arch" ($null -eq (Resolve-Supported "AMD Radeon RX 5700 XT"))
    Check "RX 5500 XT gets no supported arch" ($null -eq (Resolve-Supported "AMD Radeon RX 5500 XT"))
    Check "RX 9070 XT still maps to gfx1201"  ((Resolve-Supported "AMD Radeon RX 9070 XT") -eq 'gfx1201')

    # No arch may appear in both tables: one routes to a wheel index, the other
    # exists precisely because nothing routes.
    $bothTables = @($unsupportedNameArchTable | ForEach-Object { $_.A }) |
        Where-Object { @($nameArchTable | ForEach-Object { $_.A }) -contains $_ }
    Check "the two tables share no arch"   ($bothTables.Count -eq 0)

    # --- the wording, in the source that prints it ----------------------------
    # CRLF normalisation is mandatory: both files ship CRLF, so any needle
    # spanning a line break never matches the raw text.
    $src = (Get-Content -Raw $path) -replace "`r`n", "`n"

    # Every ordering claim below is preceded by a "was found" guard, so a renamed
    # branch fails loudly instead of comparing two -1s and passing vacuously.
    $unsupArm = 'step "gpu" "AMD GPU detected ($'
    $unknownArm = 'step "gpu" "AMD GPU detected -- arch unknown"'
    Check "unsupported gpu arm is present"  ($src.Contains($unsupArm))
    Check "arch-unknown gpu arm is present" ($src.Contains($unknownArm))
    Check "unsupported arm precedes arch-unknown arm" `
        ($src.IndexOf($unsupArm) -ge 0 -and $src.IndexOf($unknownArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($unknownArm))

    $disclaimer = "setting UNSLOTH_ROCM_GFX_ARCH will not change that."
    Check "unsupported arm says the override cannot help" ($src.Contains($disclaimer))

    $sdkAdvice = 'substep "Could not determine the GPU arch'
    Check "HIP SDK advice still exists for genuinely unknown cards" ($src.Contains($sdkAdvice))
    Check "HIP SDK advice comes after the unsupported arm" `
        ($src.IndexOf($sdkAdvice) -ge 0 -and $src.IndexOf($unsupArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($sdkAdvice))

    # The scope guard, in source: the unsupported lookup must never assign the
    # arch the installers route on.
    $tableSrc = Get-AssignmentSource $path '$unsupportedNameArchTable'
    Check "the table assigns no routable arch" (-not ($tableSrc -match 'gfx1(0[3-9]|1|2)'))

    # The table has to be READ, not just declared. Everything above evaluates it in
    # isolation, so a lookup that was never wired to the message arms would sail
    # through: assert the consumer the installer actually uses.
    $consumer = if ($file -eq "install.ps1") {
        'foreach ($row in $unsupportedNameArchTable) {'
    } else {
        '-Table $unsupportedNameArchTable'
    }
    Check "the table is consumed by the arch resolver" ($src.Contains($consumer))
    Check "the resolver feeds the variable the arms read" ($src -match 'ROCmUnsupportedGfxArch\s*=\s*(\$row\.A|Get-GfxArchFromGpuName)')
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
