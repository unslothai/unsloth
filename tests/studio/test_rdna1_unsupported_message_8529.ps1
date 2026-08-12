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

    # --- Polaris, the second card in the cluster (#8458) ----------------------
    # #8458 is an RX 580: Polaris 20, gfx803, also pre-RDNA 2 and also with no
    # ROCm PyTorch wheels. It reaches this table for the message only.
    Check "RX 580 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 580") -eq 'gfx803')
    Check "RX 580 Series -> gfx803"        ((Resolve-Unsupported "AMD Radeon RX 580 Series") -eq 'gfx803')
    Check "RX 570 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 570") -eq 'gfx803')
    Check "RX 590 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 590") -eq 'gfx803')
    Check "RX 480 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 480") -eq 'gfx803')
    Check "RX 470 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 470") -eq 'gfx803')

    # Polaris 11/12 is a different die and is deliberately absent: this table is
    # only worth having while it never guesses an arch.
    Check "RX 560 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 560"))
    Check "RX 550 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 550"))
    Check "RX 460 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 460"))

    # The collision this row is one keystroke away from: "RX 570" is a prefix of
    # "RX 5700" and "RX 550" of "RX 5500". Matched ALONE, because table order
    # already stops the RDNA 1 rows from ever reaching the Polaris pattern, so a
    # dropped (?!0) guard changes nothing observable until someone reorders.
    $polarisPattern = @($unsupportedNameArchTable | Where-Object { $_.A -eq 'gfx803' })[0].P
    foreach ($rdna1 in @("AMD Radeon RX 5700 XT", "AMD Radeon RX 5700", "AMD Radeon RX 5500 XT")) {
        Check "Polaris pattern alone does not claim '$rdna1'" (-not ($rdna1 -match $polarisPattern))
    }

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

    # --- the Vulkan pointer (#8458) -------------------------------------------
    # Torch is the end of the road on these cards; llama.cpp is not. Asserted
    # against the lines that PRINT, never the whole file: every phrase below also
    # appears in the comments that explain the branch, so a file-wide search stays
    # green after the message itself has been gutted.
    # Single quotes count too: the Vulkan setter is emitted as a literal so
    # PowerShell prints $env:... instead of expanding it, and a double-quote-only
    # filter would drop exactly the line under test and pass on an empty set.
    $emitted = @(($src -split "`n") | Where-Object {
        $_ -match 'substep\s+["'']' -and $_.TrimStart() -notmatch '^#'
    })
    Check "the emitted advice offers Vulkan" `
        (($emitted -join "`n").Contains("through Vulkan"))
    # In PowerShell syntax, and verified by PARSING it rather than by matching text:
    # a bare UNSLOTH_LLAMA_CPP_BACKEND=vulkan parses as a command name, so a user who
    # pastes it sets nothing, re-runs the installer and gets the same CPU bundle --
    # the #8458 failure mode reintroduced by the fix for it.
    $setter = '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan"'
    Check "the emitted advice teaches the current spelling" `
        (($emitted -join "`n").Contains($setter))
    $posix = @($emitted | Where-Object { $_ -match 'UNSLOTH_LLAMA_CPP_BACKEND=vulkan' })
    Check "no emitted line gives a POSIX assignment" ($posix.Count -eq 0)
    $setterAst = [System.Management.Automation.Language.Parser]::ParseInput(
        $setter, [ref]$null, [ref]$null)
    Check "the taught setter parses as an assignment, not a command" `
        (@($setterAst.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true)).Count -eq 1)

    # UNSLOTH_FORCE_VULKAN still works, but force_vulkan_requested() resolves
    # UNSLOTH_LLAMA_CPP_BACKEND first and only falls back to the legacy name when
    # the new one is absent or unparseable, so =hip stays a real opt-out. New text
    # must not spread the legacy spelling. Scoped to emitters: setup.ps1 legitimately
    # READS the legacy variable for back-compat and that is untouched here.
    $teachesLegacy = @($emitted | Where-Object { $_ -match 'UNSLOTH_FORCE_VULKAN' })
    Check "the legacy spelling is not taught" ($teachesLegacy.Count -eq 0)

    # WHEN to set it, per SITE. install.ps1 prints this advice at two places, so a
    # file-level "install time" search is satisfied by whichever site still has it
    # and gutting the other one passes.
    $mentions = @(0..($emitted.Count - 1) | Where-Object {
        $emitted[$_].Contains($setter)
    })
    Check "at least one site names the Vulkan variable" ($mentions.Count -ge 1)
    foreach ($i in $mentions) {
        $window = ($emitted[$i..([Math]::Min($i + 3, $emitted.Count - 1))]) -join "`n"
        Check "the advice at emitted line $($i + 1) says it applies at install time" `
            ($window.Contains("install time"))
    }

    # --- which arm actually WINS, evaluated rather than read ------------------
    # The ordering checks above compare source offsets, which cannot see a branch
    # that is unreachable because an earlier condition already matched. This runs
    # the real if/elseif chain: parse it, then evaluate every clause condition in
    # order against the #8529 host and assert the FIRST true one is the unsupported
    # arm. The host is the one that reported the bug and then followed the old
    # advice: an RX 5700 XT, no ROCm runtime, and the HIP SDK now installed.
    $chainAst = [System.Management.Automation.Language.Parser]::ParseFile(
        $path, [ref]$null, [ref]$null)
    $chains = @($chainAst.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.IfStatementAst] -and
        ($n.Clauses | Where-Object { $_.Item1.Extent.Text -match 'ROCmUnsupportedGfxArch' }) -and
        ($n.Clauses | Where-Object { $_.Item2.Extent.Text -match 'step "gpu"' })
    }, $true))
    Check "the gpu report chain was found" ($chains.Count -eq 1)
    if ($chains.Count -eq 1) {
        $HasNvidiaSmi = $false
        $script:IsIntelXpu = $false
        $HasROCm = $false
        $HipSdkInstalled = $true            # they installed it because we told them to
        $ROCmGpuLabel = "AMD Radeon RX 5700 XT"
        $ROCmGfxArch = $null
        $script:ROCmGfxArch = $null
        $ROCmUnsupportedGfxArch = "gfx1010"
        $script:ROCmUnsupportedGfxArch = "gfx1010"
        $winner = -1
        $unsupIdx = -1
        for ($c = 0; $c -lt $chains[0].Clauses.Count; $c++) {
            $cond = $chains[0].Clauses[$c].Item1.Extent.Text
            if ($cond -match 'ROCmUnsupportedGfxArch' -and $cond -notmatch '-not') { $unsupIdx = $c }
            if ($winner -lt 0 -and [bool](& ([scriptblock]::Create($cond)))) { $winner = $c }
        }
        Check "the unsupported arm exists in the chain" ($unsupIdx -ge 0)
        Check "an RDNA 1 host with the HIP SDK installed reaches the unsupported arm" `
            ($winner -eq $unsupIdx)
    }

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
