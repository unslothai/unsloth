#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Behavioural test for the $nameArchTable GPU-name -> gfx inference in install.ps1 and
# studio/setup.ps1, on the Radeon AI PRO R9700 (Navi 48, gfx1201; issues #7624 and #7307).
# The card's name carries neither "9070" nor "9080", so the first-match-wins table used to
# return nothing on a host with no HIP SDK and the installer fell back to CPU torch, which
# surfaces as "GPU not detected" on a single-R9700 Windows box.
# The Python parity suite evaluates these rows with Python's re; this runs them through
# PowerShell's own -match, which is the engine that actually ships.
# Run: pwsh -NoProfile -File tests/studio/test_amd_name_arch_r9700.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Returns the shipped `$nameArchTable = @( ... )` literal, sliced on balanced parens so the
# rows under test are the ones in the file rather than a copy pasted into this test.
function Get-NameArchTableSource($path) {
    # CRLF normalised: these installers ship with Windows line endings and every source-text
    # match below would silently miss otherwise.
    $src = (Get-Content -Raw $path) -replace "`r`n", "`n"
    $start = $src.IndexOf('$nameArchTable = @(')
    if ($start -lt 0) { throw "$path : `$nameArchTable = @( was not found (renamed or restructured?)" }
    $i = $src.IndexOf("(", $start)
    $depth = 0
    while ($i -lt $src.Length) {
        if ($src[$i] -eq "(") { $depth++ }
        elseif ($src[$i] -eq ")") { $depth--; if ($depth -eq 0) { return $src.Substring($start, $i - $start + 1) } }
        $i++
    }
    throw "$path : unterminated `$nameArchTable"
}

# First-match-wins, exactly as both installers consume the table.
function Resolve-Arch($table, $name) {
    foreach ($row in $table) { if ($name -match $row.P) { return $row.A } }
    return $null
}

# The card as its reporters saw it, plus the spellings WMI and amd-smi vary between.
$r9700Names = @(
    "AMD Radeon AI PRO R9700",
    "AMD Radeon(TM) AI PRO R9700",
    "Radeon AI PRO R9700",
    "AMD Radeon AI PRO R9700 32GB"
)

# Every other name the table already answers for, so the new alternation is shown not to
# steal a row from a neighbour, plus names that must keep resolving to nothing. The ATI
# Radeon 9700 PRO is the reason the pattern is "R9700" and not a bare "9700": that 2002
# card would match a loose token and be handed RDNA 4 wheels.
$otherNames = @(
    @{ N = "AMD Radeon RX 9070 XT";        A = "gfx1201" },
    @{ N = "AMD Radeon RX 9070 GRE";       A = "gfx1201" },
    @{ N = "AMD Radeon RX 9060 XT";        A = "gfx1200" },
    @{ N = "AMD Radeon 8060S Graphics";    A = "gfx1151" },
    @{ N = "AMD Radeon 890M Graphics";     A = "gfx1150" },
    @{ N = "AMD Radeon 860M Graphics";     A = "gfx1152" },
    @{ N = "AMD Radeon RX 7900 XTX";       A = "gfx1100" },
    @{ N = "AMD Radeon PRO W7900";         A = "gfx1100" },
    @{ N = "AMD Radeon RX 7800 XT";        A = "gfx1101" },
    @{ N = "AMD Radeon PRO V710";          A = "gfx1101" },
    @{ N = "AMD Radeon RX 7700S";          A = "gfx1102" },
    @{ N = "AMD Radeon PRO W7600";         A = "gfx1102" },
    @{ N = "AMD Radeon 780M Graphics";     A = "gfx1103" },
    @{ N = "AMD Radeon RX 6900 XT";        A = "gfx1030" },
    @{ N = "AMD Radeon RX 6600 XT";        A = "gfx1032" },
    @{ N = "AMD Radeon RX 6500 XT";        A = "gfx1034" },
    @{ N = "ATI Radeon 9700 PRO";          A = $null },
    @{ N = "ATI Radeon 9800 PRO";          A = $null },
    @{ N = "AMD Radeon R9 Fury X";         A = $null },
    @{ N = "AMD Radeon RX 5700 XT";        A = $null },
    @{ N = "AMD Radeon Pro WX 9100";       A = $null },
    @{ N = "AMD Instinct MI300X";          A = $null },
    @{ N = "NVIDIA GeForce RTX 4090";      A = $null },
    @{ N = "Microsoft Basic Display Adapter"; A = $null }
)

foreach ($file in @("install.ps1", "studio/setup.ps1")) {
    $path = Join-Path $root $file
    Write-Host ""
    Write-Host "=== $file ==="

    $tableSrc = Get-NameArchTableSource $path
    $nameArchTable = $null
    Invoke-Expression ("`$nameArchTable = " + $tableSrc.Substring($tableSrc.IndexOf("@(")))
    # Guard against a vacuous pass: an empty or truncated slice would resolve everything to
    # $null and every negative case below would "pass".
    Check "the shipped nameArchTable was found and parsed" ($nameArchTable -and $nameArchTable.Count -ge 12)
    Check "every parsed row has a pattern and an arch" (@($nameArchTable | Where-Object { -not $_.P -or -not $_.A }).Count -eq 0)

    foreach ($name in $r9700Names) {
        Check "'$name' -> gfx1201" ((Resolve-Arch $nameArchTable $name) -eq "gfx1201")
    }
    foreach ($row in $otherNames) {
        $got = Resolve-Arch $nameArchTable $row.N
        if ($null -eq $row.A) { Check "'$($row.N)' still matches nothing" ($null -eq $got) }
        else { Check "'$($row.N)' -> $($row.A)" ($got -eq $row.A) }
    }

    # The R9700 alternation must live on the gfx1201 arm specifically. Resolving correctly
    # would also be satisfied by an arm added anywhere, and a later arm would be shadowed
    # the day another row grows a pattern that matches first.
    $gfx1201Rows = @($nameArchTable | Where-Object { $_.A -eq "gfx1201" })
    Check "exactly one gfx1201 arm" ($gfx1201Rows.Count -eq 1)
    Check "the gfx1201 arm carries R9700" ($gfx1201Rows[0].P -match "R9700")
    Check "no other arm carries R9700" (@($nameArchTable | Where-Object { $_.A -ne "gfx1201" -and $_.P -match "R9700" }).Count -eq 0)
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
