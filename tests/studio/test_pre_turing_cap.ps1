#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Behavioural test for the pre-Turing cu126 cap (Get-NvidiaCu126Verdict,
# Get-CudaFamilyCappedForPreTuring) in install.ps1 and studio/setup.ps1.
# test_cross_platform_parity.py only greps for the call spelling, so a selector that
# computes the verdict and discards it still passes there. This runs both copies.
# Run: pwsh -NoProfile -File tests/studio/test_pre_turing_cap.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Returns the source text of each named function. The caller Invoke-Expression's it at
# script scope; doing that inside a function would lose the helpers on return.
function Get-HelperSources($path, $names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $out = @()
    foreach ($name in $names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -lt 1) { throw "expected $name in $path, found none" }
        $out += $fn[0].Extent.Text
    }
    return $out
}

# Stubs for the installers' printers, so this file does not depend on the ANSI helpers.
# Both use Write-Host, so neither can pollute a function's return value.
function substep { param([string]$Message, [string]$Color = "DarkGray") }
function Write-StudioStdoutMirror { param([string]$Line) }

# Drives Get-NvidiaCu126Verdict without spawning a process: $script:FakeSmiStdout is what
# a -StdoutOnly probe returns, $script:FakeSmiRc its exit code.
function Invoke-NvidiaSmiBounded {
    param([string]$Exe, [string[]]$SmiArgs = @(), [int]$TimeoutSec = 10, [switch]$StdoutOnly)
    $global:LASTEXITCODE = $script:FakeSmiRc
    # The real helper appends stderr to stdout without -StdoutOnly. Reproducing that is
    # the point: the caller MUST pass the switch.
    if ($StdoutOnly) { return $script:FakeSmiStdout }
    return ($script:FakeSmiStdout + "`n" + $script:FakeSmiStderr)
}

foreach ($file in @("install.ps1", "studio/setup.ps1")) {
    $path = Join-Path $root $file
    Write-Host ""
    Write-Host "=== $file ==="
    foreach ($srcText in (Get-HelperSources $path @("Get-NvidiaCu126Verdict",
                                                   "Get-CudaFamilyCappedForPreTuring"))) {
        Invoke-Expression $srcText
    }

    # --- the verdict table -----------------------------------------------------
    $script:FakeSmiRc = 0
    $script:FakeSmiStderr = ""
    function Verdict($rows, $floor = 75) {
        $script:FakeSmiStdout = ($rows -join "`n")
        return (Get-NvidiaCu126Verdict "nvidia-smi" $floor)
    }
    Check "V100 sm_70 under floor 75 -> cu126"        ((Verdict @("7.0")) -eq 'cu126')
    Check "V100 sm_70 under floor 70 -> no cap"       ((Verdict @("7.0") 70) -eq '')
    Check "GTX980 sm_52 -> cu126"                     ((Verdict @("5.2")) -eq 'cu126')
    Check "GTX1080 sm_61 -> cu126"                    ((Verdict @("6.1")) -eq 'cu126')
    Check "T4 sm_75 -> no cap"                        ((Verdict @("7.5")) -eq '')
    Check "H100 sm_90 -> no cap"                      ((Verdict @("9.0")) -eq '')
    Check "B200 sm_100 -> no cap"                     ((Verdict @("10.0")) -eq '')
    Check "Volta + Ampere -> cu126"                   ((Verdict @("7.0", "8.6")) -eq 'cu126')
    Check "Volta + Blackwell -> uncovered"            ((Verdict @("7.0", "12.0")) -eq 'uncovered')
    Check "Kepler sm_37 -> uncovered"                 ((Verdict @("3.7")) -eq 'uncovered')
    Check "CRLF rows still parse"                     ((Verdict @("7.0`r", "8.6`r")) -eq 'cu126')
    Check "padded rows still parse"                   ((Verdict @("  7.0  ")) -eq 'cu126')
    Check "blank rows are skipped"                    ((Verdict @("7.0", "", "8.6")) -eq 'cu126')
    Check "'N/A' row poisons the inventory"           ((Verdict @("7.0", "N/A")) -eq '')
    Check "'[N/A]' row poisons the inventory"         ((Verdict @("7.0", "[N/A]")) -eq '')
    Check "'[Not Supported]' poisons the inventory"   ((Verdict @("7.0", "[Not Supported]")) -eq '')
    Check "decimal comma poisons the inventory"       ((Verdict @("7,0")) -eq '')
    Check "empty inventory -> no cap"                 ((Verdict @("")) -eq '')
    Check "no exe -> no cap"                          ((Get-NvidiaCu126Verdict "" 75) -eq '')
    $script:FakeSmiRc = 1
    Check "non-zero exit -> no cap"                   ((Verdict @("7.0")) -eq '')
    $script:FakeSmiRc = 0

    # --- -StdoutOnly is load-bearing ------------------------------------------
    # A driver warning on stderr is ordinary (corrupted infoROM, ECC pending). Without the
    # switch it lands in the CSV, the inventory reads as unparseable, and a V100 silently
    # keeps cu130 -- issue #7765 all over again.
    $script:FakeSmiStdout = "7.0"
    $script:FakeSmiStderr = "WARNING: infoROM is corrupted at gpu 0000:00:04.0"
    Check "stderr noise does not reach the CSV parse" ((Get-NvidiaCu126Verdict "nvidia-smi" 75) -eq 'cu126')
    $src = Get-Content -Raw $path
    Check "the probe call passes -StdoutOnly"         ($src -match 'compute_cap.*-StdoutOnly')
    $script:FakeSmiStderr = ""

    # --- the cap only rewrites the families it can replace ---------------------
    $script:FakeSmiStdout = "7.0"
    Check "cap cu130 on a V100 -> cu126"              ((Get-CudaFamilyCappedForPreTuring 'cu130' "nvidia-smi") -eq 'cu126')
    Check "cap cu128 on a V100 -> cu126 (floor 75)"   ((Get-CudaFamilyCappedForPreTuring 'cu128' "nvidia-smi") -eq 'cu126')
    Check "cap cu126 is a no-op"                      ((Get-CudaFamilyCappedForPreTuring 'cu126' "nvidia-smi") -eq 'cu126')
    Check "cap cu124 is a no-op"                      ((Get-CudaFamilyCappedForPreTuring 'cu124' "nvidia-smi") -eq 'cu124')
    Check "cap cpu is a no-op"                        ((Get-CudaFamilyCappedForPreTuring 'cpu' "nvidia-smi") -eq 'cpu')
    $r = Get-CudaFamilyCappedForPreTuring 'cu130' "nvidia-smi"
    Check "returns a single string, not an array"     (-not ($r -is [array]))
    $script:FakeSmiStdout = "7.0`n12.0"
    Check "uncovered mix keeps the driver family"     ((Get-CudaFamilyCappedForPreTuring 'cu130' "nvidia-smi") -eq 'cu130')
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
