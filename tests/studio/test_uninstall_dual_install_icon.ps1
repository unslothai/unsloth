#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A native uninstall must keep the shared unsloth.ico while a WSL shortcut still points at it.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_dual_install_icon.ps1

$ErrorActionPreference = "Stop"
$uninstallPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "scripts", "uninstall.ps1")
$uninstallPath = (Resolve-Path $uninstallPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($uninstallPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "uninstall.ps1 has parse errors" }
# The helper delegates deletion to _RemoveTreeKeeping, so both must be extracted.
$wanted = @("_RemoveDataDirKeepingWslIcon", "_RemoveTreeKeeping")
$fn = @()
foreach ($name in $wanted) {
    $found = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }.GetNewClosure(), $true)
    if ($found.Count -ne 1) { throw "expected exactly one $name, found $($found.Count)" }
    $fn += $found[0]
}

function _Substep { param([string]$Msg, [string]$Color = "Gray") }
function _RemovePath {
    param([string]$Path)
    if ($Path -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}
foreach ($f in $fn) { . ([ScriptBlock]::Create($f.Extent.Text)) }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("undi_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
    $progA = Join-Path $work "shortcutsA"
    New-Item -ItemType Directory -Force -Path $progA | Out-Null
    Set-Content -LiteralPath (Join-Path $progA "Unsloth Studio (WSL - Ubuntu-24.04).lnk") -Value "x"
    $dataA = Join-Path $work "dataA"
    New-Item -ItemType Directory -Force -Path $dataA | Out-Null
    Set-Content -LiteralPath (Join-Path $dataA "unsloth.ico") -Value "ICO"
    Set-Content -LiteralPath (Join-Path $dataA "launch-studio.ps1") -Value "L"
    Set-Content -LiteralPath (Join-Path $dataA "studio.conf") -Value "C"
    _RemoveDataDirKeepingWslIcon -DataDir $dataA -ShortcutDirs @($progA)
    Check "A: data dir kept"            (Test-Path -LiteralPath $dataA)
    Check "A: unsloth.ico preserved"    (Test-Path -LiteralPath (Join-Path $dataA "unsloth.ico"))
    Check "A: launch-studio.ps1 removed" (-not (Test-Path -LiteralPath (Join-Path $dataA "launch-studio.ps1")))
    Check "A: studio.conf removed"      (-not (Test-Path -LiteralPath (Join-Path $dataA "studio.conf")))

    $progB = Join-Path $work "shortcutsB"
    New-Item -ItemType Directory -Force -Path $progB | Out-Null
    Set-Content -LiteralPath (Join-Path $progB "Unrelated.lnk") -Value "x"
    $dataB = Join-Path $work "dataB"
    New-Item -ItemType Directory -Force -Path $dataB | Out-Null
    Set-Content -LiteralPath (Join-Path $dataB "unsloth.ico") -Value "ICO"
    Set-Content -LiteralPath (Join-Path $dataB "launch-studio.ps1") -Value "L"
    _RemoveDataDirKeepingWslIcon -DataDir $dataB -ShortcutDirs @($progB)
    Check "B: whole data dir removed"   (-not (Test-Path -LiteralPath $dataB))

    $dataC = Join-Path $work "dataC"
    New-Item -ItemType Directory -Force -Path $dataC | Out-Null
    Set-Content -LiteralPath (Join-Path $dataC "unsloth.ico") -Value "ICO"
    _RemoveDataDirKeepingWslIcon -DataDir $dataC -ShortcutDirs @()
    Check "C: removed when no shortcut dirs" (-not (Test-Path -LiteralPath $dataC))

    $dataD = Join-Path $work "doesNotExist"
    _RemoveDataDirKeepingWslIcon -DataDir $dataD -ShortcutDirs @($progA)
    Check "D: missing dir is a safe no-op" (-not (Test-Path -LiteralPath $dataD))
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host ""; Write-Host "FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host ""; Write-Host "All tests passed."; exit 0
