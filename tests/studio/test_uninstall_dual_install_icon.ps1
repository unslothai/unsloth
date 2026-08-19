#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for _RemoveDataDirKeepingWslIcon in scripts/uninstall.ps1.
#
# A native uninstall must NOT delete the shared unsloth.ico while a WSL shortcut
# still points at it (else that shortcut blanks). Extracts the helper via AST and
# runs it on a temp data dir with a controlled ShortcutDirs list (dual-install vs
# native-only), so no real Desktop / Start Menu is touched.
#
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_dual_install_icon.ps1

$ErrorActionPreference = "Stop"
$uninstallPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "scripts", "uninstall.ps1")
$uninstallPath = (Resolve-Path $uninstallPath).Path

# --- Extract the nested helper function source (not the whole uninstaller) ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($uninstallPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "uninstall.ps1 has parse errors" }
$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "_RemoveDataDirKeepingWslIcon"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one _RemoveDataDirKeepingWslIcon, found $($fn.Count)" }

# Stubs the helper depends on (nested in Uninstall-UnslothStudio in production).
function _Substep { param([string]$Msg, [string]$Color = "Gray") }
function _RemovePath {
    param([string]$Path)
    if ($Path -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}
. ([ScriptBlock]::Create($fn[0].Extent.Text))   # defines _RemoveDataDirKeepingWslIcon

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("undi_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
    # ----- Case A: a WSL shortcut survives -> keep unsloth.ico, drop the rest, keep dir -----
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

    # ----- Case B: no WSL shortcut -> whole dir removed (native-only uninstall) -----
    $progB = Join-Path $work "shortcutsB"
    New-Item -ItemType Directory -Force -Path $progB | Out-Null
    Set-Content -LiteralPath (Join-Path $progB "Unrelated.lnk") -Value "x"
    $dataB = Join-Path $work "dataB"
    New-Item -ItemType Directory -Force -Path $dataB | Out-Null
    Set-Content -LiteralPath (Join-Path $dataB "unsloth.ico") -Value "ICO"
    Set-Content -LiteralPath (Join-Path $dataB "launch-studio.ps1") -Value "L"
    _RemoveDataDirKeepingWslIcon -DataDir $dataB -ShortcutDirs @($progB)
    Check "B: whole data dir removed"   (-not (Test-Path -LiteralPath $dataB))

    # ----- Case C: empty shortcut dirs -> whole dir removed -----
    $dataC = Join-Path $work "dataC"
    New-Item -ItemType Directory -Force -Path $dataC | Out-Null
    Set-Content -LiteralPath (Join-Path $dataC "unsloth.ico") -Value "ICO"
    _RemoveDataDirKeepingWslIcon -DataDir $dataC -ShortcutDirs @()
    Check "C: removed when no shortcut dirs" (-not (Test-Path -LiteralPath $dataC))

    # ----- Case D: missing data dir -> no-op, no throw -----
    $dataD = Join-Path $work "doesNotExist"
    _RemoveDataDirKeepingWslIcon -DataDir $dataD -ShortcutDirs @($progA)
    Check "D: missing dir is a safe no-op" (-not (Test-Path -LiteralPath $dataD))
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host ""; Write-Host "FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host ""; Write-Host "All tests passed."; exit 0
