#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.ps1's venv rollback helpers. The functions are AST-extracted
# so the top-level installer is never executed.

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$helperNames = @(
    "Start-StudioVenvRollback",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Restore-StudioVenvRollback",
    "Complete-StudioVenvRollback"
)
foreach ($name in $helperNames) {
    $fn = $ast.FindAll({ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in install.ps1, found $($fn.Count)" }
    Invoke-Expression $fn[0].Extent.Text
}

function substep { param([string]$Message, [string]$Color) }

$failures = 0
function Check($name, $condition) {
    if ($condition) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Reset-RollbackState($target) {
    $script:StudioVenvRollbackDir = $null
    $script:StudioVenvRollbackTarget = $target
    $script:StudioVenvRollbackActive = $false
}

$StudioHome = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth-rollback-$([guid]::NewGuid().ToString('N'))"
$VenvDir = Join-Path $StudioHome "unsloth_studio"
[System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null

try {
    Write-Host "Successful replacement"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    Start-StudioVenvRollback -ExistingDir $VenvDir
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "new")
    Complete-StudioVenvRollback
    Check "new environment remains" ((Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "new")
    Check "current rollback is removed" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Stale cleanup"
    $stale = Join-Path $StudioHome "unsloth_studio.rollback.20000101000000.2147483647"
    $active = Join-Path $StudioHome "unsloth_studio.rollback.20000101000001.$PID"
    $unrecognized = Join-Path $StudioHome "unsloth_studio.rollback.user-data"
    [System.IO.Directory]::CreateDirectory($stale) | Out-Null
    [System.IO.Directory]::CreateDirectory($active) | Out-Null
    [System.IO.Directory]::CreateDirectory($unrecognized) | Out-Null
    Remove-StaleStudioVenvRollbacks
    Check "dead-owner rollback is removed" (-not (Test-Path -LiteralPath $stale))
    Check "live-owner rollback is preserved" (Test-Path -LiteralPath $active)
    Check "unrecognized rollback name is preserved" (Test-Path -LiteralPath $unrecognized)
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $active -Recurse -Force
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $unrecognized -Recurse -Force

    Write-Host "Failure restoration"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old-again")
    Reset-RollbackState $VenvDir
    $committed = $false
    try {
        try {
            Start-StudioVenvRollback -ExistingDir $VenvDir
            [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
            [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "partial")
            throw "simulated install failure"
        } finally {
            if (-not $committed) { Restore-StudioVenvRollback }
        }
    } catch {
        if ($_.Exception.Message -ne "simulated install failure") { throw }
    }
    Check "finally restores the previous environment" (
        (Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "old-again"
    )
    Check "failure restoration consumes the rollback" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Locked-file retry"
    $retryDir = Join-Path $StudioHome "retry"
    [System.IO.Directory]::CreateDirectory($retryDir) | Out-Null
    $script:removeAttempts = 0
    function Remove-Item {
        param(
            [string]$LiteralPath,
            [switch]$Recurse,
            [switch]$Force,
            [object]$ErrorAction
        )
        $script:removeAttempts++
        if ($script:removeAttempts -lt 3) { throw "simulated lock" }
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
    }
    try {
        $removed = Remove-StudioVenvTreeWithRetry -Path $retryDir -Label "test rollback"
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Remove-Item -Force
    }
    Check "locked rollback deletion retries" ($removed -and $script:removeAttempts -eq 3)
} finally {
    if (Test-Path -LiteralPath $StudioHome) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $StudioHome -Recurse -Force
    }
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
