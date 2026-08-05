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
    "Restore-StudioVenvDirectoryMerge",
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
    $script:StudioVenvRollbackIsPartialMove = $false
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
    $deadRollback = Join-Path $StudioHome "unsloth_studio.rollback.20260101000000.999999"
    $liveRollback = Join-Path $StudioHome "unsloth_studio.rollback.20260101000000.$PID"
    $otherRollback = Join-Path $StudioHome "unsloth_studio.rollback.custom"
    [System.IO.Directory]::CreateDirectory($deadRollback) | Out-Null
    [System.IO.Directory]::CreateDirectory($liveRollback) | Out-Null
    [System.IO.Directory]::CreateDirectory($otherRollback) | Out-Null

    Remove-StaleStudioVenvRollbacks
    Check "dead-owner rollback is removed" (-not (Test-Path -LiteralPath $deadRollback))
    Check "live-owner rollback is preserved" (Test-Path -LiteralPath $liveRollback)
    Check "unrecognized rollback name is preserved" (Test-Path -LiteralPath $otherRollback)
    Remove-Item -LiteralPath $liveRollback -Recurse -Force
    Remove-Item -LiteralPath $otherRollback -Recurse -Force

    Write-Host "Failure restoration"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old-again")
    Reset-RollbackState $VenvDir
    $committed = $false
    try {
        Start-StudioVenvRollback -ExistingDir $VenvDir
        try {
            [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
            [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "failed-install")
            [System.IO.File]::WriteAllText((Join-Path $VenvDir "failed_package.py"), "junk")
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
    Check "failed reinstall artifacts are wiped during full rollback" (-not (Test-Path -LiteralPath (Join-Path $VenvDir "failed_package.py")))
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

    Write-Host "Partial Move-Item failure in Start-StudioVenvRollback"
    $partialVenvDir = Join-Path $StudioHome "unsloth_studio_partial"
    [System.IO.Directory]::CreateDirectory($partialVenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $partialVenvDir "fileA"), "partA")
    [System.IO.File]::WriteAllText((Join-Path $partialVenvDir "fileB"), "partB")
    Reset-RollbackState $partialVenvDir

    # Intercept Move-Item to simulate a partial move failure:
    # moves fileA to candidate, leaves fileB in ExistingDir, then throws an exception
    function Move-Item {
        param(
            [string]$LiteralPath,
            [string]$Destination,
            [switch]$Force,
            [object]$ErrorAction
        )
        [System.IO.Directory]::CreateDirectory($Destination) | Out-Null
        Microsoft.PowerShell.Management\Move-Item -LiteralPath (Join-Path $LiteralPath "fileA") -Destination (Join-Path $Destination "fileA")
        throw "simulated partial move failure"
    }

    $failedAsExpected = $false
    try {
        Start-StudioVenvRollback -ExistingDir $partialVenvDir
    } catch {
        if ($_.Exception.Message -match "simulated partial move failure") {
            $failedAsExpected = $true
        }
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Move-Item -Force -ErrorAction SilentlyContinue
    }

    Check "Start-StudioVenvRollback threw expected error" $failedAsExpected
    Check "rollback tracking stays active after partial move failure" $script:StudioVenvRollbackActive
    Check "rollback dir is recorded after partial move failure" ($null -ne $script:StudioVenvRollbackDir)

    # Now verify that Restore-StudioVenvRollback can restore fileA back into $partialVenvDir
    $savedRollbackDir = $script:StudioVenvRollbackDir
    Restore-StudioVenvRollback
    Check "fileA was restored back to ExistingDir" (Test-Path -LiteralPath (Join-Path $partialVenvDir "fileA"))
    Check "fileB remains in ExistingDir" (Test-Path -LiteralPath (Join-Path $partialVenvDir "fileB"))
    Check "rollback directory was cleaned up after restoration" (-not (Test-Path -LiteralPath $savedRollbackDir))

    Write-Host "Nested directory partial Move-Item failure in Start-StudioVenvRollback"
    $nestedVenvDir = Join-Path $StudioHome "unsloth_studio_nested"
    $nestedScriptsDir = Join-Path $nestedVenvDir "Scripts"
    [System.IO.Directory]::CreateDirectory($nestedScriptsDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $nestedScriptsDir "unsloth.exe"), "unsloth_bin")
    [System.IO.File]::WriteAllText((Join-Path $nestedScriptsDir "python.exe"), "python_bin")
    Reset-RollbackState $nestedVenvDir

    # Intercept Move-Item to simulate moving Scripts/python.exe to candidate while Scripts/unsloth.exe stays in ExistingDir
    function Move-Item {
        param(
            [string]$LiteralPath,
            [string]$Destination,
            [switch]$Force,
            [object]$ErrorAction
        )
        $destScripts = Join-Path $Destination "Scripts"
        [System.IO.Directory]::CreateDirectory($destScripts) | Out-Null
        Microsoft.PowerShell.Management\Move-Item -LiteralPath (Join-Path $LiteralPath "Scripts\python.exe") -Destination (Join-Path $destScripts "python.exe")
        throw "simulated nested partial move failure"
    }

    try {
        Start-StudioVenvRollback -ExistingDir $nestedVenvDir
    } catch {
        # expected
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Move-Item -Force -ErrorAction SilentlyContinue
    }

    $savedNestedRollback = $script:StudioVenvRollbackDir
    Restore-StudioVenvRollback
    Check "unsloth.exe was preserved in Scripts directory during rollback restore" (Test-Path -LiteralPath (Join-Path $nestedScriptsDir "unsloth.exe"))
    Check "python.exe was restored into Scripts directory" (Test-Path -LiteralPath (Join-Path $nestedScriptsDir "python.exe"))
    Check "nested rollback directory was cleaned up after restoration" (-not (Test-Path -LiteralPath $savedNestedRollback))
} finally {
    if (Test-Path -LiteralPath $StudioHome) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $StudioHome -Recurse -Force
    }
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
