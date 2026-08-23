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

# The helpers under test. Anything they call is pulled in below, so a helper that
# gains a dependency does not have to be added here by hand.
$helperNames = @(
    "Start-StudioVenvRollback",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Restore-StudioVenvRollback",
    "Complete-StudioVenvRollback"
)

# install.ps1 nests these inside Install-UnslothStudio, so at runtime PowerShell's
# dynamic scoping hands each one its siblings. An extracted function has no such
# frame, so extract its callees too or the first call dies with "term not
# recognized" (#9501 added Test-StudioPathPresent and hit exactly that).
$defs = @{}
foreach ($fn in $ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)) {
    if (-not $defs.ContainsKey($fn.Name)) { $defs[$fn.Name] = @() }
    $defs[$fn.Name] += $fn
}

# Output sinks this file stubs below. Extracting install.ps1's real ones would leave
# which definition wins depending on the order of this file.
$stubbedHere = @("substep", "Write-StudioLine")

$needed = [System.Collections.Generic.List[string]]::new()
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($name in $helperNames) { $queue.Enqueue($name) }
while ($queue.Count -gt 0) {
    $name = $queue.Dequeue()
    if ($needed.Contains($name)) { continue }
    if (-not $defs.ContainsKey($name)) { throw "install.ps1 does not define $name" }
    if ($defs[$name].Count -ne 1) {
        throw "expected exactly one $name in install.ps1, found $($defs[$name].Count)"
    }
    $needed.Add($name)
    foreach ($call in $defs[$name][0].FindAll({ param($node)
        $node -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $called = $call.GetCommandName()
        if ($called -and $defs.ContainsKey($called) -and $stubbedHere -notcontains $called) {
            $queue.Enqueue($called)
        }
    }
}
foreach ($name in $needed) { Invoke-Expression $defs[$name][0].Extent.Text }

function substep { param([string]$Message, [string]$Color) }
# The rollback helpers report through install.ps1's UTF-8 stdout sink on their warn
# and error branches. None of the cases below take one today, but this file runs
# under "Stop", so an undefined sink would abort the suite rather than fail a check.
function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }

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
