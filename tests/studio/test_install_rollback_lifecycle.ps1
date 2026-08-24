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

# The six functions under test. Extracting only these is what broke when #9501 added a
# sibling helper they call: the list is maintained by hand, so it goes stale the moment
# a helper grows a dependency, and the failure surfaces as "not recognized" inside an
# unrelated case rather than as a missing extraction. Extract the transitive closure
# instead, so a new callee is picked up without anyone remembering to add it here.
$subjectNames = @(
    "Start-StudioVenvRollback",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Restore-StudioVenvRollback",
    "Complete-StudioVenvRollback"
)

$definitions = @{}
foreach ($node in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)) {
    if ($definitions.ContainsKey($node.Name)) {
        throw "install.ps1 defines $($node.Name) more than once; the extraction cannot tell which one is under test"
    }
    $definitions[$node.Name] = $node
}

foreach ($name in $subjectNames) {
    if (-not $definitions.ContainsKey($name)) {
        throw "expected $name in install.ps1, found no definition (renamed or removed?)"
    }
}

# Sinks this file stubs below. The walk stops at them, so their own dependencies (ANSI
# colouring and the like) are not dragged in for functions that are replaced anyway.
$stubbedNames = @("substep", "Write-StudioLine")

# Breadth-first over the call graph, following only names install.ps1 itself defines.
# Anything else is a real cmdlet or one of the stubs below and must not be extracted.
$extracted = [System.Collections.Generic.List[string]]::new()
$seen = @{}
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($name in $subjectNames) { $queue.Enqueue($name) }
while ($queue.Count -gt 0) {
    $name = $queue.Dequeue()
    if ($seen.ContainsKey($name)) { continue }
    $seen[$name] = $true
    $extracted.Add($name)
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if ($callee -and $definitions.ContainsKey($callee) -and
            $stubbedNames -notcontains $callee -and -not $seen.ContainsKey($callee)) {
            $queue.Enqueue($callee)
        }
    }
}

foreach ($name in $extracted) { Invoke-Expression $definitions[$name].Extent.Text }

$pulledIn = @($extracted | Where-Object { $subjectNames -notcontains $_ })
if ($pulledIn.Count -gt 0) { Write-Host "Extracted dependencies: $($pulledIn -join ', ')" }

function substep { param([string]$Message, [string]$Color) }
# The rollback helpers report through install.ps1's UTF-8 stdout sink on their warn
# and error branches. None of the cases below take one today, but this file runs
# under "Stop", so an undefined sink would abort the suite rather than fail a check.
function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }

# Fail here, with the caller named, rather than 60 lines further down as "the term X is
# not recognized" inside whichever case happened to reach it first. The closure above
# makes an install.ps1-defined callee unreachable by construction; this catches the rest,
# including a helper that calls a sink this file forgot to stub.
$unresolved = [System.Collections.Generic.List[string]]::new()
foreach ($name in $extracted) {
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if (-not $callee) { continue }
        if (-not (Get-Command -Name $callee -ErrorAction SilentlyContinue)) {
            $unresolved.Add("$callee (called by $name)")
        }
    }
}
if ($unresolved.Count -gt 0) {
    throw "extracted helpers call commands that do not resolve: $(($unresolved | Sort-Object -Unique) -join '; ')"
}

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
