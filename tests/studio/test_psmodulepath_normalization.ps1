#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Windows PowerShell 5.1 cannot load its Security module with PS7's PSModulePath
# (PowerShell/PowerShell#18681), killing the uv installer. The system dir must be
# PREPENDED, and Refresh-Environment must not reload it from the registry.
# Run: pwsh -NoProfile -File tests/studio/test_psmodulepath_normalization.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")
$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Get-Ast($path) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    return $ast
}

Write-Host "the normalization block prepends, in both entry points"
foreach ($pair in @(@{ Name = "studio/setup.ps1"; Path = $setupPath }, @{ Name = "install.ps1"; Path = $installPath })) {
    $ast = Get-Ast $pair.Path
    $text = $ast.Extent.Text
    Check "$($pair.Name) parses" $true
    Check "$($pair.Name) guards on PSEdition" ($text -match "PSVersionTable\.PSEdition -ne 'Core'")
    Check "$($pair.Name) targets the 5.1 system module dir" ($text -match "System32\\WindowsPowerShell\\v1\.0\\Modules")
    Check "$($pair.Name) puts the system dir first" (
        $text -match '\(@\(\$_UnslothSystemModules\)\s*\+\s*\$_UnslothKept\)'
    )
    Check "$($pair.Name) does not append it instead" (
        -not ($text -match '\(\$_UnslothKept\s*\+\s*@\(\$_UnslothSystemModules\)\)')
    )
}

Write-Host "Refresh-Environment leaves PSModulePath alone"
$setupAst = Get-Ast $setupPath
$fn = $setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Refresh-Environment"
}, $true)
Check "exactly one Refresh-Environment" ($fn.Count -eq 1)
$fnText = $fn[0].Extent.Text
Check "it skips PSModulePath as well as Path" ($fnText -match "\`$key -eq 'PSModulePath'")

# Computed, not listed: off Windows the body returns early, so a stale list goes unnoticed.
$setupFunctions = @{}
foreach ($f in $setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    $setupFunctions[$f.Name] = $f.Extent.Text
}
$needed = [ordered]@{}
$pending = [System.Collections.Generic.Queue[string]]::new()
$pending.Enqueue("Refresh-Environment")
$seen = @{ "Refresh-Environment" = $true }
while ($pending.Count -gt 0) {
    $current = $pending.Dequeue()
    $body = $setupFunctions[$current]
    if (-not $body) { continue }
    # Comments dropped: a function named in prose is not a call.
    $code = ($body -split "`r?`n" | Where-Object { -not ($_.TrimStart().StartsWith("#")) }) -join "`n"
    foreach ($name in $setupFunctions.Keys) {
        if ($seen[$name]) { continue }
        if ($code -match ("(?<![\w-])" + [regex]::Escape($name) + "(?![\w-])")) {
            $seen[$name] = $true
            $needed[$name] = $true
            $pending.Enqueue($name)
        }
    }
}
Check "the closure found Refresh-Environment's own helpers (bites)" ($needed.Count -ge 1)

# Off Windows this only proves the function is callable; the AST check holds the line.
$savedPath = $env:Path
$savedModulePath = $env:PSModulePath
try {
    foreach ($name in $needed.Keys) { Invoke-Expression $setupFunctions[$name] }
    Invoke-Expression $fnText
    $sentinel = "C:\__unsloth_sentinel__;C:\Windows\System32\WindowsPowerShell\v1.0\Modules"
    $env:PSModulePath = $sentinel
    Refresh-Environment
    Check "a normalized PSModulePath survives a refresh" ($env:PSModulePath -eq $sentinel)
} finally {
    $env:Path = $savedPath
    $env:PSModulePath = $savedModulePath
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "Results: $failures failed" -ForegroundColor Red
    exit 1
}
Write-Host "Results: all passed"
