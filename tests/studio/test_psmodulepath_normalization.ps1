#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for the PSModulePath normalization in studio/setup.ps1 and install.ps1.
#
# Windows PowerShell 5.1 cannot load its own Microsoft.PowerShell.Security when it
# inherits PowerShell 7's PSModulePath, which happens whenever a process sits
# between pwsh and powershell.exe (PowerShell/PowerShell#18681). Astral's uv
# installer calls Get-ExecutionPolicy out of that module, so the install dies.
#
# Two properties have to hold together, and the second is easy to lose: the
# system module directory must be PREPENDED (appending still finds the PS7 copy
# first), and Refresh-Environment must not reload PSModulePath from the registry
# afterwards, because most of its callers run before the uv installer.
#
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
    # Prepended, not appended: @($sys) + $kept, never $kept + @($sys).
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

# Everything Refresh-Environment calls, defined before it runs. Extracting the one function
# alone worked on Linux and failed on Windows, which is the only host where this leg does
# anything: GetEnvironmentVariables('Machine') is empty off Windows, so the body returned before
# reaching Test-ActiveCondaEnvironment and the missing definition never showed. A helper added to
# Refresh-Environment is therefore invisible here until a Windows runner picks it up, so the
# closure is computed rather than listed.
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
    # Whole-line comments dropped first: a function named in prose is not a call, and pulling it
    # in would grow this closure to satisfy a mention rather than a dependency.
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

# Behavioural: the registry reload must not clobber a value already normalized. Off Windows the
# machine environment block is empty, so this leg only proves the function is callable there; the
# AST check above is what holds the line cross-platform.
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
