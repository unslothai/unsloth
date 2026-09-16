#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Refresh a rewritten shortcut's icon through a child interpreter where the shell cannot define
# the type to do it itself.
#
# The installer tells Explorer about each shortcut it wrote with SHChangeNotify, because the
# global broadcast alone misses a same-name .lnk rewritten in place. That call needs a type
# defined at runtime, which Constrained Language Mode and WDAC Dynamic Code Security both refuse,
# and on those hosts the refresh simply did not happen: the shortcut worked, its icon was stale.
#
# This is cosmetic in both directions. A stale icon is not a broken install, and nothing in this
# path may fail one, so every check below is as much about the rung staying silent as about it
# working.
# Run: pwsh -NoProfile -File tests/studio/test_early_python_icon_refresh.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = Join-Path $root "install.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }
foreach ($name in @(
    "Invoke-StudioEarlyPythonScript", "Invoke-StudioEarlyPython", "Get-StudioEarlyPython",
    "Invoke-StudioPythonShellIconRefresh"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

$script:StudioEarlyPythonProbed = $false
$script:StudioEarlyPython = $null
if (-not (Get-StudioEarlyPython)) {
    Write-Host "  SKIP  no Python on this host, which is the fallback case and not a failure" -ForegroundColor Yellow
    exit 0
}

# ------------------------------------------------------------- the rung, through a stub runner

$script:RunnerCalls = 0
$script:RunnerArgs = @()
$script:RunnerOutput = "ok"
$script:RunnerThrows = $false
function Invoke-StudioEarlyPythonScript {
    param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
    $script:RunnerCalls++
    $script:RunnerArgs = $ScriptArgs
    if ($script:RunnerThrows) { throw "the child went wrong" }
    return $script:RunnerOutput
}

$savedOs = $env:OS
try {
    $env:OS = "Windows_NT"

    $script:RunnerCalls = 0
    $links = @("C:\a\Unsloth Studio.lnk", "C:\b\Unsloth Studio.lnk")
    $ok = Invoke-StudioPythonShellIconRefresh -Paths $links
    Check "a child that reports success is believed" ($ok -eq $true)
    Check "every shortcut written is named to the child, not just the last" (
        $script:RunnerCalls -eq 1 -and ($script:RunnerArgs -join "|") -eq ($links -join "|"))

    # Bites control: the answer has to come from the child. Anything else is not success, or the
    # rung would report a refresh on a host where nothing happened.
    $script:RunnerOutput = ""
    Check "control: a silent child is not success" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false)
    $script:RunnerOutput = "something else"
    Check "control: an unexpected answer is not success" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false)
    $script:RunnerOutput = "ok"

    # Nothing here may fail an install. A throw from the runner has to surface as a false, since
    # the caller's own catch is what would otherwise be relied on to swallow it.
    $script:RunnerThrows = $true
    $threw = $false
    try { $null = Invoke-StudioPythonShellIconRefresh -Paths $links } catch { $threw = $true }
    Check "a child that throws does not escape into the install" ($threw -eq $false)
    $script:RunnerThrows = $false

    # No shortcuts written means nothing to announce per item, but the global broadcast still
    # costs one child and is harmless, so the contract is only that it does not throw.
    Check "an empty shortcut list is handled" (
        $null -ne (Invoke-StudioPythonShellIconRefresh -Paths @()))

    $env:OS = "Linux"
    $script:RunnerCalls = 0
    Check "off Windows the rung declines: there is no shell32 there" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false -and $script:RunnerCalls -eq 0)
} finally {
    if ($null -eq $savedOs) { Remove-Item Env:OS -ErrorAction SilentlyContinue } else { $env:OS = $savedOs }
}

# ------------------------------------------------------ the probe, and the rung above it

$fnAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Invoke-StudioPythonShellIconRefresh"
}, $true)[0]
$probeText = $fnAst.Extent.Text

# The per-item notification is the one that matters: SHCNE_UPDATEITEM with SHCNF_PATHW. The
# global SHCNE_ASSOCCHANGED broadcast misses a same-name .lnk rewritten in place, which is what
# an update does every single time.
Check "the probe sends SHCNE_UPDATEITEM with SHCNF_PATHW per shortcut" (
    $probeText -match "SHChangeNotify\(0x00002000,0x0005,p,None\)")
Check "the probe still sends the global SHCNE_ASSOCCHANGED broadcast" (
    $probeText -match "SHChangeNotify\(0x08000000,0,None,None\)")
# ctypes defaults an undeclared argument to a C int, which would truncate a pointer on 64 bit.
Check "the probe declares SHChangeNotify's signature" (
    $probeText -match "SHChangeNotify\.argtypes" -and $probeText -match "LPCWSTR")

# The rung above is untouched: this is a fallback, not a replacement. If the type can be defined,
# the installer must still use it and never reach the child.
$source = Get-Content -LiteralPath $installPs1 -Raw
Check "the type-defining rung is still tried first" (
    $source -match "UnslothShellIconRefresh`" -as \[type\]")
Check "the child is reached only from the catch of that rung" (
    $source -match "(?s)\}\s*catch\s*\{[^}]*?Invoke-StudioPythonShellIconRefresh")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
