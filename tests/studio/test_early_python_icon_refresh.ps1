#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Shortcut icon refresh via a child interpreter where WDAC refuses the runtime type; must never fail an install.
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

# The kill switch would make discovery decline; its own cases below set it explicitly.
Remove-Item Env:UNSLOTH_EARLY_PYTHON_PROBE -ErrorAction SilentlyContinue
$script:StudioEarlyPythonProbed = $false
$script:StudioEarlyPython = $null
if (-not (Get-StudioEarlyPython)) {
    # A runnable python on PATH means a broken extraction, not a host without Python.
    $onPath = $null
    foreach ($n in @("python3", "python")) {
        foreach ($cmd in @(Get-Command $n -All -CommandType Application -ErrorAction SilentlyContinue)) {
            if ($onPath -or -not $cmd.Source) { continue }
            $job = Start-Job -ArgumentList $cmd.Source -ScriptBlock {
                param($exe)
                $o = & $exe -I -S -c "import os,sys;sys.stdout.write(os.path.realpath('.') if sys.version_info >= (3, 8) else '')" 2>$null
                [pscustomobject]@{ Out = "$o"; Code = $LASTEXITCODE }
            }
            $ran = $null
            if (Wait-Job $job -Timeout 30) { $ran = Receive-Job $job -ErrorAction SilentlyContinue } else { Stop-Job $job }
            Remove-Job $job -Force
            if ($ran -and $ran.Code -eq 0 -and -not [string]::IsNullOrWhiteSpace($ran.Out)) { $onPath = $cmd }
        }
    }
    if ($onPath) {
        Write-Host "  FAIL  Get-StudioEarlyPython found nothing, yet $($onPath.Source) is on PATH." -ForegroundColor Red
        Write-Host "        That is a broken extraction in this file, not a host without Python." -ForegroundColor Red
        exit 1
    }
    Write-Host "  SKIP  no Python on this host, which is the fallback case and not a failure" -ForegroundColor Yellow
    exit 0
}


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

    # Only the child's answer counts as success.
    $script:RunnerOutput = ""
    Check "control: a silent child is not success" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false)
    $script:RunnerOutput = "something else"
    Check "control: an unexpected answer is not success" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false)
    $script:RunnerOutput = "ok"

    $script:RunnerThrows = $true
    $threw = $false
    try { $null = Invoke-StudioPythonShellIconRefresh -Paths $links } catch { $threw = $true }
    Check "a child that throws does not escape into the install" ($threw -eq $false)
    $script:RunnerThrows = $false

    Check "an empty shortcut list is handled" (
        $null -ne (Invoke-StudioPythonShellIconRefresh -Paths @()))

    # A passed interpreter wins over a cached discovery miss from before the install.
    $savedFinder = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython {
        if ($script:StudioEarlyPythonProbed) { return $script:StudioEarlyPython }
        $script:StudioEarlyPythonProbed = $true
        $script:StudioEarlyPython = $script:FakeHostPython
        return $script:StudioEarlyPython
    }
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $script:FakeHostPython = $null
    $null = Get-StudioEarlyPython
    $script:FakeHostPython = "C:\Studio\venv\Scripts\python.exe"

    Check "control: discovery is still stuck on the cached miss (bites)" (
        $null -eq (Get-StudioEarlyPython))
    $script:RunnerCalls = 0
    $ok = Invoke-StudioPythonShellIconRefresh -Paths $links -Exe $script:FakeHostPython
    Check "a fresh install refreshes through the interpreter it just installed" (
        $ok -eq $true -and $script:RunnerCalls -eq 1)
    ${function:Get-StudioEarlyPython} = $savedFinder
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null

    $shortcutFn = @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq "New-StudioShortcuts"
    }, $true))[0].Extent.Text
    Check "New-StudioShortcuts passes its own interpreter to the refresh" (
        $shortcutFn -match '(?s)Invoke-StudioPythonShellIconRefresh[^\r\n]*[\r\n\s`]*-Paths \$createdShortcutPaths -Exe \$ManagedPythonPath')

    # UNSLOTH_EARLY_PYTHON_PROBE=0 must win even over a passed interpreter (a past regression).
    $savedProbe = $env:UNSLOTH_EARLY_PYTHON_PROBE
    try {
        $env:UNSLOTH_EARLY_PYTHON_PROBE = "0"
        $script:RunnerCalls = 0
        Check "with the probe disabled an explicit interpreter is still refused" (
            (Invoke-StudioPythonShellIconRefresh -Paths $links -Exe "C:\Studio\venv\Scripts\python.exe") -eq $false)
        Check "and no child process was started" ($script:RunnerCalls -eq 0)
        $env:UNSLOTH_EARLY_PYTHON_PROBE = " 0 "
        Check "the switch is read with surrounding whitespace trimmed" (
            (Invoke-StudioPythonShellIconRefresh -Paths $links -Exe "C:\Studio\venv\Scripts\python.exe") -eq $false)
        $env:UNSLOTH_EARLY_PYTHON_PROBE = "1"
        $script:RunnerCalls = 0
        Check "control: with the probe enabled the same call goes through" (
            (Invoke-StudioPythonShellIconRefresh -Paths $links -Exe "C:\Studio\venv\Scripts\python.exe") -eq $true -and
            $script:RunnerCalls -eq 1)
    } finally {
        if ($null -eq $savedProbe) { Remove-Item Env:UNSLOTH_EARLY_PYTHON_PROBE -ErrorAction SilentlyContinue }
        else { $env:UNSLOTH_EARLY_PYTHON_PROBE = $savedProbe }
    }

    $env:OS = "Linux"
    $script:RunnerCalls = 0
    Check "off Windows the rung declines: there is no shell32 there" (
        (Invoke-StudioPythonShellIconRefresh -Paths $links) -eq $false -and $script:RunnerCalls -eq 0)
} finally {
    if ($null -eq $savedOs) { Remove-Item Env:OS -ErrorAction SilentlyContinue } else { $env:OS = $savedOs }
}


$fnAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Invoke-StudioPythonShellIconRefresh"
}, $true)[0]
$probeText = $fnAst.Extent.Text

# Per-item SHCNE_UPDATEITEM is required (the global broadcast misses in-place .lnk rewrites); SHCNF_FLUSH since the child exits at once.
Check "the probe sends SHCNE_UPDATEITEM with SHCNF_PATHW per shortcut, flushed" (
    $probeText -match "SHChangeNotify\(0x00002000,0x1005,p,None\)")
Check "the probe still sends the global SHCNE_ASSOCCHANGED broadcast, flushed" (
    $probeText -match "SHChangeNotify\(0x08000000,0x1000,None,None\)")
Check "the probe declares SHChangeNotify's signature" (
    $probeText -match "SHChangeNotify\.argtypes" -and $probeText -match "LPCWSTR")

# Fallback only: when the type can be defined the installer must not reach the child.
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
