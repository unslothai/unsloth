#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Refresh a rewritten shortcut's icon through a child interpreter where the shell cannot define
# the type to do it itself.
#
# The installer tells Explorer about each shortcut it wrote with SHChangeNotify, because the
# global broadcast alone misses a same-name .lnk rewritten in place. That call needs a type
# defined at runtime, which WDAC Dynamic Code Security refuses, and on those hosts the refresh
# simply did not happen: the shortcut worked, its icon was stale. (Constrained Language Mode never
# reaches it: WScript.Shell is not an allowed COM object there, so no shortcut is written.)
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
    # "No interpreter" is a real and supported state, so it is a skip. But it is also what a
    # BROKEN EXTRACTION looks like from here: a helper this file forgot to pull out of
    # install.ps1 makes Get-StudioEarlyPython fail, the probe finds nothing, and the suite exits 0
    # having tested nothing. That happened once and CI recorded it as a pass. Tell the two apart.
    # Only an interpreter that actually runs counts. A Store execution alias, a broken
    # executable or a Python too old for the probe is on PATH yet is correctly rejected, and
    # that host is the supported skip. This check uses none of the extracted helpers, so a
    # broken extraction still shows up as a runnable interpreter the ladder did not find.
    $onPath = $null
    foreach ($n in @("python3", "python")) {
        foreach ($cmd in @(Get-Command $n -All -CommandType Application -ErrorAction SilentlyContinue)) {
            if ($onPath -or -not $cmd.Source) { continue }
            # In a job with a deadline: a shim that starts and never exits must be a skip, not a hang.
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

    # An interpreter handed in is used even when discovery still holds a miss from before the
    # install provided one: probe first with nothing on the host, install after, then refresh.
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
    $null = Get-StudioEarlyPython          # the install lock's probe, on a host with no Python
    $script:FakeHostPython = "C:\Studio\venv\Scripts\python.exe"   # then the install provides one

    Check "control: discovery is still stuck on the cached miss (bites)" (
        $null -eq (Get-StudioEarlyPython))
    $script:RunnerCalls = 0
    $ok = Invoke-StudioPythonShellIconRefresh -Paths $links -Exe $script:FakeHostPython
    Check "a fresh install refreshes through the interpreter it just installed" (
        $ok -eq $true -and $script:RunnerCalls -eq 1)
    ${function:Get-StudioEarlyPython} = $savedFinder
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null

    # And the shortcut writer hands it over. The check above drives the helper directly, so it
    # would pass on its own with the call site still relying on discovery.
    $shortcutFn = @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq "New-StudioShortcuts"
    }, $true))[0].Extent.Text
    Check "New-StudioShortcuts passes its own interpreter to the refresh" (
        $shortcutFn -match '(?s)Invoke-StudioPythonShellIconRefresh[^\r\n]*[\r\n\s`]*-Paths \$createdShortcutPaths -Exe \$ManagedPythonPath')

    # The kill switch must still win, even with an interpreter handed straight in.
    #
    # UNSLOTH_EARLY_PYTHON_PROBE=0 means "do not spawn an interpreter on this host". That is a
    # statement about the host, not about how the path was obtained, and Get-StudioEarlyPython
    # honours it. Passing $ManagedPythonPath to fix the fresh-install case routed around
    # discovery and therefore around the switch, so a host that had opted out got a child process
    # anyway. Driven both ways, because the fix is one line and easy to lose.
    $savedProbe = $env:UNSLOTH_EARLY_PYTHON_PROBE
    try {
        $env:UNSLOTH_EARLY_PYTHON_PROBE = "0"
        $script:RunnerCalls = 0
        Check "with the probe disabled an explicit interpreter is still refused" (
            (Invoke-StudioPythonShellIconRefresh -Paths $links -Exe "C:\Studio\venv\Scripts\python.exe") -eq $false)
        Check "and no child process was started" ($script:RunnerCalls -eq 0)
        # Whitespace and other values must not accidentally disable it.
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

# ------------------------------------------------------ the probe, and the rung above it

$fnAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Invoke-StudioPythonShellIconRefresh"
}, $true)[0]
$probeText = $fnAst.Extent.Text

# The per-item notification is the one that matters: SHCNE_UPDATEITEM with SHCNF_PATHW. The
# global SHCNE_ASSOCCHANGED broadcast misses a same-name .lnk rewritten in place, which is what
# an update does every single time.
# Both carry SHCNF_FLUSH (0x1000): the child exits right after, and an unflushed notification is
# only queued, so it can be lost with the child still answering ok.
Check "the probe sends SHCNE_UPDATEITEM with SHCNF_PATHW per shortcut, flushed" (
    $probeText -match "SHChangeNotify\(0x00002000,0x1005,p,None\)")
Check "the probe still sends the global SHCNE_ASSOCCHANGED broadcast, flushed" (
    $probeText -match "SHChangeNotify\(0x08000000,0x1000,None,None\)")
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
