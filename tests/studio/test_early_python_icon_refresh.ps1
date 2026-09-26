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
    "Invoke-StudioPythonShellIconRefresh",
    # What Get-StudioEarlyPython reaches on Windows: the elevation gate and its helpers. Off
    # Windows it never calls them, which is how a missing one once passed here and failed CI.
    "Test-StudioChildScriptDirectoryElevated", "Get-StudioSystem32Tool", "Test-StudioPathUnderAdminRoot",
    "Test-StudioSddlRightsAreWrite", "Test-StudioSddlPrincipalIsAdminOnly",
    "Test-StudioSddlWritableByNonAdmin", "Test-StudioDirectoryIsAdminOnly", "Test-StudioInterpreterFileIsAdminOnly",
    "Get-StudioLexicalParent", "Write-StudioFinalPathDegraded", "Write-StudioLine"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

if ($env:OS -eq "Windows_NT") { function Test-StudioChildScriptDirectoryElevated { return $false } }

$script:StudioEarlyPythonProbed = $false
$script:StudioEarlyPython = $null
if (-not (Get-StudioEarlyPython)) {
    # "No interpreter" is a real and supported state, so it is a skip. But it is also what a
    # BROKEN EXTRACTION looks like from here: a helper this file forgot to pull out of
    # install.ps1 makes Get-StudioEarlyPython fail, the probe finds nothing, and the suite exits 0
    # having tested nothing. That happened once and CI recorded it as a pass. Tell the two apart.
    # The documented opt-out first. UNSLOTH_EARLY_PYTHON_PROBE=0 means "do not spawn an interpreter
    # on this host", so discovery returning nothing is the switch working, not a broken extraction.
    if ("$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0") {
        Write-Host "  SKIP  UNSLOTH_EARLY_PYTHON_PROBE=0, so this rung is switched off by request" -ForegroundColor Yellow
        exit 0
    }
    $elevatedHost = $false
    if ($env:OS -eq "Windows_NT") { try { $elevatedHost = [bool](Test-StudioChildScriptDirectoryElevated) } catch { $elevatedHost = $true } }
    $usable = $null
    foreach ($n in @("python3", "python")) {
        foreach ($src in @(Get-Command $n -All -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)) {
            if ($usable) { break }
            if ([string]::IsNullOrWhiteSpace($src)) { continue }
            if ("$src" -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]') { continue }
            # The installer's own elevated rule: an elevated run refuses an interpreter a standard user
            # can replace, so such a candidate is not one discovery should have found.
            if ($elevatedHost -and -not (Test-StudioPathUnderAdminRoot -Path $src)) { continue }
            $here = Split-Path -Parent $src
            if ([string]::IsNullOrWhiteSpace($here)) { continue }
            # In a job with a deadline: a shim that starts and never exits must be a skip, not a hang.
            $job = Start-Job -ArgumentList $src, $here -ScriptBlock {
                param($exe, $dir)
                "$(& $exe -I -S -c "import pathlib,sys`nsys.exit(2) if sys.version_info < (3,8) else None`nsys.stdout.write(str(pathlib.Path(sys.argv[1]).resolve(strict=True)))" $dir 2>$null)"
            }
            $answer = ""
            if (Wait-Job $job -Timeout 30) { $answer = "$(Receive-Job $job -ErrorAction SilentlyContinue)".Trim() } else { Stop-Job $job }
            Remove-Job $job -Force
            if (-not [string]::IsNullOrWhiteSpace($answer)) { $usable = $src }
        }
    }
    if ($usable) {
        Write-Host "  FAIL  Get-StudioEarlyPython found nothing, yet $usable answers the same probe." -ForegroundColor Red
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

    Check "an elevated run still refreshes through the venv interpreter" (
        $shortcutFn -notmatch 'Test-StudioPathUnderAdminRoot' -and
        $shortcutFn -notmatch 'iconRefreshSafe')

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

$source = Get-Content -LiteralPath $installPs1 -Raw
Check "no emitted shell-icon type is left to try first" ($source -notmatch "UnslothShellIconRefresh")
Check "the refresh goes straight through the child interpreter" (
    $source -match "Invoke-StudioPythonShellIconRefresh")
# Cosmetic, and it must stay that way: a failure here cannot be allowed to fail an install.
# Single-quoted, so PowerShell does not eat the `$ as an escape and quietly change the pattern.
Check "the call is still wrapped so a failure cannot fail the install" (
    $source -match '(?s)try \{[^{}]*Invoke-StudioPythonShellIconRefresh[^{}]*\} catch \{\}')

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
