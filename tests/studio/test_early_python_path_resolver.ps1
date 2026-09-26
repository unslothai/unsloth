#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The Python rung of Resolve-StudioFinalPathInfo, used when the native resolver answers nothing.
# Run: pwsh -NoProfile -File tests/studio/test_early_python_path_resolver.ps1

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
    "Get-StudioEarlyPython", "Invoke-StudioEarlyPythonScript", "Invoke-StudioEarlyPython",
    "Get-StudioPythonFinalPath",
    "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

# The native rung, forced off.
function Initialize-StudioFinalPathNativeType { return $false }
function Get-StudioNativeFinalPath { param([string]$Path) return $null }
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("earlypy-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    # The kill switch would make discovery decline; its own cases below set it explicitly.
    Remove-Item Env:UNSLOTH_EARLY_PYTHON_PROBE -ErrorAction SilentlyContinue
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $exe = Get-StudioEarlyPython
    if (-not $exe) {
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
    Write-Host "  interpreter: $exe"

    $real = Join-Path $tmp "real"
    New-Item -ItemType Directory -Force -Path $real | Out-Null
    $alias = Join-Path $tmp "alias"
    $madeAlias = $false
    try {
        if ($IsWindows -or $env:OS -eq "Windows_NT") {
            New-Item -ItemType Junction -Path $alias -Target $real -ErrorAction Stop | Out-Null
        } else {
            New-Item -ItemType SymbolicLink -Path $alias -Target $real -ErrorAction Stop | Out-Null
        }
        $madeAlias = $true
    } catch {
        Write-Host "  SKIP  could not create a directory alias: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    if ($madeAlias) {
        # Control: the lexical rung may fold a symlink on pwsh 7, but it is never exact.
        $savedFinder = ${function:Get-StudioEarlyPython}
        function Get-StudioEarlyPython { return $null }
        $inexact = Resolve-StudioFinalPathInfo -Path $alias
        Check "without the rung the alias identity is inexact (bites)" ($inexact.Exact -eq $false)
        ${function:Get-StudioEarlyPython} = $savedFinder

        $viaAlias = Get-StudioPythonFinalPath -Path $alias
        $viaReal = Get-StudioPythonFinalPath -Path $real
        Check "Python folds the alias onto its target" (
            -not [string]::IsNullOrWhiteSpace($viaAlias) -and $viaAlias -eq $viaReal)

        $infoAlias = Resolve-StudioFinalPathInfo -Path $alias
        $infoReal = Resolve-StudioFinalPathInfo -Path $real
        Check "the resolver reports the alias as exact" ($infoAlias.Exact -eq $true)
        Check "the resolver gives one identity for both spellings" ($infoAlias.Path -eq $infoReal.Path)
    }

    # No interpreter: lexical and inexact, as before. Restored afterwards so later checks use the rung.
    $savedFinder2 = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { return $null }
    $script:StudioEarlyPythonProbed = $false
    $script:StudioPythonFinalPathCache = $null
    $noPy = Resolve-StudioFinalPathInfo -Path $real
    Check "with no interpreter the identity is inexact, as before" ($noPy.Exact -eq $false)
    Check "with no interpreter a usable path still comes back" (
        -not [string]::IsNullOrWhiteSpace($noPy.Path))
    ${function:Get-StudioEarlyPython} = $savedFinder2
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    Check "the interpreter is back after the no-interpreter case" ($null -ne (Get-StudioEarlyPython))

    # The answer is hashed into a lock name, so non-ASCII must survive 5.1's console codepage.
    $unicodeName = "studio-ünïcôde-日本語-ß"
    $unicodeDir = Join-Path $tmp $unicodeName
    New-Item -ItemType Directory -Force -Path $unicodeDir | Out-Null
    $unicodeAnswer = Get-StudioPythonFinalPath -Path $unicodeDir
    Check "a non-ASCII path survives the child process" (
        -not [string]::IsNullOrWhiteSpace($unicodeAnswer) -and
        $unicodeAnswer.EndsWith($unicodeName))

    # Trim() would drop the U+00A0 and name the sibling instead.
    $nbspName = "studio-nbsp" + [char]0x00A0
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp "studio-nbsp") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp $nbspName) | Out-Null
    # .NET Framework strips it on the way in, so 5.1 cannot create or name such a directory at all.
    if (@(Get-ChildItem -LiteralPath $tmp -Name) -ccontains $nbspName) {
        $nbspAnswer = Get-StudioPythonFinalPath -Path (Join-Path $tmp $nbspName)
        Check "a trailing non-breaking space is kept" ("$nbspAnswer".EndsWith($nbspName))
    } else {
        Write-Host "  SKIP  this host strips a trailing U+00A0 from paths" -ForegroundColor Yellow
    }

    $gateScript = "import pathlib,sys" + [char]10 +
        "sys.stdout.buffer.write(str(pathlib.Path(sys.argv[1]).resolve(strict=False)).encode('utf-8'))"
    $gateAnswer = & $exe -I -c $gateScript $unicodeDir
    Check "the answer matches the runtime gate's own resolve()" ($unicodeAnswer -eq "$gateAnswer".Trim())

    $loopA = Join-Path $tmp "loop-a"
    $loopOk = $false
    try {
        New-Item -ItemType SymbolicLink -Path $loopA -Target (Join-Path $tmp "loop-b") -ErrorAction Stop | Out-Null
        New-Item -ItemType SymbolicLink -Path (Join-Path $tmp "loop-b") -Target $loopA -ErrorAction Stop | Out-Null
        $loopOk = $true
    } catch {}
    if ($loopOk) {
        $loopAnswer = Get-StudioPythonFinalPath -Path $loopA
        Check "a link loop is not promoted to an exact identity" ([string]::IsNullOrWhiteSpace($loopAnswer))
        # Test-Path reports the link itself, so the walk stops AT the link.
        $loopChild = Resolve-StudioFinalPathInfo -Path (Join-Path $loopA "studio")
        Check "a missing path under a link loop is not exact" ($loopChild.Exact -eq $false)
    }

    $dangling = Join-Path $tmp "dangling"
    $danglingOk = $false
    try {
        New-Item -ItemType SymbolicLink -Path $dangling -Target (Join-Path $tmp "gone") -ErrorAction Stop | Out-Null
        $danglingOk = $true
    } catch {
        # 5.1 refuses a link to a missing target: link first, then remove the target.
        try {
            $gone = Join-Path $tmp "gone"
            New-Item -ItemType Directory -Force -Path $gone | Out-Null
            New-Item -ItemType SymbolicLink -Path $dangling -Target $gone -ErrorAction Stop | Out-Null
            Remove-Item -LiteralPath $gone -Recurse -Force
            $danglingOk = $true
        } catch {}
    }
    if ($danglingOk) {
        foreach ($suffix in @("", "studio", "a\b")) {
            $probePath = if ($suffix) { Join-Path $dangling $suffix } else { $dangling }
            $info = Resolve-StudioFinalPathInfo -Path $probePath
            Check "a dangling link is not exact (suffix '$suffix')" ($info.Exact -eq $false)
        }
        # Where Test-Path follows the link, the stripped entry's reparse bit alone must keep it inexact.
        function Test-Path {
            param([string]$LiteralPath)
            if ($LiteralPath.StartsWith($dangling)) { return $false }
            return (Microsoft.PowerShell.Management\Test-Path -LiteralPath $LiteralPath)
        }
        try {
            foreach ($suffix in @("", "studio", "a\b")) {
                $probePath = if ($suffix) { Join-Path $dangling $suffix } else { $dangling }
                $info = Resolve-StudioFinalPathInfo -Path $probePath
                Check "a dangling link Test-Path reports missing is not exact (suffix '$suffix')" ($info.Exact -eq $false)
            }
        } finally {
            Remove-Item Function:Test-Path
        }
    }

    # 5.1 has no ArgumentList, so arguments go through one quoted string.
    $spacedDir = Join-Path $tmp "a dir with spaces"
    New-Item -ItemType Directory -Force -Path $spacedDir | Out-Null
    $spacedAnswer = Get-StudioPythonFinalPath -Path $spacedDir
    Check "a path containing spaces resolves" (
        -not [string]::IsNullOrWhiteSpace($spacedAnswer) -and $spacedAnswer.EndsWith("spaces"))

    $src = Get-Content -Raw -LiteralPath $installPs1
    Check "argument construction does not assume ArgumentList exists" (
        $src -match 'PSObject\.Properties\["ArgumentList"\]')
    Check "the probe refuses an interpreter older than 3.8" (
        $src -match 'sys\.version_info\s*<\s*\(3,\s*8\)')

    $refuser = Join-Path $tmp $(if ($IsWindows -or $env:OS -eq "Windows_NT") { "refuse.cmd" } else { "refuse.sh" })
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "@echo off`r`nexit /b 2`r`n" | Set-Content -LiteralPath $refuser -Encoding ASCII
    } else {
        "#!/bin/sh`nexit 2`n" | Set-Content -LiteralPath $refuser
        & chmod +x $refuser
    }
    Check "an interpreter that exits non-zero is rejected" (
        $null -eq (Invoke-StudioEarlyPython -Exe $refuser -Path $real))

    $slow = Join-Path $tmp $(if ($IsWindows -or $env:OS -eq "Windows_NT") { "slow.cmd" } else { "slow.sh" })
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "@echo off`r`nping -n 30 127.0.0.1 >nul`r`n" | Set-Content -LiteralPath $slow -Encoding ASCII
    } else {
        "#!/bin/sh`nsleep 30`n" | Set-Content -LiteralPath $slow
        & chmod +x $slow
    }
    $started = [DateTime]::UtcNow
    $hung = Invoke-StudioEarlyPython -Exe $slow -Path $real -TimeoutMs 2000
    $elapsed = ([DateTime]::UtcNow - $started).TotalSeconds
    Check "a hung interpreter returns null" ($null -eq $hung)
    Check "a hung interpreter is killed at the deadline, not waited out" ($elapsed -lt 15)

    # The interpreter exits at once but leaves a child holding stdout: the read is bounded too.
    $holder = Join-Path $tmp $(if ($IsWindows -or $env:OS -eq "Windows_NT") { "holder.cmd" } else { "holder.sh" })
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "@echo off`r`nstart /b `"`" ping -n 20 127.0.0.1`r`nexit /b 0`r`n" | Set-Content -LiteralPath $holder -Encoding ASCII
    } else {
        "#!/bin/sh`nsleep 20 &`nprintf '%s' `"`$5`"`nexit 0`n" | Set-Content -LiteralPath $holder
        & chmod +x $holder
    }
    $started = [DateTime]::UtcNow
    $held = Invoke-StudioEarlyPython -Exe $holder -Path $real -TimeoutMs 2000
    $elapsed = ([DateTime]::UtcNow - $started).TotalSeconds
    Check "an answer still held open past the deadline is declined" ($null -eq $held)
    Check "and the wait for it ends at the deadline" ($elapsed -lt 10)

    # This runs before the install lock, so finding an interpreter must write nothing.
    $before = @(Get-ChildItem -LiteralPath $tmp -Recurse -Force).Count
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    Remove-Item Function:\Get-StudioEarlyPython -ErrorAction SilentlyContinue
    foreach ($name in @("Get-StudioEarlyPython")) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        Invoke-Expression $fn[0].Extent.Text
    }
    $null = Get-StudioEarlyPython
    $after = @(Get-ChildItem -LiteralPath $tmp -Recurse -Force).Count
    Check "finding an interpreter writes nothing" ($after -eq $before)

    # -I alone still imports site, so a sitecustomize could print into the answer or hang.
    $exe = Get-StudioEarlyPython
    $isoOnly = (& $exe -I -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    $isoPlus = (& $exe -I -S -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    Check "-I alone leaves site imported on this interpreter" ("$isoOnly".Trim() -eq "0")
    Check "-S is what turns site off" ("$isoPlus".Trim() -eq "1")

    # Read from the source: a planted sitecustomize is shadowed by the host's own.
    $launcherFn = @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq "Invoke-StudioEarlyPythonScript"
    }, $true))[0].Extent.Text
    Check "the launcher runs every probe with -S as well as -I" (
        $launcherFn -match '@\("-I",\s*"-S",\s*"-c"')
    # A miss recorded before $VenvDir existed (the --tauri path) is probed again once it does.
    $venvHome = Join-Path $tmp "venvhome"
    $venvBin = if ($IsWindows -or $env:OS -eq "Windows_NT") { "Scripts" } else { "bin" }
    $venvLeaf = if ($IsWindows -or $env:OS -eq "Windows_NT") { "python.exe" } else { "python3" }
    New-Item -ItemType Directory -Force -Path (Join-Path $venvHome $venvBin) | Out-Null
    Copy-Item -LiteralPath $exe -Destination (Join-Path $venvHome (Join-Path $venvBin $venvLeaf)) -Force
    function Get-Command { param($Name, [switch]$All, $CommandType, $ErrorAction) return @() }
    Remove-Variable -Name VenvDir -Scope Script -ErrorAction SilentlyContinue
    Remove-Variable -Name VenvDir -Scope Global -ErrorAction SilentlyContinue
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $script:StudioEarlyPythonProbedWithoutVenv = $false
    Check "with no venv and no system Python the probe finds nothing" ($null -eq (Get-StudioEarlyPython))
    Check "and it recorded that the venv was unknown when it looked" (
        $script:StudioEarlyPythonProbedWithoutVenv -eq $true)
    $global:VenvDir = $venvHome
    $found = Get-StudioEarlyPython
    Check "once the venv directory is known the venv interpreter is found after all" (
        -not [string]::IsNullOrWhiteSpace($found))
    Check "and it is the one inside the venv, not some other copy" (
        "$found" -like ("*" + $venvBin + "*"))
    $script:ReprobeCount = 0
    function Test-Path { param($LiteralPath, $PathType, $ErrorAction) $script:ReprobeCount++; return $false }
    $null = Get-StudioEarlyPython
    Check "a hit is not probed again" ($script:ReprobeCount -eq 0)
    # An uninspectable candidate (throws under Stop) is skipped without escaping.
    function Test-Path { param($LiteralPath, $PathType, $ErrorAction)
        throw [System.UnauthorizedAccessException]::new("Access to the path '$LiteralPath' is denied.") }
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $threw = $null
    try { $none = Get-StudioEarlyPython } catch { $threw = $_ }
    Check "a candidate that cannot be inspected is skipped, not fatal" ($null -eq $threw -and $null -eq $none)
    $savedFinder2 = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { throw "discovery failed" }
    $script:StudioPythonFinalPathCache = $null
    $threw = $null
    try { $none = Get-StudioPythonFinalPath -Path $real } catch { $threw = $_ }
    Check "a discovery failure declines to the lexical rung instead of throwing" ($null -eq $threw -and $null -eq $none)
    ${function:Get-StudioEarlyPython} = $savedFinder2
    Remove-Item Function:Test-Path -ErrorAction SilentlyContinue
    Remove-Item Function:Get-Command -ErrorAction SilentlyContinue
    Remove-Variable -Name VenvDir -Scope Global -ErrorAction SilentlyContinue
    $script:ResolveCalls = 0
    $savedInvoke = ${function:Invoke-StudioEarlyPython}
    function Invoke-StudioEarlyPython { param($Exe, $Path, $TimeoutMs) $script:ResolveCalls++; return $Path }
    $savedFinder3 = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { return "python3" }
    $script:StudioPythonFinalPathCache = $null
    $null = Get-StudioPythonFinalPath -Path $real
    $null = Get-StudioPythonFinalPath -Path $real
    $null = Get-StudioPythonFinalPath -Path $real
    Check "three calls for one path spawn one child" ($script:ResolveCalls -eq 1)
    $null = Get-StudioPythonFinalPath -Path $tmp
    Check "and a different path still spawns its own" ($script:ResolveCalls -eq 2)
    function Invoke-StudioEarlyPython { param($Exe, $Path, $TimeoutMs) $script:ResolveCalls++; return $null }
    $missPath = Join-Path $tmp "no-such-thing"
    $null = Get-StudioPythonFinalPath -Path $missPath
    $null = Get-StudioPythonFinalPath -Path $missPath
    Check "a miss is remembered as well as an answer" ($script:ResolveCalls -eq 3)
    ${function:Invoke-StudioEarlyPython} = $savedInvoke
    ${function:Get-StudioEarlyPython} = $savedFinder3
    $script:StudioPythonFinalPathCache = $null

} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All early-Python path resolver checks passed"
exit 0
