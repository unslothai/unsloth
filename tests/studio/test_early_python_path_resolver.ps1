#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ask Python for an exact path identity, which is the only way the installer asks for one now.
#
# pathlib.Path.resolve calls GetFinalPathNameByHandleW on Windows, so it follows junctions, symlinks
# and SUBST drives, expands 8.3 names and reports the stored casing: the same answer the emitted
# CreateFileW / GetFinalPathNameByHandleW rung used to give, from the same system call.
# unsloth_cli/_studio_runtime_gate.py already derives the runtime lock name from the same
# expression, so an answer from this rung agrees with a running Unsloth by construction rather than
# by two implementations happening to match.
#
# This rung was additive when it was added, sitting under an emitted resolver. That resolver is gone
# -- it was the largest single contributor to a behavioural antivirus verdict on the shipped
# installer -- so this is the top rung, and the lexical answer with Exact = $false is what sits
# below it.
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
$wanted = @(
    "Get-StudioEarlyPython", "Invoke-StudioEarlyPython", "Invoke-StudioEarlyPythonScript",
    "Get-StudioPythonFinalPath",
    "New-StudioChildScriptDirectory",
    "Test-StudioChildScriptDirectoryElevated",
    "Get-StudioSystem32Tool",
    "Test-StudioPathUnderAdminRoot",
    "Test-StudioSddlRightsAreWrite", "Test-StudioSddlPrincipalIsAdminOnly",
    "Test-StudioSddlWritableByNonAdmin", "Test-StudioDirectoryIsAdminOnly",
    "Get-StudioLexicalParent",
    "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo", "Resolve-StudioFinalPathsInOneChild",
    # Called by Resolve-StudioFinalPathInfo on the rung below this one. Extracted rather than
    # stubbed: whether the degradation is announced is part of what the ladder owes its caller.
    "Write-StudioFinalPathDegraded"
)
# Source text of a named function, for the checks that read code rather than run it.
function Get-FunctionTextOrEmpty($path, $name) {
    $a = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$null)
    $f = @($a.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true))
    if ($f.Count -lt 1) { return "" }
    return $f[0].Extent.Text
}

$extracted = @{}
foreach ($name in $wanted) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}


# The emitted rung that used to sit above this one is gone, so nothing needs forcing off: the
# interpreter IS the exact rung now, and the lexical answer is what sits below it.
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("earlypy-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $exe = Get-StudioEarlyPython
    if (-not $exe) {
        # "No interpreter" is a real and supported state, so it is a skip and not a failure. But
        # it is also what a BROKEN EXTRACTION looks like from here: a helper this file forgot to
        # pull out of install.ps1 makes Get-StudioEarlyPython fail, the probe finds nothing, and
        # the suite exits 0 having tested nothing. That happened, and CI recorded it as a pass.
        # So the two are told apart before deciding: if this host has a python on PATH, the
        # extraction is at fault, not the host.
        # The documented opt-out first. UNSLOTH_EARLY_PYTHON_PROBE=0 means "do not spawn an interpreter
        # on this host", so discovery returning nothing is the switch working, not a broken extraction.
        if ("$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0") {
            Write-Host "  SKIP  UNSLOTH_EARLY_PYTHON_PROBE=0, so this rung is switched off by request" -ForegroundColor Yellow
            exit 0
        }
        # USABLE, not merely present. The installer rejects an interpreter that cannot complete its own
        # probe, so presence on PATH is not proof that discovery should have found one: Python 2, any
        # Python below 3.8 (whose Windows resolve() does not follow links), a broken executable, and a
        # Windows Store App Execution Alias are all on PATH and all correctly refused. Failing here on
        # those hosts blames this file for the installer behaving as designed.
        #
        # So ask the same question Invoke-StudioEarlyPython asks, of the same candidates, and only then
        # decide. The WindowsApps aliases are excluded WITHOUT running them: they are zero-length stubs
        # that open the Microsoft Store, which a test must not do to whoever is running it.
        $usable = $null
        foreach ($n in @("python3", "python")) {
            foreach ($src in @(Get-Command $n -All -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)) {
                if ($usable) { break }
                if ([string]::IsNullOrWhiteSpace($src)) { continue }
                if ("$src" -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]') { continue }
                $here = Split-Path -Parent $src
                if ([string]::IsNullOrWhiteSpace($here)) { continue }
                $answer = ""
                try {
                    $answer = "$(& $src -I -S -c "import pathlib,sys`nsys.exit(2) if sys.version_info < (3,8) else None`nsys.stdout.write(str(pathlib.Path(sys.argv[1]).resolve(strict=True)))" $here 2>$null)".Trim()
                } catch { $answer = "" }
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

    # ---- The batch, which exists for one caller and one number ----
    #
    # Get-RunningStudioVenvProcesses resolves the image path of every running process, and the
    # install scan repeats that for each protected root. One child per resolution meant a few
    # hundred interpreter launches on an ordinary machine, where the resolver this replaced made
    # an in-process call. Counted here rather than reasoned about.
    $script:StudioPythonFinalPathCache = $null
    $batchDir = Join-Path $tmp "batch"
    $null = New-Item -ItemType Directory -Path $batchDir -Force
    $batchPaths = @()
    foreach ($i in 1..12) {
        $leaf = Join-Path $batchDir "f$i.txt"
        Set-Content -LiteralPath $leaf -Value "x" -Encoding utf8
        $batchPaths += $leaf
    }

    # A counter around the single-path rung, so what is measured is children avoided rather than
    # a cache being populated.
    $script:RealInvoke = ${function:Invoke-StudioEarlyPython}
    $script:SingleCalls = 0
    function Invoke-StudioEarlyPython {
        param([string]$Exe, [string]$Path, [int]$TimeoutMs = 10000)
        $script:SingleCalls++
        return (& $script:RealInvoke -Exe $Exe -Path $Path -TimeoutMs $TimeoutMs)
    }

    $script:SingleCalls = 0
    foreach ($leaf in $batchPaths) { $null = Get-StudioPythonFinalPath -Path $leaf }
    $unbatched = $script:SingleCalls
    Check "unbatched, every resolution starts its own child (bites)" ($unbatched -eq $batchPaths.Count)

    $script:StudioPythonFinalPathCache = $null
    $script:SingleCalls = 0
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    Check "the batch starts no single-path child of its own" ($script:SingleCalls -eq 0)
    foreach ($leaf in $batchPaths) { $null = Get-StudioPythonFinalPath -Path $leaf }
    Check "and every resolution after it is served without one" ($script:SingleCalls -eq 0)

    # An answer, not just an absence of children. The batch must return exactly what the
    # single-path rung returns for the same path, or it is a second implementation of identity.
    $script:StudioPythonFinalPathCache = $null
    $one = Get-StudioPythonFinalPath -Path $batchPaths[0]
    $script:StudioPythonFinalPathCache = $null
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    $batched = Get-StudioPythonFinalPath -Path $batchPaths[0]
    Check "the batched answer is a real path (bites)" (-not [string]::IsNullOrWhiteSpace($one))
    Check "the batch answers byte for byte what the single rung answers" ($batched -ceq $one)

    # A path the child cannot resolve is recorded as a miss rather than left absent, or the whole
    # scan re-asks it once per protected root.
    $script:StudioPythonFinalPathCache = $null
    $missing = Join-Path $batchDir "no-such-file"
    Resolve-StudioFinalPathsInOneChild -Paths @($missing)
    Check "an unresolvable path is recorded" ($script:StudioPythonFinalPathCache.ContainsKey($missing))
    Check "and recorded as no exact answer" ($null -eq $script:StudioPythonFinalPathCache[$missing])
    $script:SingleCalls = 0
    $null = Get-StudioPythonFinalPath -Path $missing
    Check "and is not re-asked" ($script:SingleCalls -eq 0)

    # A batch that FAILS is not a batch of misses, and telling them apart is the whole point of
    # the child answering for every path it was given. A child that never started or hit the
    # timeout says nothing; caching that as "all unresolvable" switches the single-path rung off
    # for the rest of the run, and the process scan then compares lexical spellings, so an
    # executable reached through a junction stops matching its protected root and the installer
    # overwrites an environment that is in use.
    $script:StudioPythonFinalPathCache = $null
    $script:RealScript = ${function:Invoke-StudioEarlyPythonScript}
    function Invoke-StudioEarlyPythonScript {
        param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
        return ""
    }
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    $cachedAfterFailure = 0
    if ($null -ne $script:StudioPythonFinalPathCache) { $cachedAfterFailure = $script:StudioPythonFinalPathCache.Count }
    Check "a silent child caches nothing at all" ($cachedAfterFailure -eq 0)
    ${function:Invoke-StudioEarlyPythonScript} = $script:RealScript
    $script:SingleCalls = 0
    $null = Get-StudioPythonFinalPath -Path $batchPaths[0]
    Check "so the single-path rung still gets its turn" ($script:SingleCalls -eq 1)

    # And a child that answers about only SOME of what it was asked leaves the rest uncached,
    # rather than inferring a miss from an absence. A truncated pipe is not evidence.
    $script:StudioPythonFinalPathCache = $null
    function Invoke-StudioEarlyPythonScript {
        param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
        return ($script:PartialAnswer)
    }
    $script:PartialAnswer = "$($batchPaths[0])|$($batchPaths[0])"
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    Check "a partial answer caches only what was answered" (
        $script:StudioPythonFinalPathCache.Count -eq 1 -and
        $script:StudioPythonFinalPathCache.ContainsKey($batchPaths[0]))
    Check "and leaves the unanswered paths for the single rung" (
        -not $script:StudioPythonFinalPathCache.ContainsKey($batchPaths[1]))
    ${function:Invoke-StudioEarlyPythonScript} = $script:RealScript

    # The BOM. Windows PowerShell 5.1 is the interpreter the desktop app spawns, and there
    # -Encoding UTF8 means UTF-8 WITH a BOM: "any Unicode encoding, except UTF7, always creates a
    # BOM" (about_Character_Encoding, 5.1). Read as plain utf-8 the BOM joins the FIRST path,
    # resolve() rejects it, and exactly one entry per batch silently loses its exact identity on
    # every Windows host. This engine writes no BOM, so the behaviour cannot be reproduced here;
    # what is checked is that the reader tolerates one either way, run for real.
    $bomFile = Join-Path $batchDir "with-bom.txt"
    $bomBytes = [byte[]](0xEF, 0xBB, 0xBF) + [System.Text.Encoding]::UTF8.GetBytes(($batchPaths -join "`n"))
    [System.IO.File]::WriteAllBytes($bomFile, $bomBytes)
    $readerProbe = Get-FunctionTextOrEmpty $installPs1 "Resolve-StudioFinalPathsInOneChild"
    Check "the reader strips a byte order mark" ($readerProbe -match "encoding='utf-8-sig'")
    Check "and does not read the list as plain utf-8" ($readerProbe -notmatch "encoding='utf-8'\)")
    # Run it: the same expression against a real BOM-prefixed file must resolve every line,
    # including the first, or the check above is only spelling.
    $bomScript = "import pathlib,sys" + [char]10 +
        "n=0" + [char]10 +
        "with open(sys.argv[1],'r',encoding='utf-8-sig') as fh:" + [char]10 +
        "    for line in fh:" + [char]10 +
        "        p=line.rstrip('\r\n')" + [char]10 +
        "        if not p: continue" + [char]10 +
        "        try:" + [char]10 +
        "            pathlib.Path(p).resolve(strict=True)" + [char]10 +
        "            n+=1" + [char]10 +
        "        except Exception:" + [char]10 +
        "            pass" + [char]10 +
        "sys.stdout.buffer.write(str(n).encode('utf-8'))"
    $bomCount = Invoke-StudioEarlyPythonScript -Exe $exe -Script $bomScript -ScriptArgs @($bomFile) -TimeoutMs 30000
    Check "a BOM-prefixed list resolves every line, first one included" (
        "$bomCount".Trim() -eq "$($batchPaths.Count)")
    # The control that makes it bite: plain utf-8 must LOSE the first line.
    $bomScriptPlain = $bomScript -replace "utf-8-sig", "utf-8"
    $plainCount = Invoke-StudioEarlyPythonScript -Exe $exe -Script $bomScriptPlain -ScriptArgs @($bomFile) -TimeoutMs 30000
    Check "and plain utf-8 really does lose it (bites)" (
        "$plainCount".Trim() -eq "$($batchPaths.Count - 1)")

    # The list goes through a file, not argv: a few hundred image paths pass the 32767 character
    # Windows command line limit, and that would fail the whole batch rather than one path.
    $batchText = Get-FunctionTextOrEmpty $installPs1 "Resolve-StudioFinalPathsInOneChild"
    Check "the batch hands the child a list FILE rather than an argv of paths" (
        $batchText -match 'Set-Content -LiteralPath \$listFile' -and
        $batchText -match 'ScriptArgs @\(\$listFile\)')
    # Same expression as the single-path rung, or the two can disagree about identity.
    Check "the batch resolves with the same expression the single rung uses" (
        $batchText -match 'pathlib\.Path\(p\)\.resolve\(strict=True\)')
    Check "and keeps the same version gate" ($batchText -match 'sys\.version_info < \(3,8\)')
    Check "and applies the same two post-checks" (
        $batchText -match 'Split-Path -IsAbsolute \$answer' -and
        $batchText -match 'Test-Path -LiteralPath \$answer')

    # ---- Windows Store App Execution Aliases ----
    #
    # Windows ships python.exe and python3.exe in WindowsApps as zero-length reparse stubs that
    # satisfy Get-Command and Test-Path and OPEN THE MICROSOFT STORE when run. On a first install
    # with no CPython that is the first candidate on PATH, so probing it shows the user a Store
    # window and then burns the probe timeout waiting for output that never arrives.
    # Comments stripped first: the comment that explains the filter NAMES WindowsApps, so the
    # positive check passed against prose with the code deleted.
    $discoveryText = Get-FunctionTextOrEmpty $installPs1 "Get-StudioEarlyPython"
    $discoveryCode = ($discoveryText -split "`r?`n" |
        Where-Object { -not ($_.TrimStart().StartsWith("#")) }) -join "`n"
    Check "the comment stripper kept the code (bites)" ($discoveryCode -match 'Get-Command')
    Check "discovery skips the WindowsApps execution aliases" (
        $discoveryCode -match 'WindowsApps')
    foreach ($case in @(
        @{ P = "C:\Users\a\AppData\Local\Microsoft\WindowsApps\python3.exe"; Skip = $true },
        @{ P = "C:\Users\a\AppData\Local\Microsoft\WindowsApps\python.exe"; Skip = $true },
        @{ P = "C:/Users/a/AppData/Local/Microsoft/WindowsApps/python.exe"; Skip = $true },
        @{ P = "C:\Python312\python.exe"; Skip = $false },
        @{ P = "C:\Users\a\.unsloth\studio\.venv\Scripts\python.exe"; Skip = $false },
        @{ P = "/usr/bin/python3"; Skip = $false }
    )) {
        $hit = [bool]("$($case.P)" -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]')
        Check "the alias filter $(if ($case.Skip) { 'skips' } else { 'keeps' }) $($case.P)" ($hit -eq $case.Skip)
    }
    # ---- an elevated run will not launch an interpreter a standard user can replace ----
    #
    # Whatever this ladder settles on is launched WITH THE ADMINISTRATOR TOKEN when the install is
    # elevated. A per-user CPython, or any on PATH from a directory the same user's medium
    # integrity token can write, is then a way to have arbitrary code run elevated: replace the
    # executable and wait for the next install. The labelled child directory does not help, since
    # it protects the script being run and not the thing running it.
    # The roots come from the environment, so they are set here rather than assumed: a Linux
    # runner has no C: drive, and Join-Path on one throws rather than composing a string.
    $savedProgramFiles = $env:ProgramFiles
    $savedSystemRoot = $env:SystemRoot
    $env:ProgramFiles = "C:\Program Files"
    $env:SystemRoot = "C:\Windows"
    # The ACL half is driven through a stubbed Get-Acl, because the real one is Windows-only and
    # this file runs on every OS in the parity lane. The strings are the shapes Windows really
    # prints: System32's, %WINDIR%\Temp's, and one for a directory an attacker created.
    $script:AclTable = @{}
    $script:AclDefault = ("O:S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464G:SY" +
        "D:AI(A;;FA;;;S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464)" +
        "(A;OICIIO;GA;;;S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464)" +
        "(A;;0x1301bf;;;SY)(A;OICIIO;GA;;;SY)(A;;0x1301bf;;;BA)(A;OICIIO;GA;;;BA)" +
        "(A;;0x1200a9;;;BU)(A;OICIIO;GXGR;;;BU)(A;;0x1200a9;;;AC)(A;OICIIO;GXGR;;;AC)")
    $script:AclThrows = $false
    function Get-Acl {
        param($LiteralPath, $Path, $ErrorAction)
        if ($script:AclThrows) { throw "access denied" }
        $key = "$LiteralPath".TrimEnd('\', '/')
        $sddl = $script:AclDefault
        if ($script:AclTable.ContainsKey($key)) { $sddl = $script:AclTable[$key] }
        return [pscustomobject]@{ Sddl = $sddl }
    }
    # %WINDIR%\Temp as Windows really ships it: the two 0x1000xx ACEs are the Create Files/Write
    # Data and Create Folders/Append Data grants to BUILTIN\Users that AppLocker's own default
    # rule documentation calls out as the reason a path rule over %WINDIR% is a bypass.
    $tempSddl = ("O:BAG:SYD:PAI(A;;FA;;;SY)(A;OICIIO;GA;;;SY)(A;;FA;;;BA)(A;OICIIO;GA;;;BA)" +
        "(A;;0x100004;;;BU)(A;;0x100002;;;BU)(A;OICIIO;GA;;;CO)")
    $ownedSddl = "O:S-1-5-21-1-2-3-1001G:SYD:PAI(A;;FA;;;SY)(A;;FA;;;BA)"

    Check "the admin-root test accepts a system-wide location" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Python312\python.exe") -eq $true)
    # A directory any user can create, whose name merely starts with a protected one.
    Check "and is not fooled by a look-alike root" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files Evil\python.exe") -eq $false)
    Check "and rejects a per-user install" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Users\me\AppData\Local\Programs\Python\python.exe") -eq $false)
    Check "and rejects an empty path" ((Test-StudioPathUnderAdminRoot -Path "") -eq $false)

    # ---- being under a protected root is necessary, not sufficient ----
    #
    # %WINDIR%\Temp is below $env:SystemRoot and every standard user can create files in it, by
    # design and by Microsoft's documentation. The prefix-only version of this helper called it
    # administrator-protected and the elevated run would have launched it.
    Check "a location test alone would have accepted C:\Windows\Temp (bites)" (
        "C:\Windows\Temp\python.exe" -like "C:\Windows\*")
    Check "but the compatibility directories are refused outright" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Temp\python.exe") -eq $false)
    foreach ($leaf in @("Tasks", "Tracing", "System32\Tasks", "System32\spool\drivers\color",
                        "System32\com\dmp", "SysWOW64\FxsTmp")) {
        Check "and so is C:\Windows\$leaf" (
            (Test-StudioPathUnderAdminRoot -Path "C:\Windows\$leaf\python.exe") -eq $false)
    }
    # And the ACL is read as well, so a directory that is user-writable without being on that
    # list is refused too. This is the general case the list is only a cheap backstop for.
    $script:AclTable["C:\Windows\Sloppy"] = $tempSddl
    Check "a user-writable directory under a protected root is refused by its ACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Sloppy\python.exe") -eq $false)
    # A parent that grants Users write is enough on its own, and the leaf's own ACL reading as
    # administrator-only does not redeem it. That is not a hypothetical: write access to a
    # directory is permission to create a junction in it, and Get-Acl on a junction reports the
    # TARGET's descriptor, so a link the attacker can retarget at will reads here as System32.
    # This row leaves the leaf on the default System32 descriptor for exactly that reason, so it
    # can only be refused by walking up.
    Check "and so is a directory whose parent is user-writable, however tight its own ACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Sloppy\mine\python.exe") -eq $false)
    # Owner, separately: an attacker who owns a directory holds WRITE_DAC whatever its DACL says.
    $script:AclTable["C:\Program Files\Mine"] = $ownedSddl
    Check "a directory owned by a standard user is refused however tight its DACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Mine\python.exe") -eq $false)
    $script:AclTable.Remove("C:\Program Files\Mine")
    Check "control: the same path passes once its owner is administrators (bites)" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Mine\python.exe") -eq $true)
    # Unknown declines. An ACL that cannot be read is not evidence that it is a safe one.
    $script:AclThrows = $true
    Check "an unreadable ACL declines rather than trusting the location" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Python312\python.exe") -eq $false)
    $script:AclThrows = $false

    # The SDDL reader on its own, over the shapes above and the ones that must not parse.
    Check "System32's own descriptor is not user-writable" (
        (Test-StudioSddlWritableByNonAdmin -Sddl $script:AclDefault) -eq $false)
    Check "Temp's is" ((Test-StudioSddlWritableByNonAdmin -Sddl $tempSddl) -eq $true)
    Check "an owner that is a plain user is enough" (
        (Test-StudioSddlWritableByNonAdmin -Sddl $ownedSddl) -eq $true)
    Check "a descriptor this cannot parse is treated as writable" (
        (Test-StudioSddlWritableByNonAdmin -Sddl "not a descriptor") -eq $true)
    Check "and so is an empty one" ((Test-StudioSddlWritableByNonAdmin -Sddl "") -eq $true)
    # The owner really is read as BA and not as BAG: the owner and the group run together in the
    # string, and a class that stops at the next colon reads the owner as "BAG" and declines
    # every genuinely administrator-owned directory on the machine.
    Check "the owner is split from the group correctly" (
        (Test-StudioSddlWritableByNonAdmin -Sddl "O:BAG:SYD:PAI(A;;FA;;;BA)") -eq $false)
    Check "read and execute is not write" (
        (Test-StudioSddlRightsAreWrite -Rights "0x1200a9") -eq $false)
    Check "create files is" ((Test-StudioSddlRightsAreWrite -Rights "0x100002") -eq $true)
    Check "append data is" ((Test-StudioSddlRightsAreWrite -Rights "0x100004") -eq $true)
    Check "a full 32-bit mask does not overflow into a Double" (
        (Test-StudioSddlRightsAreWrite -Rights "0xffffffff") -eq $true)
    Check "the word forms are read too" ((Test-StudioSddlRightsAreWrite -Rights "FA") -eq $true)
    Check "and a read-only word form is not a write" (
        (Test-StudioSddlRightsAreWrite -Rights "FRFX") -eq $false)
    Check "an inherit-only CREATOR OWNER grant is not an effective one" (
        (Test-StudioSddlWritableByNonAdmin -Sddl "O:BAG:SYD:PAI(A;;FA;;;BA)(A;OICIIO;GA;;;CO)") -eq $false)
    Check "but the same grant without the inherit-only flag is" (
        (Test-StudioSddlWritableByNonAdmin -Sddl "O:BAG:SYD:PAI(A;;FA;;;BA)(A;OICI;GA;;;CO)") -eq $true)
    Check "a deny ACE is not a grant" (
        (Test-StudioSddlWritableByNonAdmin -Sddl "O:BAG:SYD:PAI(A;;FA;;;BA)(D;;FA;;;BU)") -eq $false)
    Remove-Item Function:Get-Acl -ErrorAction SilentlyContinue
    if ($null -eq $savedProgramFiles) { Remove-Item Env:ProgramFiles -ErrorAction SilentlyContinue } else { $env:ProgramFiles = $savedProgramFiles }
    if ($null -eq $savedSystemRoot) { Remove-Item Env:SystemRoot -ErrorAction SilentlyContinue } else { $env:SystemRoot = $savedSystemRoot }

    # Driven through the real discovery: elevated, with this host's own interpreter, which is not
    # under a Windows protected root, so the rung has to decline and say why.
    $savedOsElev = $env:OS
    try {
        $env:OS = "Windows_NT"
        function Test-StudioChildScriptDirectoryElevated { return $true }
        $script:StudioEarlyPythonProbed = $false
        $script:StudioEarlyPython = $null
        $script:StudioEarlyPythonProbedWithoutVenv = $false
        $script:StudioFinalPathWarned = $false
        $elevatedAnswer = Get-StudioEarlyPython
        Check "an elevated run declines a user-writable interpreter" (
            [string]::IsNullOrWhiteSpace($elevatedAnswer))
        # Bites control: the same call without elevation finds the interpreter this host has, so
        # the row above is about the rule and not about discovery being broken.
        function Test-StudioChildScriptDirectoryElevated { return $false }
        $script:StudioEarlyPythonProbed = $false
        $script:StudioEarlyPython = $null
        $script:StudioEarlyPythonProbedWithoutVenv = $false
        Check "control: an unelevated run still finds one" (
            -not [string]::IsNullOrWhiteSpace((Get-StudioEarlyPython)))
    } finally {
        Remove-Item Function:Test-StudioChildScriptDirectoryElevated -ErrorAction SilentlyContinue
        if ($null -eq $savedOsElev) { Remove-Item Env:OS -ErrorAction SilentlyContinue } else { $env:OS = $savedOsElev }
        $script:StudioEarlyPythonProbed = $false
        $script:StudioEarlyPython = $null
    }

} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All early-Python path resolver checks passed"
exit 0
