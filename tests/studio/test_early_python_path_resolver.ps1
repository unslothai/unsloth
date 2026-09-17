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
    "Remove-StudioTrailingNewline",
    "Invoke-StudioEarlyPythonScriptViaCmdlets", "Get-StudioPythonFinalPath",
    "New-StudioChildScriptDirectory",
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
    $extracted[$name] = $fn[0]
    Invoke-Expression $fn[0].Extent.Text
}


# The emitted rung that used to sit above this one is gone, so nothing needs forcing off: the
# interpreter IS the exact rung now, and the lexical answer is what sits below it.
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }

# The list above is hand-written, and install.ps1 moves under it: splitting a body out into a
# new helper leaves the helper unlisted, and the only symptom is "The term X is not recognized"
# raised from inside whichever check happens to call it first. Close the list over what the
# extracted bodies actually call. Asked of the session rather than of the list, so the stubs
# just above count as answers: what matters is that the name resolves when a check calls it.
$defined = @{}
foreach ($fn in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    $defined[$fn.Name] = $true
}
$missing = @()
foreach ($entry in $extracted.GetEnumerator()) {
    foreach ($call in $entry.Value.Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst] }, $true)) {
        $called = $call.GetCommandName()
        if ($called -and $defined.ContainsKey($called) -and
            -not (Get-Command -Name $called -ErrorAction SilentlyContinue)) {
            $missing += "$called (called by $($entry.Key))"
        }
    }
}
if ($missing.Count -gt 0) {
    throw ("install.ps1 functions these checks can reach but nothing here defines: " +
        (($missing | Sort-Object -Unique) -join ", ") +
        ". Extract them above, or stub them here.")
}

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
        # The control that makes the rest meaningful, and it is about Exact rather than about the
        # string. On PowerShell 7 Get-StudioLexicalPath already folds a symlink, because
        # Resolve-StudioLinkTarget can call ResolveLinkTarget there; on Windows PowerShell 5.1 it
        # sees only the raw reparse target, and it can never expand an 8.3 name or recover stored
        # casing on any version. What it can never do on ANY host is claim the answer is exact,
        # and Exact is what the install and runtime locks key on. So assert that: without this
        # rung the alias resolves inexactly, which is the state the rung exists to fix.
        $savedFinder = ${function:Get-StudioEarlyPython}
        function Get-StudioEarlyPython { return $null }
        $inexact = Resolve-StudioFinalPathInfo -Path $alias
        Check "without the rung the alias identity is inexact (bites)" ($inexact.Exact -eq $false)
        ${function:Get-StudioEarlyPython} = $savedFinder

        $viaAlias = Get-StudioPythonFinalPath -Path $alias
        $viaReal = Get-StudioPythonFinalPath -Path $real
        Check "Python folds the alias onto its target" (
            -not [string]::IsNullOrWhiteSpace($viaAlias) -and $viaAlias -eq $viaReal)

        # The integration check: the resolver as a whole now reports an EXACT identity for the
        # alias, which is what the install and runtime locks key on.
        $infoAlias = Resolve-StudioFinalPathInfo -Path $alias
        $infoReal = Resolve-StudioFinalPathInfo -Path $real
        Check "the resolver reports the alias as exact" ($infoAlias.Exact -eq $true)
        Check "the resolver gives one identity for both spellings" ($infoAlias.Path -eq $infoReal.Path)
    }

    # No interpreter means today's behaviour, unchanged: lexical answer, not exact. Verified by
    # replacing the finder rather than by reasoning about it, and restored afterwards: leaving it
    # stubbed made every later check silently exercise the disabled rung instead of the real one.
    $savedFinder2 = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { return $null }
    $script:StudioEarlyPythonProbed = $false
    $noPy = Resolve-StudioFinalPathInfo -Path $real
    Check "with no interpreter the identity is inexact, as before" ($noPy.Exact -eq $false)
    Check "with no interpreter a usable path still comes back" (
        -not [string]::IsNullOrWhiteSpace($noPy.Path))
    ${function:Get-StudioEarlyPython} = $savedFinder2
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    Check "the interpreter is back after the no-interpreter case" ($null -ne (Get-StudioEarlyPython))

    # Non-ASCII survives the child boundary. This string is hashed into a lock name that the
    # running Unsloth derives from _studio_runtime_gate.py's own resolve(), so a byte that does
    # not round-trip is not a cosmetic defect: the two sides compute different names for one
    # directory and neither excludes the other. Windows PowerShell 5.1 decodes a child's stdout
    # with the console codepage unless told otherwise, which is exactly how that happens.
    $unicodeName = "studio-ünïcôde-日本語-ß"
    $unicodeDir = Join-Path $tmp $unicodeName
    New-Item -ItemType Directory -Force -Path $unicodeDir | Out-Null
    $unicodeAnswer = Get-StudioPythonFinalPath -Path $unicodeDir
    Check "a non-ASCII path survives the child process" (
        -not [string]::IsNullOrWhiteSpace($unicodeAnswer) -and
        $unicodeAnswer.EndsWith($unicodeName))

    # The resolver must produce what the running Unsloth produces, since both are hashed into
    # lock names. Compare against _studio_runtime_gate.py's own expression rather than against
    # another spelling of it.
    $gateScript = "import pathlib,sys" + [char]10 +
        "sys.stdout.buffer.write(str(pathlib.Path(sys.argv[1]).resolve(strict=False)).encode('utf-8'))"
    $gateAnswer = & $exe -I -c $gateScript $unicodeDir
    Check "the answer matches the runtime gate's own resolve()" ($unicodeAnswer -eq "$gateAnswer".Trim())

    # A symlink loop must not be promoted to an exact identity. realpath is non-strict, so it
    # returns a best-effort string rather than raising, and promoting that to Exact would let a
    # caller treat an unresolved path as a vouched-for one.
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

    # A path with spaces must survive argument construction. On Windows PowerShell 5.1 there is no
    # ProcessStartInfo.ArgumentList, so the arguments go through a single quoted string and the
    # quoting has to be right. This exercises the result on whichever branch the host takes.
    $spacedDir = Join-Path $tmp "a dir with spaces"
    New-Item -ItemType Directory -Force -Path $spacedDir | Out-Null
    $spacedAnswer = Get-StudioPythonFinalPath -Path $spacedDir
    Check "a path containing spaces resolves" (
        -not [string]::IsNullOrWhiteSpace($spacedAnswer) -and $spacedAnswer.EndsWith("spaces"))

    # The 5.1 branch itself, asserted on the source: ArgumentList does not exist in .NET
    # Framework, so using it unconditionally makes every candidate fail on the one host that
    # matters most, and it fails invisibly because the finder just reports "no Python".
    $src = Get-Content -Raw -LiteralPath $installPs1
    Check "argument construction does not assume ArgumentList exists" (
        $src -match 'PSObject\.Properties\["ArgumentList"\]')
    # Likewise the version gate: before 3.8 Windows path resolution did not follow junctions, so
    # such an interpreter would return the alias spelling and this rung would call it exact.
    Check "the probe refuses an interpreter older than 3.8" (
        $src -match 'sys\.version_info\s*<\s*\(3,\s*8\)')

    # An interpreter that answers non-zero is rejected rather than trusted, which is the path the
    # version gate above takes on an old Python.
    $refuser = Join-Path $tmp $(if ($IsWindows -or $env:OS -eq "Windows_NT") { "refuse.cmd" } else { "refuse.sh" })
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "@echo off`r`nexit /b 2`r`n" | Set-Content -LiteralPath $refuser -Encoding ASCII
    } else {
        "#!/bin/sh`nexit 2`n" | Set-Content -LiteralPath $refuser
        & chmod +x $refuser
    }
    Check "an interpreter that exits non-zero is rejected" (
        $null -eq (Invoke-StudioEarlyPython -Exe $refuser -Path $real))

    # A hung interpreter must not hang the installer. The whole point of running this before the
    # install lock is that it cannot be allowed to wedge the run.
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

    # Finding an interpreter must not write anything: this runs before the install lock, so a
    # mutation here would be a mutation before exclusion is established.
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

    # site stays out of the probe, and the reason is measured rather than quoted.
    #
    # Isolated mode implies -E, -P and -s, but NOT -S, so site is still imported and a
    # system-level sitecustomize still runs. On a corporate host that is instrumentation: it can
    # print to stdout and corrupt the single line this rung reads back, it can hang and burn the
    # timeout on a good interpreter, and it can patch pathlib, which would let an influenced
    # answer be marked exact. The first check asks the real interpreter under test, so a Python
    # that ever does imply -S would show up here as a stale justification rather than pass.
    $exe = Get-StudioEarlyPython
    $isoOnly = (& $exe -I -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    $isoPlus = (& $exe -I -S -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    Check "-I alone leaves site imported on this interpreter" ("$isoOnly".Trim() -eq "0")
    Check "-S is what turns site off" ("$isoPlus".Trim() -eq "1")

    # And every launcher passes it. Read out of the source, because the end-to-end drive is not
    # available here: sitecustomize is resolved on sys.path and the stdlib directory precedes
    # site-packages, so a planted copy is shadowed by the host's own on any machine that has one.
    #
    # Written against every argument vector rather than one named function, because which
    # function builds it moves: the resolver assembles its own here and delegates to a shared
    # runner once that exists, and the runner has a second cmdlet-only launcher beside it for
    # Constrained Language Mode. A check naming one of them passes while another drops the flag.
    $vectors = [regex]::Matches((Get-Content -Raw -LiteralPath $installPs1), '@\(\s*"-I"[^)]*\)')
    Check "at least one launcher argument vector was found (bites)" ($vectors.Count -ge 1)
    foreach ($v in $vectors) {
        Check "the launcher at offset $($v.Index) runs with -S as well as -I" (
            $v.Value -match '"-I",\s*"-S"')
    }

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
    # Comments stripped first, both ways round: the comment that explains the filter NAMES
    # WindowsApps, so the positive check passed against prose with the code deleted, and the
    # comment that explains the absence of a length read names .Length.
    $discoveryText = Get-FunctionTextOrEmpty $installPs1 "Get-StudioEarlyPython"
    $discoveryCode = ($discoveryText -split "`r?`n" |
        Where-Object { -not ($_.TrimStart().StartsWith("#")) }) -join "`n"
    Check "the comment stripper kept the code (bites)" ($discoveryCode -match 'Get-Command')
    Check "discovery skips the WindowsApps execution aliases" (
        $discoveryCode -match 'WindowsApps')
    # By location, not by reading a file's length: that is a property access on a FileInfo, which
    # Constrained Language Mode refuses, and this ladder exists for hosts under CLM.
    Check "and does not read a property CLM refuses to get there" (
        $discoveryCode -notmatch '\.Length\b')
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
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All early-Python path resolver checks passed"
exit 0
