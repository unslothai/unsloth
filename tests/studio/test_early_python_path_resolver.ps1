#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Run: pwsh -NoProfile -File tests/studio/test_early_python_path_resolver.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = Join-Path $root "install.ps1"
$onWindows = $IsWindows -or $env:OS -eq "Windows_NT"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }
function Get-Fn($name) {
    $f = @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true))
    if ($f.Count -lt 1) { return "" }
    return $f[0].Extent.Text
}
foreach ($name in @(
    "Get-StudioEarlyPython", "Invoke-StudioEarlyPython", "Invoke-StudioEarlyPythonScript",
    "Get-StudioPythonFinalPath", "New-StudioChildScriptDirectory", "Test-StudioChildScriptDirectoryElevated",
    "Get-StudioSystem32Tool", "Test-StudioPathUnderAdminRoot",
    "Test-StudioSddlRightsAreWrite", "Test-StudioSddlPrincipalIsAdminOnly",
    "Test-StudioSddlWritableByNonAdmin", "Test-StudioDirectoryIsAdminOnly",
    "Get-StudioLexicalParent", "Test-StudioInterpreterFileIsAdminOnly",
    "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo", "Resolve-StudioFinalPathsInOneChild", "Write-StudioFinalPathDegraded"
)) {
    $text = Get-Fn $name
    if (-not $text) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $text
}

if ($env:OS -eq "Windows_NT") { function Test-StudioChildScriptDirectoryElevated { return $false } }
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }

function Reset-EarlyPython {
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $script:StudioEarlyPythonProbedWithoutVenv = $false
}
function Save-Env([string[]]$names) {
    $h = @{}; foreach ($n in $names) { $h[$n] = [Environment]::GetEnvironmentVariable($n) }; return $h
}
function Restore-Env($saved) { foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) } }
function New-FakeExe($stem, $cmdBody, $shBody) {
    $p = Join-Path $tmp ($stem + $(if ($onWindows) { ".cmd" } else { ".sh" }))
    if ($onWindows) { "@echo off`r`n$cmdBody`r`n" | Set-Content -LiteralPath $p -Encoding ASCII }
    else { "#!/bin/sh`n$shBody`n" | Set-Content -LiteralPath $p; & chmod +x $p }
    return $p
}
# Returns the SDDL from $script:AclTable, else $script:AclDefault; throws while $script:AclThrows.
$aclStub = {
    param($LiteralPath, $Path, $ErrorAction)
    if ($script:AclThrows) { throw "access denied" }
    $key = "$LiteralPath".TrimEnd('\', '/')
    $sddl = $script:AclDefault
    if ($script:AclTable.ContainsKey($key)) { $sddl = $script:AclTable[$key] }
    return [pscustomobject]@{ Sddl = $sddl }
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("earlypy-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    Remove-Item Env:UNSLOTH_EARLY_PYTHON_PROBE -ErrorAction SilentlyContinue
    Reset-EarlyPython
    $exe = Get-StudioEarlyPython
    if (-not $exe) {
        # "No interpreter" is a supported state and a skip, but it is also what a BROKEN EXTRACTION
        # looks like (a forgotten helper makes discovery find nothing and CI records a pass). If this
        # host has a python that answers the same probe, the extraction is at fault, not the host.
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
                # The installer's elevated rule: a candidate a standard user can replace is not one to find.
                if ($elevatedHost -and -not (Test-StudioPathUnderAdminRoot -Path $src)) { continue }
                $here = Split-Path -Parent $src
                if ([string]::IsNullOrWhiteSpace($here)) { continue }
                # A job with a deadline: a shim that never exits must be a skip, not a hang.
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
    Write-Host "  interpreter: $exe"

    $real = Join-Path $tmp "real"
    New-Item -ItemType Directory -Force -Path $real | Out-Null
    $alias = Join-Path $tmp "alias"
    $madeAlias = $false
    try {
        New-Item -ItemType $(if ($onWindows) { "Junction" } else { "SymbolicLink" }) -Path $alias -Target $real -ErrorAction Stop | Out-Null
        $madeAlias = $true
    } catch {
        Write-Host "  SKIP  could not create a directory alias: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    if ($madeAlias) {
        # Control: the lexical rung may fold a symlink on pwsh 7, but it is never exact.
        $savedFinder = ${function:Get-StudioEarlyPython}
        function Get-StudioEarlyPython { return $null }
        Check "without the rung the alias identity is inexact (bites)" ((Resolve-StudioFinalPathInfo -Path $alias).Exact -eq $false)
        ${function:Get-StudioEarlyPython} = $savedFinder

        $viaAlias = Get-StudioPythonFinalPath -Path $alias
        Check "Python folds the alias onto its target" (
            -not [string]::IsNullOrWhiteSpace($viaAlias) -and $viaAlias -eq (Get-StudioPythonFinalPath -Path $real))
        $infoAlias = Resolve-StudioFinalPathInfo -Path $alias
        Check "the resolver reports the alias as exact" ($infoAlias.Exact -eq $true)
        Check "the resolver gives one identity for both spellings" ($infoAlias.Path -eq (Resolve-StudioFinalPathInfo -Path $real).Path)
    }

    $savedFinder = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { return $null }
    $script:StudioEarlyPythonProbed = $false
    $script:StudioPythonFinalPathCache = $null
    $noPy = Resolve-StudioFinalPathInfo -Path $real
    Check "with no interpreter the identity is inexact, as before" ($noPy.Exact -eq $false)
    Check "with no interpreter a usable path still comes back" (-not [string]::IsNullOrWhiteSpace($noPy.Path))
    ${function:Get-StudioEarlyPython} = $savedFinder
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    Check "the interpreter is back after the no-interpreter case" ($null -ne (Get-StudioEarlyPython))

    # The answer is hashed into a lock name, so non-ASCII must survive 5.1's console codepage.
    $unicodeName = "studio-ünïcôde-日本語-ß"
    $unicodeDir = Join-Path $tmp $unicodeName
    New-Item -ItemType Directory -Force -Path $unicodeDir | Out-Null
    $unicodeAnswer = Get-StudioPythonFinalPath -Path $unicodeDir
    Check "a non-ASCII path survives the child process" (
        -not [string]::IsNullOrWhiteSpace($unicodeAnswer) -and $unicodeAnswer.EndsWith($unicodeName))

    # Trim() would drop the U+00A0 and name the sibling instead.
    $nbspName = "studio-nbsp" + [char]0x00A0
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp "studio-nbsp") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp $nbspName) | Out-Null
    # .NET Framework strips it on the way in, so 5.1 cannot create or name such a directory at all.
    if (@(Get-ChildItem -LiteralPath $tmp -Name) -ccontains $nbspName) {
        Check "a trailing non-breaking space is kept" ("$(Get-StudioPythonFinalPath -Path (Join-Path $tmp $nbspName))".EndsWith($nbspName))
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
        Check "a link loop is not promoted to an exact identity" ([string]::IsNullOrWhiteSpace((Get-StudioPythonFinalPath -Path $loopA)))
        # Test-Path reports the link itself, so the walk stops AT the link.
        Check "a missing path under a link loop is not exact" ((Resolve-StudioFinalPathInfo -Path (Join-Path $loopA "studio")).Exact -eq $false)
    }

    $dangling = Join-Path $tmp "dangling"
    $gone = Join-Path $tmp "gone"
    $danglingOk = $false
    try {
        New-Item -ItemType SymbolicLink -Path $dangling -Target $gone -ErrorAction Stop | Out-Null
        $danglingOk = $true
    } catch {
        # 5.1 refuses a link to a missing target: link first, then remove the target.
        try {
            New-Item -ItemType Directory -Force -Path $gone | Out-Null
            New-Item -ItemType SymbolicLink -Path $dangling -Target $gone -ErrorAction Stop | Out-Null
            Remove-Item -LiteralPath $gone -Recurse -Force
            $danglingOk = $true
        } catch {}
    }
    if ($danglingOk) {
        $checkDangling = {
            param($label)
            foreach ($suffix in @("", "studio", "a\b")) {
                $probePath = if ($suffix) { Join-Path $dangling $suffix } else { $dangling }
                Check "$label (suffix '$suffix')" ((Resolve-StudioFinalPathInfo -Path $probePath).Exact -eq $false)
            }
        }
        & $checkDangling "a dangling link is not exact"
        # Where Test-Path follows the link, the stripped entry's reparse bit alone must keep it inexact.
        function Test-Path {
            param([string]$LiteralPath)
            if ($LiteralPath.StartsWith($dangling)) { return $false }
            return (Microsoft.PowerShell.Management\Test-Path -LiteralPath $LiteralPath)
        }
        try { & $checkDangling "a dangling link Test-Path reports missing is not exact" }
        finally { Remove-Item Function:Test-Path }
    }

    # 5.1 has no ArgumentList, so arguments go through one quoted string.
    $spacedDir = Join-Path $tmp "a dir with spaces"
    New-Item -ItemType Directory -Force -Path $spacedDir | Out-Null
    $spacedAnswer = Get-StudioPythonFinalPath -Path $spacedDir
    Check "a path containing spaces resolves" (
        -not [string]::IsNullOrWhiteSpace($spacedAnswer) -and $spacedAnswer.EndsWith("spaces"))

    $src = Get-Content -Raw -LiteralPath $installPs1
    Check "argument construction does not assume ArgumentList exists" ($src -match 'PSObject\.Properties\["ArgumentList"\]')
    Check "the probe refuses an interpreter older than 3.8" ($src -match 'sys\.version_info\s*<\s*\(3,\s*8\)')

    $refuser = New-FakeExe "refuse" "exit /b 2" "exit 2"
    Check "an interpreter that exits non-zero is rejected" ($null -eq (Invoke-StudioEarlyPython -Exe $refuser -Path $real))

    $slow = New-FakeExe "slow" "ping -n 30 127.0.0.1 >nul" "sleep 30"
    $started = [DateTime]::UtcNow
    $hung = Invoke-StudioEarlyPython -Exe $slow -Path $real -TimeoutMs 2000
    $elapsed = ([DateTime]::UtcNow - $started).TotalSeconds
    Check "a hung interpreter returns null" ($null -eq $hung)
    Check "a hung interpreter is killed at the deadline, not waited out" ($elapsed -lt 15)

    # The interpreter exits at once but leaves a child holding stdout: the read is bounded too.
    $holder = New-FakeExe "holder" "start /b `"`" ping -n 20 127.0.0.1`r`nexit /b 0" "sleep 20 &`nprintf '%s' `"`$5`"`nexit 0"
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
    Invoke-Expression (Get-Fn "Get-StudioEarlyPython")
    $null = Get-StudioEarlyPython
    Check "finding an interpreter writes nothing" (@(Get-ChildItem -LiteralPath $tmp -Recurse -Force).Count -eq $before)

    # -I alone still imports site, so a sitecustomize could print into the answer or hang.
    $exe = Get-StudioEarlyPython
    $isoOnly = (& $exe -I -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    $isoPlus = (& $exe -I -S -c "import sys;print(sys.flags.no_site)" 2>$null | Select-Object -First 1)
    Check "-I alone leaves site imported on this interpreter" ("$isoOnly".Trim() -eq "0")
    Check "-S is what turns site off" ("$isoPlus".Trim() -eq "1")

    # Read from the source: a planted sitecustomize is shadowed by the host's own.
    $launcherFn = Get-Fn "Invoke-StudioEarlyPythonScript"
    Check "the launcher runs every probe with -S as well as -I" ($launcherFn -match '@\("-I",\s*"-S",\s*"-B",\s*"-c"')
    Check "and with -B, so a pre-lock probe writes no bytecode" ($launcherFn -match '"-B"')

    # A miss recorded before $VenvDir existed (the --tauri path) is probed again once it does.
    $venvHome = Join-Path $tmp "venvhome"
    $venvBin = if ($onWindows) { "Scripts" } else { "bin" }
    $venvLeaf = if ($onWindows) { "python.exe" } else { "python3" }
    New-Item -ItemType Directory -Force -Path (Join-Path $venvHome $venvBin) | Out-Null
    Copy-Item -LiteralPath $exe -Destination (Join-Path $venvHome (Join-Path $venvBin $venvLeaf)) -Force
    function Get-Command { param($Name, [switch]$All, $CommandType, $ErrorAction) return @() }
    Remove-Variable -Name VenvDir -Scope Script -ErrorAction SilentlyContinue
    Remove-Variable -Name VenvDir -Scope Global -ErrorAction SilentlyContinue
    Reset-EarlyPython
    Check "with no venv and no system Python the probe finds nothing" ($null -eq (Get-StudioEarlyPython))
    Check "and it recorded that the venv was unknown when it looked" ($script:StudioEarlyPythonProbedWithoutVenv -eq $true)
    $global:VenvDir = $venvHome
    $found = Get-StudioEarlyPython
    Check "once the venv directory is known the venv interpreter is found after all" (-not [string]::IsNullOrWhiteSpace($found))
    Check "and it is the one inside the venv, not some other copy" ("$found" -like ("*" + $venvBin + "*"))
    $script:ReprobeCount = 0
    function Test-Path { param($LiteralPath, $PathType, $ErrorAction) $script:ReprobeCount++; return $false }
    $null = Get-StudioEarlyPython
    Check "a hit is not probed again" ($script:ReprobeCount -eq 0)
    # An uninspectable candidate (an unreadable directory throws under Stop) is skipped, and nothing
    # escapes into the lock-name hash. The cache is saved and restored around these throw tests.
    $savedEarly = @($script:StudioEarlyPython, $script:StudioEarlyPythonProbed, $script:StudioEarlyPythonProbedWithoutVenv)
    function Test-Path { param($LiteralPath, $PathType, $ErrorAction)
        throw [System.UnauthorizedAccessException]::new("Access to the path '$LiteralPath' is denied.") }
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $threw = $null
    try { $none = Get-StudioEarlyPython } catch { $threw = $_ }
    Check "a candidate that cannot be inspected is skipped, not fatal" ($null -eq $threw -and $null -eq $none)
    $savedFinder = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { throw "discovery failed" }
    $script:StudioPythonFinalPathCache = $null
    $threw = $null
    try { $none = Get-StudioPythonFinalPath -Path $real } catch { $threw = $_ }
    Check "a discovery failure declines to the lexical rung instead of throwing" ($null -eq $threw -and $null -eq $none)
    ${function:Get-StudioEarlyPython} = $savedFinder
    Remove-Item Function:Test-Path -ErrorAction SilentlyContinue
    $script:StudioEarlyPython, $script:StudioEarlyPythonProbed, $script:StudioEarlyPythonProbedWithoutVenv = $savedEarly
    Remove-Item Function:Get-Command -ErrorAction SilentlyContinue
    Remove-Variable -Name VenvDir -Scope Global -ErrorAction SilentlyContinue

    $script:ResolveCalls = 0
    $savedInvoke = ${function:Invoke-StudioEarlyPython}
    function Invoke-StudioEarlyPython { param($Exe, $Path, $TimeoutMs) $script:ResolveCalls++; return $Path }
    $savedFinder = ${function:Get-StudioEarlyPython}
    function Get-StudioEarlyPython { return "python3" }
    $script:StudioPythonFinalPathCache = $null
    1..3 | ForEach-Object { $null = Get-StudioPythonFinalPath -Path $real }
    Check "three calls for one path spawn one child" ($script:ResolveCalls -eq 1)
    $null = Get-StudioPythonFinalPath -Path $tmp
    Check "and a different path still spawns its own" ($script:ResolveCalls -eq 2)
    function Invoke-StudioEarlyPython { param($Exe, $Path, $TimeoutMs) $script:ResolveCalls++; return $null }
    $missPath = Join-Path $tmp "no-such-thing"
    1..2 | ForEach-Object { $null = Get-StudioPythonFinalPath -Path $missPath }
    Check "a miss is remembered as well as an answer" ($script:ResolveCalls -eq 3)
    ${function:Invoke-StudioEarlyPython} = $savedInvoke
    ${function:Get-StudioEarlyPython} = $savedFinder

    $script:StudioPythonFinalPathCache = $null
    $batchDir = Join-Path $tmp "batch"
    $null = New-Item -ItemType Directory -Path $batchDir -Force
    $batchPaths = @(foreach ($i in 1..12) {
        $leaf = Join-Path $batchDir "f$i.txt"
        Set-Content -LiteralPath $leaf -Value "x" -Encoding utf8
        $leaf
    })

    $script:RealInvoke = ${function:Invoke-StudioEarlyPython}
    function Invoke-StudioEarlyPython {
        param([string]$Exe, [string]$Path, [int]$TimeoutMs = 10000)
        $script:SingleCalls++
        return (& $script:RealInvoke -Exe $Exe -Path $Path -TimeoutMs $TimeoutMs)
    }
    $script:SingleCalls = 0
    foreach ($leaf in $batchPaths) { $null = Get-StudioPythonFinalPath -Path $leaf }
    Check "unbatched, every resolution starts its own child (bites)" ($script:SingleCalls -eq $batchPaths.Count)

    $script:StudioPythonFinalPathCache = $null
    $script:SingleCalls = 0
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    Check "the batch starts no single-path child of its own" ($script:SingleCalls -eq 0)
    foreach ($leaf in $batchPaths) { $null = Get-StudioPythonFinalPath -Path $leaf }
    Check "and every resolution after it is served without one" ($script:SingleCalls -eq 0)

    # The batch must return exactly what the single-path rung returns, or it is a second identity.
    $script:StudioPythonFinalPathCache = $null
    $one = Get-StudioPythonFinalPath -Path $batchPaths[0]
    $script:StudioPythonFinalPathCache = $null
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    $batched = Get-StudioPythonFinalPath -Path $batchPaths[0]
    Check "the batched answer is a real path (bites)" (-not [string]::IsNullOrWhiteSpace($one))
    Check "the batch answers byte for byte what the single rung answers" ($batched -ceq $one)

    $script:StudioPythonFinalPathCache = $null
    $missing = Join-Path $batchDir "no-such-file"
    Resolve-StudioFinalPathsInOneChild -Paths @($missing)
    Check "an unresolvable path is recorded" ($script:StudioPythonFinalPathCache.ContainsKey($missing))
    Check "and recorded as no exact answer" ($null -eq $script:StudioPythonFinalPathCache[$missing])
    $script:SingleCalls = 0
    $null = Get-StudioPythonFinalPath -Path $missing
    Check "and is not re-asked" ($script:SingleCalls -eq 0)

    # A child that answers nothing, or only SOME paths, leaves the rest uncached: absence is not a miss.
    $script:RealScript = ${function:Invoke-StudioEarlyPythonScript}
    function Invoke-StudioEarlyPythonScript {
        param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
        return ($script:PartialAnswer)
    }
    $script:PartialAnswer = ""
    $script:StudioPythonFinalPathCache = $null
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    $cachedAfterFailure = 0
    if ($null -ne $script:StudioPythonFinalPathCache) { $cachedAfterFailure = $script:StudioPythonFinalPathCache.Count }
    Check "a silent child caches nothing at all" ($cachedAfterFailure -eq 0)
    ${function:Invoke-StudioEarlyPythonScript} = $script:RealScript
    $script:SingleCalls = 0
    $null = Get-StudioPythonFinalPath -Path $batchPaths[0]
    Check "so the single-path rung still gets its turn" ($script:SingleCalls -eq 1)

    $script:StudioPythonFinalPathCache = $null
    function Invoke-StudioEarlyPythonScript {
        param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
        return ($script:PartialAnswer)
    }
    $script:PartialAnswer = "$($batchPaths[0])|$($batchPaths[0])"
    Resolve-StudioFinalPathsInOneChild -Paths $batchPaths
    Check "a partial answer caches only what was answered" (
        $script:StudioPythonFinalPathCache.Count -eq 1 -and $script:StudioPythonFinalPathCache.ContainsKey($batchPaths[0]))
    Check "and leaves the unanswered paths for the single rung" (-not $script:StudioPythonFinalPathCache.ContainsKey($batchPaths[1]))
    ${function:Invoke-StudioEarlyPythonScript} = $script:RealScript

    $batchText = Get-Fn "Resolve-StudioFinalPathsInOneChild"
    Check "the reader strips a byte order mark" ($batchText -match "encoding='utf-8-sig'")
    Check "and does not read the list as plain utf-8" ($batchText -notmatch "encoding='utf-8'\)")
    # Run it: the same expression against a real BOM-prefixed file must resolve every line, and the
    # plain utf-8 control must LOSE the first one, or the check above is only spelling.
    $bomFile = Join-Path $batchDir "with-bom.txt"
    [System.IO.File]::WriteAllBytes($bomFile, [byte[]](0xEF, 0xBB, 0xBF) + [System.Text.Encoding]::UTF8.GetBytes(($batchPaths -join "`n")))
    $bomScript = @(
        "import pathlib,sys", "n=0",
        "with open(sys.argv[1],'r',encoding='utf-8-sig') as fh:",
        "    for line in fh:",
        "        p=line.rstrip('\r\n')",
        "        if not p: continue",
        "        try:",
        "            pathlib.Path(p).resolve(strict=True)",
        "            n+=1",
        "        except Exception:",
        "            pass",
        "sys.stdout.buffer.write(str(n).encode('utf-8'))"
    ) -join [char]10
    foreach ($row in @(
        @("a BOM-prefixed list resolves every line, first one included", $bomScript, $batchPaths.Count),
        @("and plain utf-8 really does lose it (bites)", ($bomScript -replace "utf-8-sig", "utf-8"), ($batchPaths.Count - 1))
    )) {
        $count = Invoke-StudioEarlyPythonScript -Exe $exe -Script $row[1] -ScriptArgs @($bomFile) -TimeoutMs 30000
        Check $row[0] ("$count".Trim() -eq "$($row[2])")
    }

    # The list goes through a file, not argv: a few hundred paths pass the 32767 character limit.
    Check "the batch hands the child a list FILE rather than an argv of paths" (
        $batchText -match 'Set-Content -LiteralPath \$listFile' -and $batchText -match 'ScriptArgs @\(\$listFile\)')
    Check "the batch resolves with the same expression the single rung uses" ($batchText -match 'pathlib\.Path\(p\)\.resolve\(strict=True\)')
    Check "and keeps the same version gate" ($batchText -match 'sys\.version_info < \(3,8\)')
    Check "and applies the same two post-checks" (
        $batchText -match 'Split-Path -IsAbsolute \$answer' -and $batchText -match 'Test-Path -LiteralPath \$answer')

    $discoveryCode = ((Get-Fn "Get-StudioEarlyPython") -split "`r?`n" |
        Where-Object { -not ($_.TrimStart().StartsWith("#")) }) -join "`n"
    Check "the comment stripper kept the code (bites)" ($discoveryCode -match 'Get-Command')
    Check "discovery skips the WindowsApps execution aliases" ($discoveryCode -match 'WindowsApps')
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

    $savedWinEnv = Save-Env @("ProgramFiles", "SystemRoot")
    $env:ProgramFiles = "C:\Program Files"
    $env:SystemRoot = "C:\Windows"
    $script:AclTable = @{}
    $script:AclDefault = ("O:S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464G:SY" +
        "D:AI(A;;FA;;;S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464)" +
        "(A;OICIIO;GA;;;S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464)" +
        "(A;;0x1301bf;;;SY)(A;OICIIO;GA;;;SY)(A;;0x1301bf;;;BA)(A;OICIIO;GA;;;BA)" +
        "(A;;0x1200a9;;;BU)(A;OICIIO;GXGR;;;BU)(A;;0x1200a9;;;AC)(A;OICIIO;GXGR;;;AC)")
    $script:AclThrows = $false
    ${function:Get-Acl} = $aclStub
    $tempSddl = ("O:BAG:SYD:PAI(A;;FA;;;SY)(A;OICIIO;GA;;;SY)(A;;FA;;;BA)(A;OICIIO;GA;;;BA)" +
        "(A;;0x100004;;;BU)(A;;0x100002;;;BU)(A;OICIIO;GA;;;CO)")
    $ownedSddl = "O:S-1-5-21-1-2-3-1001G:SYD:PAI(A;;FA;;;SY)(A;;FA;;;BA)"
    $adminRootRows = @(
        @("the admin-root test accepts a system-wide location", "C:\Program Files\Python312\python.exe", $true),
        @("and is not fooled by a look-alike root", "C:\Program Files Evil\python.exe", $false),
        @("and rejects a per-user install", "C:\Users\me\AppData\Local\Programs\Python\python.exe", $false),
        @("and rejects an empty path", "", $false)
    )
    foreach ($r in $adminRootRows) { Check $r[0] ((Test-StudioPathUnderAdminRoot -Path $r[1]) -eq $r[2]) }

    Check "a location test alone would have accepted C:\Windows\Temp (bites)" ("C:\Windows\Temp\python.exe" -like "C:\Windows\*")
    Check "but the compatibility directories are refused outright" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Temp\python.exe") -eq $false)
    foreach ($leaf in @("Tasks", "Tracing", "System32\Tasks", "System32\spool\drivers\color",
                        "System32\com\dmp", "SysWOW64\FxsTmp")) {
        Check "and so is C:\Windows\$leaf" ((Test-StudioPathUnderAdminRoot -Path "C:\Windows\$leaf\python.exe") -eq $false)
    }
    $script:AclTable["C:\Windows\Sloppy"] = $tempSddl
    Check "a user-writable directory under a protected root is refused by its ACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Sloppy\python.exe") -eq $false)
    Check "and so is a directory whose parent is user-writable, however tight its own ACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Windows\Sloppy\mine\python.exe") -eq $false)
    $script:AclTable["C:\Program Files\Mine"] = $ownedSddl
    Check "a directory owned by a standard user is refused however tight its DACL" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Mine\python.exe") -eq $false)
    $script:AclTable.Remove("C:\Program Files\Mine")
    Check "control: the same path passes once its owner is administrators (bites)" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Mine\python.exe") -eq $true)
    # An ACL that cannot be read is not evidence that it is a safe one.
    $script:AclThrows = $true
    Check "an unreadable ACL declines rather than trusting the location" (
        (Test-StudioPathUnderAdminRoot -Path "C:\Program Files\Python312\python.exe") -eq $false)
    $script:AclThrows = $false

    foreach ($r in @(
        @("System32's own descriptor is not user-writable", $script:AclDefault, $false),
        @("Temp's is", $tempSddl, $true),
        @("an owner that is a plain user is enough", $ownedSddl, $true),
        @("a descriptor this cannot parse is treated as writable", "not a descriptor", $true),
        @("and so is an empty one", "", $true),
        @("the owner is split from the group correctly", "O:BAG:SYD:PAI(A;;FA;;;BA)", $false),
        @("an inherit-only CREATOR OWNER grant is not an effective one", "O:BAG:SYD:PAI(A;;FA;;;BA)(A;OICIIO;GA;;;CO)", $false),
        @("but the same grant without the inherit-only flag is", "O:BAG:SYD:PAI(A;;FA;;;BA)(A;OICI;GA;;;CO)", $true),
        @("a deny ACE is not a grant", "O:BAG:SYD:PAI(A;;FA;;;BA)(D;;FA;;;BU)", $false)
    )) { Check $r[0] ((Test-StudioSddlWritableByNonAdmin -Sddl $r[1]) -eq $r[2]) }
    foreach ($r in @(
        @("read and execute is not write", "0x1200a9", $false),
        @("create files is", "0x100002", $true),
        @("append data is", "0x100004", $true),
        @("a full 32-bit mask does not overflow into a Double", "0xffffffff", $true),
        @("the word forms are read too", "FA", $true),
        @("and a read-only word form is not a write", "FRFX", $false)
    )) { Check $r[0] ((Test-StudioSddlRightsAreWrite -Rights $r[1]) -eq $r[2]) }
    Remove-Item Function:Get-Acl -ErrorAction SilentlyContinue
    Restore-Env $savedWinEnv

    # Driven through the real discovery, elevated, with this host's interpreter, which is not under a
    # Windows protected root, so the rung has to decline.
    $savedOsElev = Save-Env @("OS")
    try {
        $env:OS = "Windows_NT"
        function Test-StudioChildScriptDirectoryElevated { return $true }
        Reset-EarlyPython
        $script:StudioFinalPathWarned = $false
        Check "an elevated run declines a user-writable interpreter" ([string]::IsNullOrWhiteSpace((Get-StudioEarlyPython)))
        function Test-StudioChildScriptDirectoryElevated { return $false }
        Reset-EarlyPython
        Check "control: an unelevated run still finds one" (-not [string]::IsNullOrWhiteSpace((Get-StudioEarlyPython)))
        $hostPy = (Get-Command python3 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if ($savedOsElev["OS"] -ne "Windows_NT" -and $hostPy) {
            $venvRoot = Join-Path $tmp "elevated-venv"
            New-Item -ItemType Directory -Force -Path (Join-Path $venvRoot "bin") | Out-Null
            $venvPy = Join-Path $venvRoot "bin/python3"
            New-Item -ItemType SymbolicLink -Path $venvPy -Target $hostPy | Out-Null
            function Test-StudioChildScriptDirectoryElevated { return $true }
            $VenvDir = $venvRoot
            Reset-EarlyPython
            Check "an elevated run still uses the existing install's venv interpreter" ((Get-StudioEarlyPython) -eq $venvPy)
            Remove-Variable -Name VenvDir -ErrorAction SilentlyContinue
        }
    } finally {
        function Test-StudioChildScriptDirectoryElevated { return $false }
        Restore-Env $savedOsElev
        $script:StudioEarlyPythonProbed = $false
        $script:StudioEarlyPython = $null
    }

    $hostPy3 = (Get-Command python3 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
    if ($onWindows -or -not $hostPy3) {
        Write-Host "  SKIP  interpreter-file checks need a non-Windows host with python3" -ForegroundColor Yellow
    } else {
        $savedEnv2 = Save-Env @("ProgramFiles", "SystemRoot", "OS", "PATH")
        $pfRoot = Join-Path $tmp "pf"
        $pfBin = Join-Path $pfRoot "bin"
        New-Item -ItemType Directory -Force -Path $pfBin | Out-Null
        $wrapper = Join-Path $pfBin "python3"
        Set-Content -LiteralPath $wrapper -Value ("#!/bin/sh`nexec '" + $hostPy3 + "' `"`$@`"`n") -NoNewline
        & chmod +x $wrapper
        $script:AclTable = @{}
        $script:AclThrows = $false
        ${function:Get-Acl} = $aclStub
        $reprobe = { Reset-EarlyPython; return (Get-StudioEarlyPython) }
        $fileOk = { (Test-StudioInterpreterFileIsAdminOnly -Path $wrapper) }
        $walkOk = { (Test-StudioPathUnderAdminRoot -Path $wrapper) }
        try {
            $env:ProgramFiles = $pfRoot
            $env:SystemRoot = Join-Path $tmp "win"
            $env:OS = "Windows_NT"
            $env:PATH = $pfBin + [System.IO.Path]::PathSeparator + $savedEnv2["PATH"]
            function Test-StudioChildScriptDirectoryElevated { return $true }

            Check "the file helper accepts an administrator-only regular file" ((& $fileOk) -eq $true)
            Check "control: the directory walk alone accepts this location" ((& $walkOk) -eq $true)
            Check "control: an elevated run accepts an administrator-only file under an administrator-only root" (
                (& $reprobe) -eq $wrapper)

            $script:AclTable[$wrapper] = "O:BAG:SYD:PAI(A;;FA;;;SY)(A;;FA;;;BA)(A;;0x1301bf;;;BU)"
            Check "the directory walk still accepts it, so it cannot see the file's ACL (bites)" ((& $walkOk) -eq $true)
            Check "the file helper refuses a user-writable file under admin-only directories" ((& $fileOk) -eq $false)
            Check "an elevated run declines a PATH interpreter whose own DACL a standard user can write" (
                [string]::IsNullOrWhiteSpace((& $reprobe)))

            $script:AclTable[$wrapper] = "O:S-1-5-21-1-2-3-1001G:SYD:PAI(A;;FA;;;SY)(A;;FA;;;BA)"
            Check "the file helper refuses a file owned by a standard user" ((& $fileOk) -eq $false)
            $script:AclTable.Remove($wrapper)

            $script:AclThrows = $true
            Check "an unreadable file ACL declines" ((& $fileOk) -eq $false)
            $script:AclThrows = $false
            Check "the file helper refuses an empty path" ((Test-StudioInterpreterFileIsAdminOnly -Path "") -eq $false)
            Check "and a missing file" ((Test-StudioInterpreterFileIsAdminOnly -Path (Join-Path $pfBin "absent")) -eq $false)

            Remove-Item -LiteralPath $wrapper -Force
            New-Item -ItemType SymbolicLink -Path $wrapper -Target $hostPy3 | Out-Null
            Check "the directory walk accepts the symlinked candidate (bites)" ((& $walkOk) -eq $true)
            Check "the file helper refuses a symlinked interpreter" ((& $fileOk) -eq $false)
            Check "an elevated run declines a symlinked PATH interpreter" ([string]::IsNullOrWhiteSpace((& $reprobe)))
            function Test-StudioChildScriptDirectoryElevated { return $false }
            Check "control: an unelevated run is unaffected by the file check" (-not [string]::IsNullOrWhiteSpace((& $reprobe)))
        } finally {
            function Test-StudioChildScriptDirectoryElevated { return $false }
            Remove-Item Function:Get-Acl -ErrorAction SilentlyContinue
            Restore-Env $savedEnv2
            $script:StudioEarlyPythonProbed = $false
            $script:StudioEarlyPython = $null
        }
    }
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All early-Python path resolver checks passed"
exit 0
