#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The ctypes process-image rung between Get-Process and WMI. The runner is exercised for real;
# the table is Windows-only, so its parsing and caching go through a stubbed runner.
# Run: pwsh -NoProfile -File tests/studio/test_early_python_process_image.ps1

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
    "New-StudioChildScriptDirectory", "Test-StudioChildScriptDirectoryElevated",
    "Get-StudioPythonProcessImageTable", "Get-StudioProcessImagePath",
    # What Get-StudioEarlyPython reaches on Windows: the elevation gate and its helpers. Off
    # Windows it never calls them, which is how a missing one once passed here and failed CI.
    "Get-StudioSystem32Tool", "Test-StudioPathUnderAdminRoot",
    "Test-StudioSddlRightsAreWrite", "Test-StudioSddlPrincipalIsAdminOnly",
    "Test-StudioSddlWritableByNonAdmin", "Test-StudioDirectoryIsAdminOnly", "Test-StudioInterpreterFileIsAdminOnly",
    "Get-StudioLexicalParent", "Write-StudioFinalPathDegraded"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

# The emitted rung that used to sit at the top is gone. Get-Process, which is the rung above this
# one now, and the WMI rung below it are both forced off: this is the state the ctypes rung exists
# to improve, and on a host where either of them answers nothing here ever runs.
function Get-Process { param($Id, $ErrorAction) throw "no such process" }
function Get-CimInstance { param($ClassName, $ErrorAction) throw "WMI is unavailable" }
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }

function Reset-RungState {
    $script:StudioPythonProcessImageTable = $null
    $script:StudioPythonProcessImageProbed = $false
    $script:StudioProcessImageTable = $null
    $script:StudioProcessImageWarned = $true
}

# ---------------------------------------------------------------- the generic runner, for real

$script:StudioEarlyPythonProbed = $false
$script:StudioEarlyPython = $null
$exe = Get-StudioEarlyPython
if (-not $exe) {
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
    # USABLE, not merely present. The installer rejects an interpreter that cannot complete its own
    # probe, so presence on PATH is not proof that discovery should have found one: Python 2, any
    # Python below 3.8 (whose Windows resolve() does not follow links), a broken executable, and a
    # Windows Store App Execution Alias are all on PATH and all correctly refused. Failing here on
    # those hosts blames this file for the installer behaving as designed.
    #
    # So ask the same question Invoke-StudioEarlyPython asks, of the same candidates, and only then
    # decide. The WindowsApps aliases are excluded WITHOUT running them: they are zero-length stubs
    # that open the Microsoft Store, which a test must not do to whoever is running it.
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
Write-Host "  interpreter: $exe"

$out = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import sys;sys.stdout.write('ok')"
Check "the runner returns the child's stdout" ($out -eq "ok")

# The 5.1 branch quotes by hand: a space, and a trailing backslash that could escape its quote.
$probeArgs = @("a b", "c\")
$echo = Invoke-StudioEarlyPythonScript -Exe $exe `
    -Script "import sys;sys.stdout.write('|'.join(sys.argv[1:]))" -ScriptArgs $probeArgs
Check "arguments survive quoting, including a space and a trailing backslash" ($echo -eq "a b|c\")

$nonZero = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import sys;sys.stdout.write('x');sys.exit(4)"
Check "a non-zero exit is refused even though the child printed" ($null -eq $nonZero)

$zero = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import sys;sys.stdout.write('x')"
Check "control: the identical script exiting 0 does answer" ($zero -eq "x")

$started = [System.Diagnostics.Stopwatch]::StartNew()
$hung = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import time;time.sleep(90)" -TimeoutMs 2000
$started.Stop()
Check "a hung child is killed, not waited on" ($null -eq $hung -and $started.Elapsed.TotalSeconds -lt 30)

$utf8 = Invoke-StudioEarlyPythonScript -Exe $exe `
    -Script "import sys;sys.stdout.buffer.write('\u00e9\u4e2d'.encode('utf-8'))"
Check "non-ASCII output survives the host's console codepage" ($utf8 -eq ([char]0xE9 + [string][char]0x4E2D))

$missing = Invoke-StudioEarlyPythonScript -Exe (Join-Path $root "no-such-interpreter") -Script "pass"
Check "an interpreter that does not exist yields null, not a throw" ($null -eq $missing)

# ------------------------------------------------------- the table, through a stubbed runner

$script:RunnerCalls = 0
$script:RunnerOutput = "10|C:\a\python.exe`n20|C:\b\python.exe"
function Invoke-StudioEarlyPythonScript {
    param([string]$Exe, [string]$Script, [string[]]$ScriptArgs = @(), [int]$TimeoutMs = 10000)
    $script:RunnerCalls++
    return $script:RunnerOutput
}

$savedOs = $env:OS
try {
    $env:OS = "Windows_NT"

    Reset-RungState
    $table = Get-StudioPythonProcessImageTable
    Check "pid|path lines become a table" (
        $table -and $table.Count -eq 2 -and $table[10] -eq "C:\a\python.exe" -and $table[20] -eq "C:\b\python.exe")

    Reset-RungState
    $script:RunnerOutput = "30|C:\c\python.exe`r`n`r`nnot-a-row`r`n|no-pid`r`n40|"
    $table = Get-StudioPythonProcessImageTable
    Check "blank, pid-less, path-less and malformed rows are dropped, the good one kept" (
        $table -and $table.Count -eq 1 -and $table[30] -eq "C:\c\python.exe")

    Reset-RungState
    $script:RunnerOutput = ""
    Check "a child that answers nothing yields null, so the WMI rung is still reached" (
        $null -eq (Get-StudioPythonProcessImageTable))

    # A $null table must not re-probe: "not asked" and "asked, got nothing" need the probed flag.
    Reset-RungState
    $script:RunnerCalls = 0
    $script:RunnerOutput = ""
    $null = Get-StudioProcessImagePath -ProcessId 1
    $null = Get-StudioProcessImagePath -ProcessId 2
    $null = Get-StudioProcessImagePath -ProcessId 3
    Check "a declining probe runs once per installer run, not once per process" ($script:RunnerCalls -eq 1)

    Reset-RungState
    $script:RunnerCalls = 0
    $script:RunnerOutput = "77|C:\d\python.exe"
    $first = Get-StudioProcessImagePath -ProcessId 77
    $second = Get-StudioProcessImagePath -ProcessId 77
    Check "the ladder returns the Python answer when the rungs above decline" (
        $first -eq "C:\d\python.exe" -and $second -eq "C:\d\python.exe")
    Check "an answering probe is also cached" ($script:RunnerCalls -eq 1)

    Reset-RungState
    $script:RunnerOutput = "77|C:\d\python.exe"
    Check "control: an unknown pid falls through to the rung below" (
        $null -eq (Get-StudioProcessImagePath -ProcessId 999))

    $env:OS = "Linux"
    Reset-RungState
    $script:RunnerCalls = 0
    Check "off Windows the rung declines: QueryFullProcessImageNameW does not exist there" (
        $null -eq (Get-StudioPythonProcessImageTable) -and $script:RunnerCalls -eq 0)
} finally {
    if ($null -eq $savedOs) { Remove-Item Env:OS -ErrorAction SilentlyContinue } else { $env:OS = $savedOs }
}

# ------------------------------------------------------------------------ the embedded probe

$probeFn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Get-StudioPythonProcessImageTable"
}, $true)[0]
$probeText = $probeFn.Extent.Text

# ctypes defaults restype to int, which truncates a HANDLE and closes the wrong one.
Check "the probe declares OpenProcess's return type" (
    $probeText -match "OpenProcess\.restype\s*=\s*wintypes\.HANDLE")
Check "the probe declares CloseHandle's argument type" (
    $probeText -match "CloseHandle\.argtypes")
# 0x400 is refused by protected and cross-session processes; 0x1000 is not.
Check "the probe asks for the limited-information right only" ($probeText -match "OpenProcess\(0x1000,")
# EnumProcesses reports the bytes it used. Equal to the buffer size means it may have run out, so
# only a strictly smaller figure proves the enumeration was complete.
# Exactness. There is no longer a native rung to agree WITH: the emitted
# UnslothStudioProcessImageV1 type is gone and this ctypes probe is the only caller of
# OpenProcess and QueryFullProcessImageNameW in the installer. What still matters is that the
# call it makes is the one the native rung used to make, because Test-StudioProtectedPathMatch
# compares paths this returns against paths recorded by earlier installs that used the native
# rung. Same access right, same flags argument (0 is the Win32 path form; 1 would return
# \Device\HarddiskVolume1\... instead), same buffer size. Those three constants are pinned
# here directly rather than by comparison, since there is nothing left to compare against.
Check "the probe asks for PROCESS_QUERY_LIMITED_INFORMATION, as the native rung did" (
    $probeText -match "OpenProcess\(0x1000,")
Check "it passes flags 0, so it gets the Win32 path form and not the device form" (
    $probeText -match "QueryFullProcessImageNameW\(h,0,")
Check "it sizes the buffer as the native rung did" (
    $probeText -match "create_unicode_buffer\(32768\)")
# And the rung it replaced really is gone, rather than merely unreferenced by this test.
$installWhole = [System.IO.File]::ReadAllText($installPs1)
Check "no emitted process-image type remains" ($installWhole -notmatch "UnslothStudioProcessImageV1")
Check "no native process-image helper remains" (
    $installWhole -notmatch "Get-StudioNativeProcessImagePath|Initialize-StudioProcessImageNativeType")

# EnumProcesses filling the buffer exactly may mean truncation; only a smaller count is complete.
Check "the probe grows its buffer until the enumeration is provably complete" (
    $probeText -match "b\.value\s*<\s*ctypes\.sizeof\(a\)")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
