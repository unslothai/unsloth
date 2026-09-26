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
    "Get-StudioPythonProcessImageTable", "Get-StudioProcessImagePath"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

# The rungs above this one, forced off.
function Initialize-StudioProcessImageNativeType { return $false }
function Get-StudioNativeProcessImagePath { param([int]$ProcessId) return $null }
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

    # Interpreter discovery can throw; the WMI rung below this one must still get its turn.
    Reset-RungState
    $savedTable = ${function:Get-StudioPythonProcessImageTable}
    function Get-StudioPythonProcessImageTable { throw "interpreter discovery failed" }
    function Get-CimInstance { param($ClassName, $ErrorAction)
        return @([pscustomobject]@{ ProcessId = 55; ExecutablePath = "C:\w\python.exe" }) }
    $threw = $null; $viaWmi = $null
    try { $viaWmi = Get-StudioProcessImagePath -ProcessId 55 } catch { $threw = $_ }
    Check "a table probe that throws still reaches the WMI rung" (
        $null -eq $threw -and $viaWmi -eq "C:\w\python.exe")
    function Get-CimInstance { param($ClassName, $ErrorAction) throw "WMI is unavailable" }
    ${function:Get-StudioPythonProcessImageTable} = $savedTable

    # Validating the answer is part of the null-on-error contract: an access error declines.
    $script:RunnerOutput = [System.IO.Path]::GetTempPath()
    function Test-Path { param($LiteralPath, $PathType, $ErrorAction)
        throw [System.UnauthorizedAccessException]::new("Access to the path is denied.") }
    $threw = $null; $answer = "unset"
    try { $answer = Invoke-StudioEarlyPython -Exe "python3" -Path $root } catch { $threw = $_ }
    Remove-Item Function:Test-Path -ErrorAction SilentlyContinue
    Check "an access error validating the answer declines instead of throwing" (
        $null -eq $threw -and $null -eq $answer)

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
# Both rungs must return the same string or Test-StudioProtectedPathMatch sees false differences.
$nativeInit = ($ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Get-StudioNativeProcessImagePath" }, $true)[0]).Extent.Text
$nativeRight = if ($nativeInit -match '\$queryLimitedInformation\s*=\s*\[uint32\](0x[0-9a-fA-F]+)') { $Matches[1] } else { "" }
Check "the native rung asks for the same access right the probe does" (
    $nativeRight -eq "0x1000" -and
    $nativeInit -match "OpenProcess\(\s*\r?\n?\s*\`$queryLimitedInformation," -and
    $probeText -match "OpenProcess\(0x1000,")
Check "both pass flags 0, so both get the Win32 path form and not the device form" (
    $nativeInit -match "QueryFullProcessImageNameW\(\s*\`$handle,\s*\[uint32\]0," -and
    $probeText -match "QueryFullProcessImageNameW\(h,0,")
Check "both size the buffer the same" (
    $nativeInit -match "32768" -and $probeText -match "create_unicode_buffer\(32768\)")

# EnumProcesses filling the buffer exactly may mean truncation; only a smaller count is complete.
Check "the probe grows its buffer until the enumeration is provably complete" (
    $probeText -match "b\.value\s*<\s*ctypes\.sizeof\(a\)")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
