#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ask Python for the process image table when the native helper cannot be built.
#
# Get-StudioProcessImagePath decides whether a venv is in use by a running Unsloth, and every
# rung that fails makes the installer likelier to conclude "nothing is running" and overwrite an
# environment that is open. The native rung needs a type defined at runtime; Get-Process needs
# PROCESS_VM_READ, which is refused across a security boundary; the last rung needs a healthy WMI
# repository. This adds a rung that needs none of those: ctypes calling
# QueryFullProcessImageNameW with PROCESS_QUERY_LIMITED_INFORMATION, in one child for the whole
# run rather than one per process.
#
# Strictly additive. It runs only after Get-Process has already declined, and returning $null
# leaves the WMI rung reached exactly as before.
#
# The table itself is Windows-only, so on any other host the rung declines and the checks below
# drive the parsing and the caching through a stubbed runner. The generic child-process runner is
# cross-platform and is exercised for real.
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
    "Remove-StudioTrailingNewline", "Invoke-StudioEarlyPythonScript",
    "Invoke-StudioEarlyPythonScriptViaCmdlets", "Invoke-StudioEarlyPython", "Get-StudioEarlyPython",
    "Get-StudioPythonProcessImageTable", "Get-StudioProcessImagePath"
)) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in install.ps1, found none" }
    Invoke-Expression $fn[0].Extent.Text
}

# The two rungs above the new one, forced off. This is the state it exists to improve.
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
    Write-Host "  SKIP  no Python on this host, which is the fallback case and not a failure" -ForegroundColor Yellow
    exit 0
}
Write-Host "  interpreter: $exe"

$out = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import sys;sys.stdout.write('ok')"
Check "the runner returns the child's stdout" ($out -eq "ok")

# Arguments have to survive the 5.1 branch that builds a command line by hand, so the awkward
# shapes are the point: a space, and a trailing backslash that would otherwise escape its own
# closing quote.
$probeArgs = @("a b", "c\")
$echo = Invoke-StudioEarlyPythonScript -Exe $exe `
    -Script "import sys;sys.stdout.write('|'.join(sys.argv[1:]))" -ScriptArgs $probeArgs
Check "arguments survive quoting, including a space and a trailing backslash" ($echo -eq "a b|c\")

$nonZero = Invoke-StudioEarlyPythonScript -Exe $exe -Script "import sys;sys.stdout.write('x');sys.exit(4)"
Check "a non-zero exit is refused even though the child printed" ($null -eq $nonZero)

# Bites control: the same script exiting 0 does return its output, so the check above is about the
# exit code and not about the runner simply never answering.
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

# --------------------------------------------- the Constrained Language Mode launcher, for real
#
# Not a detail. CLM is one of the two policies that stop a type being defined at runtime, which is
# the reason this whole ladder exists, and CLM also refuses New-Object and every method call on
# System.Diagnostics.Process. A launcher that only worked outside CLM would be absent from half
# the population it is for, while looking perfectly healthy everywhere it is not needed.

$echoScript = "import sys;sys.stdout.write('|'.join(sys.argv[1:]))"
$awkward = @("a b", "c\")

$primaryAnswer = Invoke-StudioEarlyPythonScript -Exe $exe -Script $echoScript -ScriptArgs $awkward
$cmdletAnswer = Invoke-StudioEarlyPythonScriptViaCmdlets -Exe $exe -Script $echoScript -ScriptArgs $awkward
Check "the cmdlet launcher answers at all" ($cmdletAnswer -eq "a b|c\")
# The property that matters. Start-Process redirection appends a trailing newline the child never
# wrote, so without normalising both ends a caller silently reads one of two different strings
# depending on the host's language mode.
Check "both launchers return the same string, byte for byte" ($primaryAnswer -ceq $cmdletAnswer)

# One payload is not a contract. The check above used a single line with no newline in it, and
# passed for a year of edits while the two launchers actually DISAGREED on anything containing a
# CRLF: the ProcessStartInfo launcher reads the child's bytes straight through, and the cmdlet
# launcher's redirection goes via a file. Measured at the time this was added: a child writing
# 61 0d 0a 62 came back 61 0d 0a 62 from one and 61 0a 62 from the other.
#
# Driven over a table instead, with the shapes a child actually produces: an embedded CRLF, an
# embedded LF, trailing newlines of both kinds, no newline at all, and a tab.
function Get-LauncherHex($s) {
    if ($null -eq $s) { return "<null>" }
    return ((([System.Text.Encoding]::UTF8.GetBytes($s)) | ForEach-Object { $_.ToString("x2") }) -join " ")
}
$payloads = @(
    @{ Name = "an embedded CRLF";      Py = "b'a\r\nb'" },
    @{ Name = "an embedded LF";        Py = "b'a\nb'" },
    @{ Name = "a trailing CRLF";       Py = "b'x\r\n'" },
    @{ Name = "several trailing LFs";  Py = "b'p\nq\n\n'" },
    @{ Name = "no newline at all";     Py = "b'solo'" },
    @{ Name = "an embedded tab";       Py = "b'tab\ttab'" }
)
foreach ($case in $payloads) {
    $src = "import sys" + [char]10 + "sys.stdout.buffer.write($($case.Py))"
    $viaPrimary = Invoke-StudioEarlyPythonScript -Exe $exe -Script $src
    $viaCmdlet = Invoke-StudioEarlyPythonScriptViaCmdlets -Exe $exe -Script $src
    Check "the launchers agree on $($case.Name)" ($viaPrimary -ceq $viaCmdlet)
    if ($viaPrimary -cne $viaCmdlet) {
        Write-Host "        primary=$(Get-LauncherHex $viaPrimary)  cmdlet=$(Get-LauncherHex $viaCmdlet)" -ForegroundColor Red
    }
}

# And the bites control for the whole group: the harness must be able to SEE a difference, or
# every row above passes because both launchers are broken in the same way or neither ran.
Check "control: the comparison distinguishes different strings" (
    ("a`nb" -ceq "a`r`nb") -eq $false)
# Parenthesised on purpose: an unparenthesised concatenation in an argument position binds its
# first term to -Script and the rest positionally, which lands [char]10 on -TimeoutMs.
$roundTripSrc = "import sys" + [char]10 + "sys.stdout.buffer.write(b'tab\ttab')"
Check "control: a payload really did round-trip" (
    (Invoke-StudioEarlyPythonScript -Exe $exe -Script $roundTripSrc) -ceq "tab`ttab")

$cmdletUtf8 = Invoke-StudioEarlyPythonScriptViaCmdlets -Exe $exe `
    -Script "import sys;sys.stdout.buffer.write((chr(0xe9)+chr(0x4e2d)).encode('utf-8'))"
Check "the cmdlet launcher keeps non-ASCII intact" (
    $cmdletUtf8 -eq ([string][char]0xE9 + [string][char]0x4E2D))

Check "the cmdlet launcher refuses a non-zero exit" (
    $null -eq (Invoke-StudioEarlyPythonScriptViaCmdlets -Exe $exe -Script "import sys;sys.exit(4)"))

$cmdletStart = Get-Date
$cmdletHung = Invoke-StudioEarlyPythonScriptViaCmdlets -Exe $exe -Script "import time;time.sleep(90)" -TimeoutMs 2000
Check "the cmdlet launcher kills a hung child too" (
    $null -eq $cmdletHung -and ((Get-Date) - $cmdletStart).TotalSeconds -lt 30)

# The handover, driven in a runspace that is genuinely in Constrained Language Mode.
#
# Setting $ExecutionContext.SessionState.LanguageMode partway through a script is NOT enough and
# was tried first: PowerShell fixes a function's language mode when the function is defined, so
# functions defined before the switch keep running in FullLanguage and the primary launcher
# succeeds. The check passed while proving nothing. A runspace created with
# InitialSessionState.LanguageMode set is constrained before anything is defined in it, which is
# what a locked-down host actually looks like.
$clmFunctions = (@(
    "Remove-StudioTrailingNewline", "Invoke-StudioEarlyPythonScript",
    "Invoke-StudioEarlyPythonScriptViaCmdlets"
) | ForEach-Object {
    $name = $_
    ($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)[0]).Extent.Text
}) -join "`n"

$rs = $null
$ps = $null
try {
    $iss = [System.Management.Automation.Runspaces.InitialSessionState]::CreateDefault2()
    $iss.LanguageMode = "ConstrainedLanguage"
    $rs = [System.Management.Automation.Runspaces.RunspaceFactory]::CreateRunspace($iss)
    $rs.Open()
    $ps = [System.Management.Automation.PowerShell]::Create()
    $ps.Runspace = $rs
    $null = $ps.AddScript($clmFunctions + @"

Write-Output "MODE=`$(`$ExecutionContext.SessionState.LanguageMode)"
try { `$null = New-Object System.Diagnostics.ProcessStartInfo; Write-Output 'PSI-ALLOWED' }
catch { Write-Output 'PSI-BLOCKED' }
Write-Output ("ANSWER=" + (Invoke-StudioEarlyPythonScript -Exe '$exe' -Script "$echoScript" -ScriptArgs @('a b', 'c\')))
"@)
    $clmOut = @($ps.Invoke() | ForEach-Object { "$_".Trim() })
    Check "the runspace really is constrained" ($clmOut -contains "MODE=ConstrainedLanguage")
    # The premise of the whole fallback, measured rather than assumed.
    Check "Constrained Language Mode really does refuse ProcessStartInfo" ($clmOut -contains "PSI-BLOCKED")
    # And therefore the primary launcher cannot answer there, so this can only have come from the
    # fallback. Removing the fallback call from the catch makes this check fail.
    Check "under Constrained Language Mode the answer still comes back, through the fallback" (
        $clmOut -contains "ANSWER=a b|c\")
} finally {
    if ($ps) { $ps.Dispose() }
    if ($rs) { $rs.Dispose() }
}

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

    # The caching contract. Get-RunningStudioVenvProcesses calls Get-StudioProcessImagePath once
    # per process on the machine, so a rung that re-probes on every miss would spawn a child per
    # process: slower than the WMI rung it is meant to sit in front of. A $null table is the case
    # that matters, since "no table yet" and "asked, got nothing" look identical without a
    # separate probed flag.
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

    # Bites control: a pid the table does not carry must fall through rather than be invented.
    # Here the rung below throws, so falling through has to end at null.
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

# ctypes defaults every return value to a C int. A HANDLE truncated to a signed 32-bit int is
# sign-extended back on the way into CloseHandle, so the handle closed is not the handle opened
# and the child leaks one per process it inspects. Declaring the signature is what prevents it.
Check "the probe declares OpenProcess's return type" (
    $probeText -match "OpenProcess\.restype\s*=\s*wintypes\.HANDLE")
Check "the probe declares CloseHandle's argument type" (
    $probeText -match "CloseHandle\.argtypes")
# PROCESS_QUERY_LIMITED_INFORMATION. PROCESS_QUERY_INFORMATION (0x400) is the right that a
# protected or cross-session process refuses, which is the whole reason for this rung.
Check "the probe asks for the limited-information right only" ($probeText -match "OpenProcess\(0x1000,")
# EnumProcesses reports the bytes it used. Equal to the buffer size means it may have run out, so
# only a strictly smaller figure proves the enumeration was complete.
Check "the probe grows its buffer until the enumeration is provably complete" (
    $probeText -match "b\.value\s*<\s*ctypes\.sizeof\(a\)")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
