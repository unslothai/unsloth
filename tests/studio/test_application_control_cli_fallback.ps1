#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The installer must not depend on the generated unsloth.exe console script (#8490).
#
# On Windows, packaging materializes `unsloth = unsloth_cli:app` as an unsigned launcher .exe.
# AppLocker, WDAC and Smart App Control deny it while the venv's python.exe -- a copy of the
# signed CPython binary -- still runs, so the install died at "running unsloth studio setup"
# with no exit code to report and no diagnostic worth reading.
#
# These checks pin the three things that fix has to get right: the failure is classified off the
# exception (1260), never off $LASTEXITCODE, which no process was created to set; the CLI is
# reached through the interpreter with the trampoline intact as ONE argument; and the .cmd
# companion is byte-stable, so a re-run rewrites nothing.
# Run: pwsh -NoProfile -File tests/studio/test_application_control_cli_fallback.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$install = Join-Path $repo "install.ps1"

# The one canonical spelling, repeated in install.ps1, studio/src-tauri/src/process.rs and
# unsloth_cli/commands/studio.py. Written out here rather than read from any of them, so a
# silent edit on any side fails a check instead of being copied into the expectation. Both
# halves are load bearing; the rationale is on WINDOWS_CLI_ENTRYPOINT in process.rs.
# Written out for the same reason as the trampoline: an edit on either side has to fail a
# check rather than be copied into the expectation. This one gates a recursive delete.
$ShimMarker = "unsloth-studio-managed-launcher"
$Trampoline = "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"

function Get-FunctionText {
    param([string] $Path, [string] $Name)
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in $Path, found $($fn.Count)" }
    return $fn[0].Extent.Text
}

# Every installer function the extracted ones call has to be in the harness, or a
# code path that reaches it dies with CommandNotFound and the check above it passes
# for the wrong reason. Exactly that shipped twice: Write-StudioLine missing from the
# mutex harness in tests/python, and Test-UnslothCmdShimFile missing from Invoke-Write
# below the moment the writer started asking it.
function Assert-HarnessComplete {
    param([string] $Extracted, [string[]] $Provided, [string] $Label)
    $installerFunctions = @([regex]::Matches(
        (Get-Content -Raw -LiteralPath $install), '(?m)^    function ([\w-]+) \{'
    ) | ForEach-Object { $_.Groups[1].Value })
    $called = @([regex]::Matches($Extracted, '(?<![\w-])([A-Z][\w]*-[\w-]+)') |
        ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
    $missing = @($called | Where-Object { $installerFunctions -contains $_ -and $Provided -notcontains $_ })
    Check "$Label harness defines everything it calls" ($missing.Count -eq 0)
    if ($missing.Count -gt 0) { Write-Host "        missing: $($missing -join ', ')" }
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$blockFn    = Get-FunctionText $install "Test-ApplicationControlBlock"
$cmdlineFn  = Get-FunctionText $install "Get-ManagedUnslothCliCommandLine"
$invokeFn   = Get-FunctionText $install "Invoke-ManagedUnslothCli"
$relFn      = Get-FunctionText $install "Get-RelativeShimPath"
$contentFn  = Get-FunctionText $install "Get-UnslothCmdShimContent"
$writeFn    = Get-FunctionText $install "Write-UnslothCmdShim"
$probeFn    = Get-FunctionText $install "Test-ShimLaunchBlocked"
$shimFileFn = Get-FunctionText $install "Test-UnslothCmdShimFile"
$preferFn   = Get-FunctionText $install "Test-UnslothCmdShimPreferred"

# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the policy code"    ($blockFn -match '1260')
Check "extraction kept the utf8 pin"       ($cmdlineFn -match '-X')
Check "extraction kept the classifier call" ($invokeFn -match 'Test-ApplicationControlBlock')
Check "extraction kept the walk-up"        ($relFn -match '\.\.')
Check "extraction kept the dp0 prefix"     ($contentFn -match '%~dp0')
Check "extraction kept the compare"        ($writeFn -match 'Compare-Object \$existing \$desired')
Check "extraction kept the probe start"    ($probeFn -match 'ProcessStartInfo')
Check "extraction kept the content marker" ($shimFileFn -match 'from unsloth_cli import app')
Check "extraction kept the exe test"       ($preferFn -match 'ShimExe')

# The whole point of the classifier: the value it must NOT consult.
Check "the classifier never reads LASTEXITCODE" (-not ($blockFn -match 'LASTEXITCODE'))
# ...and the invoker must not treat a missing process as exit code 0.
Check "a blocked launch is not an exit code" ($invokeFn -match '\$script:ManagedUnslothCliExit = \$null')
Check "a non-policy launch failure rethrows" ($invokeFn -match 'if \(-not \(Test-ApplicationControlBlock \$_\)\) \{ throw \}')

# --- Test-ApplicationControlBlock -----------------------------------------------------------
# The real shape: PowerShell wraps the CreateProcess failure, so the 1260 is only reachable by
# walking InnerException. A classifier that inspects only the outer exception sees nothing.
function Invoke-Block {
    param($Record)
    $sb = [scriptblock]::Create(@"
param(`$Record)
$blockFn
Test-ApplicationControlBlock `$Record
"@)
    return (& $sb $Record)
}

function New-Win32 {
    param([int] $Code)
    return (New-Object System.ComponentModel.Win32Exception($Code))
}

function New-Record {
    param([Exception] $Exception)
    return (New-Object System.Management.Automation.ErrorRecord(
        $Exception, "NativeCommandFailed", [System.Management.Automation.ErrorCategory]::ResourceUnavailable, $null))
}

Write-Host "an Application Control block is recognised wherever PowerShell buried it"
Check "bare Win32Exception 1260"  (Invoke-Block (New-Win32 1260))
Check "wrapped once"              (Invoke-Block (New-Object System.Exception("outer", (New-Win32 1260))))
Check "wrapped twice"             (Invoke-Block (New-Object System.Exception("a", (New-Object System.Exception("b", (New-Win32 1260))))))
Check "inside an ErrorRecord"     (Invoke-Block (New-Record (New-Object System.Exception("outer", (New-Win32 1260)))))
# Some wrappers keep only the HRESULT form of the same code.
$hresultOnly = New-Object System.Exception("policy")
$hresultOnly.GetType().GetField("_HResult", "Instance,NonPublic").SetValue($hresultOnly, -2147023636)
Check "HRESULT 0x800704EC only"   (Invoke-Block $hresultOnly)
Check "HRESULT inside an ErrorRecord" (Invoke-Block (New-Record $hresultOnly))

Write-Host "and nothing else is mistaken for one"
# 5 is ERROR_ACCESS_DENIED, the neighbouring failure Test-AccessDeniedError owns. Reporting a
# permissions problem as a security policy sends the user to the wrong administrator.
Check "access denied is not a policy block" (-not (Invoke-Block (New-Win32 5)))
Check "file not found is not a policy block" (-not (Invoke-Block (New-Win32 2)))
Check "a plain exception"          (-not (Invoke-Block (New-Object System.Exception("boom"))))
Check "a plain ErrorRecord"        (-not (Invoke-Block (New-Record (New-Object System.Exception("boom")))))
Check "null"                       (-not (Invoke-Block $null))
# 1260 as an exit code is a program's own choice and says nothing about policy.
Check "an exit code of 1260 is not a block" (-not (Invoke-Block 1260))

# --- Get-ManagedUnslothCliCommandLine -------------------------------------------------------
# Start-Process joins an -ArgumentList array with spaces and quotes NOTHING, so handing it the
# trampoline as an array element gives python eleven arguments instead of one. This builds the
# command line itself.
function Invoke-CommandLine {
    param([string[]] $Arguments = @())
    $sb = [scriptblock]::Create(@"
param(`$Arguments)
`$script:UnslothCmdShimMarker = "$ShimMarker"
`$script:UnslothCliTrampoline = "$Trampoline"
$cmdlineFn
Get-ManagedUnslothCliCommandLine -Arguments `$Arguments
"@)
    return (& $sb $Arguments)
}

$line = Invoke-CommandLine @("studio", "-p", "8888")
Write-Host "the interpreter command line survives Start-Process"
Check "the trampoline is one quoted token" ($line.Contains('"' + $Trampoline + '"'))
Check "-X utf8 leads"               ($line.StartsWith("-X utf8 "))
Check "-c follows the utf8 flag"    ($line.IndexOf("-X utf8") -lt $line.IndexOf("-c"))
# -I would also discard PYTHONPATH, PYTHONWARNINGS and user site-packages, which the
# console script honours; the trampoline drops the cwd entry by itself instead.
Check "the interpreter is not isolated" (-not ($line -match '(^|\s)-I(\s|$)'))
Check "caller arguments come last"  ($line.EndsWith("studio -p 8888"))
Check "no arguments still builds"   ((Invoke-CommandLine).Trim().EndsWith('sys.exit(app())"'))
# Start-Process joins the line back with spaces, so an unquoted spaced argument would
# reach the child as several. Only bare subcommands are passed today; the signature is
# what invites the mistake.
$spaced = Invoke-CommandLine @("studio", "run", "--model", "C:\my models\a b.gguf")
Check "a spaced argument is quoted"  ($spaced.EndsWith('studio run --model "C:\my models\a b.gguf"'))
Check "and unspaced ones are not"    ($spaced -match '\srun\s--model\s')

# --- Invoke-ManagedUnslothCli ---------------------------------------------------------------
# Runs for real against this host's python3 -- the trampoline is swapped for a stub, because the
# thing under test is the plumbing (exit code, argument fidelity, output routing), not unsloth.
$python = (Get-Command python3 -ErrorAction SilentlyContinue)
if (-not $python) { $python = (Get-Command python -ErrorAction SilentlyContinue) }

function Invoke-Managed {
    param([string] $Python, [string] $Trampoline, [string[]] $Arguments = @())
    $sb = [scriptblock]::Create(@"
param(`$Python, `$Trampoline, `$Arguments)
`$script:UnslothCliTrampoline = `$Trampoline
$blockFn
$invokeFn
Invoke-ManagedUnslothCli -Python `$Python -Arguments `$Arguments
[pscustomobject]@{ Exit = `$script:ManagedUnslothCliExit }
"@)
    return (& $sb $Python $Trampoline $Arguments)
}

if ($python) {
    Write-Host "the CLI runs through the interpreter, and its exit code comes back intact"
    # Every argument after -c lands in sys.argv[1:], so this proves the caller's arguments
    # arrive unsplit AND that the exit code is the child's.
    $countArgs = 'import sys; sys.exit(len(sys.argv) - 1)'
    $r = Invoke-Managed $python.Source $countArgs @("studio", "setup", "--verbose")
    Check "argument count reaches the child"  ($r[-1].Exit -eq 3)
    $r = Invoke-Managed $python.Source 'import sys; sys.exit(7)' @()
    Check "a nonzero exit is reported"        ($r[-1].Exit -eq 7)
    $r = Invoke-Managed $python.Source 'import sys; sys.exit(0)' @()
    Check "a clean exit is reported as 0"     ($r[-1].Exit -eq 0)
    # An argument with spaces must stay one argument.
    $r = Invoke-Managed $python.Source 'import sys; sys.exit(0 if sys.argv[1] == "a b c" else 9)' @("a b c")
    Check "a spaced argument stays whole"     ($r[-1].Exit -eq 0)
    # The child's stdout must reach the caller's pipeline rather than being swallowed into a
    # return value; that is why the exit code is published in a variable instead.
    $r = Invoke-Managed $python.Source 'print("hello from the child")' @()
    Check "the child's stdout is not swallowed" (($r | Where-Object { "$_" -eq "hello from the child" }).Count -eq 1)
    Check "and the exit code is still separate" ($r[-1].Exit -eq 0)

    Write-Host "a launch failure that is not a policy block is not silently swallowed"
    $threw = $false
    try { Invoke-Managed (Join-Path ([System.IO.Path]::GetTempPath()) "unsloth-no-such-interpreter") 'pass' @() | Out-Null }
    catch { $threw = $true }
    Check "a missing interpreter rethrows"    $threw
} else {
    Write-Host "  SKIP  interpreter checks (no python3 on this host)" -ForegroundColor Yellow
}

# --- Get-RelativeShimPath -------------------------------------------------------------------
function Invoke-Relative {
    param([string] $From, [string] $To)
    $sb = [scriptblock]::Create(@"
param(`$From, `$To)
$relFn
Get-RelativeShimPath -From `$From -To `$To
"@)
    return (& $sb $From $To)
}

Write-Host "the shim reaches the interpreter without naming the install path"
Check "sibling directory"   ((Invoke-Relative "C:\Users\me\.unsloth\studio\bin" "C:\Users\me\.unsloth\studio\unsloth_studio\Scripts\python.exe") -eq "..\unsloth_studio\Scripts\python.exe")
Check "legacy .venv layout" ((Invoke-Relative "C:\s\bin" "C:\s\.venv\Scripts\python.exe") -eq "..\.venv\Scripts\python.exe")
Check "trailing separator"  ((Invoke-Relative "C:\s\bin\" "C:\s\unsloth_studio\Scripts\python.exe") -eq "..\unsloth_studio\Scripts\python.exe")
Check "spaces in the path"  ((Invoke-Relative "C:\Users\Jane Doe\.unsloth\studio\bin" "C:\Users\Jane Doe\.unsloth\studio\unsloth_studio\Scripts\python.exe") -eq "..\unsloth_studio\Scripts\python.exe")
Check "deeper base"         ((Invoke-Relative "C:\a\b\c\bin" "C:\a\python.exe") -eq "..\..\..\python.exe")
# A venv on another volume has no relative form; the caller must fall back rather than emit junk.
Check "different volume -> null" ($null -eq (Invoke-Relative "C:\s\bin" "D:\v\Scripts\python.exe"))
Check "empty input -> null"      ($null -eq (Invoke-Relative "" "C:\s\python.exe"))
# A descending range counts backwards in PowerShell, so an unguarded tail would answer
# "a\C:" here rather than refusing.
Check "target above the base -> null" ($null -eq (Invoke-Relative "C:\a\b\bin" "C:\a"))
Check "identical paths -> null"       ($null -eq (Invoke-Relative "C:\a\bin" "C:\a\bin"))

# --- Get-UnslothCmdShimContent --------------------------------------------------------------
function Invoke-Content {
    param([string] $ShimDir, [string] $PythonPath)
    $sb = [scriptblock]::Create(@"
param(`$ShimDir, `$PythonPath)
`$script:UnslothCmdShimMarker = "$ShimMarker"
`$script:UnslothCliTrampoline = "$Trampoline"
$relFn
$contentFn
Get-UnslothCmdShimContent -ShimDir `$ShimDir -PythonPath `$PythonPath
"@)
    return (& $sb $ShimDir $PythonPath)
}

$shimDir = "C:\Users\Jane Doe\.unsloth\studio\bin"
$shimPy = "C:\Users\Jane Doe\.unsloth\studio\unsloth_studio\Scripts\python.exe"
$body = Invoke-Content $shimDir $shimPy

Write-Host "the .cmd is a pure function of %~dp0"
Check "it starts with echo off"     ($body.StartsWith("@echo off`r`n"))
Check "it reaches python via %~dp0" ($body -match '"%~dp0\.\.\\unsloth_studio\\Scripts\\python\.exe"')
# The whole reason for %~dp0: a profile path with a space, a '$' or a bracket never has to be
# escaped, because it is never written down.
Check "the install path is absent"  (-not ($body -match 'Jane Doe'))
Check "the trampoline is quoted"    ($body.Contains('-X utf8 -c "' + $Trampoline + '" %*'))
Check "the shim is not isolated"    (-not ($body -match '\s-I\s'))
# cmd /V:ON, or the machine-wide DelayedExpansion default, eats a '!' out of every
# argument before python sees it unless the scope turns it off.
Check "delayed expansion is off"    ($body.Contains("`r`nsetlocal DisableDelayedExpansion`r`n"))
Check "it forwards every argument"  ($body -match '%\*')
Check "it propagates the exit code" ($body -match '@exit /b %errorlevel%')
Check "CRLF line endings"           (($body -split "`r`n").Count -ge 6 -and -not ($body -match "(?<!`r)`n"))
Check "it is deterministic"         ($body -eq (Invoke-Content $shimDir $shimPy))
# Metacharacters that would otherwise reparse: cmd expands %VAR% even inside double quotes,
# so a literal '%' in an absolute fallback has to be doubled.
$crossVolume = Invoke-Content "C:\s\bin" "D:\100%%tools\py\python.exe"
Check "cross-volume falls back absolute" ($crossVolume -match 'D:\\100')
Check "and doubles every percent"        ($crossVolume -match '100%%%%tools')
$weird = Invoke-Content "C:\s (x)\bin" "C:\s (x)\unsloth_studio\Scripts\python.exe"
Check "brackets and parens stay out"     (-not ($weird -match '\(x\)'))

# --- Write-UnslothCmdShim -------------------------------------------------------------------
# Runs against a real directory: the idempotency contract is about bytes on disk.
function Invoke-Write {
    param([string] $ShimDir, [string] $PythonPath)
    $sb = [scriptblock]::Create(@"
param(`$ShimDir, `$PythonPath)
function substep { param(`$Message, `$Color) Write-Output "SUBSTEP: `$Message" }
`$script:UnslothCmdShimMarker = "$ShimMarker"
`$script:UnslothCliTrampoline = "$Trampoline"
$relFn
$contentFn
$shimFileFn
$writeFn
Write-UnslothCmdShim -ShimDir `$ShimDir -PythonPath `$PythonPath
"@)
    return (& $sb $ShimDir $PythonPath)
}

Assert-HarnessComplete `
    -Extracted ($relFn + "`n" + $contentFn + "`n" + $shimFileFn + "`n" + $writeFn) `
    -Provided @("Get-RelativeShimPath", "Get-UnslothCmdShimContent", "Test-UnslothCmdShimFile",
                "Write-UnslothCmdShim") `
    -Label "the .cmd writer"

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-shim-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
try {
    $binDir = Join-Path $tmp "bin"
    $null = New-Item -ItemType Directory -Path $binDir -Force
    $pyPath = Join-Path (Join-Path (Join-Path $tmp "unsloth_studio") "Scripts") "python.exe"
    $cmdPath = Join-Path $binDir "unsloth.cmd"

    Invoke-Write $binDir $pyPath | Out-Null
    Write-Host "the .cmd is written once and then left alone"
    Check "the file appears"        (Test-Path -LiteralPath $cmdPath -PathType Leaf)
    $firstBytes = [System.IO.File]::ReadAllBytes($cmdPath)
    # cmd.exe reads a UTF-8 BOM as part of the first command, so there must not be one.
    Check "no BOM"                  (-not ($firstBytes.Length -ge 3 -and $firstBytes[0] -eq 0xEF -and $firstBytes[1] -eq 0xBB -and $firstBytes[2] -eq 0xBF))
    Check "ASCII only"              (-not ($firstBytes | Where-Object { $_ -gt 127 }))

    $stampBefore = (Get-Item -LiteralPath $cmdPath).LastWriteTimeUtc
    Start-Sleep -Milliseconds 50
    Invoke-Write $binDir $pyPath | Out-Null
    $stampAfter = (Get-Item -LiteralPath $cmdPath).LastWriteTimeUtc
    # Same bytes means no write at all, not a rewrite that happens to match: a rewrite would
    # move the timestamp and churn the file on every no-op reinstall.
    Check "a re-run rewrites nothing" ($stampBefore -eq $stampAfter)
    Check "the bytes are unchanged"   (-not (Compare-Object $firstBytes ([System.IO.File]::ReadAllBytes($cmdPath))))
    # No temp file is left behind either way. Filtered by name rather than by -Filter:
    # "unsloth.cmd.*" keeps 8.3 wildcard semantics and matches bare "unsloth.cmd" too.
    Check "no leftover temp file"     (@(Get-ChildItem -LiteralPath $binDir | Where-Object { $_.Name -ne "unsloth.cmd" }).Count -eq 0)

    # A stale shim from an install at a different location must be replaced.
    [System.IO.File]::WriteAllText($cmdPath, "@echo off`r`nrem stale`r`n")
    Invoke-Write $binDir $pyPath | Out-Null
    Check "stale content is replaced"  (-not ((Get-Content -Raw -LiteralPath $cmdPath) -match 'stale'))

    # Byte identical except for a leading BOM. ReadAllText strips it before comparing,
    # so a text compare calls this file unchanged and leaves cmd.exe reading the BOM as
    # part of `@echo off` forever.
    $withBom = [byte[]](0xEF, 0xBB, 0xBF) + $firstBytes
    [System.IO.File]::WriteAllBytes($cmdPath, $withBom)
    Invoke-Write $binDir $pyPath | Out-Null
    Check "a BOM-prefixed shim is rewritten" (-not (Compare-Object ([System.IO.File]::ReadAllBytes($cmdPath)) $firstBytes))

    # Case is the other thing -eq ignores. Same reasoning: cmd is case insensitive for
    # commands but the trampoline it carries is Python, where SYS is not sys.
    [System.IO.File]::WriteAllText($cmdPath, ((Get-Content -Raw -LiteralPath $cmdPath).ToUpperInvariant()))
    Invoke-Write $binDir $pyPath | Out-Null
    Check "an upper-cased shim is rewritten"  (-not (Compare-Object ([System.IO.File]::ReadAllBytes($cmdPath)) $firstBytes))

    # A run killed between the write and the rename leaves its temp behind, and the next
    # run has a different PID. Nothing else would ever collect it.
    $orphan = Join-Path $binDir "unsloth.cmd.999999.tmp"
    [System.IO.File]::WriteAllText($orphan, "half written")
    Invoke-Write $binDir $pyPath | Out-Null
    Check "an orphaned temp file is swept"    (-not (Test-Path -LiteralPath $orphan))
    Check "sweeping did not disturb the shim" (-not (Compare-Object ([System.IO.File]::ReadAllBytes($cmdPath)) $firstBytes))

    # ...but only ours, and only when its owner is gone. A live PID's temp belongs to a
    # concurrent install that is about to rename it.
    $live = Join-Path $binDir "unsloth.cmd.$PID.tmp"
    [System.IO.File]::WriteAllText($live, "in flight")
    $foreignTemp = Join-Path $binDir "something-else.tmp"
    [System.IO.File]::WriteAllText($foreignTemp, "not ours")
    Invoke-Write $binDir $pyPath | Out-Null
    Check "a live PID's temp is left alone"   (Test-Path -LiteralPath $live)
    Check "an unrelated temp is left alone"   (Test-Path -LiteralPath $foreignTemp)
    Remove-Item -LiteralPath $live, $foreignTemp -Force

    # A directory in the way is reported, not thrown out of the installer.
    $blockedDir = Join-Path $tmp "bin2"
    $null = New-Item -ItemType Directory -Path (Join-Path $blockedDir "unsloth.cmd") -Force
    $out = Invoke-Write $blockedDir $pyPath
    Check "a directory in the way warns" (($out -join "`n") -match 'SUBSTEP: cannot write')
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

# --- wiring ---------------------------------------------------------------------------------
# Normalised to LF once: on a CRLF checkout every \n-anchored pattern below matches nothing and
# the -not checks pass vacuously.
$installText = (Get-Content -Raw $install) -replace "`r`n", "`n"

Write-Host "nothing the installer drives starts the generated console script"
Check "the setup handoff goes through python" ($installText -match '(?m)^\s*Invoke-ManagedUnslothCli -Python \$VenvPython -Arguments \$studioArgs$')
Check "and reads the published exit code"     ($installText -match '\$setupExit = \$script:ManagedUnslothCliExit')
Check "a blocked launch is reported, not printed as an empty exit code" (
    $installText -match '(?s)if \(\$null -eq \$setupExit\) \{\s*return \(Exit-InstallFailure \(Write-ApplicationControlBlocked')
Check "autostart goes through python"         ($installText -match 'Start-Process -FilePath \$VenvPython')
Check "the shortcuts launcher takes python"   ($installText -match 'New-StudioShortcuts -ManagedPythonPath \$VenvPython')
Check "the generated launcher runs the interpreter" ($installText -match '\$studioPython -replace')
# The executing forms, specifically. `Test-Path $UnslothExe` and the hardlink still name the
# file on purpose: it is still generated, still shimmed, just never started.
Check "no call operator on the exe"    (-not ($installText -match '&\s*\$UnslothExe'))
Check "no Start-Process on the exe"    (-not ($installText -match 'Start-Process -FilePath \$UnslothExe'))

# A policy DENIES the launcher and leaves it on disk; antivirus QUARANTINES it, deleting it
# out of a venv that still runs. Nothing past this point starts it, so its absence must not
# abort the install for exactly the machines this change is for.
Write-Host "a quarantined launcher does not abort the install"
Check "absence asks the interpreter first" (
    $installText -match '(?s)if \(-not \(Test-Path -LiteralPath \$UnslothExe\)\) \{.*?Invoke-ManagedUnslothCli -Python \$VenvPython -Arguments @\("--version"\)')
Check "and only then gives up" (
    $installText -match '(?s)if \(\$cliImportable\) \{.*?\} else \{.*?Exit-InstallFailure "unsloth CLI was not installed correctly"')
Check "a venv with no interpreter at all still fails" (
    $installText -match '(?s)\$cliImportable = \$false\s*\n\s*if \(Test-Path -LiteralPath \$VenvPython\)')
# Anti-vacuity: the guidance for a genuinely CLI-less wheel has to survive, since that is
# the case this check was written for and the one the fallback must not swallow.
Check "the older-unsloth guidance is still there" (
    $installText -match 'an older unsloth version was installed that does not include the Unsloth CLI')

# Write-UnslothCmdShim warns and leaves an unwritable file alone, so a foreign bin\unsloth.cmd
# can survive a run. Counting it as a launcher would put its directory on PATH and advertise
# someone else's command as the policy-safe way in.
# The bare Test-Path that used to gate this is checked for above, in "PATH accepts either
# launcher". The only surviving occurrence is inside Test-UnslothCmdShimFile's own guard,
# which is where it belongs, so the negative is anchored to the assignment rather than to
# the string appearing anywhere in the file.
Check "PATH counts only our own .cmd" (
    -not ($installText -match '\$ShimUsable = .*Test-Path -LiteralPath \$ShimCmd'))
# ...and it is still installed and still hardlinked, or every unaffected machine regresses.
Check "the exe is still hardlinked"    ($installText -match 'New-Item -ItemType HardLink -Path \$ShimExe -Target \$UnslothExe')
Check "the exe existence gate remains" ($installText -match '\$UnslothExe = Join-Path \$VenvDir "Scripts\\unsloth\.exe"')

Write-Host "the launcher and the shortcuts are rewritten only when they change"
# A reinstall and every `unsloth studio update` regenerate both. Rewriting identical bytes
# moves their timestamps, which is the churn the idempotency contract exists to prevent.
# Bytes, including the BOM. ReadAllText decodes the preamble away and PowerShell's -eq
# is case insensitive, so a text compare calls a BOM-less launcher written by an older
# installer unchanged and it never gains the BOM Windows PowerShell 5.1 needs.
Check "launch-studio.ps1 is compared as bytes" (
    $installText -match '(?s)\$launcherUnchanged =\s*\(@\(Compare-Object \$existingLauncher \$desiredLauncher -SyncWindow 0\)\.Count -eq 0\)')
Check "the BOM is part of what is compared"    ($installText -match '\$utf8Bom\.GetPreamble\(\) \+ \$utf8Bom\.GetBytes\(\$launcherContent\)')
Check "and skipped when unchanged"             ($installText -match '(?s)if \(-not \$launcherUnchanged\) \{\s*\[System\.IO\.File\]::WriteAllBytes\(\$launcherPs1')
Check "the .lnk is saved only when changed"   ($installText -match '(?s)if \(-not \$shortcutUnchanged\) \{.*?\$shortcut\.Save\(\)')

Write-Host "the escape hatch reaches installs that only ever run an update"
# `unsloth studio update` drives install.ps1 through -ShortcutsOnly and returns before the
# main shim block, so without this an install made before the .cmd existed never gets one.
Check "-ShortcutsOnly writes the .cmd" ($installText -match '(?s)if \(\$ShortcutsOnly\) \{.*?Write-UnslothCmdShim -ShimDir \$ShortcutShimDir.*?New-StudioShortcuts')
# An installer older than the shim directory never created one, and `unsloth studio
# update` is the only route those installs ever take back into install.ps1. Testing
# only that the write appears in the branch missed that it was conditional on a
# directory that population does not have.
Check "-ShortcutsOnly creates the bin dir first" (
    $installText -match '(?s)if \(\$ShortcutsOnly\) \{.*?\[System\.IO\.Directory\]::CreateDirectory\(\$ShortcutShimDir\).*?Write-UnslothCmdShim -ShimDir \$ShortcutShimDir')
Check "PATH accepts either launcher"   ($installText -match '(?s)\$ShimUsable = \(Test-Path -LiteralPath \$ShimExe -PathType Leaf\) -or\s*\(Test-UnslothCmdShimFile -Path \$ShimCmd\)')
Check "and the env-mode export too"    ($installText -match "StudioRedirectMode -eq 'env' -and \`$ShimUsable")

Write-Host "the launch instructions are probed, not assumed"
# PATHEXT resolves .EXE before .CMD, so bare `unsloth` picks the blocked file. Printing the
# .cmd unconditionally would change what every unaffected machine sees, so it is gated.
Check "the decline branch probes"      ($installText -match 'if \(Test-UnslothCmdShimPreferred -ShimExe \$ShimExe -ShimCmd \$ShimCmd\) \{\n\s*substep "unsloth\.cmd studio -p 8888"')
Check "the manual branch probes"       ($installText -match '\$_shimBlocked = Test-UnslothCmdShimPreferred -ShimExe \$ShimExe -ShimCmd \$ShimCmd')
Check "and prints the bare form otherwise" ($installText -match '\$_bareLaunch = if \(\$_shimBlocked\) \{ "unsloth\.cmd studio -p 8888" \} else \{ "unsloth studio -p 8888" \}')

Write-Host "the trampoline is the one the desktop already uses"
# WINDOWS_CLI_ENTRYPOINT lives in process.rs, shared by the backend spawn, the auth
# provisioning, the health probe and the updater. update.rs only calls the builder now.
$processRs = Join-Path $repo "studio/src-tauri/src/process.rs"
$rsText = Get-Content -Raw $processRs
$studioPy = Join-Path $repo "unsloth_cli/commands/studio.py"
# Python splits the literal across source lines to stay inside the line length, so join
# adjacent string literals back before comparing. Matching the raw text instead would fail
# the moment someone rewrapped the constant without changing its value.
$pyText = (Get-Content -Raw $studioPy) -replace '"\s*\r?\n\s*"', ''
# Drift here is silent: two spellings of argv[0] both "work" until one of them stops being
# recognised as the console-script entry point. All three are checked against the literal
# at the top of this file, so editing any one of them alone fails here.
Check "install.ps1 carries the trampoline" ($installText.Contains('$script:UnslothCliTrampoline = "' + $Trampoline + '"'))
Check "process.rs agrees"                  ($rsText.Contains($Trampoline))
Check "studio.py agrees"                   ($pyText.Contains($Trampoline))
Check "nothing still isolates with -I"     (-not ($installText -match '"-X", "utf8", "-I"'))

Write-Host "a bare unsloth.cmd does not hand an unrelated directory to the uninstaller"
$uninstallText = Get-Content -Raw (Join-Path $repo "scripts/uninstall.ps1")
Check "ownership is content-checked" ($uninstallText -match '_IsUnslothCmdShim \(Join-Path \$Path "bin\\unsloth\.cmd"\)')
Check "and the marker is the trampoline" ($uninstallText -match 'from unsloth_cli import app')
$setupText = Get-Content -Raw (Join-Path $repo "studio/setup.ps1")
Check "setup.ps1 guards the same way"   ($setupText -match 'Test-UnslothCmdShimFile \(Join-Path \$StudioHome "bin\\unsloth\.cmd"\)')
Check "install.ps1 guards the same way" ($installText -match 'Test-UnslothCmdShimFile \(Join-Path \$StudioHome "bin\\unsloth\.cmd"\)')

# The predicate itself, run for real: a file with the right name and the wrong contents
# must not qualify, because the callers delete whole directories on the strength of it.
function Invoke-ShimFileCheck {
    param([string] $Path)
    $sb = [scriptblock]::Create(@"
param(`$Path)
$shimFileFn
Test-UnslothCmdShimFile -Path `$Path
"@)
    return (& $sb $Path)
}

$ownTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-own-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
try {
    $null = New-Item -ItemType Directory -Path $ownTmp -Force
    $realShim = Join-Path $ownTmp "unsloth.cmd"
    [System.IO.File]::WriteAllText($realShim, (Invoke-Content "C:\s\bin" "C:\s\unsloth_studio\Scripts\python.exe"))
    Check "our own shim qualifies"      (Invoke-ShimFileCheck $realShim)

    $impostor = Join-Path $ownTmp "impostor.cmd"
    [System.IO.File]::WriteAllText($impostor, "@echo off`r`nunsloth %*`r`n")
    Check "someone else's wrapper does not" (-not (Invoke-ShimFileCheck $impostor))

    Check "a missing file does not"     (-not (Invoke-ShimFileCheck (Join-Path $ownTmp "absent.cmd")))
    Check "an empty path does not"      (-not (Invoke-ShimFileCheck ""))
    $asDir = Join-Path $ownTmp "dir.cmd"
    $null = New-Item -ItemType Directory -Path $asDir -Force
    Check "a directory does not"        (-not (Invoke-ShimFileCheck $asDir))
    # Bounded read: a huge file named unsloth.cmd is not our few-hundred-byte shim, and
    # slurping it to find out would be the wrong trade.
    $huge = Join-Path $ownTmp "huge.cmd"
    [System.IO.File]::WriteAllText($huge, ("x" * 9000) + "from unsloth_cli import app")
    Check "an oversized file does not"  (-not (Invoke-ShimFileCheck $huge))
} finally {
    Remove-Item -LiteralPath $ownTmp -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Test-ShimLaunchBlocked, chatty launcher ----------------------------------------------
# Both pipes are redirected, so a launcher that fills one blocks writing while the probe
# blocks waiting and neither side moves. Reachable since the trampoline stopped passing -I:
# PYTHONPROFILEIMPORTTIME=1 now reaches the child and produces ~24 KB of stderr against a
# 4 KB pipe buffer. Measured on this runner: undrained, the probe burns its full 20s
# timeout; drained, it returns in tens of milliseconds.
$probeTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-probe-" + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $probeTmp
try {
    $onWindows = $env:OS -eq "Windows_NT"
    $line = "." * 64
    # Has to outrun the pipe buffer to mean anything: Linux defaults to 64 KB, so 26 KB
    # of output would sail through undrained and the timing check below would be vacuous.
    $repeats = 2000
    if ($onWindows) {
        $chatty = Join-Path $probeTmp "chatty.cmd"
        $body = "@echo off`r`n"
        $body += "for /L %%i in (1,1,$repeats) do @echo $line`r`n"
        $body += "for /L %%i in (1,1,$repeats) do @echo $line 1>&2`r`n"
        [System.IO.File]::WriteAllText($chatty, $body, (New-Object System.Text.UTF8Encoding($false)))
    } else {
        $chatty = Join-Path $probeTmp "chatty.sh"
        $body = "#!/bin/sh`n"
        $body += "i=0`n"
        $body += "while [ `$i -lt $repeats ]; do echo '$line'; echo '$line' >&2; i=`$((i+1)); done`n"
        [System.IO.File]::WriteAllText($chatty, $body, (New-Object System.Text.UTF8Encoding($false)))
        & chmod +x $chatty
    }

    $sb = [scriptblock]::Create(@"
param(`$Path)
$blockFn
$probeFn
Test-ShimLaunchBlocked -Path `$Path
"@)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $blocked = & $sb $chatty
    $sw.Stop()

    Write-Host "a launcher that outfills the pipe buffer does not stall the probe"
    Check "extraction kept the drain"  ($probeFn -match 'ReadToEndAsync')
    # -1 -lt anything, so an absent drain would pass this on order alone.
    Check "drains before it waits"     ($probeFn.IndexOf('ReadToEndAsync') -ge 0 -and
                                        $probeFn.IndexOf('ReadToEndAsync') -lt $probeFn.IndexOf('WaitForExit'))
    # A launcher that ran is not blocked, whatever it printed.
    Check "a chatty launcher is not blocked" (-not $blocked)
    Check "and it did not hit the timeout"   ($sw.ElapsedMilliseconds -lt 10000)
} finally {
    Remove-Item -LiteralPath $probeTmp -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Test-UnslothCmdShimPreferred -----------------------------------------------------------
# Which launcher the printed instructions name. Test-ShimLaunchBlocked answers $false for a
# file that is not there, because nothing refused to start it, so a quarantined or never-created
# .exe used to leave the installer printing a path that does not exist.
$prefTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-pref-" + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $prefTmp
try {
    $prefSb = [scriptblock]::Create(@"
param(`$ShimExe, `$ShimCmd)
$blockFn
$probeFn
$preferFn
Test-UnslothCmdShimPreferred -ShimExe `$ShimExe -ShimCmd `$ShimCmd
"@)
    $exe = Join-Path $prefTmp "unsloth.exe"
    $cmd = Join-Path $prefTmp "unsloth.cmd"

    Write-Host "the instructions name the .cmd only when the .exe cannot serve"
    # Nothing written yet: no .cmd to name.
    Check "no .cmd, no preference"     (-not (& $prefSb $exe $cmd))
    [System.IO.File]::WriteAllText($cmd, "@echo off`r`n")
    # This is the regression: the .exe is gone, so naming it prints a dead path.
    Check "quarantined .exe prefers it" (& $prefSb $exe $cmd)
    # A runnable .exe wins, which is what every unaffected machine has.
    Copy-Item -LiteralPath (Get-Command pwsh).Source -Destination $exe -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $exe) {
        Check "a runnable .exe keeps it"   (-not (& $prefSb $exe $cmd))
    }
} finally {
    Remove-Item -LiteralPath $prefTmp -Recurse -Force -ErrorAction SilentlyContinue
}

# --- the .cmd ownership marker ---------------------------------------------------------------
# _IsStudioRoot accepts a bin\unsloth.cmd as proof that a user-named root is ours, and that
# answer gates a recursive delete. `from unsloth_cli import app` alone is a line anyone could
# have in a hand-rolled wrapper, so the generated shim carries a marker nobody writes by
# accident and every check requires it.
$markerSb = [scriptblock]::Create(@"
param(`$Path)
`$script:UnslothCmdShimMarker = "$ShimMarker"
$shimFileFn
Test-UnslothCmdShimFile -Path `$Path
"@)
$markerTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-marker-" + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $markerTmp
try {
    $ours = Join-Path $markerTmp "ours.cmd"
    [System.IO.File]::WriteAllText($ours, "@echo off`r`nrem unsloth-studio-managed-launcher`r`n... from unsloth_cli import app ...`r`n")
    $theirs = Join-Path $markerTmp "theirs.cmd"
    [System.IO.File]::WriteAllText($theirs, "@echo off`r`npython -c `"from unsloth_cli import app; sys.exit(app())`" %*`r`n")

    Write-Host "only the shim this installer generated proves ownership"
    Check "our shim is recognised"      (& $markerSb $ours)
    Check "a hand-rolled wrapper is not" (-not (& $markerSb $theirs))
    Check "the generator emits the marker" ($installText -match 'rem \$script:UnslothCmdShimMarker')
    # The uninstaller gates a recursive delete on the same answer.
    Check "the uninstaller requires it too" ($uninstallText -match 'unsloth-studio-managed-launcher')
} finally {
    Remove-Item -LiteralPath $markerTmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
