#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A single-AMD-GPU host must not install itself into a loop (#8335).
# Under Windows PowerShell 5.1, .Count on a lone CimInstance or [pscustomobject] is $null
# (strings and ints still answer 1), so setup saw "gpu none" and judged the ROCm venv stale.
# Under 5.1 this reproduces it for real; under pwsh 7 the cases assert the source shape.
# Run: pwsh -NoProfile -File tests/studio/test_amd_venv_repair_loop.ps1
#  or: powershell -NoProfile -File tests\studio\test_amd_venv_repair_loop.ps1   (5.1, Windows)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setup = Join-Path $root "studio/setup.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Source text of each named function, to be Invoke-Expression'd at script scope by the caller.
function Get-HelperSources($path, $names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $out = @()
    foreach ($name in $names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -lt 1) { throw "expected $name in $path, found none" }
        $out += $fn[0].Extent.Text
    }
    return $out
}

function substep { param([string]$Message, [string]$Color = "DarkGray") }
function Write-StudioStdoutMirror { param([string]$Line) }

Write-Host ""
Write-Host "=== the unroll, and what this host does with it ==="
$psMajor = $PSVersionTable.PSVersion.Major
$is51 = ($psMajor -lt 6)
Write-Host "  host: PowerShell $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition))"

$oneGpu = @("AMD Radeon PRO W7900")
$unwrapped = if ($oneGpu.Count -gt 0) { $oneGpu } else { @() }
$wrapped = @(if ($oneGpu.Count -gt 0) { $oneGpu } else { @() })
Check "an if-expression unrolls a one-element array" (-not ($unwrapped -is [array]))
Check "the @() wrap keeps it an array"               ($wrapped -is [array])
Check "the wrapped value still counts one GPU"       ($wrapped.Count -eq 1)

# Only some types return $null on 5.1; pinning the String case documents that.
Check "a String scalar answers 1 on every PowerShell" ((("x")).Count -eq 1)
Check "so does the unrolled string array"             ($unwrapped.Count -eq 1)

# [pscustomobject] stands in for CimInstance and is available on Linux pwsh.
$_countless = ([pscustomobject]@{ Name = "AMD Radeon PRO W7900" }).Count
if ($is51) {
    Check "5.1: a Count-less scalar answers `$null"   ($null -eq $_countless)
} else {
    Check "7: a Count-less scalar answers 1"          ($_countless -eq 1)
}

Write-Host ""
Write-Host "=== #8335 itself, against a real CIM instance ==="
# Windows only: Win32_OperatingSystem always returns one instance, reproducing one GPU.
# $IsWindows does not exist on 5.1, which is Windows by construction.
if ($is51 -or $IsWindows) {
    $_cim = @(Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue)
    Check "exactly one CIM instance to work with" ($_cim.Count -eq 1)
    if ($_cim.Count -eq 1) {
        $_old = if ($_cim.Count -gt 0) { $_cim } else { @() }
        $_new = @(if ($_cim.Count -gt 0) { $_cim } else { @() })
        Check "the unwrapped form is a bare CimInstance" (
            $_old -is [Microsoft.Management.Infrastructure.CimInstance])
        Check "the @() wrapped form stays an array"      ($_new -is [array])
        Check "the @() wrap makes the GPU-label branch fire" ($_new.Count -gt 0)
        if ($is51) {
            Check "5.1: the OLD form loses the only adapter (#8335)" (-not ($_old.Count -gt 0))
        } else {
            Check "7: the OLD form survives, so 7 cannot see #8335"  ($_old.Count -gt 0)
        }
    }
} else {
    Write-Host "  SKIP  not Windows, Get-CimInstance unavailable (source shapes below still run)"
}

Write-Host ""
Write-Host "=== Test-VenvTorchIsRocm reads the wheel off disk ==="
# AMD counterpart of Test-VenvTorchIsXpu: reads files only, launching no interpreter.
foreach ($srcText in (Get-HelperSources $setup @("Test-VenvTorchIsRocm"))) { Invoke-Expression $srcText }
# @() then [0]: a one-element result unrolls to a String, and [0] would take its first char.
$isRocmFn = @(Get-HelperSources $setup @("Test-VenvTorchIsRocm"))[0]
# An empty extraction would make every case below pass vacuously.
Check "extraction kept the version file" ($isRocmFn -match 'version\.py')

# Filesystem cmdlets are shadowed so this runs on Linux CI and checks the content read.
function Invoke-IsRocm {
    param([string] $VenvPath, [string] $VersionPyBody, [switch] $Missing, [switch] $Throws)
    $sb = [scriptblock]::Create(@"
param(`$Body, `$Missing, `$Throws)
function Join-Path {
    [CmdletBinding()] param([Parameter(Position=0)] [string] `$Path, [Parameter(Position=1)] [string] `$ChildPath)
    # Shadowed: the real cmdlet validates the drive qualifier, so "C:\..." throws on Linux CI.
    if (-not `$ChildPath) { return `$Path }
    return (`$Path.TrimEnd('\', '/') + '\' + `$ChildPath.TrimStart('\', '/'))
}
function Test-Path {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    return (-not `$Missing)
}
function Get-Content {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    if (`$Throws) { throw "access denied" }
    return (`$Body -split "``n")
}
$isRocmFn
Test-VenvTorchIsRocm -VenvPath '$VenvPath'
"@)
    return (& $sb $VersionPyBody ([bool]$Missing) ([bool]$Throws))
}

$XPU  = "__version__ = '2.9.1+xpu'`ndebug = False"
$CU   = "__version__ = '2.9.1+cu128'`ncuda: Optional[str] = '12.8'"
$BARE = "__version__ = '2.9.1'"
# Windows ROCm wheels use a three-part label (2.11.0+rocm7.13.0); Linux uses 2.8.0+rocm6.4.
# "+rocm" is what matters; the "+gfx" arm is defensive.
Check "rocm6.4 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'")
Check "rocm7.0 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+rocm7.0'")
Check "rocm7.13.0 wheel (#8335)" (Invoke-IsRocm "C:\v" "__version__ = '2.11.0+rocm7.13.0'")
Check "gfx1151 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+gfx1151'")
Check "gfx110X-all wheel"    (Invoke-IsRocm "C:\v" "__version__ = '2.7.1+gfx110X.all'")
# A dev/nightly segment precedes the local label.
Check "nightly rocm wheel"   (Invoke-IsRocm "C:\v" "__version__ = '2.12.0.dev20260801+rocm7.2'")
Check "nightly cuda wheel"   (-not (Invoke-IsRocm "C:\v" "__version__ = '2.12.0.dev20260801+cu130'"))
Check "source build"         (-not (Invoke-IsRocm "C:\v" "__version__ = '2.9.0a0+git1a2b3c'"))
Check "cuda wheel"           (-not (Invoke-IsRocm "C:\v" $CU))
Check "xpu wheel"            (-not (Invoke-IsRocm "C:\v" $XPU))
Check "untagged wheel"       (-not (Invoke-IsRocm "C:\v" $BARE))
Check "cpu wheel"            (-not (Invoke-IsRocm "C:\v" "__version__ = '2.10.0+cpu'"))
# Only __version__ decides; git_version can name a branch.
Check "gfx in git_version"   (-not (Invoke-IsRocm "C:\v" ($CU + "`ngit_version = 'rocm-branch-gfx1100'")))
# CUDA builds carry a `hip` attribute too, so only the __version__ label decides.
Check "hip attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`nhip: Optional[str] = None")))
Check "gfx attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`ngfx = 'gfx1100'")))
Check "no torch installed"   (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Missing))
Check "unreadable file"      (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Throws))
Check "no venv path"         (-not (Invoke-IsRocm "" "__version__ = '2.8.0+rocm6.4'"))

Write-Host ""
Write-Host "=== Invoke-BoundedPythonProbe keeps the reason it failed ==="
# Both installers carry a copy. The probe must keep stderr, stay bounded and drain async, so a
# real child process is launched.
$py = (Get-Command python3 -ErrorAction SilentlyContinue)
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue) }
Check "an interpreter is available to probe" ($null -ne $py)

foreach ($file in @("install.ps1", "studio/setup.ps1")) {
    $path = Join-Path $root $file
    Write-Host "--- $file"
    foreach ($srcText in (Get-HelperSources $path @("Invoke-BoundedPythonProbe"))) {
        Invoke-Expression $srcText
    }
    if ($py) {
        $ok = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'print(1 + 1)'
        Check "a good probe still answers"        ($ok.Ok -and $ok.Output.Trim() -eq "2")
        Check "a good probe reports no error"     ([string]::IsNullOrWhiteSpace($ok.Error))

        $bad = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'import sys; sys.stderr.write(chr(91) + chr(87) + chr(105) + chr(110) + chr(69) + chr(114) + chr(114) + chr(111) + chr(114) + chr(32) + chr(49) + chr(50) + chr(54) + chr(93)); sys.exit(1)'
        Check "a failed probe still reads as not Ok" (-not $bad.Ok)
        Check "the stderr text survives"             ($bad.Error -match 'WinError 126')
        Check "stderr does not leak into Output"     (-not ($bad.Output -match 'WinError'))

        $slow = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'import time; time.sleep(45)' -TimeoutSec 1
        Check "a hung probe is still bounded"        (-not $slow.Ok)
        Check "a hung probe says it timed out"       ($slow.Error -match 'did not answer within')

        Check "an empty code string is refused"      (-not (Invoke-BoundedPythonProbe -PythonExe $py.Source -Code '').Ok)
    }
    # No interpreter: Process.Start throws, and its text is all there is to report.
    $gone = Invoke-BoundedPythonProbe -PythonExe (Join-Path $root "no-such-python-XYZ") -Code 'print(1)'
    Check "a missing interpreter reads as not Ok"    (-not $gone.Ok)
    Check "a missing interpreter reports why"        (-not [string]::IsNullOrWhiteSpace($gone.Error))
}

Write-Host ""
Write-Host "=== source shapes ==="
# Normalised to LF once, or \n-anchored patterns match nothing and -not checks pass vacuously.
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"

Write-Host "the AMD WMI fallback survives a single-GPU host"
# The trailing `\n` makes the CRLF check bite; `[^\r\n]*` absorbs any guard without the `\r`.
$_wmiPat = '(?s)(\$amdGpus = @\(Get-CimInstance Win32_VideoController.*?\$ROCmGpuLabel = \$script:ROCmGpuLabels\[0\][^\r\n]*\n)'
$_wmi = if ($setupText -match $_wmiPat) { $Matches[1] } else { "" }
Check "the WMI fallback was found"        ($_wmi -ne "")
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $_wmiPat))
# The #8335 assertion: a shape, not a behaviour (see header).
Check "the healthy/all choice is @() wrapped" (
    $_wmi -match '\$wmiGpus = @\(if \(\$healthyGpus\.Count -gt 0\) \{ \$healthyGpus \} else \{ \$amdGpus \}\)')
Check "the unwrapped form is gone"        (-not ($_wmi -match '\$wmiGpus = if \('))
# Where-Object also returns a scalar on one match.
Check "the adapter list is @() wrapped"   ($_wmi -match '\$amdGpus = @\(Get-CimInstance')
Check "the healthy list is @() wrapped"   ($_wmi -match '\$healthyGpus = @\(\$amdGpus \| Where-Object')

Write-Host "a driver fault does not delete a ROCm venv"
# Ends on `\{\n` for the same CRLF reason.
$_rescuePat = '(?s)(\$_verProbe = Invoke-BoundedPythonProbe.*?\} else \{\n        # Missing python\.exe means)'
$_rescue = if ($setupText -match $_rescuePat) { $Matches[1] } else { "" }
Check "the probe chain was found"          ($_rescue -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_rescuePat))
Check "an unreadable flavour is rescued off disk" ($_rescue -match 'elseif \(Test-VenvTorchIsRocm -VenvPath \$VenvDir\)')
Check "the rescue classifies it as rocm"   ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm.*?\$installedTorchTag = "rocm"')
Check "the rescue never sets shouldRebuild" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm[^}]*\$shouldRebuild = \$true'))
# Avoids launching a second interpreter after the first hung.
Check "the rescue launches no interpreter" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm.*?Invoke-BoundedPythonProbe'))
# Matched on the substep argument, since the comment above the branch names the driver too.
Check "the rescue names the driver fix"    ($_rescue -match 'substep "[^"]*Adrenalin')
Check "other families still rebuild"       ($_rescue -match '(?s)\} else \{\s*\$shouldRebuild = \$true')
# The XPU rescue must stay ahead of it: an Arc box names a different driver.
Check "the XPU rescue is untouched"        ($_rescue -match 'elseif \(Test-VenvTorchIsXpu -VenvPath \$VenvDir\)')
Check "XPU is judged before ROCm"          (
    $_rescue.IndexOf('Test-VenvTorchIsXpu -VenvPath') -ge 0 -and
    $_rescue.IndexOf('Test-VenvTorchIsXpu -VenvPath') -lt $_rescue.IndexOf('Test-VenvTorchIsRocm -VenvPath'))

Write-Host "the stale-venv decision has no dead end left in it"
# Starts on `\$null\n` so a CRLF checkout fails the match.
$_repairPat = '(?s)(\$reason = \$null\n.*?Rename-Item -LiteralPath \$VenvDir)'
$_repair = if ($setupText -match $_repairPat) { $Matches[1] } else { "" }
Check "the stale-venv decision was found"  ($_repair -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_repairPat))
# install.ps1 restores the previous env on failure, so aborting here can never converge.
# Bounded to the branch: the delete below has its own Exit-SetupFailure calls.
$_managedPat = '(?s)(if \(\$shouldRebuild -and \$InstallerManagedSetup\) \{.*?\n    \}\n)'
$_managed = if ($_repair -match $_managedPat) { $Matches[1] } else { "" }
Check "the installer-managed branch was found" ($_managed -ne "")
Check "an installer-managed run does not abort" (-not ($_managed -match 'Exit-SetupFailure'))
# Matched on the argument, since the comment above quotes the old advice.
Check "the advice that pointed at itself is gone" (
    -not ($setupText -match 'Write-StudioLine "\s*Re-run install\.ps1'))
Check "the abort message is gone too"      (-not ($setupText -match 'The existing Unsloth environment needs repair'))
Check "an installer-managed run repairs in place" ($_managed -match '\$script:PinChangedForceReinstall = \$true')
Check "and clears the rebuild flag"        ($_managed -match '\$shouldRebuild = \$false')
# install.ps1 runs setup through the venv's own unsloth.exe, so deleting was never possible.
Check "it does not try to wipe the venv it runs from" (-not ($_managed -match 'Remove-Item'))
Check "a direct update still rebuilds"     ($_repair -match 'Stale venv detected \(\$reason\) -- rebuilding')
# A running image cannot be deleted, so an update from the venv's python repairs in place.
Check "a direct update from inside the venv repairs in place" (
    $_repair -match '(?s)\$_hostPy = Get-SetupHostInterpreterInVenv -VenvDir \$VenvDir.*?\$script:PinChangedForceReinstall = \$true.*?\$shouldRebuild = \$false')
# A rename fails whole; a recursive delete can stop at a locked file and leave a broken venv.
Check "the rebuild moves the venv aside before deleting" (
    $_repair -match 'Rename-Item -LiteralPath \$VenvDir' -and -not ($_repair -match 'Remove-Item -LiteralPath \$VenvDir'))
Check "the custom-home guard still gates the wipe" ($_repair -match '\$StudioHomeIsCustom')
# The sweep runs ahead of the guard on every run, so it asks the same ownership question.
Check "the sweep reads the same ownership guard" ($_repair -match '\$_studioRootIsOurs')
Check "the sweep wants more than a matching name" (
    $_repair -match 'foreach \(\$_sign in @\("pyvenv\.cfg", \$StudioOwnedMarker, \$StudioStaleMarker\)\)')
Check "a taken stale name takes a suffix"        ($_repair -match '\$_staleTry -lt 64')
Check "the sweep recognises that suffix"         ($_repair -match '\(\?:-\[0-9\]\+\)\?\$')
# Distinguishes a faulted GPU driver from a missing wheel for the user.
Check "the swallowed probe error is surfaced" ($_repair -match '\$_verProbe\.Error')
Check "the probe handle is declared up front" ($setupText -match '(?m)^\s*\$_verProbe = \$null')
# Assigned only inside the venv-exists block, so fresh installs would fail under Set-StrictMode.
Check "the force-reinstall flag is declared outside the venv block" (
    $setupText -match '(?m)^\$script:PinChangedForceReinstall = \$false$')

Write-Host "the surfaced probe error survives a stderr that is only whitespace"
# Blank-line stderr passes the guard but leaves nothing for [0], fatal under Set-StrictMode
# (setup.bat runs without -NoProfile). Driven for real with the old form as a control.
$_strictNewOk = $true
try {
    & {
        Set-StrictMode -Version Latest
        $probe = [pscustomobject]@{ Ok = $false; Output = ""; Error = "   `r`n`r`n  " }
        if ($probe -and -not $probe.Ok -and $probe.Error) {
            $line = $probe.Error -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1
            if ($line) { $null = $line.Trim() }
        }
    }
} catch { $script:_strictNewOk = $false; Write-Host "    threw: $($_.Exception.Message)" }
Check "the shipped form does not throw under strict mode" $_strictNewOk

$_strictOldThrew = $false
try {
    & {
        Set-StrictMode -Version Latest
        $probe = [pscustomobject]@{ Error = "   `r`n`r`n  " }
        $null = @($probe.Error -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1)[0]
    }
} catch { $script:_strictOldThrew = $true }
# Without this the check above would pass on any rewrite.
Check "the @(...)[0] form it replaced really did throw" $_strictOldThrew
Check "setup.ps1 no longer indexes that pipeline" (
    -not ($setupText -match 'Select-Object -Last 1\)\[0\]'))
$_realErr = "Traceback (most recent call last):`r`nOSError: [WinError 126] The specified module could not be found"
$_realLine = $_realErr -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1
Check "a real stderr still yields its last line" ($_realLine -match 'WinError 126')

Write-Host ""
Write-Host "=== an interpreter-less venv is not repaired in place, it is refused ==="
# A venv dir with no python.exe is incomplete, not stale. Dot-sourcing a missing Activate is
# non-terminating at "Continue", so pip installs outside the venv and exits 0. Prove it first.
$_dotSourceKeptGoing = $false
& {
    $ErrorActionPreference = "Continue"
    . (Join-Path $root "no-such-Activate-XYZ.ps1")
    $script:_dotSourceKeptGoing = $true
} 2>$null
Check "dot-sourcing a missing script does not stop the script" $_dotSourceKeptGoing

# Bounded to the reuse branch (the not-found branch has its own Exit-SetupFailure); ends after a
# non-newline token so CRLF fails the match.
$_reusePat = '(?s)(substep "reusing existing virtual environment at \$VenvDir"\n.*?Exit-SetupFailure "No interpreter at [^\n]*\n)'
$_reuse = if ($setupText -match $_reusePat) { $Matches[1] } else { "" }
Check "the reuse branch was found"         ($_reuse -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_reusePat))
Check "it refuses a venv with no interpreter" ($_reuse -match '(?s)\} else \{.*?Exit-SetupFailure "No interpreter at')
Check "the refusal precedes the activation" (
    $setupText.IndexOf('Exit-SetupFailure "No interpreter at') -lt
    $setupText.IndexOf('$ActivateScript = Join-Path $VenvDir'))
# Not a wipe: this venv may hold the only copy of the previous one's contents.
Check "it does not delete the venv"        (-not ($_reuse -match 'Remove-Item'))
Check "it says incomplete, not out of date" ($_reuse -match 'incomplete rather than out of date')
Check "a venv with an interpreter still just prints its version" (
    $_reuse -match '(?s)if \(Test-Path -LiteralPath \$_venvPyExe\) \{.*?--version')

# Activate.ps1 must exist too: install.ps1 leaves Scripts off PATH, so a missing one hits the
# same hazard.
Check "it refuses a venv with no activation script" (
    $_reuse -match 'Exit-SetupFailure "No activation script at')
Check "the activation script path is built next to the interpreter" (
    $_reuse -match '\$_venvActivate = Join-Path \$VenvDir "Scripts\\Activate\.ps1"')
# IndexOf returns -1 when absent, which would pass the ordering check alone.
Check "that refusal also precedes the activation" (
    $setupText.IndexOf('Exit-SetupFailure "No activation script at') -ge 0 -and
    $setupText.IndexOf('Exit-SetupFailure "No activation script at') -lt
    $setupText.IndexOf('$ActivateScript = Join-Path $VenvDir'))
Check "the check sits inside the interpreter-present arm" (
    $_reuse -match '(?s)if \(Test-Path -LiteralPath \$_venvPyExe\) \{.*?Exit-SetupFailure "No activation script at.*?\} else \{')
Check "it does not delete that venv either" (-not ($_reuse -match 'Remove-Item'))

Write-Host ""
Write-Host "=== a present activation script is not the same as an activated venv ==="
# Activate.ps1 updates PATH in its last statement, so a truncated copy silently does nothing and
# an unparseable one is non-terminating. Prove both hazards before asserting the guard.
$_actRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-activate-" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $_actRoot -Force | Out-Null

    $_truncated = Join-Path $_actRoot "truncated.ps1"
    Set-Content -LiteralPath $_truncated -Value '$env:UNSLOTH_ACTIVATE_PROBE = "reached"'
    $_corrupt = Join-Path $_actRoot "corrupt.ps1"
    Set-Content -LiteralPath $_corrupt -Value @('$env:UNSLOTH_ACTIVATE_PROBE = "reached"', 'if ( { unclosed')

    $_pathBefore = $env:PATH
    $env:UNSLOTH_ACTIVATE_PROBE = ""
    $Error.Clear()
    $_truncKeptGoing = $false
    & {
        $ErrorActionPreference = "Continue"
        . $_truncated
        $script:_truncKeptGoing = $true
    } 2>$null
    Check "a truncated activation script does not stop the script" $_truncKeptGoing
    Check "and raises no error at all" ($Error.Count -eq 0)
    Check "and leaves PATH exactly as it found it" ($env:PATH -eq $_pathBefore)
    # It still set VIRTUAL_ENV-like state, which is why the guard checks the resolved interpreter.
    Check "while still setting the state that precedes the PATH line" ($env:UNSLOTH_ACTIVATE_PROBE -eq "reached")

    $env:UNSLOTH_ACTIVATE_PROBE = $null

    # Runs in a child host to match setup.ps1: script-scope "Continue", no enclosing try.
    $_hostExe = try { [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName } catch { $null }
    Check "the host executable is known" ($_hostExe -and (Test-Path -LiteralPath $_hostExe))
    $_child = Join-Path $_actRoot "child.ps1"
    Set-Content -LiteralPath $_child -Value @(
        '$ErrorActionPreference = "Continue"',
        '$before = $env:PATH',
        ('. ' + "'" + $_corrupt + "'"),
        'if ($env:PATH -eq $before) { Write-Output "PATH-UNTOUCHED" }',
        'Write-Output "SURVIVED"',
        'exit 0')
    # 5.1 applies $ErrorActionPreference to native stderr (7.1+ does not), so scope "Stop" off for
    # this call; the exit code is what is judged.
    $_childOut = & {
        $ErrorActionPreference = "Continue"
        & $_hostExe -NoProfile -File $_child 2>$null
    } | Out-String
    $_childRc = $LASTEXITCODE
    Check "an unparseable activation script does not stop the script either" ($_childOut -match 'SURVIVED')
    Check "and it leaves PATH untouched too" ($_childOut -match 'PATH-UNTOUCHED')
    Check "and the run still exits 0" ($_childRc -eq 0)

    # Caught rather than thrown, so a tree without the guard reports every check as FAIL.
    $_assertFn = try { @(Get-HelperSources $setup @("Assert-VenvActivated"))[0] } catch { "" }
    # An empty extraction would make every case below pass vacuously.
    Check "extraction kept the interpreter lookup" ($_assertFn -match 'Get-Command python')
    # A VIRTUAL_ENV check is the wrong answer (see truncation above); -ne "" is the anti-vacuity half.
    Check "the guard does not settle for VIRTUAL_ENV" (($_assertFn -ne "") -and -not ($_assertFn -match 'VIRTUAL_ENV'))

    # Real directories: separators and case folding are the whole risk.
    $_venvOk = Join-Path $_actRoot "venv"
    $_venvOkPy = Join-Path (Join-Path $_venvOk "Scripts") "python.exe"
    New-Item -ItemType File -Path $_venvOkPy -Force | Out-Null
    $_venvSibling = Join-Path $_actRoot "venv2"
    $_venvSiblingPy = Join-Path (Join-Path $_venvSibling "Scripts") "python.exe"
    New-Item -ItemType File -Path $_venvSiblingPy -Force | Out-Null
    $_ambientPy = Join-Path (Join-Path $_actRoot "WindowsApps") "python.exe"
    New-Item -ItemType File -Path $_ambientPy -Force | Out-Null

    # Returns the refusal message, or $null when the guard let the run continue.
    function Invoke-AssertVenv {
        param([string] $VenvDir, [string] $PythonSource, [string] $CommandType = 'Application', [switch] $NoPython)
        $sb = [scriptblock]::Create(@"
param(`$Venv, `$Src, `$Type, `$NoPy)
function Write-StudioLine { param([string] `$Line, `$ForegroundColor) }
# Exit-SetupFailure calls `exit`, which is not catchable; a throw is, so the case can be asserted.
function Exit-SetupFailure { param([string] `$Message, [int] `$Code = 1) throw "REFUSED: `$Message" }
function Get-Command {
    [CmdletBinding()] param([Parameter(Position = 0)] `$Name, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    if (`$NoPy) { return `$null }
    return [pscustomobject]@{ Source = `$Src; CommandType = `$Type }
}
$_assertFn
Assert-VenvActivated -VenvDir `$Venv
"@)
        try { & $sb $VenvDir $PythonSource $CommandType ([bool]$NoPython) | Out-Null; return $null }
        catch { return $_.Exception.Message }
    }

    Check "an activated venv is accepted" (
        $null -eq (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_venvOkPy))
    Check "an ambient interpreter is refused" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy) -match 'did not put its interpreter on PATH')
    Check "and the refusal names where python actually resolved" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy) -match 'WindowsApps')
    Check "no python at all is refused" (
        (Invoke-AssertVenv -VenvDir $_venvOk -NoPython) -match 'did not put its interpreter on PATH')
    # A prefix compare without a separator would call venv2 a child of venv.
    Check "a sibling venv is not mistaken for this one" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_venvSiblingPy) -match 'did not put its interpreter on PATH')

    # False positives matter more: refusing a working install is worse than the bug.
    Check "a trailing separator on the venv path still passes" (
        $null -eq (Invoke-AssertVenv -VenvDir ($_venvOk + [System.IO.Path]::DirectorySeparatorChar) -PythonSource $_venvOkPy))
    Check "a differently cased venv path still passes" (
        $null -eq (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource ($_venvOkPy.ToUpperInvariant())))
    # An alias or function named python has no Source, so nothing can be proven.
    Check "a non-application python is left alone" (
        $null -eq (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy -CommandType 'Function'))
    Check "an unresolvable venv path is left alone" (
        $null -eq (Invoke-AssertVenv -VenvDir (Join-Path $_actRoot "no-such-venv") -PythonSource $_ambientPy))
} finally {
    Remove-Item -LiteralPath $_actRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Ends after a non-newline token so CRLF fails the match. The dot-source is reached through
# Enter-StudioVenv, so the region spans that helper.
$_actPat = '(?s)(\$ActivateScript = Join-Path \$VenvDir "Scripts\\Activate\.ps1"\n.*?\. \$ActivateScript\n.*?Assert-VenvActivated -VenvDir \$VenvDir\n)'
$_act = if ($setupText -match $_actPat) { $Matches[1] } else { "" }
Check "the activation block was found"     ($_act -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_actPat))
Check "the assertion follows the dot-source" (
    $setupText.IndexOf('. $ActivateScript') -ge 0 -and
    $setupText.IndexOf('. $ActivateScript') -lt
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir'))
# Fast-Install is where a wrong interpreter starts receiving the stack.
Check "it lands before Fast-Install resolves python" (
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir') -ge 0 -and
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir') -lt
    $setupText.IndexOf('function Fast-Install'))
# The re-activation after Refresh-Environment is in a catch that swallows everything.
Check "the re-activation is covered as well" (
    ([regex]::Matches($setupText, [regex]::Escape('Assert-VenvActivated -VenvDir $VenvDir'))).Count -ge 2)
Check "and that one is outside the swallowing catch" (
    $setupText.IndexOf('} catch { }') -ge 0 -and
    $setupText.IndexOf('} catch { }') -lt
    $setupText.LastIndexOf('Assert-VenvActivated -VenvDir $VenvDir'))
# It refuses only. Env: is exempt: the staged branch clears PYTHONHOME like Activate.ps1 does.
Check "the guard does not delete anything" (
    ($_act -ne "") -and -not ($_act -match 'Remove-Item(?!\s+Env:)'))

Write-Host ""
Write-Host "=== an installer-managed repair never moves a GPU wheel to another family ==="
# setup re-probes hardware after install.ps1 installed torch; a disagreeing probe (cpu or another
# GPU vendor) must not force-reinstall over install.ps1's chosen wheel and then exit 0.
$_guardPat = '(?s)(if \(\$shouldRebuild -and \$InstallerManagedSetup -and\n.*?\n    \}\n)'
$_guard = if ($setupText -match $_guardPat) { $Matches[1] } else { "" }
Check "the downgrade guard was found"       ($_guard -ne "")
Check "CRLF is normalised, not tolerated"   (-not (($setupText -replace "`n", "`r`n") -match $_guardPat))
Check "it only fires under the installer"   ($_guard -match '\$InstallerManagedSetup')
Check "it only fires on a GPU wheel"        ($_guard -match '\$installedTorchTag -and \$installedTorchTag -ne "cpu"')
# Not narrowed to a cpu rescan: GPU-to-GPU disagreements must be covered too.
Check "it is not narrowed to a cpu rescan"  (-not ($_guard -match '\$expectedTorchTag -eq "cpu"'))
# $expectedTorchTag may be unassigned when torch fails to import; reading it in the condition
# is fatal under Set-StrictMode.
Check "the condition never reads the expected tag" (
    -not (($_guard -split '\{', 2)[0] -match '\$expectedTorchTag'))
Check "it clears the rebuild flag"          ($_guard -match '\$shouldRebuild = \$false')
Check "it does not raise force-reinstall"   (-not ($_guard -match 'PinChangedForceReinstall = \$true'))
Check "it does not wipe the venv"           (-not ($_guard -match 'Remove-Item'))
# A kept xpu venv needs the xpu index lock, or triton-windows lands over torch's XPU triton.
Check "a kept xpu venv still locks the xpu index" (
    $_guard -match 'if \(\$installedTorchTag -eq "xpu"\) \{ \$script:PreservedXpuVenv = \$true \}')
# Two install arms force torch back regardless of PinChangedForceReinstall, so the kept family
# must reach index selection, declared outside the venv block.

# A GPU wheel in the venv is not proof this run chose it (legacy-layout migration leaves old
# wheels), so install.ps1 reports its family and only a matching wheel is preserved.
$installText = (Get-Content -Raw (Join-Path $root "install.ps1")) -replace "`r`n", "`n"
Check "install.ps1 reports the family it settled on" (
    $installText -match '\$env:UNSLOTH_INSTALLER_TORCH_TAG = if \(\$SkipTorch\) \{ "" \} else \{')
Check "it reports the resolved flavor, not the raw index URL" (
    $installText -match '\[string\]\(Get-ExpectedTorchFlavorTag -TorchIndexUrl \$TorchIndexUrl -ROCmIndexUrl \$ROCmIndexUrl\)')
# Assigned unconditionally so a second install in the same session does not inherit it; ""
# clears it on every edition. The -match half is the anti-vacuity half.
Check "the report is assigned on every run, not only when known" (
    ($installText -match '(?m)^\s*\$env:UNSLOTH_INSTALLER_TORCH_TAG = ') -and
    ($installText -notmatch 'if \([^\n]*\) \{\s*\$env:UNSLOTH_INSTALLER_TORCH_TAG = (?!\$previous)'))
Check "it is handed over before setup is invoked" (
    $installText.IndexOf('$env:UNSLOTH_INSTALLER_TORCH_TAG') -ge 0 -and
    $installText.IndexOf('$env:UNSLOTH_INSTALLER_TORCH_TAG') -lt
    $installText.IndexOf('$studioArgs = @(''studio'', ''setup'')'))
# setup.ps1 and install.ps1 ship separately, so absent or blank means unknown, not mismatch.
Check "setup.ps1 reads the reported family"  (
    $setupText -match '(?m)^\$InstallerTorchTag = if \(\[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_INSTALLER_TORCH_TAG\)\) \{ \$null \}')
Check "an absent report is unknown, not a mismatch" (
    $_guard -match '\(\(-not \$InstallerTorchTag\) -or \$installedTorchTag -eq \$InstallerTorchTag\)')
Check "the read precedes the stale check" (
    $setupText.IndexOf('$InstallerTorchTag = if ([string]::IsNullOrWhiteSpace(') -ge 0 -and
    $setupText.IndexOf('$InstallerTorchTag = if ([string]::IsNullOrWhiteSpace(') -lt
    $setupText.IndexOf('(-not $InstallerTorchTag) -or $installedTorchTag -eq $InstallerTorchTag'))

Check "the kept family is recorded for the index selection" (
    $_guard -match '\$script:PreservedInstallerTorchTag = \$installedTorchTag')
Check "that flag is declared outside the venv block" (
    $setupText -match '(?m)^\$script:PreservedInstallerTorchTag = \$null$')
Check "the declaration precedes the stale check" (
    $setupText.IndexOf('$script:PreservedInstallerTorchTag = $null') -ge 0 -and
    $setupText.IndexOf('$script:PreservedInstallerTorchTag = $null') -lt
    $setupText.IndexOf('$script:PreservedInstallerTorchTag = $installedTorchTag'))
Check "the guard precedes the in-place repair" (
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup -and') -ge 0 -and
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup -and') -lt
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup) {'))
Write-Host ""
Write-Host "--- driven end to end, over setup.ps1's own source"
# The whole decision cascade is lifted from setup.ps1 verbatim and executed, never retyped.
$_cascadePat = '(?s)(    if \(\$shouldRebuild -and \$_pinnedIdx -and \$installedTorchTag\) \{.*?\n    if \(\$shouldRebuild -and \$InstallerManagedSetup\) \{.*?\n    \}\n)'
$_cascade = if ($setupText -match $_cascadePat) { $Matches[1] } else { "" }
Check "the decision cascade was found"     ($_cascade -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_cascadePat))
# Ends on the `else { $CuTag = "cpu" }` brace so a truncated match cannot still assign it.
$_cuTagPat = '(?s)(if \(\$PinnedTorchIndexUrl\) \{\n    \$CuTag = Get-TorchIndexLeaf.*?\n\} else \{\n    \$CuTag = "cpu"\n\}\n)'
$_cuTag = if ($setupText -match $_cuTagPat) { $Matches[1] } else { "" }
Check "the index selection was found"      ($_cuTag -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_cuTagPat))
# These arms force torch back on their own; their extracted conditions are evaluated below.
Check "the AMD arm forces on any non-rocm tag" (
    $setupText -match 'if \(\$installedTorchTag -ne "rocm"\) \{ \$rocmForce = @\("--force-reinstall"\) \}')
Check "the AMD arm no longer forces unconditionally" (
    -not ($setupText -match 'Fast-Install @_rocmTrio --force-reinstall'))
Check "the XPU arm forces on any non-xpu tag" (
    $setupText -match 'if \(\$installedTorchTag -ne "xpu"\) \{ \$xpuForce = @\("--force-reinstall"\) \}')
$_amdGate = if ($setupText -match '(?m)^if \((-not \$TorchIndexPinned -and \(\$HasROCm -or \$ROCmGfxArch\) -and \$CuTag -eq "cpu")\) \{$') { $Matches[1] } else { "" }
$_xpuGate = if ($setupText -match '(?m)^if \((-not \$ROCmIndexUrl -and \$CuTag -eq "xpu")\) \{ \$XpuIndexUrl') { $Matches[1] } else { "" }
Check "the AMD reroute gate was found"     ($_amdGate -ne "")
Check "the XPU arm gate was found"         ($_xpuGate -ne "")

foreach ($srcText in (Get-HelperSources $setup @("Test-CudaFamilyLeaf"))) { Invoke-Expression $srcText }
Check "Test-CudaFamilyLeaf came across"    ((Test-CudaFamilyLeaf "cu128") -and -not (Test-CudaFamilyLeaf "rocm"))
function Get-PytorchCudaTag { return "cu128" }

# One case = one host. InstallerTag "" is no answer, $null an installer too old to say; Has*
# must agree with Expected. Runs at script scope so `$script:` writes and reads share scope.
$_cases = @(
    @{ Name = "rocm wheel the installer chose, rescan says cu128"; Installed = "rocm"; InstallerTag = "rocm"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    @{ Name = "cu128 wheel the installer chose, rescan says rocm";  Installed = "cu128"; InstallerTag = "cu128"; Expected = "rocm";  Nvidia = $false; Xpu = $false; Rocm = $true;  Gfx = "gfx1151"
       Keep = $true;  CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # The xpu promotion is gated on -not $HasNvidiaSmi.
    @{ Name = "xpu wheel the installer chose, rescan says cu128";   Installed = "xpu";   InstallerTag = "xpu"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "xpu";   Amd = $false; XpuArm = $true }
    @{ Name = "rocm wheel the installer chose, rescan says cpu";    Installed = "rocm";  InstallerTag = "rocm"; Expected = "cpu";   Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    @{ Name = "cu128 wheel the installer chose, rescan says cpu";   Installed = "cu128"; InstallerTag = "cu128"; Expected = "cpu";   Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # An installer too old to set the variable must keep preserving its choice.
    @{ Name = "rocm wheel, installer did not say, rescan says cu128"; Installed = "rocm"; InstallerTag = $null; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # Wheels the installer did NOT choose: a stale cu118 from a legacy-layout upgrade.
    @{ Name = "stale cu118 wheel the installer did not choose, rescan says cpu"; Installed = "cu118"; InstallerTag = "cpu"; Expected = "cpu"; Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # Keeping cu118 would also block the ROCm reroute, which needs $CuTag -eq "cpu".
    @{ Name = "stale cu118 wheel the installer did not choose, rescan says rocm"; Installed = "cu118"; InstallerTag = "cpu"; Expected = "rocm"; Nvidia = $false; Xpu = $false; Rocm = $true; Gfx = "gfx1151"
       Keep = $false; CuTag = "cpu";   Amd = $true;  XpuArm = $false }
    @{ Name = "stale rocm wheel the installer did not choose, rescan says cu128"; Installed = "rocm"; InstallerTag = "cpu"; Expected = "cu128"; Nvidia = $true; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # Blank is unknown, like absent, so this preserves.
    @{ Name = "rocm wheel, installer reported blank, rescan says cu128"; Installed = "rocm"; InstallerTag = ""; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # Repairs that must still happen, or the loop is back.
    @{ Name = "cpu wheel on a ROCm host";       Installed = "cpu";   InstallerTag = "rocm"; Expected = "rocm";  Nvidia = $false; Xpu = $false; Rocm = $true;  Gfx = "gfx1151"
       Keep = $false; CuTag = "cpu";   Amd = $true;  XpuArm = $false }
    @{ Name = "cpu wheel on a CUDA host";       Installed = "cpu";   InstallerTag = "cu128"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # A cu* move is a repair and escapes before the guard.
    @{ Name = "cu126 wheel on a cu128 host";    Installed = "cu126"; InstallerTag = "cu128"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    @{ Name = "torch that will not import";     Installed = $null;   InstallerTag = "cu128"; Expected = $null;   Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
)
# Executed from setup.ps1 so the blank-value behaviour matches the running edition.
$_tagReadPat = '(?m)^(\$InstallerTorchTag = if \(\[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_INSTALLER_TORCH_TAG\)\) \{ \$null \}\n\s+else \{[^\n]*\}\n)'
$_tagRead = if ($setupText -match $_tagReadPat) { $Matches[1] } else { "" }
Check "the installer-report read was found" ($_tagRead -ne "")

foreach ($case in $_cases) {
    if ($null -eq $case.InstallerTag) { Remove-Item Env:UNSLOTH_INSTALLER_TORCH_TAG -ErrorAction SilentlyContinue }
    else { $env:UNSLOTH_INSTALLER_TORCH_TAG = $case.InstallerTag }
    # Guarded: Invoke-Expression "" is terminating under "Stop" and would hide every case below.
    if ($_tagRead) { Invoke-Expression $_tagRead }
    else { Remove-Variable -Name InstallerTorchTag -Scope Script -ErrorAction SilentlyContinue }
    $script:shouldRebuild = $true
    $script:InstallerManagedSetup = $true
    $script:installedTorchTag = $case.Installed
    # Left unassigned when the probe failed, as setup.ps1 does.
    if ($null -ne $case.Expected) { $script:expectedTorchTag = $case.Expected }
    else { Remove-Variable -Name expectedTorchTag -Scope Script -ErrorAction SilentlyContinue }
    $script:_pinnedIdx = $null
    $script:_verProbe = $null
    $script:PinChangedForceReinstall = $false
    $script:PreservedXpuVenv = $false
    $script:PreservedInstallerTorchTag = $null
    $script:reason = $null
    Invoke-Expression $_cascade

    $script:PinnedTorchIndexUrl = $null
    $script:TorchIndexPinned = $false
    $script:HasNvidiaSmi = $case.Nvidia
    $script:IsIntelXpu = $case.Xpu
    $script:HasROCm = $case.Rocm
    $script:ROCmGfxArch = $case.Gfx
    Invoke-Expression $_cuTag
    $script:ROCmIndexUrl = $null
    $_amdRuns = [bool](Invoke-Expression $_amdGate)
    if ($_amdRuns) { $script:ROCmIndexUrl = "https://repo.amd.com/rocm/whl/gfx1151/" }
    $_xpuRuns = [bool](Invoke-Expression $_xpuGate)
    $_forced = $script:PinChangedForceReinstall -or $_amdRuns -or ($_xpuRuns -and $installedTorchTag -ne "xpu")

    Write-Host "  case: $($case.Name)"
    Check "    it is not rebuilt from scratch"  (-not $script:shouldRebuild)
    Check "    in-place repair = $(-not $case.Keep)" ($script:PinChangedForceReinstall -eq (-not $case.Keep))
    Check "    index family = $($case.CuTag)"   ($script:CuTag -eq $case.CuTag)
    Check "    AMD arm runs = $($case.Amd)"     ($_amdRuns -eq $case.Amd)
    Check "    XPU arm runs = $($case.XpuArm)"  ($_xpuRuns -eq $case.XpuArm)
    Check "    torch is force-reinstalled = $(-not $case.Keep)" ($_forced -eq (-not $case.Keep))
    # Only when the wheel was kept, or a repaired case drags the old family into the install arms.
    Check "    kept family recorded = $($case.Keep)" (
        [bool]$script:PreservedInstallerTorchTag -eq $case.Keep)
}
Remove-Item Env:UNSLOTH_INSTALLER_TORCH_TAG -ErrorAction SilentlyContinue

# Prove the hazard: reading an unassigned variable throws under Set-StrictMode.
$_strictUnassignedThrew = $false
try {
    & {
        Set-StrictMode -Version Latest
        $installedTorchTag = $null
        $null = ($installedTorchTag -eq "cpu" -or $expectedTorchTag -eq "cpu")
    }
} catch { $script:_strictUnassignedThrew = $true }
Check "reading the unassigned expected tag really does throw" $_strictUnassignedThrew

Write-Host ""
Write-Host "=== the AMD fast-path probe is bounded too ==="
# The rescue keeps a venv whose import hung, so the AMD fast path must be bounded. The leading
# \n and indent matter: two earlier `elseif ($script:ROCmGfxArch) {` would match otherwise.
$_amdFastPat = '(?s)(\n        if \(\$script:ROCmGfxArch\) \{\n.*?reinstalling ROCm PyTorch[^\n]*\n)'
$_amdFast = if ($setupText -match $_amdFastPat) { $Matches[1] } else { "" }
Check "the AMD fast-path escape was found"  ($_amdFast -ne "")
Check "CRLF is normalised, not tolerated"   (-not (($setupText -replace "`n", "`r`n") -match $_amdFastPat))
Check "no bare interpreter call is left"    (-not ($_amdFast -match '&\s*python -c'))
Check "it goes through the bounded probe"   ($_amdFast -match 'Invoke-BoundedPythonProbe -PythonExe "python"')
Check "it still asks torch.cuda.is_available" ($_amdFast -match 'torch\.cuda\.is_available')
# A timed-out probe must read as CPU, the safe direction.
Check "an unanswered probe still reads as CPU" ($_amdFast -match '\$_torchIsCpu = -not \$_rocmTorchProbe\.Ok')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
