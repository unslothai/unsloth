#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A single-AMD-GPU host must not install itself into a loop (#8335).
#
# Two defects met there. WMI found exactly one Radeon, the if-expression holding it unrolled to a
# scalar, that scalar's .Count read $null under Windows PowerShell 5.1, so setup reported "gpu
# none" and judged the installed ROCm venv stale against a required "cpu". The stale branch under
# install.ps1 then aborted, install.ps1's failure path restored the previous environment, and the
# next run reached the same verdict -- which is why that abort has been reported from four
# unrelated triggers (#5942, #7275, #8335, and a driver crash on Discord).
#
# Read this before adding a case, because the obvious repro does not work. It is NOT true that "a
# scalar's .Count is $null on 5.1": a String or an Int32 answers 1 there exactly as on 7, so
# anyone checking the claim that way concludes there is no bug. $null comes back only for objects
# whose PSObject carries no Count of its own -- [pscustomobject], which Microsoft documents, and
# CimInstance, which it does not, and the WMI fallback assigns the second kind. Measured on
# windows-latest, PowerShell 5.1.26100.33158 (Desktop) against a real Get-CimInstance result, with
# pwsh 7.6.4 on the same runner:
#
#   value                       5.1 .Count   7 .Count
#   'a'                         1            1
#   [pscustomobject]@{...}      $null        1
#   CimInstance (one instance)  $null        1
#   @(if (...) { ... })         1            1
#
# So this file adapts to the host: under 5.1 it reproduces #8335 for real on any Windows machine
# with no AMD GPU required, and under pwsh 7 (including the Linux pwsh most contributors run) the
# unroll still happens but the consequence does not, so those cases assert the source shape. Every
# check states which it is.
#
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

# Source text of each named function, Invoke-Expression'd at script scope by the caller (inside a
# function the helpers would be lost on return). Recursive, so it reaches install.ps1's copies too.
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

# Stubs for the installers' printers. Both use Write-Host, so neither pollutes a return value.
function substep { param([string]$Message, [string]$Color = "DarkGray") }
function Write-StudioStdoutMirror { param([string]$Line) }

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== the unroll, and what this host does with it ==="
$psMajor = $PSVersionTable.PSVersion.Major
$is51 = ($psMajor -lt 6)
Write-Host "  host: PowerShell $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition))"

# The unroll itself is not version-specific and happens everywhere, so it is asserted flat.
$oneGpu = @("AMD Radeon PRO W7900")
$unwrapped = if ($oneGpu.Count -gt 0) { $oneGpu } else { @() }
$wrapped = @(if ($oneGpu.Count -gt 0) { $oneGpu } else { @() })
Check "an if-expression unrolls a one-element array" (-not ($unwrapped -is [array]))
Check "the @() wrap keeps it an array"               ($wrapped -is [array])
Check "the wrapped value still counts one GPU"       ($wrapped.Count -eq 1)

# The consequence IS version-specific, and only for some types. Pinning the String case stops this
# being read as "5.1 returns $null for scalars", which is what makes the defect easy to dismiss.
Check "a String scalar answers 1 on every PowerShell" ((("x")).Count -eq 1)
Check "so does the unrolled string array"             ($unwrapped.Count -eq 1)

# The type that bites. [pscustomobject] is the portable CimInstance stand-in: same split, and
# available on the Linux pwsh where most of this suite runs.
$_countless = ([pscustomobject]@{ Name = "AMD Radeon PRO W7900" }).Count
if ($is51) {
    Check "5.1: a Count-less scalar answers `$null"   ($null -eq $_countless)
} else {
    Check "7: a Count-less scalar answers 1"          ($_countless -eq 1)
}

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== #8335 itself, against a real CIM instance ==="
# Windows only, because Get-CimInstance is. Win32_OperatingSystem always returns exactly one
# instance, so it reproduces "exactly one AMD GPU" on any Windows host without one -- which is why
# this defect never showed up in CI. On 5.1 it is a real repro of #8335 and a real regression test
# for the @() wrap. $IsWindows does not exist on 5.1, where the answer is Windows by construction.
if ($is51 -or $IsWindows) {
    $_cim = @(Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue)
    Check "exactly one CIM instance to work with" ($_cim.Count -eq 1)
    if ($_cim.Count -eq 1) {
        # Byte-for-byte the shape of the WMI fallback, old form and new.
        $_old = if ($_cim.Count -gt 0) { $_cim } else { @() }
        $_new = @(if ($_cim.Count -gt 0) { $_cim } else { @() })
        Check "the unwrapped form is a bare CimInstance" (
            $_old -is [Microsoft.Management.Infrastructure.CimInstance])
        Check "the @() wrapped form stays an array"      ($_new -is [array])
        # `if ($wmiGpus.Count -gt 0)` is the line that gates $script:ROCmGpuLabels: the bug and
        # the fix, stated as the guard actually reads.
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

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Test-VenvTorchIsRocm reads the wheel off disk ==="
# The AMD counterpart of Test-VenvTorchIsXpu, so a faulted HIP runtime -- an `import torch` that
# raises at the DLL load or never returns -- is not answered by deleting a good ROCm environment.
# Free by design: no interpreter is launched, so a CPU-only host pays nothing on every update.
foreach ($srcText in (Get-HelperSources $setup @("Test-VenvTorchIsRocm"))) { Invoke-Expression $srcText }
# @() around the call, then [0]: a one-name request returns a one-element array that unrolls to a
# String on the way out, and [0] on a String is its first character. The same unroll, one file over.
$isRocmFn = @(Get-HelperSources $setup @("Test-VenvTorchIsRocm"))[0]
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the version file" ($isRocmFn -match 'version\.py')

# Runs on Linux CI too, so the filesystem cmdlets are shadowed: the helper builds a Windows path
# and this asserts the CONTENT it read, not what the OS resolves.
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
# Read off the indexes, not assumed. repo.amd.com/rocm/whl/<arch>/torch/ -- the only place Windows
# ROCm wheels exist -- publishes a THREE-component label with the arch in the URL, not the version
# (torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl, what the #8335 reporter ended up on);
# download.pytorch.org/whl/rocm6.4 publishes the two-component 2.8.0+rocm6.4, Linux only. So
# "+rocm" is the label that matters and the "+gfx" arm is defensive, asserted here to stay so.
Check "rocm6.4 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'")
Check "rocm7.0 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+rocm7.0'")
# Three-part, so also the case a two-part `+rocm\d+\.\d+$` anchor would miss.
Check "rocm7.13.0 wheel (#8335)" (Invoke-IsRocm "C:\v" "__version__ = '2.11.0+rocm7.13.0'")
Check "gfx1151 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+gfx1151'")
Check "gfx110X-all wheel"    (Invoke-IsRocm "C:\v" "__version__ = '2.7.1+gfx110X.all'")
# A dev/nightly release segment sits before the local label, so it must not shift the match.
Check "nightly rocm wheel"   (Invoke-IsRocm "C:\v" "__version__ = '2.12.0.dev20260801+rocm7.2'")
Check "nightly cuda wheel"   (-not (Invoke-IsRocm "C:\v" "__version__ = '2.12.0.dev20260801+cu130'"))
# A source build carries a git hash where the flavour would be, so the label alone cannot answer:
# the `hip` field decides. Both shapes are real -- torch annotates it on current releases and did
# not on older ones -- and AMD's own builds ship exactly this (2.5.0a0+git1234567, hip 6.2.41134).
Check "source build, no hip" (-not (Invoke-IsRocm "C:\v" "__version__ = '2.9.0a0+git1a2b3c'"))
Check "amd build, hip set"   (Invoke-IsRocm "C:\v" "__version__ = '2.5.0a0+git1234567'`nhip: Optional[str] = '6.2.41134'")
Check "unannotated hip"      (Invoke-IsRocm "C:\v" "__version__ = '2.5.0a0+git1234567'`nhip = '6.2.41134'")
# Quoted and non-empty, or `hip: Optional[str] = None` would make every venv a ROCm one.
Check "empty hip"            (-not (Invoke-IsRocm "C:\v" "__version__ = '2.9.0a0+git1a2b3c'`nhip: Optional[str] = ''"))
Check "cuda wheel"           (-not (Invoke-IsRocm "C:\v" $CU))
Check "xpu wheel"            (-not (Invoke-IsRocm "C:\v" $XPU))
Check "untagged wheel"       (-not (Invoke-IsRocm "C:\v" $BARE))
Check "cpu wheel"            (-not (Invoke-IsRocm "C:\v" "__version__ = '2.10.0+cpu'"))
# git_version is a real line in torch/version.py and can name a branch. Only __version__ decides.
Check "gfx in git_version"   (-not (Invoke-IsRocm "C:\v" ($CU + "`ngit_version = 'rocm-branch-gfx1100'")))
# Every CUDA and CPU build's version.py carries `hip: Optional[str] = None`, so an unquoted value
# must not count, or a CUDA venv would be kept as ROCm forever.
Check "hip attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`nhip: Optional[str] = None")))
Check "gfx attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`ngfx = 'gfx1100'")))
Check "no torch installed"   (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Missing))
Check "unreadable file"      (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Throws))
Check "no venv path"         (-not (Invoke-IsRocm "" "__version__ = '2.8.0+rocm6.4'"))

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Invoke-BoundedPythonProbe keeps the reason it failed ==="
# Both installers carry a copy, so both are driven. The probe drained stderr and threw it away, so
# "the HIP DLLs will not load", "torch is not installed" and "the import never came back" arrived
# at the caller as one silent False -- and the caller deletes the environment over that answer.
# The bounding and the async drain must survive intact, so a real child process is launched.
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

        # The shape of a driver fault: torch imports, the HIP DLLs do not load, python exits
        # nonzero with the real cause on stderr and nothing on stdout.
        $bad = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'import sys; sys.stderr.write(chr(91) + chr(87) + chr(105) + chr(110) + chr(69) + chr(114) + chr(114) + chr(111) + chr(114) + chr(32) + chr(49) + chr(50) + chr(54) + chr(93)); sys.exit(1)'
        Check "a failed probe still reads as not Ok" (-not $bad.Ok)
        Check "the stderr text survives"             ($bad.Error -match 'WinError 126')
        # Keeping stderr must not have changed which stream is which.
        Check "stderr does not leak into Output"     (-not ($bad.Output -match 'WinError'))

        # The case the helper exists for: still bounded, and now says so.
        $slow = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'import time; time.sleep(45)' -TimeoutSec 1
        Check "a hung probe is still bounded"        (-not $slow.Ok)
        Check "a hung probe says it timed out"       ($slow.Error -match 'did not answer within')

        Check "an empty code string is refused"      (-not (Invoke-BoundedPythonProbe -PythonExe $py.Source -Code '').Ok)
    }
    # No interpreter at that path: Process.Start throws, and its text is all there is to report.
    $gone = Invoke-BoundedPythonProbe -PythonExe (Join-Path $root "no-such-python-XYZ") -Code 'print(1)'
    Check "a missing interpreter reads as not Ok"    (-not $gone.Ok)
    Check "a missing interpreter reports why"        (-not [string]::IsNullOrWhiteSpace($gone.Error))
}

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== source shapes ==="
# Normalised to LF once rather than per pattern: on a CRLF checkout every \n-anchored pattern below
# matches nothing and the -not checks over an empty region pass vacuously. A real incident, not a
# hypothetical (see test_setup_xpu_runtime_prereport.ps1).
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"

Write-Host "the AMD WMI fallback survives a single-GPU host"
# The trailing `\n` is what makes the CRLF check below bite: on a CRLF checkout the line ends
# `\r\n`, so the pattern fails rather than matching a silently empty region. `[^\r\n]*` absorbs
# whatever guards the assignment (#8577 wrapped it in `if (-not $HasROCm) { ... }`) WITHOUT
# absorbing the `\r`, which would hand the CRLF check a match and retire it silently.
$_wmiPat = '(?s)(\$amdGpus = @\(Get-CimInstance Win32_VideoController.*?\$ROCmGpuLabel = \$script:ROCmGpuLabels\[0\][^\r\n]*\n)'
$_wmi = if ($setupText -match $_wmiPat) { $Matches[1] } else { "" }
Check "the WMI fallback was found"        ($_wmi -ne "")
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $_wmiPat))
# This is the #8335 assertion. It is a shape, not a behaviour: see the header.
Check "the healthy/all choice is @() wrapped" (
    $_wmi -match '\$wmiGpus = @\(if \(\$healthyGpus\.Count -gt 0\) \{ \$healthyGpus \} else \{ \$amdGpus \}\)')
Check "the unwrapped form is gone"        (-not ($_wmi -match '\$wmiGpus = if \('))
# Same reason for the two lists it chooses between: Where-Object also returns a scalar on one match.
Check "the adapter list is @() wrapped"   ($_wmi -match '\$amdGpus = @\(Get-CimInstance')
Check "the healthy list is @() wrapped"   ($_wmi -match '\$healthyGpus = @\(\$amdGpus \| Where-Object')

Write-Host "a driver fault does not delete a ROCm venv"
# Ends on `\{\n`, not on the comment text, for the same CRLF reason as above.
$_rescuePat = '(?s)(\$_verProbe = Invoke-BoundedPythonProbe.*?\} else \{\n        # Missing python\.exe means)'
$_rescue = if ($setupText -match $_rescuePat) { $Matches[1] } else { "" }
Check "the probe chain was found"          ($_rescue -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_rescuePat))
Check "an unreadable flavour is rescued off disk" ($_rescue -match 'elseif \(Test-VenvTorchIsRocm -VenvPath \$VenvDir\)')
Check "the rescue classifies it as rocm"   ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm.*?\$installedTorchTag = "rocm"')
# It must NOT set the rebuild flag, or the whole point is lost.
Check "the rescue never sets shouldRebuild" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm[^}]*\$shouldRebuild = \$true'))
# The disk read is the point: a second interpreter after the first hung is what it avoids.
Check "the rescue launches no interpreter" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm.*?Invoke-BoundedPythonProbe'))
# Matched on the substep ARGUMENT, not the region: the comment above the branch names the driver
# too, so a loose match stays green with the message itself deleted.
Check "the rescue names the driver fix"    ($_rescue -match 'substep "[^"]*Adrenalin')
# Non-AMD, non-Intel families must still rebuild on an unreadable flavour, exactly as before.
Check "other families still rebuild"       ($_rescue -match '(?s)\} else \{\s*\$shouldRebuild = \$true')
# The XPU rescue this one is modelled on has to stay, and ahead of it: an Arc box names a
# different driver.
Check "the XPU rescue is untouched"        ($_rescue -match 'elseif \(Test-VenvTorchIsXpu -VenvPath \$VenvDir\)')
Check "XPU is judged before ROCm"          (
    $_rescue.IndexOf('Test-VenvTorchIsXpu -VenvPath') -ge 0 -and
    $_rescue.IndexOf('Test-VenvTorchIsXpu -VenvPath') -lt $_rescue.IndexOf('Test-VenvTorchIsRocm -VenvPath'))

Write-Host "the stale-venv decision has no dead end left in it"
# Starts on `\$null\n`, so a CRLF checkout fails the match instead of returning an empty region.
$_repairPat = '(?s)(\$reason = \$null\n.*?Remove-Item -LiteralPath \$VenvDir -Recurse)'
$_repair = if ($setupText -match $_repairPat) { $Matches[1] } else { "" }
Check "the stale-venv decision was found"  ($_repair -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_repairPat))
# The abort itself. install.ps1 is the caller under $InstallerManagedSetup, it moved the previous
# environment aside earlier in the same run and its failure path moves it straight back, so exiting
# here landed on the byte-identical starting state and nothing in that cycle can converge. Bounded
# to the branch: the delete below has two legitimate Exit-SetupFailure calls of its own.
$_managedPat = '(?s)(if \(\$shouldRebuild -and \$InstallerManagedSetup\) \{.*?\n    \}\n)'
$_managed = if ($_repair -match $_managedPat) { $Matches[1] } else { "" }
Check "the installer-managed branch was found" ($_managed -ne "")
Check "an installer-managed run does not abort" (-not ($_managed -match 'Exit-SetupFailure'))
# Matched on the Write-StudioLine ARGUMENT, so the comment above the branch, which quotes the old
# advice to explain why it went, cannot keep this green.
Check "the advice that pointed at itself is gone" (
    -not ($setupText -match 'Write-StudioLine "\s*Re-run install\.ps1'))
Check "the abort message is gone too"      (-not ($setupText -match 'The existing Unsloth environment needs repair'))
# What replaces it: the in-place repair an index-pin change already takes, honoured by every arm.
Check "an installer-managed run repairs in place" ($_managed -match '\$script:PinChangedForceReinstall = \$true')
Check "and clears the rebuild flag"        ($_managed -match '\$shouldRebuild = \$false')
# Deleting was never available here: install.ps1 runs setup through the venv's own unsloth.exe.
Check "it does not try to wipe the venv it runs from" (-not ($_managed -match 'Remove-Item'))
# A direct `unsloth studio update` keeps its own self-repair, and the custom-home guard stays in
# front of the delete.
Check "a direct update still rebuilds"     ($_repair -match 'Stale venv detected \(\$reason\) -- rebuilding')
Check "the custom-home guard still gates the wipe" ($_repair -match '\$StudioHomeIsCustom')
# Why it failed, not just that it failed: this line separates a faulted GPU driver from a missing
# wheel, and the user is about to be told what happened to their environment.
Check "the swallowed probe error is surfaced" ($_repair -match '\$_verProbe\.Error')
Check "the probe handle is declared up front" ($setupText -match '(?m)^\s*\$_verProbe = \$null')
# Read by four install arms and raised by the repair above, but assigned only inside the
# venv-exists block, so on a fresh install every one of those reads is of a variable that was never
# created: harmless bare (falsy is the wanted answer), fatal under a caller's Set-StrictMode.
Check "the force-reinstall flag is declared outside the venv block" (
    $setupText -match '(?m)^\$script:PinChangedForceReinstall = \$false$')

Write-Host "the surfaced probe error survives a stderr that is only whitespace"
# The guard is `-and $_verProbe.Error`, and a stderr of blank lines passes it while Where-Object
# drops every one, so [0] into what is left is fatal under a caller's Set-StrictMode -- and
# studio/setup.bat launches setup WITHOUT -NoProfile, so a profile can set one. Driven for real
# rather than asserted as a shape, with the replaced form as a control.
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
# Without this the check above would pass on any rewrite, including one that never had the bug.
Check "the @(...)[0] form it replaced really did throw" $_strictOldThrew
Check "setup.ps1 no longer indexes that pipeline" (
    -not ($setupText -match 'Select-Object -Last 1\)\[0\]'))
# A real one-line stderr must still be picked up, or the whole point of keeping it is lost.
$_realErr = "Traceback (most recent call last):`r`nOSError: [WinError 126] The specified module could not be found"
$_realLine = $_realErr -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1
Check "a real stderr still yields its last line" ($_realLine -match 'WinError 126')

Write-Host ""
Write-Host "=== an interpreter-less venv is not repaired in place, it is refused ==="
# The one state the in-place repair made WORSE than the abort it replaced. A venv directory with no
# Scripts\python.exe is incomplete, not stale -- no interpreter to force-reinstall torch through --
# and the abort used to catch it by accident, because it caught every stale verdict. Without a
# guard it now reaches the activation, a dot-source of a path that does not exist, which is NOT a
# terminating error at the "Continue" the pip section runs at: setup prints one red line and keeps
# going, every `python` / `uv pip` after it resolves against whatever is on PATH, the whole stack
# lands outside the venv and the run can still exit 0. Prove the hazard before asserting the guard.
$_dotSourceKeptGoing = $false
& {
    $ErrorActionPreference = "Continue"
    . (Join-Path $root "no-such-Activate-XYZ.ps1")
    $script:_dotSourceKeptGoing = $true
} 2>$null
Check "dot-sourcing a missing script does not stop the script" $_dotSourceKeptGoing

# Bounded to the reuse branch: the "not found" branch above has an Exit-SetupFailure of its own,
# and matching the whole region would pass on that one. Both ends sit immediately after a
# non-newline token, so a CRLF checkout fails the match instead of yielding an empty region.
$_reusePat = '(?s)(substep "reusing existing virtual environment at \$VenvDir"\n.*?Exit-SetupFailure "No interpreter at [^\n]*\n)'
$_reuse = if ($setupText -match $_reusePat) { $Matches[1] } else { "" }
Check "the reuse branch was found"         ($_reuse -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_reusePat))
Check "it refuses a venv with no interpreter" ($_reuse -match '(?s)\} else \{.*?Exit-SetupFailure "No interpreter at')
# It must fail BEFORE the dot-source, or the hazard proven above is still live.
Check "the refusal precedes the activation" (
    $setupText.IndexOf('Exit-SetupFailure "No interpreter at') -lt
    $setupText.IndexOf('$ActivateScript = Join-Path $VenvDir'))
# Not a wipe: this venv may be the only place the previous one's contents still are.
Check "it does not delete the venv"        (-not ($_reuse -match 'Remove-Item'))
# The message has to say incomplete, not stale, or it reads as the loop-causing advice again.
Check "it says incomplete, not out of date" ($_reuse -match 'incomplete rather than out of date')
# A healthy venv must be untouched by all of this.
Check "a venv with an interpreter still just prints its version" (
    $_reuse -match '(?s)if \(Test-Path -LiteralPath \$_venvPyExe\) \{.*?--version')

# The interpreter is not the only file that has to be there. Everything after this reaches the venv
# through the dot-sourced Activate.ps1 and a bare `python` (Fast-Install resolves its target with
# (Get-Command python).Source), and install.ps1 leaves the venv's Scripts directory off PATH on
# purpose, so a venv that kept python.exe but lost Activate.ps1 hits the SAME hazard proven above,
# one file over. Newly reachable, because a stale verdict now repairs where it used to abort.
Check "it refuses a venv with no activation script" (
    $_reuse -match 'Exit-SetupFailure "No activation script at')
Check "the activation script path is built next to the interpreter" (
    $_reuse -match '\$_venvActivate = Join-Path \$VenvDir "Scripts\\Activate\.ps1"')
# The -ge 0 is not decoration: IndexOf answers -1 when the refusal is not there, and -1 is less
# than every other offset, so the ordering alone passes on a tree that never grew it.
Check "that refusal also precedes the activation" (
    $setupText.IndexOf('Exit-SetupFailure "No activation script at') -ge 0 -and
    $setupText.IndexOf('Exit-SetupFailure "No activation script at') -lt
    $setupText.IndexOf('$ActivateScript = Join-Path $VenvDir'))
# Checked on the arm where the interpreter EXISTS, or it never fires on its own.
Check "the check sits inside the interpreter-present arm" (
    $_reuse -match '(?s)if \(Test-Path -LiteralPath \$_venvPyExe\) \{.*?Exit-SetupFailure "No activation script at.*?\} else \{')
Check "it does not delete that venv either" (-not ($_reuse -match 'Remove-Item'))

Write-Host ""
Write-Host "=== a present activation script is not the same as an activated venv ==="
# Both refusals above check that a FILE is there; neither can tell whether dot-sourcing it took
# effect. Activate.ps1 prepends the venv to PATH in its very last statement, 28 lines after it sets
# $env:VIRTUAL_ENV, so a copy truncated by an interrupted `python -m venv` runs to its last
# complete statement and returns having changed nothing that matters, and an unparseable one is a
# ParserError, non-terminating at the "Continue" the pip section runs at. Prove both hazards live
# before asserting the guard, exactly as the missing-script case above is proven.
$_actRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-activate-" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $_actRoot -Force | Out-Null

    $_truncated = Join-Path $_actRoot "truncated.ps1"
    # Everything Activate.ps1 does BEFORE its final PATH line, and nothing after it.
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
    # The worst part of this one: there is no red line to notice, unlike the missing-file case.
    Check "and raises no error at all" ($Error.Count -eq 0)
    Check "and leaves PATH exactly as it found it" ($env:PATH -eq $_pathBefore)
    # It still set the state a VIRTUAL_ENV check would have trusted, which is why the guard looks
    # at the resolved interpreter instead.
    Check "while still setting the state that precedes the PATH line" ($env:UNSLOTH_ACTIVATE_PROBE -eq "reached")

    $env:UNSLOTH_ACTIVATE_PROBE = $null

    # The unparseable case has to run in a child host, in setup.ps1's actual shape: script-scope
    # "Continue" and no enclosing try. A ParserError is non-terminating there, but inside a try
    # under this file's "Stop" it converts and unwinds, proving the opposite. setup.ps1 is itself
    # launched as a fresh host process, so the child is the faithful reproduction, not a dodge.
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
    # The corrupt script writes its parse error to stderr, and Windows PowerShell 5.1 wraps a
    # native command's stderr in a NativeCommandError and applies $ErrorActionPreference to it,
    # so the "Stop" set at the top of this file would end the run here. 7.1 stopped applying the
    # preference to native stderr, which is why this passes under pwsh and only fails on the 5.1
    # leg. Scope the preference to the call; the exit code is what this actually judges, and it is
    # read on the next line.
    $_childOut = & {
        $ErrorActionPreference = "Continue"
        & $_hostExe -NoProfile -File $_child 2>$null
    } | Out-String
    $_childRc = $LASTEXITCODE
    Check "an unparseable activation script does not stop the script either" ($_childOut -match 'SURVIVED')
    Check "and it leaves PATH untouched too" ($_childOut -match 'PATH-UNTOUCHED')
    # The whole point: the run can still report success over an environment it never entered.
    Check "and the run still exits 0" ($_childRc -eq 0)

    # ── the guard itself, executed ──
    # Caught rather than thrown, so a tree without the guard reports every check below as FAIL
    # instead of unwinding at the first one and hiding the rest.
    $_assertFn = try { @(Get-HelperSources $setup @("Assert-VenvActivated"))[0] } catch { "" }
    # An empty or wrong extraction would make every case below pass vacuously.
    Check "extraction kept the interpreter lookup" ($_assertFn -match 'Get-Command python')
    # A VIRTUAL_ENV check is the tempting wrong answer, and the truncation above is why it fails.
    # The -ne "" is the anti-vacuity half: -not on an empty region passes against no guard at all.
    Check "the guard does not settle for VIRTUAL_ENV" (($_assertFn -ne "") -and -not ($_assertFn -match 'VIRTUAL_ENV'))

    # Real directories rather than a Get-Item stub: separators, trailing separators and case
    # folding are the whole risk, and a stub would just re-implement the bug.
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
    # The state this section exists for: activation no-opped, so `python` is still the ambient one.
    Check "an ambient interpreter is refused" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy) -match 'did not put its interpreter on PATH')
    Check "and the refusal names where python actually resolved" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy) -match 'WindowsApps')
    Check "no python at all is refused" (
        (Invoke-AssertVenv -VenvDir $_venvOk -NoPython) -match 'did not put its interpreter on PATH')
    # A prefix compare without a separator would call venv2 a child of venv.
    Check "a sibling venv is not mistaken for this one" (
        (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_venvSiblingPy) -match 'did not put its interpreter on PATH')

    # ── the false-positive side, which matters more than the true-positive side ──
    # A guard that refuses a working install is worse than the bug it closes.
    Check "a trailing separator on the venv path still passes" (
        $null -eq (Invoke-AssertVenv -VenvDir ($_venvOk + [System.IO.Path]::DirectorySeparatorChar) -PythonSource $_venvOkPy))
    # Windows paths are case-insensitive, and the two sides can disagree on case.
    Check "a differently cased venv path still passes" (
        $null -eq (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource ($_venvOkPy.ToUpperInvariant())))
    # An alias or function named python carries no Source, so nothing can be proven about it.
    Check "a non-application python is left alone" (
        $null -eq (Invoke-AssertVenv -VenvDir $_venvOk -PythonSource $_ambientPy -CommandType 'Function'))
    # An unresolvable venv root leaves the guard nothing to stand on, and the two refusals above
    # already cover a venv that is not there.
    Check "an unresolvable venv path is left alone" (
        $null -eq (Invoke-AssertVenv -VenvDir (Join-Path $_actRoot "no-such-venv") -PythonSource $_ambientPy))
} finally {
    Remove-Item -LiteralPath $_actRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# ── where it is wired in ──
# Both ends sit immediately after a non-newline token, so a CRLF checkout fails the match rather
# than yielding an empty region every -not check would pass against. The dot-source is reached
# through Enter-StudioVenv, which picks it or the staged-root PATH activation, so the region
# spans that helper rather than requiring the dot-source on the very next line.
$_actPat = '(?s)(\$ActivateScript = Join-Path \$VenvDir "Scripts\\Activate\.ps1"\n.*?\. \$ActivateScript\n.*?Assert-VenvActivated -VenvDir \$VenvDir\n)'
$_act = if ($setupText -match $_actPat) { $Matches[1] } else { "" }
Check "the activation block was found"     ($_act -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_actPat))
# A post-condition, so it has to come AFTER the dot-source, not alongside the two pre-conditions.
Check "the assertion follows the dot-source" (
    $setupText.IndexOf('. $ActivateScript') -ge 0 -and
    $setupText.IndexOf('. $ActivateScript') -lt
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir'))
# Nothing between here and Fast-Install re-checks, and Fast-Install is where a wrong interpreter
# starts receiving the stack.
Check "it lands before Fast-Install resolves python" (
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir') -ge 0 -and
    $setupText.IndexOf('Assert-VenvActivated -VenvDir $VenvDir') -lt
    $setupText.IndexOf('function Fast-Install'))
# The re-activation after Refresh-Environment sits inside a catch that swallows everything, so
# the second call site is not a duplicate of the first.
Check "the re-activation is covered as well" (
    ([regex]::Matches($setupText, [regex]::Escape('Assert-VenvActivated -VenvDir $VenvDir'))).Count -ge 2)
Check "and that one is outside the swallowing catch" (
    $setupText.IndexOf('} catch { }') -ge 0 -and
    $setupText.IndexOf('} catch { }') -lt
    $setupText.LastIndexOf('Assert-VenvActivated -VenvDir $VenvDir'))
# It refuses; it does not repair, and it does not wipe a venv install.ps1 may still restore from.
# Env: is exempt: the staged branch clears PYTHONHOME the way Activate.ps1 does, and that removes
# a variable, not a file.
Check "the guard does not delete anything" (
    ($_act -ne "") -and -not ($_act -match 'Remove-Item(?!\s+Env:)'))

Write-Host ""
Write-Host "=== an installer-managed repair never moves a GPU wheel to another family ==="
# install.ps1 resolves the index and installs the torch trio ITSELF, minutes before it invokes
# setup, which then probes the hardware again from scratch. When that second probe fails (a
# Get-CimInstance that throws, an nvidia-smi that does not answer, the single-Radeon unroll at the
# top of this file) setup lands somewhere else, reads the +cu / +rocm / +xpu wheel install.ps1 just
# placed as stale, and the in-place repair --force-reinstalls the other family over it. Then setup
# exits 0, install.ps1 counts the run a success and drops the rollback copy: the abort this repair
# replaced at least failed loudly, and committing a wrong install silently is a worse trade.
#
# "cpu" was only the first direction. The same disagreement runs GPU-to-GPU on any box holding two
# vendors' cards, where one scan answers in install.ps1 and the other in setup: +rocm read as
# cu128, +cu128 read as rocm, +xpu read as either. install.ps1 has already reconciled the venv
# against the index it chose, so a GPU wheel sitting here IS its answer.
$_guardPat = '(?s)(if \(\$shouldRebuild -and \$InstallerManagedSetup -and\n.*?\n    \}\n)'
$_guard = if ($setupText -match $_guardPat) { $Matches[1] } else { "" }
Check "the downgrade guard was found"       ($_guard -ne "")
Check "CRLF is normalised, not tolerated"   (-not (($setupText -replace "`n", "`r`n") -match $_guardPat))
# The exact condition, because each term is load-bearing.
Check "it only fires under the installer"   ($_guard -match '\$InstallerManagedSetup')
Check "it only fires on a GPU wheel"        ($_guard -match '\$installedTorchTag -and \$installedTorchTag -ne "cpu"')
# NOT narrowed to a cpu rescan: that spelling left every GPU-to-GPU direction running straight
# into the in-place repair below.
Check "it is not narrowed to a cpu rescan"  (-not ($_guard -match '\$expectedTorchTag -eq "cpu"'))
# $expectedTorchTag is assigned only inside the `if (-not $shouldRebuild)` block above, so on a
# venv whose torch would not import it was never created and reading it in the CONDITION would be
# fatal under a caller's Set-StrictMode. In the body it is reached only after $installedTorchTag
# has answered, which is what makes the message safe.
Check "the condition never reads the expected tag" (
    -not (($_guard -split '\{', 2)[0] -match '\$expectedTorchTag'))
Check "it clears the rebuild flag"          ($_guard -match '\$shouldRebuild = \$false')
# No --force-reinstall is raised, so the arm below leaves the GPU wheel in place (a +cu / +rocm /
# +xpu build already satisfies a bare torch>= range).
Check "it does not raise force-reinstall"   (-not ($_guard -match 'PinChangedForceReinstall = \$true'))
Check "it does not wipe the venv"           (-not ($_guard -match 'Remove-Item'))
# A kept xpu venv needs the same index lock the direct-update escape takes, or triton-windows
# lands over torch's XPU triton with nothing to swap it back.
Check "a kept xpu venv still locks the xpu index" (
    $_guard -match 'if \(\$installedTorchTag -eq "xpu"\) \{ \$script:PreservedXpuVenv = \$true \}')
# Keeping the wheel is not enough on its own: two install arms a thousand lines below force torch
# back regardless of $script:PinChangedForceReinstall, so the kept family has to reach the index
# selection. Same treatment as $script:PreservedXpuVenv, declaration outside the venv block
# included.

# ── the half of the guard that keeps it from preserving a wheel nobody chose ──
# "There is a GPU wheel in the venv" is not evidence that THIS install run put it there:
# install.ps1's migrated-venv arm installs unsloth alone and never touches torch, and its flavor
# repair no-ops whenever its own expected tag is 'cpu' or unrecognised, so an ordinary upgrade off
# the legacy ~/.unsloth/studio/.venv layout hands setup whatever wheel the previous install left,
# on whatever hardware that was. Preserving THAT is a permanently wrong environment setup then
# refuses to repair, and on a mapped AMD host the kept cu* tag also blocks the ROCm reroute (it
# needs $CuTag -eq "cpu"). So install.ps1 says which family it settled on, and only a wheel
# matching it counts as its answer.
$installText = (Get-Content -Raw (Join-Path $root "install.ps1")) -replace "`r`n", "`n"
Check "install.ps1 reports the family it settled on" (
    $installText -match '\$env:UNSLOTH_INSTALLER_TORCH_TAG = if \(\$SkipTorch\) \{ "" \} else \{')
Check "it reports the resolved flavor, not the raw index URL" (
    $installText -match '\[string\]\(Get-ExpectedTorchFlavorTag -TorchIndexUrl \$TorchIndexUrl -ROCmIndexUrl \$ROCmIndexUrl\)')
# Unconditional, like $env:UNSLOTH_NO_TORCH next to it: a second install in the same PowerShell
# session must not inherit the first one's answer. Assigning "" clears it on every edition (7.5+
# keeps a present blank value, 5.1 and 7.0-7.4 remove the variable) and setup reads both as
# unknown. The -match half is the anti-vacuity half: a bare -notmatch passes against a tree that
# never grew the assignment at all.
Check "the report is assigned on every run, not only when known" (
    ($installText -match '(?m)^\s*\$env:UNSLOTH_INSTALLER_TORCH_TAG = ') -and
    ($installText -notmatch 'if \([^\n]*\) \{\s*\$env:UNSLOTH_INSTALLER_TORCH_TAG'))
Check "it is handed over before setup is invoked" (
    $installText.IndexOf('$env:UNSLOTH_INSTALLER_TORCH_TAG') -ge 0 -and
    $installText.IndexOf('$env:UNSLOTH_INSTALLER_TORCH_TAG') -lt
    $installText.IndexOf('$studioArgs = @(''studio'', ''setup'')'))
# setup.ps1 ships in the pip package and install.ps1 is fetched from unsloth.ai, so the two can be
# different ages. An absent variable therefore means "unknown", not "mismatch", or a cached older
# installer would force-reinstall over its own correct choice. Blank counts as absent on 7.5+.
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
# ...and it has to run BEFORE the repair, or the repair has already fired.
Check "the guard precedes the in-place repair" (
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup -and') -ge 0 -and
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup -and') -lt
    $setupText.IndexOf('if ($shouldRebuild -and $InstallerManagedSetup) {'))
Write-Host ""
Write-Host "--- driven end to end, over setup.ps1's own source"
# A hand-copied simulation drifts and keeps passing after the file it claims to test changed,
# which is exactly how the cpu-only spelling survived review. So the WHOLE decision -- pin escape,
# cu*-to-cu* escape, direct-update xpu escape, this guard, the in-place repair -- is lifted out of
# setup.ps1 verbatim and executed, as is the index selection. Nothing below is retyped.
$_cascadePat = '(?s)(    if \(\$shouldRebuild -and \$_pinnedIdx -and \$installedTorchTag\) \{.*?\n    if \(\$shouldRebuild -and \$InstallerManagedSetup\) \{.*?\n    \}\n)'
$_cascade = if ($setupText -match $_cascadePat) { $Matches[1] } else { "" }
Check "the decision cascade was found"     ($_cascade -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_cascadePat))
# Ends on the `else { $CuTag = "cpu" }` brace, so a truncated match cannot leave a region that
# still assigns $CuTag.
$_cuTagPat = '(?s)(if \(\$PinnedTorchIndexUrl\) \{\n    \$CuTag = Get-TorchIndexLeaf.*?\n\} else \{\n    \$CuTag = "cpu"\n\}\n)'
$_cuTag = if ($setupText -match $_cuTagPat) { $Matches[1] } else { "" }
Check "the index selection was found"      ($_cuTag -ne "")
Check "CRLF is normalised, not tolerated"  (-not (($setupText -replace "`n", "`r`n") -match $_cuTagPat))
# The two install arms that force torch back on their own. Their CONDITIONS are extracted and
# evaluated below, so "the kept wheel survives" is answered by the shipped gates. The AMD arm's
# --force-reinstall is unconditional inside its block; the XPU arm's is keyed off the installed tag.
Check "the AMD arm still force-reinstalls unconditionally" (
    $setupText -match 'Fast-Install @_rocmTrio --force-reinstall --index-url \$ROCmIndexUrl')
Check "the XPU arm forces on any non-xpu tag" (
    $setupText -match 'if \(\$installedTorchTag -ne "xpu"\) \{ \$xpuForce = @\("--force-reinstall"\) \}')
$_amdGate = if ($setupText -match '(?m)^if \((-not \$TorchIndexPinned -and \(\$HasROCm -or \$ROCmGfxArch\) -and \$CuTag -eq "cpu")\) \{$') { $Matches[1] } else { "" }
$_xpuGate = if ($setupText -match '(?m)^if \((-not \$ROCmIndexUrl -and \$CuTag -eq "xpu")\) \{ \$XpuIndexUrl') { $Matches[1] } else { "" }
Check "the AMD reroute gate was found"     ($_amdGate -ne "")
Check "the XPU arm gate was found"         ($_xpuGate -ne "")

# The real Test-CudaFamilyLeaf, because the index selection calls it on the preserved tag.
foreach ($srcText in (Get-HelperSources $setup @("Test-CudaFamilyLeaf"))) { Invoke-Expression $srcText }
Check "Test-CudaFamilyLeaf came across"    ((Test-CudaFamilyLeaf "cu128") -and -not (Test-CudaFamilyLeaf "rocm"))
function Get-PytorchCudaTag { return "cu128" }

# One case = one host. Installed is what setup found in the venv; InstallerTag is the family
# install.ps1 reported ("" is the installer with no answer, $null an installer too old to say);
# Expected is what setup's rescan concluded, and the Has* values are that same rescan, so they have
# to agree with it. Run at script scope (Invoke-Expression inside a function would put the
# extracted `$script:` writes and the reads in different scopes).
$_cases = @(
    # +rocm venv, rescan found the GeForce in the same box and not the Radeon.
    @{ Name = "rocm wheel the installer chose, rescan says cu128"; Installed = "rocm"; InstallerTag = "rocm"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # +cu128 venv, nvidia-smi did not answer this time and the Radeon did.
    @{ Name = "cu128 wheel the installer chose, rescan says rocm";  Installed = "cu128"; InstallerTag = "cu128"; Expected = "rocm";  Nvidia = $false; Xpu = $false; Rocm = $true;  Gfx = "gfx1151"
       Keep = $true;  CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # +xpu venv on an Arc + GeForce box: the promotion above is gated on -not $HasNvidiaSmi.
    @{ Name = "xpu wheel the installer chose, rescan says cu128";   Installed = "xpu";   InstallerTag = "xpu"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "xpu";   Amd = $false; XpuArm = $true }
    # The direction round one closed, kept as a regression: no GPU found at all.
    @{ Name = "rocm wheel the installer chose, rescan says cpu";    Installed = "rocm";  InstallerTag = "rocm"; Expected = "cpu";   Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    @{ Name = "cu128 wheel the installer chose, rescan says cpu";   Installed = "cu128"; InstallerTag = "cu128"; Expected = "cpu";   Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # An installer too old to carry the variable at all -- a normal pairing, since the two ship
    # separately. Unknown has to keep preserving or an older installer loses its own choice.
    @{ Name = "rocm wheel, installer did not say, rescan says cu128"; Installed = "rocm"; InstallerTag = $null; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # ── the wheels the installer did NOT choose ──
    # An upgrade off the legacy layout: install.ps1's migrated arm installs unsloth alone and never
    # touches torch, and its flavor repair no-ops when its own expected tag is 'cpu', so a +cu118
    # wheel from a previous install on different hardware arrives untouched. Both scans agree there
    # is no NVIDIA GPU; preserving it would pin the dependency pass onto a cu118 index regardless.
    @{ Name = "stale cu118 wheel the installer did not choose, rescan says cpu"; Installed = "cu118"; InstallerTag = "cpu"; Expected = "cpu"; Nvidia = $false; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # The same stale wheel on a mapped AMD host, the costliest case: preserving it would ALSO set
    # $CuTag to cu118, and the ROCm reroute below needs $CuTag -eq "cpu", so the Radeon would never
    # get a ROCm wheel and setup would still exit 0.
    @{ Name = "stale cu118 wheel the installer did not choose, rescan says rocm"; Installed = "cu118"; InstallerTag = "cpu"; Expected = "rocm"; Nvidia = $false; Xpu = $false; Rocm = $true; Gfx = "gfx1151"
       Keep = $false; CuTag = "cpu";   Amd = $true;  XpuArm = $false }
    # Mirror image: a stale +rocm wheel left by an old Radeon install, on a box that is now
    # NVIDIA and that the installer resolved as CPU (its own ROCm scan found nothing to map).
    @{ Name = "stale rocm wheel the installer did not choose, rescan says cu128"; Installed = "rocm"; InstallerTag = "cpu"; Expected = "cu128"; Nvidia = $true; Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # The installer ran but had no answer (a custom index whose leaf names no flavor): blank is
    # unknown, exactly like absent, so this preserves.
    @{ Name = "rocm wheel, installer reported blank, rescan says cu128"; Installed = "rocm"; InstallerTag = ""; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $true;  CuTag = "cpu";   Amd = $false; XpuArm = $false }
    # ...and the repairs that MUST still happen, or this guard has reintroduced the loop.
    @{ Name = "cpu wheel on a ROCm host";       Installed = "cpu";   InstallerTag = "rocm"; Expected = "rocm";  Nvidia = $false; Xpu = $false; Rocm = $true;  Gfx = "gfx1151"
       Keep = $false; CuTag = "cpu";   Amd = $true;  XpuArm = $false }
    @{ Name = "cpu wheel on a CUDA host";       Installed = "cpu";   InstallerTag = "cu128"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # A cu* family move is a repair, not a family change, and escapes before the guard.
    @{ Name = "cu126 wheel on a cu128 host";    Installed = "cu126"; InstallerTag = "cu128"; Expected = "cu128"; Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
    # Torch does not import at all: the venv is broken, not a family disagreement.
    @{ Name = "torch that will not import";     Installed = $null;   InstallerTag = "cu128"; Expected = $null;   Nvidia = $true;  Xpu = $false; Rocm = $false; Gfx = $null
       Keep = $false; CuTag = "cu128"; Amd = $false; XpuArm = $false }
)
# Normalised by setup.ps1's own line, lifted out and executed rather than retyped, so the
# empty-string case is answered by the shipped code on whichever edition is running (7.5+ keeps a
# blank value, 5.1 and 7.0-7.4 remove the variable on that assignment).
$_tagReadPat = '(?m)^(\$InstallerTorchTag = if \(\[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_INSTALLER_TORCH_TAG\)\) \{ \$null \}\n\s+else \{[^\n]*\}\n)'
$_tagRead = if ($setupText -match $_tagReadPat) { $Matches[1] } else { "" }
Check "the installer-report read was found" ($_tagRead -ne "")

foreach ($case in $_cases) {
    if ($null -eq $case.InstallerTag) { Remove-Item Env:UNSLOTH_INSTALLER_TORCH_TAG -ErrorAction SilentlyContinue }
    else { $env:UNSLOTH_INSTALLER_TORCH_TAG = $case.InstallerTag }
    # Guarded, not bare: Invoke-Expression on an empty string is a terminating error at this file's
    # script-scope "Stop", so a tree whose read line is missing would unwind here and leave every
    # behavioural case below unreported rather than FAIL.
    if ($_tagRead) { Invoke-Expression $_tagRead }
    else { Remove-Variable -Name InstallerTorchTag -Scope Script -ErrorAction SilentlyContinue }
    $script:shouldRebuild = $true
    $script:InstallerManagedSetup = $true
    $script:installedTorchTag = $case.Installed
    # Left UNASSIGNED when the probe failed, exactly as setup.ps1 leaves it, so a condition reading
    # it before $installedTorchTag has answered is caught here rather than on a user's box.
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
    # The AMD arm sets $ROCmIndexUrl, and the XPU gate reads it.
    if ($_amdRuns) { $script:ROCmIndexUrl = "https://repo.amd.com/rocm/whl/gfx1151/" }
    $_xpuRuns = [bool](Invoke-Expression $_xpuGate)
    # The three arms' force decisions, as the file spells them.
    $_forced = $script:PinChangedForceReinstall -or $_amdRuns -or ($_xpuRuns -and $installedTorchTag -ne "xpu")

    Write-Host "  case: $($case.Name)"
    Check "    it is not rebuilt from scratch"  (-not $script:shouldRebuild)
    Check "    in-place repair = $(-not $case.Keep)" ($script:PinChangedForceReinstall -eq (-not $case.Keep))
    Check "    index family = $($case.CuTag)"   ($script:CuTag -eq $case.CuTag)
    Check "    AMD arm runs = $($case.Amd)"     ($_amdRuns -eq $case.Amd)
    Check "    XPU arm runs = $($case.XpuArm)"  ($_xpuRuns -eq $case.XpuArm)
    # The claim the whole guard rests on: no arm re-lands torch over the kept wheel.
    Check "    torch is force-reinstalled = $(-not $case.Keep)" ($_forced -eq (-not $case.Keep))
    # Only when the wheel was actually kept, or a repaired case drags the old family through the
    # install arms below.
    Check "    kept family recorded = $($case.Keep)" (
        [bool]$script:PreservedInstallerTorchTag -eq $case.Keep)
}
Remove-Item Env:UNSLOTH_INSTALLER_TORCH_TAG -ErrorAction SilentlyContinue

# The guard must not read $expectedTorchTag before $installedTorchTag has answered, and the cases
# above prove the cascade survives it. The hazard itself, rather than taken on trust: under a
# caller's Set-StrictMode -- studio/setup.bat launches setup WITHOUT -NoProfile, so a profile can
# set one -- reading a variable that was never assigned throws.
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
# The disk-based rescue above KEEPS a venv whose `import torch` never came back, and on a direct
# update that venv reaches the AMD fast-path check, where an unbounded call waits forever: setup
# would hang instead of finishing, on precisely the host the rescue was added for. Before the
# rescue this could not happen, because the same venv was deleted. The leading \n and indent
# matter: `elseif ($script:ROCmGfxArch) {` ENDS in that same text and there are two of those
# higher up, so a bare anchor starts the region 1700 lines early and every -not check goes green.
$_amdFastPat = '(?s)(\n        if \(\$script:ROCmGfxArch\) \{\n.*?reinstalling ROCm PyTorch[^\n]*\n)'
$_amdFast = if ($setupText -match $_amdFastPat) { $Matches[1] } else { "" }
Check "the AMD fast-path escape was found"  ($_amdFast -ne "")
Check "CRLF is normalised, not tolerated"   (-not (($setupText -replace "`n", "`r`n") -match $_amdFastPat))
Check "no bare interpreter call is left"    (-not ($_amdFast -match '&\s*python -c'))
Check "it goes through the bounded probe"   ($_amdFast -match 'Invoke-BoundedPythonProbe -PythonExe "python"')
Check "it still asks torch.cuda.is_available" ($_amdFast -match 'torch\.cuda\.is_available')
# A probe that does not answer must keep reading as CPU: one dependency pass is the safe direction,
# and reading a timeout as "the GPU is fine" would fast-path past the ROCm install.
Check "an unanswered probe still reads as CPU" ($_amdFast -match '\$_torchIsCpu = -not \$_rocmTorchProbe\.Ok')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
