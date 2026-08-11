#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A single-AMD-GPU host must not install itself into a loop (#8335).
#
# Two defects met on that host. WMI found exactly one Radeon, the if-expression holding it
# unrolled to a scalar, and a scalar's .Count is $null under Windows PowerShell 5.1, so setup
# reported "gpu none" and judged the installed ROCm venv stale against a required "cpu". The
# stale branch under install.ps1 then aborted with "re-run install.ps1", install.ps1's failure
# path restored the previous environment, and the next run reached the same verdict. Nothing
# about that pair converges, which is why the same abort has been reported from four unrelated
# triggers (#5942, #7275, #8335, and a driver crash on Discord).
#
# Read this before adding a case: PowerShell 7 returns .Count = 1 for a scalar, so the pwsh that
# runs this file CANNOT reproduce the 5.1 half of #8335. What it can do is prove the unroll
# itself still happens here, and assert the @() wrap that fixes it as a source shape. The 5.1
# leg lives in CI. No check below claims otherwise.
#
# Run: pwsh -NoProfile -File tests/studio/test_amd_venv_repair_loop.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setup = Join-Path $root "studio/setup.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Returns the source text of each named function. The caller Invoke-Expression's it at script
# scope; doing that inside a function would lose the helpers on return. Recursive, so it also
# reaches install.ps1's copies, which live inside Install-UnslothStudio.
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

# Stubs for the installers' printers, so this file does not depend on the ANSI helpers.
# Both use Write-Host, so neither can pollute a function's return value.
function substep { param([string]$Message, [string]$Color = "DarkGray") }
function Write-StudioStdoutMirror { param([string]$Line) }

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== the unroll this pwsh CAN see ==="
# The bug is the unroll, and it happens on every PowerShell. What differs is only the
# consequence: 5.1 answers $null to a scalar's .Count and 7 answers 1, so the "-gt 0" test that
# follows is false there and true here. Hence the source assertion further down.
$oneGpu = @("AMD Radeon PRO W7900")
$unwrapped = if ($oneGpu.Count -gt 0) { $oneGpu } else { @() }
$wrapped = @(if ($oneGpu.Count -gt 0) { $oneGpu } else { @() })
Check "an if-expression unrolls a one-element array here too" (-not ($unwrapped -is [array]))
Check "the @() wrap keeps it an array"                        ($wrapped -is [array])
Check "the wrapped value still counts one GPU"                ($wrapped.Count -eq 1)
# Stated as a check so it cannot quietly stop being true, and so nobody reads the file as a 5.1
# repro: on 5.1 this same expression answers $null.
Check "this pwsh answers 1, not `$null, to a scalar .Count"   ($unwrapped.Count -eq 1)
Check "so the harness is PowerShell 7, not 5.1"               ($PSVersionTable.PSVersion.Major -ge 6)

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Test-VenvTorchIsRocm reads the wheel off disk ==="
# The AMD counterpart of Test-VenvTorchIsXpu. It exists so a faulted HIP runtime -- an
# `import torch` that raises at the DLL load or never returns -- does not get answered by
# deleting a perfectly good ROCm environment. Free by design: no interpreter is launched, so a
# CPU-only host pays nothing for it on every update.
foreach ($srcText in (Get-HelperSources $setup @("Test-VenvTorchIsRocm"))) { Invoke-Expression $srcText }
# @() around the call, then [0]: a one-name request returns a one-element array, which unrolls
# to a String on the way out, and [0] on a String is its first character. The same unroll this
# file is about, one directory over.
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
# download.pytorch.org labels the wheel +rocmX.Y; the AMD Windows wheels label it +gfxNNNN. Both
# are ROCm builds and the reporter of #8335 was running the second kind.
Check "rocm6.4 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'")
Check "rocm7.0 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+rocm7.0'")
Check "gfx1151 wheel"        (Invoke-IsRocm "C:\v" "__version__ = '2.9.0+gfx1151'")
Check "gfx110X-all wheel"    (Invoke-IsRocm "C:\v" "__version__ = '2.7.1+gfx110X.all'")
Check "cuda wheel"           (-not (Invoke-IsRocm "C:\v" $CU))
Check "xpu wheel"            (-not (Invoke-IsRocm "C:\v" $XPU))
Check "untagged wheel"       (-not (Invoke-IsRocm "C:\v" $BARE))
# version.py on a CUDA build carries `hip: Optional[str] = None`, and the ROCm builds carry a
# `gfx` line of their own. Only the local label on __version__ decides, or a CUDA venv would be
# kept as ROCm and never repaired.
Check "hip attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`nhip: Optional[str] = None")))
Check "gfx attr but cuda wheel"  (-not (Invoke-IsRocm "C:\v" ($CU + "`ngfx = 'gfx1100'")))
Check "no torch installed"   (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Missing))
Check "unreadable file"      (-not (Invoke-IsRocm "C:\v" "__version__ = '2.8.0+rocm6.4'" -Throws))
Check "no venv path"         (-not (Invoke-IsRocm "" "__version__ = '2.8.0+rocm6.4'"))

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Invoke-BoundedPythonProbe keeps the reason it failed ==="
# Both installers carry a copy, so both are driven. The probe drained stderr and threw it away,
# which made "the HIP DLLs will not load", "torch is not installed" and "the import never came
# back" arrive at the caller as one silent False -- and the caller deletes the environment over
# that answer. The bounding and the async drain must survive intact, so a real child process is
# launched rather than a mock.
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
        # Draining stdout is what stops a noisy import from deadlocking on a full pipe; keeping
        # stderr must not have changed which stream is which.
        Check "stderr does not leak into Output"     (-not ($bad.Output -match 'WinError'))

        # A wedged import is the case the whole helper exists for. It must still be bounded, and
        # it must now say that that is what happened.
        $slow = Invoke-BoundedPythonProbe -PythonExe $py.Source -Code 'import time; time.sleep(45)' -TimeoutSec 1
        Check "a hung probe is still bounded"        (-not $slow.Ok)
        Check "a hung probe says it timed out"       ($slow.Error -match 'did not answer within')

        Check "an empty code string is refused"      (-not (Invoke-BoundedPythonProbe -PythonExe $py.Source -Code '').Ok)
    }
    # No interpreter at that path at all: Process.Start throws, and the exception text is the
    # only thing there is to report.
    $gone = Invoke-BoundedPythonProbe -PythonExe (Join-Path $root "no-such-python-XYZ") -Code 'print(1)'
    Check "a missing interpreter reads as not Ok"    (-not $gone.Ok)
    Check "a missing interpreter reports why"        (-not [string]::IsNullOrWhiteSpace($gone.Error))
}

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== source shapes ==="
# Normalised to LF once, rather than per pattern: on a CRLF checkout every \n-anchored pattern
# below matches nothing, and the -not checks over an empty region pass vacuously. That is a real
# incident, not a hypothetical (see test_setup_xpu_runtime_prereport.ps1).
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"

Write-Host "the AMD WMI fallback survives a single-GPU host"
# The trailing `\]\n` is what makes the CRLF check below bite: on a CRLF checkout the line ends
# `[0]\r\n`, so the pattern fails rather than matching a region that is silently empty.
$_wmiPat = '(?s)(\$amdGpus = @\(Get-CimInstance Win32_VideoController.*?\$ROCmGpuLabel = \$script:ROCmGpuLabels\[0\]\n)'
$_wmi = if ($setupText -match $_wmiPat) { $Matches[1] } else { "" }
Check "the WMI fallback was found"        ($_wmi -ne "")
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $_wmiPat))
# This is the #8335 assertion. It is a shape, not a behaviour: see the header.
Check "the healthy/all choice is @() wrapped" (
    $_wmi -match '\$wmiGpus = @\(if \(\$healthyGpus\.Count -gt 0\) \{ \$healthyGpus \} else \{ \$amdGpus \}\)')
Check "the unwrapped form is gone"        (-not ($_wmi -match '\$wmiGpus = if \('))
# The two lists it chooses between are wrapped for the same reason; the Where-Object above them
# returns a scalar on a single match too.
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
# The disk read is the point: launching a second interpreter after the first one hung is exactly
# what the rescue is avoiding.
Check "the rescue launches no interpreter" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsRocm.*?Invoke-BoundedPythonProbe'))
# Matched on the substep ARGUMENT, not the region: the comment above the branch names the driver
# too, so a loose match stays green with the message itself deleted.
Check "the rescue names the driver fix"    ($_rescue -match 'substep "[^"]*Adrenalin')
# Non-AMD, non-Intel families must still rebuild on an unreadable flavour, exactly as before.
Check "other families still rebuild"       ($_rescue -match '(?s)\} else \{\s*\$shouldRebuild = \$true')
# The XPU rescue this one is modelled on has to still be there, and ahead of it: an Arc box is
# not an AMD box and its message names a different driver.
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
# environment aside earlier in the same run, and its failure path moves it straight back -- so
# exiting here landed on the byte-identical starting state and the next run reached the same
# verdict. Nothing in that cycle can converge. Bounded to the branch: the delete below still
# has two legitimate Exit-SetupFailure calls of its own.
$_managedPat = '(?s)(if \(\$shouldRebuild -and \$InstallerManagedSetup\) \{.*?\n    \}\n)'
$_managed = if ($_repair -match $_managedPat) { $Matches[1] } else { "" }
Check "the installer-managed branch was found" ($_managed -ne "")
Check "an installer-managed run does not abort" (-not ($_managed -match 'Exit-SetupFailure'))
# Matched on the Write-StudioLine ARGUMENT, so the comment above the branch, which quotes the
# old advice to explain why it went, cannot keep this green on its own.
Check "the advice that pointed at itself is gone" (
    -not ($setupText -match 'Write-StudioLine "\s*Re-run install\.ps1'))
Check "the abort message is gone too"      (-not ($setupText -match 'The existing Unsloth environment needs repair'))
# What replaces it: the same in-place repair an index-pin change and a cu* family change already
# take, which every torch install arm below honours with --force-reinstall.
Check "an installer-managed run repairs in place" ($_managed -match '\$script:PinChangedForceReinstall = \$true')
Check "and clears the rebuild flag"        ($_managed -match '\$shouldRebuild = \$false')
# Deleting was never available on this path anyway: install.ps1 runs setup through the venv's own
# unsloth.exe, so python.exe is locked by the process executing the script.
Check "it does not try to wipe the venv it runs from" (-not ($_managed -match 'Remove-Item'))
# A direct `unsloth studio update` keeps its own self-repair, and the custom-home guard that
# stops it wiping an unrelated environment stays in front of the delete.
Check "a direct update still rebuilds"     ($_repair -match 'Stale venv detected \(\$reason\) -- rebuilding')
Check "the custom-home guard still gates the wipe" ($_repair -match '\$StudioHomeIsCustom')
# Why it failed, not just that it failed: this is the line that separates a faulted GPU driver
# from a missing wheel, and the user is about to be told what happened to their environment.
Check "the swallowed probe error is surfaced" ($_repair -match '\$_verProbe\.Error')
Check "the probe handle is declared up front" ($setupText -match '(?m)^\s*\$_verProbe = \$null')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
