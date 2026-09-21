#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A Windows code integrity block must not be reported as a driver fault (#6588, #6648).
#
# On a Smart App Control machine `import torch` dies on an unsigned ROCm DLL out of AMD's
# own wheel. The probe in studio/setup.ps1 sees a failed import, the rescue arm keeps the
# venv (correct) and then told the user to update their AMD driver (not correct, and it
# sends them somewhere that cannot help). Nothing we ship can make that DLL load, so the
# one useful thing the installer can do is name the cause.
#
# These checks pin the classifier itself: the statuses and winerrors it must catch, the
# things it must NOT call a policy block, and that the reasons are spelled the way
# studio/backend/utils/code_integrity.py spells them, since the two are read together in
# one bug report.
# Run: pwsh -NoProfile -File tests/studio/test_torch_probe_code_integrity.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setup = Join-Path $repo "studio/setup.ps1"

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

# Extracted from the shipped file, never pasted, so an edit to setup.ps1 is what these run.
. ([scriptblock]::Create((Get-FunctionText -Path $setup -Name "Get-CodeIntegrityBlockReason")))

$failures = @()
function Check {
    param([string] $What, [scriptblock] $Body)
    try {
        if (& $Body) { Write-Host "ok   $What" } else { $script:failures += $What; Write-Host "FAIL $What" }
    } catch {
        $script:failures += "$What ($_)"
        Write-Host "FAIL $What ($_)"
    }
}

# The three NTSTATUS refusals, as Python prints them in a traceback.
Check "0xc0e90002 is a block" {
    $r = Get-CodeIntegrityBlockReason -Text "OSError: [WinError -1058930686] Error status 0xc0e90002"
    $r -eq "Smart App Control or an Application Control policy blocked the image"
}
Check "0xC0000428 is an invalid hash, not a policy verdict" {
    (Get-CodeIntegrityBlockReason -Text "failed with 0xC0000428") -eq
        "the image failed code integrity validation (invalid or missing signature)"
}
Check "0xc0000602 is the fail-fast" {
    (Get-CodeIntegrityBlockReason -Text "exited 0xc0000602") -eq
        "the image was refused by a code integrity fail-fast"
}

# The winerror spellings, which is how this actually reaches the probe: #6648 recorded
# 4551 against rocfft.dll out of the ROCm wheel.
Check "WinError 4551 is a block" {
    $text = "ImportError: DLL load failed while importing _C: [WinError 4551] An Application Control policy has blocked this file"
    (Get-CodeIntegrityBlockReason -Text $text) -eq "code integrity blocked the image"
}
Check "WinError 1260 is an administrator policy" {
    (Get-CodeIntegrityBlockReason -Text "OSError: [WinError 1260] blocked by policy") -eq
        "an Application Control policy blocked this program"
}
Check "WinError 577 is an unverifiable signature" {
    (Get-CodeIntegrityBlockReason -Text "[WinError 577] bad signature") -eq
        "Windows could not verify the digital signature of the image"
}

# Free text, for the shapes that carry no number at all.
Check "the Smart App Control wording is a block" {
    (Get-CodeIntegrityBlockReason -Text "This app was blocked by Smart App Control") -eq
        "Smart App Control blocked this program"
}
Check "the Application Control wording is a block" {
    (Get-CodeIntegrityBlockReason -Text "An Application Control policy has blocked this file") -eq
        "an Application Control policy blocked this program"
}

# The negatives matter more than the positives here: calling an ordinary failure a policy
# block would send a user with a genuinely broken install to turn off a security feature.
Check "a missing DLL is not a block" {
    $null -eq (Get-CodeIntegrityBlockReason -Text "ImportError: DLL load failed while importing _C: [WinError 126] The specified module could not be found.")
}
Check "Bad Image alone is not a block" {
    $null -eq (Get-CodeIntegrityBlockReason -Text "rocfft.dll is either not designed to run on Windows or it contains an error.")
}
Check "an ordinary ImportError is not a block" {
    $null -eq (Get-CodeIntegrityBlockReason -Text "ModuleNotFoundError: No module named 'torch'")
}
Check "empty and whitespace are not blocks" {
    ($null -eq (Get-CodeIntegrityBlockReason -Text "")) -and
    ($null -eq (Get-CodeIntegrityBlockReason -Text "   `n  ")) -and
    ($null -eq (Get-CodeIntegrityBlockReason -Text $null))
}
Check "a status that merely appears in a path is still matched only as a status" {
    # 0xc0e90002 inside a longer hex run is a different number.
    $null -eq (Get-CodeIntegrityBlockReason -Text "checksum 0xc0e90002ff")
}

# Drift guard: the reasons are read beside the backend's in one bug report, so they have to
# be the same sentences. Read out of the Python rather than restated here.
$py = Get-Content (Join-Path $repo "studio/backend/utils/code_integrity.py") -Raw
foreach ($reason in @(
    "Smart App Control or an Application Control policy blocked the image",
    "the image failed code integrity validation (invalid or missing signature)",
    "the image was refused by a code integrity fail-fast",
    "Windows could not verify the digital signature of the image",
    "an Application Control policy blocked this program",
    "code integrity blocked the image",
    "Smart App Control blocked this program"
)) {
    Check "the backend spells it the same way: $reason" { $py.Contains($reason) }
}

# The ambiguous pair, which is the distinction the backend draws and this side has to keep:
# Windows raises both for a damaged file too, so they must not rule out a reinstall.
. ([scriptblock]::Create((Get-FunctionText -Path $setup -Name "Test-CodeIntegrityReasonIsAmbiguous")))
$script:CodeIntegrityAmbiguousReasons = @(
    "the image failed code integrity validation (invalid or missing signature)",
    "Windows could not verify the digital signature of the image"
)

Check "0xC0000428 is ambiguous, not a settled policy verdict" {
    Test-CodeIntegrityReasonIsAmbiguous -Reason (Get-CodeIntegrityBlockReason -Text "failed with 0xC0000428")
}
Check "WinError 577 is ambiguous too" {
    Test-CodeIntegrityReasonIsAmbiguous -Reason (Get-CodeIntegrityBlockReason -Text "[WinError 577] bad signature")
}
Check "a Smart App Control block is not ambiguous" {
    -not (Test-CodeIntegrityReasonIsAmbiguous -Reason (Get-CodeIntegrityBlockReason -Text "blocked by Smart App Control"))
}
Check "0xc0e90002 and WinError 4551 are not ambiguous" {
    (-not (Test-CodeIntegrityReasonIsAmbiguous -Reason (Get-CodeIntegrityBlockReason -Text "Error status 0xc0e90002"))) -and
    (-not (Test-CodeIntegrityReasonIsAmbiguous -Reason (Get-CodeIntegrityBlockReason -Text "[WinError 4551] blocked")))
}
Check "no reason at all is not ambiguous" {
    -not (Test-CodeIntegrityReasonIsAmbiguous -Reason $null)
}
Check "the ambiguous set is the backend's _INVALID_HASH_REASONS" {
    # Both members come from _REASON_INVALID_HASH_STATUS / _REASON_INVALID_HASH_WINERROR,
    # and the backend keys its own "a reinstall may clear this" message on the same pair.
    $ok = $true
    foreach ($reason in $script:CodeIntegrityAmbiguousReasons) { if (-not $py.Contains($reason)) { $ok = $false } }
    $ok -and $py.Contains("_INVALID_HASH_REASONS = frozenset({_REASON_INVALID_HASH_STATUS, _REASON_INVALID_HASH_WINERROR})")
}

# And the wiring: the driver advice must be behind the classifier in all three arms, or the
# classifier is dead code.
$setupText = Get-Content $setup -Raw
Check "the three rescue arms consult the classifier" {
    ([regex]::Matches($setupText, "Write-CodeIntegrityTorchNotice -Reason \`$_probeBlockReason")).Count -eq 3
}
Check "a settled policy block does not force a reinstall of the same wheels" {
    $setupText.Contains('-not $_verProbe.TimedOut -and -not $_blockRulesOutDamage')
}
Check "an ambiguous status still reaches the force-reinstall" {
    # The whole point of the distinction: 0xC0000428 can be a corrupt wheel, and that is
    # repaired by reinstalling the same family in place.
    $setupText.Contains('$_blockRulesOutDamage = $_probeBlockReason -and')
}
# The closing sentence has to match what setup does next, since this notice is emitted
# from a kept environment, from one whose wheels are about to be force-reinstalled, and
# from the rebuild path.
. ([scriptblock]::Create((Get-FunctionText -Path $setup -Name "Write-CodeIntegrityTorchNotice")))
# Script-scope data the notice reads, restated here and drift-checked against the file below.
$script:CodeIntegrityAdminPolicyReasons = @("an Application Control policy blocked this program")
$script:CodeIntegritySmartAppControlReasons = @("Smart App Control blocked this program")
function substep { param([string]$Text, [string]$Colour) $script:said += $Text }

Check "a kept environment is the only one told it is kept" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "code integrity blocked the image" -Action "kept"
    ($script:said -join " ").Contains("kept as it is")
}
Check "a forced reinstall says the wheels are being replaced, not kept" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "the image failed code integrity validation (invalid or missing signature)" -Action "reinstall"
    $joined = $script:said -join " "
    $joined.Contains("reinstall the same wheels in place") -and (-not $joined.Contains("kept as it is"))
}
Check "the rebuild path says the environment is being rebuilt" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "code integrity blocked the image" -Action "rebuild"
    $joined = $script:said -join " "
    $joined.Contains("rebuild this environment") -and (-not $joined.Contains("kept as it is"))
}
Check "an unknown action is refused rather than guessed" {
    $script:said = @()
    $refused = $false
    try { Write-CodeIntegrityTorchNotice -Reason "x" -Action "wipe" } catch { $refused = $true }
    $refused
}
Check "the CUDA arm decides before it speaks" {
    $setupText.Contains('Write-CodeIntegrityTorchNotice -Reason $_probeBlockReason -Action $_blockAction')
}
Check "the rebuild notice is chosen after the guards that cancel the rebuild" {
    # Those guards turn the rebuild into an in-place reinstall on every installer-managed run
    # and every direct update, so a notice above them describes a path setup does not take.
    $notice = $setupText.IndexOf('Write-CodeIntegrityTorchNotice -Reason $_rebuildBlockReason -Action $_rebuildAction')
    $lastGuard = $setupText.LastIndexOf('$script:PinChangedForceReinstall = $true')
    ($notice -gt 0) -and ($lastGuard -gt 0) -and ($notice -gt $lastGuard)
}
Check "the rebuild notice says reinstall when the rebuild was cancelled" {
    $setupText.Contains('$_rebuildAction = "reinstall"') -and
    $setupText.Contains('if ($shouldRebuild) { $_rebuildAction = "rebuild" }')
}

# An exit code is the only thing a fail-fast leaves behind, so the classifier has to be fed it.
. ([scriptblock]::Create((Get-FunctionText -Path $setup -Name "Get-ProbeFailureText")))

Check "a probe killed with a code integrity NTSTATUS and no stderr is still classified" {
    # 0xC0000602 as Windows hands it back through Process.ExitCode.
    $probe = [pscustomobject]@{ Ok = $false; Output = ""; Error = ""; TimedOut = $false; ExitCode = -1073740286 }
    (Get-CodeIntegrityBlockReason -Text (Get-ProbeFailureText -Probe $probe)) -eq
        "the image was refused by a code integrity fail-fast"
}
Check "the stderr text survives beside the exit code" {
    $probe = [pscustomobject]@{ Ok = $false; Output = ""; Error = "[WinError 4551] blocked"; TimedOut = $false; ExitCode = -1073740286 }
    $text = Get-ProbeFailureText -Probe $probe
    $text.Contains("WinError 4551") -and $text.Contains("0xc0000602")
}
Check "an ordinary python failure is not dressed up as a status" {
    # exit 1 with a real traceback: nothing here is a block, and 0x00000001 must not appear.
    $probe = [pscustomobject]@{ Ok = $false; Output = ""; Error = "ModuleNotFoundError: No module named 'torch'"; TimedOut = $false; ExitCode = 1 }
    $text = Get-ProbeFailureText -Probe $probe
    (-not $text.Contains("0x")) -and ($null -eq (Get-CodeIntegrityBlockReason -Text $text))
}
Check "a timeout's exit code says nothing about the installation" {
    # Kill() leaves an arbitrary code behind, and "never answered" is the caller's own case.
    $probe = [pscustomobject]@{ Ok = $false; Output = ""; Error = "python did not answer within 30 seconds"; TimedOut = $true; ExitCode = -1073740286 }
    -not (Get-ProbeFailureText -Probe $probe).Contains("0x")
}
Check "a probe that never ran is answered, not thrown at" {
    (Get-ProbeFailureText -Probe $null) -eq "" -and
    (Get-ProbeFailureText -Probe ([pscustomobject]@{ Ok = $false; Output = ""; Error = ""; TimedOut = $false; ExitCode = $null })) -eq ""
}
Check "the probe helper keeps the exit code instead of only reducing it to Ok" {
    $helper = Get-FunctionText -Path $setup -Name "Invoke-BoundedPythonProbe"
    $helper.Contains('ExitCode = $null') -and $helper.Contains('$result.ExitCode = $proc.ExitCode')
}
Check "both probe call sites classify the exit code, not stderr alone" {
    ([regex]::Matches($setupText, [regex]::Escape('Get-CodeIntegrityBlockReason -Text (Get-ProbeFailureText -Probe $_verProbe)'))).Count -eq 2
}

# The XPU and ROCm arms: ambiguous means a reinstall may clear it, and the notice says so, so
# they have to actually do it rather than print the promise and keep a damaged venv.
Check "an ambiguous block repairs the XPU and ROCm venvs too" {
    $setupText.Contains('$_ambiguousBlockRepair = [bool]($_probeBlockReason -and $_willForceReinstall)') -and
    ([regex]::Matches($setupText, [regex]::Escape('Write-CodeIntegrityTorchNotice -Reason $_probeBlockReason -Action $_ambiguousBlockAction'))).Count -eq 2 -and
    # Three arms set the flag now: CUDA on any non-timeout failure, XPU and ROCm on an
    # ambiguous block. The flag is what makes the install pass use --force-reinstall.
    ([regex]::Matches($setupText, [regex]::Escape('$script:TorchImportDefinitivelyFailed = $true'))).Count -eq 3
}
Check "a driver fault with no block reason still keeps those venvs as they are" {
    # #8335 / #7275: these arms exist to stop a faulted driver costing a whole install, so the
    # repair is gated on a block reason, not on any failed import.
    $setupText.Contains('$_ambiguousBlockAction = "kept"')
}
Check "no PowerShell 7 only if-expression reached the argument" {
    # 5.1 cannot parse `-Action (if ...)`, and this file runs on both engines.
    -not ($setupText -match 'Action \(\s*if ')
}

Check "the rebuild path does not call an unknown wheel a GPU build" {
    # Reached with a CPU-only wheel or none at all, since all three family predicates
    # declined, so the GPU sentences would be describing something else.
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "code integrity blocked the image" -Action "rebuild" -GpuBuild $false
    $joined = $script:said -join " "
    (-not $joined.Contains("GPU build")) -and (-not $joined.Contains("PyTorch GPU runtime")) -and
    $joined.Contains("nothing in it runs")
}
Check "a known GPU family still gets the GPU wording" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "code integrity blocked the image" -Action "kept"
    ($script:said -join " ").Contains("holds a GPU build")
}
Check "the rebuild call site says the family is unknown" {
    $setupText.Contains('-Action $_rebuildAction -GpuBuild $false')
}

# Which policy to send the user to, told apart the way the backend tells them apart.
Check "an administrator policy is not answered with Smart App Control advice" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "an Application Control policy blocked this program" -Action "kept"
    $joined = $script:said -join " "
    $joined.Contains("AppLocker, WDAC or Group Policy") -and
    $joined.Contains("does not affect an administrator policy") -and
    (-not $joined.Contains("On a device you administer"))
}
Check "a Smart App Control block says how to turn it off" {
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "Smart App Control blocked this program" -Action "kept"
    $joined = $script:said -join " "
    $joined.Contains("no per-application exception") -and (-not $joined.Contains("AppLocker"))
}
Check "a reason that proves neither still offers both" {
    # 0xc0e90002 is SAC or WDAC and does not say which, so neither branch may claim it.
    $script:said = @()
    Write-CodeIntegrityTorchNotice -Reason "Smart App Control or an Application Control policy blocked the image" -Action "kept"
    ($script:said -join " ").Contains("On a device you administer")
}
Check "the two policy sets are the backend's" {
    $py.Contains("_ADMIN_POLICY_REASONS = frozenset({_REASON_ADMIN_POLICY})") -and
    $py.Contains("_SMART_APP_CONTROL_REASONS = frozenset({_REASON_SMART_APP_CONTROL})") -and
    $setupText.Contains('$script:CodeIntegrityAdminPolicyReasons = @("an Application Control policy blocked this program")') -and
    $setupText.Contains('$script:CodeIntegritySmartAppControlReasons = @("Smart App Control blocked this program")')
}

Check "the notice does not tell a user with no importable torch that the CPU is fine" {
    (-not $setupText.Contains("Training on the CPU is unaffected")) -and
    $setupText.Contains("it cannot run on the CPU either")
}

if ($failures.Count) {
    Write-Host ""
    Write-Host "$($failures.Count) check(s) failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
Write-Host ""
Write-Host "all checks passed"
