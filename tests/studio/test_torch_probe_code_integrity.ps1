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

# And the wiring: the driver advice must be behind the classifier in all three arms, or the
# classifier is dead code.
$setupText = Get-Content $setup -Raw
Check "the three rescue arms consult the classifier" {
    ([regex]::Matches($setupText, "Write-CodeIntegrityTorchNotice -Reason \`$_probeBlockReason")).Count -eq 3
}
Check "a policy block does not force a reinstall of the same wheels" {
    $setupText.Contains('-not $_verProbe.TimedOut -and -not $_probeBlockReason')
}

if ($failures.Count) {
    Write-Host ""
    Write-Host "$($failures.Count) check(s) failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
Write-Host ""
Write-Host "all checks passed"
