# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

<#
.SYNOPSIS
    Ask AMSI whether it will let a script compile, without running the script.

.DESCRIPTION
    This is the probe that actually exercises the failure in #10805.

    PowerShell hands a script block to AMSI at COMPILE time, before the first statement runs. So the
    thing to measure is compilation, not execution, and `[scriptblock]::Create` on the file's text
    reproduces the exact failing path with none of the side effects of an install: no download, no
    venv, no registry, no shortcut. A block surfaces as a ParseException whose
    FullyQualifiedErrorId is `ScriptContainedMaliciousContent` (or `ScriptHasAdminBlockedContent`
    where a policy rather than a signature is responsible).

    Matching is on the error ID and never on message text, which is localised. install.rs already
    makes that distinction and this follows it.

    `[scriptblock]::Create` is banned in the shipped installers by
    tests/studio/test_installer_av_shapes.py::test_no_remote_script_is_executed_in_process. That test
    parametrises over the shipped scripts only, so this probe is unaffected -- and it is deliberate:
    submitting text to the compiler is the whole measurement, and there is no other way to ask AMSI
    the question. Do not "fix" this file to satisfy that test.

.PARAMETER Path
    One script to submit.

    Deliberately one, not a list. In `-File` mode -- which is how Windows PowerShell 5.1 has to be
    invoked, and 5.1 is the interpreter that matters here -- the argument binder does not split a
    comma-separated value into an array: it arrives as a single string and the probe reports "file
    not found" for a path that exists. Observed while smoke-running this. One file per invocation
    also means `-Control` can fire in the SAME process as the measurement it vouches for, rather
    than in a sibling process that merely shares a host configuration.

.PARAMETER Control
    Submit Microsoft's documented AMSI test sample instead of a file.

    Non-negotiable, and the reason this script exists rather than three lines of inline YAML: a
    runner where AMSI is not live, or where Defender is in passive mode because another product is
    primary, produces "no block" for every input. That is identical to a clean verdict and has
    shipped as one before. If the control does not fire, this probe reports that it could not
    measure, which must never be read as clean.

.PARAMETER OutFile
    Where to write the JSON result.
#>

[CmdletBinding()]
param(
    [string]$Path,
    [switch]$Control,
    [Parameter(Mandatory = $true)][string]$OutFile
)

$ErrorActionPreference = 'Continue'

# The error IDs that mean "the scanner refused this", separated because they have different causes
# and different fixes. A signature match is something we can get cleared; an administrator policy
# block is not.
$blockedIds = @{
    'ScriptContainedMaliciousContent' = 'an AMSI provider matched the content'
    'ScriptHasAdminBlockedContent'    = 'an administrative policy blocked the content'
}

function Submit-ToCompiler {
    param([string]$Label, [string]$Text)

    $result = [ordered]@{
        label     = $Label
        length    = $Text.Length
        blocked   = $false
        errorId   = ''
        errorType = ''
        message   = ''
        reason    = ''
    }
    try {
        # Compilation only. Nothing is invoked, so submitting the real installer here cannot install
        # anything, and that is what makes running this on every candidate cheap and safe.
        $null = [scriptblock]::Create($Text)
    } catch {
        $result.errorType = $_.Exception.GetType().FullName
        $result.errorId = [string]$_.FullyQualifiedErrorId
        # Truncated: a provider can echo a long span of the submitted script back, and the whole
        # point of this work is to avoid writing suspicious-looking text into files.
        $msg = [string]$_.Exception.Message
        if ($msg.Length -gt 400) { $msg = $msg.Substring(0, 400) + ' ...' }
        $result.message = $msg
        foreach ($id in $blockedIds.Keys) {
            if ($result.errorId -like "*$id*") {
                $result.blocked = $true
                $result.reason = $blockedIds[$id]
                break
            }
        }
        if (-not $result.blocked) {
            # A syntax error is not a scanner verdict, and conflating the two would let a broken
            # candidate masquerade as a detection (or the reverse, which is worse).
            $result.reason = 'the script did not compile, for a reason unrelated to a scanner'
        }
    }
    return $result
}

$results = New-Object System.Collections.ArrayList

if ($Control) {
    # Assembled from fragments so this repository is not itself a sample carrying the signature.
    # Microsoft's own documentation splits it the same way. Reassembled only in memory, never
    # written to disk.
    $sample = 'AMSI Test Sample: ' + '7e72c3ce-' + '861b-4339-' + '8740-' + '0ac1484c1386'
    [void]$results.Add((Submit-ToCompiler -Label 'control (AMSI test sample)' -Text $sample))
}

foreach ($file in @($Path) | Where-Object { $_ }) {
    if (-not (Test-Path -LiteralPath $file)) {
        [void]$results.Add([ordered]@{
            label = $file; blocked = $false; errorId = ''; reason = 'file not found'; missing = $true
        })
        continue
    }
    $text = $null
    try {
        $text = Get-Content -Raw -LiteralPath $file -ErrorAction Stop
    } catch {
        [void]$results.Add([ordered]@{
            label = $file; blocked = $false; reason = "could not read: $($_.Exception.Message)"
            unreadable = $true
        })
        continue
    }
    [void]$results.Add((Submit-ToCompiler -Label $file -Text $text))
}

$payload = [ordered]@{
    host         = $Host.Name
    psVersion    = [string]$PSVersionTable.PSVersion
    psEdition    = [string]$PSVersionTable.PSEdition
    languageMode = [string]$ExecutionContext.SessionState.LanguageMode
    results      = @($results)
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$payload | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $OutFile -Encoding utf8

foreach ($r in $results) {
    $verdict = if ($r.blocked) { 'BLOCKED' } elseif ($r.errorId) { 'error' } else { 'compiled' }
    Write-Host ("  {0,-9} {1}{2}" -f $verdict, $r.label,
                $(if ($r.reason) { "  [$($r.reason)]" } else { '' }))
    if ($r.errorId) { Write-Host "            id: $($r.errorId)" }
}

# Always zero. Whether a block is a failure is a decision about controls and about which side of a
# differential it appeared on, and the caller holds both. A probe that exits non-zero on a detection
# cannot be used to establish a baseline.
exit 0
