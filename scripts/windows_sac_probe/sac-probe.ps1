# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
<#
.SYNOPSIS
    Collect evidence about Windows code integrity blocking the Unsloth Studio
    llama.cpp runtime.

.DESCRIPTION
    Four stages, run in order, so a machine is never left in a modified state by
    accident:

        prepare   record the baseline, update everything, turn security up
        run       inventory signatures and exercise Studio
        collect   export the event log evidence into one zip
        revert    put the machine back the way prepare found it

    What this deliberately does NOT do: turn Smart App Control on or off.
    Switching it through Settings is a one-way operation that cannot be undone
    without reinstalling Windows, and the registry route needs BitLocker
    suspended and a recovery-mode boot. This script only ever adds and removes
    an *audit* App Control policy, which is reversible, and reports whichever
    mode it found the machine in.

    Evidence grading matters when reading the output. Event 3077 is an enforced
    block. Event 3076 is only "would have been blocked" audit evidence. 3089
    carries the correlated signature detail, matched to a 3077 by ActivityID and
    never by timestamp alone.

.EXAMPLE
    .\sac-probe.ps1 -Stage prepare
    .\sac-probe.ps1 -Stage run
    .\sac-probe.ps1 -Stage collect
    .\sac-probe.ps1 -Stage revert
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('prepare', 'run', 'collect', 'revert')]
    [string] $Stage,

    # Where evidence and the baseline are kept. Must survive between stages.
    [string] $WorkDir = "$env:USERPROFILE\unsloth-sac-probe",

    # Path to SmartAppControlAuditNoISG.bin from https://aka.ms/sacauditpolicies.
    # Optional: without it, prepare still records state and raises the Defender
    # settings, it just cannot add audit logging for what would be blocked.
    [string] $AuditPolicy,

    # Label for this run, so the four matrix cells do not overwrite each other.
    # Use something like "custom-b10798-sac-on" or "upstream-b10830-sac-off".
    [string] $Label = 'run',

    # Model to exercise. Any GGUF repo Studio can load.
    [string] $Model = 'unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL',

    # prepare only: skip winget/Defender updates, which are slow.
    [switch] $SkipUpdates,

    # run only: skip the Playwright scenario and just do the signature inventory.
    [switch] $SkipStudio
)

$ErrorActionPreference = 'Stop'

# The NoISG audit policy lives in the EFI system partition; the full audit
# policy replaces the active runtime policy. Both GUIDs and both destinations
# are Microsoft's, from the Smart App Control testing documentation.
$NOISG_GUID = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'
$LLAMA_DIR = Join-Path $env:USERPROFILE '.unsloth\llama.cpp'
$PE_EXT = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')
# 3076 audit, 3077 enforced block, 3089 signature detail, 3033/3099 policy and
# validation failures, 3090/3091/3092 allow-and-origin context.
$CI_EVENT_IDS = @(3033, 3076, 3077, 3089, 3090, 3091, 3092, 3099)

function Write-Section([string] $Text) {
    Write-Host ''
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Assert-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this from an elevated PowerShell. Right click Windows Terminal or PowerShell and pick "Run as administrator".'
    }
}

function Get-RunDir {
    $dir = Join-Path $WorkDir $Label
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return $dir
}

function Get-SacState {
    # 0 off, 1 enforcement, 2 evaluation. Absent on hardware that never offered it.
    $state = $null
    try {
        $state = (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' `
            -Name 'VerifiedAndReputablePolicyState' -ErrorAction Stop).VerifiedAndReputablePolicyState
    } catch {
        $state = $null
    }
    $name = switch ($state) {
        0 { 'off' }
        1 { 'enforcement' }
        2 { 'evaluation' }
        default { 'absent' }
    }
    # CiTool is the authority on what is actually enforced right now; the
    # registry value alone can disagree with the loaded policy set.
    $policies = @()
    try {
        $raw = & CiTool.exe -lp --json 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            $parsed = ($raw | Out-String | ConvertFrom-Json)
            foreach ($p in $parsed.Policies) {
                $policies += [pscustomobject]@{
                    FriendlyName = $p.FriendlyName
                    PolicyID     = $p.PolicyID
                    IsEnforced   = $p.IsEnforced
                }
            }
        }
    } catch {
        # CiTool is absent on older builds. Not fatal; the registry value stands.
    }
    return [pscustomobject]@{
        RegistryState = $state
        Mode          = $name
        Policies      = $policies
    }
}

function Save-Baseline([string] $dir) {
    $mp = $null
    try { $mp = Get-MpPreference } catch { }
    $status = $null
    try { $status = Get-MpComputerStatus } catch { }

    $baseline = [pscustomobject]@{
        CapturedAt              = (Get-Date).ToString('o')
        ComputerName            = $env:COMPUTERNAME
        WindowsBuild            = (Get-CimInstance Win32_OperatingSystem).BuildNumber
        WindowsVersion          = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').DisplayVersion
        Sac                     = Get-SacState
        DisableRealtimeMonitoring = if ($mp) { $mp.DisableRealtimeMonitoring } else { $null }
        MAPSReporting           = if ($mp) { $mp.MAPSReporting } else { $null }
        SubmitSamplesConsent    = if ($mp) { $mp.SubmitSamplesConsent } else { $null }
        CloudBlockLevel         = if ($mp) { $mp.CloudBlockLevel } else { $null }
        PUAProtection           = if ($mp) { $mp.PUAProtection } else { $null }
        AMProductVersion        = if ($status) { $status.AMProductVersion } else { $null }
        AntivirusSignatureVersion = if ($status) { $status.AntivirusSignatureVersion } else { $null }
        AuditPolicyApplied      = $false
    }
    $path = Join-Path $dir 'baseline.json'
    $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
    Write-Host "baseline written to $path"
    return $baseline
}

function Invoke-Prepare {
    Assert-Elevated
    $dir = Get-RunDir
    Write-Section 'Baseline'
    $baseline = Save-Baseline $dir
    Write-Host ("Smart App Control: {0} (registry state {1})" -f $baseline.Sac.Mode, $baseline.Sac.RegistryState)
    foreach ($p in $baseline.Sac.Policies) {
        Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
    }

    if (-not $SkipUpdates) {
        Write-Section 'Updates'
        # Both are best effort. A machine that cannot reach the update service
        # is still worth probing, and failing here would waste the operator's time.
        try {
            Write-Host 'Update-MpSignature ...'
            Update-MpSignature -ErrorAction Stop
        } catch { Write-Warning "Update-MpSignature failed: $_" }
        try {
            Write-Host 'winget upgrade --all ...'
            & winget upgrade --all --accept-source-agreements --accept-package-agreements --silent 2>&1 |
                Tee-Object -FilePath (Join-Path $dir 'winget-upgrade.log') | Out-Null
        } catch { Write-Warning "winget upgrade failed: $_" }
    }

    Write-Section 'Raise security settings'
    # Deliberately only the reversible ones. revert restores each from baseline.
    try {
        Set-MpPreference -DisableRealtimeMonitoring $false
        Set-MpPreference -MAPSReporting Advanced
        Set-MpPreference -SubmitSamplesConsent SendAllSamples
        Set-MpPreference -CloudBlockLevel High
        Set-MpPreference -PUAProtection Enabled
        Write-Host 'Defender: real-time on, MAPS advanced, cloud block level high, PUA on'
    } catch {
        Write-Warning "could not set Defender preferences: $_"
    }

    Write-Section 'CodeIntegrity log'
    # The default 1 MB fills quickly once a policy is auditing every load, and a
    # wrapped log silently loses the events this whole exercise exists to catch.
    try {
        & wevtutil.exe sl Microsoft-Windows-CodeIntegrity/Operational /e:true /ms:67108864
        Write-Host 'CodeIntegrity/Operational enabled, max size 64 MB'
    } catch {
        Write-Warning "could not configure the CodeIntegrity log: $_"
    }

    if ($AuditPolicy) {
        Write-Section 'Audit policy'
        if (-not (Test-Path -LiteralPath $AuditPolicy)) {
            throw "audit policy not found: $AuditPolicy"
        }
        # The NoISG policy checks signatures only and skips the cloud reputation
        # lookup, which is why it works even with Smart App Control off. That is
        # the half of the verdict signing actually changes, so it is the useful
        # one here. It logs 3076 for anything it would refuse.
        & mountvol.exe S: /S
        $dest = "S:\efi\microsoft\boot\cipolicies\active\$NOISG_GUID.cip"
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Copy-Item -LiteralPath $AuditPolicy -Destination $dest -Force
        & CiTool.exe -r
        Write-Host "applied $(Split-Path $AuditPolicy -Leaf) as $NOISG_GUID and refreshed policy"

        $baseline.AuditPolicyApplied = $true
        $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $dir 'baseline.json') -Encoding UTF8

        Write-Section 'Policy state after applying'
        $after = Get-SacState
        foreach ($p in $after.Policies) {
            Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
        }
    } else {
        Write-Host ''
        Write-Host 'No -AuditPolicy given. Download the sample policies from https://aka.ms/sacauditpolicies'
        Write-Host 'and re-run with -AuditPolicy <path to SmartAppControlAuditNoISG.bin> to log what'
        Write-Host 'would be blocked. Without it, only real enforcement (3077) shows up, and only on a'
        Write-Host 'machine where Smart App Control is genuinely on.'
    }

    # Marks the window collect will export. Written last so it covers the run.
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $dir 'window-start.txt') -Encoding UTF8
    Write-Host ''
    Write-Host "prepare complete. Next: .\sac-probe.ps1 -Stage run -Label $Label"
}

function Get-SignatureInventory([string] $root) {
    if (-not (Test-Path -LiteralPath $root)) {
        Write-Warning "no llama.cpp install at $root"
        return @()
    }
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $PE_EXT -contains $_.Extension.ToLowerInvariant() } |
        ForEach-Object {
            $sig = Get-AuthenticodeSignature -LiteralPath $_.FullName
            [pscustomobject]@{
                Name       = $_.Name
                FullName   = $_.FullName
                Length     = $_.Length
                SHA256     = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                Status     = [string]$sig.Status
                # Status alone is not enough. An unsigned file and one whose
                # chain did not build both report UnknownError.
                StatusMessage = $sig.StatusMessage
                Subject    = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
                Thumbprint = if ($sig.SignerCertificate) { $sig.SignerCertificate.Thumbprint } else { $null }
                TimeStamped = if ($sig.TimeStamperCertificate) { $true } else { $false }
            }
        }
}

function Invoke-Run {
    Assert-Elevated
    $dir = Get-RunDir

    Write-Section 'Signature inventory'
    $inventory = @(Get-SignatureInventory $LLAMA_DIR)
    $inventory | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'signature-inventory.json') -Encoding UTF8
    $inventory | Export-Csv -LiteralPath (Join-Path $dir 'signature-inventory.csv') -NoTypeInformation -Encoding UTF8

    $total = $inventory.Count
    $valid = @($inventory | Where-Object { $_.Status -eq 'Valid' }).Count
    Write-Host "$valid of $total PE files report a valid Authenticode signature"
    if ($total -gt 0 -and $valid -lt $total) {
        Write-Host 'unsigned or unverifiable:' -ForegroundColor Yellow
        $inventory | Where-Object { $_.Status -ne 'Valid' } |
            Select-Object Name, Status | Format-Table -AutoSize | Out-String | Write-Host
    }

    if ($SkipStudio) {
        Write-Host 'skipping the Studio scenario (-SkipStudio)'
    } else {
        Write-Section 'Studio scenario'
        $scenario = Join-Path $PSScriptRoot 'studio_scenario.py'
        $log = Join-Path $dir 'studio-scenario.log'
        Write-Host "python $scenario --model $Model --out $dir"
        # $ErrorActionPreference is Stop for the script, but a scenario that
        # fails is a result rather than an accident: that is what we came to
        # measure. Capture it and carry on to collect.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & python $scenario --model $Model --out $dir 2>&1 | Tee-Object -FilePath $log
            Write-Host "scenario exit code: $LASTEXITCODE"
        } catch {
            Write-Warning "scenario failed: $_"
        } finally {
            $ErrorActionPreference = $prev
        }
    }

    Write-Host ''
    Write-Host "run complete. Next: .\sac-probe.ps1 -Stage collect -Label $Label"
}

function Invoke-Collect {
    Assert-Elevated
    $dir = Get-RunDir

    $startPath = Join-Path $dir 'window-start.txt'
    $start = if (Test-Path -LiteralPath $startPath) {
        [datetime]::Parse((Get-Content -LiteralPath $startPath -Raw).Trim())
    } else {
        (Get-Date).AddHours(-2)
    }
    Write-Section "Events since $($start.ToString('o'))"

    $events = @()
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
            LogName   = 'Microsoft-Windows-CodeIntegrity/Operational'
            StartTime = $start
        } -ErrorAction Stop | Where-Object { $CI_EVENT_IDS -contains $_.Id })
    } catch {
        Write-Warning "no CodeIntegrity events in the window: $_"
    }

    $shaped = $events | ForEach-Object {
        [pscustomobject]@{
            TimeCreated = $_.TimeCreated.ToString('o')
            Id          = $_.Id
            # 3077 is an enforced block; 3076 only says it would have been.
            Kind        = switch ($_.Id) {
                3076 { 'audit-would-block' }
                3077 { 'ENFORCED-BLOCK' }
                3089 { 'signature-detail' }
                default { 'context' }
            }
            ActivityID  = $_.ActivityId
            Message     = $_.Message
        }
    }
    $shaped | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.json') -Encoding UTF8
    $shaped | Format-List | Out-String |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.txt') -Encoding UTF8

    $blocks = @($shaped | Where-Object { $_.Id -eq 3077 }).Count
    $audits = @($shaped | Where-Object { $_.Id -eq 3076 }).Count
    Write-Host "$blocks enforced block(s) (3077), $audits audit would-block(s) (3076), $($shaped.Count) event(s) total"

    # Whole-log export as well, since the shaped view drops fields and a
    # reviewer may need the raw record.
    try {
        & wevtutil.exe epl Microsoft-Windows-CodeIntegrity/Operational (Join-Path $dir 'CodeIntegrity-Operational.evtx') /ow:true
    } catch { Write-Warning "evtx export failed: $_" }

    try {
        Get-MpThreatDetection -ErrorAction Stop |
            Select-Object InitialDetectionTime, ThreatID, Resources |
            ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
    } catch {
        '[]' | Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
    }

    Get-SacState | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $dir 'sac-state-after.json') -Encoding UTF8

    # Studio's own logs, which carry the request timings and the backend errors.
    $studioLogs = Join-Path $env:USERPROFILE '.unsloth\studio\logs'
    if (Test-Path -LiteralPath $studioLogs) {
        Copy-Item -LiteralPath $studioLogs -Destination (Join-Path $dir 'studio-logs') -Recurse -Force -ErrorAction SilentlyContinue
    }

    $zip = Join-Path $WorkDir ("unsloth-sac-{0}-{1}-{2}.zip" -f $env:COMPUTERNAME, $Label, (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Compress-Archive -Path (Join-Path $dir '*') -DestinationPath $zip -Force
    Write-Host ''
    Write-Host "evidence: $zip" -ForegroundColor Green
    Write-Host 'Attach that zip to the pull request. It contains your user name in file paths;'
    Write-Host 'redact it if you like, but keep the file names and the rest of each path.'
    Write-Host ''
    Write-Host "When you are done: .\sac-probe.ps1 -Stage revert -Label $Label"
}

function Invoke-Revert {
    Assert-Elevated
    $dir = Get-RunDir
    $baselinePath = Join-Path $dir 'baseline.json'
    if (-not (Test-Path -LiteralPath $baselinePath)) {
        throw "no baseline at $baselinePath; nothing to revert to. Was -Label $Label used for prepare?"
    }
    $baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json

    if ($baseline.AuditPolicyApplied) {
        Write-Section 'Remove audit policy'
        & mountvol.exe S: /S
        $dest = "S:\efi\microsoft\boot\cipolicies\active\$NOISG_GUID.cip"
        if (Test-Path -LiteralPath $dest) {
            Remove-Item -LiteralPath $dest -Force
            & CiTool.exe -r
            Write-Host 'audit policy removed and policy refreshed'
        } else {
            Write-Host 'audit policy already absent'
        }
    }

    Write-Section 'Restore Defender preferences'
    # Only what prepare changed, and only where a baseline value was captured.
    try {
        if ($null -ne $baseline.DisableRealtimeMonitoring) { Set-MpPreference -DisableRealtimeMonitoring $baseline.DisableRealtimeMonitoring }
        if ($null -ne $baseline.MAPSReporting)        { Set-MpPreference -MAPSReporting $baseline.MAPSReporting }
        if ($null -ne $baseline.SubmitSamplesConsent) { Set-MpPreference -SubmitSamplesConsent $baseline.SubmitSamplesConsent }
        if ($null -ne $baseline.CloudBlockLevel)      { Set-MpPreference -CloudBlockLevel $baseline.CloudBlockLevel }
        if ($null -ne $baseline.PUAProtection)        { Set-MpPreference -PUAProtection $baseline.PUAProtection }
        Write-Host 'Defender preferences restored'
    } catch {
        Write-Warning "could not restore Defender preferences: $_"
    }

    Write-Host ''
    Write-Host 'revert complete. Smart App Control itself was never changed by this script.'
}

switch ($Stage) {
    'prepare' { Invoke-Prepare }
    'run'     { Invoke-Run }
    'collect' { Invoke-Collect }
    'revert'  { Invoke-Revert }
}
