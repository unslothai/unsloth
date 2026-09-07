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

    # prepare only: skip the Defender signature update, which is slow.
    [switch] $SkipUpdates,

    # prepare only: also run winget upgrade --all. Off by default because revert
    # cannot undo it: the baseline records no package versions.
    [switch] $UpgradePackages,

    # run only: skip the Studio scenario and just do the signature inventory.
    [switch] $SkipStudio,

    # prepare only: do not install Unsloth Studio when it is missing.
    [switch] $SkipInstall,

    # Port the probe expects Studio on.
    [int] $Port = 8888
)

$ErrorActionPreference = 'Stop'

# The NoISG audit policy lives in the EFI system partition; the full audit
# policy replaces the active runtime policy. Both GUIDs and both destinations
# are Microsoft's, from the Smart App Control testing documentation.
$NOISG_GUID = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'
$NOISG_DEST = "S:\efi\microsoft\boot\cipolicies\active\$NOISG_GUID.cip"
$CI_LOG = 'Microsoft-Windows-CodeIntegrity/Operational'

# Where Studio keeps its state, resolved the way Studio resolves it: the
# UNSLOTH_STUDIO_HOME override (STUDIO_HOME alias), else the legacy home.
function Get-StudioHome {
    $override = if ($env:UNSLOTH_STUDIO_HOME) { $env:UNSLOTH_STUDIO_HOME } else { $env:STUDIO_HOME }
    if ($override) { return $override }
    return (Join-Path $env:USERPROFILE '.unsloth\studio')
}

# The runtime Studio actually loads. UNSLOTH_LLAMA_CPP_PATH is an explicit
# install dir; a custom Studio home keeps its runtime under <home>\llama.cpp;
# only the legacy home uses ~\.unsloth\llama.cpp. Inventorying the wrong tree
# would omit exactly the files the events are about.
function Get-LlamaDir {
    if ($env:UNSLOTH_LLAMA_CPP_PATH) { return $env:UNSLOTH_LLAMA_CPP_PATH }
    $override = if ($env:UNSLOTH_STUDIO_HOME) { $env:UNSLOTH_STUDIO_HOME } else { $env:STUDIO_HOME }
    if ($override) { return (Join-Path $override 'llama.cpp') }
    return (Join-Path $env:USERPROFILE '.unsloth\llama.cpp')
}
$LLAMA_DIR = Get-LlamaDir
$PE_EXT = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')
# 3076 audit, 3077 enforced block, 3089 signature detail, 3033/3099 policy and
# validation failures, 3090/3091/3092 allow-and-origin context.
$CI_EVENT_IDS = @(3033, 3076, 3077, 3089, 3090, 3091, 3092, 3099)

# Native commands do not throw under $ErrorActionPreference = 'Stop' in Windows
# PowerShell; their exit code has to be read, or a failed mount or refresh
# reads as success and a later absence of events as an allow verdict.
function Invoke-Native([string] $Exe, [string[]] $Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') exited $LASTEXITCODE"
    }
}

# The audit policy lives in the EFI system partition. Mount it only if S: is
# not already something, and hand back whether this call mounted it so the
# caller's finally can unmount exactly what it mounted.
function Mount-Efi {
    if (Test-Path -LiteralPath 'S:\') {
        # S: is already something. Only the EFI system partition may be used as
        # is: the policy tree written onto a data or network volume installs
        # nothing, and the missing 3076 events would then read as an allow.
        if (-not (Test-Path -LiteralPath 'S:\EFI\Microsoft\Boot')) {
            throw 'S: is mapped to a volume that is not the EFI system partition; free the drive letter and run again'
        }
        return $false
    }
    Invoke-Native 'mountvol.exe' @('S:', '/S')
    return $true
}

function Dismount-Efi([bool] $Mounted) {
    if ($Mounted) {
        & mountvol.exe S: /D
        if ($LASTEXITCODE -ne 0) { Write-Warning "could not unmount S: (mountvol exited $LASTEXITCODE)" }
    }
}

function Test-PolicyActive([string] $Guid) {
    $bare = $Guid.Trim('{', '}').ToLowerInvariant()
    foreach ($p in (Get-SacState).Policies) {
        if ((([string]$p.PolicyID) -replace '[{}]', '') -eq $bare) { return $true }
    }
    return $false
}

# `wevtutil gl` prints `enabled: true` and `maxSize: 1052672`; both are
# restored by revert, so prepare must record them.
function Get-CiLogSettings {
    $enabled = $null
    $maxSize = $null
    try {
        $lines = & wevtutil.exe gl $CI_LOG 2>$null
        if ($LASTEXITCODE -eq 0) {
            foreach ($line in $lines) {
                if ($line -match '^\s*enabled:\s*(\S+)') { $enabled = ($Matches[1] -eq 'true') }
                if ($line -match '^\s*maxSize:\s*(\d+)') { $maxSize = [int64]$Matches[1] }
            }
        }
    } catch { }
    return [pscustomobject]@{ Enabled = $enabled; MaxSize = $maxSize }
}

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

function Get-StudioPython {
    <#
        The managed interpreter, or $null when Studio is not installed.

        Deliberately not the generated unsloth.exe. Windows materialises the
        console script as an unsigned PE, and AppLocker, WDAC and Smart App
        Control deny it while the signed interpreter beside it keeps running
        (issue 8490). On exactly the machines this probe targets, calling
        unsloth.exe would fail for a reason that has nothing to do with what we
        are measuring.
    #>
    # The same precedence as Get-StudioHome, so a Studio configured through
    # the STUDIO_HOME alias is found rather than reinstalled beside itself.
    $legacy = Join-Path $env:USERPROFILE '.unsloth\studio'
    $roots = @((Get-StudioHome))
    if ($roots[0] -ne $legacy) { $roots += $legacy }
    foreach ($root in $roots) {
        $py = Join-Path $root 'unsloth_studio\Scripts\python.exe'
        if (Test-Path -LiteralPath $py) { return $py }
    }
    return $null
}

function Test-StudioResponding([int] $port) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/liveness" -TimeoutSec 5 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Install-Studio {
    Write-Section 'Install Unsloth Studio'
    # The documented install command, run exactly as a user would. Piping keeps
    # the script off disk, which matters here: the copy served from unsloth.ai
    # is not Authenticode signed, so downloading it first would attach
    # Mark-of-the-Web and the default RemoteSigned policy would refuse to run
    # it. Reproducing the user's real path is also the point.
    Write-Host 'irm https://unsloth.ai/install.ps1 | iex'
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Invoke-Expression (Invoke-RestMethod -Uri 'https://unsloth.ai/install.ps1' -TimeoutSec 120)
    } catch {
        Write-Warning "installer failed: $_"
    } finally {
        $ErrorActionPreference = $prev
    }
    return (Get-StudioPython)
}

function Start-Studio([string] $python, [int] $port, [string] $logPath) {
    Write-Host "starting Studio on port $port"
    # -X utf8 -I -m unsloth_cli is the supported entry point on a locked-down
    # machine, per unsloth_cli/__main__.py. -I drops the working directory from
    # sys.path so a stray unsloth_cli folder cannot shadow the package.
    # Not $args: that is a PowerShell automatic variable.
    $cliArgs = @('-X', 'utf8', '-I', '-m', 'unsloth_cli', 'studio', '-p', "$port")
    Start-Process -FilePath $python -ArgumentList $cliArgs `
        -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" `
        -WindowStyle Hidden | Out-Null

    # Studio imports torch on a warm thread, so first start is slow. Poll rather
    # than sleep, and give it long enough that a slow machine is not called dead.
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 5
        if (Test-StudioResponding $port) {
            Write-Host "Studio answering on port $port after $($i * 5)s"
            return $true
        }
    }
    Write-Warning "Studio did not answer on port $port within 5 minutes; see $logPath"
    return $false
}

function Initialize-Studio([string] $dir) {
    <# Ensure a Studio exists and is answering, installing it if needed. #>
    Write-Section 'Unsloth Studio'
    if (Test-StudioResponding $Port) {
        Write-Host "Studio already answering on port $Port"
        return
    }

    $python = Get-StudioPython
    if (-not $python) {
        if ($SkipInstall) {
            Write-Warning 'Studio is not installed and -SkipInstall was given; the run stage will have nothing to drive.'
            return
        }
        Write-Host 'Studio is not installed on this machine.'
        $python = Install-Studio
        if (-not $python) {
            Write-Warning 'Studio still not found after the installer ran. Install it by hand, then re-run this stage.'
            return
        }
    }
    Write-Host "managed interpreter: $python"
    Start-Studio $python $Port (Join-Path $dir 'studio-start.log') | Out-Null
}

function Save-Baseline([string] $dir) {
    $mp = $null
    try { $mp = Get-MpPreference } catch { }
    $status = $null
    try { $status = Get-MpComputerStatus } catch { }
    $ciLog = Get-CiLogSettings

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
        CiLogEnabled            = $ciLog.Enabled
        CiLogMaxSize            = $ciLog.MaxSize
        AuditPolicyApplied      = $false
        # A policy with the NoISG GUID that was there before prepare: kept
        # aside and put back by revert rather than deleted as ours.
        AuditPolicyPreexisting  = $false
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
    $baselinePath = Join-Path $dir 'baseline.json'
    if (Test-Path -LiteralPath $baselinePath) {
        # A retry of this label. The machine already carries whatever the first
        # pass changed, so snapshotting it again would record the raised
        # settings, and our own policy, as the state revert should restore.
        $baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
        Write-Warning "reusing the baseline captured at $($baseline.CapturedAt) by an earlier prepare of this label; revert restores that state"
    } else {
        $baseline = Save-Baseline $dir
    }
    Write-Host ("Smart App Control: {0} (registry state {1})" -f $baseline.Sac.Mode, $baseline.Sac.RegistryState)
    foreach ($p in $baseline.Sac.Policies) {
        Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
    }

    if (-not $SkipUpdates) {
        Write-Section 'Updates'
        # Best effort. A machine that cannot reach the update service is still
        # worth probing, and failing here would waste the operator's time.
        try {
            Write-Host 'Update-MpSignature ...'
            Update-MpSignature -ErrorAction Stop
        } catch { Write-Warning "Update-MpSignature failed: $_" }
    }
    if ($UpgradePackages) {
        # Opt-in: this changes every winget-managed package on the machine and
        # revert cannot put them back, so it is not part of the reversible run.
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
        Invoke-Native 'wevtutil.exe' @('sl', $CI_LOG, '/e:true', '/ms:67108864')
        Write-Host ("CodeIntegrity/Operational enabled, max size 64 MB (was enabled={0}, maxSize={1})" -f $baseline.CiLogEnabled, $baseline.CiLogMaxSize)
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
        $mounted = Mount-Efi
        try {
            # A file already there is somebody's policy only if this label did
            # not put it there: on a retry it is ours, and saving it as
            # pre-existing would make revert reinstall it.
            if ((Test-Path -LiteralPath $NOISG_DEST) -and -not $baseline.AuditPolicyApplied) {
                # Keep it so revert restores it instead of deleting an
                # administrator's policy.
                Copy-Item -LiteralPath $NOISG_DEST -Destination (Join-Path $dir 'preexisting-policy.cip') -Force
                $baseline.AuditPolicyPreexisting = $true
                Write-Warning "a policy with $NOISG_GUID was already installed; saved to preexisting-policy.cip and will be restored by revert"
            }
            New-Item -ItemType Directory -Force -Path (Split-Path $NOISG_DEST) | Out-Null
            Copy-Item -LiteralPath $AuditPolicy -Destination $NOISG_DEST -Force
            # Persisted before the refresh: if CiTool fails, the copied file is
            # on the EFI partition and revert must know to remove or restore it.
            $baseline.AuditPolicyApplied = $true
            $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
            Invoke-Native 'CiTool.exe' @('-r')
        } finally {
            Dismount-Efi $mounted
        }
        Write-Host "applied $(Split-Path $AuditPolicy -Leaf) as $NOISG_GUID and refreshed policy"

        Write-Section 'Policy state after applying'
        $after = Get-SacState
        foreach ($p in $after.Policies) {
            Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
        }
        # Printing whatever is listed is not verification. If the policy did
        # not take, a run with no 3076 events would read as an allow verdict
        # when it was an invalid setup. Only checkable where CiTool exists.
        if ($after.Policies.Count -gt 0 -and -not (Test-PolicyActive $NOISG_GUID)) {
            throw "the audit policy $NOISG_GUID is not in the active policy set after refresh; the machine is left as prepare found it apart from the copied file, run revert"
        }
        if ($after.Policies.Count -eq 0) {
            Write-Warning 'CiTool is not available here, so the policy could not be verified as active; read the 3076 count with that in mind'
        }
    } else {
        Write-Host ''
        Write-Host 'No -AuditPolicy given. Download the sample policies from https://aka.ms/sacauditpolicies'
        Write-Host 'and re-run with -AuditPolicy <path to SmartAppControlAuditNoISG.bin> to log what'
        Write-Host 'would be blocked. Without it, only real enforcement (3077) shows up, and only on a'
        Write-Host 'machine where Smart App Control is genuinely on.'
    }

    # Marks the window collect will export. Set before Studio is installed and
    # started on purpose: the install downloads the llama.cpp bundle and the
    # first launch loads it, so both are inside the observed window and any
    # code integrity event they raise is captured.
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $dir 'window-start.txt') -Encoding UTF8

    Initialize-Studio $dir

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

    # A machine that was prepared earlier may have been rebooted since, which is
    # itself part of the reported behaviour: Smart App Control re-evaluates from
    # a cleared cache after a restart. Bring Studio back rather than failing.
    # Not under -SkipStudio: a signature-only run must not start, let alone
    # install, Studio inside the evidence window.
    if (-not $SkipStudio -and -not (Test-StudioResponding $Port)) { Initialize-Studio $dir }

    Write-Section 'Signature inventory'
    Write-Host "runtime: $LLAMA_DIR"
    $inventory = @(Get-SignatureInventory $LLAMA_DIR)
    # -InputObject: an empty pipeline writes an empty file, not `[]`.
    ConvertTo-Json -InputObject @($inventory) -Depth 4 |
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
        if (-not $env:UNSLOTH_STUDIO_PASSWORD) {
            Write-Warning 'UNSLOTH_STUDIO_PASSWORD is not set; the scenario needs it (see README) and will stop at login'
        }
        Write-Host "python $scenario --model $Model --out $dir"
        # $ErrorActionPreference is Stop for the script, but a scenario that
        # fails is a result rather than an accident: that is what we came to
        # measure. Capture it and carry on to collect.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & python $scenario --model $Model --out $dir --port $Port 2>&1 | Tee-Object -FilePath $log
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
    if (-not (Test-Path -LiteralPath $startPath)) {
        # Get-RunDir creates the directory, so a mistyped label or a collect
        # without a prepare lands here. Inventing a window would export
        # unrelated events as evidence for this scenario.
        throw "no window-start.txt under ${dir}: prepare did not run for label '$Label', so there is no event window to collect"
    }
    $start = [datetime]::Parse((Get-Content -LiteralPath $startPath -Raw).Trim())
    Write-Section "Events since $($start.ToString('o'))"

    $events = @()
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
            LogName   = $CI_LOG
            StartTime = $start
        } -ErrorAction Stop | Where-Object { $CI_EVENT_IDS -contains $_.Id })
    } catch {
        Write-Warning "no CodeIntegrity events in the window: $_"
    }

    $shaped = @($events | ForEach-Object {
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
    })
    # -InputObject: zero events is a normal and important result for an
    # allowed bundle, and piping an empty array writes an empty file that no
    # consumer can tell from a failed collection. This writes `[]`.
    ConvertTo-Json -InputObject $shaped -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.json') -Encoding UTF8
    $shaped | Format-List | Out-String |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.txt') -Encoding UTF8

    $blocks = @($shaped | Where-Object { $_.Id -eq 3077 }).Count
    $audits = @($shaped | Where-Object { $_.Id -eq 3076 }).Count
    Write-Host "$blocks enforced block(s) (3077), $audits audit would-block(s) (3076), $($shaped.Count) event(s) total"

    # Whole-log export as well, since the shaped view drops fields and a
    # reviewer may need the raw record.
    try {
        Invoke-Native 'wevtutil.exe' @('epl', $CI_LOG, (Join-Path $dir 'CodeIntegrity-Operational.evtx'), '/ow:true')
    } catch { Write-Warning "evtx export failed: $_" }

    try {
        # Captured into an array first: a clean machine yields nothing, and a
        # pipeline with no input writes an empty file rather than [].
        $detections = @(Get-MpThreatDetection -ErrorAction Stop | Select-Object InitialDetectionTime, ThreatID, Resources)
        ConvertTo-Json -InputObject $detections -Depth 4 |
            Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
    } catch {
        '[]' | Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
    }

    Get-SacState | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $dir 'sac-state-after.json') -Encoding UTF8

    # Studio's own logs, which carry the request timings and the backend errors.
    $studioLogs = Join-Path (Get-StudioHome) 'logs'
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
        $saved = Join-Path $dir 'preexisting-policy.cip'
        $mounted = Mount-Efi
        try {
            if ($baseline.AuditPolicyPreexisting -and (Test-Path -LiteralPath $saved)) {
                # Not ours to delete: put back the policy prepare found.
                Copy-Item -LiteralPath $saved -Destination $NOISG_DEST -Force
                Invoke-Native 'CiTool.exe' @('-r')
                Write-Host 'pre-existing audit policy restored and policy refreshed'
            } elseif (Test-Path -LiteralPath $NOISG_DEST) {
                Remove-Item -LiteralPath $NOISG_DEST -Force
                Invoke-Native 'CiTool.exe' @('-r')
                Write-Host 'audit policy removed and policy refreshed'
            } else {
                Write-Host 'audit policy already absent'
            }
        } finally {
            Dismount-Efi $mounted
        }
    }

    Write-Section 'Restore CodeIntegrity log'
    # Only where prepare recorded a value; an absent baseline field is an
    # older baseline.json and the setting is left alone.
    try {
        if ($null -ne $baseline.CiLogMaxSize -and $null -ne $baseline.CiLogEnabled) {
            $enabled = if ($baseline.CiLogEnabled) { 'true' } else { 'false' }
            Invoke-Native 'wevtutil.exe' @('sl', $CI_LOG, "/e:$enabled", "/ms:$($baseline.CiLogMaxSize)")
            Write-Host ("CodeIntegrity/Operational restored: enabled={0}, maxSize={1}" -f $enabled, $baseline.CiLogMaxSize)
        } else {
            Write-Host 'no CodeIntegrity log baseline recorded; left as is'
        }
    } catch {
        Write-Warning "could not restore the CodeIntegrity log settings: $_"
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
