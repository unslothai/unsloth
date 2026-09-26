# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
<#
.SYNOPSIS
    Collect evidence about Windows code integrity blocking the Unsloth Studio
    llama.cpp runtime.

.DESCRIPTION
    Stages prepare, run, collect, revert. Never toggles Smart App Control (one-way);
    only adds and removes a reversible audit policy. 3077 = enforced block,
    3076 = audit only, 3089 matched to 3077 by ActivityID, never by timestamp.

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
    [string] $AuditPolicy,

    # Label for this run, so the four matrix cells do not overwrite each other.
    [string] $Label = 'run',

    [string] $Model = 'unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL',

    # prepare only: skip the Defender signature update, which is slow.
    [switch] $SkipUpdates,

    [switch] $UpgradePackages,

    # run only: skip the Studio scenario and just do the signature inventory.
    [switch] $SkipStudio,

    # prepare only: do not install Unsloth Studio when it is missing.
    [switch] $SkipInstall,

    [switch] $SendSamples,

    [int] $Port = 8888
)

$ErrorActionPreference = 'Stop'

$NOISG_GUID = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'
$NOISG_DEST = "S:\efi\microsoft\boot\cipolicies\active\$NOISG_GUID.cip"
$CI_LOG = 'Microsoft-Windows-CodeIntegrity/Operational'

# A configured path the way Studio normalises one (utils/paths/storage_roots.studio_root)
function Resolve-ConfiguredPath([string] $value) {
    if (-not $value) { return $null }
    $trimmed = $value.Trim()
    if (-not $trimmed) { return $null }
    $userHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
    if ($trimmed -eq '~') { $trimmed = $userHome }
    elseif ($trimmed.StartsWith('~\') -or $trimmed.StartsWith('~/')) {
        $trimmed = Join-Path $userHome $trimmed.Substring(2)
    }
    try { return [IO.Path]::GetFullPath($trimmed) } catch { return $trimmed }
}

# The configured Studio home, in Studio's precedence
function Get-StudioHomeOverride {
    $override = Resolve-ConfiguredPath $env:UNSLOTH_STUDIO_HOME
    if (-not $override) { $override = Resolve-ConfiguredPath $env:STUDIO_HOME }
    return $override
}

function Get-StudioHome {
    $override = Get-StudioHomeOverride
    if ($override) { return $override }
    return (Join-Path $env:USERPROFILE '.unsloth\studio')
}

# The runtime Studio actually loads, in Studio's own order (llama_cpp.py _find_llama_server_binary)
function Get-LlamaDir {
    $binary = Resolve-ConfiguredPath $env:LLAMA_SERVER_PATH
    if ($binary) { return (Split-Path -Parent $binary) }
    $installDir = Resolve-ConfiguredPath $env:UNSLOTH_LLAMA_CPP_PATH
    if ($installDir) { return $installDir }
    $override = Get-StudioHomeOverride
    if ($override) { return (Join-Path $override 'llama.cpp') }
    return (Join-Path $env:USERPROFILE '.unsloth\llama.cpp')
}
# The managed venv.
$VENV_DIR = Join-Path (Get-StudioHome) 'unsloth_studio'

function Resolve-LlamaDir([string] $dir) {
    $selection = Join-Path $dir 'runtime-selection.json'
    if (Test-Path -LiteralPath $selection) {
        $sel = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
        if ($sel.resolved_binary) {
            Write-Host ("runtime selected by Studio ({0}): {1}" -f $sel.source, $sel.resolved_binary)
            # The selected root, not the binary's directory.
            if ($sel.path -and (Test-Path -LiteralPath $sel.path -PathType Container)) {
                return $sel.path
            }
            return (Split-Path -Parent $sel.resolved_binary)
        }
        Write-Warning "Studio reported no resolvable llama-server (source $($sel.source), path $($sel.path)); inventorying $(Get-LlamaDir)"
    } elseif (-not $env:LLAMA_SERVER_PATH -and -not $env:UNSLOTH_LLAMA_CPP_PATH) {
        Write-Warning 'no runtime-selection.json from the scenario: a folder selected in Studio settings cannot be seen from here, so the managed default is inventoried'
    }
    return (Get-LlamaDir)
}
$PE_EXT = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')

# unsloth_cli reads UNSLOTH_STUDIO_PASSWORD itself and treats it as set-the-initial-password, so launching with it set hard-errors on any Studio that already has one
$STUDIO_PASSWORD = $env:UNSLOTH_STUDIO_PASSWORD
Remove-Item Env:\UNSLOTH_STUDIO_PASSWORD -ErrorAction SilentlyContinue
$CI_EVENT_IDS = @(3033, 3076, 3077, 3089, 3090, 3091, 3092, 3099)

# Native commands do not throw under $ErrorActionPreference = 'Stop' in Windows PowerShell
function Invoke-Native([string] $Exe, [string[]] $Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') exited $LASTEXITCODE"
    }
}

# The audit policy lives in the EFI system partition.
$EFI_OWNED_MARKER = Join-Path $WorkDir '.efi-mounted-by-probe'

function Mount-Efi {
    if (Test-Path -LiteralPath 'S:\') {
        if (-not (Test-Path -LiteralPath 'S:\EFI\Microsoft\Boot')) {
            throw 'S: is mapped to a volume that is not the EFI system partition; free the drive letter and run again'
        }
        if (Test-Path -LiteralPath $EFI_OWNED_MARKER) {
            Write-Warning 'S: is the EFI system partition left mounted by an earlier stage of this probe; this stage will unmount it'
            return $true
        }
        return $false
    }
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
    Invoke-Native 'mountvol.exe' @('S:', '/S')
    (Get-Date).ToString('o') | Set-Content -LiteralPath $EFI_OWNED_MARKER -Encoding UTF8
    return $true
}

# Set when a dismount failed, and read by prepare and revert before either reports success.
$script:EfiStillMounted = $false

function Dismount-Efi([bool] $Mounted) {
    if ($Mounted) {
        & mountvol.exe S: /D
        if ($LASTEXITCODE -ne 0) {
            $script:EfiStillMounted = $true
            Write-Warning "could not unmount S: (mountvol exited $LASTEXITCODE)"
        } else {
            Remove-Item -LiteralPath $EFI_OWNED_MARKER -Force -ErrorAction SilentlyContinue
        }
    }
}

# Retries a dismount an earlier stage could not complete.
function Clear-EfiOwnership {
    if (-not (Test-Path -LiteralPath $EFI_OWNED_MARKER)) { return }
    if (-not (Test-Path -LiteralPath 'S:\')) {
        Remove-Item -LiteralPath $EFI_OWNED_MARKER -Force -ErrorAction SilentlyContinue
        return
    }
    if (-not (Test-Path -LiteralPath 'S:\EFI\Microsoft\Boot')) {
        # Somebody else's volume took the letter after our mount went away.
        Write-Warning 'S: is no longer the EFI system partition; leaving the drive letter alone and dropping this probe''s claim on it'
        Remove-Item -LiteralPath $EFI_OWNED_MARKER -Force -ErrorAction SilentlyContinue
        return
    }
    Write-Warning 'S: was left mounted by an earlier stage of this probe; unmounting it'
    Dismount-Efi $true
}

function Test-PolicyActive([string] $Guid) {
    $bare = $Guid.Trim('{', '}').ToLowerInvariant()
    foreach ($p in (Get-SacState).Policies) {
        if ((([string]$p.PolicyID) -replace '[{}]', '') -eq $bare) { return $true }
    }
    return $false
}

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

# Does a preference that reads back as $actual carry the value $expected asked for? $true match, $false mismatch, $null when it could not be decided.
function Test-MpPreferenceMatch($actual, $expected, [Type] $type) {
    if ($null -eq $actual) { return $false }
    if ($expected -is [bool] -or $actual -is [bool]) { return ([bool]$actual -eq [bool]$expected) }
    if ([string]$actual -eq [string]$expected) { return $true }
    if ($type -and $type.IsEnum) {
        try {
            $want = [int]([Enum]::Parse($type, [string]$expected, $true))
            $have = [int]([Enum]::Parse($type, [string]$actual, $true))
            return ($have -eq $want)
        } catch { }
    }
    return $null
}

function Get-MpPreferenceType([string] $name) {
    try {
        $cmd = Get-Command Set-MpPreference -ErrorAction Stop
        $p = $cmd.Parameters[$name]
        if ($p) { return $p.ParameterType }
    } catch { }
    return $null
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

function Get-RollbackPolicyPath([string] $dir) {
    return (Join-Path (Join-Path $dir 'rollback') 'preexisting-policy.cip')
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
    }
    return [pscustomobject]@{
        RegistryState = $state
        Mode          = $name
        Policies      = $policies
    }
}

function Get-StudioPython {
    <# The managed interpreter. Not unsloth.exe: that console script is an unsigned PE that SAC/WDAC deny (issue 8490). #>
    $legacy = Join-Path $env:USERPROFILE '.unsloth\studio'
    $roots = @((Get-StudioHome))
    if ($roots[0] -ne $legacy) { $roots += $legacy }
    foreach ($root in $roots) {
        $py = Join-Path $root 'unsloth_studio\Scripts\python.exe'
        if (Test-Path -LiteralPath $py) { return $py }
    }
    return $null
}

# The venv the RUNNING Studio uses, not the default location.
function Resolve-VenvDir([string] $dir) {
    # Recorded by run and read back by collect when $dir is given, the way the llama.cpp selection already is.
    if ($dir) {
        $recorded = Join-Path $dir 'venv-selection.txt'
        if (Test-Path -LiteralPath $recorded) {
            $recordedPath = (Get-Content -LiteralPath $recorded -Raw).Trim()
            if ($recordedPath) { return $recordedPath }
        }
    }
    $studioPython = Get-StudioPython
    if ($studioPython) { return (Split-Path -Parent (Split-Path -Parent $studioPython)) }
    return $VENV_DIR
}

function Resolve-StudioHomeFor([string] $dir) {
    # run records the home the backend was launched with.
    if ($dir) {
        $recordedHome = Join-Path $dir 'studio-home.txt'
        if (Test-Path -LiteralPath $recordedHome) {
            $homePath = "$(Get-Content -LiteralPath $recordedHome -Raw)".Trim()
            if ($homePath) { return $homePath }
        }
    }
    $venv = Resolve-VenvDir $dir
    if ($venv) {
        $parent = Split-Path -Parent $venv
        if ($parent) { return $parent }
    }
    return (Get-StudioHome)
}

function Resolve-StudioPythonFor([string] $dir) {
    $venv = Resolve-VenvDir $dir
    if ($venv) {
        $py = Join-Path $venv 'Scripts\python.exe'
        if (Test-Path -LiteralPath $py) { return $py }
    }
    return (Get-StudioPython)
}

function Test-StudioResponding([int] $port) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/liveness" -TimeoutSec 5 -UseBasicParsing
        if ($r.StatusCode -ne 200) { return $false }
        # Identity, not a bare status code.
        $body = $null
        try { $body = $r.Content | ConvertFrom-Json } catch { return $false }
        return ($body.service -eq 'Unsloth UI Backend')
    } catch {
        return $false
    }
}

function Get-ScopeTail([string] $root) {
    $trimmed = ($root -replace '^[A-Za-z]:', '').TrimEnd('\', '/')
    # A runtime at the root of a volume trims to nothing, and the tail would then be a lone separator, which every path in this machine-wide channel contains.
    if (-not $trimmed) {
        throw "cannot scope CodeIntegrity events to '${root}': a runtime or venv at the root of a volume leaves no path tail, so every event on this machine would be counted as an Unsloth verdict. Move it into a subdirectory, point UNSLOTH_LLAMA_CPP_PATH or LLAMA_SERVER_PATH at that, and run this label again from prepare."
    }
    return ([Management.Automation.WildcardPattern]::Escape($trimmed) + '\')
}

function Get-EventDataMap($record) {
    $map = [ordered]@{}
    try {
        $xml = [xml] $record.ToXml()
        $i = 0
        foreach ($node in $xml.Event.EventData.Data) {
            $i++
            $name = if ($node.Name) { $node.Name } else { "Data$i" }
            $map[$name] = $node.'#text'
        }
    } catch {
    }
    return $map
}

function Install-Studio {
    Write-Section 'Install Unsloth Studio'
    # The documented install command, run exactly as a user would.
    Write-Host 'irm https://unsloth.ai/install.ps1 | iex'
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Out-Host, not a bare call.
        Invoke-Expression (Invoke-RestMethod -Uri 'https://unsloth.ai/install.ps1' -TimeoutSec 120) | Out-Host
    } catch {
        Write-Warning "installer failed: $_"
    } finally {
        $ErrorActionPreference = $prev
    }
    return (Get-StudioPython)
}

function Start-Studio([string] $python, [int] $port, [string] $logPath) {
    Write-Host "starting Studio on port $port"
    # -X utf8 -I -m unsloth_cli is the supported entry point on a locked-down machine, per unsloth_cli/__main__.py.
    $cliArgs = @('-X', 'utf8', '-I', '-m', 'unsloth_cli', 'studio', '-p', "$port")
    # One record, so one probe Studio at a time.
    if (-not (Stop-ProbeStudio (Get-RunDir))) {
        throw "the Studio this probe started earlier (see probe-studio.json under $(Get-RunDir)) is still running and could not be stopped; stop it by hand and run this stage again."
    }
    $proc = Start-Process -FilePath $python -ArgumentList $cliArgs `
        -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" `
        -WindowStyle Hidden -PassThru
    try {
        @{ Id = $proc.Id; StartTicks = $proc.StartTime.ToUniversalTime().Ticks } | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path (Get-RunDir) 'probe-studio.json') -Encoding UTF8 -ErrorAction Stop
    } catch {
        $recordError = $_
        Stop-ProcessTree $proc.Id
        throw "could not record the elevated Studio process for revert, so it was stopped rather than left running unrecorded: $recordError"
    }

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

function Stop-Studio([int] $port) {
    <# Stop the Studio answering on $port, children (llama-server, workers) first. #>
    # Only the listeners that can be serving the endpoint Test-StudioResponding actually verified, which is http://127.0.0.1:$port.
    $owners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -eq '127.0.0.1' -or $_.LocalAddress -eq '0.0.0.0' } |
        Select-Object -ExpandProperty OwningProcess -Unique)
    if (-not $owners) {
        Write-Warning "Studio answers on http://127.0.0.1:$port but no local process listens on 127.0.0.1 or 0.0.0.0 there; it cannot be restarted from here"
        return $false
    }
    foreach ($owner in $owners) {
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $owner" -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
    }
    foreach ($i in 1..30) {
        Start-Sleep -Seconds 1
        if (-not (Test-StudioResponding $port)) { return $true }
    }
    Write-Warning "Studio is still answering on port $port after being stopped"
    return $false
}

function Stop-ProcessTree([int] $id) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $id" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree ([int] $_.ProcessId) }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

function Stop-ProbeStudio([string] $dir) {
    <# Stop the elevated Studio this probe started, if it is still that process. Returns $false if it survives. #>
    $record = Join-Path $dir 'probe-studio.json'
    if (-not (Test-Path -LiteralPath $record)) { return $true }
    $r = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $p = Get-Process -Id ([int] $r.Id) -ErrorAction SilentlyContinue
    # Start time guards PID reuse
    if ($p -and $p.StartTime.ToUniversalTime().Ticks -eq [int64] $r.StartTicks) {
        Stop-ProcessTree $p.Id
        if (-not $p.WaitForExit(10000)) { return $false }
    }
    Remove-Item -LiteralPath $record -Force -ErrorAction SilentlyContinue
    return $true
}

function Initialize-Studio([string] $dir, [bool] $allowInstall) {
    <# Only prepare may install. prepare restarts a running Studio so its native loads land inside the event window. #>
    Write-Section 'Unsloth Studio'
    if (Test-StudioResponding $Port) {
        if (-not $allowInstall) {
            Write-Host "Studio already answering on port $Port"
            return
        }
        Write-Host "Studio is already answering on port $Port; restarting it so its startup loads land inside the window"
        if (-not (Stop-Studio $Port)) {
            throw "Studio is still answering on port $Port and could not be restarted, so its startup loads cannot be observed in this window. Stop it by hand and run prepare again."
        }
    }

    $python = Get-StudioPython
    if (-not $python) {
        if (-not $allowInstall) {
            Write-Warning 'Studio is not installed; the run stage does not install it, so there is nothing to drive. Run prepare without -SkipInstall, or install by hand.'
            return
        }
        if ($SkipInstall) {
            Write-Warning 'Studio is not installed and -SkipInstall was given; the run stage will have nothing to drive.'
            return
        }
        Write-Host 'Studio is not installed on this machine.'
        # Which of the trees the installer creates did not exist beforehand.
        $unslothHome = Join-Path $env:USERPROFILE '.unsloth'
        # node and whisper.cpp do NOT always live under %USERPROFILE%\.unsloth.
        $override = Get-StudioHomeOverride
        $installRoots = @($unslothHome, $override, (Split-Path -Parent (Get-LlamaDir))) |
            Where-Object { $_ } | Select-Object -Unique
        $candidates = @(
            $unslothHome,
            (Get-StudioHome),
            (Join-Path (Get-StudioHome) 'unsloth_studio'),
            (Get-LlamaDir)
        ) + @($installRoots | ForEach-Object {
            (Join-Path $_ 'node')
            (Join-Path $_ 'whisper.cpp')
            (Join-Path $_ '.cache')
        }) | Where-Object { $_ } | Select-Object -Unique
        $absentBefore = @($candidates | Where-Object { -not (Test-Path -LiteralPath $_) })
        $python = Install-Studio
        $created = @($absentBefore | Where-Object { Test-Path -LiteralPath $_ })
        $baselinePath = Join-Path $dir 'baseline.json'
        if ($created.Count -gt 0 -and (Test-Path -LiteralPath $baselinePath)) {
            $b = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
            $b.StudioInstalledByProbe = $true
            # Merged, not replaced.
            $b.StudioInstallRoots = @(@($b.StudioInstallRoots) + $created |
                Where-Object { $_ } | Select-Object -Unique)
            Save-ProbeBaseline $b $baselinePath
        }
        if (-not $python) {
            # prepare cannot go on: no Studio to start, so nothing loads inside the window.
            if ($allowInstall) {
                throw 'the installer ran but produced no managed interpreter, so there is no Studio to start and this window would observe nothing. Install Studio by hand and run prepare again; the machine is left prepared, so run revert if you stop here.'
            }
            Write-Warning 'Studio still not found after the installer ran. Install it by hand, then re-run this stage.'
            return
        }
    }
    Write-Host "managed interpreter: $python"
    # Under raw-logs\, which collect redacts into studio-logs\ and never archives as is
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'raw-logs') | Out-Null
    $startLog = Join-Path (Join-Path $dir 'raw-logs') 'studio-start.log'
    # Not discarded.
    if (-not (Start-Studio $python $Port $startLog)) {
        if ($allowInstall) {
            throw "Studio did not answer on port $Port within 5 minutes, so its startup loads cannot be observed in this window. See $startLog, fix the cause and run prepare again."
        }
        Write-Warning "Studio did not answer on port $Port; the scenario will have nothing to drive. See $startLog."
    }
}

function Save-Baseline([string] $dir) {
    $mp = $null
    $mpError = $null
    try { $mp = Get-MpPreference } catch { $mpError = $_ }
    $status = $null
    try { $status = Get-MpComputerStatus } catch { }
    $ciLog = Get-CiLogSettings

    # Fail before mutating anything whose old value we could not read.
    if (-not $mp) {
        throw "could not read the Defender preferences that prepare changes, so revert would not be able to restore them. Refusing to modify this machine. Underlying error: $mpError"
    }
    $missing = @('DisableRealtimeMonitoring', 'MAPSReporting', 'SubmitSamplesConsent', 'CloudBlockLevel', 'PUAProtection') |
        Where-Object { $null -eq $mp.$_ }
    if ($missing.Count -gt 0) {
        throw "Defender preference(s) $($missing -join ', ') read back as null, so revert could not restore them. Refusing to modify this machine."
    }
    if ($null -eq $ciLog.Enabled -or $null -eq $ciLog.MaxSize) {
        throw "could not read the $CI_LOG channel settings (enabled=$($ciLog.Enabled), maxSize=$($ciLog.MaxSize)), which prepare raises and revert restores. Refusing to modify this machine."
    }

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
        # revert repairs ACLs only on trees this run created; see Invoke-Revert.
        StudioInstalledByProbe  = $false
        StudioInstallRoots      = @()
        AuditPolicyApplied      = $false
        # $true when an unsigned control raised 3076/3077 under the applied policy, $false never (prepare throws), $null when no control could be built.
        AuditPolicyControlFired = $null
        # The same question for the cell that runs with no -AuditPolicy at all, where the only thing that can produce a verdict is Smart App Control enforcing for real.
        SacControlFired         = $null
        AuditPolicyPreexisting  = $false
        RevertCompletedAt       = $null
    }
    $path = Join-Path $dir 'baseline.json'
    Save-ProbeBaseline $baseline $path
    Write-Host "baseline written to $path"
    return $baseline
}

# Proves the audit policy is not merely listed but actually evaluating loads.
function Save-ProbeBaseline($Baseline, [string] $Path) {
    $tmp = "$Path.tmp"
    $Baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $tmp -Encoding UTF8
    if (Test-Path -LiteralPath $Path) {
        [System.IO.File]::Replace($tmp, $Path, [NullString]::Value)
    } else {
        Move-Item -LiteralPath $tmp -Destination $Path
    }
}

function Test-EventFromPolicy($record, [string] $Guid, [string] $NamePattern = '*AuditNoISG*') {
    return (Test-EventDataFromPolicy (Get-EventDataMap $record) $Guid $NamePattern)
}

function Test-EventDataFromPolicy($data, [string] $Guid, [string] $NamePattern = '*AuditNoISG*') {
    if ($null -eq $data) { return $false }
    $bare = $Guid.Trim('{', '}').ToLowerInvariant()
    foreach ($key in @($data.Keys)) {
        $value = [string] $data[$key]
        if (-not $value) { continue }
        if ($value.ToLowerInvariant().Contains($bare)) { return $true }
        if ($key -like 'PolicyName*' -and $NamePattern -and $value -like $NamePattern) { return $true }
    }
    return $false
}

function Test-AuditPolicyEvaluating([int[]] $AcceptIds = @(3076, 3077), [string] $FromPolicy = $null) {
    $dir = Join-Path $WorkDir ".control-$Label"
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    try {
        $control = Join-Path $dir 'unsigned-control.exe'
        $since = (Get-Date).AddSeconds(-1)
        try {
            # Windows PowerShell 5.1 only.
            $src = 'public class UnsignedControl { public static void Main() { System.Console.WriteLine("control"); } }'
            Add-Type -TypeDefinition $src -OutputAssembly $control -OutputType ConsoleApplication -ErrorAction Stop
        } catch {
            Write-Warning "could not build an unsigned control binary ($_); the audit policy is listed as active but has NOT been shown to evaluate loads here. Run prepare from Windows PowerShell 5.1 for a verified cell."
            return $null
        }
        if (-not (Test-Path -LiteralPath $control)) {
            Write-Warning 'Add-Type reported success but produced no control binary; the audit policy has NOT been shown to evaluate loads here.'
            return $null
        }
        # Launch failure is not an error here.
        try { & $control 2>&1 | Out-Null } catch { }
        # Polled, not slept once.
        foreach ($attempt in 1..10) {
            Start-Sleep -Seconds 3
            $fired = @(Get-WinEvent -FilterHashtable @{
                LogName = $CI_LOG; StartTime = $since
            } -ErrorAction SilentlyContinue |
                Where-Object {
                    $AcceptIds -contains $_.Id -and $_.Message -like '*unsigned-control*' -and
                    (-not $FromPolicy -or (Test-EventFromPolicy $_ $FromPolicy))
                })
            if ($fired.Count -gt 0) {
                $what = if ($fired[0].Id -eq 3077) { 'refused on this machine' } else { 'evaluated on this machine' }
                Write-Host "positive control raised $($fired[0].Id): unsigned code is $what"
                return $true
            }
        }
        return $false
    } finally {
        # Never left behind, and never staged into the evidence
        Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# A label whose baseline is not spent still has its changes on this machine.
function Get-UnrevertedLabel([string] $dir) {
    foreach ($other in @(Get-ChildItem -LiteralPath (Split-Path -Parent $dir) -Directory -ErrorAction SilentlyContinue)) {
        if ($other.Name -eq (Split-Path -Leaf $dir)) { continue }
        $path = Join-Path $other.FullName 'baseline.json'
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $b = $null
        try { $b = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json } catch { }
        if ($b -and -not $b.RevertCompletedAt) { return $other.Name }
    }
    return $null
}

function Invoke-Prepare {
    Assert-Elevated
    $dir = Get-RunDir
    $unreverted = Get-UnrevertedLabel $dir
    if ($unreverted) {
        throw "label '$unreverted' has not been reverted, so this machine still carries its changes (and its audit policy would be taken for a pre-existing one). Run .\sac-probe.ps1 -Stage revert -Label $unreverted first, then prepare label '$Label'."
    }
    $ROLLBACK_POLICY = Get-RollbackPolicyPath $dir
    Write-Section 'Baseline'
    $baselinePath = Join-Path $dir 'baseline.json'
    $previous = $null
    if (Test-Path -LiteralPath $baselinePath) {
        $previous = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
    }
    if ($previous -and -not $previous.RevertCompletedAt) {
        # A retry of this label whose changes are still on the machine.
        $baseline = $previous
        Write-Warning "reusing the baseline captured at $($baseline.CapturedAt) by an earlier prepare of this label; revert restores that state"
    } else {
        # No baseline, or one whose revert completed
        if ($previous) {
            Write-Host "the previous run of label '$Label' was reverted at $($previous.RevertCompletedAt); capturing a fresh baseline"
        }
        $baseline = Save-Baseline $dir
    }
    Write-Host ("Smart App Control: {0} (registry state {1})" -f $baseline.Sac.Mode, $baseline.Sac.RegistryState)
    foreach ($p in $baseline.Sac.Policies) {
        Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
    }

    if (-not $SkipUpdates) {
        Write-Section 'Updates'
        # Best effort.
        try {
            Write-Host 'Update-MpSignature ...'
            Update-MpSignature -ErrorAction Stop
        } catch { Write-Warning "Update-MpSignature failed: $_" }
    }
    # Always cleared, so the transcript in the run directory describes this pass
    Remove-Item -LiteralPath (Join-Path $dir 'winget-upgrade.log') -Force -ErrorAction SilentlyContinue
    if ($UpgradePackages) {
        try {
            Write-Host 'winget upgrade --all ...'
            & winget upgrade --all --accept-source-agreements --accept-package-agreements --silent 2>&1 |
                Tee-Object -FilePath (Join-Path $dir 'winget-upgrade.log') | Out-Null
        } catch { Write-Warning "winget upgrade failed: $_" }
    }

    Write-Section 'Raise security settings'
    # Deliberately only the reversible ones.
    $wanted = [ordered]@{
        DisableRealtimeMonitoring = $false
        MAPSReporting             = 'Advanced'
        CloudBlockLevel           = 'High'
        PUAProtection             = 'Enabled'
    }
    if ($SendSamples) {
        $wanted['SubmitSamplesConsent'] = 'SendAllSamples'
    }
    $mpFailed = @()
    foreach ($name in $wanted.Keys) {
        try {
            $params = @{ $name = $wanted[$name] }
            Set-MpPreference @params
        } catch {
            $mpFailed += "${name}: $_"
            Write-Warning "could not set Defender $name : $_"
        }
    }
    # Read back, because a tamper-protected or policy-managed preference is IGNORED rather than refused
    $applied = $null
    try { $applied = Get-MpPreference } catch { }
    if (-not $applied) {
        $mpFailed += 'the Defender preferences could not be read back, so this cell cannot show that it carries the documented baseline'
    } else {
        foreach ($name in $wanted.Keys) {
            $actual = $applied.$name
            $expected = $wanted[$name]
            $same = Test-MpPreferenceMatch $actual $expected (Get-MpPreferenceType $name)
            if ($false -eq $same) {
                $mpFailed += "${name}: set to $expected but reads back as $actual (tamper protection or policy?)"
                Write-Warning "Defender $name reads back as $actual after being set to $expected"
            } elseif ($null -eq $same) {
                $mpFailed += "${name}: set to $expected and reads back as $actual, which could not be compared, so this cell cannot show it carries that setting"
                Write-Warning "Defender $name reads back as $actual, which could not be compared to $expected"
            }
        }
    }
    $mpErrorPath = Join-Path $dir 'defender-preference-errors.txt'
    if ($mpFailed.Count -gt 0) {
        $mpFailed -join "`n" | Set-Content -LiteralPath $mpErrorPath -Encoding UTF8
        Write-Warning "$($mpFailed.Count) Defender preference(s) were NOT applied; this cell does not carry the documented baseline. See defender-preference-errors.txt."
    } else {
        Remove-Item -LiteralPath $mpErrorPath -Force -ErrorAction SilentlyContinue
        Write-Host 'Defender: real-time on, MAPS advanced, cloud block level high, PUA on'
        if ($SendSamples) { Write-Host 'Defender: sample submission set to SendAllSamples (-SendSamples)' }
    }

    Write-Section 'CodeIntegrity log'
    # The default 1 MB fills quickly once a policy is auditing every load, and a wrapped log silently loses the events this whole exercise exists to catch.
    try {
        Invoke-Native 'wevtutil.exe' @('sl', $CI_LOG, '/e:true', '/ms:67108864')
    } catch {
        throw "could not enable $CI_LOG at 64 MB, so no code integrity event could be collected: $_"
    }
    $ciNow = Get-CiLogSettings
    if (-not $ciNow.Enabled) {
        throw "$CI_LOG is still disabled after wevtutil sl (policy-controlled?); the probe cannot collect evidence here"
    }
    if ($null -eq $ciNow.MaxSize -or $ciNow.MaxSize -lt 67108864) {
        throw "$CI_LOG is $($ciNow.MaxSize) bytes after wevtutil sl asked for 64 MB (policy-controlled?); auditing every load would wrap the channel before collect reads it, so the probe cannot collect trustworthy evidence here"
    }
    Write-Host ("CodeIntegrity/Operational enabled, max size {0} (was enabled={1}, maxSize={2})" -f $ciNow.MaxSize, $baseline.CiLogEnabled, $baseline.CiLogMaxSize)

    if ($AuditPolicy) {
        Write-Section 'Audit policy'
        if (-not (Test-Path -LiteralPath $AuditPolicy)) {
            throw "audit policy not found: $AuditPolicy"
        }
        # The NoISG policy checks signatures only and skips the cloud reputation lookup, which is why it works even with Smart App Control off.
        $policyCopied = $false
      try {
        $mounted = Mount-Efi
        try {
            # A file already there is somebody's policy only if this label did not put it there
            if ((Test-Path -LiteralPath $NOISG_DEST) -and -not $baseline.AuditPolicyApplied) {
                # Keep it so revert restores it instead of deleting an administrator's policy.
                New-Item -ItemType Directory -Force -Path (Split-Path $ROLLBACK_POLICY) | Out-Null
                Copy-Item -LiteralPath $NOISG_DEST -Destination $ROLLBACK_POLICY -Force
                $baseline.AuditPolicyPreexisting = $true
                Write-Warning "a policy with $NOISG_GUID was already installed; saved to rollback\preexisting-policy.cip and will be restored by revert"
            }
            # Persisted before anything on the EFI partition changes
            $baseline.AuditPolicyApplied = $true
            Save-ProbeBaseline $baseline $baselinePath
            New-Item -ItemType Directory -Force -Path (Split-Path $NOISG_DEST) | Out-Null
            $policyCopied = $true
            Copy-Item -LiteralPath $AuditPolicy -Destination $NOISG_DEST -Force
            Invoke-Native 'CiTool.exe' @('-r')
        } finally {
            Dismount-Efi $mounted
        }
        if ($script:EfiStillMounted) {
            throw "the EFI system partition is still mounted as S: and could not be unmounted; the next stage would see it as already mounted and never retry, so run 'mountvol S: /D' by hand before continuing"
        }
        Write-Host "applied $(Split-Path $AuditPolicy -Leaf) as $NOISG_GUID and refreshed policy"

        Write-Section 'Policy state after applying'
        $after = Get-SacState
        foreach ($p in $after.Policies) {
            Write-Host ("  policy {0} enforced={1}" -f $p.FriendlyName, $p.IsEnforced)
        }
        # Printing whatever is listed is not verification.
        if ($after.Policies.Count -gt 0 -and -not (Test-PolicyActive $NOISG_GUID)) {
            throw "the audit policy $NOISG_GUID is not in the active policy set after refresh"
        }
        if ($after.Policies.Count -eq 0) {
            # No policy list means no evidence the policy is active, and a run with no 3076 events would then read as an allow verdict.
            throw "CiTool listed no policies, so the audit policy cannot be verified as active"
        }

        Write-Section 'Positive control'
        $controlFired = Test-AuditPolicyEvaluating -FromPolicy $NOISG_GUID
        if ($false -eq $controlFired) {
            throw "the audit policy $NOISG_GUID is listed as active but an unsigned control binary raised no 3076 or 3077 attributed to that policy, so it is not evaluating loads on this machine; a run with no events would be meaningless. Run revert and re-apply the policy."
        }
        $baseline.AuditPolicyControlFired = $controlFired
        Save-ProbeBaseline $baseline $baselinePath
      } catch {
        $failure = $_
        # A failed prepare must not leave the policy it applied behind.
        if ($policyCopied) {
            Write-Warning 'prepare failed after applying the audit policy; rolling the policy back'
            try {
                $mounted = Mount-Efi
                try {
                    if ($baseline.AuditPolicyPreexisting) {
                        Copy-Item -LiteralPath $ROLLBACK_POLICY -Destination $NOISG_DEST -Force
                    } elseif (Test-Path -LiteralPath $NOISG_DEST) {
                        Remove-Item -LiteralPath $NOISG_DEST -Force
                    }
                    Invoke-Native 'CiTool.exe' @('-r')
                } finally {
                    Dismount-Efi $mounted
                }
                if (-not $baseline.AuditPolicyPreexisting -and (Test-PolicyActive $NOISG_GUID)) {
                    Write-Warning "the audit policy $NOISG_GUID was removed but is still active until Windows restarts; restart, then run .\sac-probe.ps1 -Stage revert -Label $Label"
                } else {
                    $baseline.AuditPolicyApplied = $false
                    Save-ProbeBaseline $baseline $baselinePath
                    Write-Host 'audit policy rolled back'
                }
            } catch {
                Write-Warning "could not roll back the audit policy: $_ (run .\sac-probe.ps1 -Stage revert -Label $Label)"
            }
        }
        throw $failure
      }
    } else {
        Write-Host ''
        Write-Host 'No -AuditPolicy given. Download the sample policies from https://aka.ms/sacauditpolicies'
        Write-Host 'and re-run with -AuditPolicy <path to SmartAppControlAuditNoISG.bin> to log what'
        Write-Host 'would be blocked. Without it, only real enforcement (3077) shows up, and only on a'
        Write-Host 'machine where Smart App Control is genuinely on.'
        # So establish that it is, rather than letting collect infer an allow from a registry read.
        $sacBefore = Get-SacState
        if ($sacBefore.Mode -eq 'enforcement') {
            Write-Section 'Positive control'
            # 3077 only.
            $sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)
            if ($false -eq $sacFired) {
                throw "Smart App Control reads as 'enforcement' but an unsigned control binary ran here without being refused with a 3077, so nothing is enforcing against unsigned code on this boot and a window with no events would be meaningless. Re-run prepare with -AuditPolicy, or on a machine where enforcement is live."
            }
            $baseline | Add-Member -NotePropertyName SacControlFired -NotePropertyValue $sacFired -Force
            Save-ProbeBaseline $baseline $baselinePath
        }
    }

    # Marks the window collect will export.
    foreach ($stale in @(
        'scenario-status.json', 'scenario-results.json', 'runtime-selection.json',
        'venv-selection.txt', 'studio-home.txt',
        'signature-inventory.json', 'signature-inventory.csv',
        'venv-signature-inventory.json', 'venv-signature-inventory.csv',
        'code-integrity-events.json', 'code-integrity-events.txt',
        'CodeIntegrity-Operational.evtx', 'defender-detections.json',
        'defender-query-error.txt', 'sac-state-after.json',
        'events-collection-error.txt', 'inventory-enumeration-errors.txt'
    )) {
        Remove-Item -LiteralPath (Join-Path $dir $stale) -Force -ErrorAction SilentlyContinue
    }
    foreach ($staleDir in @('studio-logs', 'raw-logs')) {
        Remove-Item -LiteralPath (Join-Path $dir $staleDir) -Recurse -Force -ErrorAction SilentlyContinue
    }

    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $dir 'window-start.txt') -Encoding UTF8

    Initialize-Studio $dir $true

    Write-Host ''
    Write-Host "prepare complete. Next: .\sac-probe.ps1 -Stage run -Label $Label"
}

$script:InventoryErrors = @()

function Get-SignatureInventory([string] $root) {
    if (-not (Test-Path -LiteralPath $root)) {
        Write-Warning "nothing to inventory at $root"
        return @()
    }
    $enumErrors = @()
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue -ErrorVariable enumErrors |
        Where-Object { $PE_EXT -contains $_.Extension.ToLowerInvariant() } |
        ForEach-Object {
            $sig = Get-AuthenticodeSignature -LiteralPath $_.FullName
            [pscustomobject]@{
                Name       = $_.Name
                FullName   = $_.FullName
                Length     = $_.Length
                SHA256     = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                Status     = [string]$sig.Status
                StatusMessage = $sig.StatusMessage
                Subject    = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
                Thumbprint = if ($sig.SignerCertificate) { $sig.SignerCertificate.Thumbprint } else { $null }
                TimeStamped = if ($sig.TimeStamperCertificate) { $true } else { $false }
            }
        }
    if ($enumErrors.Count -gt 0) {
        # A subtree that could not be read is exactly where the access-denied native module being investigated would sit, and the caller only rejects a completely empty result.
        $script:InventoryErrors += @($enumErrors | ForEach-Object { "${root}: $_" })
        Write-Warning "$($enumErrors.Count) path(s) under $root could not be enumerated; this inventory is PARTIAL. See inventory-enumeration-errors.txt."
    }
}

function Invoke-Run {
    Assert-Elevated
    $dir = Get-RunDir

    # Before anything else.
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'baseline.json')) -or
        -not (Test-Path -LiteralPath (Join-Path $dir 'window-start.txt'))) {
        throw "label '$Label' has no baseline.json and window-start.txt under ${dir}: prepare did not complete for it, so there is nothing to run against. Check the -Label, or run prepare for it first."
    }

    $runBaselinePath = Join-Path $dir 'baseline.json'
    if (Test-Path -LiteralPath $runBaselinePath) {
        $runBaseline = Get-Content -LiteralPath $runBaselinePath -Raw | ConvertFrom-Json
        # A label whose revert already completed.
        if ($runBaseline.RevertCompletedAt) {
            throw "label '$Label' was reverted at $($runBaseline.RevertCompletedAt), so its baseline is spent and the event window on disk belongs to the run that was undone. Running now would file events from that window against this run's inventory. Run prepare for this label again (or use a new -Label) first."
        }
        # The window on disk has to belong to THIS baseline, and the check above cannot see when it does not.
        $runStartPath = Join-Path $dir 'window-start.txt'
        if (Test-Path -LiteralPath $runStartPath) {
            $runStart = $null
            $runPrepared = $null
            try {
                $runStart = [datetime]::Parse(
                    (Get-Content -LiteralPath $runStartPath -Raw).Trim(),
                    [cultureinfo]::InvariantCulture)
                $runPrepared = [datetime]::Parse([string]$runBaseline.CapturedAt,
                    [cultureinfo]::InvariantCulture)
            } catch { }
            if ($runStart -and $runPrepared -and $runStart -lt $runPrepared) {
                throw "the event window on disk opened at $($runStart.ToString('o')), before this label's baseline was captured at $($runPrepared.ToString('o')): prepare took a new baseline and then failed before it reopened the window, so collect would export the previous run's events against this run's inventory. Run prepare for label '$Label' again first."
            }
        }
        if ($runBaseline.AuditPolicyApplied) {
            Write-Section 'Audit policy still active'
            $state = Get-SacState
            if ($state.Policies.Count -eq 0) {
                throw "CiTool listed no policies, so the audit policy applied by prepare cannot be verified as active; this run would measure nothing. Run revert and prepare again."
            }
            if (-not (Test-PolicyActive $NOISG_GUID)) {
                throw "the audit policy $NOISG_GUID is no longer in the active policy set (a reboot since prepare?), so this run would not be audited and its empty window would read as an allow. Run revert and prepare again."
            }
            Write-Host "audit policy $NOISG_GUID is active"
            $bootedAt = $null
            try { $bootedAt = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime } catch { }
            $preparedAt = $null
            # Invariant, and on the string form, because CapturedAt arrives here in two shapes and the default parse reads neither safely.
            try {
                $preparedAt = [datetime]::Parse([string]$runBaseline.CapturedAt,
                    [cultureinfo]::InvariantCulture)
            } catch { }
            if ($bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt) {
                Write-Host 'the machine booted after this baseline was captured; re-running the positive control on this boot'
                $controlFired = Test-AuditPolicyEvaluating -FromPolicy $NOISG_GUID
                if ($false -eq $controlFired) {
                    throw "the audit policy $NOISG_GUID is listed as active but an unsigned control raised no 3076 or 3077 attributed to that policy on this boot, so it is not evaluating loads and this run would be meaningless. Run revert and prepare again."
                }
                $runBaseline.AuditPolicyControlFired = $controlFired
                Save-ProbeBaseline $runBaseline $runBaselinePath
            }
        } elseif ([string]$runBaseline.Sac.Mode -eq 'enforcement') {
            # The same revalidation for the cell with no audit policy, where the only thing that can produce a verdict is Smart App Control refusing code for real.
            Write-Section 'Smart App Control still enforcing'
            $sacNow = Get-SacState
            if ($sacNow.Mode -ne 'enforcement') {
                throw "Smart App Control read as 'enforcement' when this label was prepared but reads as '$($sacNow.Mode)' now (a reboot since prepare?), so nothing here can log a verdict and this run's empty window would read as an allow. Run revert and prepare again."
            }
            Write-Host 'Smart App Control is enforcing'
            $bootedAt = $null
            try { $bootedAt = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime } catch { }
            $preparedAt = $null
            # Invariant, and on the string form, because CapturedAt arrives here in two shapes and the default parse reads neither safely.
            try {
                $preparedAt = [datetime]::Parse([string]$runBaseline.CapturedAt,
                    [cultureinfo]::InvariantCulture)
            } catch { }
            if (($bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt) -or
                ($true -ne $runBaseline.SacControlFired)) {
                Write-Host 'confirming on this boot that unsigned code is actually refused here'
                # 3077 only, for the reason prepare gives
                $sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)
                if ($false -eq $sacFired) {
                    throw "Smart App Control reads as 'enforcement' but an unsigned control ran on this boot without being refused with a 3077, so it is not enforcing against unsigned code and this run would be meaningless. Run revert and prepare again."
                }
                $runBaseline | Add-Member -NotePropertyName SacControlFired -NotePropertyValue $sacFired -Force
                Save-ProbeBaseline $runBaseline $runBaselinePath
            }
        }
    }

    # A machine that was prepared earlier may have been rebooted since, which is itself part of the reported behaviour
    if (-not $SkipStudio -and -not (Test-StudioResponding $Port)) { Initialize-Studio $dir $false }

    # The venv, separately.
    Write-Section 'Venv signature inventory'
    $studioPython = Get-StudioPython
    $venvDir = Resolve-VenvDir
    Write-Host "venv: $venvDir"
    # Recorded for collect, which may run from a shell where a custom UNSLOTH_STUDIO_HOME was never set.
    $venvDir | Set-Content -LiteralPath (Join-Path $dir 'venv-selection.txt') -Encoding UTF8
    Get-StudioHome | Set-Content -LiteralPath (Join-Path $dir 'studio-home.txt') -Encoding UTF8
    $venvInventory = @(Get-SignatureInventory $venvDir)
    ConvertTo-Json -InputObject @($venvInventory) -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'venv-signature-inventory.json') -Encoding UTF8
    $venvInventory | Export-Csv -LiteralPath (Join-Path $dir 'venv-signature-inventory.csv') -NoTypeInformation -Encoding UTF8
    $venvTotal = $venvInventory.Count
    if ($venvTotal -eq 0) {
        # The venv is where the only enforced block so far landed
        throw "no PE files found under $venvDir; Studio's environment was not found, so the cell is invalid. Check UNSLOTH_STUDIO_HOME or install Studio first"
    }
    $venvValid = @($venvInventory | Where-Object { $_.Status -eq 'Valid' }).Count
    Write-Host "$venvValid of $venvTotal PE files in the venv report a valid Authenticode signature"
    if ($venvTotal -gt 0) {
        $venvInventory | Where-Object { $_.Status -ne 'Valid' } |
            Group-Object { ($_.FullName -split '\\site-packages\\')[-1].Split('\')[0] } |
            Sort-Object Count -Descending | Select-Object -First 10 |
            Format-Table @{n = 'package'; e = { $_.Name } }, Count -AutoSize | Out-String | Write-Host
    }

    $scenarioStatus = [pscustomobject]@{ Ran = $false; ExitCode = $null; Reason = 'skipped' }
    if ($SkipStudio) {
        Write-Host 'skipping the Studio scenario (-SkipStudio)'
    } else {
        Write-Section 'Studio scenario'
        $scenario = Join-Path $PSScriptRoot 'studio_scenario.py'
        New-Item -ItemType Directory -Force -Path (Join-Path $dir 'raw-logs') | Out-Null
        $log = Join-Path (Join-Path $dir 'raw-logs') 'studio-scenario.log'
        if (-not $STUDIO_PASSWORD) {
            Write-Warning 'UNSLOTH_STUDIO_PASSWORD is not set; the scenario needs it (see README) and will stop at login'
        }
        $scenarioArgs = @($scenario, '--model', $Model, '--out', $dir, '--port', "$Port")
        # Neither UNSLOTH_STUDIO_PASSWORD (unsloth_cli claims that name and hard-errors, see $STUDIO_PASSWORD) nor --password on the command line.
        if ($STUDIO_PASSWORD) { $env:SAC_PROBE_STUDIO_PASSWORD = $STUDIO_PASSWORD }
        Write-Host "python $scenario --model $Model --out $dir --port $Port"
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & python @scenarioArgs 2>&1 | Tee-Object -FilePath $log
            $code = $LASTEXITCODE
            $scenarioStatus = [pscustomobject]@{
                Ran      = $true
                ExitCode = $code
                Reason   = if ($code -eq 0) { 'ok' } else { 'scenario exited non-zero' }
            }
            Write-Host "scenario exit code: $code"
        } catch {
            $scenarioStatus = [pscustomobject]@{ Ran = $true; ExitCode = $null; Reason = "$_" }
            Write-Warning "scenario failed: $_"
        } finally {
            Remove-Item Env:\SAC_PROBE_STUDIO_PASSWORD -ErrorAction SilentlyContinue
            $ErrorActionPreference = $prev
        }
    }
    $scenarioStatus | ConvertTo-Json -Depth 3 |
        Set-Content -LiteralPath (Join-Path $dir 'scenario-status.json') -Encoding UTF8


    # After the scenario, which records the runtime Studio resolved
    Write-Section 'Signature inventory'
    $llamaDir = Resolve-LlamaDir $dir
    Write-Host "runtime: $llamaDir"
    $inventory = @(Get-SignatureInventory $llamaDir)
    ConvertTo-Json -InputObject @($inventory) -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'signature-inventory.json') -Encoding UTF8
    $inventory | Export-Csv -LiteralPath (Join-Path $dir 'signature-inventory.csv') -NoTypeInformation -Encoding UTF8

    $total = $inventory.Count
    if ($total -eq 0) {
        # A runtime with no PE files is a stale UNSLOTH_LLAMA_CPP_PATH, an absent install or a failed enumeration
        throw "no PE files found under ${llamaDir}: there is no llama.cpp runtime on this machine, so the cell is invalid. Run 'python -X utf8 -I -m unsloth_cli studio setup' to download one (or point UNSLOTH_LLAMA_CPP_PATH / UNSLOTH_STUDIO_HOME at an existing install), then re-run this stage"
    }
    $valid = @($inventory | Where-Object { $_.Status -eq 'Valid' }).Count
    Write-Host "$valid of $total PE files report a valid Authenticode signature"
    if ($total -gt 0 -and $valid -lt $total) {
        Write-Host 'unsigned or unverifiable:' -ForegroundColor Yellow
        $inventory | Where-Object { $_.Status -ne 'Valid' } |
            Select-Object Name, Status | Format-Table -AutoSize | Out-String | Write-Host
    }

    $enumPath = Join-Path $dir 'inventory-enumeration-errors.txt'
    if ($script:InventoryErrors.Count -gt 0) {
        $script:InventoryErrors -join "`n" | Set-Content -LiteralPath $enumPath -Encoding UTF8
        Write-Warning "$($script:InventoryErrors.Count) path(s) could not be enumerated; the inventories in this cell are PARTIAL and inventory-enumeration-errors.txt names them. Do not read an absent file as an absent PE."
    } else {
        Remove-Item -LiteralPath $enumPath -Force -ErrorAction SilentlyContinue
    }

    Write-Host ''
    if (-not $SkipStudio -and $scenarioStatus.ExitCode -ne 0) {
        # A scenario that never authenticated or never loaded a model produces a window with nothing in it, and an empty window reads exactly like a clean allow.
        Write-Warning 'the scenario did NOT complete, so this cell has not exercised model loading.'
        Write-Warning "See $log. Fix the cause and re-run this stage before collecting."
    }
    Write-Host "run complete. Next: .\sac-probe.ps1 -Stage collect -Label $Label"
}

# Reads the event window until delivery has settled.
function Read-SettledCiEvents([datetime] $Start, [int[]] $Ids, [int] $Attempts = 20, [int] $IntervalSeconds = 3, [int] $MinSettleSeconds = 30) {
    $previous = -1
    $waited = 0
    $events = @()
    foreach ($attempt in 1..$Attempts) {
        $events = @()
        try {
            $events = @(Get-WinEvent -FilterHashtable @{
                LogName   = $CI_LOG
                StartTime = $Start
            } -ErrorAction Stop | Where-Object { $Ids -contains $_.Id })
        } catch {
            if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') { throw }
        }
        if ($events.Count -eq $previous -and $waited -ge $MinSettleSeconds) { break }
        $previous = $events.Count
        if ($attempt -lt $Attempts) {
            Start-Sleep -Seconds $IntervalSeconds
            $waited += $IntervalSeconds
        }
    }
    return $events
}

function Invoke-Collect {
    Assert-Elevated
    $dir = Get-RunDir

    $startPath = Join-Path $dir 'window-start.txt'
    if (-not (Test-Path -LiteralPath $startPath)) {
        # Get-RunDir creates the directory, so a mistyped label or a collect without a prepare lands here.
        throw "no window-start.txt under ${dir}: prepare did not run for label '$Label', so there is no event window to collect"
    }
    $start = [datetime]::Parse((Get-Content -LiteralPath $startPath -Raw).Trim())
    Write-Section "Events since $($start.ToString('o'))"

    # Everything that went wrong while gathering evidence.
    $collectionProblems = @()
    $events = @()
    try {
        $events = @(Read-SettledCiEvents $start $CI_EVENT_IDS)
        if ($events.Count -eq 0) { Write-Host 'no CodeIntegrity events in the window' }
    } catch {
        # Only "nothing matched" is an empty window, and Read-SettledCiEvents already treats it as one.
        $_.ToString() | Set-Content -LiteralPath (Join-Path $dir 'events-collection-error.txt') -Encoding UTF8
        throw "could not read $CI_LOG, so the event window was not collected: $_"
    }
    # A retry that got through clears the earlier attempt's marker
    Remove-Item -LiteralPath (Join-Path $dir 'events-collection-error.txt') -Force -ErrorAction SilentlyContinue

    $tail = Get-ScopeTail (Resolve-LlamaDir $dir)
    $venvTail = Get-ScopeTail (Resolve-VenvDir $dir)
    $shaped = @($events | ForEach-Object {
        $msg = $_.Message
        $data = Get-EventDataMap $_
        # The evaluated file, not the whole message.
        $subject = $msg
        foreach ($field in @('File Name', 'FileNameBuffer')) {
            if ($data.Contains($field) -and $data[$field]) { $subject = [string]$data[$field]; break }
        }
        $scope =
            if ($subject -like "*$tail*") { 'llama.cpp' }
            elseif ($subject -like "*$venvTail*") { 'venv' }
            else { 'other' }
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
            Scope       = $scope
            ScopeFrom   = 'path'
            ActivityID  = $_.ActivityId
            Message     = $msg
            # 3089's rendered Message is the fixed string "Signature information for another event.
            EventData   = $data
        }
    })
    $scopeByActivity = @{}
    foreach ($e in $shaped) {
        if ($e.Scope -ne 'other' -and $e.ActivityID -and -not $scopeByActivity.ContainsKey($e.ActivityID)) {
            $scopeByActivity[$e.ActivityID] = $e.Scope
        }
    }
    $inherited = 0
    foreach ($e in $shaped) {
        if ($e.Scope -eq 'other' -and $e.ActivityID -and $scopeByActivity.ContainsKey($e.ActivityID)) {
            $e.Scope = $scopeByActivity[$e.ActivityID]
            $e.ScopeFrom = 'correlation'
            $inherited++
        }
    }
    if ($inherited -gt 0) {
        Write-Host "$inherited event(s) scoped by ActivityID correlation rather than by path"
    }

    ConvertTo-Json -InputObject $shaped -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.json') -Encoding UTF8
    $shaped | Format-List | Out-String |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.txt') -Encoding UTF8

    $ours = @($shaped | Where-Object { $_.Scope -ne 'other' })
    $blocks = @($ours | Where-Object { $_.Id -eq 3077 }).Count
    # A 3076 is this probe's audit verdict only when the NoISG policy it installed raised it.
    $auditApplied = $false
    $baselineForAttribution = Join-Path $dir 'baseline.json'
    if (Test-Path -LiteralPath $baselineForAttribution) {
        try {
            $auditApplied = [bool] (Get-Content -LiteralPath $baselineForAttribution -Raw | ConvertFrom-Json).AuditPolicyApplied
        } catch { $auditApplied = $false }
    }
    $isOurAudit = {
        param($e)
        $e.Id -eq 3076 -and $auditApplied -and (Test-EventDataFromPolicy $e.EventData $NOISG_GUID)
    }
    $audits = @($ours | Where-Object { & $isOurAudit $_ }).Count
    $otherPolicyAudits = @($ours | Where-Object { $_.Id -eq 3076 }).Count - $audits
    if ($otherPolicyAudits -gt 0) {
        $ourPolicy = if ($auditApplied) { 'the NoISG audit policy this probe installed' } else { 'this probe (no audit policy was applied)' }
        $collectionProblems += "$otherPolicyAudits audit event(s) (3076) on Unsloth paths did not come from $ourPolicy; they are not counted as signature verdicts, and the machine carries another audit policy that can confound this cell"
        Write-Warning "$otherPolicyAudits 3076 event(s) on Unsloth paths came from another policy and are not counted (see code-integrity-events.json)"
    }
    $foreign = @($shaped | Where-Object { $_.Scope -eq 'other' }).Count
    Write-Host "Unsloth paths: $blocks enforced block(s) (3077), $audits audit would-block(s) (3076), $($ours.Count) event(s)"
    foreach ($g in ($ours | Group-Object Scope | Sort-Object Name)) {
        $b = @($g.Group | Where-Object { $_.Id -eq 3077 }).Count
        $a = @($g.Group | Where-Object { & $isOurAudit $_ }).Count
        Write-Host ("  {0,-10} {1} x 3077, {2} x 3076, {3} event(s)" -f $g.Name, $b, $a, $g.Count)
    }
    # Whether a model was actually loaded inside this window.
    $scenarioStatus = $null
    $statusPath = Join-Path $dir 'scenario-status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $scenarioStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    }
    $loadOk = $false
    $failedSteps = @()
    $resultsPath = Join-Path $dir 'scenario-results.json'
    if (Test-Path -LiteralPath $resultsPath) {
        $results = Get-Content -LiteralPath $resultsPath -Raw | ConvertFrom-Json
        $loadOk = [bool]$results.steps.load.ok
        $failedSteps = @($results.steps.PSObject.Properties | Where-Object { -not $_.Value.ok } | ForEach-Object { $_.Name })
    }
    $scenarioProblem = $null
    if (-not $scenarioStatus) {
        $scenarioProblem = 'no scenario-status.json: the run stage did not complete for this label, so nothing is known to have loaded inside this window and an empty event list is a NULL result, not an allow'
    } elseif ($scenarioStatus.Reason -eq 'skipped') {
        $scenarioProblem = 'the Studio scenario was skipped (-SkipStudio), so this zip is a signature inventory only: it does not show whether anything was blocked at load time'
    } elseif ($scenarioStatus.ExitCode -ne 0 -and -not $loadOk) {
        $scenarioProblem = "the Studio scenario did NOT load a model (exit $($scenarioStatus.ExitCode): $($scenarioStatus.Reason); failed steps: $($failedSteps -join ', ')), so an empty event list here is a NULL result, not an allow"
    }
    if ($scenarioProblem) { $collectionProblems += $scenarioProblem }

    # An empty window is an allow only if the policy was shown to be evaluating loads.
    $verdicts = $blocks + $audits
    $baselineForControl = Join-Path $dir 'baseline.json'
    $sacNow = Get-SacState
    if ($verdicts -eq 0 -and (Test-Path -LiteralPath $baselineForControl)) {
        $b = Get-Content -LiteralPath $baselineForControl -Raw | ConvertFrom-Json
        # Both ends of the window, and neither is redundant.
        $sacMode = [string]$sacNow.Mode
        $sacAtPrepare = [string]$b.Sac.Mode
        if ($b.AuditPolicyApplied -and $true -ne $b.AuditPolicyControlFired) {
            $collectionProblems += 'no positive control confirmed the audit policy was evaluating loads, so a window with no 3076 or 3077 here is a NULL result, not an allow'
            Write-Warning 'No Unsloth path raised a 3076 or 3077, but no positive control confirmed the audit policy was evaluating loads on this machine. Do NOT report this cell as "not blocked".'
        } elseif (-not $loadOk) {
            # Recorded as a collection problem above
            Write-Warning 'No Unsloth path raised a 3076 or 3077, but nothing was observed loading in this window (see collection-warnings.txt in the zip). Do NOT report this cell as "not blocked".'
        } elseif ($b.AuditPolicyApplied) {
            Write-Host 'no Unsloth path raised a 3076 or 3077, and the positive control confirmed the policy was evaluating loads'
        } elseif ($sacMode -ne 'enforcement' -or $sacAtPrepare -ne 'enforcement') {
            $collectionProblems += "no audit policy was applied and Smart App Control is '$sacMode' now ('$sacAtPrepare' when this label was prepared), so nothing on this machine could log a 3076 and nothing could enforce a 3077 for the whole window: a window with no verdict is a NULL result, not an allow. Re-run prepare with -AuditPolicy."
            Write-Warning "No Unsloth path raised a 3076 or 3077, but this cell ran with no audit policy and Smart App Control '$sacMode' ('$sacAtPrepare' at prepare), so no verdict could have been logged either way. Do NOT report this cell as `"not blocked`"; re-run prepare with -AuditPolicy."
        } elseif ($true -ne $b.SacControlFired) {
            # Enforcement in the registry is an intent, not an observation.
            $collectionProblems += 'no audit policy was applied and no unsigned positive control confirmed that Smart App Control actually refused code on the boot that measured, so a window with no 3077 here is a NULL result, not an allow. Re-run prepare and run from Windows PowerShell 5.1, which can build the control.'
            Write-Warning 'No Unsloth path raised a 3077 and Smart App Control reads as enforcing, but no unsigned positive control confirmed it was refusing code on this boot. Do NOT report this cell as "not blocked".'
        } else {
            Write-Host 'no Unsloth path raised a 3077, Smart App Control is enforcing, and an unsigned positive control confirmed it refuses code on this boot, so this window is a real allow'
        }
    }

    # Reported, never folded into the totals above.
    Write-Host "unrelated to Unsloth: $foreign event(s) (kept in the export, excluded from the counts)"
    Write-Host "$($shaped.Count) event(s) in the window overall"

    $windowStart = $start.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
    $query = "*[System[TimeCreated[@SystemTime>='$windowStart']]]"
    try {
        Invoke-Native 'wevtutil.exe' @('epl', $CI_LOG, (Join-Path $dir 'CodeIntegrity-Operational.evtx'), "/q:$query", '/ow:true')
    } catch {
        $collectionProblems += "raw evtx export failed, so CodeIntegrity-Operational.evtx is missing from this zip: $_"
        Write-Warning "evtx export failed: $_"
    }

    try {
        # Captured into an array first
        $detections = @(Get-MpThreatDetection -ErrorAction Stop |
            Where-Object { $_.InitialDetectionTime -ge $start } |
            Select-Object InitialDetectionTime, ThreatID, Resources)
        ConvertTo-Json -InputObject $detections -Depth 4 |
            Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
        Remove-Item -LiteralPath (Join-Path $dir 'defender-query-error.txt') -Force -ErrorAction SilentlyContinue
    } catch {
        # A failed query wrote the same `[]` a clean machine writes, so the evidence claimed a quarantine-free window when Defender had simply not answered.
        $_.ToString() | Set-Content -LiteralPath (Join-Path $dir 'defender-query-error.txt') -Encoding UTF8
        Remove-Item -LiteralPath (Join-Path $dir 'defender-detections.json') -Force -ErrorAction SilentlyContinue
        Write-Warning "Defender detections were NOT collected: $_. defender-query-error.txt records why; do not read the absence of detections as a clean window."
    }

    $sacNow | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $dir 'sac-state-after.json') -Encoding UTF8

    # Studio's own logs, which carry the request timings and the backend errors.
    $sources = @()
    $studioLogs = Join-Path (Resolve-StudioHomeFor $dir) 'logs'
    if (Test-Path -LiteralPath $studioLogs) { $sources += $studioLogs }
    $rawLogs = Join-Path $dir 'raw-logs'
    if (Test-Path -LiteralPath $rawLogs) { $sources += $rawLogs }
    if ($sources.Count -gt 0) {
        $python = Resolve-StudioPythonFor $dir
        if ($python) {
            $redactor = Join-Path $PSScriptRoot 'redact_logs.py'
            foreach ($source in $sources) {
                try {
                    Invoke-Native $python @('-X', 'utf8', '-I', $redactor, $source, (Join-Path $dir 'studio-logs'), '--since', $start.ToString('o'))
                } catch {
                    $collectionProblems += "log redaction failed for ${source}, so studio-logs\ is missing from this zip: $_"
                    Write-Warning "logs under $source were not copied (redaction failed): $_"
                    Remove-Item -LiteralPath (Join-Path $dir 'studio-logs') -Recurse -Force -ErrorAction SilentlyContinue
                    break
                }
            }
        } else {
            # Same absence, same marker
            $collectionProblems += 'Studio logs were not copied: no managed interpreter to run the redactor, so studio-logs\ is missing from this zip'
            Write-Warning 'Studio logs were not copied: no managed interpreter to run the redactor'
        }
    }

    # Stage a copy before compressing.
    $stage = Join-Path $WorkDir ".stage-$Label"
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $dir -Recurse -File) {
        $rel = $item.FullName.Substring($dir.Length).TrimStart('\')
        if ($rel -like 'rollback\*' -or $rel -like 'raw-logs\*') { continue }
        $target = Join-Path $stage $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        try {
            # FileShare::ReadWrite so a writer holding the file does not block the read, which Copy-Item cannot express.
            $src = [System.IO.File]::Open($item.FullName, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            try {
                $dst = [System.IO.File]::Create($target)
                try { $src.CopyTo($dst) } finally { $dst.Dispose() }
            } finally { $src.Dispose() }
        } catch {
            $collectionProblems += "${rel} could not be staged and is missing from this zip: $_"
            Write-Warning "could not stage $($item.FullName): $_"
        }
    }
    if ($collectionProblems.Count -gt 0) {
        # Into the staged copy, after the loop, so it is inside the archive the operator attaches rather than only in a console they will not send.
        $collectionProblems -join "`n" |
            Set-Content -LiteralPath (Join-Path $stage 'collection-warnings.txt') -Encoding UTF8
        Write-Warning "this collection is INCOMPLETE: $($collectionProblems.Count) problem(s), recorded in collection-warnings.txt inside the zip"
    }

    $zip = Join-Path $WorkDir ("unsloth-sac-{0}-{1}-{2}.zip" -f $env:COMPUTERNAME, $Label, (Get-Date -Format 'yyyyMMdd-HHmmss'))
    try {
        Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force -ErrorAction Stop
    } catch {
        # Never announce a path Compress-Archive did not produce.
        Write-Host "::error::could not write $zip : $_"
        throw
    } finally {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $zip)) {
        throw "Compress-Archive reported success but $zip does not exist"
    }

    Write-Host ''
    Write-Host "evidence: $zip" -ForegroundColor Green

    # An empty window is only a result if the scenario actually ran.
    if ($loadOk) {
        if ($scenarioStatus -and $scenarioStatus.ExitCode -ne 0) {
            Write-Warning "The model loaded, so the load-time events are valid; later scenario step(s) failed: $($failedSteps -join ', ') (exit $($scenarioStatus.ExitCode))."
        }
    } elseif ($scenarioProblem) {
        Write-Warning $scenarioProblem
        Write-Warning 'Do not report this cell as "not blocked".'
    }

    Write-Host ''
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
    $ROLLBACK_POLICY = Get-RollbackPolicyPath $dir

    Clear-EfiOwnership

    # A baseline this script already considers spent.
    if ($baseline.RevertCompletedAt) {
        if ($script:EfiStillMounted) {
            throw "the EFI system partition is still mounted as S: and could not be unmounted; run 'mountvol S: /D' by hand"
        }
        Write-Host "label '$Label' was already reverted at $($baseline.RevertCompletedAt); this baseline is spent and nothing was changed. Run prepare again for a new run on this label."
        return
    }

    # The policy block may throw (mount, copy or CiTool).
    $policyError = $null
    if ($baseline.AuditPolicyApplied) {
      try {
        Write-Section 'Remove audit policy'
        $saved = $ROLLBACK_POLICY
        $mounted = Mount-Efi
        try {
            if ($baseline.AuditPolicyPreexisting -and -not (Test-Path -LiteralPath $saved)) {
                # Falling through to the removal branch would delete an administrator's policy and then report a completed rollback.
                throw "the baseline says a policy with $NOISG_GUID was already installed, but the saved copy is missing from $saved, so it cannot be restored. Nothing was changed; restore that .cip by hand (or remove AuditPolicyPreexisting from baseline.json once you have) and run revert again."
            }
            if ($baseline.AuditPolicyPreexisting) {
                Copy-Item -LiteralPath $saved -Destination $NOISG_DEST -Force
                Write-Host 'pre-existing audit policy restored'
            } elseif (Test-Path -LiteralPath $NOISG_DEST) {
                Remove-Item -LiteralPath $NOISG_DEST -Force
                Write-Host 'audit policy removed'
            } else {
                Write-Host 'audit policy file already absent'
            }
            # Refreshed in every branch
            Invoke-Native 'CiTool.exe' @('-r')
            Write-Host 'policy refreshed'
        } finally {
            Dismount-Efi $mounted
        }
        # Before Windows 11 24H2 a removed policy can stay active until a restart.
        if (-not $baseline.AuditPolicyPreexisting -and (Test-PolicyActive $NOISG_GUID)) {
            throw "the audit policy $NOISG_GUID was removed and refreshed but is still active; restart Windows, then run .\sac-probe.ps1 -Stage revert -Label $Label again"
        }
        # Only once the refresh succeeded and the policy is gone
        $baseline.AuditPolicyApplied = $false
        Save-ProbeBaseline $baseline (Join-Path $dir 'baseline.json')
      } catch {
        $policyError = $_
        Write-Warning "audit policy was not reverted: $_ (continuing with the other settings; run revert again)"
      }
    }

    Write-Section 'Restore CodeIntegrity log'
    $ciRestoreError = $null
    try {
        if ($null -ne $baseline.CiLogMaxSize -and $null -ne $baseline.CiLogEnabled) {
            $enabled = if ($baseline.CiLogEnabled) { 'true' } else { 'false' }
            Invoke-Native 'wevtutil.exe' @('sl', $CI_LOG, "/e:$enabled", "/ms:$($baseline.CiLogMaxSize)")
            Write-Host ("CodeIntegrity/Operational restored: enabled={0}, maxSize={1}" -f $enabled, $baseline.CiLogMaxSize)
        } else {
            Write-Host 'no CodeIntegrity log baseline recorded; left as is'
        }
    } catch {
        $ciRestoreError = $_
        Write-Warning "could not restore the CodeIntegrity log settings: $_"
    }

    Write-Section 'Restore Defender preferences'
    # Only what prepare changed, and only where a baseline value was captured.
    $restores = @(
        @{ Name = 'DisableRealtimeMonitoring'; Value = $baseline.DisableRealtimeMonitoring },
        @{ Name = 'MAPSReporting';             Value = $baseline.MAPSReporting },
        @{ Name = 'SubmitSamplesConsent';      Value = $baseline.SubmitSamplesConsent },
        @{ Name = 'CloudBlockLevel';           Value = $baseline.CloudBlockLevel },
        @{ Name = 'PUAProtection';             Value = $baseline.PUAProtection }
    )
    $failed = 0
    foreach ($r in $restores) {
        if ($null -eq $r.Value) { continue }
        try {
            $params = @{ $r.Name = $r.Value }
            Set-MpPreference @params
        } catch {
            $failed++
            Write-Warning "could not restore $($r.Name): $_"
        }
    }
    $restored = $null
    try { $restored = Get-MpPreference } catch { }
    if (-not $restored) {
        $failed++
        Write-Warning 'could not read the Defender preferences back, so this revert cannot show the machine was restored'
    } else {
        foreach ($r in $restores) {
            if ($null -eq $r.Value) { continue }
            $same = Test-MpPreferenceMatch $restored.($r.Name) $r.Value (Get-MpPreferenceType $r.Name)
            if ($true -ne $same) {
                $failed++
                Write-Warning "$($r.Name) reads back as $($restored.($r.Name)) after being restored to $($r.Value)"
            }
        }
    }
    if ($failed -eq 0) { Write-Host 'Defender preferences restored' }
    else { Write-Warning "$failed Defender preference(s) were not restored; see above" }
    # Carried to the exit status below rather than left as a console warning
    $restoreFailures = $failed

    Write-Section 'Stop the elevated Studio'
    $studioStillRunning = -not (Stop-ProbeStudio $dir)
    if ($studioStillRunning) { Write-Warning 'the Studio prepare/run started as administrator is still running; stop it by hand' }
    else { Write-Host 'no probe-started Studio left running; start Studio normally (unelevated) to use it' }

    Write-Section 'Restore Studio tree access'
    # Every stage is elevated, so a Studio that prepare installs is installed as administrator, and the trees the installer creates (llama.cpp, whisper.cpp, node, .cache) come out owned by BUILTIN\Administrators.
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $override = Get-StudioHomeOverride
    $allowedRoots = @(
        $env:USERPROFILE, $override, (Get-StudioHome),
        (Get-LlamaDir), (Split-Path -Parent (Get-LlamaDir))
    ) | Where-Object { $_ } |
        ForEach-Object { try { [IO.Path]::GetFullPath($_).TrimEnd('\') } catch { $null } } |
        Where-Object { $_ } | Select-Object -Unique
    $recorded = @()
    if ($baseline.StudioInstalledByProbe) { $recorded = @($baseline.StudioInstallRoots) }
    $trees = @()
    $rejected = @()
    foreach ($path in $recorded) {
        if (-not $path) { continue }
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $full = [IO.Path]::GetFullPath($path).TrimEnd('\')
        $inside = @($allowedRoots | Where-Object {
            $full -eq $_ -or $full.StartsWith($_ + '\', [StringComparison]::OrdinalIgnoreCase)
        }).Count -gt 0
        if ($inside) { $trees += $full }
        else {
            $rejected += $full
            Write-Warning "not repairing ACLs on ${full}: outside the configured install roots ($($allowedRoots -join ', ')). prepare recorded this tree, so it is administrator-owned and still unreadable to you: set the UNSLOTH_STUDIO_HOME / UNSLOTH_LLAMA_CPP_PATH this label was prepared with and run revert again."
        }
    }
    $trees = @($trees | Select-Object -Unique)
    $rejected = @($rejected | Select-Object -Unique)
    if ($trees.Count -eq 0 -and $rejected.Count -eq 0) {
        Write-Host 'nothing to repair: this run did not install Studio'
    }
    # Counted, not just warned about.
    $aclFailures = $rejected.Count
    foreach ($tree in $trees) {
        try {
            # Grants the invoking user only.
            & icacls.exe $tree /grant "${user}:(OI)(CI)F" /T /C /Q | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Host "restored $user access to $tree" }
            else {
                $aclFailures++
                Write-Warning "icacls exited $LASTEXITCODE for $tree"
            }
        } catch {
            $aclFailures++
            Write-Warning "could not restore access to ${tree}: $_"
        }
    }

    if ($null -ne $policyError) {
        throw "revert restored the log and Defender settings but the audit policy is still applied: $policyError"
    }
    Write-Host ''
    # Every restoration is attempted first, then the failures decide the exit status.
    if ($restoreFailures -gt 0 -or $null -ne $ciRestoreError -or $aclFailures -gt 0 -or $studioStillRunning -or $script:EfiStillMounted) {
        if ($null -ne $ciRestoreError) {
            Write-Warning "the $CI_LOG channel settings were not restored: $ciRestoreError"
        }
        if ($script:EfiStillMounted) {
            Write-Warning "the EFI system partition is still mounted as S:; run 'mountvol S: /D' by hand"
        }
        throw "revert did not fully restore this machine: $restoreFailures Defender preference(s), $aclFailures Studio tree ACL repair(s), $(if ($null -ne $ciRestoreError) { 'the CodeIntegrity log settings' } else { 'no log settings' })$(if ($script:EfiStillMounted) { ' and the EFI mount' }) still differ from the baseline. See the warnings above."
    }
    # Only here, past every failure check
    $baseline | Add-Member -NotePropertyName RevertCompletedAt `
        -NotePropertyValue ((Get-Date).ToString('o')) -Force
    Save-ProbeBaseline $baseline $baselinePath
    Write-Host 'revert complete. Smart App Control itself was never changed by this script.'
}

switch ($Stage) {
    'prepare' { Invoke-Prepare }
    'run'     { Invoke-Run }
    'collect' { Invoke-Collect }
    'revert'  { Invoke-Revert }
}
