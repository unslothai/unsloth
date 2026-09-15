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

    # prepare only: also set Defender's sample submission to SendAllSamples.
    # Off by default: a file Defender uploads during the probe cannot be
    # recalled by revert, so this is not part of the reversible run.
    [switch] $SendSamples,

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

# A configured path the way Studio normalises one
# (utils/paths/storage_roots.studio_root): trimmed, ~ expanded, made absolute.
# The raw string is not usable: `UNSLOTH_STUDIO_HOME=~\my-studio` is a directory
# literally named '~' under the current one to .NET, so the venv would be
# inventoried in the wrong place and the event scoping, which matches the
# configured path against the real path in each event message, would match
# nothing and file every Unsloth event under 'other'.
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

# The configured Studio home, in Studio's precedence: UNSLOTH_STUDIO_HOME, then
# the STUDIO_HOME alias, and a value that is only whitespace counts as unset.
function Get-StudioHomeOverride {
    $override = Resolve-ConfiguredPath $env:UNSLOTH_STUDIO_HOME
    if (-not $override) { $override = Resolve-ConfiguredPath $env:STUDIO_HOME }
    return $override
}

# Where Studio keeps its state, resolved the way Studio resolves it: the
# UNSLOTH_STUDIO_HOME override (STUDIO_HOME alias), else the legacy home.
function Get-StudioHome {
    $override = Get-StudioHomeOverride
    if ($override) { return $override }
    return (Join-Path $env:USERPROFILE '.unsloth\studio')
}

# The runtime Studio actually loads, in Studio's own order
# (llama_cpp.py _find_llama_server_binary): LLAMA_SERVER_PATH names the binary,
# UNSLOTH_LLAMA_CPP_PATH an install dir, then the folder selected in Studio's
# settings, then the managed default (<home>\llama.cpp for a custom home,
# ~\.unsloth\llama.cpp for the legacy one). The settings choice is only
# visible through the API, so the scenario records it in runtime-selection.json
# and the inventory reads that when it is there. Inventorying the wrong tree
# would omit exactly the files the events are about.
function Get-LlamaDir {
    $binary = Resolve-ConfiguredPath $env:LLAMA_SERVER_PATH
    if ($binary) { return (Split-Path -Parent $binary) }
    $installDir = Resolve-ConfiguredPath $env:UNSLOTH_LLAMA_CPP_PATH
    if ($installDir) { return $installDir }
    $override = Get-StudioHomeOverride
    if ($override) { return (Join-Path $override 'llama.cpp') }
    return (Join-Path $env:USERPROFILE '.unsloth\llama.cpp')
}
# The managed venv. Studio loads far more native code from here than from the
# llama.cpp runtime: 673 PE files against 27 on a measured install, 657 of them
# unsigned, spread across torch, scipy, sklearn, numpy, av and the rest. The
# only enforced 3077 observed so far was in this tree
# (sentencepiece\_sentencepiece.cp313-win_amd64.pyd), not in llama.cpp, so an
# inventory that stops at the runtime dir omits the evidence.
$VENV_DIR = Join-Path (Get-StudioHome) 'unsloth_studio'

function Resolve-LlamaDir([string] $dir) {
    $selection = Join-Path $dir 'runtime-selection.json'
    if (Test-Path -LiteralPath $selection) {
        $sel = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
        if ($sel.resolved_binary) {
            Write-Host ("runtime selected by Studio ({0}): {1}" -f $sel.source, $sel.resolved_binary)
            # The selected root, not the binary's directory. <root>\build\bin\Release
            # is a supported layout (llama_cpp_path_settings.llama_server_candidates),
            # so the binary's parent would inventory Release\ alone and file every
            # sibling PE under the selected root as somebody else's.
            if ($sel.path -and (Test-Path -LiteralPath $sel.path -PathType Container)) {
                return $sel.path
            }
            # A direct LLAMA_SERVER_PATH selection: path IS the binary.
            return (Split-Path -Parent $sel.resolved_binary)
        }
        Write-Warning "Studio reported no resolvable llama-server (source $($sel.source), path $($sel.path)); inventorying $(Get-LlamaDir)"
    } elseif (-not $env:LLAMA_SERVER_PATH -and -not $env:UNSLOTH_LLAMA_CPP_PATH) {
        Write-Warning 'no runtime-selection.json from the scenario: a folder selected in Studio settings cannot be seen from here, so the managed default is inventoried'
    }
    return (Get-LlamaDir)
}
$PE_EXT = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')

# unsloth_cli reads UNSLOTH_STUDIO_PASSWORD itself and treats it as
# set-the-initial-password, so launching with it set hard-errors on any Studio
# that already has one: "an Unsloth admin password is already set; --password
# only sets the initial password." The scenario genuinely needs the value to log
# in. One variable, two incompatible consumers, so take it out of the
# environment here and hand it to the scenario as an argument instead.
$STUDIO_PASSWORD = $env:UNSLOTH_STUDIO_PASSWORD
Remove-Item Env:\UNSLOTH_STUDIO_PASSWORD -ErrorAction SilentlyContinue
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
# Written while the probe holds the EFI mount and removed once it is given
# back. $script:EfiStillMounted only lives as long as one PowerShell process,
# so a stage that could not unmount left the next one finding an EFI-backed S:,
# calling it pre-existing and never retrying: the partition stayed exposed and
# that later stage still reported success.
$EFI_OWNED_MARKER = Join-Path $WorkDir '.efi-mounted-by-probe'

function Mount-Efi {
    if (Test-Path -LiteralPath 'S:\') {
        # S: is already something. Only the EFI system partition may be used as
        # is: the policy tree written onto a data or network volume installs
        # nothing, and the missing 3076 events would then read as an allow.
        if (-not (Test-Path -LiteralPath 'S:\EFI\Microsoft\Boot')) {
            throw 'S: is mapped to a volume that is not the EFI system partition; free the drive letter and run again'
        }
        # Ours from an earlier stage that could not unmount: claim it, so this
        # stage's Dismount-Efi retries rather than leaving it mounted again.
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

# Set when a dismount failed, and read by prepare and revert before either
# reports success. A warning was not enough: the next stage's Mount-Efi finds
# the EFI tree already there, returns $false, and so never retries the unmount,
# leaving the EFI system partition exposed as S: for good.
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

# Retries a dismount an earlier stage could not complete. Mount-Efi is the only
# other reader of the marker, and revert calls it only inside the policy block,
# which is skipped once AuditPolicyApplied is false: a dismount that failed after
# a successful refresh cleared that flag, so the next revert never came back for
# the partition and stamped the rollback complete with S: still exposed.
# Nothing is ever mounted here.
function Clear-EfiOwnership {
    if (-not (Test-Path -LiteralPath $EFI_OWNED_MARKER)) { return }
    if (-not (Test-Path -LiteralPath 'S:\')) {
        Remove-Item -LiteralPath $EFI_OWNED_MARKER -Force -ErrorAction SilentlyContinue
        return
    }
    if (-not (Test-Path -LiteralPath 'S:\EFI\Microsoft\Boot')) {
        # Somebody else's volume took the letter after our mount went away.
        # Unmounting it would be a change to a machine we came here to restore.
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

# Does a preference that reads back as $actual carry the value $expected asked
# for? $true match, $false mismatch, $null when it could not be decided.
#
# Both sides go through the cmdlet's own parameter enum, taken from
# Set-MpPreference's metadata rather than a hand-written table. Get-MpPreference
# returns these as CIM numbers on some builds and as named enum values on
# others, and the baseline JSON stores whatever it was given, so a plain string
# compare would call MAPSReporting 0 equal to 'Advanced' on exactly the machine
# where the request was ignored. [Enum]::Parse takes the name or the number, so
# one path covers both directions.
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

# The type Set-MpPreference declares for a preference, or $null when the cmdlet
# is not there to ask.
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

# The venv the RUNNING Studio uses, not the default location. Get-StudioPython
# already resolves a custom home and the legacy layout; the inventory and the
# event scoping must agree on this directory or a runtime under a custom home
# is inventoried in one place and counted as foreign in the other.
function Resolve-VenvDir([string] $dir) {
    # Recorded by run and read back by collect when $dir is given, the way the
    # llama.cpp selection already is. Both of those resolutions go through
    # UNSLOTH_STUDIO_HOME, which an operator may have set for the shell that ran
    # the earlier stage only, and a reboot between the stages is supported. So
    # recomputing here handed collect the legacy default while run had
    # inventoried a custom venv: the real tree's events were then scoped 'other'
    # and dropped from the counts, and the one enforced 3077 this probe has ever
    # seen was inside that tree.
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

# The Studio install run measured, derived from the venv it recorded. Both of
# these went through Get-StudioHome, so a collect from a shell without a custom
# UNSLOTH_STUDIO_HOME looked for Studio's logs under the legacy default: the
# backend logs were then simply absent from the zip with nothing recording why,
# and no managed interpreter was found to redact even the probe's own raw logs.
# Deriving from the venv is also more accurate than Get-StudioHome on its own,
# since Get-StudioPython already falls back to the legacy root.
function Resolve-StudioHomeFor([string] $dir) {
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
    # No recorded venv, or it is not on this machine: the live search, which is
    # what every caller outside collect wants anyway.
    return (Get-StudioPython)
}

function Test-StudioResponding([int] $port) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/liveness" -TimeoutSec 5 -UseBasicParsing
        if ($r.StatusCode -ne 200) { return $false }
        # Identity, not a bare status code. prepare force-stops whatever owns
        # this port so the restart's loads land inside the window, and 8888 is
        # a busy default (Jupyter among others); a server that answers 200 on
        # an unknown path would have been killed as if it were Studio. The
        # service field has been in this route since the route was added
        # (studio/backend/main.py, "Speed up Studio desktop startup"), so no
        # Studio that can answer here fails this check.
        $body = $null
        try { $body = $r.Content | ConvertFrom-Json } catch { return $false }
        return ($body.service -eq 'Unsloth UI Backend')
    } catch {
        return $false
    }
}

# Every named field of an event's EventData, as an ordered hashtable. Events
# with an unnamed payload fall back to Data1, Data2, ... so nothing is dropped.
# The -like pattern that scopes an event message to one of our trees: the path
# with any drive letter dropped (device paths carry no letter), escaped so [ and
# ] read as the literal characters they are, and ending at a separator. Without
# that separator the pattern is a bare prefix, so a sibling like
# ...\llama.cpp-b10830 beside the selected ...\llama.cpp matched too, and since
# the channel is machine-wide - and running two builds side by side is what the
# matrix asks for - its 3076/3077 was counted as a verdict on the build the
# scenario actually drove.
function Get-ScopeTail([string] $root) {
    $trimmed = ($root -replace '^[A-Za-z]:', '').TrimEnd('\', '/')
    # A runtime at the root of a volume trims to nothing, and the tail would
    # then be a lone separator, which every path in this machine-wide channel
    # contains. The Git Bash msys-2.0.dll events this scoping exists to keep in
    # 'other' would be counted as Unsloth verdicts and the cell would report
    # enforced blocks it never saw. UNSLOTH_LLAMA_CPP_PATH is returned verbatim
    # by Get-LlamaDir, so 'D:\' reaches here directly, and a llama-server.exe at
    # a volume root resolves to the same place. No tail can scope a whole
    # volume, so refuse rather than grade a window against one.
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
        # A record that will not render as XML still has a usable Message.
    }
    return $map
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
        # Out-Host, not a bare call. The installer writes to the output stream,
        # and whatever is left there becomes part of this function's return
        # value, so the caller gets the whole transcript with the interpreter
        # path glued on the end instead of a path, and Start-Process is handed
        # a 5 KB string it cannot resolve.
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

function Stop-Studio([int] $port) {
    <# Stop the Studio answering on $port, children (llama-server, workers) first. #>
    # Only the listeners that can be serving the endpoint Test-StudioResponding
    # actually verified, which is http://127.0.0.1:$port. -LocalPort on its own
    # returns every listener holding that port on any local address, and this
    # function force-stops each owner AND its children: a process listening on
    # ::1 or on a LAN address would have been killed as an unverified stranger,
    # which is the same mistake the /api/liveness identity check exists to
    # prevent. IPV6_V6ONLY defaults to enabled on Windows, so an IPv6 listener
    # cannot be the one that answered 127.0.0.1 (Microsoft, "Dual-Stack Sockets
    # for IPv6 Winsock Applications"); 0.0.0.0 can be, and is kept.
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

function Initialize-Studio([string] $dir, [bool] $allowInstall) {
    <#
        Ensure a Studio exists and is answering. Installing is allowed only
        from prepare: the run stage restarts an existing Studio and never
        installs one, so a prepare -SkipInstall is honoured without the
        operator having to repeat the switch.

        prepare also restarts a Studio that is already up. Its startup is
        where the venv's native modules load, and those loads have to happen
        inside the event window and under the audit policy; a process that
        was already running loaded them before either existed.
    #>
    Write-Section 'Unsloth Studio'
    if (Test-StudioResponding $Port) {
        if (-not $allowInstall) {
            Write-Host "Studio already answering on port $Port"
            return
        }
        Write-Host "Studio is already answering on port $Port; restarting it so its startup loads land inside the window"
        # Not a silent return. prepare's whole premise is that the venv's native
        # modules load inside the window and under the audit policy; a restart
        # that failed leaves them outside it, and the empty window that follows
        # reads exactly like a clean allow.
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
        # revert repairs the ACLs on exactly these, because they are the ones
        # this elevated run brought into being owned by Administrators; an
        # existing tree is the operator's and is left alone.
        # Every tree the installer creates, each checked on its own. Listing
        # only the three roots was not enough: when ~\.unsloth already exists
        # and Studio does not, the installer still makes node\ and whisper.cpp\
        # inside it (studio/setup.ps1 lines 3169 and 5936) and those come out
        # administrator-owned exactly like the rest, leaving the user's own
        # Studio unable to read its own node runtime. The README already named
        # them; the candidate list did not.
        $unslothHome = Join-Path $env:USERPROFILE '.unsloth'
        # node and whisper.cpp do NOT always live under %USERPROFILE%\.unsloth.
        # setup.ps1 resolves $NodeParent from UNSLOTH_STUDIO_HOME / STUDIO_HOME
        # (line 3140-3166) and $UnslothHome from the parent of the managed
        # llama.cpp dir (line 1886), so a custom home moves both. Hard coding the
        # legacy root left the custom-home trees out of StudioInstallRoots and
        # revert then never repaired their ACLs.
        $override = Get-StudioHomeOverride
        $installRoots = @($unslothHome, $override, (Split-Path -Parent (Get-LlamaDir))) |
            Where-Object { $_ } | Select-Object -Unique
        # The managed venv on its own, not left to the Studio home above. A home
        # that already exists drops out of $absentBefore, and
        # <home>\unsloth_studio was not a candidate at all, so the tree holding
        # every native module Studio loads (673 PE files on a measured install)
        # came out administrator-owned with nothing recorded for revert to
        # repair. When every other candidate exists too, nothing is recorded at
        # all, StudioInstalledByProbe stays false, and revert prints "nothing to
        # repair: this run did not install Studio" over a venv the user can no
        # longer read. install.ps1 creates a custom home itself (line 1207), so
        # the reachable shapes are a home the operator made before setting
        # UNSLOTH_STUDIO_HOME, one whose unsloth_studio was moved aside on the
        # installer's own instruction, and a legacy ~\.unsloth\studio whose venv
        # was removed.
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
        # Recorded before the failure check, not after it. An installer that
        # created node\ or .cache\ and then died before producing the managed
        # interpreter still left administrator-owned trees behind, and returning
        # early used to leave StudioInstalledByProbe false so revert skipped them.
        $created = @($absentBefore | Where-Object { Test-Path -LiteralPath $_ })
        $baselinePath = Join-Path $dir 'baseline.json'
        if ($created.Count -gt 0 -and (Test-Path -LiteralPath $baselinePath)) {
            $b = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
            $b.StudioInstalledByProbe = $true
            # Merged, not replaced. A retry after a partial install finds the
            # first attempt's trees already there, so they are no longer in
            # $absentBefore and assigning $created would drop them from the
            # baseline: revert would then leave the tree the first attempt made
            # administrator-owned, which is the failure this record exists for.
            $b.StudioInstallRoots = @(@($b.StudioInstallRoots) + $created |
                Where-Object { $_ } | Select-Object -Unique)
            $b | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
        }
        if (-not $python) {
            # prepare cannot go on: there is no Studio to start, so nothing
            # loads inside the window and the cell measures nothing. Same
            # reasoning as the startup-failure throw below.
            if ($allowInstall) {
                throw 'the installer ran but produced no managed interpreter, so there is no Studio to start and this window would observe nothing. Install Studio by hand and run prepare again; the machine is left prepared, so run revert if you stop here.'
            }
            Write-Warning 'Studio still not found after the installer ran. Install it by hand, then re-run this stage.'
            return
        }
    }
    Write-Host "managed interpreter: $python"
    # Under raw-logs\, which collect redacts into studio-logs\ and never
    # archives as is: the backend's stdout can carry the same tokens its log
    # files can.
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'raw-logs') | Out-Null
    $startLog = Join-Path (Join-Path $dir 'raw-logs') 'studio-start.log'
    # Not discarded. A Studio that never answered loaded none of the venv's
    # native modules inside the window, so the stage that follows measures
    # nothing and prepare used to print "prepare complete" over it. run only
    # warns: its inventories are still worth collecting, and the scenario's own
    # failure is already recorded for collect to grade.
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

    # Fail before mutating anything whose old value we could not read. revert
    # skips a null Defender field and skips the log restore unless both channel
    # settings are present, so a silent capture failure here does not produce a
    # partial rollback later, it produces a machine that prepare changed and
    # revert cannot put back. Only the settings prepare actually writes are
    # required; AMProductVersion and the like are descriptive and may be null.
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
        # $true when an unsigned control raised 3076/3077 under the applied
        # policy, $false never (prepare throws), $null when no control could be
        # built. collect refuses to read an empty window as an allow unless
        # this is $true.
        AuditPolicyControlFired = $null
        # The same question for the cell that runs with no -AuditPolicy at all,
        # where the only thing that can produce a verdict is Smart App Control
        # enforcing for real. $true when an unsigned control was refused with a
        # 3077 (or audited with a 3076) on the boot that measured, $null when no
        # control could be built. Re-asked by run after a reboot, because the
        # registry mode in the baseline is a reading from the previous boot.
        SacControlFired         = $null
        # A policy with the NoISG GUID that was there before prepare: kept
        # aside and put back by revert rather than deleted as ours.
        AuditPolicyPreexisting  = $false
        # Stamped by a revert that fully succeeded. prepare treats a baseline
        # carrying it as spent and captures a new one.
        RevertCompletedAt       = $null
    }
    $path = Join-Path $dir 'baseline.json'
    $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
    Write-Host "baseline written to $path"
    return $baseline
}

# Proves the audit policy is not merely listed but actually evaluating loads.
# Being in CiTool's list is not that: a policy that loaded and evaluates nothing
# produces a window with no 3076 in it, which reads exactly like a clean allow,
# and the whole verdict of a Smart-App-Control-off cell rests on that window.
# The CI job runs the same control for the same reason.
#
# Returns $true (fired), $false (did not fire) or $null (no control could be
# built, so the question was not asked).
#
# $AcceptIds is what counts as the control firing, and the two callers do not
# want the same answer. Under an installed audit policy a 3076 is the expected
# result and a 3077 is the stronger one, so both count. For Smart App Control
# with no audit policy the claim being tested is that unsigned code is REFUSED,
# and only a 3077 shows that: Smart App Control in evaluation mode logs nothing
# to this channel, so a 3076 there was written by some other audit policy on the
# machine and says nothing about enforcement.
function Test-AuditPolicyEvaluating([int[]] $AcceptIds = @(3076, 3077)) {
    $dir = Join-Path $WorkDir ".control-$Label"
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    try {
        $control = Join-Path $dir 'unsigned-control.exe'
        $since = (Get-Date).AddSeconds(-1)
        try {
            # Windows PowerShell 5.1 only. PowerShell 7 dropped assembly
            # emission from Add-Type ("Both the assembly types
            # 'ConsoleApplication' and 'WindowsApplication' are not currently
            # supported"), so on pwsh the cell is marked unverified rather than
            # failed. One line, not a here-string, to keep this readable.
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
        # Launch failure is not an error here. On a machine where Smart App
        # Control is genuinely enforcing, an unsigned binary is refused and
        # never runs: that refusal is a 3077 and is stronger evidence of
        # evaluation than the 3076 an audit-only machine produces.
        try { & $control 2>&1 | Out-Null } catch { }
        # Polled, not slept once. Code integrity writes these asynchronously, so
        # a fixed wait that is slightly too short would report "not evaluating"
        # for a policy that works, which is the one conclusion this must never
        # reach by accident.
        foreach ($attempt in 1..10) {
            Start-Sleep -Seconds 3
            $fired = @(Get-WinEvent -FilterHashtable @{
                LogName = $CI_LOG; StartTime = $since
            } -ErrorAction SilentlyContinue |
                Where-Object { $AcceptIds -contains $_.Id -and $_.Message -like '*unsigned-control*' })
            if ($fired.Count -gt 0) {
                $what = if ($fired[0].Id -eq 3077) { 'refused on this machine' } else { 'evaluated on this machine' }
                Write-Host "positive control raised $($fired[0].Id): unsigned code is $what"
                return $true
            }
        }
        return $false
    } finally {
        # Never left behind, and never staged into the evidence: an unsigned
        # executable does not belong in an archive attached to a public issue.
        Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-Prepare {
    Assert-Elevated
    $dir = Get-RunDir
    $ROLLBACK_POLICY = Get-RollbackPolicyPath $dir
    Write-Section 'Baseline'
    $baselinePath = Join-Path $dir 'baseline.json'
    $previous = $null
    if (Test-Path -LiteralPath $baselinePath) {
        $previous = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
    }
    if ($previous -and -not $previous.RevertCompletedAt) {
        # A retry of this label whose changes are still on the machine. It
        # already carries whatever the first pass changed, so snapshotting it
        # again would record the raised settings, and our own policy, as the
        # state revert should restore.
        $baseline = $previous
        Write-Warning "reusing the baseline captured at $($baseline.CapturedAt) by an earlier prepare of this label; revert restores that state"
    } else {
        # No baseline, or one whose revert completed: revert put the machine
        # back and then left the file behind, so reusing it would hand the next
        # revert a snapshot of how the machine looked before the FIRST run.
        # Anything legitimately changed in between - a Defender preference, the
        # CodeIntegrity channel - would be overwritten with those stale values,
        # which for Defender means lowering protections nobody asked to lower.
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
    # One try per preference, the way revert already does it: they used to share
    # a single try, so the first policy-controlled setting threw and every later
    # Set-MpPreference was never called at all, leaving the cell running without
    # the cloud-block and PUA configuration whose reputation behaviour it
    # measures, and saying so only in one warning.
    $wanted = [ordered]@{
        DisableRealtimeMonitoring = $false
        MAPSReporting             = 'Advanced'
        CloudBlockLevel           = 'High'
        PUAProtection             = 'Enabled'
    }
    if ($SendSamples) {
        # Opt-in only: revert puts the setting back but cannot recall a sample
        # Defender uploaded in the meantime.
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
    # Read back, because a tamper-protected or policy-managed preference is
    # IGNORED rather than refused: Set-MpPreference returns without error and
    # the effective value never changes ("changes to tamper-protected settings
    # are ignored", Microsoft, Protect security settings with tamper
    # protection). Treating the absence of an exception as success let a cell
    # be graded under a baseline the machine never adopted.
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
        # In the evidence, not just the console: the zip is what a reader gets,
        # and a cell measured without the documented baseline has to say so.
        $mpFailed -join "`n" | Set-Content -LiteralPath $mpErrorPath -Encoding UTF8
        Write-Warning "$($mpFailed.Count) Defender preference(s) were NOT applied; this cell does not carry the documented baseline. See defender-preference-errors.txt."
    } else {
        Remove-Item -LiteralPath $mpErrorPath -Force -ErrorAction SilentlyContinue
        Write-Host 'Defender: real-time on, MAPS advanced, cloud block level high, PUA on'
        if ($SendSamples) { Write-Host 'Defender: sample submission set to SendAllSamples (-SendSamples)' }
    }

    Write-Section 'CodeIntegrity log'
    # The default 1 MB fills quickly once a policy is auditing every load, and a
    # wrapped log silently loses the events this whole exercise exists to catch.
    # Not best effort: a channel that stays disabled records nothing, and
    # collect would then export a clean-looking empty window for a bundle
    # that was being refused.
    try {
        Invoke-Native 'wevtutil.exe' @('sl', $CI_LOG, '/e:true', '/ms:67108864')
    } catch {
        throw "could not enable $CI_LOG at 64 MB, so no code integrity event could be collected: $_"
    }
    $ciNow = Get-CiLogSettings
    if (-not $ciNow.Enabled) {
        throw "$CI_LOG is still disabled after wevtutil sl (policy-controlled?); the probe cannot collect evidence here"
    }
    # The size as well as the switch, the check the CI job already makes. A
    # channel that was already enabled reads back enabled even when the resize
    # was refused, and the 1 MB default wraps while an audit policy logs every
    # load in the venv: the dropped 3076s then read as a clean window.
    if ($null -eq $ciNow.MaxSize -or $ciNow.MaxSize -lt 67108864) {
        throw "$CI_LOG is $($ciNow.MaxSize) bytes after wevtutil sl asked for 64 MB (policy-controlled?); auditing every load would wrap the channel before collect reads it, so the probe cannot collect trustworthy evidence here"
    }
    Write-Host ("CodeIntegrity/Operational enabled, max size {0} (was enabled={1}, maxSize={2})" -f $ciNow.MaxSize, $baseline.CiLogEnabled, $baseline.CiLogMaxSize)

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
                # administrator's policy. Under rollback\, which collect
                # leaves out of the zip: a customised policy is deployment
                # detail and is needed only here.
                New-Item -ItemType Directory -Force -Path (Split-Path $ROLLBACK_POLICY) | Out-Null
                Copy-Item -LiteralPath $NOISG_DEST -Destination $ROLLBACK_POLICY -Force
                $baseline.AuditPolicyPreexisting = $true
                Write-Warning "a policy with $NOISG_GUID was already installed; saved to rollback\preexisting-policy.cip and will be restored by revert"
            }
            # Persisted before anything on the EFI partition changes: an
            # interruption between the copy and this write would leave a
            # baseline saying no policy was applied, and revert would skip it.
            $baseline.AuditPolicyApplied = $true
            $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
            New-Item -ItemType Directory -Force -Path (Split-Path $NOISG_DEST) | Out-Null
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
        # Printing whatever is listed is not verification. If the policy did
        # not take, a run with no 3076 events would read as an allow verdict
        # when it was an invalid setup. Only checkable where CiTool exists.
        if ($after.Policies.Count -gt 0 -and -not (Test-PolicyActive $NOISG_GUID)) {
            throw "the audit policy $NOISG_GUID is not in the active policy set after refresh; the machine is left as prepare found it apart from the copied file, run revert"
        }
        if ($after.Policies.Count -eq 0) {
            # No policy list means no evidence the policy is active, and a run
            # with no 3076 events would then read as an allow verdict. Every
            # machine this probe targets ships CiTool; one that does not cannot
            # produce a result this script would stand behind.
            throw "CiTool listed no policies, so the audit policy cannot be verified as active; the copied file is left in place, run revert"
        }

        Write-Section 'Positive control'
        $controlFired = Test-AuditPolicyEvaluating
        if ($false -eq $controlFired) {
            throw "the audit policy $NOISG_GUID is listed as active but an unsigned control binary raised no 3076 or 3077 event, so it is not evaluating loads on this machine; a run with no events would be meaningless. Run revert and re-apply the policy."
        }
        $baseline.AuditPolicyControlFired = $controlFired
        $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
    } else {
        Write-Host ''
        Write-Host 'No -AuditPolicy given. Download the sample policies from https://aka.ms/sacauditpolicies'
        Write-Host 'and re-run with -AuditPolicy <path to SmartAppControlAuditNoISG.bin> to log what'
        Write-Host 'would be blocked. Without it, only real enforcement (3077) shows up, and only on a'
        Write-Host 'machine where Smart App Control is genuinely on.'
        # So establish that it is, rather than letting collect infer an allow
        # from a registry read. VerifiedAndReputablePolicyState says which mode
        # Windows intends; whether the policy actually loaded and is refusing
        # unsigned code on this boot is a separate question, and it is the one
        # the verdict rests on. Ahead of window-start.txt below, so the control's
        # own 3077 never lands in the measured window.
        $sacBefore = Get-SacState
        if ($sacBefore.Mode -eq 'enforcement') {
            Write-Section 'Positive control'
            # 3077 only. A 3076 here would have been logged by some other audit
            # policy on the machine, and an audit is by definition not the
            # refusal this cell's allow verdict rests on.
            $sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)
            if ($false -eq $sacFired) {
                throw "Smart App Control reads as 'enforcement' but an unsigned control binary ran here without being refused with a 3077, so nothing is enforcing against unsigned code on this boot and a window with no events would be meaningless. Re-run prepare with -AuditPolicy, or on a machine where enforcement is live."
            }
            # Add-Member, not an assignment: a baseline reused from an earlier
            # prepare of this label predates the field, and setting a property a
            # PSCustomObject does not carry throws.
            $baseline | Add-Member -NotePropertyName SacControlFired -NotePropertyValue $sacFired -Force
            $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
        }
    }

    # Marks the window collect will export. Set before Studio is installed and
    # started on purpose: the install downloads the llama.cpp bundle and the
    # first launch loads it, so both are inside the observed window and any
    # code integrity event they raise is captured.
    # A retry of this label reopens the window, so the previous window's
    # evidence must not survive into it. Without this, an operator who reruns
    # prepare and then skips run (or whose run dies early) gets a collect that
    # reads the OLD scenario-results.json, calls the load valid, and archives
    # the old inventories and logs against a new and empty event window.
    # baseline.json and rollback\ are deliberately kept: they describe the
    # machine state revert has to restore, which the retry did not change.
    foreach ($stale in @(
        'scenario-status.json', 'scenario-results.json', 'runtime-selection.json',
        'venv-selection.txt',
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

# Subtrees Get-ChildItem could not read, accumulated across both inventories so
# run can put them in the evidence. Without this the enumeration errors were
# discarded and a partial inventory was described as every PE under the tree.
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
                # Status alone is not enough. An unsigned file and one whose
                # chain did not build both report UnknownError.
                StatusMessage = $sig.StatusMessage
                Subject    = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
                Thumbprint = if ($sig.SignerCertificate) { $sig.SignerCertificate.Thumbprint } else { $null }
                TimeStamped = if ($sig.TimeStamperCertificate) { $true } else { $false }
            }
        }
    if ($enumErrors.Count -gt 0) {
        # A subtree that could not be read is exactly where the access-denied
        # native module being investigated would sit, and the caller only
        # rejects a completely empty result.
        $script:InventoryErrors += @($enumErrors | ForEach-Object { "${root}: $_" })
        Write-Warning "$($enumErrors.Count) path(s) under $root could not be enumerated; this inventory is PARTIAL. See inventory-enumeration-errors.txt."
    }
}

function Invoke-Run {
    Assert-Elevated
    $dir = Get-RunDir

    # The policy is re-verified here rather than trusted from prepare. A reboot
    # between the two stages is supported and is itself part of what is being
    # reported, but a policy that did not load on the new boot leaves collect
    # reading a control that fired on the previous one as proof this window was
    # audited: the scenario then runs unaudited and its empty window reads as a
    # clean allow. Only the file on the EFI partition survives a reboot; that it
    # loaded and is evaluating has to be asked again.
    $runBaselinePath = Join-Path $dir 'baseline.json'
    if (Test-Path -LiteralPath $runBaselinePath) {
        $runBaseline = Get-Content -LiteralPath $runBaselinePath -Raw | ConvertFrom-Json
        # A label whose revert already completed. revert does not delete
        # window-start.txt, so the window still open on disk is the one the
        # reverted run measured: this stage would overwrite the scenario results
        # and both inventories with a current runtime while collect kept
        # exporting events from before the revert, and file them against the
        # build measured now. The machine is no longer prepared either, so
        # nothing here would be audited. prepare and revert both refuse a spent
        # baseline; this is the third stage that has to.
        if ($runBaseline.RevertCompletedAt) {
            throw "label '$Label' was reverted at $($runBaseline.RevertCompletedAt), so its baseline is spent and the event window on disk belongs to the run that was undone. Running now would file events from that window against this run's inventory. Run prepare for this label again (or use a new -Label) first."
        }
        # The window on disk has to belong to THIS baseline, and the check above
        # cannot see when it does not. prepare captures a fresh unspent baseline
        # for a reverted label and only reopens the window at the very end, so a
        # throw in between - a CodeIntegrity channel it cannot raise, an
        # -AuditPolicy that is not there, a policy that will not load - leaves an
        # unspent baseline beside the PREVIOUS run's window-start.txt, and the
        # guard above passes on it. collect would then export the reverted run's
        # events against the inventories taken here. A retry that reuses an
        # unspent baseline keeps the window its first pass wrote, which is newer
        # than that baseline, so re-preparing the same label is unaffected.
        $runStartPath = Join-Path $dir 'window-start.txt'
        if (Test-Path -LiteralPath $runStartPath) {
            $runStart = $null
            $runPrepared = $null
            # Invariant, for the reason the two CapturedAt parses below give.
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
            # Invariant, and on the string form, because CapturedAt arrives
            # here in two shapes and the default parse reads neither safely.
            # ConvertFrom-Json silently turns an ISO timestamp into a [datetime],
            # PowerShell then stringifies that with the INVARIANT culture
            # (MM/dd/yyyy) to bind the string overload, while [datetime]::Parse
            # reads the CURRENT one. On a dd/MM machine every prepare after the
            # 12th of a month threw here and left $preparedAt null, so the
            # revalidation below was skipped and collect graded the window on a
            # control that fired on a different boot; on the 1st to the 12th it
            # parsed silently into the wrong month instead.
            try {
                $preparedAt = [datetime]::Parse([string]$runBaseline.CapturedAt,
                    [cultureinfo]::InvariantCulture)
            } catch { }
            if ($bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt) {
                # The control that fired belongs to the previous boot. Ask again
                # on this one, and record the answer where collect reads it.
                Write-Host 'the machine booted after this baseline was captured; re-running the positive control on this boot'
                $controlFired = Test-AuditPolicyEvaluating
                if ($false -eq $controlFired) {
                    throw "the audit policy $NOISG_GUID is listed as active but an unsigned control raised no 3076 or 3077 on this boot, so it is not evaluating loads and this run would be meaningless. Run revert and prepare again."
                }
                $runBaseline.AuditPolicyControlFired = $controlFired
                $runBaseline | ConvertTo-Json -Depth 6 |
                    Set-Content -LiteralPath $runBaselinePath -Encoding UTF8
            }
        } elseif ([string]$runBaseline.Sac.Mode -eq 'enforcement') {
            # The same revalidation for the cell with no audit policy, where the
            # only thing that can produce a verdict is Smart App Control refusing
            # code for real. Windows settles that state at boot
            # (VerifiedAndReputablePolicyState is reconciled against
            # VerifiedAndReputablePolicyStateMinValueSeen on the boot path), and
            # the registry value can in any case disagree with the policy set
            # that actually loaded, so prepare's reading says nothing about this
            # boot. Without this, an empty window on a machine that stopped
            # enforcing after prepare read as a clean allow.
            Write-Section 'Smart App Control still enforcing'
            $sacNow = Get-SacState
            if ($sacNow.Mode -ne 'enforcement') {
                throw "Smart App Control read as 'enforcement' when this label was prepared but reads as '$($sacNow.Mode)' now (a reboot since prepare?), so nothing here can log a verdict and this run's empty window would read as an allow. Run revert and prepare again."
            }
            Write-Host 'Smart App Control is enforcing'
            $bootedAt = $null
            try { $bootedAt = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime } catch { }
            $preparedAt = $null
            # Invariant, and on the string form, because CapturedAt arrives
            # here in two shapes and the default parse reads neither safely.
            # ConvertFrom-Json silently turns an ISO timestamp into a [datetime],
            # PowerShell then stringifies that with the INVARIANT culture
            # (MM/dd/yyyy) to bind the string overload, while [datetime]::Parse
            # reads the CURRENT one. On a dd/MM machine every prepare after the
            # 12th of a month threw here and left $preparedAt null, so the
            # revalidation below was skipped and collect graded the window on a
            # control that fired on a different boot; on the 1st to the 12th it
            # parsed silently into the wrong month instead.
            try {
                $preparedAt = [datetime]::Parse([string]$runBaseline.CapturedAt,
                    [cultureinfo]::InvariantCulture)
            } catch { }
            if (($bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt) -or
                ($true -ne $runBaseline.SacControlFired)) {
                Write-Host 'confirming on this boot that unsigned code is actually refused here'
                # 3077 only, for the reason prepare gives: an audit event is not
                # a refusal, and a 3076 here belongs to somebody else's policy.
                $sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)
                if ($false -eq $sacFired) {
                    throw "Smart App Control reads as 'enforcement' but an unsigned control ran on this boot without being refused with a 3077, so it is not enforcing against unsigned code and this run would be meaningless. Run revert and prepare again."
                }
                $runBaseline | Add-Member -NotePropertyName SacControlFired -NotePropertyValue $sacFired -Force
                $runBaseline | ConvertTo-Json -Depth 6 |
                    Set-Content -LiteralPath $runBaselinePath -Encoding UTF8
            }
        }
    }

    # A machine that was prepared earlier may have been rebooted since, which is
    # itself part of the reported behaviour: Smart App Control re-evaluates from
    # a cleared cache after a restart. Bring Studio back rather than failing.
    # Not under -SkipStudio: a signature-only run must not start, let alone
    # install, Studio inside the evidence window.
    if (-not $SkipStudio -and -not (Test-StudioResponding $Port)) { Initialize-Studio $dir $false }

    # The venv, separately. See $VENV_DIR: this is where the native code
    # actually lives, and where the only enforced block seen so far landed.
    Write-Section 'Venv signature inventory'
    # From the interpreter that actually runs Studio when there is one, so a
    # custom home or an older layout is inventoried rather than guessed at.
    $studioPython = Get-StudioPython
    $venvDir = Resolve-VenvDir
    Write-Host "venv: $venvDir"
    # Recorded for collect, which may run from a shell where a custom
    # UNSLOTH_STUDIO_HOME was never set. See Resolve-VenvDir: without this,
    # collect scoped the window against a different tree than run inventoried.
    # It is staged into the zip like everything else in the run directory, so
    # the reader can see which venv the counts were taken over.
    $venvDir | Set-Content -LiteralPath (Join-Path $dir 'venv-selection.txt') -Encoding UTF8
    $venvInventory = @(Get-SignatureInventory $venvDir)
    ConvertTo-Json -InputObject @($venvInventory) -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'venv-signature-inventory.json') -Encoding UTF8
    $venvInventory | Export-Csv -LiteralPath (Join-Path $dir 'venv-signature-inventory.csv') -NoTypeInformation -Encoding UTF8
    $venvTotal = $venvInventory.Count
    if ($venvTotal -eq 0) {
        # The venv is where the only enforced block so far landed; a zip
        # without it would read as a machine with nothing to block.
        throw "no PE files found under $venvDir; Studio's environment was not found, so the cell is invalid. Check UNSLOTH_STUDIO_HOME or install Studio first"
    }
    $venvValid = @($venvInventory | Where-Object { $_.Status -eq 'Valid' }).Count
    Write-Host "$venvValid of $venvTotal PE files in the venv report a valid Authenticode signature"
    if ($venvTotal -gt 0) {
        # By package, because "657 unsigned" is not actionable and "scipy 106,
        # torch 9 at 274 MB" is.
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
        # Neither UNSLOTH_STUDIO_PASSWORD (unsloth_cli claims that name and
        # hard-errors, see $STUDIO_PASSWORD) nor --password on the command line.
        # An argv secret is readable by anything that can see the process table
        # and is captured verbatim by process-creation auditing and EDR
        # telemetry, for the whole length of a scenario that can run minutes.
        # A distinct variable, set for this child only and removed after, is
        # visible to the process and its children and nothing else.
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
            # In the finally, not on the two ordinary paths. Ctrl+C stops the
            # pipeline without entering either of them, and a finally block runs
            # even then (about_Try_Catch_Finally). A script runs inside the
            # operator's own elevated session, so a variable left behind stays
            # in that console and is inherited by everything started from it
            # afterwards; the point of the separate name is that it does not.
            Remove-Item Env:\SAC_PROBE_STUDIO_PASSWORD -ErrorAction SilentlyContinue
            $ErrorActionPreference = $prev
        }
    }
    $scenarioStatus | ConvertTo-Json -Depth 3 |
        Set-Content -LiteralPath (Join-Path $dir 'scenario-status.json') -Encoding UTF8


    # After the scenario, which records the runtime Studio resolved: the
    # files do not change in between, and the inventory must be of the
    # build the scenario drove.
    Write-Section 'Signature inventory'
    $llamaDir = Resolve-LlamaDir $dir
    Write-Host "runtime: $llamaDir"
    $inventory = @(Get-SignatureInventory $llamaDir)
    # -InputObject: an empty pipeline writes an empty file, not `[]`.
    ConvertTo-Json -InputObject @($inventory) -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'signature-inventory.json') -Encoding UTF8
    $inventory | Export-Csv -LiteralPath (Join-Path $dir 'signature-inventory.csv') -NoTypeInformation -Encoding UTF8

    $total = $inventory.Count
    if ($total -eq 0) {
        # A runtime with no PE files is a stale UNSLOTH_LLAMA_CPP_PATH, an
        # absent install or a failed enumeration; evidence from it would read
        # as a bundle with nothing unsigned.
        throw "no PE files found under ${llamaDir}: there is no llama.cpp runtime on this machine, so the cell is invalid. Run 'python -X utf8 -I -m unsloth_cli studio setup' to download one (or point UNSLOTH_LLAMA_CPP_PATH / UNSLOTH_STUDIO_HOME at an existing install), then re-run this stage"
    }
    $valid = @($inventory | Where-Object { $_.Status -eq 'Valid' }).Count
    Write-Host "$valid of $total PE files report a valid Authenticode signature"
    if ($total -gt 0 -and $valid -lt $total) {
        Write-Host 'unsigned or unverifiable:' -ForegroundColor Yellow
        $inventory | Where-Object { $_.Status -ne 'Valid' } |
            Select-Object Name, Status | Format-Table -AutoSize | Out-String | Write-Host
    }

    # Both inventories are done, so any subtree either of them could not read
    # goes into the evidence rather than scrolling past in the console.
    $enumPath = Join-Path $dir 'inventory-enumeration-errors.txt'
    if ($script:InventoryErrors.Count -gt 0) {
        $script:InventoryErrors -join "`n" | Set-Content -LiteralPath $enumPath -Encoding UTF8
        Write-Warning "$($script:InventoryErrors.Count) path(s) could not be enumerated; the inventories in this cell are PARTIAL and inventory-enumeration-errors.txt names them. Do not read an absent file as an absent PE."
    } else {
        Remove-Item -LiteralPath $enumPath -Force -ErrorAction SilentlyContinue
    }

    Write-Host ''
    if (-not $SkipStudio -and $scenarioStatus.ExitCode -ne 0) {
        # A scenario that never authenticated or never loaded a model produces
        # a window with nothing in it, and an empty window reads exactly like a
        # clean allow. Say so here rather than letting collect imply a result.
        Write-Warning 'the scenario did NOT complete, so this cell has not exercised model loading.'
        Write-Warning "See $log. Fix the cause and re-run this stage before collecting."
    }
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

    # Everything that went wrong while gathering evidence. Written into the
    # staged copy below, because a console warning does not travel with the zip
    # and its recipient would otherwise read a package missing a core artifact
    # as a complete one.
    $collectionProblems = @()
    $events = @()
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
            LogName   = $CI_LOG
            StartTime = $start
        } -ErrorAction Stop | Where-Object { $CI_EVENT_IDS -contains $_.Id })
    } catch {
        # Only "nothing matched" is an empty window. A channel that could not
        # be read is a failed collection, and writing [] for it would ship a
        # clean-looking allow verdict with no events behind it.
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {
            Write-Host 'no CodeIntegrity events in the window'
        } else {
            $_.ToString() | Set-Content -LiteralPath (Join-Path $dir 'events-collection-error.txt') -Encoding UTF8
            throw "could not read $CI_LOG, so the event window was not collected: $_"
        }
    }
    # A retry that got through clears the earlier attempt's marker; leaving it
    # put an artifact claiming this window was never collected into a zip that
    # collected it. defender-query-error.txt is cleaned the same way.
    Remove-Item -LiteralPath (Join-Path $dir 'events-collection-error.txt') -Force -ErrorAction SilentlyContinue

    # The CodeIntegrity channel is machine-wide. An unrelated Git Bash session
    # contributed msys-2.0.dll, head.exe and tail.exe to one run, so a raw count
    # is not a statement about Unsloth. Scope every event by the file it names
    # before anyone reads a verdict off the totals. Nothing is discarded: the
    # export keeps all of it, and 'other' is reported separately.
    #
    # Device paths (\Device\HarddiskVolume3\Users\...) appear alongside drive
    # letters in this channel, so match on the tail rather than anchoring at
    # the root. Resolved the same way the inventory resolves it, so a runtime
    # chosen in Studio settings is scoped in rather than counted as foreign.
    #
    # Escaped, because -like reads [ and ] as pattern syntax rather than as the
    # literal path characters they are here. A directory really called [llama]
    # is legal and supported (tests/test_installer_system32_guard.py), and an
    # unescaped tail matched none of its events, filing its 3076/3077 records
    # under 'other' and dropping them out of the Unsloth headline.
    $tail = Get-ScopeTail (Resolve-LlamaDir $dir)
    $venvTail = Get-ScopeTail (Resolve-VenvDir $dir)
    $shaped = @($events | ForEach-Object {
        $msg = $_.Message
        $scope =
            if ($msg -like "*$tail*") { 'llama.cpp' }
            elseif ($msg -like "*$venvTail*") { 'venv' }
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
            # 'path' when the message named a file we own, 'correlation' when
            # the scope came from a sibling event sharing the ActivityID.
            ScopeFrom   = 'path'
            ActivityID  = $_.ActivityId
            Message     = $msg
            # 3089's rendered Message is the fixed string "Signature
            # information for another event. Match using the Correlation Id."
            # Everything worth correlating - PublisherName, IssuerName,
            # VerificationError, TotalSignatureCount, SignatureType - exists
            # only in EventData, so keeping Message alone makes the ActivityID
            # correlation this script exists to support impossible from the
            # JSON, and sends the reader back to the raw evtx.
            EventData   = (Get-EventDataMap $_)
        }
    })
    # 3089 carries no path: its rendered message is the fixed string
    # "Signature information for another event. Match using the Correlation
    # Id.", so classifying it from Message alone puts every one of them in
    # 'other'. That is exactly the signature detail the README tells a reviewer
    # to correlate to a block, and 'other' is documented as somebody else's
    # binaries, so the scoping would have taught people to discard it. Take the
    # scope from the event it is the detail for. Measured on one run: 278
    # ActivityIDs, each holding exactly one 3076 and one 3089.
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

    # -InputObject: zero events is a normal and important result for an
    # allowed bundle, and piping an empty array writes an empty file that no
    # consumer can tell from a failed collection. This writes `[]`.
    ConvertTo-Json -InputObject $shaped -Depth 4 |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.json') -Encoding UTF8
    $shaped | Format-List | Out-String |
        Set-Content -LiteralPath (Join-Path $dir 'code-integrity-events.txt') -Encoding UTF8

    $ours = @($shaped | Where-Object { $_.Scope -ne 'other' })
    $blocks = @($ours | Where-Object { $_.Id -eq 3077 }).Count
    $audits = @($ours | Where-Object { $_.Id -eq 3076 }).Count
    $foreign = @($shaped | Where-Object { $_.Scope -eq 'other' }).Count
    Write-Host "Unsloth paths: $blocks enforced block(s) (3077), $audits audit would-block(s) (3076), $($ours.Count) event(s)"
    foreach ($g in ($ours | Group-Object Scope | Sort-Object Name)) {
        $b = @($g.Group | Where-Object { $_.Id -eq 3077 }).Count
        $a = @($g.Group | Where-Object { $_.Id -eq 3076 }).Count
        Write-Host ("  {0,-10} {1} x 3077, {2} x 3076, {3} event(s)" -f $g.Name, $b, $a, $g.Count)
    }
    # Whether a model was actually loaded inside this window. Read here, ahead
    # of the verdict and of the staging loop, because a scenario that stopped
    # at login loaded no PE under the policy at all: the window is then a null
    # result however well the positive control fired. This used to be worked
    # out only after the archive had been written, so the zip carried no trace
    # of it and its reader saw an empty event list and a clean warnings file.
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
        # A non-zero exit also covers a failed search, chat or unload after a
        # load that worked; the load is what the events are about, so read that
        # step rather than the aggregate.
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

    # An empty window is an allow only if the policy was shown to be evaluating
    # loads. prepare throws when its control did not fire, so the case left here
    # is the one where no control could be built (PowerShell 7); saying nothing
    # would let that run be read as a clean result.
    # Keyed on the absence of a 3076/3077 verdict, not on the scoped event
    # count. 3033, 3090, 3091, 3092 and 3099 are context: a single scoped
    # allow-and-origin record made $ours non-empty and skipped this warning for
    # exactly the window it exists for, one holding no verdict at all.
    $verdicts = $blocks + $audits
    $baselineForControl = Join-Path $dir 'baseline.json'
    # Read once, here, and written to sac-state-after.json below: the allow
    # verdict has to rest on the state of the machine that held the window open,
    # not on the registry reading prepare took before a reboot the probe
    # explicitly supports.
    $sacNow = Get-SacState
    if ($verdicts -eq 0 -and (Test-Path -LiteralPath $baselineForControl)) {
        $b = Get-Content -LiteralPath $baselineForControl -Raw | ConvertFrom-Json
        # Both ends of the window, and neither is redundant. The baseline value
        # is what was enforcing when the loads during prepare happened; $sacNow
        # is what is enforcing now. A window that spans a change of mode cannot
        # be graded either way.
        $sacMode = [string]$sacNow.Mode
        $sacAtPrepare = [string]$b.Sac.Mode
        if ($b.AuditPolicyApplied -and $true -ne $b.AuditPolicyControlFired) {
            $collectionProblems += 'no positive control confirmed the audit policy was evaluating loads, so a window with no 3076 or 3077 here is a NULL result, not an allow'
            Write-Warning 'No Unsloth path raised a 3076 or 3077, but no positive control confirmed the audit policy was evaluating loads on this machine. Do NOT report this cell as "not blocked".'
        } elseif (-not $loadOk) {
            # Recorded as a collection problem above; what must not happen here
            # is the clean-allow line, which the control alone used to earn.
            Write-Warning 'No Unsloth path raised a 3076 or 3077, but nothing was observed loading in this window (see collection-warnings.txt in the zip). Do NOT report this cell as "not blocked".'
        } elseif ($b.AuditPolicyApplied) {
            Write-Host 'no Unsloth path raised a 3076 or 3077, and the positive control confirmed the policy was evaluating loads'
        } elseif ($sacMode -ne 'enforcement' -or $sacAtPrepare -ne 'enforcement') {
            # -AuditPolicy is optional, and without it nothing on this machine
            # can produce a verdict unless Smart App Control is genuinely
            # enforcing: enforcement logs 3077 by itself, but the policy Smart
            # App Control runs in evaluation mode does not log audit events to
            # this channel at all, and an off machine evaluates nothing
            # (Microsoft, "Test your app with Smart App Control"). So an empty
            # window on any other machine says nothing, and shipping it without
            # a marker is how it gets read as an allow.
            # Both readings are checked because they can differ: Windows settles
            # the mode on the boot path, and the probe supports a reboot between
            # the stages, so a window opened under enforcement can close on a
            # machine that stopped enforcing partway through it.
            $collectionProblems += "no audit policy was applied and Smart App Control is '$sacMode' now ('$sacAtPrepare' when this label was prepared), so nothing on this machine could log a 3076 and nothing could enforce a 3077 for the whole window: a window with no verdict is a NULL result, not an allow. Re-run prepare with -AuditPolicy."
            Write-Warning "No Unsloth path raised a 3076 or 3077, but this cell ran with no audit policy and Smart App Control '$sacMode' ('$sacAtPrepare' at prepare), so no verdict could have been logged either way. Do NOT report this cell as `"not blocked`"; re-run prepare with -AuditPolicy."
        } elseif ($true -ne $b.SacControlFired) {
            # Enforcement in the registry is an intent, not an observation. This
            # branch is what is left once no control could be built (PowerShell 7
            # cannot emit one), and an unverified cell must not be the one that
            # earns the allow line.
            $collectionProblems += 'no audit policy was applied and no unsigned positive control confirmed that Smart App Control actually refused code on the boot that measured, so a window with no 3077 here is a NULL result, not an allow. Re-run prepare and run from Windows PowerShell 5.1, which can build the control.'
            Write-Warning 'No Unsloth path raised a 3077 and Smart App Control reads as enforcing, but no unsigned positive control confirmed it was refusing code on this boot. Do NOT report this cell as "not blocked".'
        } else {
            Write-Host 'no Unsloth path raised a 3077, Smart App Control is enforcing, and an unsigned positive control confirmed it refuses code on this boot, so this window is a real allow'
        }
    }

    # Reported, never folded into the totals above. These are somebody else's
    # binaries and say nothing about whether Studio is blocked.
    Write-Host "unrelated to Unsloth: $foreign event(s) (kept in the export, excluded from the counts)"
    Write-Host "$($shaped.Count) event(s) in the window overall"

    # Raw export as well, since the shaped view drops fields and a reviewer may
    # need the original record. Bounded to the same window as the JSON: an
    # unfiltered `epl` copies the whole channel, which on an established machine
    # is months of unrelated executable paths and policy history, up to the
    # 64 MB prepare configures. The README asks operators to attach this zip, so
    # the raw export has to carry the same scoping as everything else in it.
    $windowStart = $start.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
    $query = "*[System[TimeCreated[@SystemTime>='$windowStart']]]"
    try {
        Invoke-Native 'wevtutil.exe' @('epl', $CI_LOG, (Join-Path $dir 'CodeIntegrity-Operational.evtx'), "/q:$query", '/ow:true')
    } catch {
        $collectionProblems += "raw evtx export failed, so CodeIntegrity-Operational.evtx is missing from this zip: $_"
        Write-Warning "evtx export failed: $_"
    }

    try {
        # Captured into an array first: a clean machine yields nothing, and a
        # pipeline with no input writes an empty file rather than [].
        # The probe's window only: older detections are somebody else's
        # history and their resource paths do not belong in the attached zip.
        $detections = @(Get-MpThreatDetection -ErrorAction Stop |
            Where-Object { $_.InitialDetectionTime -ge $start } |
            Select-Object InitialDetectionTime, ThreatID, Resources)
        ConvertTo-Json -InputObject $detections -Depth 4 |
            Set-Content -LiteralPath (Join-Path $dir 'defender-detections.json') -Encoding UTF8
        Remove-Item -LiteralPath (Join-Path $dir 'defender-query-error.txt') -Force -ErrorAction SilentlyContinue
    } catch {
        # A failed query wrote the same `[]` a clean machine writes, so the
        # evidence claimed a quarantine-free window when Defender had simply
        # not answered. Leave a marker instead; collect reports it and the
        # reader can tell "nothing detected" from "not asked".
        $_.ToString() | Set-Content -LiteralPath (Join-Path $dir 'defender-query-error.txt') -Encoding UTF8
        Remove-Item -LiteralPath (Join-Path $dir 'defender-detections.json') -Force -ErrorAction SilentlyContinue
        Write-Warning "Defender detections were NOT collected: $_. defender-query-error.txt records why; do not read the absence of detections as a clean window."
    }

    # The same object the verdict above was graded against, not a second reading:
    # the zip has to show the reader exactly what collect decided on.
    $sacNow | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $dir 'sac-state-after.json') -Encoding UTF8

    # Studio's own logs, which carry the request timings and the backend errors.
    # Never raw: the zip is attached to an issue, and Studio's logs can carry
    # tokens. They are copied through Studio's own redactor
    # (studio/backend/utils/log_redaction.py) by the managed interpreter that
    # ships it; without that interpreter the logs are left out, not copied.
    # raw-logs\ (the redirected Studio stdout and the scenario's console
    # output) goes through the same redactor and is never archived as is.
    $sources = @()
    # The install run measured, not the one this shell resolves. See
    # Resolve-StudioHomeFor: a collect after the supported between-stage reboot
    # looked under the legacy default and shipped a zip with no backend logs and
    # nothing saying so.
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
                    # Bounded to the probe window; the redactor also caps each
                    # log family, so an established install does not drag its
                    # whole history into the zip.
                    Invoke-Native $python @('-X', 'utf8', '-I', $redactor, $source, (Join-Path $dir 'studio-logs'), '--since', $start.ToString('o'))
                } catch {
                    # Recorded before the partial output is deleted: this branch
                    # throws away every log redacted so far, so without a marker
                    # the zip is announced complete with the documented backend,
                    # startup and scenario logs simply absent.
                    $collectionProblems += "log redaction failed for ${source}, so studio-logs\ is missing from this zip: $_"
                    Write-Warning "logs under $source were not copied (redaction failed): $_"
                    Remove-Item -LiteralPath (Join-Path $dir 'studio-logs') -Recurse -Force -ErrorAction SilentlyContinue
                    break
                }
            }
        } else {
            # Same absence, same marker: raw logs are never archived unredacted,
            # so no interpreter means no studio-logs\ in the zip either.
            $collectionProblems += 'Studio logs were not copied: no managed interpreter to run the redactor, so studio-logs\ is missing from this zip'
            Write-Warning 'Studio logs were not copied: no managed interpreter to run the redactor'
        }
    }

    # Stage a copy before compressing. Start-Studio holds the redirected stdout
    # handle for as long as Studio runs, so Compress-Archive fails with "the
    # process cannot access the file ... because it is being used by another
    # process" on a run where Studio is still up. That inverted the outcome: a
    # run where Studio had died zipped fine, and a successful one could not be
    # collected at all.
    $stage = Join-Path $WorkDir ".stage-$Label"
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $dir -Recurse -File) {
        $rel = $item.FullName.Substring($dir.Length).TrimStart('\')
        # rollback\ holds the administrator's own policy and raw-logs\ the
        # unredacted output; neither goes into the zip.
        if ($rel -like 'rollback\*' -or $rel -like 'raw-logs\*') { continue }
        $target = Join-Path $stage $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        try {
            # FileShare::ReadWrite so a writer holding the file does not block
            # the read, which Copy-Item cannot express.
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
        # Into the staged copy, after the loop, so it is inside the archive the
        # operator attaches rather than only in a console they will not send.
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

    # An empty window is only a result if the scenario actually ran. Repeated
    # here, after the archive line, so it is the last thing the operator reads;
    # the same sentence is inside the zip as a collection warning.
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

    # Ahead of the policy block, and outside it: the block below is skipped
    # whenever AuditPolicyApplied is already false, which is exactly the state a
    # revert whose dismount failed leaves behind. Reclaiming here runs on every
    # revert, so the retry the operator is told to make actually retries.
    Clear-EfiOwnership

    # A baseline this script already considers spent. Writing it back would put
    # the Defender preferences and channel settings the machine had before the
    # FIRST run over anything legitimately changed since the revert that spent
    # it, which for Defender means lowering protections nobody asked to lower.
    # That is the reason prepare captures a fresh baseline rather than reusing a
    # stamped one; revert has to refuse for the same reason. An UNSTAMPED
    # baseline is an incomplete revert and is still retried, which is what the
    # reclaim above and the operator instructions depend on.
    if ($baseline.RevertCompletedAt) {
        if ($script:EfiStillMounted) {
            throw "the EFI system partition is still mounted as S: and could not be unmounted; run 'mountvol S: /D' by hand"
        }
        Write-Host "label '$Label' was already reverted at $($baseline.RevertCompletedAt); this baseline is spent and nothing was changed. Run prepare again for a new run on this label."
        return
    }

    # The policy block may throw (mount, copy or CiTool). The log and Defender
    # restorations below do not depend on it and must still run; the failure
    # is re-raised at the end, with AuditPolicyApplied left set for a retry.
    $policyError = $null
    if ($baseline.AuditPolicyApplied) {
      try {
        Write-Section 'Remove audit policy'
        $saved = $ROLLBACK_POLICY
        $mounted = Mount-Efi
        try {
            if ($baseline.AuditPolicyPreexisting -and -not (Test-Path -LiteralPath $saved)) {
                # Falling through to the removal branch would delete an
                # administrator's policy and then report a completed rollback.
                # Nothing is touched here, so the copy is still recoverable if
                # the file turns up; only the operator can decide the rest.
                throw "the baseline says a policy with $NOISG_GUID was already installed, but the saved copy is missing from $saved, so it cannot be restored. Nothing was changed; restore that .cip by hand (or remove AuditPolicyPreexisting from baseline.json once you have) and run revert again."
            }
            if ($baseline.AuditPolicyPreexisting) {
                # Not ours to delete: put back the policy prepare found.
                Copy-Item -LiteralPath $saved -Destination $NOISG_DEST -Force
                Write-Host 'pre-existing audit policy restored'
            } elseif (Test-Path -LiteralPath $NOISG_DEST) {
                Remove-Item -LiteralPath $NOISG_DEST -Force
                Write-Host 'audit policy removed'
            } else {
                Write-Host 'audit policy file already absent'
            }
            # Refreshed in every branch: an earlier revert may have removed the
            # file and then failed here, leaving the policy active in memory
            # until reboot, and the rerun would otherwise skip the refresh.
            Invoke-Native 'CiTool.exe' @('-r')
            Write-Host 'policy refreshed'
        } finally {
            Dismount-Efi $mounted
        }
        # Only once the refresh succeeded: a rerun after a failed one must
        # still find AuditPolicyApplied set.
        $baseline.AuditPolicyApplied = $false
        $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $dir 'baseline.json') -Encoding UTF8
      } catch {
        $policyError = $_
        Write-Warning "audit policy was not reverted: $_ (continuing with the other settings; run revert again)"
      }
    }

    Write-Section 'Restore CodeIntegrity log'
    # Only where prepare recorded a value; an absent baseline field is an
    # older baseline.json and the setting is left alone.
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
        # Recorded, not just printed: prepare raised this channel to 64MB and
        # enabled it, so a failure here leaves the machine changed.
        $ciRestoreError = $_
        Write-Warning "could not restore the CodeIntegrity log settings: $_"
    }

    Write-Section 'Restore Defender preferences'
    # Only what prepare changed, and only where a baseline value was captured.
    # Each one on its own: a preference that has become policy-controlled
    # throws, and one failure must not skip the values after it.
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
    # Read back, for the same reason prepare does: a preference that became
    # tamper-protected or policy-controlled between the two stages is ignored
    # rather than refused, so $failed stays zero while the machine keeps the
    # raised setting. Without this, revert reports success and spends the
    # baseline (RevertCompletedAt) over a machine it did not restore.
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
    # Carried to the exit status below rather than left as a console warning:
    # a preference that has become policy-controlled does not restore, and an
    # operator reading only the exit code would believe the machine was put
    # back while raised settings remain.
    $restoreFailures = $failed

    Write-Section 'Restore Studio tree access'
    # Every stage is elevated, so a Studio that prepare installs is installed as
    # administrator, and the trees the installer creates (llama.cpp,
    # whisper.cpp, node, .cache) come out owned by BUILTIN\Administrators. The
    # user's own non-elevated Studio then cannot read its own runtime, and
    # nothing else here covers it: prepare never set those ACLs deliberately,
    # they are a side effect of running elevated at all, so there is no baseline
    # value to restore. Measured on a machine where prepare installed Studio:
    # llama.cpp, whisper.cpp and node all denied Get-Acl to the owning user.
    #
    # Only the trees THIS run created, recorded by prepare. Granting Full
    # Control over a tree the probe did not create is not a repair, it is a
    # permission change to somebody else's directory: UNSLOTH_STUDIO_HOME and
    # UNSLOTH_LLAMA_CPP_PATH may point at a shared or administrator-managed
    # runtime, and revert has no business widening access there. Containment is
    # re-checked here rather than trusted from the file, since baseline.json is
    # editable between stages.
    # The roots are re-derived from the live environment, not %USERPROFILE%
    # alone: a custom UNSLOTH_STUDIO_HOME may sit on another volume (D:\Unsloth)
    # and prepare records the trees the installer created there, so anchoring on
    # the profile discarded exactly the trees this repair exists for. Two gates
    # still stand between a recorded path and a grant: prepare must have seen it
    # come into being during this run, and it must resolve under a root the
    # environment designates right now.
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
        # A recorded tree that is gone needs no repair. Only one that still
        # exists and is out of scope is a rollback this revert did not carry out.
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
    # Counted, not just warned about. This loop is outside the Defender
    # accounting above, so revert used to print "revert complete" and exit zero
    # while the user's own Studio tree was still unreadable, which is the exact
    # failure the loop was added for. icacls returns nonzero when any object in
    # the tree failed even under /C, so a partial repair counts as a failure
    # here on purpose.
    # Seeded with the trees this revert refused to touch, not zero. An override
    # that was only ever set in the prepare shell is gone after a reboot, so the
    # recorded trees fall outside the roots derived here; warning and filtering
    # them out left $aclFailures at zero and revert stamped the baseline complete
    # over administrator-owned trees the user still cannot read.
    $aclFailures = $rejected.Count
    foreach ($tree in $trees) {
        try {
            # Grants the invoking user only. Nothing here widens access for
            # anyone else or touches ownership.
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
    # Every restoration is attempted first, then the failures decide the exit
    # status. Reporting success here while settings stayed raised is the same
    # class of bug as collect implying evidence it did not have.
    if ($restoreFailures -gt 0 -or $null -ne $ciRestoreError -or $aclFailures -gt 0 -or $script:EfiStillMounted) {
        if ($null -ne $ciRestoreError) {
            Write-Warning "the $CI_LOG channel settings were not restored: $ciRestoreError"
        }
        if ($script:EfiStillMounted) {
            Write-Warning "the EFI system partition is still mounted as S:; run 'mountvol S: /D' by hand"
        }
        throw "revert did not fully restore this machine: $restoreFailures Defender preference(s), $aclFailures Studio tree ACL repair(s), $(if ($null -ne $ciRestoreError) { 'the CodeIntegrity log settings' } else { 'no log settings' })$(if ($script:EfiStillMounted) { ' and the EFI mount' }) still differ from the baseline. See the warnings above."
    }
    # Only here, past every failure check: a revert that left something raised
    # still needs its baseline, and a rerun must find it unspent. Marked rather
    # than deleted so the record of what the machine looked like stays with the
    # rest of the run's evidence.
    $baseline | Add-Member -NotePropertyName RevertCompletedAt `
        -NotePropertyValue ((Get-Date).ToString('o')) -Force
    $baseline | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
    Write-Host 'revert complete. Smart App Control itself was never changed by this script.'
}

switch ($Stage) {
    'prepare' { Invoke-Prepare }
    'run'     { Invoke-Run }
    'collect' { Invoke-Collect }
    'revert'  { Invoke-Revert }
}
