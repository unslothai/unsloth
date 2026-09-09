# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

<#
.SYNOPSIS
Run a script block and report whether a C# compiler ran, or a DLL landed in a
temporary directory, while it did.

.DESCRIPTION
Two independent detectors, because either one alone can be defeated by something
that is not the code under test:

  1. Security log event 4688, process creation. Authoritative about what ran, and
     it sees a compiler spawned through any depth of child process, which is the
     case a text search of our own scripts cannot reach. Needs auditing enabled
     by the caller; the workflow does that and verifies it took.

  2. New *.dll, *.cmdline, *.rsp and *.?.cs files under every temporary directory
     in play. This is the artefact half of the same shape, and it survives the
     case where auditing is silently overridden by a machine policy. csc.exe
     writes the source and the response file next to the assembly, so the
     supporting names are watched too: an assembly written and cleaned up leaves
     the job with nothing to show, and a .cmdline left behind still names it.

Both are reported. A caller decides what to do with them; the positive control
requires both to fire, and the real measurement requires neither to.

TEMP is read from the environment rather than assumed, and the machine-wide
C:\Windows\Temp is watched alongside it, because a compile launched from a
service or an elevated child does not write where this process would.
#>

Set-StrictMode -Version Latest

$script:CompilerNames = @('csc.exe', 'vbc.exe', 'cvtres.exe', 'jsc.exe')

function Get-StudioTempRoots {
    <#
    .SYNOPSIS
    Every directory a compile could write its intermediates to, de-duplicated.
    #>
    $roots = @($env:TEMP, $env:TMP, "$env:SystemRoot\Temp", "$env:LOCALAPPDATA\Temp")
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $result = @()
    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $full = (Resolve-Path -LiteralPath $root).ProviderPath
        if ($seen.Add($full)) { $result += $full }
    }
    return $result
}

function Get-StudioTempArtifacts {
    <#
    .SYNOPSIS
    Compiler intermediates and assemblies currently sitting in the temp roots.
    #>
    $patterns = @('*.dll', '*.cmdline', '*.rsp', '*.cs', '*.err', '*.out')
    $found = @()
    foreach ($root in (Get-StudioTempRoots)) {
        foreach ($pattern in $patterns) {
            # Recurse: PowerShell compiles into a per-invocation subdirectory, not
            # into the root, so a non-recursive listing sees none of this.
            $found += Get-ChildItem -LiteralPath $root -Filter $pattern -File -Recurse `
                -Force -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName
        }
    }
    # Comma-wrapped: PowerShell unrolls an empty array to nothing on return, and the
    # caller casts this into a HashSet whose two-argument constructor rejects null. On a
    # clean runner with no matching temp files that killed the watcher before it ran even
    # the positive control.
    return ,[string[]]$found
}

function Get-StudioProcessImageName {
    <#
    .SYNOPSIS
    The image a 4688 record says was created, from the record's own field.
    .DESCRIPTION
    NewProcessName, read out of the event XML by name rather than by position, since
    the position is a property of the schema and this has to keep meaning the same
    thing if the schema gains a field. Nothing else in the record is consulted: the
    rendered message also contains the command line, so matching the message scores
    `cmd.exe /c echo csc.exe` as a compiler, which is a false report of the one thing
    this job exists to detect.
    #>
    param([Parameter(Mandatory = $true)]$Event)

    try {
        $xml = [xml]$Event.ToXml()
        foreach ($field in $xml.Event.EventData.Data) {
            if ($field.Name -eq 'NewProcessName') { return [string]$field.'#text' }
        }
    } catch { }
    return ''
}

function Select-StudioCompilerHits {
    <#
    .SYNOPSIS
    The records among $Events whose created image is a compiler.
    .DESCRIPTION
    Separate from the query so it can be exercised without a Security log, which is
    the only way to test the classification anywhere but a Windows runner that has
    just compiled something.
    #>
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][array]$Events)

    $hits = @()
    foreach ($record in $Events) {
        $image = Get-StudioProcessImageName -Event $record
        if ([string]::IsNullOrEmpty($image)) { continue }
        # Split explicitly rather than through [System.IO.Path]::GetFileName, which splits on
        # the separators of the HOST it runs on: under a Linux pwsh, which is where the
        # classification is tested, a backslash is an ordinary character and the whole path
        # came back as the leaf. The records are always Windows paths whatever reads them.
        $leaf = ($image -split '[\\/]')[-1]
        foreach ($name in $script:CompilerNames) {
            if ($leaf -eq $name) {
                $rendered = ''
                try { $rendered = [string]$record.Message } catch { }
                $hits += ("{0:o} {1} :: {2}" -f $record.TimeCreated, $image,
                    ($rendered -replace '\s+', ' '))
                break
            }
        }
    }
    return ,[string[]]$hits
}

function Get-StudioCompilerEvents {
    <#
    .SYNOPSIS
    4688 records naming a compiler image, created at or after $Since.
    .PARAMETER Since
    The instant the measured action began. Taken before the action rather than
    filtering afterwards by a fixed window, so a slow installer cannot outrun it.
    .PARAMETER Until
    The instant it ended. Both ends are needed: a runner is a shared machine, and an
    unrelated service starting a compiler after the action would otherwise be scored
    against it.

    This window is the honest bound rather than the process tree the workflow prose
    describes. 4688 carries the creator's pid, but a compile can be several processes
    deep and the intermediate pids have exited by the time this reads the log, so an
    ancestry walk is not reconstructable after the fact. The window is what can be
    measured; the positive control is what proves it measures anything.
    #>
    param(
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][datetime]$Until
    )

    $events = @()
    try {
        # Bounded at both ends. Open-ended, a compiler started by something else while the
        # recursive temp scan was still running counted against the action that had
        # already finished.
        $events = Get-WinEvent -FilterHashtable @{
            LogName   = 'Security'
            Id        = 4688
            StartTime = $Since
            EndTime   = $Until
        } -ErrorAction Stop
    } catch [System.Exception] {
        # No matching events is an exception from Get-WinEvent, not an empty set, and on a
        # clean run that is the expected result. A log this cannot READ throws the same way,
        # and swallowing both made an unreadable Security log indistinguishable from a clean
        # measurement: the job would print "no compiler" having seen nothing at all. The
        # positive control does not cover this, because it runs in an earlier step and says
        # nothing about whether the log was still readable during the two measurements.
        #
        # So the two are separated structurally rather than by matching the message text,
        # which is localised: ask the log for any one record at all. If that succeeds the log
        # is readable and the filter genuinely matched nothing; if it fails too, this
        # measurement is void and the caller must not report it as clean.
        # Bound before the probe below, whose own catch rebinds $_.
        $reason = $_.Exception.Message
        $readable = $false
        try {
            $null = Get-WinEvent -LogName 'Security' -MaxEvents 1 -ErrorAction Stop
            $readable = $true
        } catch { }
        if (-not $readable) {
            throw ("the Security log could not be read, so this run measured nothing. " +
                   "Treat it as void rather than as clean. Underlying error: $reason")
        }
        return ,[string[]]@()
    }

    return Select-StudioCompilerHits -Events @($events)
}

function Invoke-WithCompilerWatch {
    <#
    .SYNOPSIS
    Run $Action and return what the two detectors saw while it ran.
    .OUTPUTS
    A hashtable with Compilers and TempLibraries, each an array of strings, plus
    the exit state of the action. Evidence is written under $EvidenceRoot whether
    the action succeeded or not, since a failed installer is exactly when the
    trace is wanted.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$EvidenceRoot
    )

    New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

    # Baseline first, THEN open the window. The sweep walks every temp root
    # recursively and can take seconds on a runner with a large temp tree, and a
    # csc.exe the machine started during that walk predates the action: counting it
    # fails the positive control or the installer measurement for something neither
    # one did. Same reasoning as the $until below, which is taken before the second
    # sweep for the same reason.
    $before = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]](Get-StudioTempArtifacts), [StringComparer]::OrdinalIgnoreCase)
    # A second back, so a process created in the same tick as the timestamp is not
    # filtered out by a strictly-later comparison inside Get-WinEvent. Stated exactly,
    # since the sweep above runs before this: the window still reaches one second into
    # the tail of that sweep, so a compiler the machine started in that last second is
    # counted. The trade is deliberate and it errs towards a false alarm, which fails a
    # measurement loudly, rather than towards dropping a real compile.
    $since = (Get-Date).AddSeconds(-1)

    $failure = $null
    try {
        # Piped to Out-Host, not left on the success stream. The installer action tees its
        # log, and every one of those lines would otherwise be emitted as function output
        # ahead of the result hashtable, so the caller's $seen became an object array and
        # $seen.Compilers failed under Set-StrictMode against the string elements rather
        # than reporting the measurement.
        & $Action | Out-Host
    } catch {
        # Recorded and re-thrown by the caller if it cares. The detectors still
        # report, because "the installer died AND spawned a compiler" is a more
        # useful message than either half alone.
        $failure = $_
    }

    # Closed before the temp sweep, not after it: the sweep walks every temp root
    # recursively and can take seconds, and anything the machine starts during that walk
    # belongs to nobody's measurement.
    $until = Get-Date
    $compilers = @(Get-StudioCompilerEvents -Since $since -Until $until)

    $after = Get-StudioTempArtifacts
    $newArtifacts = @($after | Where-Object { -not $before.Contains($_) })
    $newLibraries = @($newArtifacts | Where-Object { $_ -match '\.(dll|cmdline|rsp)$' })

    $stem = Join-Path $EvidenceRoot $Name
    $compilers | Out-File -FilePath "$stem-compilers.txt" -Encoding utf8
    $newArtifacts | Out-File -FilePath "$stem-temp-artifacts.txt" -Encoding utf8
    if ($failure) {
        $failure | Out-String | Out-File -FilePath "$stem-error.txt" -Encoding utf8
    }

    if ($failure) { throw $failure }

    return @{
        Compilers     = $compilers
        TempLibraries = $newLibraries
        TempArtifacts = $newArtifacts
    }
}
