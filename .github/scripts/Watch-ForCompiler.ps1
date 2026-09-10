# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

<#
.SYNOPSIS
Run a script block and report whether a C# compiler ran, or a DLL landed in a
temporary directory, while it did.

.DESCRIPTION
Two independent detectors, because either alone can be defeated by something that
is not the code under test:

  1. Security log event 4688, process creation. Authoritative about what ran, and
     it sees a compiler spawned through any depth of child process, which a text
     search of our own scripts cannot reach. Needs auditing enabled by the caller;
     the workflow does that and verifies it took.

  2. New *.dll, *.cmdline, *.rsp and *.?.cs files under every temporary directory
     in play. The artefact half of the same shape, and it survives auditing being
     silently overridden by machine policy. csc.exe writes the source and the
     response file next to the assembly, so those names are watched too.

     Watched live, with a FileSystemWatcher, and not only by comparing a listing
     taken before the action against one taken after. CodeDom deletes its whole
     intermediate directory once the assembly is loaded, so on a hosted runner the
     before-and-after diff saw nothing at all while 4688 recorded

         csc.exe /noconfig /fullpaths @"...\Temp\vpmyd5eq\vpmyd5eq.cmdline"

     which is a compile this half missed entirely. The two are unioned: the listing
     catches what was left behind, the watcher catches what was cleaned up.

Both are reported. The positive control requires both to fire, and the real
measurement requires neither to.

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
    # caller casts this into a HashSet whose two-argument constructor rejects null.
    # On a clean runner that killed the watcher before the positive control ran.
    return ,[string[]]$found
}

$script:ArtifactPattern = '\.(dll|cmdline|rsp|cs|err|out)$'
$script:LibraryPattern = '\.(dll|cmdline|rsp)$'

function Start-StudioTempWatch {
    <#
    .SYNOPSIS
    Begin recording file creations under every temp root, and return the handles.
    .DESCRIPTION
    Register-ObjectEvent without an -Action: the events queue in the session's event
    manager as they are raised, and Stop-StudioTempWatch drains them afterwards. An
    -Action block would have to run for anything to be recorded, and there is nothing
    to run it while a synchronous installer holds the pipeline.

    A root that cannot be watched is skipped rather than fatal. The listing half still
    covers it, and on a machine where none of them can be watched the positive control
    is what says so.
    #>
    $handles = @()
    foreach ($root in (Get-StudioTempRoots)) {
        try {
            $watcher = New-Object System.IO.FileSystemWatcher
            $watcher.Path = $root
            $watcher.IncludeSubdirectories = $true
            $watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName
            # The default 8 KB buffer overflows on a busy temp directory, and an
            # overflow drops events silently, which here reads as a clean run.
            $watcher.InternalBufferSize = 65536
            $identifier = "StudioTempWatch-" + [guid]::NewGuid().ToString('N')
            $null = Register-ObjectEvent -InputObject $watcher -EventName Created `
                -SourceIdentifier $identifier
            $watcher.EnableRaisingEvents = $true
            $handles += [pscustomobject]@{
                Watcher          = $watcher
                SourceIdentifier = $identifier
                Root             = $root
            }
        } catch {
            continue
        }
    }
    return ,[object[]]$handles
}

function Stop-StudioTempWatch {
    <#
    .SYNOPSIS
    Stop recording and return every path created while the handles were live.
    .DESCRIPTION
    Always unregisters and disposes, including on a path that saw nothing: a leaked
    subscription keeps firing into the next measurement's queue.
    #>
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Handle)

    # Delivery is asynchronous, so the last few creations before the action returned may
    # still be in flight. Settle first, then stop raising: draining immediately dropped
    # exactly the events that matter, the ones from the end of a compile.
    Start-Sleep -Milliseconds 750

    $seen = @()
    foreach ($entry in $Handle) {
        try { $entry.Watcher.EnableRaisingEvents = $false } catch { }
        try {
            foreach ($record in @(Get-Event -SourceIdentifier $entry.SourceIdentifier `
                    -ErrorAction SilentlyContinue)) {
                $path = ''
                try { $path = [string]$record.SourceEventArgs.FullPath } catch { }
                if (-not [string]::IsNullOrWhiteSpace($path)) { $seen += $path }
                Remove-Event -EventIdentifier $record.EventIdentifier -ErrorAction SilentlyContinue
            }
        } catch { }
        Unregister-Event -SourceIdentifier $entry.SourceIdentifier -ErrorAction SilentlyContinue
        try { $entry.Watcher.Dispose() } catch { }
    }
    return ,[string[]]$seen
}

function Get-StudioProcessImageName {
    <#
    .SYNOPSIS
    The image a 4688 record says was created, from the record's own field.
    .DESCRIPTION
    NewProcessName, read out of the event XML by name rather than by position, so a
    schema that gains a field still means the same thing. Nothing else is consulted:
    the rendered message also carries the command line, so matching it would score
    `cmd.exe /c echo csc.exe` as a compiler.
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
    Separate from the query so the classification can be exercised without a
    Security log, which is the only way to test it off a Windows runner.
    #>
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][array]$Events)

    $hits = @()
    foreach ($record in $Events) {
        $image = Get-StudioProcessImageName -Event $record
        if ([string]::IsNullOrEmpty($image)) { continue }
        # Split explicitly, not via [System.IO.Path]::GetFileName, which splits on the
        # HOST's separators: under the Linux pwsh where this is tested a backslash is an
        # ordinary character and the whole path came back as the leaf. The records are
        # always Windows paths whatever reads them.
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
    unrelated service starting a compiler after the action would be scored against it.

    A window, not the process tree the workflow prose describes. 4688 carries the
    creator's pid, but a compile can be several processes deep and the intermediate
    pids have exited by the time this reads the log, so ancestry is not
    reconstructable after the fact. The positive control proves the window measures
    anything.
    #>
    param(
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][datetime]$Until
    )

    $events = @()
    try {
        # Bounded at both ends. Open-ended, a compiler started by something else during
        # the recursive temp scan counted against the action that had already finished.
        $events = Get-WinEvent -FilterHashtable @{
            LogName   = 'Security'
            Id        = 4688
            StartTime = $Since
            EndTime   = $Until
        } -ErrorAction Stop
    } catch [System.Exception] {
        # No matching events is an exception from Get-WinEvent, not an empty set, and on
        # a clean run that is expected. A log this cannot READ throws the same way, so
        # swallowing both would print "no compiler" having seen nothing at all. The
        # positive control runs in an earlier step and says nothing about whether the log
        # was readable during the measurements.
        #
        # Separate them structurally rather than by the localised message text: ask the
        # log for any one record. If that succeeds the filter genuinely matched nothing;
        # if it fails too, this measurement is void, not clean.
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
    the exit state of the action. Evidence is written under $EvidenceRoot even when
    the action fails, which is exactly when the trace is wanted.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$EvidenceRoot
    )

    New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

    # Baseline first, THEN open the window. The sweep walks every temp root
    # recursively and can take seconds, and a csc.exe the machine started during that
    # walk predates the action, so counting it fails a measurement for something it
    # did not do. Same reasoning as the $until below.
    $before = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]](Get-StudioTempArtifacts), [StringComparer]::OrdinalIgnoreCase)
    # A second back, so a process created in the same tick as the timestamp survives
    # Get-WinEvent's strictly-later comparison. That reaches one second into the tail
    # of the sweep above, so a compiler started in that second is counted: a deliberate
    # trade towards a loud false alarm rather than a dropped real compile.
    $since = (Get-Date).AddSeconds(-1)
    # Opened here, with the 4688 window, and not before the baseline: a file the
    # machine creates during that recursive sweep predates the action.
    $watch = Start-StudioTempWatch

    $failure = $null
    $live = @()
    try {
        # Out-Host, not the success stream. The installer action tees its log, and those
        # lines would be emitted as function output ahead of the result hashtable, making
        # the caller's $seen an object array whose $seen.Compilers fails under
        # Set-StrictMode instead of reporting the measurement.
        & $Action | Out-Host
    } catch {
        # Recorded and re-thrown below. The detectors still report, because "the
        # installer died AND spawned a compiler" beats either half alone.
        $failure = $_
    } finally {
        # In the finally, so an action that threw still closes its subscriptions.
        $live = @(Stop-StudioTempWatch -Handle $watch)
    }

    # Closed before the temp sweep, which can take seconds: anything the machine
    # starts during that walk belongs to nobody's measurement.
    $until = Get-Date
    $compilers = @(Get-StudioCompilerEvents -Since $since -Until $until)

    $after = Get-StudioTempArtifacts
    $left = @($after | Where-Object { -not $before.Contains($_) })
    # Only the names the compiler writes, because the watcher reports every creation
    # under temp and most of them are nobody's business.
    $transient = @($live | Where-Object { $_ -match $script:ArtifactPattern })
    $union = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $newArtifacts = @()
    foreach ($path in ($left + $transient)) {
        if ($union.Add($path)) { $newArtifacts += $path }
    }
    $newLibraries = @($newArtifacts | Where-Object { $_ -match $script:LibraryPattern })

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
