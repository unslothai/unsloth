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

     A DLL is only reported when a compile is evidenced in ITS OWN directory; see
     Select-StudioCompilerLibraries. Reporting every DLL under TEMP is not the same
     claim, and an installer that unpacks a verified archive there makes the
     difference the whole result.

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

function Get-StudioTempSubtree {
    <#
    .SYNOPSIS
    Every file under one root, walked a directory at a time so that one unreadable
    directory costs that directory and nothing else.

    .DESCRIPTION
    Get-ChildItem -Recurse is the obvious way to do this and it is not safe here.
    A temp root is shared with everything else running on the machine, so a
    directory can be removed or become unopenable partway through the walk, and
    the provider raises a Win32Exception that -ErrorAction SilentlyContinue does
    not suppress: that parameter governs non-terminating errors, and with
    $ErrorActionPreference = 'Stop' set by the caller this one ends the step.
    Observed on hosted runners as

        Get-ChildItem : The system cannot find the file specified
        + CategoryInfo : NotSpecified: (:) [Get-ChildItem], Win32Exception

    from inside the positive control, which failed the job while the installer
    under test had done nothing wrong. Walking by hand means the failure is
    contained to the one directory that raised it, and the rest of the subtree is
    still reported.

    Reparse points are not followed. A junction into an ancestor would otherwise
    walk forever, and a compile does not write through one.

    Filtering happens here rather than in the caller. A temp root can hold an
    extracted toolchain or a package cache, and this sweep runs before and after
    every measured action, so collecting every path first and selecting afterwards
    means carrying tens of thousands of strings that were never of interest.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$Patterns
    )
    # Generic lists, not PowerShell arrays. += on an array allocates a new one and
    # copies, so a large temp tree costs quadratic time in the number of entries on
    # a path that runs twice per measured action.
    $found = New-Object 'System.Collections.Generic.List[string]'
    $unread = New-Object 'System.Collections.Generic.List[string]'
    $pending = New-Object 'System.Collections.Generic.List[string]'
    $pending.Add($Root)
    $visited = 0
    while ($pending.Count -gt 0) {
        $visited++
        if ($visited -gt 200000) {
            # Not a break. A truncated snapshot is indistinguishable from a clean one
            # to the caller, and this listing is exactly what stands in when the
            # watcher cannot attach, so a silent stop turns a missed artifact into a
            # clean verdict. Better to declare the measurement void.
            throw ("the temp scan of $Root passed $visited directories without finishing. " +
                   "A partial snapshot would be read as a complete one, so this run cannot " +
                   "say whether a compiler ran.")
        }
        $dir = $pending[$pending.Count - 1]
        $pending.RemoveAt($pending.Count - 1)
        $entries = @()
        try { $entries = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop) }
        catch {
            # Recorded, not just skipped. The caller subtracts the baseline listing from the
            # final one, so a directory that fails HERE and succeeds in the other snapshot
            # silently changes the answer: files that were there all along show up as new and
            # an innocent action is reported as having compiled. The reverse hides a real
            # artifact. Which directory went unread has to survive to the comparison.
            #
            # A directory that no longer exists is not a gap. It cannot contribute a file to
            # a later listing of itself, and re-reading a path that raised for any other
            # reason is how a transient lock gets a second chance.
            if (Test-Path -LiteralPath $dir) {
                try { $entries = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop) }
                catch { $unread.Add($dir); continue }
            } else {
                continue
            }
        }
        foreach ($entry in $entries) {
            if ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
            if ($entry.PSIsContainer) { $pending.Add($entry.FullName); continue }
            foreach ($pattern in $Patterns) {
                if ($entry.Name -like $pattern) {
                    $found.Add($entry.FullName)
                    break
                }
            }
        }
    }
    return [pscustomobject]@{
        Files  = [string[]]$found.ToArray()
        Unread = [string[]]$unread.ToArray()
    }
}

function Get-StudioTempArtifacts {
    <#
    .SYNOPSIS
    Compiler intermediates and assemblies currently sitting in the temp roots, and the
    directories this sweep could not read.

    .DESCRIPTION
    Both halves are returned because the caller compares two of these snapshots and a
    directory missing from one side is not the same thing as a directory that is empty.
    See the comparison in Invoke-WithCompilerWatch.
    #>
    $patterns = @('*.dll', '*.cmdline', '*.rsp', '*.cs', '*.err', '*.out')
    $found = New-Object 'System.Collections.Generic.List[string]'
    $unread = New-Object 'System.Collections.Generic.List[string]'
    foreach ($root in (Get-StudioTempRoots)) {
        # The whole subtree: PowerShell compiles into a per-invocation subdirectory,
        # not into the root, so a non-recursive listing sees none of this.
        $scan = Get-StudioTempSubtree -Root $root -Patterns $patterns
        $found.AddRange([string[]]$scan.Files)
        $unread.AddRange([string[]]$scan.Unread)
    }
    # Arrays cast explicitly: PowerShell unrolls an empty array to nothing, and the
    # caller casts Files into a HashSet whose two-argument constructor rejects null.
    # On a clean runner that killed the watcher before the positive control ran.
    return [pscustomobject]@{
        Files  = [string[]]$found.ToArray()
        Unread = [string[]]$unread.ToArray()
    }
}

function Test-StudioPathUnder {
    <#
    .SYNOPSIS
    Is $Path inside $Directory, or the directory itself?

    .DESCRIPTION
    Compared with a trailing separator appended to the directory, so C:\Temp\ab does not
    count as being under C:\Temp\a. Withholding a sibling that merely shares a name prefix
    would drop real evidence, which is the opposite of what the caller wants.

    Both separators are accepted rather than [System.IO.Path]::DirectorySeparatorChar. That
    property is '/' under pwsh on Linux, where this script's own tests run, so pinning to it
    made every Windows-shaped path compare false and the withholding silently did nothing.

    Case-insensitive, like the rest of this script's path handling, because the paths reach
    here from two different APIs: the directory from Get-ChildItem and the file path from the
    same walk or from FileSystemWatcher.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Directory
    )
    $trimmed = $Directory.TrimEnd('\', '/')
    foreach ($separator in @('\', '/')) {
        if ($Path.StartsWith($trimmed + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

$script:ArtifactPattern = '\.(dll|cmdline|rsp|cs|err|out)$'
# A compiler's own files, wherever they appear. Nothing else writes a .cmdline.
$script:CompilerFilePattern = '\.(cmdline|rsp)$'
# What csc leaves beside the assembly it produced: the response file, the generated
# source, and the captured streams.
$script:CompilerSiblingPattern = '\.(cmdline|rsp|cs|err|out)$'

function Get-StudioParentPath {
    <#
    .SYNOPSIS
    The directory part of $Path, split on either separator.
    .DESCRIPTION
    Not Split-Path, for the reason Select-StudioCompilerHits splits by hand: under the
    Linux pwsh these functions are tested on, a backslash is an ordinary character and a
    Windows path comes back whole. The paths here are always Windows paths whatever reads
    them.
    #>
    param([Parameter(Mandatory = $true)][string]$Path)

    $at = [Math]::Max($Path.LastIndexOf('\'), $Path.LastIndexOf('/'))
    if ($at -lt 0) { return '' }
    return $Path.Substring(0, $at)
}

function Select-StudioCompilerLibraries {
    <#
    .SYNOPSIS
    The paths among $Artifacts that a C# compile accounts for.
    .DESCRIPTION
    A .cmdline or .rsp is a compiler's own file wherever it lands. A .dll on its own is
    not, and treating it as one is what failed this job on every run since it was added:
    the installer unpacks llama.cpp's checksum-verified prebuilt release into a staging
    directory under TEMP, which lands ~25 DLLs there with no compiler within reach. The
    shape under test is the one that was blocked in the field,

        powershell.exe -> csc.exe -> %TEMP%\<random>.dll

    and an unpacked archive is not it.

    So a DLL counts only when a compile is evidenced in the SAME directory. CodeDom, which
    is what Add-Type uses and what Bitdefender flagged, writes the response file, the
    generated source and the captured streams into the per-invocation directory it puts the
    assembly in, so the pairing holds for the shape this exists to catch. The workflow's
    positive control compiles a real type and REQUIRES this to fire, so a narrowing that
    went too far fails there rather than passing quietly.
    #>
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$Artifacts)

    $compileDirs = New-Object 'System.Collections.Generic.HashSet[string]' (
        [StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $Artifacts) {
        if ($path -match $script:CompilerSiblingPattern) {
            $null = $compileDirs.Add((Get-StudioParentPath -Path $path))
        }
    }

    $libraries = @()
    foreach ($path in $Artifacts) {
        if ($path -match $script:CompilerFilePattern) {
            $libraries += $path
        } elseif ($path -match '\.dll$' -and
                  $compileDirs.Contains((Get-StudioParentPath -Path $path))) {
            $libraries += $path
        }
    }
    # Returned plain, not comma-wrapped like the functions above: the caller normalises
    # with @(), and wrapping an empty array there yields a one-element array holding an
    # empty array, which reads downstream as one unnamed temporary library.
    return $libraries
}

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

function Get-StudioEventField {
    <#
    .SYNOPSIS
    One named EventData field of a 4688 record, from the record's own XML.
    .DESCRIPTION
    Read by name rather than by position, so a schema that gains a field still means the
    same thing. The rendered message is never consulted: it carries the command line too,
    so matching that would score `cmd.exe /c echo csc.exe` as a compiler.
    #>
    param(
        [Parameter(Mandatory = $true)]$Event,
        [Parameter(Mandatory = $true)][string]$Name
    )

    try {
        $xml = [xml]$Event.ToXml()
        foreach ($field in $xml.Event.EventData.Data) {
            if ($field.Name -eq $Name) { return [string]$field.'#text' }
        }
    } catch { }
    return ''
}

function Get-StudioProcessImageName {
    <#
    .SYNOPSIS
    The image a 4688 record says was created.
    #>
    param([Parameter(Mandatory = $true)]$Event)

    return Get-StudioEventField -Event $Event -Name 'NewProcessName'
}

function Test-StudioCompilerImage {
    <#
    .SYNOPSIS
    True when $Image is one of the compiler binaries, matched on the whole leaf name.
    #>
    param([string]$Image)

    if ([string]::IsNullOrEmpty($Image)) { return $false }
    # Split explicitly, not via [System.IO.Path]::GetFileName, which splits on the HOST's
    # separators: under the Linux pwsh where this is tested a backslash is an ordinary
    # character and the whole path came back as the leaf. The records are always Windows
    # paths whatever reads them.
    $leaf = ($Image -split '[\\/]')[-1]
    foreach ($name in $script:CompilerNames) {
        if ($leaf -eq $name) { return $true }
    }
    return $false
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
        if (-not (Test-StudioCompilerImage -Image $image)) { continue }
        # A compiler started BY a compiler is a step of a compile that is already being
        # scored, not a new one. csc.exe shells out to cvtres.exe to build its resource
        # blob, and counting that as a second hit says the action compiled twice.
        #
        # It also decides the cross-step bleed the $prior subtraction could not, because
        # the Security log is written with latency: the positive control's csc.exe started
        # before the installer's window, its cvtres.exe child landed inside, and neither
        # was in the log yet when the baseline was taken. The child is the only part that
        # was ever in range.
        #
        # Detection is unchanged for a compile the action really starts, because its ROOT
        # compiler is spawned by the installer's shell, not by another compiler, and the
        # window opens before the action does. What this drops is only ever the second
        # process of a chain whose first was already seen or was never in range at all.
        if (Test-StudioCompilerImage -Image (Get-StudioEventField -Event $record -Name 'ParentProcessName')) {
            continue
        }
        $rendered = ''
        try { $rendered = [string]$record.Message } catch { }
        $hits += ("{0:o} {1} :: {2}" -f $record.TimeCreated, $image,
            ($rendered -replace '\s+', ' '))
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
        #
        # Padded a second each way, then filtered exactly below: the hashtable's bounds do
        # not hold to the precision they are given. A csc.exe at 17:51:57.107 came back from
        # a window whose floor was 17:51:57.58, and failed a step that had not printed its
        # first line until 17:51:58.58. TimeCreated is the instant the claim is about. The
        # far pad covers the other end, where rounding could drop a real compile.
        $events = Get-WinEvent -FilterHashtable @{
            LogName   = 'Security'
            Id        = 4688
            StartTime = $Since.AddSeconds(-1)
            EndTime   = $Until.AddSeconds(1)
        } -ErrorAction Stop
        $events = @($events | Where-Object {
            $_.TimeCreated -ge $Since -and $_.TimeCreated -le $Until
        })
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
    $beforeScan = Get-StudioTempArtifacts
    $before = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]]$beforeScan.Files, [StringComparer]::OrdinalIgnoreCase)
    # A second back, so a process created in the same tick as the timestamp survives
    # Get-WinEvent's strictly-later comparison. That reaches one second into the tail
    # of the sweep above, so a compiler started in that second is counted: a deliberate
    # trade towards a loud false alarm rather than a dropped real compile.
    $since = (Get-Date).AddSeconds(-1)
    # That second reaches backwards, so it can reach into whatever ran BEFORE this action.
    # It did: the positive control compiles a type one step earlier, and its
    # csc.exe -> cvtres.exe landed inside the installer measurement's lookback and was
    # reported as "the installer spawned 1 compiler process(es)".
    #
    # Recorded and subtracted below, rather than moving the floor forward: a hit already in
    # the window before the action starts cannot be the action's, while moving the floor to
    # "now" would give up the same-tick protection the second is there to provide.
    $prior = @(Get-StudioCompilerEvents -Since $since -Until (Get-Date))
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
    # Each hit string carries its own round-trip timestamp, image and message, so it
    # identifies the record. Anything that was already there before the action ran is
    # dropped by identity.
    $compilers = @(
        Get-StudioCompilerEvents -Since $since -Until $until |
            Where-Object { $prior -notcontains $_ }
    )

    $afterScan = Get-StudioTempArtifacts
    $after = $afterScan.Files

    # A directory that could not be read in EITHER sweep is a hole in the comparison, not an
    # empty directory. $left below is "in the final listing and not in the baseline", so a
    # directory unread at baseline and readable afterwards hands every file that was already
    # sitting in it to $left, and the action is reported as having compiled something it did
    # not. Unread afterwards hides the opposite: a real artifact that never reaches $left.
    #
    # So paths under an unread directory are not evidence either way and are withheld from
    # $left, and which directories those were is recorded in the evidence rather than
    # dropped silently.
    $unread = @($beforeScan.Unread + $afterScan.Unread | Sort-Object -Unique)
    $watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]]@($watch | ForEach-Object { $_.Root }), [StringComparer]::OrdinalIgnoreCase)
    if ($unread.Count -gt 0) {
        foreach ($dir in $unread) {
            # Withholding is only safe while the watcher is covering that root: it reports
            # creations live, so a compile inside an unread directory still lands in
            # $transient. With no watcher on the root the listing is the only evidence there
            # is, and withholding part of it would report a hole as a clean result. That is
            # the same call as the traversal cap in Get-StudioTempSubtree, for the same
            # reason, so it is the same answer: declare the measurement void.
            $covered = @($watchedRoots | Where-Object { Test-StudioPathUnder -Path $dir -Directory $_ })
            if ($covered.Count -eq 0) {
                throw ("the temp sweep could not read $dir, and no file watcher is attached " +
                       "to the root containing it. The listing is the only evidence here and " +
                       "it is incomplete, so this run cannot say whether a compiler ran.")
            }
        }
    }

    $left = @(
        $after | Where-Object {
            $path = $_
            if ($before.Contains($path)) { return $false }
            foreach ($dir in $unread) {
                if (Test-StudioPathUnder -Path $path -Directory $dir) { return $false }
            }
            return $true
        }
    )
    # Only the names the compiler writes, because the watcher reports every creation
    # under temp and most of them are nobody's business.
    $transient = @($live | Where-Object { $_ -match $script:ArtifactPattern })
    $union = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $newArtifacts = @()
    foreach ($path in ($left + $transient)) {
        if ($union.Add($path)) { $newArtifacts += $path }
    }
    $newLibraries = @(Select-StudioCompilerLibraries -Artifacts ([string[]]$newArtifacts))

    $stem = Join-Path $EvidenceRoot $Name
    $compilers | Out-File -FilePath "$stem-compilers.txt" -Encoding utf8
    $newArtifacts | Out-File -FilePath "$stem-temp-artifacts.txt" -Encoding utf8
    # Written even when empty, so "the sweep read everything" is a statement the evidence
    # makes rather than the absence of a file, which is also what a crash looks like.
    $unread | Out-File -FilePath "$stem-unread-dirs.txt" -Encoding utf8
    if ($failure) {
        $failure | Out-String | Out-File -FilePath "$stem-error.txt" -Encoding utf8
    }

    if ($failure) { throw $failure }

    return @{
        Compilers     = $compilers
        TempLibraries = $newLibraries
        TempArtifacts = $newArtifacts
        UnreadDirs    = $unread
    }
}
