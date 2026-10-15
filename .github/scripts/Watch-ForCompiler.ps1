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

function Test-StudioPathIsGone {
    <#
    .SYNOPSIS
    Did this enumeration failure mean the directory does not exist, as opposed to could not
    be read?

    .DESCRIPTION
    The distinction decides whether a directory is dropped or recorded as a gap, so it has to
    come from the error itself. Test-Path cannot answer it. Measured under pwsh with
    $ErrorActionPreference = 'Stop':

        missing directory  ->  ItemNotFoundException, Test-Path returns $false
        denied directory   ->  Test-Path THROWS "Access to the path ... is denied"

    So probing with Test-Path both mis-answers the ACL case and can raise from inside the
    catch that was meant to contain the failure.

    Fails safe. Anything not positively identified as a missing path is reported as still
    present, which records the directory as unread: over-reporting a gap costs a withheld
    path or a voided run, while under-reporting one silently drops a directory out of the
    comparison, which is the defect this whole file exists to prevent.
    #>
    param([Parameter(Mandatory = $true)]$ErrorRecord)
    $exception = $ErrorRecord.Exception
    while ($null -ne $exception) {
        if ($exception -is [System.Management.Automation.ItemNotFoundException] -or
            $exception -is [System.IO.DirectoryNotFoundException] -or
            $exception -is [System.IO.FileNotFoundException]) {
            return $true
        }
        $exception = $exception.InnerException
    }
    return $false
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
    # Generic lists: += on arrays is quadratic.
    $found = New-Object 'System.Collections.Generic.List[string]'
    $unread = New-Object 'System.Collections.Generic.List[string]'
    $pending = New-Object 'System.Collections.Generic.List[string]'
    $pending.Add($Root)
    $visited = 0
    while ($pending.Count -gt 0) {
        $visited++
        if ($visited -gt 200000) {
            # Throw, not break: a truncated snapshot would look clean.
            throw ("the temp scan of $Root passed $visited directories without finishing. " +
                   "A partial snapshot would be read as a complete one, so this run cannot " +
                   "say whether a compiler ran.")
        }
        $dir = $pending[$pending.Count - 1]
        $pending.RemoveAt($pending.Count - 1)
        $entries = @()
        try { $entries = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop) }
        catch {
            # Record unread dirs so the before/after diff can withhold them. Classify from the error,
            # not Test-Path, which throws on ACL-denied dirs.
            if (Test-StudioPathIsGone -ErrorRecord $_) { continue }
            try { $entries = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop) }
            catch {
                if (-not (Test-StudioPathIsGone -ErrorRecord $_)) { $unread.Add($dir) }
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
        # Recursive: PowerShell compiles into a per-invocation subdirectory.
        $scan = Get-StudioTempSubtree -Root $root -Patterns $patterns
        $found.AddRange([string[]]$scan.Files)
        $unread.AddRange([string[]]$scan.Unread)
    }
    # Cast explicitly: an empty array unrolls to null, which the caller's HashSet rejects.
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

    The directory itself counts, and that is load-bearing rather than tidiness. A temp ROOT
    can be the thing that could not be enumerated, and the root is also what the watcher
    attaches to, so the coverage check asks whether an unread directory is at or under a
    watched root and gets back the root itself. Descendants-only there means a root that
    failed to enumerate twice, on a machine where the watcher did attach to it, is declared
    uncovered and the run throws - which is the temp-scan failure this whole change exists to
    contain, put back one layer up.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Directory
    )
    $trimmed = $Directory.TrimEnd('\', '/')
    if ($Path.TrimEnd('\', '/').Equals($trimmed, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    foreach ($separator in @('\', '/')) {
        if ($Path.StartsWith($trimmed + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

$script:ArtifactPattern = '\.(dll|cmdline|rsp|cs|err|out)$'
$script:CompilerFilePattern = '\.(cmdline|rsp)$'
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
    # Not comma-wrapped: the caller normalises with @(), and wrapping would nest an empty array.
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
            # The default 8 KB buffer overflows and silently drops events.
            $watcher.InternalBufferSize = 65536
            $identifier = "StudioTempWatch-" + [guid]::NewGuid().ToString('N')
            $null = Register-ObjectEvent -InputObject $watcher -EventName Created `
                -SourceIdentifier $identifier
            # Subscribe to Error so an overflowed watcher is not counted as covering its root.
            $errorIdentifier = $identifier + "-error"
            $null = Register-ObjectEvent -InputObject $watcher -EventName Error `
                -SourceIdentifier $errorIdentifier
            $watcher.EnableRaisingEvents = $true
            $handles += [pscustomobject]@{
                Watcher               = $watcher
                SourceIdentifier      = $identifier
                ErrorSourceIdentifier = $errorIdentifier
                Root                  = $root
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

    # Delivery is async; settle before draining or the last events are lost.
    Start-Sleep -Milliseconds 750

    $seen = @()
    $failedRoots = @()
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
        if ($entry.PSObject.Properties['ErrorSourceIdentifier']) {
            try {
                $errors = @(Get-Event -SourceIdentifier $entry.ErrorSourceIdentifier `
                        -ErrorAction SilentlyContinue)
                if ($errors.Count -gt 0) { $failedRoots += $entry.Root }
                foreach ($record in $errors) {
                    Remove-Event -EventIdentifier $record.EventIdentifier `
                        -ErrorAction SilentlyContinue
                }
            } catch { }
            Unregister-Event -SourceIdentifier $entry.ErrorSourceIdentifier `
                -ErrorAction SilentlyContinue
        }
        Unregister-Event -SourceIdentifier $entry.SourceIdentifier -ErrorAction SilentlyContinue
        try { $entry.Watcher.Dispose() } catch { }
    }
    return [pscustomobject]@{
        Paths       = [string[]]$seen
        FailedRoots = [string[]]$failedRoots
    }
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
    # Split explicitly: GetFileName uses the host's separators, wrong under Linux pwsh.
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
        # Skip compilers spawned by compilers (csc -> cvtres): part of an already-counted compile.
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
        # Pad a second each way, then filter exactly: the hashtable bounds are imprecise.
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
        # Get-WinEvent throws both on no matches and on an unreadable log; probe the log to tell them apart.
        # Bind $_ before the probe, whose catch rebinds it.
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

    # Baseline first, then open the window, so compiles during the sweep are not counted.
    $beforeScan = Get-StudioTempArtifacts
    $before = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]]$beforeScan.Files, [StringComparer]::OrdinalIgnoreCase)
    # Back one second so a same-tick process survives Get-WinEvent's strictly-later comparison.
    $since = (Get-Date).AddSeconds(-1)
    # That lookback can catch the previous step's compile, so record it and subtract below.
    $prior = @(Get-StudioCompilerEvents -Since $since -Until (Get-Date))
    $watch = Start-StudioTempWatch

    $failure = $null
    $live = @()
    $watchFailedRoots = @()
    try {
        # Out-Host so action output does not pollute the function's return value.
        & $Action | Out-Host
    } catch {
        $failure = $_
    } finally {
        $drained = Stop-StudioTempWatch -Handle $watch
        $live = @($drained.Paths)
        $watchFailedRoots = @($drained.FailedRoots)
    }

    # Closed before the temp sweep, which can take seconds.
    $until = Get-Date
    $compilers = @(
        Get-StudioCompilerEvents -Since $since -Until $until |
            Where-Object { $prior -notcontains $_ }
    )

    $afterScan = Get-StudioTempArtifacts
    $after = $afterScan.Files

    # Paths under a dir unread in either sweep are withheld from $left: they are not evidence either way.
    $unread = @($beforeScan.Unread + $afterScan.Unread | Sort-Object -Unique)
    # Only roots whose watcher stayed healthy count as covered.
    $watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' (
        [string[]]@(
            $watch | ForEach-Object { $_.Root } | Where-Object {
                $watchFailedRoots -notcontains $_
            }
        ), [StringComparer]::OrdinalIgnoreCase)
    # Unread dirs on unwatched roots void the run, but only at the end so an action failure wins.
    $uncovered = @()
    foreach ($dir in $unread) {
        $covered = @($watchedRoots | Where-Object { Test-StudioPathUnder -Path $dir -Directory $_ })
        if ($covered.Count -eq 0) { $uncovered += $dir }
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
    # Written even when empty, so absence of the file means a crash.
    $unread | Out-File -FilePath "$stem-unread-dirs.txt" -Encoding utf8
    $watchFailedRoots | Out-File -FilePath "$stem-watch-failures.txt" -Encoding utf8
    if ($failure) {
        $failure | Out-String | Out-File -FilePath "$stem-error.txt" -Encoding utf8
    }

    if ($failure) { throw $failure }

    $incomplete = @()
    if ($watchFailedRoots.Count -gt 0) {
        # A failed watcher alone voids the run: CodeDom deletes its temp dir, so live events are the only evidence.
        $incomplete += ("the file watcher on " + ($watchFailedRoots -join ', ') + " raised an " +
                        "error, so creations under it may have been dropped. A compile whose " +
                        "intermediates were deleted before the final sweep is visible only in " +
                        "those events")
    }
    if ($uncovered.Count -gt 0) {
        $incomplete += ("the temp sweep could not read " + ($uncovered -join ', ') + ", and no " +
                        "file watcher is attached to the root containing it, so the listing is " +
                        "the only evidence here and it is incomplete")
    }
    if ($incomplete.Count -gt 0) {
        throw (($incomplete -join '; ') + ". This run cannot say whether a compiler ran.")
    }

    return @{
        Compilers     = $compilers
        TempLibraries = $newLibraries
        TempArtifacts = $newArtifacts
        UnreadDirs    = $unread
        WatchFailures = $watchFailedRoots
    }
}
