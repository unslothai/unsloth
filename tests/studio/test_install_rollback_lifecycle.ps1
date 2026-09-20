#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.ps1's venv rollback helpers. The functions are AST-extracted
# so the top-level installer is never executed.

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

# The six functions under test. Extracting only these is what broke when #9501 added a
# sibling helper they call: the list is maintained by hand, so it goes stale the moment
# a helper grows a dependency, and the failure surfaces as "not recognized" inside an
# unrelated case rather than as a missing extraction. Extract the transitive closure
# instead, so a new callee is picked up without anyone remembering to add it here.
$subjectNames = @(
    "Start-StudioVenvRollback",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Restore-StudioVenvRollback",
    "Complete-StudioVenvRollback",
    "Restore-StudioUvCacheMarker",
    # Not called by the rollback helpers: it is the cross-volume cache notice that
    # calls it, and the rollback space warning is now gated on what it decides.
    "Test-StudioSameVolume"
)

$definitions = @{}
foreach ($node in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)) {
    if ($definitions.ContainsKey($node.Name)) {
        throw "install.ps1 defines $($node.Name) more than once; the extraction cannot tell which one is under test"
    }
    $definitions[$node.Name] = $node
}

foreach ($name in $subjectNames) {
    if (-not $definitions.ContainsKey($name)) {
        throw "expected $name in install.ps1, found no definition (renamed or removed?)"
    }
}

# Sinks this file stubs below. The walk stops at them, so their own dependencies (ANSI
# colouring and the like) are not dragged in for functions that are replaced anyway.
$stubbedNames = @("substep", "Write-StudioLine")

# Breadth-first over the call graph, following only names install.ps1 itself defines.
# Anything else is a real cmdlet or one of the stubs below and must not be extracted.
$extracted = [System.Collections.Generic.List[string]]::new()
$seen = @{}
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($name in $subjectNames) { $queue.Enqueue($name) }
while ($queue.Count -gt 0) {
    $name = $queue.Dequeue()
    if ($seen.ContainsKey($name)) { continue }
    $seen[$name] = $true
    $extracted.Add($name)
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if ($callee -and $definitions.ContainsKey($callee) -and
            $stubbedNames -notcontains $callee -and -not $seen.ContainsKey($callee)) {
            $queue.Enqueue($callee)
        }
    }
}

foreach ($name in $extracted) { Invoke-Expression $definitions[$name].Extent.Text }

$pulledIn = @($extracted | Where-Object { $subjectNames -notcontains $_ })
if ($pulledIn.Count -gt 0) { Write-Host "Extracted dependencies: $($pulledIn -join ', ')" }

function substep { param([string]$Message, [string]$Color) }
# The rollback helpers report through install.ps1's UTF-8 stdout sink on their warn
# and error branches. None of the cases below take one today, but this file runs
# under "Stop", so an undefined sink would abort the suite rather than fail a check.
function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }

# Fail here, with the caller named, rather than 60 lines further down as "the term X is
# not recognized" inside whichever case happened to reach it first. The closure above
# makes an install.ps1-defined callee unreachable by construction; this catches the rest,
# including a helper that calls a sink this file forgot to stub.
$windowsOnlyCommands = @("Get-CimInstance")
$unresolved = [System.Collections.Generic.List[string]]::new()
foreach ($name in $extracted) {
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if (-not $callee) { continue }
        # CimCmdlets ships only on Windows, and Get-StudioMountedVolume guards its one call on
        # the platform before making it, so a host without the module never reaches the name.
        # Keyed on the command being absent rather than on the platform: $env:OS can say Windows
        # on a host whose PowerShell has no CimCmdlets, and on a real Windows runner the name
        # resolves and is checked like any other.
        if ($windowsOnlyCommands -contains $callee -and
            -not (Get-Command -Name $callee -ErrorAction SilentlyContinue)) { continue }
        if (-not (Get-Command -Name $callee -ErrorAction SilentlyContinue)) {
            $unresolved.Add("$callee (called by $name)")
        }
    }
}
if ($unresolved.Count -gt 0) {
    throw "extracted helpers call commands that do not resolve: $(($unresolved | Sort-Object -Unique) -join '; ')"
}

$failures = 0
function Check($name, $condition) {
    if ($condition) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Reset-RollbackState($target) {
    $script:StudioVenvRollbackDir = $null
    $script:StudioVenvRollbackTarget = $target
    $script:StudioVenvRollbackActive = $false
    # As Install-UnslothStudio does at entry: the commit flag is per install, and the
    # restore consults it, so a previous section's commit would suppress this one.
    $script:StudioInstallCommitted = $false
}

$StudioHome = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth-rollback-$([guid]::NewGuid().ToString('N'))"
$VenvDir = Join-Path $StudioHome "unsloth_studio"
[System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null

try {
    Write-Host "Successful replacement"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    Start-StudioVenvRollback -ExistingDir $VenvDir
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "new")
    Complete-StudioVenvRollback
    Check "new environment remains" ((Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "new")
    Check "current rollback is removed" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Stale cleanup"
    $stale = Join-Path $StudioHome "unsloth_studio.rollback.20000101000000.2147483647"
    $active = Join-Path $StudioHome "unsloth_studio.rollback.20000101000001.$PID"
    $unrecognized = Join-Path $StudioHome "unsloth_studio.rollback.user-data"
    [System.IO.Directory]::CreateDirectory($stale) | Out-Null
    [System.IO.Directory]::CreateDirectory($active) | Out-Null
    [System.IO.Directory]::CreateDirectory($unrecognized) | Out-Null
    Remove-StaleStudioVenvRollbacks
    Check "dead-owner rollback is removed" (-not (Test-Path -LiteralPath $stale))
    Check "live-owner rollback is preserved" (Test-Path -LiteralPath $active)
    Check "unrecognized rollback name is preserved" (Test-Path -LiteralPath $unrecognized)
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $active -Recurse -Force
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $unrecognized -Recurse -Force

    Write-Host "Failure restoration"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old-again")
    Reset-RollbackState $VenvDir
    $committed = $false
    try {
        try {
            Start-StudioVenvRollback -ExistingDir $VenvDir
            [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
            [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "partial")
            throw "simulated install failure"
        } finally {
            if (-not $committed) { Restore-StudioVenvRollback }
        }
    } catch {
        if ($_.Exception.Message -ne "simulated install failure") { throw }
    }
    Check "finally restores the previous environment" (
        (Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "old-again"
    )
    Check "failure restoration consumes the rollback" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Locked-file retry"
    $retryDir = Join-Path $StudioHome "retry"
    [System.IO.Directory]::CreateDirectory($retryDir) | Out-Null
    $script:removeAttempts = 0
    function Remove-Item {
        param(
            [string]$LiteralPath,
            [switch]$Recurse,
            [switch]$Force,
            [object]$ErrorAction
        )
        $script:removeAttempts++
        if ($script:removeAttempts -lt 3) { throw "simulated lock" }
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
    }
    try {
        $removed = Remove-StudioVenvTreeWithRetry -Path $retryDir -Label "test rollback"
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Remove-Item -Force
    }
    Check "locked rollback deletion retries" ($removed -and $script:removeAttempts -eq 3)

    Write-Host "--no-rollback discards the old environment instead of keeping a copy (#11313)"
    # The rename still has to happen -- uv creates only into a path that is absent or empty -- so
    # what the flag changes is what survives it, not whether there is one.
    foreach ($case in @(
            @{ Label = "default"; Flag = $false; ExpectCopy = $true },
            @{ Label = "--no-rollback"; Flag = $true; ExpectCopy = $false })) {
        [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
        Reset-RollbackState $VenvDir
        $script:StudioNoRollback = $case.Flag
        Start-StudioVenvRollback -ExistingDir $VenvDir
        $copies = @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)
        Check "$($case.Label): rollback copy kept = $($case.ExpectCopy)" (($copies.Count -gt 0) -eq $case.ExpectCopy)
        Check "$($case.Label): the old environment was moved aside" (-not (Test-Path -LiteralPath $VenvDir))
        if (-not $case.ExpectCopy) {
            # Cleared before the delete, exactly as the commit path does: an interrupt must not be
            # handed a backup that is already half gone.
            Check "--no-rollback clears the restore state" (
                (-not $script:StudioVenvRollbackActive) -and ($null -eq $script:StudioVenvRollbackDir))
            # And the restore must then be a no-op rather than a failure.
            $restoreThrew = $false
            try { Restore-StudioVenvRollback } catch { $restoreThrew = $true }
            Check "--no-rollback leaves nothing for the restore to do" (
                (-not $restoreThrew) -and (-not (Test-Path -LiteralPath $VenvDir)))
        }
        foreach ($c in $copies) { Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    }
    $script:StudioNoRollback = $false

    Write-Host "--no-rollback tells the ARM64 migration the tree is gone (#11313)"
    # The migration branch decides what to do from the rollback state. Before this was tracked
    # separately, --no-rollback left it merely "inactive", which that branch reads as "not moved
    # aside yet", so it called Start-StudioVenvRollback on a directory --no-rollback had already
    # deleted. Move-Item threw and the install exited through Exit-InstallFailure, having already
    # destroyed the environment. The flag has to distinguish discarded from never-started.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioVenvRollbackDiscarded = $false
    $script:StudioNoRollback = $true
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "--no-rollback records that the tree was discarded" ($script:StudioVenvRollbackDiscarded)
    Check "and the tree really is gone" (-not (Test-Path -LiteralPath $VenvDir))
    # Replay the migration branch's own decision with that state. It must take neither the
    # "already moved aside" arm nor the "move it now" arm.
    $wouldMove = (-not $script:StudioVenvRollbackDiscarded) -and (-not $script:StudioVenvRollbackActive)
    Check "the migration would not try to move a tree that is gone" (-not $wouldMove)
    $script:StudioNoRollback = $false
    $script:StudioVenvRollbackDiscarded = $false

    # The ordinary path must still report "not discarded", or the migration would skip a tree
    # that is genuinely sitting there waiting to be kept.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "the ordinary path does not claim the tree was discarded" (-not $script:StudioVenvRollbackDiscarded)
    Check "and it is still there to be kept" ($null -ne $script:StudioVenvRollbackDir -and (Test-Path -LiteralPath $script:StudioVenvRollbackDir))
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "the volume helpers degrade to the drive root where mount points cannot be read"
    # A Windows volume mounted at a directory is not its drive letter, so both helpers ask
    # Win32_Volume first. That view exists only on Windows, and this host is not Windows, so what
    # is checkable here is the fallback: the probe answers "cannot tell" rather than throwing, and
    # the drive-root logic underneath still produces the answers the rest of this file relies on.
    # The mount-point case itself is pinned by text in tests/python/test_cross_platform_parity.py
    # and is not reproduced on any host available here.
    $probed = $null
    $probeThrew = $false
    try { $probed = Get-StudioMountedVolume -Path $StudioHome } catch { $probeThrew = $true }
    Check "the mount-point probe never throws" (-not $probeThrew)
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        # On Windows it answers for real, and may legitimately answer nothing when CIM is not
        # available to this account. What it must never do is throw or hand back something the
        # callers cannot read: this file also runs on windows-latest, where the earlier
        # "answers nothing" form asserted a Linux-only outcome and failed.
        Check "on Windows the probe answers a volume or nothing, never something unusable" (
            $null -eq $probed -or $null -ne $probed.Name)
    } else {
        Check "off Windows the probe answers nothing" ($null -eq $probed)
    }
    Check "and an empty path is not an error" ($null -eq (Get-StudioMountedVolume -Path ""))
    # The WMI answer is advisory and runs before the venv exists, so it must be bounded and taken
    # once. A degraded repository is what the bound is for; what is checkable here is that the
    # call returns rather than hanging, never throws, and is not repeated.
    $script:StudioVolumeList = $null
    $listThrew = $false
    $t0 = [System.Diagnostics.Stopwatch]::StartNew()
    try { $null = Get-StudioVolumeList } catch { $listThrew = $true }
    $firstMs = $t0.ElapsedMilliseconds
    Check "the volume query returns rather than hanging" (-not $listThrew -and $firstMs -lt 60000)
    $t0.Restart()
    $null = Get-StudioVolumeList
    Check "and is answered from the cache the second time" ($t0.ElapsedMilliseconds -lt $firstMs + 50)
    Check "a query that answered nothing still caches an answer" (
        $null -ne $script:StudioVolumeList)
    # The cache holds identity, which does not move during an install. It must NOT answer the
    # free-space question: the failure handler asks after setup has eaten the disk, and a number
    # snapshotted before the environment was built reports room that is gone.
    $script:StudioVolumeList = @([pscustomobject]@{ Name = "sentinel"; DeviceID = "sentinel"; FreeSpace = 1 })
    $null = Get-StudioVolumeList
    Check "the cached list is reused when only identity is wanted" (
        @(Get-StudioVolumeList)[0].DeviceID -eq "sentinel")
    $null = Get-StudioVolumeList -Fresh
    Check "asking for a fresh answer replaces the cached one" (
        @($script:StudioVolumeList | Where-Object { $_.DeviceID -eq "sentinel" }).Count -eq 0)
    $script:StudioVolumeList = $null
    # A link must be measured as the volume it points at, not the one it lives on. On one
    # filesystem the two numbers agree either way, so what this proves is that the resolution
    # happens at all and costs nothing: a helper that threw, or answered $null through a link,
    # would take the warning down with it. The cross-volume case needs a second volume and a
    # junction, neither of which exists on any host this suite runs on.
    $linkTarget = Join-Path $StudioHome "link-target"
    [System.IO.Directory]::CreateDirectory($linkTarget) | Out-Null
    $linkPath = Join-Path $StudioHome "link"
    $linkMade = $true
    try { New-Item -ItemType SymbolicLink -Path $linkPath -Target $linkTarget -ErrorAction Stop | Out-Null }
    catch { $linkMade = $false }
    if ($linkMade) {
        # Within a tolerance, not equal: free space on a live filesystem moves between two
        # calls, and an exact comparison fails for that reason rather than for this one.
        $viaLink = Get-StudioFreeSpaceBytes -Path $linkPath
        $viaTarget = Get-StudioFreeSpaceBytes -Path $linkTarget
        Check "free space through a link is the target's volume, not some other one" (
            $null -ne $viaLink -and $null -ne $viaTarget -and $viaLink -gt 0 -and
            [math]::Abs($viaLink - $viaTarget) -lt ([math]::Max($viaLink, $viaTarget) * 0.01))
        Check "a link and its target are one volume" (
            Test-StudioSameVolume -PathA $linkPath -PathB $linkTarget)
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $linkPath -Force -ErrorAction SilentlyContinue
    } else {
        # Windows needs Developer Mode or elevation to create one; not a failure of the code.
        Write-Host "  SKIP  this host will not create a symbolic link"
    }
    $freeHere = Get-StudioFreeSpaceBytes -Path $StudioHome
    Check "free space still comes back from the fallback" ($null -ne $freeHere -and $freeHere -gt 0)
    Check "a path and its own child are still one volume" (
        Test-StudioSameVolume -PathA $StudioHome -PathB (Join-Path $StudioHome "child"))

    Write-Host "picking the volume that holds a path, including a directory mount point"
    # The matching, with the volume list supplied rather than read from CIM, so the mount-point
    # cases run on a host that has no mount points. Every path goes through GetFullPath first,
    # exactly as the function does: on Windows a rooted path with no drive picks up the current
    # drive, so fake volume names built from a bare separator match nothing and all four cases
    # fail there. Deriving both sides from the same call keeps this true on either platform.
    $sep = [System.IO.Path]::DirectorySeparatorChar
    $mountPath = [System.IO.Path]::GetFullPath("${sep}studio")
    $otherPath = [System.IO.Path]::GetFullPath("${sep}studiofoo")
    $elsePath  = [System.IO.Path]::GetFullPath("${sep}elsewhere${sep}x")
    $rootVol  = [pscustomobject]@{ Name = [System.IO.Path]::GetPathRoot($mountPath); DeviceID = "root";  FreeSpace = 1GB }
    $mountVol = [pscustomobject]@{ Name = "$mountPath$sep";                          DeviceID = "mount"; FreeSpace = 2GB }
    $otherVol = [pscustomobject]@{ Name = "$otherPath$sep";                          DeviceID = "other"; FreeSpace = 3GB }
    $vols = @($rootVol, $mountVol, $otherVol)
    Check "a path under a mount point picks the mounted volume" (
        (Select-StudioVolumeForPath -Path (Join-Path $mountPath "cache") -Volumes $vols).DeviceID -eq "mount")
    # The regression this section exists for: the mount point itself has no trailing separator.
    Check "the mount point itself picks the mounted volume, not the drive root" (
        (Select-StudioVolumeForPath -Path $mountPath -Volumes $vols).DeviceID -eq "mount")
    Check "a sibling whose name merely starts the same does not match it" (
        (Select-StudioVolumeForPath -Path $otherPath -Volumes $vols).DeviceID -eq "other")
    Check "a path under neither falls to the root volume" (
        (Select-StudioVolumeForPath -Path $elsePath -Volumes $vols).DeviceID -eq "root")
    Check "nothing to choose from is not an error" (
        $null -eq (Select-StudioVolumeForPath -Path $mountPath -Volumes @()))
    Check "an empty path chooses nothing" (
        $null -eq (Select-StudioVolumeForPath -Path "" -Volumes $vols))

    Write-Host "the free-space warning names both figures and the opt-out, and never aborts"
    # Stub the two measurements rather than filling a real disk.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    foreach ($case in @(
            @{ Label = "less free than the venv needs"; Free = 512MB; Expect = $true },
            @{ Label = "plenty of room"; Free = 100GB; Expect = $false },
            @{ Label = "unmeasurable free space"; Free = $null; Expect = $false })) {
        $script:warnLines = @()
        function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) $script:warnLines += $Message }
        function Get-StudioTreeSizeBytes { param([string]$Path) return 1GB }
        $script:caseFree = $case.Free
        function Get-StudioFreeSpaceBytes { param([string]$Path) return $script:caseFree }
        $threw = $false
        try { Write-StudioRollbackSpaceWarning -ExistingDir $VenvDir } catch { $threw = $true }
        $joined = ($script:warnLines -join "`n")
        Check "$($case.Label): never throws" (-not $threw)
        Check "$($case.Label): warning present = $($case.Expect)" (($joined -match 'needs about 1024 MB') -eq $case.Expect)
        if ($case.Expect) {
            Check "$($case.Label): names the free space too" ($joined -match '512 MB free')
            Check "$($case.Label): names the opt-out" ($joined -match 'UNSLOTH_INSTALL_NO_ROLLBACK=1')
        }
    }
    Write-Host "--no-rollback costs disk, never hardware (#11313)"
    # The Intel scan rescues an adapter WMI cannot classify by asking the PREVIOUS environment's
    # torch whether XPU works; the replacement has no torch yet. Discarding that tree without
    # taking the verdict first routes an Arc machine to CPU wheels for having opted out of a
    # rollback copy, which is a narrower device set than the same install without the flag.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.Directory]::CreateDirectory((Join-Path $VenvDir "Scripts")) | Out-Null
    # Stands in for the interpreter: the probe is bounded and reads stdout, so what it runs only
    # has to print True the way torch.xpu.is_available() would on an Arc machine.
    $fakePy = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Set-Content -LiteralPath $fakePy -Value "@echo True"
    } else {
        Set-Content -LiteralPath $fakePy -Value "#!/bin/sh`necho True"
        & chmod +x $fakePy
    }
    Reset-RollbackState $VenvDir
    $script:StudioPreservedXpuVerdict = $false
    $script:StudioNoRollback = $true
    $script:StudioRollbackCostsFullSize = $true
    $xpuThrew = $false
    try { Start-StudioVenvRollback -ExistingDir $VenvDir } catch { $xpuThrew = $true }
    Check "taking the XPU verdict never costs the discard" (-not $xpuThrew)
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        # A .exe that is really a batch file will not start on Windows, so only the POSIX arm
        # can drive the probe end to end; here the assertion is that it is attempted and safe.
        Write-Host "  SKIP  the stand-in interpreter cannot be executed on this platform"
    } else {
        Check "the XPU verdict survives the environment being discarded" (
            $script:StudioPreservedXpuVerdict)
    }
    Check "and the environment really was discarded" (-not (Test-Path -LiteralPath $VenvDir))
    $script:StudioPreservedXpuVerdict = $false
    $script:StudioNoRollback = $false
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "the warning and the discard message both tell the truth under --no-rollback"
    # Two things Start-StudioVenvRollback gets wrong if it is written without them, and install.sh
    # is gated identically: the warning's payload is the name of the opt-out, so printing it to
    # someone who already passed that flag advises an action they have taken; and a delete that
    # could not remove the tree frees none of the space the flag exists to free, so reporting it
    # as discarded promises the user something that is still on their disk.
    function Get-StudioTreeSizeBytes { param([string]$Path) return 1GB }
    function Get-StudioFreeSpaceBytes { param([string]$Path) return 512MB }
    $script:said = @()
    function substep { param([string]$Message, [string]$Color) $script:said += $Message }
    function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) $script:said += $Message }

    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioNoRollback = $true
    # Off-volume, so the only thing keeping the warning quiet here is the flag.
    $script:StudioRollbackCostsFullSize = $true
    Start-StudioVenvRollback -ExistingDir $VenvDir
    $joined = ($script:said -join "`n")
    Check "--no-rollback does not advise the flag it was already given" (
        $joined -notmatch 'needs about')
    Check "--no-rollback still reports the discard" ($joined -match 'discarded \(--no-rollback\)')

    # And with the cache on this volume the warning is wrong even without the flag: uv hardlinks
    # every wheel within one filesystem, so the old tree shares its blocks with the cache and
    # keeping it costs metadata, while summing each file's length still bills all of them.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioNoRollback = $false
    $script:StudioRollbackCostsFullSize = $false
    $script:said = @()
    Start-StudioVenvRollback -ExistingDir $VenvDir
    $joined = ($script:said -join "`n")
    Check "a co-located cache does not warn about space the rollback does not take" (
        $joined -notmatch 'needs about')
    Check "and the rollback copy is still kept" ($script:StudioVenvRollbackActive)
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    $script:StudioNoRollback = $true
    $script:StudioRollbackCostsFullSize = $true

    # A tree the retry helper could not remove. It shadows the extracted definition, so
    # Start-StudioVenvRollback resolves to this one at call time.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:said = @()
    function Remove-StudioVenvTreeWithRetry { param([string]$Path, [string]$Label) return $false }
    try {
        $discardThrew = $false
        try { Start-StudioVenvRollback -ExistingDir $VenvDir } catch { $discardThrew = $true }
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Remove-StudioVenvTreeWithRetry -Force
    }
    $joined = ($script:said -join "`n")
    Check "a discard that could not delete never aborts the install" (-not $discardThrew)
    Check "a discard that could not delete does not claim success" (
        $joined -notmatch 'discarded \(--no-rollback\)')
    Check "a discard that could not delete names the path left on disk" (
        $joined -match [regex]::Escape($StudioHome) -and $joined -match 'unsloth_studio\.rollback\.')
    $script:StudioNoRollback = $false
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    function substep { param([string]$Message, [string]$Color) }
    function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }
} finally {
    if (Test-Path -LiteralPath $StudioHome) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $StudioHome -Recurse -Force
    }
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
