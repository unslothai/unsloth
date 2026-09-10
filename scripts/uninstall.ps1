# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Unsloth Studio uninstaller for Windows PowerShell. Run -Help for details.
# Custom roots (UNSLOTH_STUDIO_HOME / STUDIO_HOME) come from share\studio.conf.
#
# Usage: run -Help. The web one-liner is in that help text and is not repeated here, since
# AMSI scans this file in full before any of it runs and nothing reads the header.

function Uninstall-UnslothStudio {
    $ErrorActionPreference = "Continue"

    # Reset at entry: a piped web run defines this function in the caller's session, so a
    # second run in the same window would otherwise inherit the first run's flags.
    $script:RemoveFailed = $false
    $script:StudioDbRemoved = $false
    # The default root was REFUSED and still holds studio.db. Its own flag, not RemoveFailed:
    # nothing failed, so "remove those paths by hand" is the wrong thing to say about it.
    $script:StudioDbKept = $false
    # ONE budget for the run: what _RemovePath waits out is wall clock and shared, so whatever
    # the first blocked path waits, the next does not. Without it an undeletable root costs the
    # full escalation at each of the 18 call sites. Classifying the error instead does not work:
    # a delete-pending file and an antivirus hold both report access denied.
    $script:RemoveWaitBudgetMs = 20000

    function _Usage {
        Write-Host @'
Unsloth Studio uninstaller (Windows PowerShell).

Usage:
  irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\scripts\uninstall.ps1

Stops running Unsloth Studio servers, then removes the install dir, launcher
data, CLI shim, desktop and Start Menu shortcuts, the user PATH entry and the
PathBackup registry key. In a default-mode install it also removes the shared
prebuilts that sit beside the install dir:
%USERPROFILE%\.unsloth\{llama.cpp,node,whisper.cpp,.cache}. The Hugging Face
cache is left in place, as is anything else you keep under %USERPROFILE%\.unsloth.

Options:
  -Help, -h, --help, -?, /?  Print this message and exit without removing anything.

Run with no arguments to uninstall. Unrecognized arguments never trigger
removal.

Environment:
  UNSLOTH_STUDIO_HOME  Also remove this custom install root. Set it to the value
                       used at install time.
  STUDIO_HOME          Alias for the above, ignored when both are set.
'@
    }

    # Reject unknown arguments before destructive work. throw, not exit, so an embedded run reports
    # failure without killing the caller's PowerShell session.
    foreach ($arg in $args) {
        if ($arg -in @('-h', '-help', '--help', '-?', '/?')) { _Usage; return }
        Write-Host "uninstall.ps1: unrecognized argument: $arg" -ForegroundColor Red
        Write-Host "Nothing was removed. Re-run with no arguments to uninstall, or -Help."
        throw "uninstall.ps1: unrecognized argument: $arg"
    }

    function _Step { param([string]$Msg) Write-Host $Msg }
    function _Substep { param([string]$Msg, [string]$Color = "Gray") Write-Host "  $Msg" -ForegroundColor $Color }

    # Remove a file/dir/symlink if present. Idempotent; retries since a just-killed
    # process can briefly hold a handle (Windows refuses the delete until released).
    function _RemovePath {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        # Escalating: torch inductor holds unattributable DATA handles under
        # TORCHINDUCTOR_CACHE_DIR for seconds after the server stops.
        $delays = @(250, 500, 1000, 2000, 4000, 4000, 4000, 4000)
        # 2100ms per path is free, exactly the flat 700ms x3 this replaced, so no path is retried
        # less than before: a lock only this path hits is not the shared wait. $false means both
        # are spent, so stop rather than retry with no pause in between.
        $freeLeft = 2100
        function _Wait {
            param([int]$Ms, [ref]$FreeLeft)
            $free = [Math]::Min($Ms, [Math]::Max(0, $FreeLeft.Value))
            $paid = $Ms - $free
            if ($paid -gt 0) {
                if ($script:RemoveWaitBudgetMs -le 0) {
                    if ($free -le 0) { return $false }
                    $paid = 0
                } else {
                    $paid = [Math]::Min($paid, $script:RemoveWaitBudgetMs)
                    $script:RemoveWaitBudgetMs -= $paid
                }
            }
            $FreeLeft.Value -= $free
            Start-Sleep -Milliseconds ($free + $paid)
            return $true
        }
        for ($attempt = 0; $attempt -le $delays.Count; $attempt++) {
            $lastTry = ($attempt -eq $delays.Count)
            try {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            } catch {
                # Read before $wait runs, so nothing can shadow the error record.
                $failure = $_.Exception.Message
                if (-not $lastTry -and (_Wait $delays[$attempt] ([ref]$freeLeft))) { continue }
                _Substep "could not remove: $Path ($failure)" "Yellow"
                # The closing summary must not promise the data is gone.
                $script:RemoveFailed = $true
                return
            }
            # Remove-Item -Recurse can report success yet leave a transiently-locked child (e.g.
            # unsloth.ico in Explorer's icon cache); verify + retry so "removed" is never a lie.
            if (-not (Test-Path -LiteralPath $Path)) {
                _Substep "removed: $Path" "Green"
                return
            }
            if (-not $lastTry -and (_Wait $delays[$attempt] ([ref]$freeLeft))) { continue }
            _Substep "still present (files held open): $Path" "Yellow"
            $script:RemoveFailed = $true
            return
        }
    }

    # An install lock, and only an install lock.
    #
    # prebuilt_core.install_lock creates these with os.open(O_CREAT | O_EXCL) and writes a pid,
    # so the lock is always a regular file. _RemovePath deletes recursively, which in a
    # user-chosen master root would take a whole tree that merely happens to carry one of these
    # fixed names, with none of the owner-marker proof the runtime children beside it require.
    # uninstall.sh's _remove_lock_file is the same rule.
    function _RemoveLockFile {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        # A reparse point at a lock name is not a lock either, and following one would delete
        # whatever it points at.
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and -not $item.PSIsContainer -and
            -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            _RemovePath $Path
        } else {
            _Substep "keeping non-file at an install-lock path: $Path" "Yellow"
        }
    }

    # LOCALAPPDATA / APPDATA are dropped in service and CI contexts, so fall back to the known
    # folder; $null only if that fails too, which callers treat as incomplete cleanup.
    function _AppDataRoot {
        param([string]$Var, [string]$Folder)
        if (-not [string]::IsNullOrWhiteSpace($Var)) { return $Var }
        $p = try { [Environment]::GetFolderPath($Folder) } catch { $null }
        if ([string]::IsNullOrWhiteSpace($p)) { return $null }
        return $p
    }

    # Remove an install root and record whether its studio.db really went with it. That file holds
    # the chat history (backend/storage/studio_db.py), not the provider API keys, which
    # providers_db.py keeps in the browser's localStorage only; an env-mode install keeps it in a
    # custom root a bare run cannot find. The check runs on the RESOLVED target: a relocated
    # install passes the before-check through the link, but the delete unlinks only the reparse
    # point. Verifying rather than chasing the link is deliberate: following a reparse point out of
    # the expected location to delete its target is what the deny list exists to stop.
    # A removal that got part way can take the sentinels and then fail on a locked child, leaving
    # a root the next run's gate would refuse. Put the marker back so a retry recognises it.
    function _RestoreOwnerMarker {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
        # Get-Item -Force, not Test-Path: the latter follows a dangling link and answers false,
        # after which WriteAllText follows the link and writes outside the root.
        $marker = Join-Path $Path ".unsloth-studio-owned"
        if (Get-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue) { return }
        try { [System.IO.File]::WriteAllText($marker, "") } catch { }
    }

    function _RemoveRootRecordingDb {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        # Anchor a relative reparse-point target to the link's own parent, or Join-Path
        # resolves it from the uninstaller's working directory and the db test reads false.
        # -LiteralPath takes no -Parent (other parameter set; it throws) and already returns it.
        $resolveTarget = {
            param($Item, $Fallback)
            if (-not $Item -or -not $Item.Target) { return $Fallback }
            $t = @($Item.Target)[0]
            if ([string]::IsNullOrWhiteSpace($t)) { return $Fallback }
            if (-not [System.IO.Path]::IsPathRooted($t)) {
                $t = Join-Path (Split-Path -LiteralPath $Item.FullName) $t
            }
            return $t
        }
        $real = $Path
        try {
            $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
            $real = & $resolveTarget $item $Path
        } catch { }
        # The db itself can be a reparse point out of the tree: the target survives the delete.
        $dbPath = Join-Path $real "studio.db"
        $hadDb = Test-Path -LiteralPath $dbPath -PathType Leaf
        if ($hadDb) {
            try {
                $dbItem = Get-Item -LiteralPath $dbPath -Force -ErrorAction SilentlyContinue
                $dbPath = & $resolveTarget $dbItem $dbPath
            } catch { }
        }
        _RemovePath $Path
        _RestoreOwnerMarker $Path
        if ($hadDb) {
            if (Test-Path -LiteralPath $dbPath -PathType Leaf) {
                $script:RemoveFailed = $true
            } else {
                $script:StudioDbRemoved = $true
            }
        }
    }

    # Reclaim ONLY what install.ps1 puts under "<LocalAppData>\Unsloth Studio\temp": directories
    # named ust-<pid>-<hex>, then the temp dir and its parent if left empty. Deliberately narrow:
    # this runs against every LocalAppData spelling, including one that can name a different user's
    # profile, so it must never recursively delete anything it did not create. Nothing here follows
    # a link (Directory.Delete with $false unlinks a reparse point without touching its target).
    function _RemoveStudioPrivateTempTrees {
        param(
            [string[]]$Paths,
            # The spelling this uninstall is actually for. Any OTHER spelling can be
            # a different user's profile, so it gets the stricter rule below.
            [string]$PrimaryPath
        )
        # Every directory this sweep kept, returned so the callers keep it too: the wholesale
        # data-directory removal runs over the same tree, and a live owner preserved here but
        # deleted there is not preserved at all.
        $preserved = @()
        foreach ($temp in @($Paths)) {
            if ([string]::IsNullOrWhiteSpace($temp)) { continue }
            if (-not (Test-Path -LiteralPath $temp -PathType Container)) { continue }
            $isPrimary = (-not [string]::IsNullOrWhiteSpace($PrimaryPath)) -and
                [string]::Equals($temp.TrimEnd('\','/'), $PrimaryPath.TrimEnd('\','/'), [System.StringComparison]::OrdinalIgnoreCase)
            # Never descend through a link. Get-ChildItem on a reparse point enumerates the
            # TARGET, whose ordinary children carry no ReparsePoint attribute, so the recursive
            # delete below would take somebody else's tree by way of a redirected temp dir.
            #
            # How far up to look differs by spelling. For the profile this uninstall is FOR, the
            # temp directory and the "Unsloth Studio" directory above it are the two this script
            # created and the two that decide where the enumeration lands; a redirected
            # LocalAppData higher up is still that same user's own storage, and refusing there
            # would leave the installer's own temp tree behind on every host that uses folder
            # redirection. Any OTHER spelling may be another profile entirely, and a junction
            # anywhere along it -- LocalAppData, Users, the drive root -- is enough to redirect an
            # ordinary-looking "Unsloth Studio\temp", so there every ancestor has to be ordinary.
            $ancestors = @()
            if ($isPrimary) {
                $ancestors = @($temp, [System.IO.Path]::GetDirectoryName($temp))
            } else {
                $walk = $temp
                # Bounded: GetDirectoryName returns $null at the root, and the cap stops a spin.
                for ($depth = 0; $depth -lt 64; $depth++) {
                    if ([string]::IsNullOrWhiteSpace($walk)) { break }
                    $ancestors += $walk
                    $next = [System.IO.Path]::GetDirectoryName($walk)
                    if ($next -eq $walk) { break }
                    $walk = $next
                }
            }
            $linked = $false
            foreach ($ancestor in $ancestors) {
                try {
                    if ([string]::IsNullOrWhiteSpace($ancestor)) { continue }
                    $info = Get-Item -LiteralPath $ancestor -Force -ErrorAction Stop
                    if ($info.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { $linked = $true }
                } catch {}
            }
            if ($linked) {
                _Substep "skipped (reparse point): $temp" "Yellow"
                # Not ours to delete here, nor in the data-directory pass either.
                $preserved += $temp
                continue
            }
            $entries = @()
            try { $entries = @(Get-ChildItem -LiteralPath $temp -Force -ErrorAction Stop) } catch { continue }
            foreach ($entry in $entries) {
                # Shape, not prefix: "ust-legacy" and "ust-notapid-x" are not ours.
                if ($entry.Name -notmatch '^ust-[0-9]+-[0-9a-f]{8}$') { continue }
                if (-not ($entry.PSIsContainer)) { continue }
                # A LIVE owner keeps its directory, exactly as install.ps1's own sweep does. An
                # uninstall stops the Unsloth instances under the roots it knows about; one from
                # another install root, or another user, is not among them and still needs %TEMP%.
                $ownerPid = 0
                try {
                    $ownerFile = Join-Path $entry.FullName "owner.pid"
                    if ([System.IO.File]::Exists($ownerFile)) {
                        $null = [int]::TryParse(([System.IO.File]::ReadAllText($ownerFile)).Trim(), [ref]$ownerPid)
                    }
                } catch { $ownerPid = 0 }
                if ($ownerPid -gt 0) {
                    $ownerLives = $true
                    try { $null = Get-Process -Id $ownerPid -ErrorAction Stop }
                    catch [Microsoft.PowerShell.Commands.ProcessCommandException] { $ownerLives = $false }
                    catch { $ownerLives = $true }
                    if ($ownerLives) {
                        _Substep "in use by pid ${ownerPid}, left alone: $($entry.FullName)" "Yellow"
                        $preserved += $entry.FullName
                        continue
                    }
                } elseif (-not $isPrimary) {
                    # No recorded owner, and not the profile being uninstalled. install.ps1 reads
                    # that as UNKNOWN rather than abandoned, because an installer killed before
                    # writing owner.pid leaves a live Unsloth holding the directory, and another
                    # user's live %TEMP% is not this uninstall's business. Under our own profile
                    # the shape is enough, since that is what is being removed.
                    _Substep "no recorded owner in another profile, left alone: $($entry.FullName)" "Yellow"
                    $preserved += $entry.FullName
                    continue
                }
                try {
                    if ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                        [System.IO.Directory]::Delete($entry.FullName, $false)
                    } else {
                        Remove-Item -LiteralPath $entry.FullName -Recurse -Force -ErrorAction Stop
                    }
                    _Substep "removed: $($entry.FullName)" "Green"
                } catch {
                    _Substep "could not remove: $($entry.FullName) ($($_.Exception.Message))" "Yellow"
                    $script:RemoveFailed = $true
                }
            }
            # Only when empty, so a temp dir holding anything else is left alone, and likewise the
            # "Unsloth Studio" parent, which on the second spelling may not be ours to delete.
            foreach ($dir in @($temp, [System.IO.Path]::GetDirectoryName($temp))) {
                try {
                    if ((Test-Path -LiteralPath $dir -PathType Container) -and
                        -not @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop)) {
                        [System.IO.Directory]::Delete($dir, $false)
                    }
                } catch {}
            }
        }
        return $preserved
    }

    # Delete a tree except for the paths in -Keep and the directories above them, which have to
    # survive to hold them. With an empty -Keep this is _RemovePath, which is the ordinary case.
    function _RemoveTreeKeeping {
        param(
            [string]$Path,
            [string[]]$Keep
        )
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        $self = $Path.TrimEnd('\','/')
        $holdsKept = $false
        foreach ($k in @($Keep)) {
            if ([string]::IsNullOrWhiteSpace($k)) { continue }
            $kept = $k.TrimEnd('\','/')
            # The kept path itself: stop here, with everything inside it.
            if ([string]::Equals($kept, $self, [System.StringComparison]::OrdinalIgnoreCase)) { return }
            if ($kept.StartsWith(($self + '\'), [System.StringComparison]::OrdinalIgnoreCase) -or
                $kept.StartsWith(($self + '/'), [System.StringComparison]::OrdinalIgnoreCase)) {
                $holdsKept = $true
            }
        }
        if (-not $holdsKept) {
            _RemovePath $Path
            return
        }
        # Something below survives, so this directory does too; take the rest of its children.
        Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue | ForEach-Object {
            _RemoveTreeKeeping -Path $_.FullName -Keep $Keep
        }
    }

    # Remove the shared data dir, but keep unsloth.ico if a WSL shortcut still points
    # at it (else that shortcut blanks); uninstall.sh drops it when WSL is removed.
    function _RemoveDataDirKeepingWslIcon {
        param(
            [string]$DataDir,
            # WSL-shortcut search dirs; default Start Menu + Desktop, overridable for tests.
            [string[]]$ShortcutDirs = $null,
            # Paths under the data dir a previous pass kept, typically a live Unsloth's %TEMP%.
            [string[]]$Preserve = @()
        )
        if ([string]::IsNullOrWhiteSpace($DataDir)) { return }
        if (-not (Test-Path -LiteralPath $DataDir)) { return }
        # $null = not passed (use defaults); test $null, not truthiness, so an explicit @() is honored.
        if ($null -eq $ShortcutDirs) {
            # Guard $env:APPDATA: unset in service/CI contexts, where Join-Path errors noisily.
            $ShortcutDirs = @()
            if (-not [string]::IsNullOrWhiteSpace($env:APPDATA)) {
                $ShortcutDirs += Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
            }
            try {
                $desktop = [Environment]::GetFolderPath("Desktop")
                if (-not [string]::IsNullOrWhiteSpace($desktop)) { $ShortcutDirs += $desktop }
            } catch {}
        }
        $wslShortcuts = @()
        foreach ($d in $ShortcutDirs) {
            if ($d -and (Test-Path -LiteralPath $d)) {
                $wslShortcuts += Get-ChildItem -LiteralPath $d -Filter "Unsloth Studio (WSL*.lnk" -ErrorAction SilentlyContinue
            }
        }
        $keep = @()
        foreach ($p in @($Preserve)) { if (-not [string]::IsNullOrWhiteSpace($p)) { $keep += $p } }
        if (@($wslShortcuts).Count -eq 0 -and $keep.Count -eq 0) {
            _RemovePath $DataDir
            return
        }
        if (@($wslShortcuts).Count -gt 0) {
            # A WSL shortcut survives: keep its shared icon.
            _Substep "keeping $(Join-Path $DataDir 'unsloth.ico') for the WSL shortcut" "Gray"
            $keep += (Join-Path $DataDir "unsloth.ico")
        }
        _RemoveTreeKeeping -Path $DataDir -Keep $keep
    }

    # Is this bin\unsloth.cmd the launcher install.ps1 wrote, or just a file with that name? The
    # distinction decides whether a directory gets deleted recursively, and `unsloth.cmd` is a
    # plausible wrapper for anyone shipping an unsloth-based tool, so pointing UNSLOTH_STUDIO_HOME
    # at such a project must not hand its whole tree to _RemovePath.
    #
    # The trampoline is the marker: install.ps1 bakes that exact expression into the shim, no other
    # file has a reason to carry it, and it survives every layout the shim has. Bounded read: the
    # real shim is a few hundred bytes.
    function _IsUnslothCmdShim {
        param([string]$Path)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        try {
            $item = Get-Item -LiteralPath $Path -ErrorAction Stop
            if ($item.Length -gt 8192) { return $false }
            $text = [System.IO.File]::ReadAllText($Path)
        } catch {
            # Unreadable proves nothing, and "proves nothing" must not mean "delete it".
            return $false
        }
        return ($text -like "*unsloth-studio-managed-launcher*" -and $text -like "*from unsloth_cli import app*")
    }

    # A path is an Unsloth-owned root iff one of install.ps1's sentinels exists:
    #   <root>\share\studio.conf, <root>\unsloth_studio\.unsloth-studio-owned,
    #   <root>\bin\unsloth.exe, or a <root>\bin\unsloth.cmd this installer wrote.
    # The .cmd is the launcher written where an Application Control policy denies the generated
    # console script, so an install whose .exe that policy quarantined still owns its root.
    # Plus the legacy venv shapes install.ps1 still migrates: share\studio.conf is never written
    # on Windows, so every sentinel above postdates the bin\ shim dir.
    # A sentinel this gate may trust. Deliberately a plain existence test, following links:
    # refusing a reparse point here, or a marker inside one, buys nothing and costs a supported
    # install. Nothing, because anyone who can plant a junction at that path can plant a plain
    # file there instead, which this has always accepted. A supported install, because relocating
    # a multi-gigabyte venv with a junction leaves a REAL marker behind one, and refusing it
    # strands the install, which is the failure this gate exists to prevent. The link test belongs
    # in install.ps1's claim, where following one would TRUNCATE the target rather than read it.
    function _IsOwnerMarker {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        return (Test-Path -LiteralPath $Path -PathType Leaf)
    }

    # Is $Path a Python venv? What the gate reads out of one is evidence only if the directory
    # really is one; a bare name at that path is somebody else's.
    function _IsVenvDir {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        # No reparse check here either, and for the same reason: a relocated venv is still a
        # venv. The leftover scan below rejects links on its own, where the name came from a
        # wildcard rather than from us.
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
        if (Test-Path -LiteralPath (Join-Path $Path "pyvenv.cfg") -PathType Leaf) { return $true }
        return (Test-Path -LiteralPath (Join-Path $Path "Scripts\python.exe") -PathType Leaf)
    }

    # The exact shape an installer gives a moved-aside venv. install.ps1 keeps every other
    # spelling as the user's data (Test-StudioVenvRollbackMustBePreserved); do not be looser.
    function _IsInstallerLeftoverName {
        param([string]$Name)
        # Two patterns: only the rollback name carries a collision counter (install.ps1:4171).
        # No "time" alternative; that is install.sh's date(1) fallback and has no Windows twin.
        # [0-9], not \d, which matches every Unicode decimal digit; install.ps1 writes ASCII.
        if ($Name -match '^unsloth_studio\.rollback\.[0-9]{14}\.[0-9]+(\.[0-9]+)?$') { return $true }
        return ($Name -match '^\.venv\.invalid\.[0-9]{14}\.[0-9]+$')
    }

    function _IsStudioRoot {
        param([string]$Path, [switch]$ManagedDefaultRoot)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        # install.ps1 writes the first when it creates the root, before the uv cache and long
        # before the venv, so a partial install identifies itself instead of being guessed at.
        if (_IsOwnerMarker (Join-Path $Path ".unsloth-studio-owned")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path "share\studio.conf")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path "unsloth_studio\.unsloth-studio-owned")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path ".venv\.unsloth-studio-owned")) { return $true }
        if (Test-Path -LiteralPath (Join-Path $Path "bin\unsloth.exe") -PathType Leaf) { return $true }
        if (_IsUnslothCmdShim (Join-Path $Path "bin\unsloth.cmd")) { return $true }
        # Below here is INSIDE a venv, where pip puts it for any install of the wheel: it names
        # the wheel, not the owner. Proof only at the managed root, which install.ps1:4453 also
        # makes the only root that can hold the layout, or a stale UNSLOTH_STUDIO_HOME deletes it.
        if (-not $ManagedDefaultRoot) { return $false }
        foreach ($venv in @("unsloth_studio", ".venv")) {
            if (-not (_IsVenvDir (Join-Path $Path $venv))) { continue }
            if (Test-Path -LiteralPath (Join-Path $Path "$venv\Scripts\unsloth.exe") -PathType Leaf) { return $true }
            # Antivirus takes that .exe out of a venv that still runs; install.ps1:6412 repairs
            # through it, so a pre-marker root has nothing else left.
            foreach ($pkg in @("unsloth_cli", "unsloth")) {
                if (Test-Path -LiteralPath (Join-Path $Path "$venv\Lib\site-packages\$pkg") -PathType Container) { return $true }
            }
        }
        # An install that died between moving the old venv aside (install.ps1:4487, :4165) and
        # writing the marker (install.ps1:4524) leaves only these, and only install.ps1 makes
        # either name, always by renaming a venv.
        foreach ($leftover in @("unsloth_studio.rollback.*", ".venv.invalid.*")) {
            foreach ($dir in @(Get-ChildItem -LiteralPath $Path -Filter $leftover -Directory -Force -ErrorAction SilentlyContinue)) {
                # Never a reparse point: install.ps1 refuses to prune a linked rollback either.
                if (($dir.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
                if (-not (_IsInstallerLeftoverName $dir.Name)) { continue }
                if (_IsVenvDir $dir.FullName) { return $true }
            }
        }
        return $false
    }

    # Hard deny list: never recursively delete a drive root, USERPROFILE, its parent or a system dir.
    function _IsUnsafeRoot {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $true }
        $norm = $null
        try { $norm = [System.IO.Path]::GetFullPath($Path).TrimEnd('\','/') } catch { return $true }
        if ([string]::IsNullOrWhiteSpace($norm)) { return $true }
        # Drive root, e.g. C:\
        if ($norm -match '^[A-Za-z]:[\\/]?$') { return $true }
        $userProfile = $env:USERPROFILE
        if ($userProfile) {
            $userProfile = $userProfile.TrimEnd('\','/')
            if ($norm -ieq $userProfile) { return $true }
            try {
                $parent = Split-Path -LiteralPath $userProfile
                if ($parent -and ($norm -ieq $parent.TrimEnd('\','/'))) { return $true }
            } catch { }
        }
        $systemRoots = @(
            $env:SystemRoot, $env:windir, $env:ProgramFiles, ${env:ProgramFiles(x86)},
            $env:ProgramData, $env:APPDATA, $env:LOCALAPPDATA
        )
        foreach ($s in $systemRoots) {
            if (-not [string]::IsNullOrWhiteSpace($s)) {
                $s2 = $s.TrimEnd('\','/')
                if ($norm -ieq $s2) { return $true }
            }
        }
        return $false
    }

    # Parse UNSLOTH_EXE='<path>' out of a share\studio.conf and return the
    # implied install root (three dirnames up from the venv exe).
    function _RootFromConf {
        param([string]$ConfFile)
        if (-not (Test-Path -LiteralPath $ConfFile -PathType Leaf)) { return $null }
        $line = Get-Content -LiteralPath $ConfFile -ErrorAction SilentlyContinue |
            Where-Object { $_ -match "^UNSLOTH_EXE\s*=" } | Select-Object -First 1
        if (-not $line) { return $null }
        # Tolerate ' value ' single-quoted with ''  -> ' apostrophe escape.
        if ($line -match "^UNSLOTH_EXE\s*=\s*'(.*)'\s*$") {
            $exe = $Matches[1] -replace "''", "'"
            try {
                $bin = Split-Path -LiteralPath $exe
                $studio = Split-Path -LiteralPath $bin
                $root = Split-Path -LiteralPath $studio
                if ($root) { return $root }
            } catch { }
        }
        return $null
    }

    # Expand a leading ~ or ~/ ~\ to $env:USERPROFILE, so env-mode roots written with the tilde
    # shape install.ps1 supports are found here too.
    function _ExpandTilde {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
        $p = $Path.Trim()
        if ($p -eq '~') { return $env:USERPROFILE }
        if ($p.StartsWith('~/') -or $p.StartsWith('~\')) {
            if ($env:USERPROFILE) {
                return (Join-Path $env:USERPROFILE $p.Substring(2).TrimStart('/','\'))
            }
        }
        return $p
    }

    # Discover non-default Unsloth roots from env vars + studio.conf files. Mirrors install.ps1's
    # precedence: UNSLOTH_STUDIO_HOME wins and STUDIO_HOME is ignored when both are set, or
    # uninstalling install A would also delete install B from a stale STUDIO_HOME.
    # The master root UNSLOTH_HOME names, or $null. studio\ is its child and llama.cpp, node and
    # whisper.cpp are its other children, so removing the Studio root alone strands them. Trimmed
    # and tilde-expanded like storage_roots.unsloth_home() and studio\setup.ps1.
    function _MasterRoot {
        $raw = $env:UNSLOTH_HOME
        if ([string]::IsNullOrWhiteSpace($raw)) {
            # The note setup.ps1 leaves in the Studio tree, when this run has no UNSLOTH_HOME of
            # its own. `$env:UNSLOTH_HOME = 'D:\portable'; unsloth studio update` installs the
            # runtimes there and leaves nothing in a later environment, so without the note an
            # uninstall removed the Studio tree and stranded them. Every Studio root this script
            # already knows is consulted and the first readable note wins; the deny list and the
            # marker gates still apply to whatever it names, so a stale note cannot license a
            # removal the environment could not. Mirrors _master_root in uninstall.sh.
            $noteRoots = @()
            if ($env:USERPROFILE) { $noteRoots += (Join-Path $env:USERPROFILE ".unsloth\studio") }
            foreach ($override in @($env:UNSLOTH_STUDIO_HOME, $env:STUDIO_HOME)) {
                if (-not [string]::IsNullOrWhiteSpace($override)) {
                    $noteRoots += (_ExpandTilde $override.Trim())
                }
            }
            foreach ($noteRoot in $noteRoots) {
                $notePath = Join-Path $noteRoot "share\.unsloth-master-root"
                if (-not (Test-Path -LiteralPath $notePath -PathType Leaf)) { continue }
                try {
                    # One line, first only: a note that grew a second line is not one we wrote.
                    $line = @(Get-Content -LiteralPath $notePath -TotalCount 1 -ErrorAction Stop)[0]
                } catch { continue }
                if (-not [string]::IsNullOrWhiteSpace($line)) { $raw = $line; break }
            }
        }
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        $expanded = _ExpandTilde $raw.Trim()
        $norm = $null
        # The provider path first, exactly as setup.ps1's Get-CanonicalDir resolves it.
        # [IO.Path]::GetFullPath anchors a RELATIVE root at [Environment]::CurrentDirectory,
        # which PowerShell does not keep in step with its own location: after a Set-Location the
        # two disagree, so a relative UNSLOTH_HOME named one install to setup and a different one
        # here. Nothing downstream would notice -- process selection, the marker check and the
        # removal all just operate on the wrong root, and the marker only spares trees that are
        # not Unsloth's, not another Unsloth install.
        try {
            $norm = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($expanded)
        } catch { $norm = $null }
        try { $norm = [System.IO.Path]::GetFullPath($(if ($norm) { $norm } else { $expanded })).TrimEnd('\','/') }
        catch { return $null }
        if (-not $norm) { return $null }
        # The default root is left to the blocks that own it, which remove it unconditionally.
        if ($env:USERPROFILE -and ($norm -ieq (Join-Path $env:USERPROFILE ".unsloth").TrimEnd('\','/'))) {
            return $null
        }
        return $norm
    }

    function _CustomStudioRoots {
        $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        $defaultRoot = $null
        if ($env:USERPROFILE) {
            $defaultRoot = (Join-Path $env:USERPROFILE ".unsloth\studio")
        }

        $emit = {
            param($Path)
            if ([string]::IsNullOrWhiteSpace($Path)) { return }
            $expanded = _ExpandTilde $Path
            $norm = $null
            try { $norm = [System.IO.Path]::GetFullPath($expanded).TrimEnd('\','/') } catch { return }
            if (-not $norm) { return }
            if ($defaultRoot -and ($norm -ieq $defaultRoot.TrimEnd('\','/'))) { return }
            if ($seen.Add($norm)) { Write-Output $norm }
        }

        $envRoot = $null
        # IsNullOrWhiteSpace, not truthiness: storage_roots.studio_root() trims these, so a
        # whitespace-only override is unset to every resolver. A bare truthy test called it
        # present, suppressed the master-root branch below, and then discarded the whitespace
        # path, leaving <UNSLOTH_HOME>\studio installed while its runtime siblings went.
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
            $envRoot = $env:UNSLOTH_STUDIO_HOME.Trim()
        } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
            $envRoot = $env:STUDIO_HOME.Trim()
        } else {
            # Last, as in storage_roots.studio_root(): UNSLOTH_HOME names the tree, and the two
            # above name this exact directory, so either of them wins outright.
            $master = _MasterRoot
            if ($master) { $envRoot = (Join-Path $master "studio") }
        }
        if ($envRoot) {
            $expandedEnv = _ExpandTilde $envRoot
            & $emit $expandedEnv
            $confRoot = _RootFromConf (Join-Path $expandedEnv "share\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
        # Default-mode conf at LOCALAPPDATA\Unsloth Studio.
        if ($env:LOCALAPPDATA) {
            $confRoot = _RootFromConf (Join-Path $env:LOCALAPPDATA "Unsloth Studio\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
    }

    # Return $true iff the PID's image path lives under one of $KnownRoots, so an unrelated
    # process listening on a stale Unsloth port is never killed.
    function _PidUnderKnownRoot {
        param([int]$Pid_, [string[]]$KnownRoots)
        if (-not $KnownRoots -or $KnownRoots.Count -eq 0) { return $false }
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$Pid_" -ErrorAction SilentlyContinue
            if (-not $proc) { return $false }
            $exe = $proc.ExecutablePath
            if (-not $exe) { return $false }
            foreach ($r in $KnownRoots) {
                if ($r -and ($exe -ilike "$r\*")) { return $true }
            }
        } catch { }
        return $false
    }

    # Stop an Unsloth backend whose port is recorded in <DataDir>\studio.port.
    # Only kills if the listening PID's exe path is under a known Unsloth root.
    function _StopByPortFile {
        param([string]$PortFile, [string[]]$KnownRoots)
        if (-not (Test-Path -LiteralPath $PortFile -PathType Leaf)) { return }
        $port = Get-Content -LiteralPath $PortFile -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($port) { $port = $port.Trim() }
        if (-not ($port -match '^[0-9]+$')) {
            Remove-Item -LiteralPath $PortFile -Force -ErrorAction SilentlyContinue
            return
        }
        try {
            $conns = Get-NetTCPConnection -State Listen -LocalPort ([int]$port) -ErrorAction SilentlyContinue
            foreach ($c in $conns) {
                if (-not (_PidUnderKnownRoot -Pid_ ([int]$c.OwningProcess) -KnownRoots $KnownRoots)) { continue }
                try {
                    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
                } catch { }
            }
        } catch {
            # netstat fallback for older PowerShell. Require LISTENING so we never kill a process
            # whose remote endpoint just happens to be the cached port (browser -> :443 etc.).
            try {
                $lines = & netstat.exe -ano 2>$null |
                    Select-String -Pattern "LISTENING" |
                    Select-String -Pattern ":$port\s"
                foreach ($l in $lines) {
                    $parts = ($l.ToString() -split '\s+') | Where-Object { $_ }
                    $pid_ = $parts[-1]
                    if ($pid_ -match '^\d+$') {
                        if (-not (_PidUnderKnownRoot -Pid_ ([int]$pid_) -KnownRoots $KnownRoots)) { continue }
                        try { Stop-Process -Id ([int]$pid_) -Force -ErrorAction SilentlyContinue } catch { }
                    }
                }
            } catch { }
        }
        Remove-Item -LiteralPath $PortFile -Force -ErrorAction SilentlyContinue
    }

    # Stop processes whose ExecutablePath lives under an unsloth_studio venv.
    # Anchoring on the venv path avoids matching unrelated python.exe / studio.exe.
    function _StopStudioProcesses {
        param([string[]]$KnownRoots)
        # An EXPLICIT empty list means "no root qualifies", not "do not scope". @() is false in
        # PowerShell, so `if ($KnownRoots)` swept the whole machine on a run with nothing to delete.
        $scoped = $PSBoundParameters.ContainsKey('KnownRoots')
        try {
            $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.ExecutablePath -and ($_.ExecutablePath -match '\\unsloth_studio\\.*\\(unsloth|python|studio)\.exe$') -and
                    $_.CommandLine -and ($_.CommandLine -match 'studio')
                }
            foreach ($p in $procs) {
                # Optional scope: only kill if the exe is under a known root.
                if ($scoped) {
                    $match = $false
                    foreach ($r in $KnownRoots) {
                        if ($p.ExecutablePath -and ($p.ExecutablePath -ilike "$r\*")) { $match = $true; break }
                    }
                    if (-not $match) { continue }
                }
                try {
                    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
                } catch { }
            }
        } catch { }
    }

    # The Unsloth-managed subtrees underneath the reparse-point TARGET of each Unsloth home, for
    # the stop scan only. A junction or symlinked home runs its native binaries out of the PHYSICAL
    # path (the backend resolves the home before deriving <home>\stable-diffusion.cpp and launching
    # sd-server there), while _CustomStudioRoots only normalizes the string: GetFullPath is lexical
    # and leaves a reparse point untouched. The prefix scan below reads Win32_Process.ExecutablePath,
    # the real image path, so without the target a running server never matches and survives an
    # uninstall that took its tree.
    #
    # The SUBTREES, never the bare target: the delete unlinks only the reparse point, so anything
    # under the target that is not ours is neither locking nor being removed and must not be
    # force-stopped. Homes only, for the same reason: a component dir ($defaultNode, ...) can
    # itself be a link onto a shared runtime, and resolving it would scope in every process there.
    #
    # Stop scan only, deliberately. The deletes still refuse to chase a link out of the expected
    # location, which is what the deny list exists to prevent; ending our own process is not.
    function _ManagedPathsUnderReparseTargets {
        param([string[]]$Roots)
        # Everything setup.ps1 / the prebuilt installers place inside an Unsloth home.
        $managed = @(
            "unsloth_studio", "share", "bin", "llama.cpp", "whisper.cpp", "node",
            "stable-diffusion.cpp", ".cache", ".venv_t5_510", ".venv_t5_530", ".venv_t5_550"
        )
        $out = @()
        foreach ($r in @($Roots | Where-Object { $_ })) {
            try {
                $item = Get-Item -LiteralPath $r -Force -ErrorAction SilentlyContinue
                if (-not $item -or -not $item.Target) { continue }
                $t = @($item.Target)[0]
                if ([string]::IsNullOrWhiteSpace($t)) { continue }
                # A symlink target may be relative; a junction's never is. Anchor it on the link's
                # own parent, or GetFullPath would read it from the uninstaller's working directory.
                if (-not [System.IO.Path]::IsPathRooted($t)) {
                    $t = Join-Path (Split-Path -LiteralPath $item.FullName) $t
                }
                $t = [System.IO.Path]::GetFullPath($t).TrimEnd('\', '/')
                if (-not $t) { continue }
                foreach ($sub in $managed) {
                    $p = (Join-Path $t $sub).TrimEnd('\', '/')
                    if ($out -notcontains $p) { $out += $p }
                }
            } catch { }
        }
        return $out
    }

    # Stop processes that would block deleting the paths we remove. Unlike _StopStudioProcesses
    # (venv exe only), this also catches llama-server/llama-cli, the unsloth.exe shim and orphaned
    # mp workers holding a venv DLL (an open DLL handle blocks the dir delete), found by scanning
    # each candidate's loaded modules as well as its image path.
    function _StopProcessesLockingRoots {
        param([string[]]$Roots)
        $clean = @($Roots | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\','/') })
        if ($clean.Count -eq 0) { return }
        $underRoot = {
            param($p)
            if (-not $p) { return $false }
            foreach ($r in $clean) { if ($p -ieq $r -or $p -ilike "$r\*") { return $true } }
            return $false
        }
        # 1. Image path under a target root (venv python, shim, llama-server).
        try {
            foreach ($proc in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
                if ((& $underRoot $proc.ExecutablePath)) {
                    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
        } catch { }
        # 2. A loaded module under a target root (orphaned mp-fork python holding a
        #    venv DLL). Scoped to names that load our DLLs to keep the scan fast.
        try {
            $cands = Get-Process -Name python, pythonw, unsloth, llama-server, llama-cli, sd-cli, sd-server -ErrorAction SilentlyContinue
            foreach ($proc in $cands) {
                $hit = $false
                try {
                    foreach ($m in $proc.Modules) { if ((& $underRoot $m.FileName)) { $hit = $true; break } }
                } catch { }  # access denied enumerating modules -> skip
                if ($hit) { try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { } }
            }
        } catch { }
    }

    # Default install root + default data dir.
    $defaultStudioHome = if ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".unsloth\studio" } else { $null }
    $defaultDataRoot = _AppDataRoot $env:LOCALAPPDATA 'LocalApplicationData'
    $defaultDataDir = if ($defaultDataRoot) { Join-Path $defaultDataRoot "Unsloth Studio" } else { $null }
    # The SECOND LocalAppData spelling gets the private temp tree only, never the data directory.
    # install.ps1 falls through from a set-but-unusable $env:LOCALAPPDATA to the known folder when
    # it places "Unsloth Studio\temp", so that tree can outlive an uninstall; but the two spellings
    # differ mainly when they name a DIFFERENT USER's profile, and the data-dir delete is
    # recursive, sentinel-free and deny-list-free.
    $knownLocalAppData = $null
    try { $knownLocalAppData = [Environment]::GetFolderPath('LocalApplicationData') } catch { $knownLocalAppData = $null }
    # The first non-blank spelling is the one _AppDataRoot resolved: the profile this run is for.
    $primaryPrivateTemp = if ($defaultDataDir) { Join-Path $defaultDataDir "temp" } else { $null }
    $privateTempDirs = @()
    foreach ($root in @($env:LOCALAPPDATA, $knownLocalAppData)) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $candidate = Join-Path $root "Unsloth Studio\temp"
        if ($privateTempDirs -notcontains $candidate) { $privateTempDirs += $candidate }
    }
    # Default-mode ~/.unsloth holds a SHARED llama.cpp build + .cache that are siblings of studio,
    # so deleting <studio> misses them. No-op in env/custom mode (nested under the custom root,
    # removed with it). A user-set UNSLOTH_LLAMA_CPP_PATH is left alone.
    $defaultUnslothHome = if ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".unsloth" } else { $null }
    $defaultLlamaCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "llama.cpp" } else { $null }
    # Default-mode native diffusion build, a sibling of studio like llama.cpp. No-op in env/custom
    # mode and when absent. A user-set UNSLOTH_SD_CPP_PATH is left alone.
    $defaultSdCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "stable-diffusion.cpp" } else { $null }
    $defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" } else { $null }
    # Isolated Node.js runtime (install_node_prebuilt.py), a default-mode sibling of studio.
    $defaultNode = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "node" } else { $null }
    # llama.cpp atomic-install staging root (install_llama_prebuilt.py), a sibling of the install
    # dir. Usually pruned after activate, but an interrupted build leaves a "<name>.staging-XXXX"
    # tree behind and that blocks the empty-dir cleanup of ~/.unsloth below.
    $defaultStaging = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".staging" } else { $null }
    # Managed whisper.cpp dictation engine (setup.ps1 installs it at $UnslothHome\whisper.cpp), a
    # default-mode sibling of studio. No-op in env/custom mode and when absent.
    $defaultWhisperCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "whisper.cpp" } else { $null }

    # Build known-root list FIRST so the port-file kill can verify ownership.
    $customRoots = @(_CustomStudioRoots)
    $knownRoots = @()
    if ($defaultStudioHome) { $knownRoots += $defaultStudioHome }
    $knownRoots += $customRoots
    # The roots this run would actually delete. Everything that stops, deletes or edits on behalf
    # of a root takes THIS list, never $knownRoots: all of it runs before the gates below, and a
    # stale studio.conf can name a directory another application has taken over. _IsUnsafeRoot as
    # well as _IsStudioRoot, because the removal loop refuses on either.
    $ownedRoots = @()
    if ($defaultStudioHome -and (_IsStudioRoot $defaultStudioHome -ManagedDefaultRoot) -and
        -not (_IsUnsafeRoot $defaultStudioHome)) {
        $ownedRoots += $defaultStudioHome
    }
    foreach ($r in $customRoots) {
        if ((_IsStudioRoot $r) -and -not (_IsUnsafeRoot $r)) { $ownedRoots += $r }
    }

    # ── Stop running servers ──
    _Step "Stopping any running Unsloth Studio servers..."
    if ($defaultDataDir) {
        _StopByPortFile -PortFile (Join-Path $defaultDataDir "studio.port") -KnownRoots $ownedRoots
    }
    # $ownedRoots, not $customRoots: _StopByPortFile deletes the port file on its way out.
    foreach ($r in $ownedRoots) {
        _StopByPortFile -PortFile (Join-Path $r "share\studio.port") -KnownRoots $ownedRoots
    }
    _StopStudioProcesses -KnownRoots $ownedRoots
    # The app and the WebView2 helpers holding its profile open must both exit before the
    # EBWebView delete below. Same resolver as that removal, or the sweep misses the profile
    # we then try to delete and the helpers keep holding locks.
    $localAppRoot = _AppDataRoot $env:LOCALAPPDATA 'LocalApplicationData'
    $webviewProfile = if ($localAppRoot) { Join-Path $localAppRoot "ai.unsloth.studio" } else { $null }
    # Account-scoped, like every other kill here. installMode is currentUser, so the profile is
    # this account's and shared by all its sessions: a second console or RDS session must die
    # too or it re-creates the profile mid-delete, while another user's Unsloth must not be
    # touched. SIDs, so domain and locale do not matter; an unreadable owner is skipped.
    $meSid = try { [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value } catch { $null }
    $studioPids = @()
    if ($meSid) {
        try {
            # Unsloth.exe too: older releases used the product name as MAINBINARYNAME, and this
            # script is always fetched fresh from main, so it meets those installs. A missed
            # process re-creates the profile. The owner-SID filter keeps the broader name safe.
            $filter = "Name = 'unsloth-studio.exe' OR Name = 'Unsloth.exe'"
            foreach ($sp in (Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue)) {
                $spSid = try { (Invoke-CimMethod -InputObject $sp -MethodName GetOwnerSid -ErrorAction SilentlyContinue).Sid } catch { $null }
                if ($spSid -eq $meSid) { $studioPids += [int]$sp.ProcessId }
            }
        } catch { }
    }
    if ($webviewProfile) {
        # Escape "[", a wildcard class, or we miss our profile and match others. Trailing \
        # spares "<bid>2\EBWebView".
        $pattern = "*" + [System.Management.Automation.WildcardPattern]::Escape($webviewProfile) + "\*"
        try {
            $wvProcs = @(Get-CimInstance Win32_Process -Filter "Name = 'msedgewebview2.exe'" -ErrorAction SilentlyContinue)
            # Parent lookup for the ancestor walk; WebView2 procs are all the chain passes through.
            $parentOf = @{}
            foreach ($p in $wvProcs) { $parentOf[[int]$p.ProcessId] = [int]$p.ParentProcessId }
            # Chromium process model: the renderer, GPU and utility helpers are children of the
            # browser process, not of unsloth-studio.exe, so an immediate-parent check misses
            # them all and they keep EBWebView locked (Stop-Process is not recursive).
            # Depth-capped so a PID-reuse cycle cannot spin here.
            $isOurs = {
                param($StartPid)
                $cur = $StartPid
                for ($hop = 0; $hop -lt 12; $hop++) {
                    if (-not $parentOf.ContainsKey($cur)) { return $false }
                    $parent = $parentOf[$cur]
                    if ($studioPids -contains $parent) { return $true }
                    if ($parent -eq $cur -or $parent -le 0) { return $false }
                    $cur = $parent
                }
                return $false
            }
            foreach ($proc in $wvProcs) {
                # CommandLine is null across an elevation boundary: fall back to the parent chain.
                $mine = if ($proc.CommandLine) { $proc.CommandLine -ilike $pattern }
                        else { & $isOurs ([int]$proc.ProcessId) }
                if ($mine) { try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch { } }
            }
        } catch { }
    }
    if ($studioPids) {
        try { Stop-Process -Id $studioPids -Force -ErrorAction SilentlyContinue } catch { }
        # WebView2 releases the profile only once its browser processes have exited.
        Wait-Process -Id $studioPids -Timeout 10 -ErrorAction SilentlyContinue
    }
    # Only stop the default sd.cpp dir when it carries our owner marker, so stop matches the
    # marker-gated delete and a user's own sd-server at this default path is left running.
    $defaultSdCppToStop = $null
    if ($defaultSdCpp -and (Test-Path -LiteralPath $defaultSdCpp) -and (Test-Path -LiteralPath (Join-Path $defaultSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
        $defaultSdCppToStop = $defaultSdCpp
    }
    # A custom/env-mode sd.cpp build now sits UNDER its root, which the $knownRoots prefix match
    # below already covers. Older builds put it BESIDE the root at <parent>\stable-diffusion.cpp,
    # outside $knownRoots; we delete those marker-owned dirs below, so scan them too.
    # Gated too: a stale root's PARENT can hold another install's marked sd.cpp.
    $customSdCppToStop = @()
    foreach ($r in $customRoots) {
        if (-not (_IsStudioRoot $r)) { continue }
        if (_IsUnsafeRoot $r) { continue }
        $sdc = Join-Path (Split-Path -LiteralPath $r) "stable-diffusion.cpp"
        if ((Test-Path -LiteralPath $sdc) -and (Test-Path -LiteralPath (Join-Path $sdc ".unsloth-studio-owned") -PathType Leaf)) {
            $customSdCppToStop += $sdc
        }
    }
    # Also stop anything holding a handle on the exact paths we delete (llama-server,
    # the CLI shim, an mp-fork python with a venv DLL) so the dir delete isn't refused.
    # Resolved here, before the stop pass: a loaded llama-server.exe under the master root keeps
    # its own tree locked, and the removal below would exhaust its retries and leave it installed.
    $masterRootToStop = _MasterRoot
    $masterChildrenToStop = @()
    if ($masterRootToStop -and -not (_IsUnsafeRoot $masterRootToStop)) {
        foreach ($childName in @("llama.cpp", "node", "whisper.cpp", "stable-diffusion.cpp")) {
            $childPath = Join-Path $masterRootToStop $childName
            # Only the trees this uninstall is going to delete, so an unmarked neighbour's
            # process is never killed.
            if (Test-Path -LiteralPath (Join-Path $childPath ".unsloth-studio-owned") -PathType Leaf) {
                $masterChildrenToStop += $childPath
            }
        }
    }
    # $ownedRoots, not $knownRoots: see the note where the two lists are built. The master
    # children are already marker-gated above, so they carry their own proof of ownership.
    $stopRoots = @($ownedRoots) + @($defaultDataDir, $defaultLlamaCpp, $defaultCache, $defaultNode, $defaultWhisperCpp) + @($defaultSdCppToStop | Where-Object { $_ }) + @($customSdCppToStop) + @($masterChildrenToStop)
    # The reparse expansion turns one path into generic subdirectories of wherever it points
    # (node, bin, unsloth_studio), so it is gated for the same reason.
    _StopProcessesLockingRoots -Roots ($stopRoots + @(_ManagedPathsUnderReparseTargets $ownedRoots))

    # ── Remove custom-root install trees ──
    _Step "Removing data and install directories..."
    foreach ($r in $customRoots) {
        if (_IsUnsafeRoot $r) {
            _Substep "refusing to remove unsafe path: $r" "Yellow"
            # install.ps1 accepts any writable root, so a real install can sit under a deny-listed
            # path. Nothing is deleted and it holds studio.db, so say so. Mirrors uninstall.sh.
            if (Test-Path -LiteralPath $r) { $script:RemoveFailed = $true }
            continue
        }
        if (-not (_IsStudioRoot $r)) {
            _Substep "refusing to remove non-Unsloth path: $r" "Yellow"
            continue
        }
        _RemoveRootRecordingDb $r
        # Native diffusion now installs UNDER the custom root, so the removal above already took
        # it. Older builds put it BESIDE the root at <parent>\stable-diffusion.cpp, which removing
        # the root alone would leave behind. <parent> is user-chosen and "stable-diffusion.cpp" is
        # exactly what a git clone of the upstream project produces, so require our owner marker
        # (install_sd_cpp_prebuilt) before rm and keep any unowned checkout. The derived parent
        # path gets the deny-list check too.
        $customSdCpp = Join-Path (Split-Path -LiteralPath $r) "stable-diffusion.cpp"
        if (_IsUnsafeRoot $customSdCpp) {
            _Substep "refusing to remove unsafe path: $customSdCpp" "Yellow"
        } elseif ((Test-Path -LiteralPath $customSdCpp) -and -not (Test-Path -LiteralPath (Join-Path $customSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
            _Substep "keeping sd.cpp without Unsloth owner marker: $customSdCpp" "Yellow"
        } else {
            _RemovePath $customSdCpp
        }
    }
    # Default install dir (always at %USERPROFILE%\.unsloth\studio when present). Same sentinels
    # as a custom root: an ungated run takes a hand-made one, then ~/.unsloth with the prune below.
    if ($defaultStudioHome -and (Test-Path -LiteralPath $defaultStudioHome) -and
        -not (_IsStudioRoot $defaultStudioHome -ManagedDefaultRoot)) {
        _Substep "refusing to remove non-Unsloth path: $defaultStudioHome" "Yellow"
        # A refused CUSTOM root is somebody else's by definition. This is our own default path,
        # where a damaged install can sit, so a studio.db here is chat history.
        if (Test-Path -LiteralPath (Join-Path $defaultStudioHome "studio.db") -PathType Leaf) {
            $script:StudioDbKept = $true
        }
    } elseif ($defaultStudioHome) {
        _RemoveRootRecordingDb $defaultStudioHome
    }
    # Default data dir. The private temp sweep goes FIRST and hands back what it kept: the primary
    # temp directory lives under this data dir, so a wholesale removal here would erase a live
    # Unsloth's %TEMP% before the sweep ever looked at its owner.pid.
    $preservedTemp = @(_RemoveStudioPrivateTempTrees -Paths $privateTempDirs -PrimaryPath $primaryPrivateTemp)
    if ($defaultDataDir) { _RemoveDataDirKeepingWslIcon $defaultDataDir -Preserve $preservedTemp }
    # The master root's own children. Marker-gated rather than removed outright like the
    # ~/.unsloth ones below: <master> is a directory the user chose and may hold their files, so
    # only a tree an Unsloth installer marked is ours to delete. The locks and .staging are ours
    # by name (prebuilt_core.py) and carry no marker. Mirrors scripts/uninstall.sh.
    $masterRoot = _MasterRoot
    if ($masterRoot -and (_IsUnsafeRoot $masterRoot)) {
        _Substep "refusing to remove unsafe path: $masterRoot" "Yellow"
        $masterRoot = $null
    }
    if ($masterRoot) {
        foreach ($childName in @("llama.cpp", "node", "whisper.cpp", "stable-diffusion.cpp")) {
            $childPath = Join-Path $masterRoot $childName
            if ((Test-Path -LiteralPath $childPath) -and
                -not (Test-Path -LiteralPath (Join-Path $childPath ".unsloth-studio-owned") -PathType Leaf)) {
                _Substep "keeping $childName without Unsloth owner marker: $childPath" "Yellow"
            } else {
                _RemovePath $childPath
            }
        }
        foreach ($lockName in @(".llama.cpp.install.lock", ".node.install.lock",
                                ".whisper.cpp.install.lock", ".sd.cpp.install.lock")) {
            _RemoveLockFile (Join-Path $masterRoot $lockName)
        }
        # The prebuilt installers SHARE <root>\.staging and prune it only when empty, so
        # anything left in it here is not ours: remove the directory only, never its contents.
        $masterStaging = Join-Path $masterRoot ".staging"
        if ((Test-Path -LiteralPath $masterStaging) -and
            -not (Get-ChildItem -LiteralPath $masterStaging -Force -ErrorAction SilentlyContinue)) {
            _RemovePath $masterStaging
        }
        if (Test-Path -LiteralPath $masterRoot) {
            # ".*" and not "*": install_node_prebuilt renames <root>\.<name>.install.lock, so the
            # leading dot is part of the name. Without it this also matches, say, a user's
            # "backup.install.lock.stale.copy", which is not ours to delete in their own root.
            foreach ($stale in @(Get-ChildItem -LiteralPath $masterRoot -Force -ErrorAction SilentlyContinue |
                                 Where-Object { $_.Name -like ".*.install.lock.stale.*" })) {
                _RemoveLockFile $stale.FullName
            }
        }
        # Only when nothing of the user's is left.
        if ((Test-Path -LiteralPath $masterRoot) -and
            -not (Get-ChildItem -LiteralPath $masterRoot -Force -ErrorAction SilentlyContinue)) {
            _RemovePath $masterRoot
        }
    }
    # Shared llama.cpp build + cache, siblings of studio under ~/.unsloth in default mode.
    if ($defaultLlamaCpp) { _RemovePath $defaultLlamaCpp }
    # "stable-diffusion.cpp" is exactly what a git clone of leejet/stable-diffusion.cpp produces
    # and a user may keep their own checkout (or point UNSLOTH_SD_CPP_PATH) here, so require our
    # owner marker before rm: an unowned checkout or a pre-marker Unsloth build is kept.
    if ($defaultSdCpp -and (Test-Path -LiteralPath $defaultSdCpp) -and -not (Test-Path -LiteralPath (Join-Path $defaultSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
        _Substep "keeping sd.cpp without Unsloth owner marker: $defaultSdCpp" "Yellow"
    } elseif ($defaultSdCpp) {
        _RemovePath $defaultSdCpp
    }
    if ($defaultCache) { _RemovePath $defaultCache }
    # Isolated Node.js runtime, a sibling of studio under ~/.unsloth. Nested in env/custom mode.
    if ($defaultNode) { _RemovePath $defaultNode }
    if ($defaultStaging) { _RemovePath $defaultStaging }
    # Managed whisper.cpp prebuilt, a sibling of studio under ~/.unsloth. Only present when one
    # matching the pinned llama.cpp build existed at install time, so many installs lack it.
    if ($defaultWhisperCpp) { _RemovePath $defaultWhisperCpp }
    # Prebuilt install locks: every prebuilt serializes on <parent>\.<name>.install.lock
    # (prebuilt_core.py), and a stray lock keeps ~/.unsloth from being pruned below.
    if ($defaultUnslothHome) {
        foreach ($lockName in @(".llama.cpp.install.lock", ".node.install.lock", ".whisper.cpp.install.lock")) {
            _RemoveLockFile (Join-Path $defaultUnslothHome $lockName)
        }
        # Taking over an abandoned lock renames it to .stale.<pid> before unlinking; a crash
        # between the two strands the rename, so sweep any leftovers. -Force sees dotted names.
        if (Test-Path -LiteralPath $defaultUnslothHome) {
            # -like, not -Filter: the Win32 filter is unreliable for names with several dots.
            foreach ($stale in @(Get-ChildItem -LiteralPath $defaultUnslothHome -Force -ErrorAction SilentlyContinue |
                                 Where-Object { $_.Name -like ".*.install.lock.stale.*" })) {
                _RemoveLockFile $stale.FullName
            }
        }
    }
    # Drop ~/.unsloth itself, but ONLY if now empty -- never nuke unrelated content.
    if ($defaultUnslothHome -and (Test-Path -LiteralPath $defaultUnslothHome) -and
        -not (Get-ChildItem -LiteralPath $defaultUnslothHome -Force -ErrorAction SilentlyContinue)) {
        _RemovePath $defaultUnslothHome
    }

    # Runtime data, created at first launch rather than by install.ps1: LOCALAPPDATA holds the
    # EBWebView profile (a leftover copy serves a stale frontend), APPDATA the app config dir.
    _Step "Removing WebView caches and app data (ai.unsloth.studio)..."
    # Count an unresolvable root as incomplete cleanup: skipping it silently would leave the
    # profile on disk while the summary reports the session as gone.
    foreach ($known in @(
        @{ Var = $env:LOCALAPPDATA; Folder = 'LocalApplicationData' },
        @{ Var = $env:APPDATA;      Folder = 'ApplicationData' }
    )) {
        $base = _AppDataRoot $known.Var $known.Folder
        if ([string]::IsNullOrWhiteSpace($base)) {
            _Substep "could not resolve $($known.Folder); WebView data may remain" "Yellow"
            $script:RemoveFailed = $true
            continue
        }
        _RemovePath (Join-Path $base "ai.unsloth.studio")
    }

    # ── Remove desktop and Start Menu shortcuts ──
    _Step "Removing desktop and Start Menu shortcuts..."
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        if ($desktop) { _RemovePath (Join-Path $desktop "Unsloth Studio.lnk") }
    } catch { }
    if ($env:APPDATA) {
        _RemovePath (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Unsloth Studio.lnk")
    }
    # Invalidate the Win11 Start Menu tile cache so the removed shortcut's tile disappears instead
    # of lingering stale (mirrors install.ps1). Preserves start2.bin, the pin layout.
    try {
        $smehTemp = Join-Path $env:LOCALAPPDATA "Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState"
        if (Test-Path -LiteralPath $smehTemp) {
            Get-ChildItem -LiteralPath $smehTemp -Filter "TileCache_*" -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path $smehTemp "StartUnifiedTileModelCache.dat") -Force -ErrorAction SilentlyContinue
            Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
        }
    } catch { }

    # Re-sweep: the first pass may have left unsloth.ico locked by Explorer/SMEH for the native
    # shortcut; that handle is now freed. (A surviving WSL shortcut still keeps the icon.)
    $preservedTemp = @(_RemoveStudioPrivateTempTrees -Paths $privateTempDirs -PrimaryPath $primaryPrivateTemp)
    if ($defaultDataDir -and (Test-Path -LiteralPath $defaultDataDir)) {
        _RemoveDataDirKeepingWslIcon $defaultDataDir -Preserve $preservedTemp
    }

    # ── Clean user PATH and registry backup ──
    _Step "Cleaning user PATH and registry..."
    try {
        $regKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        if ($regKey) {
            try {
                $rawPath = $regKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
                if ($rawPath) {
                    $entries = $rawPath -split ';'
                    $kept = New-Object System.Collections.ArrayList
                    $removedAny = $false
                    # Only remove PATH entries inside an Unsloth root we actually own. A literal
                    # substring match on `unsloth_studio` would clobber unrelated user virtualenvs.
                    # A root this run refused keeps its PATH entry, because it keeps its files.
                    foreach ($e in $entries) {
                        if ([string]::IsNullOrWhiteSpace($e)) { continue }
                        $expanded = [Environment]::ExpandEnvironmentVariables($e).TrimEnd('\','/')
                        $isStudio = $false
                        foreach ($r in $ownedRoots) {
                            if (-not $r) { continue }
                            $rNorm = $r.TrimEnd('\','/')
                            if ($expanded -ieq $rNorm -or $expanded -ilike "$rNorm\*") {
                                $isStudio = $true; break
                            }
                        }
                        if ($isStudio) {
                            _Substep "removed PATH entry: $e" "Green"
                            $removedAny = $true
                            continue
                        }
                        [void]$kept.Add($e)
                    }
                    if ($removedAny) {
                        $newPath = ($kept -join ';')
                        $regKey.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                        try {
                            $d = "UnslothPathRefresh_" + ([guid]::NewGuid().ToString('N').Substring(0, 8))
                            [Environment]::SetEnvironmentVariable($d, '1', 'User')
                            [Environment]::SetEnvironmentVariable($d, [NullString]::Value, 'User')
                        } catch { }
                    }
                }
            } finally {
                $regKey.Close()
            }
        }
    } catch {
        _Substep "could not update user PATH: $($_.Exception.Message)" "Yellow"
    }
    # Clear the persisted Inductor cache path, when it still names a tree this run deleted.
    #
    # setup.ps1 writes TORCHINDUCTOR_CACHE_DIR to the USER environment, so it outlives the
    # install and every later PyTorch process on this account inherits it, including ones with
    # nothing to do with Unsloth. Left behind, they compile into the removed directory and
    # rebuild part of the tree the uninstall just took away.
    #
    # Only a value inside a root this run owned. A user who pointed the variable at a directory
    # of their own keeps it, and so does the shared C:\tc fallback, which is not install
    # specific and is not deleted here either.
    try {
        $persistedCache = [Environment]::GetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', 'User')
        if (-not [string]::IsNullOrWhiteSpace($persistedCache)) {
            $expandedCache = [Environment]::ExpandEnvironmentVariables($persistedCache).TrimEnd('\', '/')
            foreach ($r in $ownedRoots) {
                if (-not $r) { continue }
                $rNorm = $r.TrimEnd('\', '/')
                if ($expandedCache -ieq $rNorm -or $expandedCache -ilike "$rNorm\*") {
                    [Environment]::SetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', [NullString]::Value, 'User')
                    _Substep "cleared TORCHINDUCTOR_CACHE_DIR: $persistedCache" "Green"
                    break
                }
            }
        }
    } catch {
        _Substep "could not clear TORCHINDUCTOR_CACHE_DIR: $($_.Exception.Message)" "Yellow"
    }
    # Remove HKCU\Software\Unsloth (PathBackup lives here; install.ps1 owns it).
    try {
        Remove-Item -LiteralPath 'HKCU:\Software\Unsloth' -Recurse -Force -ErrorAction SilentlyContinue
    } catch { }

    Write-Host ""
    Write-Host "Unsloth Studio uninstalled."
    if ($script:RemoveFailed) {
        Write-Host "Note: some paths could not be removed (see 'could not remove:' above), so the"
        Write-Host "      signed-in session and local chat history may still be on disk. Remove"
        Write-Host "      those paths by hand to clear them."
    } elseif ($script:StudioDbRemoved) {
        # Scoped to what was removed: a bare run never discovers a coexisting env-mode root.
        Write-Host "Note: this also removed the app's WebView data and the studio.db it found, so"
        Write-Host "      the desktop app's session and the chat history in the install(s) removed"
        Write-Host "      above are gone."
    } else {
        # No studio.db was deleted, so only the WebView-local data is accounted for: an env-mode
        # install this run never discovered still has its keys and history.
        Write-Host "Note: this also removed the app's WebView data, so the desktop app's session"
        Write-Host "      is gone. A browser session is not affected: its tokens live in the same"
        Write-Host "      localStorage as the API keys below."
        if ($script:StudioDbKept) {
            # Named, and with no advice to delete it: the gate kept it precisely because it
            # does not look like ours.
            Write-Host "      $defaultStudioHome carries no Unsloth install marker, so it was"
            Write-Host "      left alone. The studio.db inside it is still there; look at that"
            Write-Host "      directory yourself before deciding what to do with it."
        } else {
            Write-Host "      No studio.db was found, so any chat history in an install root this run"
            Write-Host "      did not see is still on disk."
        }
    }
    Write-Host "Note: provider API keys are kept in the browser's localStorage, not in studio.db."
    Write-Host "      Unless you ran Unsloth as the desktop app, clear site data for the"
    Write-Host "      http://localhost:<port> origin you used to remove them."
    Write-Host "Note: Hugging Face model cache at %USERPROFILE%\.cache\huggingface was left in place."
    Write-Host "Remove it manually with 'Remove-Item -Recurse -Force `"$env:USERPROFILE\.cache\huggingface\hub`"' if desired."
    if (-not $env:UNSLOTH_STUDIO_HOME -and -not $env:STUDIO_HOME) {
        Write-Host ""
        Write-Host "If you installed Unsloth Studio with UNSLOTH_STUDIO_HOME or STUDIO_HOME"
        Write-Host "pointing at a custom directory, re-run this script with the same variable"
        Write-Host "set to also remove that install tree, e.g.:"
        Write-Host "  `$env:UNSLOTH_STUDIO_HOME = 'C:\your\path'; irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex"
    }
}

Uninstall-UnslothStudio @args
