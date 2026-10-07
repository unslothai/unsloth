# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Unsloth Studio uninstaller for Windows PowerShell. Run -Help for details.
# Custom roots (UNSLOTH_STUDIO_HOME / STUDIO_HOME) come from share\studio.conf.
#
# Usage: run -Help. The web one-liner is in that help text and is not repeated here: nothing
# reads this header from inside the script, and the whole file is scanned before any of it runs.
# Why several things here are written the long way: tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)

function Uninstall-UnslothStudio {
    $ErrorActionPreference = "Continue"

    # Reset at entry: a piped run defines this function in the caller's session, so a rerun
    # would otherwise inherit the previous run's flags.
    $script:RemoveFailed = $false
    $script:StudioDbRemoved = $false
    # Separate from RemoveFailed: nothing failed, the default root was refused.
    $script:StudioDbKept = $false
    # One shared wall-clock wait budget for the whole run, not per call site.
    $script:RemoveWaitBudgetMs = 20000

    function _Usage {
        Write-Host @'
Unsloth Studio uninstaller (Windows PowerShell).

Usage:
  irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\scripts\uninstall.ps1

RemoteSigned is enough for the second form: a git clone carries no mark of the
web, so this script loads unsigned. From a downloaded zip it does carry one, so
clear it first with:  Unblock-File .\scripts\uninstall.ps1

Stops running Unsloth Studio servers, then removes the install dir, launcher
data, CLI shim, desktop and Start Menu shortcuts, the user PATH entry and the
PathBackup registry key. In a default-mode install it also removes the shared
prebuilts that sit beside the install dir:
%USERPROFILE%\.unsloth\{llama.cpp,node,whisper.cpp,audio.cpp,.cache}. The Hugging
Face cache is left in place (only audio.cpp's unsloth-audiocpp-links beside it
goes), as is anything else you keep under %USERPROFILE%\.unsloth.
A shared uv package cache (`uv cache dir`) is also left when install reused one.

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

    # throw, not exit, so an embedded run does not kill the caller's session.
    foreach ($arg in $args) {
        if ($arg -in @('-h', '-help', '--help', '-?', '/?')) { _Usage; return }
        Write-Host "uninstall.ps1: unrecognized argument: $arg" -ForegroundColor Red
        Write-Host "Nothing was removed. Re-run with no arguments to uninstall, or -Help."
        throw "uninstall.ps1: unrecognized argument: $arg"
    }

    function _Step { param([string]$Msg) Write-Host $Msg }
    function _Substep { param([string]$Msg, [string]$Color = "Gray") Write-Host "  $Msg" -ForegroundColor $Color }

    # Retries because a just-killed process can briefly hold a handle.
    function _RemovePath {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        # torch inductor holds cache handles for seconds after the server stops.
        $delays = @(250, 500, 1000, 2000, 4000, 4000, 4000, 4000)
        # Each path gets 2100ms free before drawing on the shared budget.
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
                # Read before _Wait runs, so nothing can shadow the error record.
                $failure = $_.Exception.Message
                if (-not $lastTry -and (_Wait $delays[$attempt] ([ref]$freeLeft))) { continue }
                _Substep "could not remove: $Path ($failure)" "Yellow"
                $script:RemoveFailed = $true
                return
            }
            # Remove-Item -Recurse can report success yet leave a locked child, so verify.
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

    # One Directory.Delete($path, $false) call: listing first can mistake an unreadable dir for empty.
    function _RemoveDirIfEmpty {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        try {
            if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
            [System.IO.Directory]::Delete($Path, $false)
            _Substep "removed: $Path" "Green"
        } catch {
            _Substep "keeping non-empty directory: $Path" "Yellow"
        }
    }

    # The exact shape prebuilt_core.py leaves: a component lock name, ".stale.", and the pid it
    # moved aside. The leading dot alone also matched ".backup.install.lock.stale.copy", which in
    # a user-chosen master root is theirs. uninstall.sh applies the same shape.
    $script:StaleLockPattern = '^\.(llama\.cpp|node|whisper\.cpp|audio\.cpp|sd\.cpp)\.install\.lock\.stale\.[0-9]+$'

    # Install locks are always regular files (O_CREAT | O_EXCL); never delete a tree by that name.
    function _RemoveLockFile {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        # A reparse point at a lock name is not a lock; following it would delete its target.
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and -not $item.PSIsContainer -and
            -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            _RemovePath $Path
        } else {
            _Substep "keeping non-file at an install-lock path: $Path" "Yellow"
        }
    }

    # LOCALAPPDATA/APPDATA are unset in service and CI contexts; fall back to the known folder.
    function _AppDataRoot {
        param([string]$Var, [string]$Folder)
        if (-not [string]::IsNullOrWhiteSpace($Var)) { return $Var }
        $p = try { [Environment]::GetFolderPath($Folder) } catch { $null }
        if ([string]::IsNullOrWhiteSpace($p)) { return $null }
        return $p
    }

    # A partial removal can take the sentinels and then fail; restore the marker so a retry
    # recognises the root.
    function _RestoreOwnerMarker {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
        # Get-Item -Force, not Test-Path, which follows a dangling link.
        $marker = Join-Path $Path ".unsloth-studio-owned"
        if (Get-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue) { return }
        try { [System.IO.File]::WriteAllText($marker, "") } catch { }
    }

    # Checks studio.db at the resolved target but deletes only the link: never chase a reparse
    # point out of the tree to delete its target.
    function _RemoveRootRecordingDb {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        # Anchor a relative reparse target to the link's parent, not the working directory.
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
        # studio.db can itself be a reparse point out of the tree.
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

    # Reclaim only install.ps1's ust-<pid>-<hex> dirs and the then-empty parents. Other spellings
    # may be another user's profile, so never delete anything not created here or follow a link.
    function _RemoveStudioPrivateTempTrees {
        param(
            [string[]]$Paths,
            # Any other spelling may be a different user's profile.
            [string]$PrimaryPath
        )
        # Returned so the data-directory removal keeps them too.
        $preserved = @()
        foreach ($temp in @($Paths)) {
            if ([string]::IsNullOrWhiteSpace($temp)) { continue }
            if (-not (Test-Path -LiteralPath $temp -PathType Container)) { continue }
            $isPrimary = (-not [string]::IsNullOrWhiteSpace($PrimaryPath)) -and
                [string]::Equals($temp.TrimEnd('\','/'), $PrimaryPath.TrimEnd('\','/'), [System.StringComparison]::OrdinalIgnoreCase)
            # Never descend through a link. For the primary profile only the two dirs we created must be
            # ordinary; for other spellings every ancestor must be.
            $ancestors = @()
            if ($isPrimary) {
                $ancestors = @($temp, [System.IO.Path]::GetDirectoryName($temp))
            } else {
                $walk = $temp
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
                $preserved += $temp
                continue
            }
            $entries = @()
            try { $entries = @(Get-ChildItem -LiteralPath $temp -Force -ErrorAction Stop) } catch { continue }
            foreach ($entry in $entries) {
                # Shape, not prefix.
                if ($entry.Name -notmatch '^ust-[0-9]+-[0-9a-f]{8}$') { continue }
                if (-not ($entry.PSIsContainer)) { continue }
                # A live owner keeps its directory, as in install.ps1's own sweep.
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
                    # No recorded owner in another profile is unknown, not abandoned.
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
            # Only when empty; the parent may not be ours on another spelling.
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

    # Delete a tree except -Keep paths and their ancestors; empty -Keep is _RemovePath.
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
        Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue | ForEach-Object {
            _RemoveTreeKeeping -Path $_.FullName -Keep $Keep
        }
    }

    # Keep unsloth.ico while a WSL shortcut still points at it; uninstall.sh drops it with WSL.
    function _RemoveDataDirKeepingWslIcon {
        param(
            [string]$DataDir,
            [string[]]$ShortcutDirs = $null,
            # Paths a previous pass kept, typically a live Unsloth's %TEMP%.
            [string[]]$Preserve = @()
        )
        if ([string]::IsNullOrWhiteSpace($DataDir)) { return }
        if (-not (Test-Path -LiteralPath $DataDir)) { return }
        # Test $null, not truthiness, so an explicit @() is honored.
        if ($null -eq $ShortcutDirs) {
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
            _Substep "keeping $(Join-Path $DataDir 'unsloth.ico') for the WSL shortcut" "Gray"
            $keep += (Join-Path $DataDir "unsloth.ico")
        }
        _RemoveTreeKeeping -Path $DataDir -Keep $keep
    }

    # Only the shim install.ps1 wrote counts: its trampoline expression is the marker, since
    # unsloth.cmd is a plausible name for other tools. Bounded read: the real shim is tiny.
    function _IsUnslothCmdShim {
        param([string]$Path)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        try {
            $item = Get-Item -LiteralPath $Path -ErrorAction Stop
            if ($item.Length -gt 8192) { return $false }
            $text = [System.IO.File]::ReadAllText($Path)
        } catch {
            # Unreadable proves nothing, and must not mean delete.
            return $false
        }
        return ($text -like "*unsloth-studio-managed-launcher*" -and $text -like "*from unsloth_cli import app*")
    }

    # Owned iff one of install.ps1's sentinels exists (studio.conf, owner marker, bin\unsloth.exe,
    # our unsloth.cmd) or a legacy venv shape. Plain existence test that follows links on purpose.
    function _IsOwnerMarker {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        return (Test-Path -LiteralPath $Path -PathType Leaf)
    }

    function _IsVenvDir {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        # No reparse check: a relocated venv is still a venv.
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
        if (Test-Path -LiteralPath (Join-Path $Path "pyvenv.cfg") -PathType Leaf) { return $true }
        return (Test-Path -LiteralPath (Join-Path $Path "Scripts\python.exe") -PathType Leaf)
    }

    # Exactly the moved-aside venv shape; install.ps1 keeps every other spelling as user data.
    function _IsInstallerLeftoverName {
        param([string]$Name)
        # Only the rollback name has a collision counter. [0-9], not \d, which matches any Unicode digit.
        if ($Name -match '^unsloth_studio\.rollback\.[0-9]{14}\.[0-9]+(\.[0-9]+)?$') { return $true }
        return ($Name -match '^\.venv\.invalid\.[0-9]{14}\.[0-9]+$')
    }

    function _IsStudioRoot {
        param([string]$Path, [switch]$ManagedDefaultRoot)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        # install.ps1 writes the marker first, so a partial install identifies itself.
        if (_IsOwnerMarker (Join-Path $Path ".unsloth-studio-owned")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path "share\studio.conf")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path "unsloth_studio\.unsloth-studio-owned")) { return $true }
        if (_IsOwnerMarker (Join-Path $Path ".venv\.unsloth-studio-owned")) { return $true }
        if (Test-Path -LiteralPath (Join-Path $Path "bin\unsloth.exe") -PathType Leaf) { return $true }
        if (_IsUnslothCmdShim (Join-Path $Path "bin\unsloth.cmd")) { return $true }
        # Inside a venv a package names the wheel, not the owner; proof only at the managed root.
        if (-not $ManagedDefaultRoot) { return $false }
        foreach ($venv in @("unsloth_studio", ".venv")) {
            if (-not (_IsVenvDir (Join-Path $Path $venv))) { continue }
            if (Test-Path -LiteralPath (Join-Path $Path "$venv\Scripts\unsloth.exe") -PathType Leaf) { return $true }
            # The .exe can be removed from a working venv; a pre-marker root has nothing else left.
            foreach ($pkg in @("unsloth_cli", "unsloth")) {
                if (Test-Path -LiteralPath (Join-Path $Path "$venv\Lib\site-packages\$pkg") -PathType Container) { return $true }
            }
        }
        # Only install.ps1 makes these names, by renaming a venv before writing the marker.
        foreach ($leftover in @("unsloth_studio.rollback.*", ".venv.invalid.*")) {
            foreach ($dir in @(Get-ChildItem -LiteralPath $Path -Filter $leftover -Directory -Force -ErrorAction SilentlyContinue)) {
                if (($dir.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
                if (-not (_IsInstallerLeftoverName $dir.Name)) { continue }
                if (_IsVenvDir $dir.FullName) { return $true }
            }
        }
        return $false
    }

    function _RecordedUvCache {
        param([string]$Root)
        $marker = Join-Path $Root "cache\uv-cache-dir"
        if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { return $null }
        try {
            $raw = [System.IO.File]::ReadAllText($marker)
        } catch {
            return $null
        }
        $raw = $raw.TrimEnd("`r", "`n")
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        return $raw
    }

    function _UvCacheUnderRoot {
        param([string]$Cache, [string]$Root)
        if ([string]::IsNullOrWhiteSpace($Cache) -or [string]::IsNullOrWhiteSpace($Root)) { return $false }
        $normCache = $Cache.Replace('/', '\').TrimEnd('\')
        $normRoot = $Root.Replace('/', '\').TrimEnd('\')
        if ($normCache -eq $normRoot) { return $true }
        return $normCache.StartsWith($normRoot + '\', [StringComparison]::OrdinalIgnoreCase)
    }

    function _UvLeftoverCaches {
        param([string[]]$Roots)
        $saw = $false
        $leftovers = @()
        foreach ($r in $Roots) {
            $rec = _RecordedUvCache $r
            if ($null -eq $rec) { continue }
            $saw = $true
            $under = $false
            foreach ($root in $Roots) {
                # A linked root is only unlinked, so its target cache stays.
                $item = Get-Item -LiteralPath $root -Force -ErrorAction SilentlyContinue
                if ($item -and $item.LinkType) { continue }
                if (_UvCacheUnderRoot $rec $root) { $under = $true; break }
            }
            if (-not $under -and $leftovers -notcontains $rec) { $leftovers += $rec }
        }
        return @{ Saw = $saw; Leftovers = $leftovers }
    }

    # Hard deny list: never recursively delete a drive root, USERPROFILE, its parent or a system dir.
    function _IsUnsafeRoot {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $true }
        $norm = $null
        try { $norm = [System.IO.Path]::GetFullPath($Path).TrimEnd('\','/') } catch { return $true }
        if ([string]::IsNullOrWhiteSpace($norm)) { return $true }
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

    # Install root is three dirnames up from UNSLOTH_EXE in share\studio.conf.
    function _RootFromConf {
        param([string]$ConfFile)
        if (-not (Test-Path -LiteralPath $ConfFile -PathType Leaf)) { return $null }
        $line = Get-Content -LiteralPath $ConfFile -ErrorAction SilentlyContinue |
            Where-Object { $_ -match "^UNSLOTH_EXE\s*=" } | Select-Object -First 1
        if (-not $line) { return $null }
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

    # UNSLOTH_STUDIO_HOME wins and STUDIO_HOME is ignored when both are set, as in install.ps1.
    # Master root from UNSLOTH_HOME, trimmed and tilde-expanded like storage_roots.unsloth_home().
    function _MasterRoot {
        $noteStudio = $null
        $raw = $env:UNSLOTH_HOME
        if ([string]::IsNullOrWhiteSpace($raw)) {
            # Fall back to the note setup.ps1 leaves in the Studio tree; only this run's Studio root, no
            # falling through. Mirrors _master_root in uninstall.sh.
            $noteRoots = @()
            foreach ($override in @($env:UNSLOTH_STUDIO_HOME, $env:STUDIO_HOME)) {
                if (-not [string]::IsNullOrWhiteSpace($override)) {
                    $noteRoots = @(_ExpandTilde $override.Trim())
                    break
                }
            }
            if (-not $noteRoots) {
                # Read the default-mode conf directly; _CustomStudioRoots calls back into this function.
                if ($env:LOCALAPPDATA) {
                    $confRoot = _RootFromConf (Join-Path $env:LOCALAPPDATA "Unsloth Studio\studio.conf")
                    if ($confRoot) { $noteRoots += $confRoot }
                }
                if ($env:USERPROFILE) {
                    $noteRoots += (Join-Path $env:USERPROFILE ".unsloth\studio")
                }
            }
            foreach ($noteRoot in $noteRoots) {
                $notePath = Join-Path $noteRoot "share\.unsloth-master-root"
                if (-not (Test-Path -LiteralPath $notePath -PathType Leaf)) { continue }
                try {
                    # Refuse a two-line note, as storage_roots and the CLI do.
                    # -Encoding UTF8: 5.1 reads BOM-less files as ANSI.
                    $noteLines = @(Get-Content -LiteralPath $notePath -TotalCount 2 -Encoding UTF8 -ErrorAction Stop)
                } catch { continue }
                if ($noteLines.Count -gt 1 -and -not [string]::IsNullOrWhiteSpace($noteLines[1])) { continue }
                $line = if ($noteLines.Count -ge 1) { $noteLines[0] } else { $null }
                if (-not [string]::IsNullOrWhiteSpace($line)) {
                    $raw = $line
                    $noteStudio = $noteRoot
                    break
                }
            }
        }
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        $expanded = _ExpandTilde $raw.Trim()
        $norm = $null
        # Resolve through the provider like setup.ps1's Get-CanonicalDir; GetFullPath uses a stale cwd.
        try {
            $norm = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($expanded)
        } catch { $norm = $null }
        try { $norm = [System.IO.Path]::GetFullPath($(if ($norm) { $norm } else { $expanded })).TrimEnd('\','/') }
        catch { return $null }
        if (-not $norm) { return $null }
        # A note must describe the tree it was found in (containment); explicit UNSLOTH_HOME skips this.
        # Mirrors uninstall.sh and storage_roots._recorded_master_root().
        if ($noteStudio) {
            $here = $null
            try {
                $here = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($noteStudio)
            } catch { $here = $null }
            try { $here = [System.IO.Path]::GetFullPath($(if ($here) { $here } else { $noteStudio })).TrimEnd('\','/') }
            catch { $here = $null }
            if (-not $here) { return $null }
            # Decline any note in the legacy %USERPROFILE%\.unsloth\studio tree, as storage_roots does.
            # Mirrors uninstall.sh.
            if ($env:USERPROFILE) {
                $legacyStudio = $null
                try {
                    $legacyStudio = [System.IO.Path]::GetFullPath(
                        (Join-Path $env:USERPROFILE ".unsloth\studio")).TrimEnd('\','/')
                } catch { $legacyStudio = $null }
                if ($legacyStudio -and ($here -ieq $legacyStudio)) { return $null }
            }
            if (-not ($here -ieq $norm -or $here.StartsWith($norm + [System.IO.Path]::DirectorySeparatorChar,
                      [System.StringComparison]::OrdinalIgnoreCase))) {
                return $null
            }
        }
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
        # IsNullOrWhiteSpace, not truthiness: the resolvers trim these.
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
            $envRoot = $env:UNSLOTH_STUDIO_HOME.Trim()
        } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
            $envRoot = $env:STUDIO_HOME.Trim()
        } else {
            # Last, as in storage_roots.studio_root().
            $master = _MasterRoot
            if ($master) { $envRoot = (Join-Path $master "studio") }
        }
        if ($envRoot) {
            $expandedEnv = _ExpandTilde $envRoot
            & $emit $expandedEnv
            $confRoot = _RootFromConf (Join-Path $expandedEnv "share\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
        if ($env:LOCALAPPDATA) {
            $confRoot = _RootFromConf (Join-Path $env:LOCALAPPDATA "Unsloth Studio\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
    }

    # True iff the PID's image is under $KnownRoots, so an unrelated process on a stale
    # Unsloth port is never killed.
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
            # Require LISTENING so a process merely connected to that port is never killed.
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

    function _StopStudioProcesses {
        param([string[]]$KnownRoots)
        # An explicit empty list means no root qualifies; @() is falsy in PowerShell.
        $scoped = $PSBoundParameters.ContainsKey('KnownRoots')
        try {
            $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.ExecutablePath -and ($_.ExecutablePath -match '\\unsloth_studio\\.*\\(unsloth|python|studio)\.exe$') -and
                    $_.CommandLine -and ($_.CommandLine -match 'studio')
                }
            foreach ($p in $procs) {
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

    # Managed subtrees under each home's reparse target, for the stop scan only: processes run from
    # the physical path. Subtrees and homes only, never the bare target or component links.
    function _ManagedPathsUnderReparseTargets {
        param([string[]]$Roots)
        $managed = @(
            "unsloth_studio", "share", "bin", "llama.cpp", "whisper.cpp", "audio.cpp", "node",
            "stable-diffusion.cpp", ".cache", ".venv_t5_510", ".venv_t5_530", ".venv_t5_550"
        )
        $out = @()
        foreach ($r in @($Roots | Where-Object { $_ })) {
            try {
                $item = Get-Item -LiteralPath $r -Force -ErrorAction SilentlyContinue
                if (-not $item -or -not $item.Target) { continue }
                $t = @($item.Target)[0]
                if ([string]::IsNullOrWhiteSpace($t)) { continue }
                # A relative symlink target is anchored on the link's parent, not the working directory.
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

    # Unlike _StopStudioProcesses, also catches llama-server, the shim and orphaned workers
    # holding a venv DLL (via loaded modules).
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
        try {
            foreach ($proc in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
                if ((& $underRoot $proc.ExecutablePath)) {
                    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
        } catch { }
        # Module scan is scoped to names that load our DLLs, to keep it fast.
        try {
            $cands = Get-Process -Name python, pythonw, unsloth, llama-server, llama-cli, sd-cli, sd-server -ErrorAction SilentlyContinue
            foreach ($proc in $cands) {
                $hit = $false
                try {
                    foreach ($m in $proc.Modules) { if ((& $underRoot $m.FileName)) { $hit = $true; break } }
                } catch { }
                if ($hit) { try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { } }
            }
        } catch { }
    }

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
    $primaryPrivateTemp = if ($defaultDataDir) { Join-Path $defaultDataDir "temp" } else { $null }
    $privateTempDirs = @()
    foreach ($root in @($env:LOCALAPPDATA, $knownLocalAppData)) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $candidate = Join-Path $root "Unsloth Studio\temp"
        if ($privateTempDirs -notcontains $candidate) { $privateTempDirs += $candidate }
    }
    # Default-mode siblings of studio under ~/.unsloth. A user-set UNSLOTH_LLAMA_CPP_PATH is left alone.
    $defaultUnslothHome = if ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".unsloth" } else { $null }
    $defaultLlamaCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "llama.cpp" } else { $null }
    # A user-set UNSLOTH_SD_CPP_PATH is left alone.
    $defaultSdCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "stable-diffusion.cpp" } else { $null }
    $defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" } else { $null }
    $defaultNode = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "node" } else { $null }
    # An interrupted build leaves .staging behind, which blocks pruning ~/.unsloth.
    $defaultStaging = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".staging" } else { $null }
    $defaultWhisperCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "whisper.cpp" } else { $null }
    $defaultAudioCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "audio.cpp" } else { $null }

    # Build known roots first so the port-file kill can verify ownership.
    $customRoots = @(_CustomStudioRoots)
    $knownRoots = @()
    if ($defaultStudioHome) { $knownRoots += $defaultStudioHome }
    $knownRoots += $customRoots
    # Everything that stops, deletes or edits for a root uses this list, never $knownRoots:
    # a stale studio.conf can name a directory another application now owns.
    $ownedRoots = @()
    if ($defaultStudioHome -and (_IsStudioRoot $defaultStudioHome -ManagedDefaultRoot) -and
        -not (_IsUnsafeRoot $defaultStudioHome)) {
        $ownedRoots += $defaultStudioHome
    }
    foreach ($r in $customRoots) {
        if ((_IsStudioRoot $r) -and -not (_IsUnsafeRoot $r)) { $ownedRoots += $r }
    }

    $uvCaches = _UvLeftoverCaches $ownedRoots

    _Step "Stopping any running Unsloth Studio servers..."
    if ($defaultDataDir) {
        _StopByPortFile -PortFile (Join-Path $defaultDataDir "studio.port") -KnownRoots $ownedRoots
    }
    # _StopByPortFile deletes the port file, so only owned roots.
    foreach ($r in $ownedRoots) {
        _StopByPortFile -PortFile (Join-Path $r "share\studio.port") -KnownRoots $ownedRoots
    }
    _StopStudioProcesses -KnownRoots $ownedRoots
    # The app and WebView2 helpers must exit before the EBWebView delete; same resolver as that delete.
    $localAppRoot = _AppDataRoot $env:LOCALAPPDATA 'LocalApplicationData'
    $webviewProfile = if ($localAppRoot) { Join-Path $localAppRoot "ai.unsloth.studio" } else { $null }
    # Account-scoped by SID: every session of this account must die, other users' must not.
    $meSid = try { [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value } catch { $null }
    $studioPids = @()
    if ($meSid) {
        try {
            # Older releases used Unsloth.exe as the binary name; the SID filter keeps that safe.
            $filter = "Name = 'unsloth-studio.exe' OR Name = 'Unsloth.exe'"
            foreach ($sp in (Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue)) {
                $spSid = try { (Invoke-CimMethod -InputObject $sp -MethodName GetOwnerSid -ErrorAction SilentlyContinue).Sid } catch { $null }
                if ($spSid -eq $meSid) { $studioPids += [int]$sp.ProcessId }
            }
        } catch { }
    }
    if ($webviewProfile) {
        # Escape `[` (a wildcard class). Trailing \ spares "<bid>2\EBWebView".
        $pattern = "*" + [System.Management.Automation.WildcardPattern]::Escape($webviewProfile) + "\*"
        try {
            $wvProcs = @(Get-CimInstance Win32_Process -Filter "Name = 'msedgewebview2.exe'" -ErrorAction SilentlyContinue)
            $parentOf = @{}
            foreach ($p in $wvProcs) { $parentOf[[int]$p.ProcessId] = [int]$p.ParentProcessId }
            # WebView2 helpers are children of the browser process, not the app.
            # Depth-capped so a PID-reuse cycle cannot spin.
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
                # CommandLine is null across an elevation boundary.
                $mine = if ($proc.CommandLine) { $proc.CommandLine -ilike $pattern }
                        else { & $isOurs ([int]$proc.ProcessId) }
                if ($mine) { try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch { } }
            }
        } catch { }
    }
    if ($studioPids) {
        try { Stop-Process -Id $studioPids -Force -ErrorAction SilentlyContinue } catch { }
        Wait-Process -Id $studioPids -Timeout 10 -ErrorAction SilentlyContinue
    }
    # Stop the default sd.cpp only when marked, matching the marker-gated delete.
    $defaultSdCppToStop = $null
    if ($defaultSdCpp -and (Test-Path -LiteralPath $defaultSdCpp) -and (Test-Path -LiteralPath (Join-Path $defaultSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
        $defaultSdCppToStop = $defaultSdCpp
    }
    # Older builds put sd.cpp beside the root, outside $knownRoots; gated by marker too.
    $customSdCppToStop = @()
    foreach ($r in $customRoots) {
        if (-not (_IsStudioRoot $r)) { continue }
        if (_IsUnsafeRoot $r) { continue }
        $sdc = Join-Path (Split-Path -LiteralPath $r) "stable-diffusion.cpp"
        if ((Test-Path -LiteralPath $sdc) -and (Test-Path -LiteralPath (Join-Path $sdc ".unsloth-studio-owned") -PathType Leaf)) {
            $customSdCppToStop += $sdc
        }
    }
    # Resolved before the stop pass: a loaded llama-server keeps its own tree locked.
    $masterRootToStop = _MasterRoot
    $masterChildrenToStop = @()
    if ($masterRootToStop -and -not (_IsUnsafeRoot $masterRootToStop)) {
        foreach ($childName in @("llama.cpp", "node", "whisper.cpp", "audio.cpp", "stable-diffusion.cpp")) {
            $childPath = Join-Path $masterRootToStop $childName
            if (Test-Path -LiteralPath (Join-Path $childPath ".unsloth-studio-owned") -PathType Leaf) {
                $masterChildrenToStop += $childPath
            }
        }
    }
    $stopRoots = @($ownedRoots) + @($defaultDataDir, $defaultLlamaCpp, $defaultCache, $defaultNode, $defaultWhisperCpp, $defaultAudioCpp) + @($defaultSdCppToStop | Where-Object { $_ }) + @($customSdCppToStop) + @($masterChildrenToStop)
    # The reparse expansion yields generic subdirectories, so it is gated too.
    _StopProcessesLockingRoots -Roots ($stopRoots + @(_ManagedPathsUnderReparseTargets $ownedRoots))

    _Step "Removing data and install directories..."
    foreach ($r in $customRoots) {
        if (_IsUnsafeRoot $r) {
            _Substep "refusing to remove unsafe path: $r" "Yellow"
            # A refused root still holds studio.db, so report incomplete. Mirrors uninstall.sh.
            if (Test-Path -LiteralPath $r) { $script:RemoveFailed = $true }
            continue
        }
        if (-not (_IsStudioRoot $r)) {
            _Substep "refusing to remove non-Unsloth path: $r" "Yellow"
            continue
        }
        # Flat layout: keep the user-chosen master root rather than take it with the Studio root.
        # Mirrors uninstall.sh.
        $flatMaster = $masterRootToStop
        if ($flatMaster -and ($r.TrimEnd('\', '/') -ieq $flatMaster.TrimEnd('\', '/'))) {
            _Substep "keeping $r`: UNSLOTH_HOME and the Studio root name the same directory," "Yellow"
            _Substep "so removing it would take whatever else you keep there. Delete it by hand" "Yellow"
            _Substep "once you have checked what is in it." "Yellow"
            $script:RemoveFailed = $true
            continue
        }
        _RemoveRootRecordingDb $r
        # Older builds put sd.cpp beside the root; require our marker, since an upstream clone has
        # the same name.
        $customSdCpp = Join-Path (Split-Path -LiteralPath $r) "stable-diffusion.cpp"
        if (_IsUnsafeRoot $customSdCpp) {
            _Substep "refusing to remove unsafe path: $customSdCpp" "Yellow"
        } elseif ((Test-Path -LiteralPath $customSdCpp) -and -not (Test-Path -LiteralPath (Join-Path $customSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
            _Substep "keeping sd.cpp without Unsloth owner marker: $customSdCpp" "Yellow"
        } else {
            _RemovePath $customSdCpp
        }
    }
    # Same sentinels as a custom root.
    if ($defaultStudioHome -and (Test-Path -LiteralPath $defaultStudioHome) -and
        -not (_IsStudioRoot $defaultStudioHome -ManagedDefaultRoot)) {
        _Substep "refusing to remove non-Unsloth path: $defaultStudioHome" "Yellow"
        # Our own default path, so a studio.db here is chat history.
        if (Test-Path -LiteralPath (Join-Path $defaultStudioHome "studio.db") -PathType Leaf) {
            $script:StudioDbKept = $true
        }
    } elseif ($defaultStudioHome) {
        _RemoveRootRecordingDb $defaultStudioHome
    }
    # Temp sweep first: a wholesale data-dir removal would erase a live instance's %TEMP%.
    $preservedTemp = @(_RemoveStudioPrivateTempTrees -Paths $privateTempDirs -PrimaryPath $primaryPrivateTemp)
    if ($defaultDataDir) { _RemoveDataDirKeepingWslIcon $defaultDataDir -Preserve $preservedTemp }
    # The master root's own children are marker-gated: the user chose that directory.
    # Reuse $masterRootToStop, since its note may already be gone. Mirrors scripts/uninstall.sh.
    $masterRoot = $masterRootToStop
    if ($masterRoot -and (_IsUnsafeRoot $masterRoot)) {
        _Substep "refusing to remove unsafe path: $masterRoot" "Yellow"
        $masterRoot = $null
    }
    if ($masterRoot) {
        foreach ($childName in @("llama.cpp", "node", "whisper.cpp", "audio.cpp", "stable-diffusion.cpp")) {
            $childPath = Join-Path $masterRoot $childName
            if ((Test-Path -LiteralPath $childPath) -and
                -not (Test-Path -LiteralPath (Join-Path $childPath ".unsloth-studio-owned") -PathType Leaf)) {
                _Substep "keeping $childName without Unsloth owner marker: $childPath" "Yellow"
            } else {
                _RemovePath $childPath
            }
        }
        foreach ($lockName in @(".llama.cpp.install.lock", ".node.install.lock",
                                ".whisper.cpp.install.lock", ".audio.cpp.install.lock", ".sd.cpp.install.lock")) {
            _RemoveLockFile (Join-Path $masterRoot $lockName)
        }
        # Shared .staging is pruned only when empty; never remove its contents.
        _RemoveDirIfEmpty (Join-Path $masterRoot ".staging")
        if (Test-Path -LiteralPath $masterRoot) {
            # The installer's exact shape only, or a user's file could be taken.
            foreach ($stale in @(Get-ChildItem -LiteralPath $masterRoot -Force -ErrorAction SilentlyContinue |
                                 Where-Object { $_.Name -match $script:StaleLockPattern })) {
                _RemoveLockFile $stale.FullName
            }
        }
        _RemoveDirIfEmpty $masterRoot
    }
    # Shared llama.cpp build + cache, siblings of studio under ~/.unsloth in default mode.
    if ($defaultLlamaCpp) { _RemovePath $defaultLlamaCpp }
    # Require our marker: an upstream git clone produces the same directory name.
    if ($defaultSdCpp -and (Test-Path -LiteralPath $defaultSdCpp) -and -not (Test-Path -LiteralPath (Join-Path $defaultSdCpp ".unsloth-studio-owned") -PathType Leaf)) {
        _Substep "keeping sd.cpp without Unsloth owner marker: $defaultSdCpp" "Yellow"
    } elseif ($defaultSdCpp) {
        _RemovePath $defaultSdCpp
    }
    if ($defaultCache) { _RemovePath $defaultCache }
    if ($defaultNode) { _RemovePath $defaultNode }
    if ($defaultStaging) { _RemovePath $defaultStaging }
    if ($defaultWhisperCpp) { _RemovePath $defaultWhisperCpp }
    # The installer always marks its tree, so an unmarked one is the user's.
    if ($defaultAudioCpp -and (Test-Path -LiteralPath $defaultAudioCpp) -and -not (Test-Path -LiteralPath (Join-Path $defaultAudioCpp ".unsloth-studio-owned") -PathType Leaf)) {
        _Substep "keeping audio.cpp without Unsloth owner marker: $defaultAudioCpp" "Yellow"
    } elseif ($defaultAudioCpp) {
        _RemovePath $defaultAudioCpp
    }
    # audio.cpp model links live beside the HF hub cache; the cache itself stays.
    $hfHub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
             elseif ($env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE }
             elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
             elseif ($env:XDG_CACHE_HOME) { Join-Path (Join-Path $env:XDG_CACHE_HOME "huggingface") "hub" }
             elseif ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
             else { $null }
    $hfParent = if ($hfHub) { Split-Path -Parent $hfHub } else { $null }
    if ($hfParent) { _RemovePath (Join-Path $hfParent "unsloth-audiocpp-links") }
    # %TEMP% is per user and the name carries this user's, so no other account is reached.
    if ($env:USERNAME) {
        $audioCppHome = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth-audiocpp-home-$env:USERNAME"
        if (Test-Path -LiteralPath $audioCppHome -PathType Container) { _RemovePath $audioCppHome }
    }
    # A stray prebuilt install lock keeps ~/.unsloth from being pruned.
    if ($defaultUnslothHome) {
        foreach ($lockName in @(".llama.cpp.install.lock", ".node.install.lock", ".whisper.cpp.install.lock", ".audio.cpp.install.lock")) {
            _RemoveLockFile (Join-Path $defaultUnslothHome $lockName)
        }
        # A crash between rename and unlink strands .stale.<pid> locks.
        if (Test-Path -LiteralPath $defaultUnslothHome) {
            # -match, not -Filter: the Win32 filter is unreliable with several dots.
            foreach ($stale in @(Get-ChildItem -LiteralPath $defaultUnslothHome -Force -ErrorAction SilentlyContinue |
                                 Where-Object { $_.Name -match $script:StaleLockPattern })) {
                _RemoveLockFile $stale.FullName
            }
        }
    }
    if ($defaultUnslothHome -and (Test-Path -LiteralPath $defaultUnslothHome) -and
        -not (Get-ChildItem -LiteralPath $defaultUnslothHome -Force -ErrorAction SilentlyContinue)) {
        _RemovePath $defaultUnslothHome
    }

    # Created at first launch: a leftover EBWebView profile serves a stale frontend.
    _Step "Removing WebView caches and app data (ai.unsloth.studio)..."
    # An unresolvable root counts as incomplete cleanup.
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

    _Step "Removing desktop and Start Menu shortcuts..."
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        if ($desktop) { _RemovePath (Join-Path $desktop "Unsloth Studio.lnk") }
    } catch { }
    if ($env:APPDATA) {
        _RemovePath (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Unsloth Studio.lnk")
    }
    # Invalidate the Win11 Start tile cache (mirrors install.ps1); keep start2.bin, the pin layout.
    try {
        $smehTemp = Join-Path $env:LOCALAPPDATA "Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState"
        if (Test-Path -LiteralPath $smehTemp) {
            Get-ChildItem -LiteralPath $smehTemp -Filter "TileCache_*" -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path $smehTemp "StartUnifiedTileModelCache.dat") -Force -ErrorAction SilentlyContinue
            Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
        }
    } catch { }

    # Re-sweep: unsloth.ico may have been locked by Explorer on the first pass.
    $preservedTemp = @(_RemoveStudioPrivateTempTrees -Paths $privateTempDirs -PrimaryPath $primaryPrivateTemp)
    if ($defaultDataDir -and (Test-Path -LiteralPath $defaultDataDir)) {
        _RemoveDataDirKeepingWslIcon $defaultDataDir -Preserve $preservedTemp
    }

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
                    # Only PATH entries inside a root we own; a substring match would hit user virtualenvs.
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
    # Clear the persisted Inductor cache path when it still names a tree this run deleted:
    # setup.ps1 writes TORCHINDUCTOR_CACHE_DIR to the USER environment, so every later PyTorch
    # process on this account inherits it and compiles into the removed directory. Only a value
    # inside a root this run owned; a user's own directory and the shared C:\tc fallback stay.
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
        Write-Host "Note: this also removed the app's WebView data and the studio.db it found, so"
        Write-Host "      the desktop app's session and the chat history in the install(s) removed"
        Write-Host "      above are gone."
    } else {
        Write-Host "Note: this also removed the app's WebView data, so the desktop app's session"
        Write-Host "      is gone. A browser session is not affected: its tokens live in the same"
        Write-Host "      localStorage as the API keys below."
        if ($script:StudioDbKept) {
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
    if ($uvCaches.Leftovers.Count -gt 0) {
        foreach ($p in $uvCaches.Leftovers) {
            Write-Host "Note: the uv package cache at $p was left in place (it may be shared with other tools)."
            # --cache-dir: a bare `uv cache clean` cleans whichever cache uv resolves now.
            $q = "'" + $p.Replace("'", "''") + "'"
            Write-Host "      Free it with: uv cache clean --cache-dir $q"
        }
    } elseif (-not $uvCaches.Saw) {
        Write-Host 'Note: if install reused a shared uv cache (`uv cache dir`), it was left in place.'
        Write-Host "      Free it with 'uv cache clean'."
    }
    if (-not $env:UNSLOTH_STUDIO_HOME -and -not $env:STUDIO_HOME) {
        Write-Host ""
        Write-Host "If you installed Unsloth Studio with UNSLOTH_STUDIO_HOME or STUDIO_HOME"
        Write-Host "pointing at a custom directory, re-run this script with the same variable"
        Write-Host "set to also remove that install tree, e.g.:"
        Write-Host "  `$env:UNSLOTH_STUDIO_HOME = 'C:\your\path'; irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex"
    }
}

Uninstall-UnslothStudio @args
