# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Unsloth Studio uninstaller for Windows PowerShell.
# Stops running servers and removes install dir, launcher data, CLI shim,
# desktop and Start Menu shortcuts, the user PATH entry, the PathBackup
# registry key, and WebView2 runtime data keyed by the app bundle id
# (EBWebView caches created at first desktop-app launch, not by install.ps1).
# Honors custom roots set via UNSLOTH_STUDIO_HOME / STUDIO_HOME
# at install time (read back from share\studio.conf).
#
# Usage:  irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex
# Local:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\scripts\uninstall.ps1

function Uninstall-UnslothStudio {
    $ErrorActionPreference = "Continue"

    function _Step { param([string]$Msg) Write-Host $Msg }
    function _Substep { param([string]$Msg, [string]$Color = "Gray") Write-Host "  $Msg" -ForegroundColor $Color }

    # Remove a file/dir/symlink if present. Idempotent; retries since a just-killed
    # process can briefly hold a handle (Windows refuses the delete until released).
    function _RemovePath {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (-not (Test-Path -LiteralPath $Path)) { return }
        for ($attempt = 1; $attempt -le 4; $attempt++) {
            try {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            } catch {
                if ($attempt -lt 4) { Start-Sleep -Milliseconds 700; continue }
                _Substep "could not remove: $Path ($($_.Exception.Message))" "Yellow"
                return
            }
            # Remove-Item -Recurse can report success yet leave a transiently-locked
            # child (e.g. unsloth.ico in Explorer's icon cache); verify + retry so we
            # never falsely claim "removed" or orphan the dir.
            if (-not (Test-Path -LiteralPath $Path)) {
                _Substep "removed: $Path" "Green"
                return
            }
            if ($attempt -lt 4) { Start-Sleep -Milliseconds 700; continue }
            _Substep "still present (files held open): $Path" "Yellow"
        }
    }

    # Remove the shared data dir, but keep unsloth.ico if a WSL shortcut still points
    # at it (else that shortcut blanks); uninstall.sh drops it when WSL is removed.
    function _RemoveDataDirKeepingWslIcon {
        param(
            [string]$DataDir,
            # WSL-shortcut search dirs; default Start Menu + Desktop, overridable for tests.
            [string[]]$ShortcutDirs = $null
        )
        if ([string]::IsNullOrWhiteSpace($DataDir)) { return }
        if (-not (Test-Path -LiteralPath $DataDir)) { return }
        # $null = not passed (use defaults); test $null not truthiness so an explicit
        # @() is honored (-not @() is $true).
        if ($null -eq $ShortcutDirs) {
            # Guard $env:APPDATA: it can be unset in service/CI Windows contexts, where
            # an unguarded Join-Path emits a noisy parameter-binding error.
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
        if (@($wslShortcuts).Count -eq 0) {
            _RemovePath $DataDir
            return
        }
        # A WSL shortcut survives: drop everything except its shared icon.
        _Substep "keeping $(Join-Path $DataDir 'unsloth.ico') for the WSL shortcut" "Gray"
        Get-ChildItem -LiteralPath $DataDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.Name -ne "unsloth.ico") { _RemovePath $_.FullName }
        }
    }

    # A path is an Unsloth-owned root iff one of install.ps1's sentinels exists:
    #   <root>\share\studio.conf, <root>\unsloth_studio\.unsloth-studio-owned,
    #   or <root>\bin\unsloth.exe.
    function _IsStudioRoot {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        if (Test-Path -LiteralPath (Join-Path $Path "share\studio.conf") -PathType Leaf) { return $true }
        if (Test-Path -LiteralPath (Join-Path $Path "unsloth_studio\.unsloth-studio-owned") -PathType Leaf) { return $true }
        if (Test-Path -LiteralPath (Join-Path $Path "bin\unsloth.exe") -PathType Leaf) { return $true }
        return $false
    }

    # Hard deny list. Refuse to recursively delete drive roots, USERPROFILE
    # itself, parent of USERPROFILE, or system directories.
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
                $parent = Split-Path -LiteralPath $userProfile -Parent
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
                $bin = Split-Path -LiteralPath $exe -Parent
                $studio = Split-Path -LiteralPath $bin -Parent
                $root = Split-Path -LiteralPath $studio -Parent
                if ($root) { return $root }
            } catch { }
        }
        return $null
    }

    # Expand a leading ~ or ~/ ~\ to $env:USERPROFILE so env-mode roots
    # written with the tilde shape install.ps1 supports (lines 152-154) are
    # found here too.
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

    # Discover non-default Unsloth roots from env vars + studio.conf files.
    # Mirrors install.ps1's precedence: UNSLOTH_STUDIO_HOME wins, STUDIO_HOME
    # is ignored when both are set, so uninstalling install A doesn't also
    # delete install B if the user has a stale STUDIO_HOME pointing at B.
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
        if ($env:UNSLOTH_STUDIO_HOME) {
            $envRoot = $env:UNSLOTH_STUDIO_HOME
        } elseif ($env:STUDIO_HOME) {
            $envRoot = $env:STUDIO_HOME
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

    # Return $true iff the PID's image path lives under one of $KnownRoots.
    # Prevents killing an unrelated process that happens to listen on a stale
    # Unsloth port.
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
            # netstat fallback for older PowerShell. Require LISTENING state so
            # we never kill a process whose remote endpoint just happens to be
            # the cached port (browser -> :443 etc.).
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
        try {
            $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.ExecutablePath -and ($_.ExecutablePath -match '\\unsloth_studio\\.*\\(unsloth|python|studio)\.exe$') -and
                    $_.CommandLine -and ($_.CommandLine -match 'studio')
                }
            foreach ($p in $procs) {
                # Optional scope: only kill if the exe is under a known root.
                if ($KnownRoots) {
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

    # Stop processes that would block deleting the paths we remove. Unlike
    # _StopStudioProcesses (venv exe only), this also catches llama-server/llama-cli,
    # the unsloth.exe shim, and orphaned mp workers under SYSTEM python holding a
    # venv DLL (an open DLL handle blocks the dir delete) -- found by scanning each
    # candidate's loaded modules, not just its image path.
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
            $cands = Get-Process -Name python, pythonw, unsloth, llama-server, llama-cli -ErrorAction SilentlyContinue
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
    $defaultDataDir = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "Unsloth Studio" } else { $null }
    # Default-mode ~/.unsloth holds a SHARED llama.cpp build + .cache that are
    # siblings of studio (not under it), so deleting <studio> misses them -- handle
    # explicitly. No-op in env/custom mode (nested under the custom root, removed
    # with it). A user-set UNSLOTH_LLAMA_CPP_PATH is left alone.
    $defaultUnslothHome = if ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".unsloth" } else { $null }
    $defaultLlamaCpp = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "llama.cpp" } else { $null }
    $defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" } else { $null }
    # Isolated Node.js runtime (install_node_prebuilt.py), a sibling of studio in
    # default mode. No-op in env/custom mode (nested under the custom root) and absent.
    $defaultNode = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome "node" } else { $null }
    # llama.cpp atomic-install staging root (install_llama_prebuilt.py .staging,
    # sibling of the install dir). Usually pruned after activate, but an interrupted
    # build can leave a "<name>.staging-XXXX" tree; removing it lets the empty-dir
    # cleanup of ~/.unsloth below succeed. No-op in env/custom mode and when absent.
    $defaultStaging = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".staging" } else { $null }

    # Build known-root list FIRST so the port-file kill can verify ownership.
    $customRoots = @(_CustomStudioRoots)
    $knownRoots = @()
    if ($defaultStudioHome) { $knownRoots += $defaultStudioHome }
    $knownRoots += $customRoots

    # ── Stop running servers ──
    _Step "Stopping any running Unsloth Studio servers..."
    if ($defaultDataDir) {
        _StopByPortFile -PortFile (Join-Path $defaultDataDir "studio.port") -KnownRoots $knownRoots
    }
    foreach ($r in $customRoots) {
        _StopByPortFile -PortFile (Join-Path $r "share\studio.port") -KnownRoots $knownRoots
    }
    _StopStudioProcesses -KnownRoots $knownRoots
    # Also stop anything holding a handle on the exact paths we delete (llama-server,
    # the CLI shim, an mp-fork python with a venv DLL) so the dir delete isn't refused.
    _StopProcessesLockingRoots -Roots (@($knownRoots) + @($defaultDataDir, $defaultLlamaCpp, $defaultCache, $defaultNode))

    # ── Remove custom-root install trees ──
    _Step "Removing data and install directories..."
    foreach ($r in $customRoots) {
        if (_IsUnsafeRoot $r) {
            _Substep "refusing to remove unsafe path: $r" "Yellow"
            continue
        }
        if (-not (_IsStudioRoot $r)) {
            _Substep "refusing to remove non-Unsloth path: $r" "Yellow"
            continue
        }
        _RemovePath $r
    }
    # Default install dir (always at %USERPROFILE%\.unsloth\studio when present).
    if ($defaultStudioHome) { _RemovePath $defaultStudioHome }
    # Default data dir.
    if ($defaultDataDir) { _RemoveDataDirKeepingWslIcon $defaultDataDir }
    # Default-mode shared llama.cpp build + cache (siblings of studio under
    # ~/.unsloth). No-op in env/custom mode and when absent.
    if ($defaultLlamaCpp) { _RemovePath $defaultLlamaCpp }
    if ($defaultCache) { _RemovePath $defaultCache }
    # Isolated Node.js runtime (sibling of studio under ~/.unsloth). No-op in env/
    # custom mode (nested under the custom root, removed with it) and when absent.
    if ($defaultNode) { _RemovePath $defaultNode }
    if ($defaultStaging) { _RemovePath $defaultStaging }
    # llama.cpp install lock (serializes the shared build); a stray lock keeps
    # ~/.unsloth from being pruned below. No-op in env/custom mode and when absent.
    if ($defaultUnslothHome) { _RemovePath (Join-Path $defaultUnslothHome ".llama.cpp.install.lock") }
    # Drop ~/.unsloth itself, but ONLY if now empty -- never nuke unrelated content.
    if ($defaultUnslothHome -and (Test-Path -LiteralPath $defaultUnslothHome) -and
        -not (Get-ChildItem -LiteralPath $defaultUnslothHome -Force -ErrorAction SilentlyContinue)) {
        _RemovePath $defaultUnslothHome
    }

    # WebView2/app runtime data keyed by the Tauri bundle id. Created at first
    # desktop-app launch, not by install.ps1. LOCALAPPDATA holds the EBWebView
    # profile (a leftover copy serves a stale frontend to the next install);
    # APPDATA holds the app config dir.
    _Step "Removing WebView caches and app data (ai.unsloth.studio)..."
    $bundleId = "ai.unsloth.studio"
    $webviewDataDirs = @()
    if ($env:LOCALAPPDATA) { $webviewDataDirs += Join-Path $env:LOCALAPPDATA $bundleId }
    if ($env:APPDATA) { $webviewDataDirs += Join-Path $env:APPDATA $bundleId }
    # Stop the desktop app and any msedgewebview2.exe helper using these dirs
    # first; WebView2 keeps open handles that make the delete fail.
    try { Stop-Process -Name "unsloth-studio" -Force -ErrorAction SilentlyContinue } catch { }
    try {
        foreach ($proc in (Get-CimInstance Win32_Process -Filter "Name = 'msedgewebview2.exe'" -ErrorAction SilentlyContinue)) {
            $cl = $proc.CommandLine
            if (-not $cl) { continue }
            foreach ($d in $webviewDataDirs) {
                if ($cl -ilike "*$d*") {
                    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
                    break
                }
            }
        }
    } catch { }
    foreach ($d in $webviewDataDirs) { _RemovePath $d }

    # ── Remove desktop and Start Menu shortcuts ──
    _Step "Removing desktop and Start Menu shortcuts..."
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        if ($desktop) { _RemovePath (Join-Path $desktop "Unsloth Studio.lnk") }
    } catch { }
    if ($env:APPDATA) {
        _RemovePath (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Unsloth Studio.lnk")
    }
    # Invalidate the Win11 Start Menu tile cache so the removed shortcut's tile
    # disappears promptly instead of lingering stale (mirrors install.ps1's
    # New-StudioShortcuts). Preserves start2.bin (the pin layout).
    try {
        $smehTemp = Join-Path $env:LOCALAPPDATA "Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState"
        if (Test-Path -LiteralPath $smehTemp) {
            Get-ChildItem -LiteralPath $smehTemp -Filter "TileCache_*" -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path $smehTemp "StartUnifiedTileModelCache.dat") -Force -ErrorAction SilentlyContinue
            Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
        }
    } catch { }

    # Re-sweep: the first pass may have left unsloth.ico locked by Explorer/SMEH for
    # the native shortcut; that handle is now freed. (A surviving WSL shortcut still
    # keeps the icon -- see the helper.)
    if ($defaultDataDir -and (Test-Path -LiteralPath $defaultDataDir)) { _RemoveDataDirKeepingWslIcon $defaultDataDir }

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
                    # Only remove PATH entries that live inside an Unsloth root we
                    # actually own (default or env-mode). A literal substring
                    # match on `unsloth_studio` would clobber unrelated user
                    # virtualenvs that happen to share the name.
                    foreach ($e in $entries) {
                        if ([string]::IsNullOrWhiteSpace($e)) { continue }
                        $expanded = [Environment]::ExpandEnvironmentVariables($e).TrimEnd('\','/')
                        $isStudio = $false
                        foreach ($r in $knownRoots) {
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
    # Remove HKCU\Software\Unsloth (PathBackup lives here; install.ps1 owns it).
    try {
        Remove-Item -LiteralPath 'HKCU:\Software\Unsloth' -Recurse -Force -ErrorAction SilentlyContinue
    } catch { }

    Write-Host ""
    Write-Host "Unsloth Studio uninstalled."
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
