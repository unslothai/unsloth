# Unsloth Studio uninstaller for Windows PowerShell.
# Stops running servers and removes install dir, launcher data, CLI shim,
# desktop and Start Menu shortcuts, the user PATH entry, and the PathBackup
# registry key. Honors custom roots set via UNSLOTH_STUDIO_HOME / STUDIO_HOME
# at install time (read back from share\studio.conf).
#
# Usage:  irm https://raw.githubusercontent.com/unslothai/unsloth/main/uninstall.ps1 | iex
# Local:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\uninstall.ps1

function Uninstall-UnslothStudio {
    $ErrorActionPreference = "Continue"

    function _Step { param([string]$Msg) Write-Host $Msg }
    function _Substep { param([string]$Msg, [string]$Color = "Gray") Write-Host "  $Msg" -ForegroundColor $Color }

    # Remove a file/dir/symlink only if it exists. Idempotent.
    function _RemovePath {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return }
        if (Test-Path -LiteralPath $Path) {
            try {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
                _Substep "removed: $Path" "Green"
            } catch {
                _Substep "could not remove: $Path ($($_.Exception.Message))" "Yellow"
            }
        }
    }

    # A path is a Studio-owned root iff one of install.ps1's sentinels exists:
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

    # Discover non-default Studio roots from env vars + studio.conf files.
    function _CustomStudioRoots {
        $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        $defaultRoot = $null
        if ($env:USERPROFILE) {
            $defaultRoot = (Join-Path $env:USERPROFILE ".unsloth\studio")
        }

        $emit = {
            param($Path)
            if ([string]::IsNullOrWhiteSpace($Path)) { return }
            $norm = $null
            try { $norm = [System.IO.Path]::GetFullPath($Path).TrimEnd('\','/') } catch { return }
            if (-not $norm) { return }
            if ($defaultRoot -and ($norm -ieq $defaultRoot.TrimEnd('\','/'))) { return }
            if ($seen.Add($norm)) { Write-Output $norm }
        }

        if ($env:UNSLOTH_STUDIO_HOME) {
            & $emit $env:UNSLOTH_STUDIO_HOME
            $confRoot = _RootFromConf (Join-Path $env:UNSLOTH_STUDIO_HOME "share\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
        if ($env:STUDIO_HOME) {
            & $emit $env:STUDIO_HOME
            $confRoot = _RootFromConf (Join-Path $env:STUDIO_HOME "share\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
        # Default-mode conf at LOCALAPPDATA\Unsloth Studio.
        if ($env:LOCALAPPDATA) {
            $confRoot = _RootFromConf (Join-Path $env:LOCALAPPDATA "Unsloth Studio\studio.conf")
            if ($confRoot) { & $emit $confRoot }
        }
    }

    # Stop a Studio backend whose port is recorded in <DataDir>\studio.port.
    function _StopByPortFile {
        param([string]$PortFile)
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
                try {
                    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
                } catch { }
            }
        } catch {
            # Fallback for PS versions without Get-NetTCPConnection.
            try {
                $lines = & netstat.exe -ano 2>$null | Select-String -Pattern (":$port\s")
                foreach ($l in $lines) {
                    $parts = ($l.ToString() -split '\s+') | Where-Object { $_ }
                    $pid_ = $parts[-1]
                    if ($pid_ -match '^\d+$') {
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

    # Default install root + default data dir.
    $defaultStudioHome = if ($env:USERPROFILE) { Join-Path $env:USERPROFILE ".unsloth\studio" } else { $null }
    $defaultDataDir = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "Unsloth Studio" } else { $null }

    # ── Stop running servers ──
    _Step "Stopping any running Unsloth Studio servers..."
    # Default-mode port file.
    if ($defaultDataDir) {
        _StopByPortFile (Join-Path $defaultDataDir "studio.port")
    }
    # Custom-root port files (env-mode <root>\share\studio.port).
    $customRoots = @(_CustomStudioRoots)
    foreach ($r in $customRoots) {
        _StopByPortFile (Join-Path $r "share\studio.port")
    }
    # Process-name sweep as belt-and-suspenders.
    $knownRoots = @()
    if ($defaultStudioHome) { $knownRoots += $defaultStudioHome }
    $knownRoots += $customRoots
    _StopStudioProcesses -KnownRoots $knownRoots

    # ── Remove custom-root install trees ──
    _Step "Removing data and install directories..."
    foreach ($r in $customRoots) {
        if (_IsUnsafeRoot $r) {
            _Substep "refusing to remove unsafe path: $r" "Yellow"
            continue
        }
        if (-not (_IsStudioRoot $r)) {
            _Substep "refusing to remove non-Studio path: $r" "Yellow"
            continue
        }
        _RemovePath $r
    }
    # Default install dir (always at %USERPROFILE%\.unsloth\studio when present).
    if ($defaultStudioHome) { _RemovePath $defaultStudioHome }
    # Default data dir.
    if ($defaultDataDir) { _RemovePath $defaultDataDir }

    # ── Remove desktop and Start Menu shortcuts ──
    _Step "Removing desktop and Start Menu shortcuts..."
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        if ($desktop) { _RemovePath (Join-Path $desktop "Unsloth Studio.lnk") }
    } catch { }
    if ($env:APPDATA) {
        _RemovePath (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Unsloth Studio.lnk")
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
                    foreach ($e in $entries) {
                        if ([string]::IsNullOrWhiteSpace($e)) { continue }
                        # Drop any PATH entry under .unsloth\studio or an unsloth_studio venv.
                        $expanded = [Environment]::ExpandEnvironmentVariables($e)
                        if ($expanded -match '\\\.unsloth\\studio($|\\)' -or $expanded -match '\\unsloth_studio\\') {
                            _Substep "removed PATH entry: $e" "Green"
                            $removedAny = $true
                            continue
                        }
                        # Also drop a custom-root\bin we know about.
                        $isCustom = $false
                        foreach ($r in $customRoots) {
                            $rBin = (Join-Path $r "bin").TrimEnd('\','/')
                            if ($expanded.TrimEnd('\','/') -ieq $rBin) { $isCustom = $true; break }
                        }
                        if ($isCustom) {
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
        Write-Host "  `$env:UNSLOTH_STUDIO_HOME = 'C:\your\path'; .\uninstall.ps1"
    }
}

Uninstall-UnslothStudio @args
