# Unsloth Studio Installer for Windows PowerShell
# Usage:  irm https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1 | iex
# Local:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\install.ps1 --local
# NoTorch: .\install.ps1 --no-torch  (skip PyTorch, GGUF-only mode)
# Test:   .\install.ps1 --package roland-sloth

function Install-UnslothStudio {
    $ErrorActionPreference = "Stop"
    $script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq "1")

    # ── Parse flags ──
    $StudioLocalInstall = $false
    $PackageName = "unsloth"
    $RepoRoot = ""
    $TauriMode = $false
    $SkipTorch = $false
    $argList = $args
    for ($i = 0; $i -lt $argList.Count; $i++) {
        switch ($argList[$i]) {
            "--local"    { $StudioLocalInstall = $true }
            "--tauri"    { $TauriMode = $true }
            "--no-torch" { $SkipTorch = $true }
            "--verbose"  { $script:UnslothVerbose = $true }
            "-v"         { $script:UnslothVerbose = $true }
            "--package"  {
                $i++
                if ($i -ge $argList.Count) {
                    Write-Host "[ERROR] --package requires an argument." -ForegroundColor Red
                    return
                }
                $PackageName = $argList[$i]
            }
        }
    }
    # Propagate to child processes so they also respect verbose mode.
    # Process-scoped -- does not persist.
    if ($script:UnslothVerbose) {
        $env:UNSLOTH_VERBOSE = '1'
    }

    if ($StudioLocalInstall) {
        $RepoRoot = (Resolve-Path (Split-Path -Parent $PSCommandPath)).Path
        if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
            Write-Host "[ERROR] --local must be run from the unsloth repo root (pyproject.toml not found at $RepoRoot)" -ForegroundColor Red
            return
        }
    }

    # Validate --package to prevent injection into shell/Python commands
    if ($PackageName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
        Write-Host "[ERROR] --package name contains invalid characters (allowed: a-z A-Z 0-9 . _ -)" -ForegroundColor Red
        return
    }

    # ── Tauri structured output ──
    function Write-TauriLog {
        param([string]$Tag, [string]$Message)
        if ($TauriMode) {
            Write-Host "[TAURI:$Tag] $Message"
        }
    }

    $PythonVersion = "3.13"
    $StudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
    $VenvDir = Join-Path $StudioHome "unsloth_studio"

    $Rule = [string]::new([char]0x2500, 52)
    $Sloth = [char]::ConvertFromUtf32(0x1F9A5)

    function Enable-StudioVirtualTerminal {
        if ($env:NO_COLOR) { return $false }
        try {
            if (-not ("StudioVT.Native" -as [type])) {
                Add-Type -Namespace StudioVT -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
'@ -ErrorAction Stop
            }
            $h = [StudioVT.Native]::GetStdHandle(-11)
            [uint32]$mode = 0
            if (-not [StudioVT.Native]::GetConsoleMode($h, [ref]$mode)) { return $false }
            $mode = $mode -bor 0x0004
            return [StudioVT.Native]::SetConsoleMode($h, $mode)
        } catch {
            return $false
        }
    }
    $script:StudioVtOk = Enable-StudioVirtualTerminal

    function Get-StudioAnsi {
        param(
            [Parameter(Mandatory = $true)]
            [ValidateSet('Title', 'Dim', 'Ok', 'Warn', 'Err', 'Reset')]
            [string]$Kind
        )
        $e = [char]27
        switch ($Kind) {
            'Title' { return "${e}[38;5;150m" }
            'Dim'   { return "${e}[38;5;245m" }
            'Ok'    { return "${e}[38;5;108m" }
            'Warn'  { return "${e}[38;5;136m" }
            'Err'   { return "${e}[91m" }
            'Reset' { return "${e}[0m" }
        }
    }

    Write-Host ""
    if ($script:StudioVtOk -and -not $env:NO_COLOR) {
        Write-Host ("  " + (Get-StudioAnsi Title) + $Sloth + " Unsloth Studio Installer (Windows)" + (Get-StudioAnsi Reset))
        Write-Host ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
    } else {
        Write-Host ("  {0} Unsloth Studio Installer (Windows)" -f $Sloth) -ForegroundColor DarkGreen
        Write-Host "  $Rule" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ── Helper: refresh PATH from registry (deduplicating entries) ──
    # Merge order: venv Scripts (if active) > Machine > User > current $env:Path.
    # Dedup compares both raw and expanded forms (%VAR% vs literal).
    function Refresh-SessionPath {
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $venvScripts = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts" } else { $null }
        $sources = @()
        if ($venvScripts) { $sources += $venvScripts }
        $sources += @($machine, $user, $env:Path)
        $merged = ($sources | Where-Object { $_ }) -join ";"
        $seen    = @{}
        $unique  = New-Object System.Collections.Generic.List[string]
        foreach ($p in $merged -split ";") {
            $rawKey = $p.Trim().Trim('"').TrimEnd("\").ToLowerInvariant()
            $expKey = [Environment]::ExpandEnvironmentVariables($p).Trim().Trim('"').TrimEnd("\").ToLowerInvariant()
            if ($rawKey -and -not $seen.ContainsKey($rawKey) -and -not $seen.ContainsKey($expKey)) {
                $seen[$rawKey] = $true
                if ($expKey -and $expKey -ne $rawKey) { $seen[$expKey] = $true }
                $unique.Add($p)
            }
        }
        $env:Path = $unique -join ";"
    }

    # ── Helper: safely add a directory to the persistent User PATH ──
    # Direct registry access preserves REG_EXPAND_SZ (avoids dotnet/runtime#1442).
    # Append (default) keeps existing tools first; Prepend for must-win entries.
    function Add-ToUserPath {
        param(
            [Parameter(Mandatory = $true)][string]$Directory,
            [ValidateSet('Append','Prepend')]
            [string]$Position = 'Append'
        )
        try {
            $regKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
            try {
                $rawPath = $regKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
                [string[]]$entries = if ($rawPath) { $rawPath -split ';' } else { @() } # string[] prevents scalar collapse
                $normalDir = $Directory.Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
                $expNormalDir = [Environment]::ExpandEnvironmentVariables($Directory).Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
                $kept = New-Object System.Collections.Generic.List[string]
                $matchIndices = New-Object System.Collections.Generic.List[int]
                for ($i = 0; $i -lt $entries.Count; $i++) {
                    $stripped = $entries[$i].Trim().Trim('"')
                    $rawNorm = $stripped.TrimEnd('\').ToLowerInvariant()
                    $expNorm = [Environment]::ExpandEnvironmentVariables($stripped).TrimEnd('\').ToLowerInvariant()
                    $isMatch = ($rawNorm -and ($rawNorm -eq $normalDir -or $rawNorm -eq $expNormalDir)) -or
                               ($expNorm -and ($expNorm -eq $normalDir -or $expNorm -eq $expNormalDir))
                    if ($isMatch) {
                        $matchIndices.Add($i)
                        continue
                    }
                    $kept.Add($entries[$i])
                }
                $alreadyPresent = $matchIndices.Count -gt 0
                if ($alreadyPresent -and $Position -eq 'Append') { # Append: idempotent no-op
                    return $false
                }
                if ($alreadyPresent -and $Position -eq 'Prepend' -and # Prepend: no-op if already at front
                    $matchIndices.Count -eq 1 -and $matchIndices[0] -eq 0) {
                    return $false
                }
                # One-time backup under HKCU\Software\Unsloth\PathBackup
                if ($rawPath) {
                    try {
                        $backupKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Unsloth')
                        try {
                            $existingBackup = $backupKey.GetValue('PathBackup', $null)
                            if (-not $existingBackup) {
                                $backupKey.SetValue('PathBackup', $rawPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                            }
                        } finally {
                            $backupKey.Close()
                        }
                    } catch { }
                }
                if (-not $rawPath) {
                    Write-Host "[WARN] User PATH is empty - initializing with $Directory" -ForegroundColor Yellow
                }
                $newPath = if ($rawPath) {
                    if ($Position -eq 'Prepend') {
                        (@($Directory) + $kept) -join ';'
                    } else {
                        ($kept + @($Directory)) -join ';'
                    }
                } else {
                    $Directory
                }
                if ($newPath -ceq $rawPath) { # no actual change
                    return $false
                }
                $regKey.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                # Broadcast WM_SETTINGCHANGE via dummy env-var roundtrip.
                # [NullString]::Value avoids PS 7.5+/.NET 9 $null-to-"" coercion.
                try {
                    $d = "UnslothPathRefresh_$([guid]::NewGuid().ToString('N').Substring(0,8))"
                    [Environment]::SetEnvironmentVariable($d, '1', 'User')
                    [Environment]::SetEnvironmentVariable($d, [NullString]::Value, 'User')
                } catch { }
                return $true
            } finally {
                $regKey.Close()
            }
        } catch {
            Write-Host "[WARN] Could not update User PATH: $($_.Exception.Message)" -ForegroundColor Yellow
            return $false
        }
    }

    function step {
        param(
            [Parameter(Mandatory = $true)][string]$Label,
            [Parameter(Mandatory = $true)][string]$Value,
            [string]$Color = "Green"
        )
        if ($script:StudioVtOk -and -not $env:NO_COLOR) {
            $dim = Get-StudioAnsi Dim
            $rst = Get-StudioAnsi Reset
            $val = switch ($Color) {
                'Green' { Get-StudioAnsi Ok }
                'Yellow' { Get-StudioAnsi Warn }
                'Red' { Get-StudioAnsi Err }
                'DarkGray' { Get-StudioAnsi Dim }
                default { Get-StudioAnsi Ok }
            }
            $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
            Write-Host ("  {0}{1}{2}{3}{4}{2}" -f $dim, $padded, $rst, $val, $Value)
        } else {
            $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
            Write-Host ("  {0}" -f $padded) -NoNewline -ForegroundColor DarkGray
            $fc = switch ($Color) {
                'Green' { 'DarkGreen' }
                'Yellow' { 'Yellow' }
                'Red' { 'Red' }
                'DarkGray' { 'DarkGray' }
                default { 'DarkGreen' }
            }
            Write-Host $Value -ForegroundColor $fc
        }
    }

    function substep {
        param(
            [Parameter(Mandatory = $true)][string]$Message,
            [string]$Color = "DarkGray"
        )
        if ($script:StudioVtOk -and -not $env:NO_COLOR) {
            $msgCol = switch ($Color) {
                'Yellow' { (Get-StudioAnsi Warn) }
                'Red' { (Get-StudioAnsi Err) }
                default { (Get-StudioAnsi Dim) }
            }
            $pad = "".PadRight(15)
            Write-Host ("  {0}{1}{2}{3}" -f $msgCol, $pad, $Message, (Get-StudioAnsi Reset))
        } else {
            $fc = switch ($Color) {
                'Yellow' { 'Yellow' }
                'Red' { 'Red' }
                default { 'DarkGray' }
            }
            Write-Host ("  {0,-15}{1}" -f "", $Message) -ForegroundColor $fc
        }
    }

    # Run native commands quietly by default to match install.sh behavior.
    # Full command output is shown only when --verbose / UNSLOTH_VERBOSE=1.
    function Invoke-InstallCommand {
        param(
            [Parameter(Mandatory = $true)][ScriptBlock]$Command
        )
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # Reset to avoid stale values from prior native commands.
            $global:LASTEXITCODE = 0
            if ($script:UnslothVerbose) {
                # Merge stderr into stdout so progress/warning output stays visible
                # without flipping $? on successful native commands (PS 5.1 treats
                # stderr records as errors that set $? = $false even on exit code 0).
                & $Command 2>&1 | Out-Host
            } else {
                $output = & $Command 2>&1 | Out-String
                if ($LASTEXITCODE -ne 0) {
                    Write-Host $output -ForegroundColor Red
                }
            }
            return [int]$LASTEXITCODE
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }

    function New-StudioShortcuts {
        param(
            [Parameter(Mandatory = $true)][string]$UnslothExePath
        )

        if (-not (Test-Path $UnslothExePath)) {
            substep "cannot create shortcuts, unsloth.exe not found at $UnslothExePath" "Yellow"
            return
        }
        try {
            # Persist an absolute path in launcher scripts so shortcut working
            # directory changes do not break process startup.
            $UnslothExePath = (Resolve-Path $UnslothExePath).Path
            # Escape for single-quoted embedding in generated launcher script.
            # This prevents runtime variable expansion for paths containing '$'.
            $SingleQuotedExePath = $UnslothExePath -replace "'", "''"

            $localAppDataDir = $env:LOCALAPPDATA
            if (-not $localAppDataDir -or [string]::IsNullOrWhiteSpace($localAppDataDir)) {
                substep "LOCALAPPDATA path unavailable; skipped shortcut creation" "Yellow"
                return
            }
            $appDir = Join-Path $localAppDataDir "Unsloth Studio"
            $launcherPs1 = Join-Path $appDir "launch-studio.ps1"
            $launcherVbs = Join-Path $appDir "launch-studio.vbs"
            $desktopDir = [Environment]::GetFolderPath("Desktop")
            $desktopLink = if ($desktopDir -and $desktopDir.Trim()) {
                Join-Path $desktopDir "Unsloth Studio.lnk"
            } else {
                $null
            }
            $startMenuDir = if ($env:APPDATA -and $env:APPDATA.Trim()) {
                Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
            } else {
                $null
            }
            $startMenuLink = if ($startMenuDir -and $startMenuDir.Trim()) {
                Join-Path $startMenuDir "Unsloth Studio.lnk"
            } else {
                $null
            }
            if (-not $desktopLink) {
                substep "Desktop path unavailable; skipped desktop shortcut creation" "Yellow"
            }
            if (-not $startMenuLink) {
                substep "APPDATA/Start Menu path unavailable; skipped Start menu shortcut creation" "Yellow"
            }
            $iconPath = Join-Path $appDir "unsloth.ico"
            $bundledIcon = $null
            if ($PSScriptRoot -and $PSScriptRoot.Trim()) {
                $bundledIcon = Join-Path $PSScriptRoot "studio\frontend\public\unsloth.ico"
            }
            $iconUrl = "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth.ico"

            if (-not (Test-Path $appDir)) {
                New-Item -ItemType Directory -Path $appDir -Force | Out-Null
            }

            $launcherContent = @"
`$ErrorActionPreference = 'Stop'
`$basePort = 8888
`$maxPortOffset = 20
`$timeoutSec = 60
`$pollIntervalMs = 1000

function Test-StudioHealth {
    param([Parameter(Mandatory = `$true)][int]`$Port)
    try {
        `$url = "http://127.0.0.1:`$Port/api/health"
        `$resp = Invoke-RestMethod -Uri `$url -TimeoutSec 1 -Method Get
        return (`$resp -and `$resp.status -eq 'healthy' -and `$resp.service -eq 'Unsloth UI Backend')
    } catch {
        return `$false
    }
}

function Get-CandidatePorts {
    # Fast path: only probe base port + currently listening ports in range.
    `$ports = @(`$basePort)
    try {
        `$maxPort = `$basePort + `$maxPortOffset
        `$listening = Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { `$_.LocalPort -ge `$basePort -and `$_.LocalPort -le `$maxPort } |
            Select-Object -ExpandProperty LocalPort
        `$ports = (@(`$basePort) + `$listening) | Sort-Object -Unique
    } catch {
        Write-Host "[DEBUG] Get-NetTCPConnection failed: `$(`$_.Exception.Message). Falling back to full port scan." -ForegroundColor DarkGray
        # Fallback when Get-NetTCPConnection is unavailable/restricted.
        for (`$offset = 1; `$offset -le `$maxPortOffset; `$offset++) {
            `$ports += (`$basePort + `$offset)
        }
    }
    return `$ports
}

function Find-HealthyStudioPort {
    foreach (`$candidate in (Get-CandidatePorts)) {
        if (Test-StudioHealth -Port `$candidate) {
            return `$candidate
        }
    }
    return `$null
}

function Test-PortBusy {
    param([Parameter(Mandatory = `$true)][int]`$Port)
    `$listener = `$null
    try {
        `$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, `$Port)
        `$listener.Start()
        return `$false
    } catch {
        return `$true
    } finally {
        if (`$listener) { try { `$listener.Stop() } catch {} }
    }
}

function Find-FreeLaunchPort {
    `$maxPort = `$basePort + `$maxPortOffset
    try {
        `$listening = Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { `$_.LocalPort -ge `$basePort -and `$_.LocalPort -le `$maxPort } |
            Select-Object -ExpandProperty LocalPort
        for (`$offset = 0; `$offset -le `$maxPortOffset; `$offset++) {
            `$candidate = `$basePort + `$offset
            if (`$candidate -notin `$listening) {
                return `$candidate
            }
        }
    } catch {
        # Get-NetTCPConnection unavailable or restricted; probe ports directly
        for (`$offset = 0; `$offset -le `$maxPortOffset; `$offset++) {
            `$candidate = `$basePort + `$offset
            if (-not (Test-PortBusy -Port `$candidate)) {
                return `$candidate
            }
        }
    }
    return `$null
}

# If Studio is already healthy on any expected port, just open it and exit.
`$existingPort = Find-HealthyStudioPort
if (`$existingPort) {
    Start-Process "http://localhost:`$existingPort"
    exit 0
}

`$launchMutex = [System.Threading.Mutex]::new(`$false, 'Local\UnslothStudioLauncher')
`$haveMutex = `$false
try {
    try {
        `$haveMutex = `$launchMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        `$haveMutex = `$true
    }
    if (-not `$haveMutex) {
        # Another launcher is already running; wait for it to bring Studio up
        `$deadline = (Get-Date).AddSeconds(`$timeoutSec)
        while ((Get-Date) -lt `$deadline) {
            `$port = Find-HealthyStudioPort
            if (`$port) { Start-Process "http://localhost:`$port"; exit 0 }
            Start-Sleep -Milliseconds `$pollIntervalMs
        }
        exit 0
    }

    `$powershellExe = Join-Path `$env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    `$studioExe = '$SingleQuotedExePath'
    `$launchPort = Find-FreeLaunchPort
    if (-not `$launchPort) {
        `$msg = "No free port found in range `$basePort-`$(`$basePort + `$maxPortOffset)"
        try {
            Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
            [System.Windows.Forms.MessageBox]::Show(`$msg, 'Unsloth Studio') | Out-Null
        } catch {}
        exit 1
    }
    `$studioCommand = '& "' + `$studioExe + '" studio -H 0.0.0.0 -p ' + `$launchPort
    `$launchArgs = @(
        '-NoExit',
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        `$studioCommand
    )

    try {
        `$proc = Start-Process -FilePath `$powershellExe -ArgumentList `$launchArgs -WorkingDirectory `$env:USERPROFILE -PassThru
    } catch {
        `$msg = "Could not launch Unsloth Studio terminal.`n`nError: `$(`$_.Exception.Message)"
        try {
            Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
            [System.Windows.Forms.MessageBox]::Show(`$msg, 'Unsloth Studio') | Out-Null
        } catch {}
        exit 1
    }

    `$browserOpened = `$false
    `$deadline = (Get-Date).AddSeconds(`$timeoutSec)
    while ((Get-Date) -lt `$deadline) {
        `$healthyPort = Find-HealthyStudioPort
        if (`$healthyPort) {
            Start-Process "http://localhost:`$healthyPort"
            `$browserOpened = `$true
            break
        }
        if (`$proc.HasExited) { break }
        Start-Sleep -Milliseconds `$pollIntervalMs
    }
    if (-not `$browserOpened) {
        if (`$proc.HasExited) {
            `$msg = "Unsloth Studio exited before becoming healthy. Check terminal output for errors."
        } else {
            `$msg = "Unsloth Studio is still starting but did not become healthy within `$timeoutSec seconds. Check the terminal window for the selected port and open it manually."
        }
        try {
            Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
            [System.Windows.Forms.MessageBox]::Show(`$msg, 'Unsloth Studio') | Out-Null
        } catch {}
    }
} finally {
    if (`$haveMutex) { `$launchMutex.ReleaseMutex() | Out-Null }
    `$launchMutex.Dispose()
}
exit 0
"@

            # Write UTF-8 with BOM for reliable decoding by Windows PowerShell 5.1,
            # even when install.ps1 is executed from PowerShell 7.
            $utf8Bom = New-Object System.Text.UTF8Encoding($true)
            [System.IO.File]::WriteAllText($launcherPs1, $launcherContent, $utf8Bom)
            $vbsContent = @"
Set shell = CreateObject("WScript.Shell")
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""$launcherPs1"""
shell.Run cmd, 0, False
"@
            # WSH handles UTF-16LE reliably for .vbs files with non-ASCII paths.
            Set-Content -Path $launcherVbs -Value $vbsContent -Encoding Unicode -Force

            # Prefer bundled icon from local clone/dev installs.
            # If not available, best-effort download from raw GitHub.
            # We only attach the icon if the resulting file has a valid ICO header.
            $hasValidIcon = $false
            if ($bundledIcon -and (Test-Path $bundledIcon)) {
                try {
                    Copy-Item -Path $bundledIcon -Destination $iconPath -Force
                } catch {
                    Write-Host "[DEBUG] Error copying bundled icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                }
            } elseif (-not (Test-Path $iconPath)) {
                try {
                    Invoke-WebRequest -Uri $iconUrl -OutFile $iconPath -UseBasicParsing
                } catch {
                    Write-Host "[DEBUG] Error downloading icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                }
            }

            if (Test-Path $iconPath) {
                try {
                    $bytes = [System.IO.File]::ReadAllBytes($iconPath)
                    if (
                        $bytes.Length -ge 4 -and
                        $bytes[0] -eq 0 -and
                        $bytes[1] -eq 0 -and
                        $bytes[2] -eq 1 -and
                        $bytes[3] -eq 0
                    ) {
                        $hasValidIcon = $true
                    } else {
                        Remove-Item $iconPath -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    Write-Host "[DEBUG] Error validating or removing icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                    Remove-Item $iconPath -Force -ErrorAction SilentlyContinue
                }
            }

            $wscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
            $shortcutArgs = "//B //Nologo `"$launcherVbs`""

            try {
                $wshell = New-Object -ComObject WScript.Shell
                $createdShortcutCount = 0
                foreach ($linkPath in @($desktopLink, $startMenuLink)) {
                    if (-not $linkPath -or [string]::IsNullOrWhiteSpace($linkPath)) { continue }
                    try {
                        $shortcut = $wshell.CreateShortcut($linkPath)
                        $shortcut.TargetPath = $wscriptExe
                        $shortcut.Arguments = $shortcutArgs
                        $shortcut.WorkingDirectory = $appDir
                        $shortcut.Description = "Launch Unsloth Studio"
                        if ($hasValidIcon) {
                            $shortcut.IconLocation = "$iconPath,0"
                        }
                        $shortcut.Save()
                        $createdShortcutCount++
                    } catch {
                        substep "could not create shortcut at ${linkPath}: $($_.Exception.Message)" "Yellow"
                    }
                }
                if ($createdShortcutCount -gt 0) {
                    substep "Created Unsloth Studio shortcut"
                } else {
                    substep "no Unsloth Studio shortcuts were created" "Yellow"
                }
            } catch {
                substep "shortcut creation unavailable: $($_.Exception.Message)" "Yellow"
            }
        } catch {
            substep "shortcut setup failed; skipping shortcuts: $($_.Exception.Message)" "Yellow"
        }
    }

    # ── Check winget ──
    Write-TauriLog "STEP" "Checking system dependencies"
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        step "winget" "not available" "Red"
        substep "Install it from https://aka.ms/getwinget" "Yellow"
        substep "or install Python $PythonVersion and uv manually, then re-run." "Yellow"
        return
    }

    # ── Helper: detect a working Python 3.11-3.13 on the system ──
    # Returns the version string (e.g. "3.13") or "" if none found.
    # Uses try-catch + stderr redirection so that App Execution Alias stubs
    # (WindowsApps) and other non-functional executables are probed safely
    # without triggering $ErrorActionPreference = "Stop".
    #
    # Skips Anaconda/Miniconda Python: conda-bundled CPython ships modified
    # DLL search paths that break torch's c10.dll loading on Windows.
    # Standalone CPython (python.org, winget, uv) does not have this issue.
    #
    # NOTE: A venv created from conda Python inherits conda's base_prefix
    # even if the venv path does not contain "conda". We check both the
    # executable path AND sys.base_prefix to catch this.
    $script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'

    function Test-IsCondaPython {
        param([string]$Exe)
        if ($Exe -match $script:CondaSkipPattern) { return $true }
        try {
            $basePrefix = (& $Exe -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
            if ($basePrefix -match $script:CondaSkipPattern) { return $true }
        } catch { }
        return $false
    }

    # Returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    # The resolved Path is passed to `uv venv --python` to prevent uv from
    # re-resolving the version string back to a conda interpreter.
    function Find-CompatiblePython {
        # Try the Python Launcher first (most reliable on Windows)
        # py.exe resolves to the standard CPython install, not conda.
        $pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if ($pyLauncher -and $pyLauncher.Source -notmatch $script:CondaSkipPattern) {
            foreach ($minor in @("3.13", "3.12", "3.11")) {
                try {
                    $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") {
                        $ver = $Matches[1]
                        # Resolve the actual executable path and verify it is not conda-based
                        $resolvedExe = (& $pyLauncher.Source "-$minor" -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
                        if ($resolvedExe -and (Test-Path $resolvedExe) -and -not (Test-IsCondaPython $resolvedExe)) {
                            return @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # Try python3 / python via Get-Command -All to look past stubs that
        # might shadow a real Python further down PATH.
        # Skip WindowsApps entries: the App Execution Alias stubs live there
        # and can open the Microsoft Store as a side effect. Legitimate Store
        # Python is already detected via the py launcher above (Store packages
        # include py since Python 3.11).
        # Skip Anaconda/Miniconda: check both path and sys.base_prefix.
        foreach ($name in @("python3", "python")) {
            foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
                if (-not $cmd.Source) { continue }
                if ($cmd.Source -like "*\WindowsApps\*") { continue }
                if (Test-IsCondaPython $cmd.Source) { continue }
                try {
                    $out = & $cmd.Source --version 2>&1 | Out-String
                    if ($out -match "Python (3\.1[1-3])\.\d+") {
                        return @{ Version = $Matches[1]; Path = $cmd.Source }
                    }
                } catch {}
            }
        }
        return $null
    }

    # ── Install Python if no compatible version (3.11-3.13) found ──
    # Find-CompatiblePython returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    Write-TauriLog "STEP" "Installing Python"
    $DetectedPython = Find-CompatiblePython
    if ($DetectedPython) {
        step "python" "Python $($DetectedPython.Version) already installed"
    }
    if (-not $DetectedPython) {
        substep "installing Python ${PythonVersion}..."
        $pythonPackageId = "Python.Python.$PythonVersion"
        # Temporarily lower ErrorActionPreference so that winget stderr
        # (progress bars, warnings) does not become a terminating error
        # on PowerShell 5.1 where native-command stderr is ErrorRecord.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            winget install -e --id $pythonPackageId --accept-package-agreements --accept-source-agreements
            $wingetExit = $LASTEXITCODE
        } catch { $wingetExit = 1 }
        $ErrorActionPreference = $prevEAP
        Refresh-SessionPath

        # Re-detect after install (PATH may have changed)
        $DetectedPython = Find-CompatiblePython

        if (-not $DetectedPython) {
            # Python still not functional after winget -- force reinstall.
            # This handles both real failures AND "already installed" codes where
            # winget thinks Python is present but it's not actually on PATH
            # (e.g. user partially uninstalled, or installed via a different method).
            substep "Python not found on PATH after winget. Retrying with --force..." "Yellow"
            $ErrorActionPreference = "Continue"
            try {
                winget install -e --id $pythonPackageId --accept-package-agreements --accept-source-agreements --force
                $wingetExit = $LASTEXITCODE
            } catch { $wingetExit = 1 }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath
            $DetectedPython = Find-CompatiblePython
        }

        if (-not $DetectedPython) {
            Write-Host "[ERROR] Python installation failed (exit code $wingetExit)" -ForegroundColor Red
            Write-Host "        Please install Python $PythonVersion manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-Host "        Then re-run this installer." -ForegroundColor Yellow
            return
        }
    }

    # ── Install uv if not present ──
    Write-TauriLog "STEP" "Installing uv package manager"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        substep "installing uv package manager..."
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { winget install --id=astral-sh.uv -e --accept-package-agreements --accept-source-agreements } catch {}
        $ErrorActionPreference = $prevEAP
        Refresh-SessionPath
        # Fallback: if winget didn't put uv on PATH, try the PowerShell installer
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            substep "trying alternative uv installer..." "Yellow"
            Invoke-Expression (Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1")
            Refresh-SessionPath
        }
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        step "uv" "could not be installed" "Red"
        substep "Install it from https://docs.astral.sh/uv/" "Yellow"
        return
    }

    # ── Create venv (migrate old layout if possible, otherwise fresh) ──
    # Pass the resolved executable path to uv so it does not re-resolve
    # a version string back to a conda interpreter.
    Write-TauriLog "STEP" "Creating virtual environment"
    if (-not (Test-Path $StudioHome)) {
        New-Item -ItemType Directory -Path $StudioHome -Force | Out-Null
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $_Migrated = $false

    if (Test-Path $VenvPython) {
        # New layout already exists -- nuke for fresh install
        substep "removing existing environment for fresh install..."
        Remove-Item -Recurse -Force $VenvDir
    } elseif (Test-Path (Join-Path $StudioHome ".venv\Scripts\python.exe")) {
        # Old layout (~/.unsloth/studio/.venv) exists -- validate before migrating
        $OldVenv = Join-Path $StudioHome ".venv"
        $OldPy = Join-Path $OldVenv "Scripts\python.exe"
        substep "found legacy Studio environment, validating..."
        $prevEAP2 = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $OldPy -c "import torch; A = torch.ones((2,2)); B = A + A" 2>$null | Out-Null
            $torchOk = ($LASTEXITCODE -eq 0)
        } catch { $torchOk = $false }
        $ErrorActionPreference = $prevEAP2
        if ($torchOk) {
            substep "legacy environment is healthy -- migrating..."
            Move-Item -Path $OldVenv -Destination $VenvDir -Force
            substep "moved .venv -> unsloth_studio"
            $_Migrated = $true
        } else {
            substep "legacy environment failed validation -- creating fresh environment" "Yellow"
            Remove-Item -Recurse -Force $OldVenv -ErrorAction SilentlyContinue
        }
    } elseif (Test-Path (Join-Path $env:USERPROFILE "unsloth_studio\Scripts\python.exe")) {
        # CWD-relative venv from old install.ps1 -- migrate to absolute path
        $CwdVenv = Join-Path $env:USERPROFILE "unsloth_studio"
        substep "found CWD-relative Studio environment, migrating to $VenvDir..."
        Move-Item -Path $CwdVenv -Destination $VenvDir -Force
        substep "moved ~/unsloth_studio -> ~/.unsloth/studio/unsloth_studio"
        $_Migrated = $true
    }

    if (-not (Test-Path $VenvPython)) {
        step "venv" "creating Python $($DetectedPython.Version) virtual environment"
        substep "$VenvDir"
        $venvExit = Invoke-InstallCommand { uv venv $VenvDir --python "$($DetectedPython.Path)" }
        if ($venvExit -ne 0) {
            Write-TauriLog "ERROR" "Failed to create virtual environment (exit code $venvExit)"
            Write-Host "[ERROR] Failed to create virtual environment (exit code $venvExit)" -ForegroundColor Red
            return
        }
    } else {
        step "venv" "using migrated environment"
        substep "$VenvDir"
    }

    # ── Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) ──
    $HasNvidiaSmi = $false
    $NvidiaSmiExe = $null
    try {
        $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvSmiCmd) {
            & $nvSmiCmd.Source *> $null
            if ($LASTEXITCODE -eq 0) { $HasNvidiaSmi = $true; $NvidiaSmiExe = $nvSmiCmd.Source }
        }
    } catch {}
    if (-not $HasNvidiaSmi) {
        foreach ($p in @(
            "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "$env:SystemRoot\System32\nvidia-smi.exe"
        )) {
            if (Test-Path $p) {
                try {
                    & $p *> $null
                    if ($LASTEXITCODE -eq 0) { $HasNvidiaSmi = $true; $NvidiaSmiExe = $p; break }
                } catch {}
            }
        }
    }
    if ($HasNvidiaSmi) {
        step "gpu" "NVIDIA GPU detected"
    } else {
        step "gpu" "none (chat-only / GGUF)" "Yellow"
        substep "Training and GPU inference require an NVIDIA GPU with drivers installed." "Yellow"
    }

    # ── Choose the correct PyTorch index URL based on driver CUDA version ──
    # Mirrors Get-PytorchCudaTag in setup.ps1.
    function Get-TorchIndexUrl {
        $baseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        if (-not $NvidiaSmiExe) { return "$baseUrl/cpu" }
        try {
            $output = & $NvidiaSmiExe 2>&1 | Out-String
            if ($output -match 'CUDA Version:\s+(\d+)\.(\d+)') {
                $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                if ($major -ge 13)                    { return "$baseUrl/cu130" }
                if ($major -eq 12 -and $minor -ge 8)  { return "$baseUrl/cu128" }
                if ($major -eq 12 -and $minor -ge 6)  { return "$baseUrl/cu126" }
                if ($major -ge 12) { return "$baseUrl/cu124" }
                if ($major -ge 11) { return "$baseUrl/cu118" }
                return "$baseUrl/cpu"
            }
        } catch {}
        substep "could not determine CUDA version from nvidia-smi, defaulting to cu126" "Yellow"
        return "$baseUrl/cu126"
    }
    $TorchIndexUrl = Get-TorchIndexUrl

    # ── Print CPU-only hint when no GPU detected ──
    if (-not $SkipTorch -and $TorchIndexUrl -like "*/cpu") {
        Write-Host ""
        substep "No NVIDIA GPU detected." "Yellow"
        substep "Installing CPU-only PyTorch. If you only need GGUF chat/inference," "Yellow"
        substep "re-run with --no-torch for a faster, lighter install:" "Yellow"
        substep ".\install.ps1 --no-torch" "Yellow"
        Write-Host ""
    }

    # ── Install PyTorch first, then unsloth separately ──
    #
    # Why two steps?
    #   `uv pip install unsloth --torch-backend=cpu` on Windows resolves to
    #   unsloth==2024.8 (a pre-CLI release with no unsloth.exe) because the
    #   cpu-only solver cannot satisfy newer unsloth's dependencies.
    #   Installing torch first from the explicit CUDA index, then upgrading
    #   unsloth in a second step, avoids this solver dead-end.
    #
    # Why --upgrade-package instead of --upgrade?
    #   `--upgrade unsloth` re-resolves ALL dependencies including torch,
    #   pulling torch from default PyPI and stripping the +cuXXX suffix
    #   that step 1 installed (e.g. torch 2.5.1+cu124 -> 2.10.0 with no
    #   CUDA suffix).  `--upgrade-package unsloth` upgrades ONLY unsloth
    #   to the latest version while preserving the already-pinned torch
    #   CUDA wheels.  Missing dependencies (transformers, trl, peft, etc.)
    #   are still pulled in because they are new, not upgrades.
    #
    # ── Helper: find no-torch-runtime.txt ──
    function Find-NoTorchRuntimeFile {
        if ($StudioLocalInstall -and (Test-Path (Join-Path $RepoRoot "studio\backend\requirements\no-torch-runtime.txt"))) {
            return Join-Path $RepoRoot "studio\backend\requirements\no-torch-runtime.txt"
        }
        $installed = Get-ChildItem -Path $VenvDir -Recurse -Filter "no-torch-runtime.txt" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*studio*backend*requirements*no-torch-runtime.txt" } |
            Select-Object -ExpandProperty FullName -First 1
        return $installed
    }

    if ($_Migrated) {
        # Migrated env: force-reinstall unsloth+unsloth-zoo to ensure clean state
        # in the new venv location, while preserving existing torch/CUDA
        Write-TauriLog "STEP" "Installing unsloth"
        substep "upgrading unsloth in migrated environment..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.4.8" unsloth-zoo }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.4.8" unsloth-zoo }
        }
        if ($baseInstallExit -ne 0) {
            Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return
        }
        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return
            }
        }
    } elseif ($TorchIndexUrl) {
        if ($SkipTorch) {
            substep "skipping PyTorch (--no-torch flag set)." "Yellow"
        } else {
            Write-TauriLog "STEP" "Installing PyTorch"
            substep "installing PyTorch ($TorchIndexUrl)..."
            $torchInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython "torch>=2.4,<2.11.0" torchvision torchaudio --index-url $TorchIndexUrl }
            if ($torchInstallExit -ne 0) {
                Write-TauriLog "ERROR" "Failed to install PyTorch (exit code $torchInstallExit)"
                Write-Host "[ERROR] Failed to install PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                return
            }
        }

        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "unsloth>=2026.4.8" unsloth-zoo }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } elseif ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth "unsloth>=2026.4.8" unsloth-zoo }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth -- "$PackageName" }
        }
        if ($baseInstallExit -ne 0) {
            Write-TauriLog "ERROR" "Failed to install unsloth (exit code $baseInstallExit)"
            Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return
        }

        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return
            }
        }
    } else {
        # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython unsloth-zoo "unsloth>=2026.4.8" --torch-backend=auto }
            if ($baseInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return
            }
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return
            }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --torch-backend=auto -- "$PackageName" }
            if ($baseInstallExit -ne 0) {
                Write-TauriLog "ERROR" "Failed to install unsloth (exit code $baseInstallExit)"
                Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return
            }
        }
    }

    # Hotfix: patch install_python_stack.py for Windows GUI stdout
    # The PyPI version crashes with OSError when stdout is piped from a GUI app.
    # Copy our fixed version (bundled by Tauri) over the installed one.
    # Remove this block once PyPI ships the fix from commit 18c5aae7.
    if ($TauriMode) {
        $rawPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.ScriptName }
        $scriptDir = Split-Path -Parent ($rawPath -replace '^\\\\\?\\', '')
        $fixedPy = Join-Path $scriptDir "install_python_stack.py"
        $target = Join-Path $VenvDir "Lib\site-packages\studio\install_python_stack.py"
        $sentinel = "# UNSLOTH_DESKTOP_HOTFIX_APPLIED_v1"
        $sentinelPattern = [regex]::Escape($sentinel)
        if ((Test-Path $fixedPy) -and (Test-Path $target)) {
            $installed = Get-Content $target -Raw
            if ($installed -notmatch $sentinelPattern) {
                Copy-Item $fixedPy $target -Force
                Add-Content -Path $target -Value "`n$sentinel"
                substep "patched install_python_stack.py (stdout fix)"
            } else {
                substep "install_python_stack.py already has stdout fix"
            }
        } elseif ((Test-Path $fixedPy) -and (Test-Path (Split-Path $target))) {
            Copy-Item $fixedPy $target -Force
            Add-Content -Path $target -Value "`n$sentinel"
            substep "patched install_python_stack.py (stdout fix)"
        } else {
            Write-Host "[WARN] Could not patch install_python_stack.py (bundled file or target dir missing)" -ForegroundColor Yellow
        }
    }

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, Node.js, and other dependencies automatically via winget.
    Write-TauriLog "STEP" "Running studio setup"
    step "setup" "running unsloth studio setup..."
    $UnslothExe = Join-Path $VenvDir "Scripts\unsloth.exe"
    if (-not (Test-Path $UnslothExe)) {
        Write-TauriLog "ERROR" "unsloth CLI was not installed correctly"
        Write-Host "[ERROR] unsloth CLI was not installed correctly." -ForegroundColor Red
        Write-Host "        Expected: $UnslothExe" -ForegroundColor Yellow
        Write-Host "        This usually means an older unsloth version was installed that does not include the Studio CLI." -ForegroundColor Yellow
        Write-Host "        Try re-running the installer or see: https://github.com/unslothai/unsloth?tab=readme-ov-file#-quickstart" -ForegroundColor Yellow
        return
    }
    # Tell setup.ps1 to skip base package installation (install.ps1 already did it)
    $env:SKIP_STUDIO_BASE = "1"
    $env:STUDIO_PACKAGE_NAME = $PackageName
    $env:UNSLOTH_NO_TORCH = if ($SkipTorch) { "true" } else { "false" }
    # Tauri desktop app bundles its own frontend — skip Node/npm/frontend build
    $env:SKIP_STUDIO_FRONTEND = if ($TauriMode) { "1" } else { "0" }
    # Always set STUDIO_LOCAL_INSTALL explicitly to avoid stale values from
    # a previous --local run in the same PowerShell session.
    if ($StudioLocalInstall) {
        $env:STUDIO_LOCAL_INSTALL = "1"
        $env:STUDIO_LOCAL_REPO = $RepoRoot
    } else {
        $env:STUDIO_LOCAL_INSTALL = "0"
        Remove-Item Env:STUDIO_LOCAL_REPO -ErrorAction SilentlyContinue
    }
    # Use 'studio setup' (not 'studio update') because 'update' pops
    # SKIP_STUDIO_BASE, which would cause redundant package reinstallation
    # and bypass the fast-path version check from PR #4667.
    $studioArgs = @('studio', 'setup')
    if ($script:UnslothVerbose) { $studioArgs += '--verbose' }
    & $UnslothExe @studioArgs
    $setupExit = $LASTEXITCODE
    if ($setupExit -ne 0) {
        Write-TauriLog "ERROR" "unsloth studio setup failed (exit code $setupExit)"
        Write-Host "[ERROR] unsloth studio setup failed (exit code $setupExit)" -ForegroundColor Red
        return
    }

    # ── Expose `unsloth` via a shim dir containing only unsloth.exe ──
    # We do NOT add the venv Scripts dir to PATH (it also holds python.exe
    # and pip.exe, which would hijack the user's system interpreter).
    # Hardlink preferred; falls back to copy if cross-volume or non-NTFS.
    #
    # Remove the legacy venv Scripts PATH entry that older installers wrote.
    $LegacyScriptsDir = Join-Path $VenvDir "Scripts"
    try {
        $legacyKey = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
        try {
            $rawPath = $legacyKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            if ($rawPath) {
                [string[]]$pathEntries = $rawPath -split ';'
                $normalLegacy = $LegacyScriptsDir.Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
                $expNormalLegacy = [Environment]::ExpandEnvironmentVariables($LegacyScriptsDir).Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
                $filtered = @($pathEntries | Where-Object {
                    $stripped = $_.Trim().Trim('"')
                    $rawNorm = $stripped.TrimEnd('\').ToLowerInvariant()
                    $expNorm = [Environment]::ExpandEnvironmentVariables($stripped).TrimEnd('\').ToLowerInvariant()
                    ($rawNorm -ne $normalLegacy -and $rawNorm -ne $expNormalLegacy) -and
                    ($expNorm -ne $normalLegacy -and $expNorm -ne $expNormalLegacy)
                })
                $cleanedPath = $filtered -join ';'
                if ($cleanedPath -ne $rawPath) {
                    $legacyKey.SetValue('Path', $cleanedPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
                    try {
                        $d = "UnslothPathRefresh_$([guid]::NewGuid().ToString('N').Substring(0,8))"
                        [Environment]::SetEnvironmentVariable($d, '1', 'User')
                        [Environment]::SetEnvironmentVariable($d, [NullString]::Value, 'User')
                    } catch { }
                }
            }
        } finally {
            $legacyKey.Close()
        }
    } catch { }
    $ShimDir = Join-Path $StudioHome "bin"
    New-Item -ItemType Directory -Force -Path $ShimDir | Out-Null
    $ShimExe = Join-Path $ShimDir "unsloth.exe"
    # try/catch: if unsloth.exe is locked (Studio running), keep the old shim.
    $shimUpdated = $false
    try {
        if (Test-Path $ShimExe) { Remove-Item $ShimExe -Force -ErrorAction Stop }
        try {
            New-Item -ItemType HardLink -Path $ShimExe -Target $UnslothExe -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -Path $UnslothExe -Destination $ShimExe -Force -ErrorAction Stop # fallback: copy
        }
        $shimUpdated = $true
    } catch {
        if (Test-Path $ShimExe) {
            Write-Host "[WARN] Could not refresh unsloth launcher at $ShimExe." -ForegroundColor Yellow
            Write-Host "       This usually means a running 'unsloth studio' process still holds the file open." -ForegroundColor Yellow
            Write-Host "       Close Studio and re-run the installer to pick up the latest launcher." -ForegroundColor Yellow
            Write-Host "       Continuing with the existing launcher." -ForegroundColor Yellow
        } else {
            Write-Host "[WARN] Could not create unsloth launcher at $ShimExe" -ForegroundColor Yellow
            Write-Host "       $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "       Launch unsloth studio directly via '$UnslothExe' until the next successful install." -ForegroundColor Yellow
        }
    }
    # Only add to PATH when the launcher actually exists on disk.
    $pathAdded = $false
    if (Test-Path $ShimExe) {
        $pathAdded = Add-ToUserPath -Directory $ShimDir -Position 'Prepend'
    }
    if ($shimUpdated -and $pathAdded) {
        step "path" "added unsloth launcher to PATH"
    }
    Refresh-SessionPath  # sync current session with registry

    # ── Tauri mode: done, skip shortcuts and auto-launch ──
    if ($TauriMode) {
        Write-TauriLog "DONE" ""
        return
    }

    New-StudioShortcuts -UnslothExePath $UnslothExe

    # Launch studio automatically in interactive terminals;
    # in non-interactive environments (CI, Docker) just print instructions.
    $IsInteractive = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        & $UnslothExe studio -H 0.0.0.0 -p 8888
    } else {
        step "launch" "manual commands:"
        substep "& `"$VenvDir\Scripts\Activate.ps1`""
        substep "unsloth studio -H 0.0.0.0 -p 8888"
        Write-Host ""
    }
}

Install-UnslothStudio @args
