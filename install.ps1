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
    $SkipTorch = $false
    $argList = $args
    for ($i = 0; $i -lt $argList.Count; $i++) {
        switch ($argList[$i]) {
            "--local"    { $StudioLocalInstall = $true }
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
    function Refresh-SessionPath {
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $merged  = "$machine;$user;$env:Path"
        $seen    = @{}
        $unique  = @()
        foreach ($p in $merged -split ";") {
            $key = $p.TrimEnd("\").ToLowerInvariant()
            if ($key -and -not $seen.ContainsKey($key)) {
                $seen[$key] = $true
                $unique += $p
            }
        }
        $env:Path = $unique -join ";"
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

    # ── AMD Windows ROCm helpers ──
    # Detect AMD HIP SDK 7.1.x / 7.2.x on Windows. Returns a hashtable
    # @{ Major = 7; Minor = 2; Path = "C:\Program Files\AMD\ROCm\7.2" }
    # or $null when the SDK is missing, unreadable, or at an unsupported
    # version. Callers differentiate "not installed" vs "installed but wrong
    # version" by re-parsing $env:HIP_PATH themselves.
    function Get-HipSdkVersion {
        # Primary signal: HIP_PATH env var, e.g. "C:\Program Files\AMD\ROCm\7.2\"
        # The final path component is the version. Validate bin\ exists so
        # a stale env var pointing at a half-uninstalled SDK is rejected.
        $parseComponent = {
            param($name)
            if ([string]::IsNullOrWhiteSpace($name)) { return $null }
            $m = [regex]::Match($name.Trim(), '^(\d+)\.(\d+)')
            if ($m.Success) {
                return @{ Major = [int]$m.Groups[1].Value; Minor = [int]$m.Groups[2].Value }
            }
            return $null
        }

        $hipPath = $env:HIP_PATH
        if ($hipPath) {
            $trimmed = $hipPath.TrimEnd('\','/')
            if (Test-Path (Join-Path $trimmed 'bin')) {
                $parsed = & $parseComponent (Split-Path $trimmed -Leaf)
                if ($parsed) {
                    $parsed.Path = $trimmed
                    return $parsed
                }
            }
        }

        # Fallback: scan C:\Program Files\AMD\ROCm\
        $rocmRoot = Join-Path $env:ProgramFiles 'AMD\ROCm'
        if (Test-Path $rocmRoot) {
            $best = $null
            try {
                $dirs = Get-ChildItem -Path $rocmRoot -Directory -ErrorAction SilentlyContinue
            } catch { $dirs = @() }
            foreach ($d in $dirs) {
                if (-not (Test-Path (Join-Path $d.FullName 'bin'))) { continue }
                $parsed = & $parseComponent $d.Name
                if ($null -eq $parsed) { continue }
                if ($null -eq $best -or
                    $parsed.Major -gt $best.Major -or
                    ($parsed.Major -eq $best.Major -and $parsed.Minor -gt $best.Minor)) {
                    $parsed.Path = $d.FullName
                    $best = $parsed
                }
            }
            if ($best) { return $best }
        }

        return $null
    }

    # Map a ROCm release version to the full Radeon Windows wheel set.
    # Returns @{ SdkCore, SdkDevel, SdkLibraries, SdkTarball, Torch,
    # Torchvision, Torchaudio } or $null when unsupported. AMD's docs at
    # rocm.docs.amd.com/projects/radeon-ryzen/.../install-pytorch.html
    # require a two-step install: first the rocm_sdk_* wheels (~1.4 GB;
    # ship the runtime that torch links against), then torch itself. Both
    # are mandatory -- torch import fails with missing DLLs otherwise.
    # Wheels are cp312 only.
    function Get-RocmWheelUrls {
        param([Parameter(Mandatory = $true)]$Version)
        $base721 = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/'
        $base711 = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/'
        if ($Version.Major -eq 7 -and $Version.Minor -eq 2) {
            return @{
                SdkCore       = $base721 + 'rocm_sdk_core-7.2.1-py3-none-win_amd64.whl'
                SdkDevel      = $base721 + 'rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl'
                SdkLibraries  = $base721 + 'rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl'
                SdkTarball    = $base721 + 'rocm-7.2.1.tar.gz'
                Torch         = $base721 + 'torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
                Torchvision   = $base721 + 'torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
                Torchaudio    = $base721 + 'torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
            }
        }
        if ($Version.Major -eq 7 -and $Version.Minor -eq 1) {
            # 7.1.1 stamps SDK wheels with `0.1.dev0`; torch gets rocmsdk date tag.
            return @{
                SdkCore       = $base711 + 'rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl'
                SdkDevel      = $base711 + 'rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl'
                SdkLibraries  = $base711 + 'rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl'
                SdkTarball    = $base711 + 'rocm-0.1.dev0.tar.gz'
                Torch         = $base711 + 'torch-2.9.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl'
                Torchvision   = $base711 + 'torchvision-0.24.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl'
                Torchaudio    = $base711 + 'torchaudio-2.9.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl'
            }
        }
        return $null
    }

    # Default Windows ROCm release when HIP_PATH is absent. HIP_PATH is an
    # optional hint, NOT a prerequisite -- regular torch users only need
    # an AMD graphics driver (26.2.2+ for 7.2.1) and Python 3.12. PyTorch
    # does not publish Windows ROCm wheels on download.pytorch.org (see
    # the "ROCm is not available on Windows" note on pytorch.org), so
    # repo.radeon.com is the only source until pytorch/pytorch#159520
    # lands upstream Windows ROCm hosting in a future release.
    $DefaultWindowsRocmVersion = @{ Major = 7; Minor = 2 }

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
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        step "winget" "not available" "Red"
        substep "Install it from https://aka.ms/getwinget" "Yellow"
        substep "or install Python $PythonVersion and uv manually, then re-run." "Yellow"
        return
    }

    # ── Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) ──
    # Runs before Python detection so the AMD/ROCm path can require Python
    # 3.12 (the only cp tag Radeon publishes Windows wheels for).
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

    # AMD GPU presence via WMI -- always available on Windows, no elevation,
    # no dependency on HIP SDK being installed. We deliberately avoid
    # hipinfo.exe here because it ships inside the HIP SDK and would create
    # a chicken-and-egg where we cannot prompt "install HIP SDK" on the hosts
    # that need the prompt.
    $HasAmdGpu = $false
    try {
        $videoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        if ($videoControllers) {
            $amdMatches = @($videoControllers | Where-Object { $_.Name -match 'AMD|Radeon' })
            if ($amdMatches.Count -gt 0) { $HasAmdGpu = $true }
        }
    } catch {}
    # NVIDIA wins on mixed systems -- matches Linux install.sh behaviour so
    # users with both cards get the NVIDIA torch path they expect.
    if ($HasNvidiaSmi) { $HasAmdGpu = $false }

    # Probe HIP SDK as an OPTIONAL version hint. The HIP SDK developer
    # toolkit is NOT a prerequisite for running torch on Windows -- AMD's
    # install docs only require the graphics driver (26.2.2+ for 7.2.1)
    # and Python 3.12. We use $HipSdkVersion when present to select a
    # matching ROCm wheel release; otherwise we fall back to the newest
    # stable release ($DefaultWindowsRocmVersion).
    $HipSdkVersion = $null
    if ($HasAmdGpu) {
        $HipSdkVersion = Get-HipSdkVersion
    }

    if ($HasNvidiaSmi) {
        step "gpu" "NVIDIA GPU detected"
    } elseif ($HasAmdGpu) {
        if ($HipSdkVersion) {
            step "gpu" ("AMD GPU detected (HIP SDK {0}.{1} hint)" -f $HipSdkVersion.Major, $HipSdkVersion.Minor)
        } else {
            step "gpu" ("AMD GPU detected (will use rocm-rel-{0}.{1}.x)" -f $DefaultWindowsRocmVersion.Major, $DefaultWindowsRocmVersion.Minor)
            substep "HIP SDK not found (optional). Ensure AMD graphics driver is up to date:" "DarkGray"
            substep "https://www.amd.com/en/support/download/drivers.html" "DarkGray"
        }
    } else {
        step "gpu" "none (chat-only / GGUF)" "Yellow"
        substep "Training and GPU inference require an NVIDIA or AMD GPU with drivers installed." "Yellow"
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
    #
    # $PreferredVersions is a list of minor versions to search, in priority
    # order. Defaults to 3.13, 3.12, 3.11. AMD/ROCm callers pass @("3.12")
    # because Radeon's Windows wheels are cp312 only.
    function Find-CompatiblePython {
        param([string[]]$PreferredVersions = @("3.13", "3.12", "3.11"))

        # Build a version regex from the requested minor list. We accept
        # "3.<minor>.<patch>" where <minor> is in the preferred list.
        $minors = @()
        foreach ($v in $PreferredVersions) {
            if ($v -match '^3\.(\d+)$') { $minors += $Matches[1] }
        }
        if ($minors.Count -eq 0) { return $null }
        $minorAlt = ($minors | Sort-Object -Unique) -join '|'
        $verRegex = "Python (3\.(?:$minorAlt))\.\d+"

        # Try the Python Launcher first (most reliable on Windows)
        # py.exe resolves to the standard CPython install, not conda.
        $pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if ($pyLauncher -and $pyLauncher.Source -notmatch $script:CondaSkipPattern) {
            foreach ($minor in $PreferredVersions) {
                try {
                    $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
                    if ($out -match $verRegex) {
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
                    if ($out -match $verRegex) {
                        return @{ Version = $Matches[1]; Path = $cmd.Source }
                    }
                } catch {}
            }
        }
        return $null
    }

    # ── Install Python if no compatible version found ──
    # Find-CompatiblePython returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    # AMD path requires Python 3.12 specifically (Radeon only publishes cp312
    # wheels for Windows), so we pass a narrower preference list AND skip the
    # winget auto-install (the user must have 3.12 already).
    if ($HasAmdGpu) {
        $PythonPreferred = @("3.12")
    } else {
        $PythonPreferred = @("3.13", "3.12", "3.11")
    }
    $DetectedPython = Find-CompatiblePython -PreferredVersions $PythonPreferred
    if ($DetectedPython) {
        step "python" "Python $($DetectedPython.Version) already installed"
    }
    if (-not $DetectedPython -and $HasAmdGpu) {
        Write-Host "[ERROR] AMD ROCm path requires Python 3.12 (Radeon's Windows wheels are cp312 only)." -ForegroundColor Red
        Write-Host "        Install Python 3.12 from https://www.python.org/downloads/ or via:" -ForegroundColor Yellow
        Write-Host "        winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
        Write-Host "        Then re-run this installer." -ForegroundColor Yellow
        return
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
        $DetectedPython = Find-CompatiblePython -PreferredVersions $PythonPreferred

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
            $DetectedPython = Find-CompatiblePython -PreferredVersions $PythonPreferred
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
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
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
            Write-Host "[ERROR] Failed to create virtual environment (exit code $venvExit)" -ForegroundColor Red
            return
        }
    } else {
        step "venv" "using migrated environment"
        substep "$VenvDir"
    }

    # GPU detection moved earlier in the flow (see "Detect GPU" block above
    # the Python detection); $HasNvidiaSmi, $NvidiaSmiExe, $HasAmdGpu, and
    # $HipSdkVersion are already populated by this point.

    # ── Choose the correct PyTorch index URL based on driver CUDA version ──
    # Mirrors Get-PytorchCudaTag in setup.ps1.
    function Get-TorchIndexUrl {
        $baseUrl = "https://download.pytorch.org/whl"
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

    # The AMD/ROCm path does not use a --index-url; instead it installs
    # explicit wheel URLs from repo.radeon.com. $TorchIndexUrl stays $null
    # on that branch so the NVIDIA/CPU path is visibly bypassed. HIP_PATH
    # is a hint only -- we fall back to $DefaultWindowsRocmVersion when
    # it is absent or points at an unsupported version, because the HIP
    # SDK is NOT a runtime prerequisite for torch on Windows.
    $TorchIndexUrl = $null
    $RocmWheelUrls = $null
    $RocmReleaseVersion = $null
    if ($HasAmdGpu) {
        if ($HipSdkVersion) {
            $RocmWheelUrls = Get-RocmWheelUrls -Version $HipSdkVersion
            if ($RocmWheelUrls) {
                $RocmReleaseVersion = $HipSdkVersion
            }
        }
        if (-not $RocmWheelUrls) {
            if ($HipSdkVersion) {
                substep ("HIP SDK {0}.{1} is too old; falling back to rocm-rel-{2}.{3}.x" -f $HipSdkVersion.Major, $HipSdkVersion.Minor, $DefaultWindowsRocmVersion.Major, $DefaultWindowsRocmVersion.Minor) "Yellow"
            }
            $RocmWheelUrls = Get-RocmWheelUrls -Version $DefaultWindowsRocmVersion
            $RocmReleaseVersion = $DefaultWindowsRocmVersion
        }
    } else {
        $TorchIndexUrl = Get-TorchIndexUrl
    }

    # ── Print CPU-only hint when no GPU detected ──
    if (-not $SkipTorch -and -not $HasAmdGpu -and $TorchIndexUrl -like "*/cpu") {
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
        substep "upgrading unsloth in migrated environment..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.4.4" unsloth-zoo }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.4.4" unsloth-zoo }
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
    } elseif ($HasAmdGpu) {
        if ($SkipTorch) {
            substep "skipping PyTorch (--no-torch flag set)." "Yellow"
        } elseif (-not $RocmWheelUrls) {
            # Should be unreachable because the detection block above
            # always falls back to $DefaultWindowsRocmVersion, but guard
            # anyway so a future refactor that drops the fallback does
            # not install CPU torch on an AMD host.
            Write-Host "[ERROR] Could not resolve Windows ROCm wheel URLs." -ForegroundColor Red
            Write-Host "        This is a bug; please file it at github.com/unslothai/unsloth/issues" -ForegroundColor Yellow
            return
        } else {
            # Verify the venv's Python is 3.12 -- Radeon's wheels are cp312
            # only and pip would fail with a confusing error otherwise.
            $venvPyVer = ''
            try {
                $venvPyVer = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null | Out-String).Trim()
            } catch {}
            if ($venvPyVer -and $venvPyVer -ne "3.12") {
                Write-Host "[ERROR] Radeon Windows ROCm wheels require Python 3.12 (venv has $venvPyVer)." -ForegroundColor Red
                Write-Host "        Install Python 3.12 from https://www.python.org/downloads/ and re-run." -ForegroundColor Yellow
                return
            }

            substep ("installing Radeon ROCm wheels for rocm-rel-{0}.{1}.x ..." -f $RocmReleaseVersion.Major, $RocmReleaseVersion.Minor)
            substep "Step 1/2: ROCm SDK runtime (~1.4 GB -- this will take a while)"
            # Use python -m pip (NOT uv) because (a) AMD's documented
            # procedure uses pip, (b) uv has known wheel-corruption issues
            # on these big ROCm/bnb wheels (unslothai/unsloth#4966), and
            # (c) pip's dep resolver is the combination AMD validates. We
            # install all four SDK artefacts in one command so pip does
            # not reset torch between them.
            $sdkInstallExit = Invoke-InstallCommand {
                & $VenvPython -m pip install --no-cache-dir --force-reinstall `
                    $RocmWheelUrls.SdkCore `
                    $RocmWheelUrls.SdkDevel `
                    $RocmWheelUrls.SdkLibraries `
                    $RocmWheelUrls.SdkTarball
            }
            if ($sdkInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install ROCm SDK wheels (exit code $sdkInstallExit)" -ForegroundColor Red
                Write-Host "        Verify your AMD graphics driver is recent and repo.radeon.com is reachable." -ForegroundColor Yellow
                return
            }

            substep "Step 2/2: PyTorch + torchvision + torchaudio (~820 MB)"
            $torchInstallExit = Invoke-InstallCommand {
                & $VenvPython -m pip install --no-cache-dir --force-reinstall `
                    $RocmWheelUrls.Torch `
                    $RocmWheelUrls.Torchvision `
                    $RocmWheelUrls.Torchaudio
            }
            if ($torchInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install ROCm PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                Write-Host "        Update your AMD graphics driver: https://www.amd.com/en/support/download/drivers.html" -ForegroundColor Yellow
                return
            }
        }

        substep "installing unsloth (this may take a few minutes)..."
        # --no-deps prevents uv from re-resolving torch back to default PyPI
        # (which would strip the +rocm suffix). bitsandbytes has no Windows
        # ROCm wheel so it is deliberately NOT installed here; 4-bit
        # quantization is not available on Windows AMD yet.
        if ($SkipTorch) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "unsloth>=2026.4.4" unsloth-zoo }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } elseif ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth "unsloth>=2026.4.4" unsloth-zoo }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth "$PackageName" }
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
            substep "installing PyTorch ($TorchIndexUrl)..."
            $torchInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython "torch>=2.4,<2.11.0" torchvision torchaudio --index-url $TorchIndexUrl }
            if ($torchInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                return
            }
        }

        substep "installing unsloth (this may take a few minutes)..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "unsloth>=2026.4.4" unsloth-zoo }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } elseif ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth "unsloth>=2026.4.4" unsloth-zoo }
        } else {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython --upgrade-package unsloth "$PackageName" }
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
    } else {
        # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
        substep "installing unsloth (this may take a few minutes)..."
        if ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython unsloth-zoo "unsloth>=2026.4.4" --torch-backend=auto }
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
            $baseInstallExit = Invoke-InstallCommand { uv pip install --python $VenvPython "$PackageName" --torch-backend=auto }
            if ($baseInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return
            }
        }
    }

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, Node.js, and other dependencies automatically via winget.
    step "setup" "running unsloth studio setup..."
    $UnslothExe = Join-Path $VenvDir "Scripts\unsloth.exe"
    if (-not (Test-Path $UnslothExe)) {
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
        Write-Host "[ERROR] unsloth studio setup failed (exit code $setupExit)" -ForegroundColor Red
        return
    }

    New-StudioShortcuts -UnslothExePath $UnslothExe

    # ── Add venv Scripts dir to User PATH so `unsloth studio` works from any terminal ──
    $ScriptsDir = Join-Path $VenvDir "Scripts"
    $UserPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $UserPath -or $UserPath -notlike "*$ScriptsDir*") {
        if ($UserPath) {
            [System.Environment]::SetEnvironmentVariable("Path", "$ScriptsDir;$UserPath", "User")
        } else {
            [System.Environment]::SetEnvironmentVariable("Path", "$ScriptsDir", "User")
        }
        Refresh-SessionPath
        step "path" "added unsloth to PATH"
    }

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
