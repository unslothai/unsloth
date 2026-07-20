# Unsloth Studio Installer for Windows PowerShell
#
# Usage:  irm https://unsloth.ai/install.ps1 | iex
#         Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\install.ps1 --local
#
# irm | iex cannot forward arguments, so web installs take options as env vars set
# before the pipe (flags still work via .\install.ps1):
#   $env:UNSLOTH_NO_TORCH=1; irm https://unsloth.ai/install.ps1 | iex       # skip PyTorch (GGUF-only)
#   $env:UNSLOTH_SKIP_AUTOSTART=1; irm https://unsloth.ai/install.ps1 | iex # do not prompt to launch
#   $env:UNSLOTH_PYTHON='3.12'; irm https://unsloth.ai/install.ps1 | iex    # pin Python version
#   $env:UNSLOTH_STUDIO_HOME='C:\path'; irm https://unsloth.ai/install.ps1 | iex
#   .\install.ps1 --no-torch                                                # equivalent flag
# Or pass flags to a scriptblock: & ([scriptblock]::Create((irm https://unsloth.ai/install.ps1))) --no-torch
#
# Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME (alias) > $USERPROFILE\.unsloth\studio
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
function Install-UnslothStudio {
    $ErrorActionPreference = "Stop"
    $script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq "1")

    # ── Tauri structured output ──
    function Write-TauriLog {
        param([string]$Tag, [string]$Message)
        if ($TauriMode) {
            Write-Host "[TAURI:$Tag] $Message"
        }
    }

    function Format-TauriDiagBool {
        param([bool]$Value)
        if ($Value) { return "true" }
        return "false"
    }

    function Get-TauriDiagArch {
        $arch = [string]$env:PROCESSOR_ARCHITECTURE
        if ([string]::IsNullOrWhiteSpace($arch)) {
            try { $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() } catch { $arch = "unknown" }
        }
        $arch = $arch.ToLowerInvariant()
        switch ($arch) {
            "amd64" { return "x86_64" }
            "x64" { return "x86_64" }
            "arm64" { return "arm64" }
            "x86" { return "x86" }
            default { return ($arch -replace '[^a-z0-9_.-]', '_') }
        }
    }

    function Get-TauriTorchIndexFamily {
        param([string]$TorchIndexUrl)
        if ($SkipTorch) { return "none" }
        if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) { return "none" }
        # Drop query/fragment first so a token-authenticated pin classifies by family.
        $leaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
        if (@("cpu", "cu118", "cu124", "cu126", "cu128", "cu130") -contains $leaf) { return $leaf }
        if ($leaf -match '^rocm[0-9]+\.[0-9]+$') { return $leaf }
        return "auto"
    }

    function Get-TauriGpuBranch {
        param([string]$TorchIndexFamily)
        if ($SkipTorch) { return "no_torch" }
        # Require a digit after "cu" so /current or /custom isn't branded CUDA (parity ^cu[0-9]).
        if ($TorchIndexFamily -match '^cu[0-9]') { return "cuda" }
        if ($TorchIndexFamily -like "rocm*") { return "rocm" }
        if ($TorchIndexFamily -eq "cpu") { return "cpu" }
        return "unknown"
    }

    function Write-TauriDiag {
        param(
            [string]$GpuBranch = "unknown",
            [string]$TorchIndexFamily = "none",
            [string]$PythonVersionForDiag = $PythonVersion
        )
        if ([string]::IsNullOrWhiteSpace($PythonVersionForDiag)) { $PythonVersionForDiag = "unknown" }
        Write-TauriLog "DIAG" "diag_schema=1 platform=windows arch=$(Get-TauriDiagArch) python_version=$($PythonVersionForDiag.ToLowerInvariant()) skip_torch=$(Format-TauriDiagBool $SkipTorch) mac_intel=false gpu_branch=$GpuBranch torch_index_family=$TorchIndexFamily"
    }

    function Exit-InstallFailure {
        param(
            [Parameter(Mandatory = $true)][string]$Message,
            [int]$Code = 1
        )
        if ($Code -eq 0) { $Code = 1 }
        Write-TauriLog "ERROR" $Message
        if (Get-Command Restore-StudioVenvRollback -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioVenvRollback
        }
        if ($TauriMode) {
            exit $Code
        }
        throw $Message
    }

    # ── Parse flags ──
    $StudioLocalInstall = $false
    $PackageName = "unsloth"
    $RepoRoot = ""
    $TauriMode = $false
    $SkipTorch = $false
    $SkipAutostart = $false
    $ShortcutsOnly = $false
    $WithLlamaCppDir = ""
    $argList = $args
    for ($i = 0; $i -lt $argList.Count; $i++) {
        switch ($argList[$i]) {
            "--local"    { $StudioLocalInstall = $true }
            "--tauri"    { $TauriMode = $true }
            "--no-torch" { $SkipTorch = $true }
            "--verbose"  { $script:UnslothVerbose = $true }
            "-v"         { $script:UnslothVerbose = $true }
            "--shortcuts-only" { $ShortcutsOnly = $true }
            "--package"  {
                $i++
                if ($i -ge $argList.Count) {
                    Write-Host "[ERROR] --package requires an argument." -ForegroundColor Red
                    return (Exit-InstallFailure "--package requires an argument.")
                }
                $PackageName = $argList[$i]
            }
            "--with-llama-cpp-dir" {
                $i++
                if ($i -ge $argList.Count) {
                    Write-Host "[ERROR] --with-llama-cpp-dir requires a path argument." -ForegroundColor Red
                    return (Exit-InstallFailure "--with-llama-cpp-dir requires a path argument.")
                }
                $WithLlamaCppDir = $argList[$i]
            }
        }
    }

    # Env-var equivalent for web installs; an explicit flag still wins.
    if ($env:UNSLOTH_NO_TORCH -in @('1', 'true', 'yes', 'on')) { $SkipTorch = $true }
    if ($env:UNSLOTH_SKIP_AUTOSTART -in @('1', 'true', 'yes', 'on')) { $SkipAutostart = $true }

    # Propagate to child processes so they also respect verbose mode.
    # Process-scoped -- does not persist.
    if ($script:UnslothVerbose) {
        $env:UNSLOTH_VERBOSE = '1'
    }

    if ($StudioLocalInstall) {
        $RepoRoot = (Resolve-Path (Split-Path -Parent $PSCommandPath)).Path
        if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
            Write-Host "[ERROR] --local must be run from the unsloth repo root (pyproject.toml not found at $RepoRoot)" -ForegroundColor Red
            return (Exit-InstallFailure "--local must be run from the unsloth repo root")
        }
    }

    # Validate --package to prevent injection into shell/Python commands
    if ($PackageName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
        Write-Host "[ERROR] --package name contains invalid characters (allowed: a-z A-Z 0-9 . _ -)" -ForegroundColor Red
        return (Exit-InstallFailure "--package name contains invalid characters")
    }

    # UNSLOTH_PYTHON pins the version (mirrors install.sh --python); default 3.13.
    $PythonVersion = if ($env:UNSLOTH_PYTHON) { $env:UNSLOTH_PYTHON } else { "3.13" }
    # python.org fallback patch, used only when winget is unavailable/broken AND
    # the live python.org listing can't be fetched. The installer URL scheme is
    # stable so an older patch still installs. Bump alongside $PythonVersion.
    $PythonFallbackFullVersion = "3.13.13"

    # Resolve install destinations. Priority: UNSLOTH_STUDIO_HOME, then
    # STUDIO_HOME alias, then USERPROFILE-redirect, then default.
    # Reject whitespace-only values so " " is treated as unset (matches the
    # Python resolvers' .strip()), preventing install/runtime layout drift.
    $envOverrideVar = $null
    $envOverride = $null
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
        $envOverrideVar = "UNSLOTH_STUDIO_HOME"
        $envOverride = $env:UNSLOTH_STUDIO_HOME.Trim()
    } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
        $envOverrideVar = "STUDIO_HOME"
        $envOverride = $env:STUDIO_HOME.Trim()
    }

    # Custom Unsloth roots are not supported with --tauri (desktop app still
    # resolves %USERPROFILE%\.unsloth\studio). Pass through if override == legacy.
    if ($TauriMode -and $envOverride) {
        $_tauriOverride = $envOverride
        if ($_tauriOverride -eq "~" -or $_tauriOverride -like "~/*" -or $_tauriOverride -like "~\*") {
            $_tauriOverride = (Join-Path $env:USERPROFILE $_tauriOverride.Substring(1).TrimStart('/','\'))
        }
        try {
            $_tauriOverride = [System.IO.Path]::GetFullPath($_tauriOverride)
        } catch {}
        $_legacyTauriRoot = Join-Path $env:USERPROFILE ".unsloth\studio"
        try {
            $_legacyTauriRoot = [System.IO.Path]::GetFullPath($_legacyTauriRoot)
        } catch {}
        # Strip trailing separators so ".../studio\" matches ".../studio".
        $_trimSeps = @(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
        $_tauriOverride = $_tauriOverride.TrimEnd($_trimSeps)
        $_legacyTauriRoot = $_legacyTauriRoot.TrimEnd($_trimSeps)
        if ($_tauriOverride -ne $_legacyTauriRoot) {
            Write-Host "ERROR: $envOverrideVar is not supported with --tauri." -ForegroundColor Red
            Write-Host "       The desktop app still uses the legacy %USERPROFILE%\.unsloth\studio root." -ForegroundColor Red
            Write-Host "       Run install.ps1 without --tauri for custom-root shell installs," -ForegroundColor Yellow
            Write-Host "       or unset the env var for default desktop installs." -ForegroundColor Yellow
            throw "$envOverrideVar is not supported with --tauri."
        }
    }

    $defaultProfile = $null
    try { $defaultProfile = [Environment]::GetFolderPath("UserProfile") } catch {}

    # LOCALAPPDATA may be unset in service / CI contexts; Join-Path would abort
    # under ErrorActionPreference=Stop without this guard.
    $defaultDataDir = if ($env:LOCALAPPDATA -and -not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Join-Path $env:LOCALAPPDATA "Unsloth Studio"
    } else { $null }

    if ($envOverride) {
        # Tilde expansion: env vars aren't subject to it when quoted on assignment.
        if ($envOverride -eq "~" -or $envOverride -like "~/*" -or $envOverride -like "~\*") {
            $envOverride = (Join-Path $env:USERPROFILE $envOverride.Substring(1).TrimStart('/','\'))
        }
        try {
            # .NET API: New-Item -Path treats brackets as wildcards and has no
            # -LiteralPath in PS 5.1, so a root like C:\studio[abc] would fail.
            [System.IO.Directory]::CreateDirectory($envOverride) | Out-Null
            $StudioHome = (Resolve-Path -LiteralPath $envOverride).Path
        } catch {
            Write-Host "ERROR: $envOverrideVar=$envOverride cannot be created or accessed." -ForegroundColor Red
            throw "$envOverrideVar=$envOverride cannot be created or accessed."
        }
        $probe = Join-Path $StudioHome (".unsloth-write-probe-" + [guid]::NewGuid())
        try {
            # WriteAllText: literal-path safe + closes handle so Remove-Item works.
            [System.IO.File]::WriteAllText($probe, "")
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "ERROR: $envOverrideVar=$StudioHome is not writable." -ForegroundColor Red
            throw "$envOverrideVar=$StudioHome is not writable."
        }
        $StudioDataDir = Join-Path $StudioHome "share"
        $StudioRedirectMode = 'env'
    } elseif ($defaultProfile -and $env:USERPROFILE -and ($env:USERPROFILE -ne $defaultProfile)) {
        $StudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
        $StudioDataDir = $defaultDataDir
        $StudioRedirectMode = 'profile'
    } else {
        $StudioHome = Join-Path $env:USERPROFILE ".unsloth\studio"
        $StudioDataDir = $defaultDataDir
        $StudioRedirectMode = 'default'
    }
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

    # Redact index-URL credentials (userinfo + ?query= + #fragment) from captured installer
    # output before printing on failure; uv/pip errors echo the failing --index-url verbatim.
    # Mirrors the other installers. Verbose mode streams uncaptured, so it isn't redacted.
    function Redact-InstallOutput {
        param([string]$Text)
        if (-not $Text) { return $Text }
        $Text = $Text -replace '(https?://)[^/@\s`]+@', '$1<redacted>@'
        $Text = $Text -replace '([?&][^=\s&`]+)=[^&#\s`]+', '$1=<redacted>'
        # A #token=... fragment is as sensitive as a query; URL-anchored.
        return $Text -replace '(https?://[^\s`#]+)#[^\s`]+', '$1#<redacted>'
    }

    # Run native commands quietly by default to match install.sh behavior.
    # Full command output is shown only when --verbose / UNSLOTH_VERBOSE=1.
    function Invoke-InstallCommand {
        param(
            [Parameter(Mandatory = $true)][ScriptBlock]$Command
        )
        # Installer-pinned index installs (torch) must beat an inherited uv mirror (#6898):
        # for --default-index, clear the uv index env vars (restore in finally) and set
        # UV_NO_CONFIG=1 so a uv.toml/pyproject index can't outrank the CLI pin (uv 0.10).
        $savedUvIndex = $null
        if ($Command.ToString() -match '--default-index') {
            $savedUvIndex = @{}
            foreach ($n in 'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX', 'UV_EXTRA_INDEX_URL', 'UV_TORCH_BACKEND', 'UV_FIND_LINKS', 'UV_CONFIG_FILE', 'UV_NO_CONFIG') {
                $savedUvIndex[$n] = [Environment]::GetEnvironmentVariable($n)
                Remove-Item "Env:$n" -ErrorAction SilentlyContinue
            }
            $env:UV_NO_CONFIG = '1'
        }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # Reset to avoid stale values from prior native commands.
            $global:LASTEXITCODE = 0
            if ($script:UnslothVerbose) {
                # Merge stderr into stdout so progress/warning output stays visible
                # without flipping $? on successful native commands (PS 5.1 treats
                # stderr records as errors that set $? = $false even on exit code 0).
                # Redact per record: uv echoes index URLs (credentials and all) in
                # its errors, and verbose mode must not bypass the quiet path's
                # redaction. ForEach-Object/Out-Host leave $LASTEXITCODE untouched.
                & $Command 2>&1 | ForEach-Object { Redact-InstallOutput "$_" } | Out-Host
            } else {
                $output = & $Command 2>&1 | Out-String
                if ($LASTEXITCODE -ne 0) {
                    Write-Host (Redact-InstallOutput $output) -ForegroundColor Red
                }
            }
            return [int]$LASTEXITCODE
        } finally {
            $ErrorActionPreference = $prevEap
            if ($savedUvIndex) {
                Remove-Item "Env:UV_NO_CONFIG" -ErrorAction SilentlyContinue
                foreach ($n in $savedUvIndex.Keys) { if ($null -ne $savedUvIndex[$n]) { Set-Item "Env:$n" $savedUvIndex[$n] } }
            }
        }
    }

    # Retry Invoke-InstallCommand on transient uv download failures with backoff.
    # Returns the last exit code on permanent failure so rollback still fires.
    function Invoke-InstallCommandRetry {
        param(
            [Parameter(Mandatory = $true, Position = 0)][ScriptBlock]$Command,
            [string]$Label = "install step"
        )
        # Sanitize overrides to a default of 3 (a typo must not disable retries; =1 disables).
        # TryParse with bounds avoids an Int32 overflow throw. Bounds: 1..100 retries, 0..3600s.
        $maxAttempts = 3
        $parsedAttempts = 0
        if ([int]::TryParse($env:UNSLOTH_INSTALL_RETRIES, [ref]$parsedAttempts) -and $parsedAttempts -ge 1 -and $parsedAttempts -le 100) {
            $maxAttempts = $parsedAttempts
        }
        $delay = 3
        $parsedDelay = 0
        if ([int]::TryParse($env:UNSLOTH_INSTALL_RETRY_DELAY, [ref]$parsedDelay) -and $parsedDelay -ge 0 -and $parsedDelay -le 3600) {
            $delay = $parsedDelay
        }
        $attempt = 1
        while ($true) {
            $code = Invoke-InstallCommand $Command
            if ($code -eq 0) { return 0 }
            if ($attempt -ge $maxAttempts) { return $code }
            substep ("retrying ""$Label"" after transient failure (attempt $($attempt + 1)/$maxAttempts, waiting ${delay}s)...") "Yellow"
            Start-Sleep -Seconds $delay
            $attempt++
            $delay = $delay * 2
        }
    }

    function New-StudioShortcuts {
        param(
            [Parameter(Mandatory = $true)][string]$UnslothExePath
        )

        if (-not (Test-Path -LiteralPath $UnslothExePath)) {
            substep "cannot create shortcuts, unsloth.exe not found at $UnslothExePath" "Yellow"
            return
        }
        try {
            # Persist an absolute path in launcher scripts so shortcut working
            # directory changes do not break process startup.
            $UnslothExePath = (Resolve-Path -LiteralPath $UnslothExePath).Path
            # Escape for single-quoted embedding in generated launcher script.
            # This prevents runtime variable expansion for paths containing '$'.
            $SingleQuotedExePath = $UnslothExePath -replace "'", "''"

            # $StudioDataDir = LOCALAPPDATA\Unsloth Studio, or $StudioHome\share in env-mode.
            if (-not $StudioDataDir -or [string]::IsNullOrWhiteSpace($StudioDataDir)) {
                substep "DataDir path unavailable; skipped shortcut creation" "Yellow"
                return
            }
            $appDir = $StudioDataDir
            $launcherPs1 = Join-Path $appDir "launch-studio.ps1"
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

            if (-not (Test-Path -LiteralPath $appDir)) {
                [System.IO.Directory]::CreateDirectory($appDir) | Out-Null
            }

            # Same-install discriminator: per-install opaque id written once at
            # install time and read by both this launcher and the backend
            # (/api/health). Replaces the older sha256(resolved $StudioHome)
            # scheme to (a) avoid leaking the install path on -H 0.0.0.0
            # deployments and (b) sidestep launcher/backend canonicalization
            # drift (Resolve-Path vs Path.resolve() junction handling). Lives
            # at $StudioHome\share\ (not $appDir) so the backend can find it
            # via _STUDIO_ROOT_RESOLVED / "share" / "studio_install_id"
            # regardless of mode. 32 bytes of crypto random -> 64 hex chars.
            $_studioIdDir = Join-Path $StudioHome "share"
            if (-not (Test-Path -LiteralPath $_studioIdDir)) {
                [System.IO.Directory]::CreateDirectory($_studioIdDir) | Out-Null
            }
            $_studioIdFile = Join-Path $_studioIdDir "studio_install_id"
            $_studioRootId = ""
            if ((Test-Path -LiteralPath $_studioIdFile) -and `
                ((Get-Item -LiteralPath $_studioIdFile).Length -gt 0)) {
                $_studioRootId = ([System.IO.File]::ReadAllText($_studioIdFile)).Trim()
            }
            if (-not $_studioRootId) {
                $_idBytes = New-Object byte[] 32
                [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($_idBytes)
                $_studioRootId = -join ($_idBytes | ForEach-Object { $_.ToString('x2') })
                # Atomic write: write to a temp sibling then rename, so a partial
                # install cannot leave a half-written id.
                $_idTmp = $_studioIdFile + ".$PID.tmp"
                [System.IO.File]::WriteAllText($_idTmp, $_studioRootId)
                Move-Item -LiteralPath $_idTmp -Destination $_studioIdFile -Force
            }

            # Env-mode: persist UNSLOTH_STUDIO_HOME (and llama path) so fresh
            # shells don't need to re-export, and bake per-install $portFile /
            # $mutexName so concurrent custom-root launchers cannot serialize
            # through one global mutex on 8888..8908. Default installs get an
            # empty prefix to match pre-PR behavior.
            $studioHomeExport = if ($StudioRedirectMode -eq 'env') {
                # When override == legacy default, llama.cpp stays at
                # ~/.unsloth/llama.cpp (one shared build). Canonicalize the
                # legacy side so the comparison survives path normalization.
                $_legacyStudio = Join-Path $env:USERPROFILE ".unsloth\studio"
                if (Test-Path -LiteralPath $_legacyStudio -PathType Container) {
                    $_legacyStudio = (Resolve-Path -LiteralPath $_legacyStudio).Path
                }
                $_llamaPath = if ($StudioHome -eq $_legacyStudio) {
                    Join-Path $env:USERPROFILE ".unsloth\llama.cpp"
                } else {
                    Join-Path $StudioHome "llama.cpp"
                }
                $_sq = $StudioHome -replace "'", "''"
                $_llama = $_llamaPath -replace "'", "''"
                $_appDirSq = $appDir -replace "'", "''"
                $_appBytes = [Text.Encoding]::UTF8.GetBytes($appDir)
                $_appHash = ([BitConverter]::ToString(
                    [Security.Cryptography.SHA256]::Create().ComputeHash($_appBytes)
                ) -replace '-', '').Substring(0, 16)
                # UNSLOTH_LLAMA_CPP_PATH is a pre-existing user override; only default if unset.
                "`$env:UNSLOTH_STUDIO_HOME = '$_sq'`nif (-not `$env:UNSLOTH_LLAMA_CPP_PATH) {`n    `$env:UNSLOTH_LLAMA_CPP_PATH = '$_llama'`n}`n`$portFile = '$_appDirSq\studio.port'`n`$mutexName = 'Local\UnslothStudioLauncher-$_appHash'`n"
            } else {
                "`$portFile = `$null`n`$mutexName = 'Local\UnslothStudioLauncher'`n"
            }

            $launcherContent = @"
$studioHomeExport`$ErrorActionPreference = 'Stop'
`$basePort = 8888
`$maxPortOffset = 20
`$timeoutSec = 60
`$pollIntervalMs = 1000
`$_ExpectedStudioRootId = '$_studioRootId'

function Test-StudioHealth {
    param([Parameter(Mandatory = `$true)][int]`$Port)
    try {
        `$url = "http://127.0.0.1:`$Port/api/health"
        `$resp = Invoke-RestMethod -Uri `$url -TimeoutSec 1 -Method Get
        if (-not (`$resp -and `$resp.status -eq 'healthy' -and `$resp.service -eq 'Unsloth UI Backend')) { return `$false }
        # why: verify the backend belongs to THIS install via the install-time
        # hex digest; raw path is not leaked over /api/health.
        if (`$_ExpectedStudioRootId -and `$resp.studio_root_id -ne `$_ExpectedStudioRootId) { return `$false }
        return `$true
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
    if (`$portFile) {
        if (Test-Path -LiteralPath `$portFile) {
            `$cached = Get-Content -LiteralPath `$portFile -ErrorAction SilentlyContinue | Select-Object -First 1
            if (`$cached -match '^\d+`$') {
                `$cachedPort = [int]`$cached
                if (Test-StudioHealth -Port `$cachedPort) { return `$cachedPort }
                Remove-Item -LiteralPath `$portFile -Force -ErrorAction SilentlyContinue
            }
        }
        return `$null
    }
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

# If Unsloth is already healthy on any expected port, just open it and exit.
`$existingPort = Find-HealthyStudioPort
if (`$existingPort) {
    Start-Process "http://localhost:`$existingPort"
    exit 0
}

`$launchMutex = [System.Threading.Mutex]::new(`$false, `$mutexName)
`$haveMutex = `$false
try {
    try {
        `$haveMutex = `$launchMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        `$haveMutex = `$true
    }
    if (-not `$haveMutex) {
        # Another launcher is already running; wait for it to bring Unsloth up
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
    # Single-quote the path in the child -Command so `$` / backtick in custom
    # roots don't get reparsed; double any apostrophes so 'O''Brien' survives.
    `$studioCommand = "& '" + (`$studioExe -replace "'", "''") + "' studio -p " + `$launchPort
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
        if (Test-StudioHealth -Port `$launchPort) {
            if (`$portFile) {
                try {
                    [System.IO.File]::WriteAllText(`$portFile, "`$launchPort`n")
                } catch {}
            }
            Start-Process "http://localhost:`$launchPort"
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
            # No .vbs launcher is written. A WScript.Shell .vbs that spawns a hidden
            # ExecutionPolicy-Bypass PowerShell is exactly the shape VBS-dropper
            # heuristics score (e.g. Kaspersky HEUR:Trojan.VBS.Agent.gen). The .lnk
            # shortcuts instead point straight at powershell.exe running
            # launch-studio.ps1 with a hidden window (selected below).

            # Delete any launch-studio.vbs left by a pre-hardening install. New
            # installs no longer generate it, but an upgrade that merely stopped
            # generating it would leave the exact file AV flags on disk, so remove
            # it explicitly. Covers default and env-mode installs (same $appDir).
            $legacyLauncherVbs = Join-Path $appDir "launch-studio.vbs"
            if (Test-Path -LiteralPath $legacyLauncherVbs) {
                Remove-Item -LiteralPath $legacyLauncherVbs -Force -ErrorAction SilentlyContinue
            }

            # Prefer bundled icon from local clone/dev installs.
            # If not available, best-effort download from raw GitHub.
            # We only attach the icon if the resulting file has a valid ICO header.
            # Snapshot the existing icon first so we can tell whether it actually
            # changed and gate the heavier icon-cache refresh on a real change.
            $preIconHash = $null
            if (Test-Path -LiteralPath $iconPath) {
                try { $preIconHash = (Get-FileHash -LiteralPath $iconPath -Algorithm SHA256).Hash } catch {}
            }
            $hasValidIcon = $false
            if ($bundledIcon -and (Test-Path -LiteralPath $bundledIcon)) {
                try {
                    Copy-Item -LiteralPath $bundledIcon -Destination $iconPath -Force
                } catch {
                    Write-Host "[DEBUG] Error copying bundled icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                }
            } elseif (-not (Test-Path -LiteralPath $iconPath)) {
                try {
                    Invoke-WebRequest -Uri $iconUrl -OutFile $iconPath -UseBasicParsing
                } catch {
                    Write-Host "[DEBUG] Error downloading icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                }
            }

            if (Test-Path -LiteralPath $iconPath) {
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
                        Remove-Item -LiteralPath $iconPath -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    Write-Host "[DEBUG] Error validating or removing icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                    Remove-Item -LiteralPath $iconPath -Force -ErrorAction SilentlyContinue
                }
            }

            # Did the icon content actually change vs the previous install?
            # Only a real change (or a first/removed icon) should trigger the heavy
            # refresh; a no-op reinstall with no icon at all must not.
            $iconChanged = $false
            if ($hasValidIcon) {
                if (-not $preIconHash) {
                    $iconChanged = $true
                } else {
                    try {
                        $postIconHash = (Get-FileHash -LiteralPath $iconPath -Algorithm SHA256).Hash
                        $iconChanged = ($postIconHash -ne $preIconHash)
                    } catch { $iconChanged = $true }
                }
            } elseif ($preIconHash) {
                # A previously present icon was removed or invalidated.
                $iconChanged = $true
            }

            # Env-mode: skip persistent Desktop / Start Menu .lnk shortcuts
            # that may point at a deleted workspace; launcher + icon stay.
            if ($StudioRedirectMode -eq 'env') {
                substep "wrote launcher at $launcherPs1 (persistent shortcuts skipped in env-override mode)"
                return
            }

            # Whether this is effectively a first install (no pre-existing .lnk).
            # Used to gate the heavier icon-cache refresh below so a no-op reinstall
            # does not repeatedly clear caches / restart StartMenuExperienceHost --
            # a behavioral cluster AV heuristics can score as dropper-like.
            $firstInstall = -not (
                ($desktopLink -and (Test-Path -LiteralPath $desktopLink)) -or
                ($startMenuLink -and (Test-Path -LiteralPath $startMenuLink))
            )

            # Launch transport for the shortcuts: powershell.exe runs
            # launch-studio.ps1 with a hidden window. We deliberately avoid a
            # .vbs/WScript.Shell wrapper -- that script-engine shape is what AV
            # VBS-dropper heuristics score (Kaspersky HEUR:Trojan.VBS.Agent.gen).
            $powershellForLnk = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
            $shortcutTarget = $powershellForLnk
            $shortcutArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcherPs1`""

            try {
                $wshell = New-Object -ComObject WScript.Shell
                $createdShortcutCount = 0
                $createdShortcutPaths = @()
                foreach ($linkPath in @($desktopLink, $startMenuLink)) {
                    if (-not $linkPath -or [string]::IsNullOrWhiteSpace($linkPath)) { continue }
                    try {
                        $shortcut = $wshell.CreateShortcut($linkPath)
                        $shortcut.TargetPath = $shortcutTarget
                        $shortcut.Arguments = $shortcutArgs
                        $shortcut.WorkingDirectory = $appDir
                        # Start minimized so the brief PowerShell console flash is muted.
                        $shortcut.WindowStyle = 7
                        $shortcut.Description = "Launch Unsloth Studio"
                        if ($hasValidIcon) {
                            $shortcut.IconLocation = "$iconPath,0"
                        }
                        $shortcut.Save()
                        $createdShortcutCount++
                        $createdShortcutPaths += $linkPath
                    } catch {
                        substep "could not create shortcut at ${linkPath}: $($_.Exception.Message)" "Yellow"
                    }
                }
                if ($createdShortcutCount -gt 0) {
                    substep "Created Unsloth Studio shortcut"
                    # Always do the cheap, non-disruptive per-item refresh so a
                    # rewritten same-name .lnk renders with its new target/icon
                    # immediately (a same-name .lnk recreated across reinstalls keeps
                    # Explorer's cached per-item icon). The reliable fix (no explorer
                    # restart) is a per-item SHChangeNotify SHCNE_UPDATEITEM +
                    # SHCNF_PATHW per .lnk; the global SHCNE_ASSOCCHANGED broadcast
                    # alone does NOT recover a stale item.
                    try {
                        Add-Type -Namespace UnslothShell -Name IconRefresh -MemberDefinition '[System.Runtime.InteropServices.DllImport("shell32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public static extern void SHChangeNotify(int eventId, uint flags, string item1, System.IntPtr item2);' -ErrorAction SilentlyContinue
                        # SHCNE_UPDATEITEM (0x00002000) + SHCNF_PATHW (0x0005) per shortcut
                        foreach ($scPath in $createdShortcutPaths) {
                            try { [UnslothShell.IconRefresh]::SHChangeNotify(0x00002000, 0x0005, $scPath, [System.IntPtr]::Zero) } catch {}
                        }
                        # SHCNE_ASSOCCHANGED (0x08000000) global refresh (belt-and-suspenders)
                        [UnslothShell.IconRefresh]::SHChangeNotify(0x08000000, 0, $null, [System.IntPtr]::Zero)
                    } catch {}
                    # Heavier on-disk icon-cache clear + StartMenuExperienceHost tile
                    # rebuild only when the icon actually changed or this is a first
                    # install. Running "clear icon cache + kill StartMenuExperienceHost"
                    # on every no-op reinstall is a dropper-like behavioral cluster and
                    # is unnecessary when the icon is unchanged (the per-item notify
                    # above already refreshes the rewritten shortcut).
                    if ($firstInstall -or $iconChanged) {
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -ClearIconCache 2>$null } catch {}
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -show 2>$null } catch {}
                        # Win11's Start Menu (StartMenuExperienceHost) keeps its OWN
                        # pre-rendered tile-icon cache that ie4uinit/explorer restart do NOT
                        # invalidate, so a rewritten same-name shortcut shows the old tile
                        # until the host restarts. Drop only the render caches (NEVER
                        # start2.bin -- the pinned layout) and let the host rebuild.
                        # Best-effort; Win10 has no such host (Test-Path skips it).
                        try {
                            $smehTemp = Join-Path $env:LOCALAPPDATA "Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState"
                            if (Test-Path -LiteralPath $smehTemp) {
                                Get-ChildItem -LiteralPath $smehTemp -Filter "TileCache_*" -ErrorAction SilentlyContinue |
                                    Remove-Item -Force -ErrorAction SilentlyContinue
                                Remove-Item -LiteralPath (Join-Path $smehTemp "StartUnifiedTileModelCache.dat") -Force -ErrorAction SilentlyContinue
                                Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
                            }
                        } catch {}
                    }
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

    # Regen .lnk + launcher only; used by `unsloth studio update`.
    if ($ShortcutsOnly) {
        if ($TauriMode) { return }
        $UnslothExe = Join-Path $VenvDir "Scripts\unsloth.exe"
        if (-not (Test-Path -LiteralPath $UnslothExe)) {
            Write-Host "[ERROR] unsloth.exe missing at $UnslothExe; run install.ps1 first." -ForegroundColor Red
            # throw (not Exit-InstallFailure) so non-Tauri callers see rc != 0.
            throw "unsloth.exe missing"
        }
        New-StudioShortcuts -UnslothExePath $UnslothExe
        return
    }

    # ── Check winget ──
    # winget is only needed to install Python or uv. If both are
    # already on PATH (Windows ARM64 GitHub-hosted runners, manual
    # python.org + Astral uv installs, corporate locked-down hosts
    # without the Store, etc.) the script can proceed without it.
    # We defer the hard failure to the Python / uv install branches
    # below, where winget is actually invoked.
    Write-TauriLog "STEP" "Checking system dependencies"
    $script:WingetAvailable = [bool](Get-Command winget -ErrorAction SilentlyContinue)
    if ($script:WingetAvailable) {
        step "winget" "available"
    } else {
        step "winget" "not available -- will require Python + uv to be already installed" "Yellow"
        substep "Get it from https://aka.ms/getwinget if Python / uv are not already on PATH." "Yellow"
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
        # Prefer the requested $PythonVersion, then newest-first fallback.
        $minors = @($PythonVersion) + (@("3.13", "3.12", "3.11") | Where-Object { $_ -ne $PythonVersion })
        # Enumerate every py.exe on PATH with -All (Windows PowerShell 5.1
        # returns only the first launcher without it) and search each for a
        # supported, non-conda interpreter.
        foreach ($pyLauncher in @(Get-Command py -All -CommandType Application -ErrorAction SilentlyContinue)) {
            if ($pyLauncher.Source -match $script:CondaSkipPattern) { continue }
            foreach ($minor in $minors) {
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

    # ── Fallback: install CPython directly from python.org ──
    # Used when winget is unavailable or fails (notably msstore cert-pinning error
    # 0x8a15005e, which aborts `winget install` unless --source winget is given).
    # Downloads the official installer and runs it silently as a per-user install
    # (no UAC), putting python.exe + the py launcher on PATH. Mirrors the uv ->
    # astral.sh fallback below. Returns @{ Version; Path } or $null.
    function Install-PythonFromPythonOrg {
        # python.org ships one installer per architecture.
        $archSuffix = switch (Get-TauriDiagArch) {
            "x86_64" { "-amd64" }
            "arm64"  { "-arm64" }
            "x86"    { "" }
            default  { $null }
        }
        if ($null -eq $archSuffix) {
            substep "No python.org installer is available for this architecture." "Yellow"
            return $null
        }

        # Resolve the latest $PythonVersion.x patch from the python.org listing,
        # falling back to a same-minor version if the listing cannot be fetched.
        # Use the pinned full version only when it matches the requested minor so a
        # non-default UNSLOTH_PYTHON (e.g. 3.12) doesn't silently install 3.13.
        $full = if ($PythonFallbackFullVersion -like "$PythonVersion.*") { $PythonFallbackFullVersion } else { "$PythonVersion.0" }
        try {
            $listing = [string](Invoke-RestMethod -Uri "https://www.python.org/ftp/python/" -UseBasicParsing -TimeoutSec 20)
            $patches = [regex]::Matches($listing, ([regex]::Escape($PythonVersion) + '\.(\d+)/')) |
                ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Descending
            if ($patches.Count -gt 0) { $full = "$PythonVersion.$($patches[0])" }
        } catch {}

        $file = "python-$full$archSuffix.exe"
        $url  = "https://www.python.org/ftp/python/$full/$file"
        $dest = Join-Path ([System.IO.Path]::GetTempPath()) $file
        substep "downloading Python $full from python.org..." "Yellow"
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        } catch {
            substep "python.org download failed: $($_.Exception.Message)" "Yellow"
            return $null
        }

        # Per-user install => no UAC. PrependPath puts python + py on PATH;
        # Include_launcher installs py.exe (preferred by Find-CompatiblePython).
        substep "installing Python $full (silent, per-user)..."
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_launcher=1",
            # Launcher per-user too: Include_launcher defaults InstallLauncherAllUsers=1,
            # which needs admin and would break this non-admin per-user fallback.
            "InstallLauncherAllUsers=0",
            "Include_pip=1",
            "AssociateFiles=0",
            "Shortcuts=0"
        )
        $rc = 1
        try {
            $proc = Start-Process -FilePath $dest -ArgumentList $installArgs -Wait -PassThru
            $rc = $proc.ExitCode
        } catch {
            substep "python.org installer failed to start: $($_.Exception.Message)" "Yellow"
        } finally {
            Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
        }
        if ($rc -ne 0) {
            substep "python.org installer exited with code $rc." "Yellow"
        }
        Refresh-SessionPath
        return (Find-CompatiblePython)
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
        $wingetExit = $null

        if ($script:WingetAvailable) {
            # --source winget avoids the msstore source, which can fail with
            # cert-pinning error 0x8a15005e and abort the whole `winget install`
            # (winget then demands --source). Python and uv both live in the
            # winget source, so pinning it is correct and faster.
            #
            # Lower ErrorActionPreference so winget stderr (progress/warnings) is
            # not a terminating error on PS 5.1 (native stderr is ErrorRecord).
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                winget install -e --id $pythonPackageId --source winget --accept-package-agreements --accept-source-agreements
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
                    winget install -e --id $pythonPackageId --source winget --accept-package-agreements --accept-source-agreements --force
                    $wingetExit = $LASTEXITCODE
                } catch { $wingetExit = 1 }
                $ErrorActionPreference = $prevEAP
                Refresh-SessionPath
                $DetectedPython = Find-CompatiblePython
            }
        }

        # Fall back to python.org if winget is unavailable OR couldn't install a
        # working Python (missing/broken winget, msstore cert errors --source
        # winget can't fix). Keeps the install automatic instead of failing out.
        if (-not $DetectedPython) {
            if ($script:WingetAvailable) {
                substep "winget could not install Python -- falling back to python.org..." "Yellow"
            } else {
                substep "winget is unavailable -- installing Python from python.org..." "Yellow"
            }
            $DetectedPython = Install-PythonFromPythonOrg
        }

        if (-not $DetectedPython) {
            $exitNote = if ($null -ne $wingetExit) { " (winget exit code $wingetExit)" } else { "" }
            Write-Host "[ERROR] Python installation failed$exitNote" -ForegroundColor Red
            Write-Host "        Please install Python $PythonVersion manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-Host "        Then re-run this installer." -ForegroundColor Yellow
            return (Exit-InstallFailure "Python installation failed")
        }
    }
    $DiagPythonVersion = $PythonVersion
    if ($DetectedPython) { $DiagPythonVersion = $DetectedPython.Version }
    $InitialGpuBranch = "unknown"
    if ($SkipTorch) { $InitialGpuBranch = "no_torch" }
    Write-TauriDiag -GpuBranch $InitialGpuBranch -TorchIndexFamily "none" -PythonVersionForDiag $DiagPythonVersion

    # ── Install uv ──
    Write-TauriLog "STEP" "Installing uv package manager"
    $UvMinVersion = "0.8.16"
    function Test-UvVersionOk {
        $cmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $cmd) { return $false }
        try {
            $raw = (& uv --version 2>$null | Select-Object -First 1)
        } catch {
            return $false
        }
        if ($raw -notmatch 'uv\s+([0-9]+(?:\.[0-9]+)+)') { return $false }
        try {
            return ([version]$Matches[1] -ge [version]$UvMinVersion)
        } catch {
            return $false
        }
    }

    if (-not (Test-UvVersionOk)) {
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            substep "updating uv package manager..."
        } else {
            substep "installing uv package manager..."
        }
        if ($script:WingetAvailable) {
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try { winget upgrade --id=astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements } catch {}
            if (-not (Test-UvVersionOk)) {
                try { winget install --id=astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements } catch {}
            }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath
        }
        # Fallback: if winget is unavailable or didn't put uv on PATH,
        # use Astral's official PowerShell installer. This is the only
        # supported path on hosts without winget (Windows ARM64 runners,
        # corporate machines without the Store, etc.).
        if (-not (Test-UvVersionOk)) {
            substep "installing uv via https://astral.sh/uv/install.ps1..." "Yellow"
            Invoke-Expression (Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1")
            Refresh-SessionPath
        }
    }

    # A freshly installed uv can sit later on PATH than an older one (active
    # venv, Scoop/pipx shim). Prefer a just-installed uv from a known location.
    if (-not (Test-UvVersionOk)) {
        $origPath = $env:PATH
        foreach ($d in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME,
                         (Join-Path $env:USERPROFILE ".local\bin"),
                         (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"))) {
            if ($d -and (Test-Path $d)) {
                $env:PATH = "$d;$origPath"
                if (Test-UvVersionOk) { break }
                $env:PATH = $origPath
            }
        }
    }

    if (-not (Test-UvVersionOk)) {
        step "uv" "could not be installed" "Red"
        substep "Install it from https://docs.astral.sh/uv/" "Yellow"
        return (Exit-InstallFailure "uv could not be installed")
    }

    # When bytecode compilation is enabled, large installs can exceed uv's 60s
    # default on slow machines. Default to 180s, preserving overrides ("0" disables).
    if (-not $env:UV_COMPILE_BYTECODE_TIMEOUT) {
        $env:UV_COMPILE_BYTECODE_TIMEOUT = "180"
    }

    # uv >= 0.8.16 retries HTTP/2 streaming body errors; raise retries and read
    # timeout for large wheel downloads. User-provided values are preserved.
    if (-not $env:UV_HTTP_RETRIES) {
        $env:UV_HTTP_RETRIES = "5"
    }
    if (-not $env:UV_HTTP_TIMEOUT) {
        $env:UV_HTTP_TIMEOUT = "180"
    }

    # ── Create venv (migrate old layout if possible, otherwise fresh) ──
    # Pass the resolved executable path to uv so it does not re-resolve
    # a version string back to a conda interpreter.
    Write-TauriLog "STEP" "Creating virtual environment"
    if (-not (Test-Path -LiteralPath $StudioHome)) {
        # .NET API: New-Item -Path treats brackets as wildcards.
        [System.IO.Directory]::CreateDirectory($StudioHome) | Out-Null
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $_Migrated = $false
    $script:StudioVenvRollbackDir = $null
    $script:StudioVenvRollbackTarget = $VenvDir
    $script:StudioVenvRollbackActive = $false

    function Start-StudioVenvRollback {
        param([Parameter(Mandatory = $true)][string]$ExistingDir)
        $stamp = Get-Date -Format "yyyyMMddHHmmss"
        $candidate = Join-Path $StudioHome "unsloth_studio.rollback.$stamp.$PID"
        $suffix = 0
        # -LiteralPath: a custom $StudioHome may contain [ ] * ? which
        # plain Test-Path / Move-Item would interpret as wildcards.
        while (Test-Path -LiteralPath $candidate) {
            $suffix++
            $candidate = Join-Path $StudioHome "unsloth_studio.rollback.$stamp.$PID.$suffix"
        }
        Move-Item -LiteralPath $ExistingDir -Destination $candidate -ErrorAction Stop
        $script:StudioVenvRollbackDir = $candidate
        $script:StudioVenvRollbackTarget = $ExistingDir
        $script:StudioVenvRollbackActive = $true
        substep "previous environment preserved for rollback"
    }

    function Restore-StudioVenvRollback {
        if (-not $script:StudioVenvRollbackActive) { return }
        $backup = $script:StudioVenvRollbackDir
        $target = $script:StudioVenvRollbackTarget
        if (-not $backup -or -not (Test-Path -LiteralPath $backup)) {
            $script:StudioVenvRollbackActive = $false
            return
        }
        substep "restoring previous environment after failed install..." "Yellow"
        try {
            if (Test-Path -LiteralPath $target) {
                Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
            }
            Move-Item -LiteralPath $backup -Destination $target -Force -ErrorAction Stop
            substep "restored previous environment"
            $script:StudioVenvRollbackActive = $false
            $script:StudioVenvRollbackDir = $null
        } catch {
            Write-Host "[WARN] Could not restore previous environment from $backup to $target" -ForegroundColor Yellow
            Write-Host "       $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    function Complete-StudioVenvRollback {
        if (-not $script:StudioVenvRollbackActive) { return }
        $backup = $script:StudioVenvRollbackDir
        if ($backup -and (Test-Path -LiteralPath $backup)) {
            Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue
        }
        $script:StudioVenvRollbackActive = $false
        $script:StudioVenvRollbackDir = $null
    }

    if (Test-Path -LiteralPath $VenvPython) {
        # why: matching guard to the .venv branch below -- in env-mode
        # $StudioHome is a user-chosen workspace, so refuse to nuke an
        # existing $StudioHome\unsloth_studio that lacks Unsloth sentinels.
        # -PathType Leaf rejects a directory at the sentinel path. Accept the
        # in-VENV ownership marker so partial-install retries are not blocked.
        if (
            $StudioRedirectMode -eq 'env' -and
            -not (Test-Path -LiteralPath (Join-Path $VenvDir ".unsloth-studio-owned") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "share\studio.conf") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe") -PathType Leaf)
        ) {
            Write-Host "[ERROR] $VenvDir already exists but does not look like an Unsloth Studio install." -ForegroundColor Red
            Write-Host "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME." -ForegroundColor Yellow
            throw "Refusing to delete non-Unsloth venv at $VenvDir"
        }
        # New layout already exists -- replace only after preserving rollback copy.
        substep "preserving existing environment for rollback..."
        try {
            Start-StudioVenvRollback -ExistingDir $VenvDir
        } catch {
            Write-Host "[ERROR] Could not prepare existing environment for reinstall: $($_.Exception.Message)" -ForegroundColor Red
            return (Exit-InstallFailure "Could not prepare existing environment for reinstall")
        }
    } elseif (
        $StudioRedirectMode -ne 'env' `
        -and (Test-Path -LiteralPath (Join-Path $StudioHome ".venv\Scripts\python.exe"))
    ) {
        # Old layout (~/.unsloth/studio/.venv) exists -- validate before migrating.
        # Skip in env-mode so we don't blow away an unrelated .venv at the
        # workspace root (e.g. user's existing project Python venv).
        $OldVenv = Join-Path $StudioHome ".venv"
        $OldPy = Join-Path $OldVenv "Scripts\python.exe"
        substep "found legacy Unsloth environment, validating..."
        $prevEAP2 = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($SkipTorch) {
                & $OldPy -c "import sys; print(sys.executable)" 2>$null | Out-Null
            } else {
                & $OldPy -c "import torch; A = torch.ones((2,2)); B = A + A" 2>$null | Out-Null
            }
            $legacyOk = ($LASTEXITCODE -eq 0)
        } catch { $legacyOk = $false }
        $ErrorActionPreference = $prevEAP2
        if ($legacyOk) {
            substep "legacy environment is healthy -- migrating..."
            Move-Item -LiteralPath $OldVenv -Destination $VenvDir -Force
            substep "moved .venv -> unsloth_studio"
            $_Migrated = $true
        } else {
            substep "legacy environment failed validation -- creating fresh environment" "Yellow"
            $invalidVenv = Join-Path $StudioHome (".venv.invalid.{0}.{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), $PID)
            Move-Item -LiteralPath $OldVenv -Destination $invalidVenv -Force -ErrorAction SilentlyContinue
        }
    } elseif (
        $StudioRedirectMode -ne 'env' `
        -and (Test-Path -LiteralPath (Join-Path $env:USERPROFILE "unsloth_studio\Scripts\python.exe"))
    ) {
        # CWD-relative venv from old install.ps1 -> migrate to absolute path.
        # Skip in env-mode so we don't relocate the default-install venv into
        # the workspace root.
        $CwdVenv = Join-Path $env:USERPROFILE "unsloth_studio"
        substep "found CWD-relative Unsloth environment, migrating to $VenvDir..."
        Move-Item -LiteralPath $CwdVenv -Destination $VenvDir -Force
        substep "moved ~/unsloth_studio -> ~/.unsloth/studio/unsloth_studio"
        $_Migrated = $true
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        step "venv" "creating Python $($DetectedPython.Version) virtual environment"
        substep "$VenvDir"
        $venvExit = Invoke-InstallCommand { uv venv $VenvDir --python "$($DetectedPython.Path)" }
        if ($venvExit -ne 0) {
            Write-Host "[ERROR] Failed to create virtual environment (exit code $venvExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to create virtual environment (exit code $venvExit)" $venvExit)
        }
    } else {
        step "venv" "using migrated environment"
        substep "$VenvDir"
    }

    # Mark the freshly-created venv as Unsloth-owned so a partial install can be
    # repaired by re-running install.ps1; the env-mode deletion guard above
    # accepts this marker as the primary sentinel.
    if (Test-Path -LiteralPath $VenvDir -PathType Container) {
        try { [System.IO.File]::WriteAllText((Join-Path $VenvDir ".unsloth-studio-owned"), "") } catch {}
    }

    # ── Helper: run amd-smi without triggering a UAC elevation prompt ──
    # amd-smi on Windows auto-elevates to read GPU/APU memory, surfacing a confusing
    # DiskPart UAC prompt mid-install (Unsloth backend amd.py hits the same).
    # __COMPAT_LAYER=RunAsInvoker forces it (and helpers it spawns) to run
    # un-elevated; on failure the WMI name -> gfx fallback still resolves the arch.
    function Invoke-AmdSmiNoElevate {
        param(
            [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
            [Parameter(Position = 1)][string[]]$SmiArgs = @(),
            [int]$TimeoutSec = 30
        )
        # RunAsInvoker blocks the auto-elevation/UAC prompt; the timeout bounds a
        # flaky amd-smi that can otherwise spin for minutes (30s mirrors amd.py).
        $prevCompat = [Environment]::GetEnvironmentVariable('__COMPAT_LAYER', 'Process')
        $env:__COMPAT_LAYER = 'RunAsInvoker'
        try {
            # [Process]::Start, NOT Start-Process -PassThru: the latter leaves
            # .ExitCode $null after WaitForExit on PS 5.1, so $LASTEXITCODE (checked
            # by callers) reads non-zero and kills detection. Async reads drain the
            # pipes (no deadlock); amd-smi args have no spaces so a plain join is safe.
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Exe
            $psi.Arguments = ($SmiArgs -join ' ')
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
                try { $proc.Kill() } catch {}
                $global:LASTEXITCODE = 124
                return ""
            }
            $global:LASTEXITCODE = $proc.ExitCode
            return ($outTask.Result + "`n" + $errTask.Result)
        } catch {
            $global:LASTEXITCODE = 1
            return ""
        } finally {
            if ($null -eq $prevCompat) {
                Remove-Item Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
            } else {
                $env:__COMPAT_LAYER = $prevCompat
            }
        }
    }

    # ── Helper: run nvidia-smi under a timeout ──
    # A wedged NVIDIA driver can make nvidia-smi block during init or after a
    # reset; WaitForExit bounds it (mirrors Invoke-AmdSmiNoElevate) so detection
    # cannot hang the installer. No RunAsInvoker compat layer: nvidia-smi does
    # not auto-elevate. Returns combined stdout+stderr; "" on timeout/failure.
    function Invoke-NvidiaSmiBounded {
        param(
            [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
            [Parameter(Position = 1)][string[]]$SmiArgs = @(),
            [int]$TimeoutSec = 10
        )
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Exe
            $psi.Arguments = ($SmiArgs -join ' ')
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
                try { $proc.Kill() } catch {}
                $global:LASTEXITCODE = 124
                return ""
            }
            $global:LASTEXITCODE = $proc.ExitCode
            return ($outTask.Result + "`n" + $errTask.Result)
        } catch {
            $global:LASTEXITCODE = 1
            return ""
        }
    }

    # ── Helper: nvidia-smi -L lists at least one real GPU ──
    # Exit code 0 alone is not enough: a stale/driverless nvidia-smi can exit 0
    # while listing no GPU, which would mark an AMD host NVIDIA and suppress
    # ROCm detection. Require a "GPU <n>:" data row.
    function Test-NvidiaSmiHasGpu {
        param([Parameter(Mandatory = $true)][string]$Exe)
        $out = Invoke-NvidiaSmiBounded $Exe @('-L')
        return ($LASTEXITCODE -eq 0 -and $out -match '(?m)^GPU\s+\d+:')
    }

    # ── Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) ──
    $HasNvidiaSmi = $false
    $NvidiaSmiExe = $null
    try {
        $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvSmiCmd -and (Test-NvidiaSmiHasGpu $nvSmiCmd.Source)) {
            $HasNvidiaSmi = $true; $NvidiaSmiExe = $nvSmiCmd.Source
        }
    } catch {}
    if (-not $HasNvidiaSmi) {
        foreach ($p in @(
            "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "$env:SystemRoot\System32\nvidia-smi.exe"
        )) {
            if (Test-Path $p) {
                try {
                    if (Test-NvidiaSmiHasGpu $p) { $HasNvidiaSmi = $true; $NvidiaSmiExe = $p; break }
                } catch {}
            }
        }
    }
    # ── AMD ROCm detection (Windows) — mirrors setup.ps1 ──
    $HasROCm = $false
    $HipSdkInstalled = $false   # HIP SDK binary found (independent of device accessibility)
    $ROCmGpuLabel = $null
    $ROCmVersion = $null
    $ROCmGfxArch = $null
    if (-not $HasNvidiaSmi) {
        # hipinfo: PATH first, then HIP_PATH/ROCM_PATH bin fallback (mirrors NVIDIA smi path resolution).
        # AMD HIP SDK sets HIP_PATH but may not add the bin dir to PATH depending on install type.
        # Ignore the venv hipInfo.exe (AMD wheel, on PATH): not a HIP SDK, so
        # amd-smi would still auto-elevate. Cf. _path_inside_venv().
        function Test-HipinfoIsVenvInternal {
            param([AllowNull()][string]$HipinfoPath)
            if ([string]::IsNullOrWhiteSpace($HipinfoPath)) { return $false }
            # Also derive the venv from the setup python + default Unsloth home, so
            # the venv hipInfo is caught when VenvDir/VIRTUAL_ENV are unset.
            $venvRoots = @()
            if ($env:VIRTUAL_ENV) { $venvRoots += $env:VIRTUAL_ENV }
            $vd = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue
            if ($vd) { $venvRoots += $vd }
            if ($env:UNSLOTH_SETUP_PYTHON) {
                try { $venvRoots += (Split-Path -Parent (Split-Path -Parent $env:UNSLOTH_SETUP_PYTHON)) } catch {}
            }
            if ($env:USERPROFILE) { $venvRoots += (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio") }
            # A custom Unsloth home (UNSLOTH_STUDIO_HOME / STUDIO_HOME alias) moves the
            # venv off the default path; seed it too or its hipInfo escapes the filter.
            $studioHomeEnv = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() } else { $null }
            if ($studioHomeEnv) {
                # Expand a leading ~ like the canonical resolver; else GetFullPath
                # keeps the literal ~ (cwd-relative) and the hipInfo escapes the filter.
                if (($studioHomeEnv -eq "~" -or $studioHomeEnv -like "~/*" -or $studioHomeEnv -like "~\*") -and -not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                    # A bare "~" leaves an empty child path; Join-Path rejects that on
                    # PS 5.1, so use USERPROFILE directly and only join a real remainder.
                    $studioHomeRest = $studioHomeEnv.Substring(1).TrimStart('/', '\')
                    $studioHomeEnv = if ($studioHomeRest) { Join-Path $env:USERPROFILE $studioHomeRest } else { $env:USERPROFILE }
                }
                $venvRoots += (Join-Path $studioHomeEnv "unsloth_studio")
            }
            try { $hip = [System.IO.Path]::GetFullPath($HipinfoPath).TrimEnd('\', '/') } catch { return $false }
            foreach ($root in $venvRoots) {
                if ([string]::IsNullOrWhiteSpace($root)) { continue }
                try { $r = [System.IO.Path]::GetFullPath($root).TrimEnd('\', '/') } catch { continue }
                # Skip a bare drive root (e.g. a non-venv UNSLOTH_SETUP_PYTHON like
                # C:\Python311\python.exe yields C:) -- it would match every path on that drive.
                if ($r -match '^[a-zA-Z]:$') { continue }
                if ($hip.Equals($r, [System.StringComparison]::OrdinalIgnoreCase) -or
                    $hip.StartsWith($r + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                    return $true
                }
            }
            return $false
        }
        # Scan all hipinfo and keep the first non-venv one (the venv copy from the
        # bnb fix could shadow a real HIP SDK's). -CommandType Application matches
        # only real executables, not a user alias/function named hipinfo.
        $hipinfoExe = Get-Command hipinfo -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-HipinfoIsVenvInternal $_.Source) } |
            Select-Object -First 1
        if (-not $hipinfoExe) {
            # Iterate the env roots (mirrors the Python list) and take the first non-venv
            # bin\hipinfo.exe, so a venv-internal HIP_PATH can't mask a real SDK in ROCM_PATH.
            $hipMissingLabel = $null; $hipMissingRoot = $null; $hipMissingCandidate = $null
            foreach ($hipEnvLabel in @("HIP_PATH", "HIP_PATH_57", "ROCM_PATH")) {
                $hipRoot = [Environment]::GetEnvironmentVariable($hipEnvLabel)
                if ([string]::IsNullOrWhiteSpace($hipRoot)) { continue }
                $hipinfoCandidate = Join-Path $hipRoot "bin\hipinfo.exe"
                if (-not (Test-Path $hipinfoCandidate)) {
                    if (-not $hipMissingLabel) { $hipMissingLabel = $hipEnvLabel; $hipMissingRoot = $hipRoot; $hipMissingCandidate = $hipinfoCandidate }
                    continue
                }
                if (Test-HipinfoIsVenvInternal $hipinfoCandidate) { continue }   # venv copy (AMD wheel): not a HIP SDK
                Write-Host "  [WARN] hipinfo not on PATH -- located via ${hipEnvLabel}: $hipinfoCandidate" -ForegroundColor Yellow
                Write-Host "         Add '$(Join-Path $hipRoot 'bin')' to your PATH to suppress this warning" -ForegroundColor Yellow
                Write-Host "         Quick fix: [Environment]::SetEnvironmentVariable('PATH',`$env:PATH+';$(Join-Path $hipRoot 'bin')','User')" -ForegroundColor Yellow
                $hipinfoExe = [PSCustomObject]@{ Source = $hipinfoCandidate }
                break
            }
            if ((-not $hipinfoExe) -and $hipMissingLabel) {
                Write-Host "  [WARN] ${hipMissingLabel}=$hipMissingRoot is set but hipinfo.exe not found at $hipMissingCandidate" -ForegroundColor Yellow
                Write-Host "         HIP SDK install may be incomplete -- re-install from:" -ForegroundColor Yellow
                Write-Host "         https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" -ForegroundColor Yellow
            }
        }
        if ($hipinfoExe) {
            $HipSdkInstalled = $true   # binary found → SDK is installed regardless of device state
            try {
                $hipOut = & $hipinfoExe.Source 2>&1 | Out-String
                if ($hipOut -match "(?i)gcnArchName") {
                    # hipinfo can crash after printing gcnArchName (#6043).
                    # Once the arch is printed, keep the ROCm wheel path.
                    $HasROCm = $true
                    $_hipAllArches = @([regex]::Matches($hipOut, "(?im)^\s*gcnArchName\s*:\s*(\S+)") | ForEach-Object { ($_.Groups[1].Value -split ':')[0].Trim().ToLower() })
                    $_hipVisIdx = if ($env:HIP_VISIBLE_DEVICES -match '^\d') { [int]($env:HIP_VISIBLE_DEVICES -split ',')[0] } elseif ($env:ROCR_VISIBLE_DEVICES -match '^\d') { [int]($env:ROCR_VISIBLE_DEVICES -split ',')[0] } else { 0 }
                    if ($_hipAllArches.Count -gt 0) {
                        $ROCmGfxArch  = if ($_hipVisIdx -lt $_hipAllArches.Count) { $_hipAllArches[$_hipVisIdx] } else { $_hipAllArches[0] }
                        $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                    } else {
                        $ROCmGpuLabel = "AMD ROCm"
                    }
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "  [INFO] hipinfo exited with code $LASTEXITCODE but reported gcnArchName -- treating as ROCm-capable (see #6043)" -ForegroundColor Cyan
                    }
                } elseif ($LASTEXITCODE -ne 0) {
                    # hipinfo ran but returned a HIP runtime error without any gcnArchName
                    # output (e.g. "no ROCm-capable device detected"), or crashed before
                    # printing device info.
                    $firstLine = ($hipOut -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -First 1)
                    Write-Host "  [WARN] hipinfo returned a HIP runtime error (exit $LASTEXITCODE)" -ForegroundColor Yellow
                    Write-Host "         $firstLine" -ForegroundColor Yellow
                    Write-Host "         Ensure ROCm drivers are installed: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" -ForegroundColor Yellow
                }
            } catch {}
        }
        # On hosts without a working HIP runtime amd-smi elevates a child at runtime,
        # popping a UAC/DiskPart prompt RunAsInvoker can't suppress (manifest is
        # asInvoker). So only probe when a HIP SDK is present (hipinfo found ->
        # un-elevated) or the user opts in; else fall through to WMI name inference
        # (enough to pick ROCm wheels + the ROCm llama.cpp prebuilt).
        # An explicit opt-out (UNSLOTH_ENABLE_AMD_SMI=0/false/no/off) wins over the
        # HIP-SDK heuristic: a HIP SDK binary with a broken runtime can still pop the
        # prompt, so $HipSdkInstalled must NOT silently re-enable it.
        $amdSmiOptOut = $env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(0|false|no|off)$'
        $amdSmiAllowed = (-not $amdSmiOptOut) -and ($HipSdkInstalled -or ($env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(1|true|yes|on)$'))
        if (-not $HasROCm -and $amdSmiAllowed) {
            $amdSmiExe = Get-Command "amd-smi" -ErrorAction SilentlyContinue
            if ($amdSmiExe) {
                try {
                    $smiOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('list')
                    if ($LASTEXITCODE -eq 0 -and $smiOut -match "(?im)^GPU\s*[:\[]\s*\d") {
                        $HasROCm = $true
                        # Mirror the hipinfo path: collect all gfx tokens in enumeration
                        # order and pick the runtime-visible one via HIP_VISIBLE_DEVICES.
                        $_smiVisIdx = if ($env:HIP_VISIBLE_DEVICES -match '^\d') { [int]($env:HIP_VISIBLE_DEVICES -split ',')[0] } elseif ($env:ROCR_VISIBLE_DEVICES -match '^\d') { [int]($env:ROCR_VISIBLE_DEVICES -split ',')[0] } else { 0 }
                        # Attempt 1: newer amd-smi versions embed the gfx arch in list output.
                        $_smiGfxTokens = @([regex]::Matches($smiOut, "(?i)\b(gfx\d+[a-z]?)\b") | ForEach-Object { $_.Groups[1].Value.ToLower() })
                        if ($_smiGfxTokens.Count -gt 0) {
                            $ROCmGfxArch = if ($_smiVisIdx -lt $_smiGfxTokens.Count) { $_smiGfxTokens[$_smiVisIdx] } else { $_smiGfxTokens[0] }
                            $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                        } else {
                            # Attempt 2: 'static --asic' exposes ASIC details on ROCm 6+,
                            # including the GFX target needed for wheel index selection.
                            $smiAsicOut = ""
                            try { $smiAsicOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('static','--asic') } catch {}
                            $_asicGfxTokens = @([regex]::Matches($smiAsicOut, "(?i)\b(gfx\d+[a-z]?)\b") | ForEach-Object { $_.Groups[1].Value.ToLower() })
                            if ($_asicGfxTokens.Count -gt 0) {
                                $ROCmGfxArch = if ($_smiVisIdx -lt $_asicGfxTokens.Count) { $_asicGfxTokens[$_smiVisIdx] } else { $_asicGfxTokens[0] }
                                $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                            } elseif ($smiAsicOut -match "(?im)Market.?Name\s*[:\|]\s*([^\r\n]+)") {
                                $ROCmGpuLabel = "AMD ROCm ($($Matches[1].Trim()))"
                            } else {
                                $ROCmGpuLabel = "AMD ROCm"
                            }
                        }
                    }
                } catch {}
            }
        }
        if (-not $HasROCm) {
            try {
                $wmiGpu = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match "AMD|Radeon" } |
                    Select-Object -First 1
                if ($wmiGpu) { $ROCmGpuLabel = $wmiGpu.Name }
            } catch {}
        }
        # ── Arch resolution: env-var override → name inference ──────────────
        # Runs even when the probe can't confirm a runtime ($HasROCm false): the
        # WMI-name gfx arch drives both ROCm llama.cpp and torch. repo.amd.com
        # wheels bundle their own runtime (no HIP SDK), so a mapped arch installs
        # ROCm torch directly below -- no wasted CPU base.
        if (-not $ROCmGfxArch) {
            # 1. Manual override: set UNSLOTH_ROCM_GFX_ARCH=gfx1151 before running.
            if ($env:UNSLOTH_ROCM_GFX_ARCH) {
                $ROCmGfxArch = $env:UNSLOTH_ROCM_GFX_ARCH.Trim().ToLower()
                $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $ROCmGfxArch" "Cyan"
            }
            # 2. Best-effort name → arch lookup from marketing name (amd-smi / WMI).
            #    Targets only arches the ROCm prebuilts cover
            #    (gfx120X/110X/1151/1150/103X); unknown names fall back to CPU.
            elseif ($ROCmGpuLabel) {
                $nameArchTable = @(
                    @{ P = "9070 XT|9080";                                        A = "gfx1201" }  # RDNA 4 (RX 9070 XT / 9080)
                    @{ P = "9070|9060";                                           A = "gfx1200" }  # RDNA 4 (RX 9070 / 9060)
                    @{ P = "8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max"; A = "gfx1151" }  # RDNA 3.5 (Strix Halo: Radeon 8060S/8050S/8040S iGPU, Ryzen AI Max+)
                    @{ P = "890M|880M|860M|840M|Strix Point|Krackan|HX 37[05]|AI 9 HX|AI 9 36[05]|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33"; A = "gfx1150" }  # RDNA 3.5 (Strix/Krackan Point: Radeon 890M/880M iGPU, Ryzen AI 9 HX 370/375)
                    @{ P = "RX 7900|RX 7800|RX 7700(?!S)|PRO W7900|PRO W7800|PRO W7700"; A = "gfx1100" }  # RDNA 3 desktop/workstation (Navi 31)
                    @{ P = "RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500|PRO V710"; A = "gfx1102" }  # RDNA 3 (Navi 33)
                    @{ P = "780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme"; A = "gfx1103" }  # RDNA 3 iGPU (Phoenix / Hawk Point)
                    @{ P = "RX 6900|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900";  A = "gfx1030" }  # RDNA 2 (Navi 21) -- gfx103X family
                    @{ P = "RX 6650|RX 6600|PRO W6600|PRO W6650";                  A = "gfx1032" }  # RDNA 2 (Navi 23) -- gfx103X family
                    @{ P = "RX 6500|RX 6400|RX 6300|PRO W6400|PRO W6500";          A = "gfx1034" }  # RDNA 2 (Navi 24) -- gfx103X family
                )
                foreach ($row in $nameArchTable) {
                    if ($ROCmGpuLabel -match $row.P) {
                        $ROCmGfxArch = $row.A
                        $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                        substep "gfx arch inferred from GPU name: $ROCmGfxArch" "Cyan"
                        substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$ROCmGfxArch to skip inference next time" "Cyan"
                        break
                    }
                }
            }
        }
        # Capture ROCm version for wheel selection (hipconfig, then amd-smi).
        # Run whenever the HIP SDK binary is present, not just when the device is accessible --
        # hipconfig --version works even when hipinfo reports no ROCm device (driver issue).
        if ($HasROCm -or $HipSdkInstalled) {
            $hipConfigExe = Get-Command hipconfig -ErrorAction SilentlyContinue
            if (-not $hipConfigExe) {
                $hipRoot = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { $null }
                if ($hipRoot) {
                    $hipConfigCandidate = Join-Path $hipRoot "bin\hipconfig.exe"
                    if (Test-Path $hipConfigCandidate) {
                        $hipConfigEnvLabel = if ($env:HIP_PATH) { "HIP_PATH" } else { "ROCM_PATH" }
                        Write-Host "  [WARN] hipconfig not on PATH -- located via ${hipConfigEnvLabel}: $hipConfigCandidate" -ForegroundColor Yellow
                        $hipConfigExe = [PSCustomObject]@{ Source = $hipConfigCandidate }
                    }
                }
            }
            if ($hipConfigExe) {
                try {
                    $hipVerOut = & $hipConfigExe.Source --version 2>&1 | Out-String
                    if ($LASTEXITCODE -eq 0) {
                        $hipVerLine = ($hipVerOut -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
                        if ($hipVerLine -match '(\d+\.\d+)') {
                            $ROCmVersion     = $Matches[1]
                            $ROCmVersionFull = $hipVerLine
                        }
                    }
                } catch {}
            }
            if (-not $ROCmVersion -and $amdSmiAllowed) {
                $amdSmiVer = Get-Command "amd-smi" -ErrorAction SilentlyContinue
                if ($amdSmiVer) {
                    try {
                        $smiVerOut = Invoke-AmdSmiNoElevate $amdSmiVer.Source @('version')
                        if ($LASTEXITCODE -eq 0 -and $smiVerOut -match 'ROCm version:\s*(\d+\.\d+)') {
                            $ROCmVersion = $Matches[1]
                        }
                    } catch {}
                }
            }
        }
    }

    # ── Optional WSL-ROCm driver hint ────────────────────────────────────────
    # An AMD GPU can also be used inside WSL2, but only with Adrenalin >= 26.2.2
    # (first production ROCDXG/WSL release); native Windows GPU works with any
    # recent driver. We can't auto-install it (AMD referrer-gates downloads, no
    # winget package), so just point at AMD's page. Shown only when the installed
    # driver predates 26.2.2 (Feb 2026); suppress with UNSLOTH_SKIP_AMD_DRIVER_HINT=1.
    function Show-AmdWslDriverHint {
        if ($env:UNSLOTH_SKIP_AMD_DRIVER_HINT) { return }
        try {
            $amd = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1
            if (-not $amd) { return }
            $drvDate = $null
            try {
                if ($amd.DriverDate -is [datetime]) {
                    # Get-CimInstance returns DriverDate already parsed.
                    $drvDate = $amd.DriverDate
                } elseif ($amd.DriverDate) {
                    # Get-WmiObject style WMI datetime string.
                    $drvDate = [Management.ManagementDateTimeConverter]::ToDateTime([string]$amd.DriverDate)
                }
            } catch {}
            # Older than 26.2.2 (Feb 2026) => can't expose the GPU to WSL ROCm.
            # Unreadable date => still show the hint (informational, suppressible).
            if ($drvDate -and $drvDate -ge (Get-Date '2026-02-01')) { return }
            substep "Tip: to use this GPU inside WSL too, install AMD Adrenalin 26.2.2+ (for WSL2)." "Cyan"
            substep "  Your current driver predates it; native Windows GPU is unaffected. Get it from AMD:" "Cyan"
            substep "    https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html" "Cyan"
            substep "  Then reboot and run this installer inside an Ubuntu-24.04 WSL distro." "Cyan"
            # If WSL isn't installed yet, point at the command that provisions it
            # (best-effort; wsl.exe absent => no WSL).
            $hasWsl = $false
            try { $hasWsl = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue) } catch {}
            if (-not $hasWsl) {
                substep "  No WSL yet? Install it first:  wsl --install -d Ubuntu-24.04" "Cyan"
            }
            substep "  (suppress: set UNSLOTH_SKIP_AMD_DRIVER_HINT=1)" "Cyan"
        } catch {}
    }

    if ($HasNvidiaSmi) {
        step "gpu" "NVIDIA GPU detected"
    } elseif ($HasROCm) {
        step "gpu" $ROCmGpuLabel
        $hipSdkPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "on system PATH" }
        substep "HIP SDK: $hipSdkPath"
        if ($ROCmVersionFull) { substep "hipconfig: $ROCmVersionFull" }
    } elseif ($HipSdkInstalled -and $ROCmGpuLabel) {
        # HIP SDK is installed but ROCm can't see the device (driver issue, not SDK issue)
        $sdkVer = if ($ROCmVersionFull) { " (HIP $ROCmVersionFull)" } else { "" }
        step "gpu" "AMD GPU detected -- not ROCm-accessible$sdkVer" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        substep "[WARN] HIP SDK is installed but hipinfo reports no ROCm-capable device." "Yellow"
        substep "       This is a driver issue, not an SDK issue." "Yellow"
        substep "       Ensure the ROCm compute driver is installed alongside the display driver:" "Yellow"
        substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    } elseif ($ROCmGfxArch) {
        # Known arch: Unsloth setup installs AMD's bundled-runtime ROCm PyTorch wheels
        # (repo.amd.com), which ship their own runtime -- HIP SDK optional.
        step "gpu" "AMD ROCm ($ROCmGfxArch)" "Cyan"
        substep "Detected: $ROCmGpuLabel" "Cyan"
        substep "GPU PyTorch uses AMD's bundled-runtime ROCm wheels -- HIP SDK not required (optional)." "Cyan"
    } elseif ($ROCmGpuLabel) {
        step "gpu" "AMD GPU detected -- arch unknown" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        substep "Could not determine the GPU arch -- install the HIP SDK or set" "Yellow"
        substep "UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm PyTorch:" "Yellow"
        substep "https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    } else {
        step "gpu" "none (chat-only / GGUF)" "Yellow"
        substep "Training and GPU inference require an NVIDIA or AMD ROCm GPU." "Yellow"
    }
    # On an AMD GPU (no NVIDIA), surface the optional WSL-ROCm driver hint.
    if (-not $HasNvidiaSmi -and ($ROCmGfxArch -or $ROCmGpuLabel)) { Show-AmdWslDriverHint }

    # Trim trailing slashes from the URL PATH only, preserving ?query / #fragment: a whole-URL
    # TrimEnd corrupts a token ending in "/", a single strip leaves .../cu128// empty. Shared.
    function Trim-IndexPathSlashes {
        param([string]$Url)
        $value = $Url.Trim()
        $idx = $value.IndexOfAny([char[]]@('?', '#'))
        if ($idx -lt 0) {
            return $value.TrimEnd('/')
        }
        return $value.Substring(0, $idx).TrimEnd('/') + $value.Substring($idx)
    }

    # ── Choose the correct PyTorch index URL based on driver CUDA version ──
    # Mirrors Get-PytorchCudaTag in setup.ps1.
    function Get-TorchIndexUrl {
        $baseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        # Explicit pin -- skip ALL GPU probing (headless / CI / cross-install).
        # UNSLOTH_TORCH_INDEX_URL wins (full URL, verbatim); _FAMILY is the leaf appended
        # to the mirror base. Matches install.sh / install_python_stack.py.
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
            return (Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL)
        }
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY)) {
            return "$baseUrl/$($env:UNSLOTH_TORCH_INDEX_FAMILY.Trim().Trim('/'))"
        }
        if (-not $NvidiaSmiExe) { return "$baseUrl/cpu" }
        try {
            $output = Invoke-NvidiaSmiBounded $NvidiaSmiExe
            # Newer NVIDIA drivers (e.g. 610.x on Windows) print
            # "CUDA UMD Version: X.Y" instead of the legacy "CUDA Version: X.Y".
            # Accept both spellings so we don't fall through to the cu126 default.
            if ($output -match 'CUDA(?: UMD)? Version:\s+(\d+)\.(\d+)') {
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

    # Strip userinfo AND query/fragment so an authenticated pin never leaks. Shared with
    # _strip_index_url_credentials (install.sh / py / setup.ps1).
    function Remove-IndexUrlCredentials {
        param([string]$Url)
        $sep = $Url.IndexOf('://')
        if ($sep -lt 0) { return $Url }
        $scheme = $Url.Substring(0, $sep)
        $rest = $Url.Substring($sep + 3)
        # Drop query / fragment (may hold auth tokens).
        $q = $rest.IndexOfAny([char[]]('?', '#'))
        if ($q -ge 0) { $rest = $rest.Substring(0, $q) }
        $slash = $rest.IndexOf('/')
        $authority = if ($slash -ge 0) { $rest.Substring(0, $slash) } else { $rest }
        $at = $authority.LastIndexOf('@')
        $host_ = if ($at -ge 0) { $authority.Substring($at + 1) } else { $authority }
        if ($slash -ge 0) { return "${scheme}://${host_}$($rest.Substring($slash))" }
        return "${scheme}://${host_}"
    }

    # ── Torch flavor helpers (to repair a stale CPU / wrong-CUDA wheel) ──
    # torch.__version__ -> flavor tag (cuXXX / rocm / cpu); untagged wheel = cpu,
    # matching setup.ps1's stale-venv parse.
    function ConvertTo-TorchFlavorTag {
        param([string]$TorchVersion)
        if (-not $TorchVersion) { return $null }
        if ($TorchVersion -match '\+(cu\d+)') { return $Matches[1] }
        if ($TorchVersion -match '\+rocm')    { return 'rocm' }
        if ($TorchVersion -match '\+cpu')     { return 'cpu' }
        return 'cpu'
    }

    # Expected tag from the index leaf: cuXXX / cpu / rocm ($ROCmIndexUrl or a
    # gfx* leaf -> rocm). $null on an unknown leaf (odd mirror) so repair no-ops.
    function Get-ExpectedTorchFlavorTag {
        param([string]$TorchIndexUrl, [string]$ROCmIndexUrl)
        if (-not [string]::IsNullOrWhiteSpace($ROCmIndexUrl)) { return 'rocm' }
        if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) { return $null }
        # Drop query/fragment first so .../cu128?token=x classifies as cu128 (else it reinstalls every run).
        $leaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
        if ($leaf -match '^cu\d+$') { return $leaf }
        if ($leaf -eq 'cpu')        { return 'cpu' }
        if ($leaf -match '^rocm')   { return 'rocm' }
        # gfx must be followed by a digit (an architecture leaf); gfx-private is custom.
        if ($leaf -match '^gfx[0-9]') { return 'rocm' }
        return $null
    }

    # Installed torch flavor tag in $PythonExe's venv, or $null if absent. Uses
    # ProcessStartInfo (not &) so stderr doesn't trip $ErrorActionPreference.
    function Get-InstalledTorchTag {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $PythonExe
            $psi.Arguments = '-c "import torch; print(torch.__version__)"'
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            # Drain BOTH streams async, then WaitForExit. A synchronous ReadToEnd()
            # before the wait would block forever if a wedged "import torch" never
            # closes stdout; leaving the redirected stderr undrained would deadlock a
            # child that floods it past the pipe buffer. Async reads let a noisy-but-
            # exiting probe finish, while a truly hung one still hits the 30s timeout
            # and is killed -- bounded either way.
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            $finished = $proc.WaitForExit(30000)
            if (-not $finished) { try { $proc.Kill() } catch {}; return $null }
            $torchVer = $outTask.GetAwaiter().GetResult().Trim()
            [void]$errTask.GetAwaiter().GetResult()
            if ($proc.ExitCode -ne 0 -or -not $torchVer) { return $null }
            return ConvertTo-TorchFlavorTag $torchVer
        } catch { return $null }
    }

    # An explicit pin is authoritative: the AMD ROCm reroute below must not rewrite it
    # (e.g. a deliberate cpu pin on an AMD host).
    $TorchIndexPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) -or `
                        (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY))
    $TorchIndexUrl = Get-TorchIndexUrl

    # ── GPU arch → newest compatible Windows ROCm wheel release ──
    # Wheels bundle their own ROCm runtime; the installed HIP SDK version does
    # not constrain which release to use.  Always picks the newest release that
    # supports the GPU architecture.
    # ── AMD Windows ROCm: arch-aware pip index (repo.amd.com) ──
    # Wheels bundle their own ROCm runtime and support all Python versions.
    # Override with UNSLOTH_ROCM_WINDOWS_MIRROR for air-gapped / mirror installs.
    $ROCmIndexUrl = $null
    $ROCmTorchFloor = $null
    $PinnedRocmVisionSpec = $null
    $PinnedRocmAudioSpec = $null
    if (-not $TorchIndexPinned -and ($HasROCm -or $ROCmGfxArch) -and $TorchIndexUrl -like "*/cpu" -and -not $SkipTorch) {
        $amdIndexBase = if ($env:UNSLOTH_ROCM_WINDOWS_MIRROR) { $env:UNSLOTH_ROCM_WINDOWS_MIRROR.TrimEnd('/') } else { "https://repo.amd.com/rocm/whl" }
        $archFamilyMap = @{
            "gfx1201" = "gfx120X-all"; "gfx1200" = "gfx120X-all"  # RDNA 4
            "gfx1151" = "gfx1151";     "gfx1150" = "gfx1150"       # RDNA 3.5 (Strix Halo/Point)
            "gfx1103" = "gfx110X-all"; "gfx1102" = "gfx110X-all"   # RDNA 3
            "gfx1101" = "gfx110X-all"; "gfx1100" = "gfx110X-all"
            "gfx90a"  = "gfx90a";      "gfx908"  = "gfx908"        # MI200/MI100
        }
        # gfx120X (RDNA 4) and gfx1151/gfx1150 (Strix) have a null-pointer bug in
        # torch._C._grouped_mm on torch <2.11.0 (rocm7.12 and rocm7.1 respectively).
        # TheRock issues #5284 and #3284. Force torch>=2.11.0 so pip never resolves
        # to the broken 2.10.0 wheels even though they exist on the AMD index.
        # The <2.12.0 ceiling matches the Linux install_python_stack.py constraint
        # for the same arches: AMD actively publishes new versions on their index,
        # so without a ceiling a future 2.12.0+rocmX.Y wheel would be pulled in
        # automatically before it has been validated on these architectures.
        # Bump the ceiling here (and in install_python_stack.py) when 2.12.x is
        # confirmed working on gfx120X / Strix.
        $torchFloorMap = @{
            "gfx1201" = "torch>=2.11.0,<2.12.0"; "gfx1200" = "torch>=2.11.0,<2.12.0"
            "gfx1151" = "torch>=2.11.0,<2.12.0"; "gfx1150" = "torch>=2.11.0,<2.12.0"
        }
        # Companion ranges track the torch ceiling so pip resolves a consistent
        # trio on AMD's per-arch index (each published independently). Mirrors
        # setup.ps1 / install_python_stack.py; bump all three together for 2.12.x.
        $torchvisionFloorMap = @{
            "gfx1201" = "torchvision>=0.26.0,<0.27.0"; "gfx1200" = "torchvision>=0.26.0,<0.27.0"
            "gfx1151" = "torchvision>=0.26.0,<0.27.0"; "gfx1150" = "torchvision>=0.26.0,<0.27.0"
        }
        $torchaudioFloorMap = @{
            "gfx1201" = "torchaudio>=2.11.0,<2.12.0"; "gfx1200" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1151" = "torchaudio>=2.11.0,<2.12.0"; "gfx1150" = "torchaudio>=2.11.0,<2.12.0"
        }
        $archFamily = if ($ROCmGfxArch -and $archFamilyMap.ContainsKey($ROCmGfxArch)) { $archFamilyMap[$ROCmGfxArch] } else { $null }
        if ($archFamily) {
            $ROCmIndexUrl = "$amdIndexBase/$archFamily/"
            $ROCmTorchFloor = if ($ROCmGfxArch -and $torchFloorMap.ContainsKey($ROCmGfxArch)) { $torchFloorMap[$ROCmGfxArch] } else { $null }
            $archLabel = if ($ROCmGfxArch) { $ROCmGfxArch } else { "AMD GPU" }
            substep "$archLabel -- AMD repo.amd.com index selected" "Cyan"
            if ($ROCmTorchFloor) {
                substep "  enforcing $ROCmTorchFloor (known _grouped_mm bug in older wheels)" "Cyan"
            }
        } elseif ($ROCmGfxArch) {
            substep "AMD GPU ($ROCmGfxArch) not in supported arch list -- falling back to CPU-only PyTorch" "Yellow"
        } else {
            substep "AMD GPU detected but arch unknown -- falling back to CPU-only PyTorch" "Yellow"
        }
    }

    # A gfx*/rocm pin skips the auto-reroute above, but the generic CPU/CUDA install below
    # would use torch>=2.4,<2.11 and pull a known-bad wheel on the gfx115x/gfx120x/rocm>=7.2
    # indexes (the _grouped_mm bug). Route a pinned ROCm index through the ROCm path.
    if ($TorchIndexPinned -and -not $ROCmIndexUrl -and -not $SkipTorch) {
        $_pinLeaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLower()
        $_pinRocm211 = $false
        # Anchor ($) so a suffixed custom leaf (rocm7.2-private) falls through to verbatim.
        if ($_pinLeaf -match '^rocm(\d+)\.(\d+)$') {
            # Only KNOWN-2.11 rocm (rocm7.2) gets the floor. Matches Test-RocmKnown211Version.
            $_pinRocm211 = ([int]$Matches[1] -eq 7 -and [int]$Matches[2] -eq 2)
        }
        # Only the 2.11-allowlist gfx arches need the floor; others publish <2.11 and stay bare.
        $_pinGfx211 = @('gfx120x-all', 'gfx1151', 'gfx1150') -contains $_pinLeaf
        if ($_pinGfx211 -or $_pinRocm211) {
            $ROCmIndexUrl = $TorchIndexUrl
            $ROCmTorchFloor = "torch>=2.11.0,<2.12.0"
            $PinnedRocmVisionSpec = "torchvision>=0.26.0,<0.27.0"
            $PinnedRocmAudioSpec = "torchaudio>=2.11.0,<2.12.0"
            substep "pinned ROCm index ($_pinLeaf) -- enforcing $ROCmTorchFloor" "Cyan"
        } elseif ($_pinLeaf -match '^gfx[0-9]' -or $_pinLeaf -match '^rocm[0-9]+(\.[0-9]+)?$') {
            # Other gfx / older rocm (<=7.1) ship torch <2.11; route via the ROCm path with
            # bare specs. Only EXACT rocm<digits>/gfx* are families; a suffixed leaf is verbatim.
            $ROCmIndexUrl = $TorchIndexUrl
        }
    }

    if ($ROCmIndexUrl) {
        $TorchIndexFamily = "rocm"
    } else {
        $TorchIndexFamily = Get-TauriTorchIndexFamily $TorchIndexUrl
    }
    $GpuBranch = Get-TauriGpuBranch $TorchIndexFamily
    Write-TauriDiag -GpuBranch $GpuBranch -TorchIndexFamily $TorchIndexFamily -PythonVersionForDiag $DetectedPython.Version

    # ── Print CPU-only hint when no GPU detected ──
    if (-not $SkipTorch -and -not $ROCmIndexUrl -and $TorchIndexUrl -like "*/cpu") {
        Write-Host ""
        if ($ROCmGfxArch) {
            # Only an unmapped arch reaches here (a mapped one set $ROCmIndexUrl
            # above). No ROCm torch wheels for this arch (e.g. RDNA2 gfx103X) -> CPU.
            substep "Installing CPU PyTorch -- no ROCm PyTorch wheels are available for $ROCmGfxArch." "Yellow"
            substep "PyTorch (training and Transformers inference) runs on CPU on this GPU." "Yellow"
        } else {
            if ($HipSdkInstalled -and -not $HasROCm) {
                substep "Installing CPU-only PyTorch (HIP SDK found but GPU not ROCm-accessible)." "Yellow"
            } elseif ($ROCmGpuLabel) {
                substep "Installing CPU-only PyTorch (AMD GPU arch unknown -- install the HIP SDK" "Yellow"
                substep "or set UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm)." "Yellow"
            } else {
                substep "No NVIDIA GPU detected." "Yellow"
            }
            substep "Installing CPU-only PyTorch. If you only need GGUF chat/inference," "Yellow"
            substep "re-run with --no-torch for a faster, lighter install:" "Yellow"
            substep ".\install.ps1 --no-torch" "Yellow"
        }
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
        $installed = Get-ChildItem -LiteralPath $VenvDir -Recurse -Filter "no-torch-runtime.txt" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*studio*backend*requirements*no-torch-runtime.txt" } |
            Select-Object -ExpandProperty FullName -First 1
        return $installed
    }

    if ($_Migrated) {
        # Migrated env: force-reinstall unsloth+unsloth-zoo for a clean state, preserving
        # existing torch/CUDA unless the flavor repair below re-lands it.
        Write-TauriLog "STEP" "Installing unsloth"
        substep "upgrading unsloth in migrated environment..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated no-torch)" { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.7.3" "unsloth-zoo>=2026.7.3" }
            if ($baseInstallExit -eq 0) {
                # Resolve pydantic WITH deps so pip pins pydantic-core
                # to the matching version (no-torch-runtime.txt below
                # is --no-deps). All transitive deps are torch-free.
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { uv pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } else {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated)" { uv pip install --python $VenvPython --reinstall-package unsloth --reinstall-package unsloth-zoo "unsloth>=2026.7.3" "unsloth-zoo>=2026.7.3" }
        }
        if ($baseInstallExit -ne 0) {
            Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
        }
        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        }
    } elseif ($TorchIndexUrl -or $ROCmIndexUrl) {
        if ($SkipTorch) {
            substep "skipping PyTorch (--no-torch flag set)." "Yellow"
        } elseif ($ROCmIndexUrl) {
            Write-TauriLog "STEP" "Installing PyTorch (AMD ROCm Windows)"
            substep "installing PyTorch from $(Remove-IndexUrlCredentials $ROCmIndexUrl)..."
            $torchSpec = if ($ROCmTorchFloor) { $ROCmTorchFloor } else { "torch" }
            # Pin the companions to match $torchSpec; bare names can resolve an
            # ABI-incompatible torchvision/torchaudio on AMD's per-arch index.
            $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
            $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
            $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (AMD ROCm)" { uv pip install --python $VenvPython --force-reinstall --default-index $ROCmIndexUrl $torchSpec $visionSpec $audioSpec }
            if ($torchInstallExit -ne 0) {
                # Transient AMD-index failure: fall back to a CPU base (Unsloth setup retries
                # ROCm). Use an explicit CPU index -- for a pinned ROCm index $TorchIndexUrl IS
                # the ROCm mirror, so reusing it would just retry it.
                $CpuFallbackIndexUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { "$($env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/'))/cpu" } else { "https://download.pytorch.org/whl/cpu" }
                substep "ROCm PyTorch install failed (exit $torchInstallExit); using a CPU base, Unsloth setup retries ROCm." "Yellow"
                # --force-reinstall: a failed ROCm install can leave an unpinned ROCm
                # torch (e.g. 2.10.0+rocm on gfx110X/gfx90a) that still satisfies the CPU
                # torch>= range, so without it uv would keep the ROCm build and only swap
                # the companions -- a mismatched venv the flavor-repair block won't fix.
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (CPU fallback)" { uv pip install --python $VenvPython --force-reinstall "torch>=2.4,<2.11.0" "torchvision>=0.19,<0.26.0" "torchaudio>=2.4,<2.11.0" --default-index $CpuFallbackIndexUrl }
                if ($torchInstallExit -ne 0) {
                    Write-Host "[ERROR] Failed to install PyTorch (ROCm and CPU base both failed, exit code $torchInstallExit)" -ForegroundColor Red
                    return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
                }
                # CPU base is in; drop the ROCm expectation so the flavor-repair
                # block below won't retry the just-failed index and abort. setup.ps1
                # reinstalls ROCm afterwards (recomputes its own index URL).
                $ROCmIndexUrl = $null
                $ROCmTorchFloor = $null
            }
        } else {
            Write-TauriLog "STEP" "Installing PyTorch"
            substep "installing PyTorch ($(Remove-IndexUrlCredentials $TorchIndexUrl))..."
            # Bounded trio on every index (torchaudio 2.11 dropped its exact torch
            # pin, so a bare companion can drift from a capped torch). cu<digits>
            # families ship torch 2.11.x with paired triton-windows 3.6, so the
            # ceiling widens to <2.12 there; other leaves keep the 2.10 line.
            # Mirrors install.sh and _CUDA_TORCH_PKG_SPEC.
            $_idxLeaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
            if ($_idxLeaf -match '^cu[0-9]+$') {
                $_pinTorchSpec = "torch>=2.4,<2.12.0"
                $_pinVisionSpec = "torchvision>=0.19,<0.27.0"
                $_pinAudioSpec = "torchaudio>=2.4,<2.12.0"
            } else {
                $_pinTorchSpec = "torch>=2.4,<2.11.0"
                $_pinVisionSpec = "torchvision>=0.19,<0.26.0"
                $_pinAudioSpec = "torchaudio>=2.4,<2.11.0"
            }
            $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch" { uv pip install --python $VenvPython $_pinTorchSpec $_pinVisionSpec $_pinAudioSpec --default-index $TorchIndexUrl }
            if ($torchInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
            }
        }

        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (no-torch)" { uv pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "unsloth>=2026.7.3" "unsloth-zoo>=2026.7.3" }
            if ($baseInstallExit -eq 0) {
                # Same pydantic-with-deps trick as the migrated branch.
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { uv pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { uv pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } elseif ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (local)" { uv pip install --python $VenvPython --upgrade-package unsloth "unsloth>=2026.7.3" "unsloth-zoo>=2026.7.3" }
        } else {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth" { uv pip install --python $VenvPython --upgrade-package unsloth -- "$PackageName" }
        }
        if ($baseInstallExit -ne 0) {
            Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
        }

        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        }
    } else {
        # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (auto torch backend)" { uv pip install --python $VenvPython "unsloth-zoo>=2026.7.3" "unsloth>=2026.7.3" --torch-backend=auto }
            if ($baseInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
            }
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand { uv pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { uv pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-Host "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        } else {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (auto torch backend)" { uv pip install --python $VenvPython --torch-backend=auto -- "$PackageName" }
            if ($baseInstallExit -ne 0) {
                Write-Host "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
            }
        }
    }

    # ── Enforce the installed torch flavor matches the detected GPU build ──
    # PEP 440 ignores the +cpu/+cuXXX/+rocm local label in a version range, so uv
    # keeps a stale torch==X+cpu against a CUDA index and setup.ps1 then loops on
    # "torch cpu != required cuXXX". Reinstall the right triplet when a GPU build is
    # expected: CUDA from $TorchIndexUrl, ROCm from $ROCmIndexUrl (repo.amd.com gfx*
    # is a PEP 503 index uv resolves via --default-index, same URL the fresh ROCm install
    # above uses). --no-torch / CPU-only hosts (expected cpu) are no-ops.
    if (-not $SkipTorch) {
        $expectedTorchTag = Get-ExpectedTorchFlavorTag -TorchIndexUrl $TorchIndexUrl -ROCmIndexUrl $ROCmIndexUrl
        if ($expectedTorchTag -and $expectedTorchTag -ne 'cpu') {
            $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
            if ($installedTorchTag -and $installedTorchTag -ne $expectedTorchTag) {
                if ($expectedTorchTag -eq 'rocm' -and $ROCmIndexUrl) {
                    # AMD: a migrated venv can keep a stale CPU torch the fresh ROCm path
                    # would have force-reinstalled. Repair from the same repo.amd.com index.
                    $rocmSpec = if ($ROCmTorchFloor) { $ROCmTorchFloor } else { "torch" }
                    # Pin companions like the fresh ROCm path (bare names can pull an
                    # ABI-incompatible torchvision/torchaudio from the per-arch index).
                    $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
                    $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
                    substep "PyTorch flavor mismatch (installed $installedTorchTag, need ROCm) -- reinstalling correct build..." "Yellow"
                    $torchFixExit = Invoke-InstallCommand { uv pip install --python $VenvPython --force-reinstall --default-index $ROCmIndexUrl $rocmSpec $visionSpec $audioSpec }
                    if ($torchFixExit -ne 0) {
                        Write-Host "[ERROR] Failed to reinstall PyTorch with the correct ROCm build (exit code $torchFixExit)" -ForegroundColor Red
                        return (Exit-InstallFailure "Failed to reinstall PyTorch (ROCm) (exit code $torchFixExit)" $torchFixExit)
                    }
                    $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
                } elseif ($expectedTorchTag -ne 'rocm') {
                    # CUDA: stale +cpu (or wrong cuXXX) against a CUDA index -> reinstall triplet.
                    # Same leaf-gated ceiling as the install above: cu* serves 2.11.
                    $_fixLeaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
                    if ($_fixLeaf -match '^cu[0-9]+$') {
                        $_fixTorchSpec = "torch>=2.4,<2.12.0"; $_fixVisionSpec = "torchvision>=0.19,<0.27.0"; $_fixAudioSpec = "torchaudio>=2.4,<2.12.0"
                    } else {
                        $_fixTorchSpec = "torch>=2.4,<2.11.0"; $_fixVisionSpec = "torchvision>=0.19,<0.26.0"; $_fixAudioSpec = "torchaudio>=2.4,<2.11.0"
                    }
                    substep "PyTorch flavor mismatch (installed $installedTorchTag, need $expectedTorchTag) -- reinstalling correct build..." "Yellow"
                    $torchFixExit = Invoke-InstallCommand { uv pip install --python $VenvPython $_fixTorchSpec $_fixVisionSpec $_fixAudioSpec --default-index $TorchIndexUrl --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio }
                    if ($torchFixExit -ne 0) {
                        Write-Host "[ERROR] Failed to reinstall PyTorch with the correct CUDA build (exit code $torchFixExit)" -ForegroundColor Red
                        return (Exit-InstallFailure "Failed to reinstall PyTorch ($expectedTorchTag) (exit code $torchFixExit)" $torchFixExit)
                    }
                    $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
                }
            }
            # Safety net (incl. AMD): GPU build expected but still CPU -> warn loudly.
            if ($installedTorchTag -eq 'cpu') {
                Write-Host ""
                Write-Host "  [WARN] PyTorch is CPU-only but a $expectedTorchTag GPU build was expected for this machine." -ForegroundColor Yellow
                Write-Host "  [WARN] Training and GPU inference will run on CPU until this is fixed." -ForegroundColor Yellow
                Write-Host "  [WARN] Re-run this installer, or reinstall the GPU build manually for your GPU." -ForegroundColor Yellow
            }
        }
    }

    # Overlay Tauri-bundled studio fixes that may be ahead of PyPI. Skipped
    # for --local: the editable install above already makes _PACKAGE_ROOT in
    # unsloth_cli/commands/studio.py resolve to the repo (PEP 660 __file__).
    # Source paths match the Tauri bundle layout in studio/src-tauri/tauri.conf.json,
    # which bundles install_python_stack.py at the bundle root next to install.ps1.
    if ($TauriMode) {
        $rawPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.ScriptName }
        if ($rawPath) {
            # Strip leading \\?\ extended-length prefix if the launcher passed one.
            $scriptDir = Split-Path -Parent ($rawPath -replace '^\\\\\?\\', '')
            $overlayMap = [ordered]@{
                "install_python_stack.py" = "Lib\site-packages\studio\install_python_stack.py"
            }
            foreach ($rel in $overlayMap.Keys) {
                $src = Join-Path $scriptDir $rel
                $dst = Join-Path $VenvDir $overlayMap[$rel]
                # -LiteralPath: $VenvDir derives from $StudioHome which may
                # contain [ ] * ? when the user overrode UNSLOTH_STUDIO_HOME.
                if (-not (Test-Path -LiteralPath $src)) { continue }
                $dstParent = Split-Path -Parent $dst
                if (-not (Test-Path -LiteralPath $dstParent)) {
                    Write-Host "[WARN] Overlay target dir missing: $dstParent; studio setup may use stale bundled file" -ForegroundColor Yellow
                    continue
                }
                try {
                    if (-not (Test-Path -LiteralPath $dst)) {
                        # Backfill: target file missing but parent dir exists.
                        Copy-Item -LiteralPath $src -Destination $dst -Force
                        substep ("backfilled bundled " + (Split-Path -Leaf $rel))
                    } else {
                        # Hash-compare so re-runs are no-ops when files already match.
                        $srcHash = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
                        $dstHash = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash
                        if ($srcHash -ne $dstHash) {
                            Copy-Item -LiteralPath $src -Destination $dst -Force
                            substep ("applied bundled " + (Split-Path -Leaf $rel))
                        }
                    }
                } catch {
                    Write-Host "[WARN] Could not overlay $($rel): $($_.Exception.Message); studio setup may use stale bundled file" -ForegroundColor Yellow
                }
            }
        }
    }

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, and other dependencies automatically via winget. Node.js is
    # NOT installed via winget -- setup.ps1 uses an isolated Node it manages and
    # never touches the system Node/npm.
    Write-TauriLog "STEP" "Running studio setup"
    step "setup" "running unsloth studio setup..."
    $UnslothExe = Join-Path $VenvDir "Scripts\unsloth.exe"
    if (-not (Test-Path -LiteralPath $UnslothExe)) {
        Write-TauriLog "ERROR" "unsloth CLI was not installed correctly"
        Write-Host "[ERROR] unsloth CLI was not installed correctly." -ForegroundColor Red
        Write-Host "        Expected: $UnslothExe" -ForegroundColor Yellow
        Write-Host "        This usually means an older unsloth version was installed that does not include the Unsloth CLI." -ForegroundColor Yellow
        Write-Host "        Try re-running the installer or see: https://github.com/unslothai/unsloth?tab=readme-ov-file#-quickstart" -ForegroundColor Yellow
        return (Exit-InstallFailure "unsloth CLI was not installed correctly")
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
    # Propagate UNSLOTH_STUDIO_HOME only for env-override installs; otherwise
    # an inherited value would put llama.cpp in the wrong place.
    $previousUnslothStudioHome = $env:UNSLOTH_STUDIO_HOME
    $hadPreviousUnslothStudioHome = ($null -ne $previousUnslothStudioHome)
    if ($StudioRedirectMode -eq 'env') {
        $env:UNSLOTH_STUDIO_HOME = $StudioHome
    } else {
        Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
    }
    $studioArgs = @('studio', 'setup')
    if ($script:UnslothVerbose) { $studioArgs += '--verbose' }
    if ($WithLlamaCppDir) {
        if (-not (Test-Path -LiteralPath $WithLlamaCppDir -PathType Container)) {
            Write-Host "[ERROR] --with-llama-cpp-dir path does not exist: $WithLlamaCppDir" -ForegroundColor Red
            return (Exit-InstallFailure "--with-llama-cpp-dir path does not exist.")
        }
        $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR = (Resolve-Path -LiteralPath $WithLlamaCppDir).Path
    }
    $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED = "1"
    # Hand the venv interpreter to setup.ps1 so it reuses the Python we already
    # resolved and built the venv with, instead of re-probing the system (which
    # can trip over an unsupported `python` 3.14 or a Store stub on PATH even
    # though the venv is fine). setup.ps1 Test-Path-guards this before use.
    $env:UNSLOTH_SETUP_PYTHON = Join-Path $VenvDir "Scripts\python.exe"
    try {
        & $UnslothExe @studioArgs
        $setupExit = $LASTEXITCODE
    } finally {
        if ($hadPreviousUnslothStudioHome) {
            $env:UNSLOTH_STUDIO_HOME = $previousUnslothStudioHome
        } else {
            Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
        }
        Remove-Item Env:UNSLOTH_LOCAL_LLAMA_CPP_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_SETUP_PYTHON -ErrorAction SilentlyContinue
    }
    if ($setupExit -ne 0) {
        Write-Host "[ERROR] unsloth studio setup failed (exit code $setupExit)" -ForegroundColor Red
        return (Exit-InstallFailure "unsloth studio setup failed (exit code $setupExit)" $setupExit)
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
    [System.IO.Directory]::CreateDirectory($ShimDir) | Out-Null
    $ShimExe = Join-Path $ShimDir "unsloth.exe"
    # Fatal preflight outside the lock-handling try/catch -- a directory at
    # the shim path must not be downgraded to "Continuing with the existing
    # launcher", or the install finishes with no usable shim.
    if (Test-Path -LiteralPath $ShimExe -PathType Container) {
        Write-Host "[ERROR] Cannot create unsloth launcher: $ShimExe is a directory." -ForegroundColor Red
        Write-Host "        Move or remove it manually, then re-run the installer." -ForegroundColor Yellow
        throw "Cannot create unsloth launcher: $ShimExe is a directory."
    }
    # try/catch: if unsloth.exe is locked (Unsloth running), keep the old shim.
    $shimUpdated = $false
    try {
        if (Test-Path -LiteralPath $ShimExe) { Remove-Item -LiteralPath $ShimExe -Force -ErrorAction Stop }
        try {
            # New-Item -ItemType HardLink does NOT accept -LiteralPath in any
            # PowerShell version, so use -Path. Wildcards in $ShimExe (e.g.
            # brackets in custom roots) glob-expand here and fall through to
            # the Copy-Item -LiteralPath fallback below.
            New-Item -ItemType HardLink -Path $ShimExe -Target $UnslothExe -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -LiteralPath $UnslothExe -Destination $ShimExe -Force -ErrorAction Stop # fallback: copy
        }
        $shimUpdated = $true
    } catch {
        if (Test-Path -LiteralPath $ShimExe) {
            Write-Host "[WARN] Could not refresh unsloth launcher at $ShimExe." -ForegroundColor Yellow
            Write-Host "       This usually means a running 'unsloth studio' process still holds the file open." -ForegroundColor Yellow
            Write-Host "       Close Unsloth and re-run the installer to pick up the latest launcher." -ForegroundColor Yellow
            Write-Host "       Continuing with the existing launcher." -ForegroundColor Yellow
        } else {
            Write-Host "[WARN] Could not create unsloth launcher at $ShimExe" -ForegroundColor Yellow
            Write-Host "       $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "       Launch unsloth studio directly via '$UnslothExe' until the next successful install." -ForegroundColor Yellow
        }
    }
    # Add to PATH only when launcher exists. Env-mode: session-only export,
    # no registry change (workspace path may be deleted later).
    $pathAdded = $false
    if (Test-Path -LiteralPath $ShimExe) {
        if ($StudioRedirectMode -ne 'env') {
            $pathAdded = Add-ToUserPath -Directory $ShimDir -Position 'Prepend'
        }
    }
    if ($shimUpdated -and $pathAdded) {
        step "path" "added unsloth launcher to PATH"
    }
    Refresh-SessionPath  # sync current session with registry
    Complete-StudioVenvRollback

    # Env-mode session export AFTER Refresh-SessionPath; otherwise a legacy
    # User PATH entry (Machine > User > current $env:Path) would win.
    if ($StudioRedirectMode -eq 'env' -and (Test-Path -LiteralPath $ShimExe)) {
        $env:Path = "$ShimDir;$env:Path"
        step "path" "exported $ShimDir for this session (no registry PATH change in env-override mode)"
    }

    # ── Tauri mode: done, skip shortcuts and auto-launch ──
    if ($TauriMode) {
        Write-TauriLog "DONE" ""
        return
    }

    # New-StudioShortcuts gates the .lnk shortcuts on env-mode internally.
    New-StudioShortcuts -UnslothExePath $UnslothExe

    # Warn if another 'unsloth' wins on PATH (different venv, system pip).
    # Mirrors install.sh; absolute path is still the most reliable launch.
    # Uses content-hash equality (Get-FileHash) so hardlinks, symlinks, and
    # identical copies of the installer's shim don't false-trigger. CommandType
    # Application restricts the probe to real executables (skips aliases,
    # functions, scripts).
    try {
        $_pathCmd = Get-Command unsloth -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($_pathCmd) {
            $_pathExe = $_pathCmd.Source
            $_installedHash = (Get-FileHash -LiteralPath $UnslothExe -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
            $_pathHash      = (Get-FileHash -LiteralPath $_pathExe   -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
            if ($_installedHash -and $_pathHash -and ($_installedHash -ne $_pathHash)) {
                Write-Host ""
                step "warning" "another 'unsloth' wins on PATH:" "Yellow"
                substep $_pathExe
                substep "this installer's binary is at:"
                substep $UnslothExe
                substep "to use this install, call the absolute path above,"
                substep "or put its dir earlier on PATH."
                Write-Host ""
            }
        }
    } catch {
        # Diagnostic only; never block install on a probe failure.
    }

    # In interactive terminals, ask the user before starting Unsloth unless the
    # caller explicitly disabled the post-install prompt.
    # In non-interactive environments (CI, Docker) just print instructions.
    $IsInteractive = (-not $SkipAutostart) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        Write-Host ""
        $reply = Read-Host "  Start Unsloth Studio now? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($reply) -or $reply -match '^[Yy]') {
            & $UnslothExe studio -p 8888
        } else {
            step "launch" "to start later, run:"
            substep "unsloth studio -p 8888"
            substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
            substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
            Write-Host ""
        }
    } else {
        step "launch" "manual commands:"
        # Single-quote the printed paths so $-vars / backticks in custom roots
        # do not reparse when the user pastes the command.
        $_actLiteral = "'" + ((Join-Path $VenvDir "Scripts\Activate.ps1") -replace "'", "''") + "'"
        if ($StudioRedirectMode -eq 'env') {
            # Env-mode skips registry PATH; print the absolute shim path.
            $_shim = Join-Path $StudioHome "bin\unsloth.exe"
            $_shimLiteral = "'" + ($_shim -replace "'", "''") + "'"
            substep "& $_shimLiteral studio -p 8888"
            substep "or activate env first:"
            substep "& $_actLiteral"
            substep "unsloth studio -p 8888"
        } else {
            substep "& $_actLiteral"
            substep "unsloth studio -p 8888"
        }
        substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
        substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
        Write-Host ""
    }
}

Install-UnslothStudio @args
