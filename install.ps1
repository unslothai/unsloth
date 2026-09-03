# Unsloth Studio Installer for Windows PowerShell
#
# Usage, options and the web one-liner: see "Unsloth Studio (web UI)" in the README
# (https://github.com/unslothai/unsloth#unsloth-studio-web-ui). Not repeated here, because
# AMSI scans this file in full before a line of it runs and nothing reads the header from inside.
#
# The web entry point cannot forward arguments, so it takes options as environment variables set
# beforehand (UNSLOTH_NO_TORCH, UNSLOTH_SKIP_AUTOSTART, UNSLOTH_PYTHON, UNSLOTH_STUDIO_HOME); a
# local run takes the equivalent flags (--no-torch, --skip-autostart, --python, --local).
#
# Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME (alias) > $USERPROFILE\.unsloth\studio
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
function Install-UnslothStudio {
    $ErrorActionPreference = "Stop"

    # The user's PowerShell profile has already run by the time this does, and the documented
    # piped web entry point documented in the README has no script file to re-launch
    # with -NoProfile, so each way a profile can reach in here is cut individually below.
    #
    # Off, not Latest: this script predates strict mode, testing environment variables that are
    # legitimately unset and reading $script: state only some branches assign. Scoped to here
    # and below, so the caller keeps its own.
    Set-StrictMode -Off

    # A profile that sets 'None' -- the startup-time tweak people copy -- is fatal on PowerShell
    # 7, which loads NO modules at startup: Test-Path, Write-Host, Select-Object,
    # ConvertFrom-Json, Get-FileHash, Invoke-WebRequest, Expand-Archive, Start-Process and
    # Get-Content stop resolving, while 5.1 preloads Utility and Management and survives. First
    # of the four, because the proxy handoff below calls ConvertTo-Json from Utility too.
    $PSModuleAutoLoadingPreference = 'All'

    # Proxy keys are kept rather than dropped: on a locked-down corporate host such an entry may
    # be the sole route to python.org and the uv release. IsMatch, not -match, so the filter
    # leaves no $Matches behind for the rest of the install.
    $_UnslothKeptDefaults = @{}
    foreach ($_UnslothDefaultKey in @($PSDefaultParameterValues.Keys)) {
        # IgnoreCase: 'invoke-webrequest:proxy' is valid PowerShell and binds the same
        # parameter, so a case-sensitive filter drops a working proxy on a technicality.
        if ($_UnslothDefaultKey -is [string] -and
            [regex]::IsMatch(
                $_UnslothDefaultKey,
                ':Proxy(Credential|UseDefaultCredentials)?$',
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            $_UnslothKeptDefaults[$_UnslothDefaultKey] = $PSDefaultParameterValues[$_UnslothDefaultKey]
        }
    }
    # One profile entry such as 'Start-Process:WindowStyle' or 'Invoke-WebRequest:TimeoutSec'
    # silently rebinds every process launch and download here. Assigning with no scope qualifier
    # shadows the caller's table for this scope and below only.
    $PSDefaultParameterValues = $_UnslothKeptDefaults

    # Windows PowerShell 5.1 redraws the Invoke-WebRequest progress bar on every read, and the
    # redraw, not the link, sets the rate: on a windows-latest runner the python.org installer
    # (27.8 MB) took 41.34s with the bar on against 0.08s with it off, and the uv archive the
    # same. That is the multi-minute "slow download" users report. -UseBasicParsing does NOT
    # avoid it and PowerShell 7 never had the cost; only this preference does. Same scoping rule
    # as the table above: no qualifier, so the caller's own preference survives a piped web run.
    $ProgressPreference = 'SilentlyContinue'

    # The kept proxies travel to studio/setup.ps1 (launched -NoProfile by unsloth_cli, and it
    # downloads the VC++ runtime and the uv installer) as JSON in _UNSLOTH_PS_PROXY_DEFAULTS,
    # since a PowerShell variable does not cross a process boundary. Credentials do not travel:
    # a PSCredential does not serialize, and an environment variable is the wrong place for one.
    $_UnslothProxyHandoff = @{}
    foreach ($_UnslothDefaultKey in @($_UnslothKeptDefaults.Keys)) {
        $_UnslothDefaultValue = $_UnslothKeptDefaults[$_UnslothDefaultKey]
        # [uri] is the form the parameter actually takes and serializes to its own string.
        if ($_UnslothDefaultValue -is [uri]) {
            $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultValue.AbsoluteUri
        } elseif ($_UnslothDefaultValue -is [string] -or $_UnslothDefaultValue -is [bool]) {
            $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultValue
        } elseif ($_UnslothDefaultValue -is [scriptblock]) {
            # A script block is the supported form for a DYNAMIC default, e.g.
            # { [uri]$env:CORP_PROXY }, evaluated per call by Invoke-WebRequest. Evaluate here
            # and hand over the RESULT: executable code must not cross into the child.
            try {
                $_UnslothDefaultResolved = & $_UnslothDefaultValue
                if ($_UnslothDefaultResolved -is [uri]) {
                    $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultResolved.AbsoluteUri
                } elseif ($_UnslothDefaultResolved -is [string] -or
                          $_UnslothDefaultResolved -is [bool]) {
                    $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultResolved
                }
            } catch { }
        }
    }
    # A FUNCTION-local, not $script: or an environment variable: under a piped web run this runs
    # in the caller's own session, and the value can carry credentials (http://user:secret@proxy
    # is the ordinary corporate form) that must not outlive the install on any of the dozens of
    # return paths. Module-qualified serializer, as in the probe: a profile alias or function
    # named ConvertTo-Json would otherwise reshape this record or throw out of the prologue.
    $UnslothProxyHandoffJson =
        if ($_UnslothProxyHandoff.Count -gt 0) {
            $_UnslothProxyHandoff | Microsoft.PowerShell.Utility\ConvertTo-Json -Compress
        }
        else { $null }

    # PowerShell 7 only, and $false is its default: a profile that flips it on turns every
    # non-zero native exit into a terminating error, which with "Stop" above would throw out of
    # the setup handoff instead of reaching Exit-InstallFailure. Harmless on 5.1, where the
    # variable does not exist.
    $PSNativeCommandUseErrorActionPreference = $false

    # Reset per invocation, for the reason at $script:IsIntelXpu further down: under
    # a piped web run, $script: is the caller's session scope, so a second run in the same
    # console would start on the first run's state. These two are the only ones no later
    # statement re-assigns unconditionally.
    $script:UvExe = 'uv'
    $script:UvInstallDestDir = $null
    # Same reason, and this one caches a decision about the machine: a policy added or
    # removed between two runs in one console, or a replaced launcher under a hash rule,
    # would otherwise be answered from the first run's probe.
    $script:ShimLaunchBlockedCache = $null
    $script:ShimLaunchBlockedPath = $null

    $script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq "1")

    # Same fix as studio/setup.ps1, for the same reason. This script also calls
    # Expand-Archive and Get-ExecutionPolicy, which resolve via PSModulePath.
    # The desktop app reaches here as Tauri -> Rust -> powershell.exe
    # (studio/src-tauri/src/install.rs), and PowerShell only rewrites
    # PSModulePath for a direct pwsh -> powershell.exe hop, so the Rust process
    # in between leaves Windows PowerShell 5.1 leading with PowerShell 7's
    # module directories and unable to load its own copy of that module.
    #
    # Not restored afterwards, deliberately. $env: is the process environment,
    # so running this script in an interactive console leaves the reordering in
    # place for that session. A try/finally would not change that for the case
    # it is raised about: the interactive path ends by running Unsloth in the
    # foreground, so the finally would not fire until the user stops the server.
    # Narrowing the trigger instead would risk skipping the fix on some chain
    # this list does not anticipate, and the cost of that is the install failing
    # outright, against a session-lived module precedence change here.
    if ($PSVersionTable.PSEdition -ne 'Core' -and $env:SystemRoot) {
        $_UnslothSystemModules = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules'
        if (Test-Path $_UnslothSystemModules) {
            # Prepended: the problem is precedence, not absence.
            $_UnslothKept = @(
                $env:PSModulePath -split ';' |
                    Where-Object { $_ -and ($_ -ne $_UnslothSystemModules) }
            )
            $env:PSModulePath = (@($_UnslothSystemModules) + $_UnslothKept) -join ';'
        }
    }

    # Same UTF-8 invariant as studio/setup.ps1, same ordering constraint: this
    # rebuilds [Console]::Out, so it precedes the first write.
    $_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false
    try {
        [Console]::OutputEncoding = $_UnslothUtf8NoBom
    } catch {
        # No console: the setter drops the cached writer before throwing, so
        # bind UTF-8 ones explicitly. Same fallback as studio/setup.ps1.
        try {
            $_UnslothStdout = New-Object System.IO.StreamWriter -ArgumentList ([Console]::OpenStandardOutput()), $_UnslothUtf8NoBom
            $_UnslothStdout.AutoFlush = $true
            [Console]::SetOut($_UnslothStdout)
            $_UnslothStderr = New-Object System.IO.StreamWriter -ArgumentList ([Console]::OpenStandardError()), $_UnslothUtf8NoBom
            $_UnslothStderr.AutoFlush = $true
            [Console]::SetError($_UnslothStderr)
        } catch { }
    }
    $OutputEncoding = $_UnslothUtf8NoBom
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'

    # Resolved once: it picks the output sink in Write-StudioLine and must not
    # change mid-run. Same probe as studio/setup.ps1.
    $script:StudioStdoutRedirected = $false
    try { $script:StudioStdoutRedirected = [Console]::IsOutputRedirected } catch { }

    # Write-Host is written by 5.1's console host with its own writer on the OEM
    # code page, not the UTF-8 [Console]::Out rebound above. The desktop app
    # spawns this script with CREATE_NO_WINDOW and decodes the pipe as UTF-8
    # (from_utf8_lossy, studio/src-tauri/src/install.rs), so the banner emoji,
    # the U+2500 rule and every warning arrived as U+FFFD. One sink instead: the
    # console handle when redirected, Write-Host when interactive, since it is
    # the only one that colorizes. Defined above the first write, for the same
    # ordering reason as the encoding block.
    function Write-StudioLine {
        param([string]$Message = "", [string]$ForegroundColor)
        if ($script:StudioStdoutRedirected) {
            try { [Console]::Out.WriteLine($Message); [Console]::Out.Flush() } catch {}
            return
        }
        if ($PSBoundParameters.ContainsKey('ForegroundColor')) {
            Write-Host $Message -ForegroundColor $ForegroundColor
        } else {
            Write-Host $Message
        }
    }

    # ── Tauri structured output ──
    function Write-TauriLog {
        param([string]$Tag, [string]$Message)
        if ($TauriMode) {
            Write-StudioLine "[TAURI:$Tag] $Message"
        }
    }

    # Mirrors _uv_download_markers in install.sh; only what was opened is closed.
    $script:UvDownloadMarkerMinBytes = 52428800
    if ($env:UNSLOTH_DL_MARKER_MIN_BYTES -match '^\d+$') {
        $script:UvDownloadMarkerMinBytes = [long]$env:UNSLOTH_DL_MARKER_MIN_BYTES
    }
    $script:UvAnnouncedDownloads = @{}
    function Write-UvDownloadMarker {
        param([string]$Line)
        if (-not $TauriMode) { return }
        if ($Line -match '(?:^|\s)Downloading (\S+) \(([0-9.]+)(KiB|MiB|GiB)\)\s*$') {
            $unit = @{ KiB = 1024L; MiB = 1048576L; GiB = 1073741824L }[$Matches[3]]
            if ([double]$Matches[2] * $unit -ge $script:UvDownloadMarkerMinBytes) {
                $script:UvAnnouncedDownloads[$Matches[1]] = $true
                Write-TauriLog "DL" "$($Matches[1]) $($Matches[2])$($Matches[3])"
            }
        } elseif ($Line -match '(?:^|\s)Downloaded (\S+)\s*$') {
            # ContainsKey, not Remove's return: Hashtable.Remove is void.
            if ($script:UvAnnouncedDownloads.ContainsKey($Matches[1])) {
                $script:UvAnnouncedDownloads.Remove($Matches[1])
                Write-TauriLog "DL_DONE" $Matches[1]
            }
        }
    }

    function Clear-TauriInstallError {
        param([string]$Message)
        if ($TauriMode) {
            Write-TauriLog "ERROR_CLEAR" $Message
            [Console]::Error.WriteLine("[TAURI:ERROR_CLEAR] $Message")
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

    # Machine arch; Get-TauriDiagArch above reports the process. An emulated x64 shell on
    # ARM64 reports AMD64, but PROCESSOR_ARCHITEW6432 is ARM64 in exactly that case.
    function Get-HostMachineArch {
        $osArch = ""
        try { $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() } catch { $osArch = "" }
        $signals = @([string]$env:PROCESSOR_ARCHITEW6432, [string]$env:PROCESSOR_ARCHITECTURE, $osArch)
        foreach ($s in $signals) {
            if ($s.ToLowerInvariant() -eq "arm64") { return "arm64" }
        }
        foreach ($s in $signals) {
            if ([string]::IsNullOrWhiteSpace($s)) { continue }
            switch ($s.ToLowerInvariant()) {
                "amd64" { return "x86_64" }
                "x64" { return "x86_64" }
                "x86" { return "x86" }
            }
        }
        return "unknown"
    }

    function Get-TauriTorchIndexFamily {
        param([string]$TorchIndexUrl)
        if ($SkipTorch) { return "none" }
        if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) { return "none" }
        # Drop query/fragment first so a token-authenticated pin classifies by family.
        $leaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
        if (@("cpu", "xpu", "cu118", "cu124", "cu126", "cu128", "cu130") -contains $leaf) { return $leaf }
        if ($leaf -match '^rocm[0-9]+\.[0-9]+$') { return $leaf }
        return "auto"
    }

    function Get-TauriGpuBranch {
        param([string]$TorchIndexFamily)
        if ($SkipTorch) { return "no_torch" }
        # Require a digit after "cu" so /current or /custom isn't branded CUDA (parity ^cu[0-9]).
        if ($TorchIndexFamily -match '^cu[0-9]') { return "cuda" }
        if ($TorchIndexFamily -like "rocm*") { return "rocm" }
        if ($TorchIndexFamily -eq "xpu") { return "xpu" }
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
        Write-TauriLog "ERROR_DEFAULT" $Message
        # Clear the release-preservation handoff on any failure: a non-Tauri `irm | iex`
        # run throws below and leaves the caller's session alive, so a leaked
        # UNSLOTH_KEPT_TORCH would let a later `studio setup`/`update` (or a retry that
        # skips the branch-entry clear) re-pin the abandoned exact torch release.
        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
        if (Get-Command Restore-StudioVenvRollback -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioVenvRollback
        }
        # Most failures return before the lock try/finally, and under `irm | iex`
        # these variables are the caller's own. Defined later, so probed like above.
        if (Get-Command Restore-StudioTempEnvironment -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioTempEnvironment
        }
        if ($TauriMode) {
            exit $Code
        }
        throw $Message
    }

    # ── Usable temporary storage ──
    # Windows picks the temp directory from TMP, then TEMP, then the profile, and
    # never checks it exists or is writable. The desktop app passes on whatever it
    # inherited (studio/src-tauri/src/install.rs sets neither); one report had it at
    # C:\Windows\TEMP, where the source Add-Type had just written was gone by the
    # time csc.exe opened it (issue #9140). The Python, uv and VC++ downloads stage
    # through it too. Probe it once and, if it cannot hold a file, point BOTH
    # variables at a directory we own: every child process and every
    # [System.IO.Path]::GetTempPath() call reads the process environment block.
    function Test-StudioDirectoryUsable {
        param(
            [string]$Path,
            # Only for a directory this installer OWNS. Probing the host's own
            # inherited TMP/TEMP must not bring it into existence: -Force creates
            # the whole parent chain, so a stale or mistyped TMP would have the
            # installer silently materialize a tree at a path nobody chose, and
            # then trust it. Absent means unusable, which is what the private
            # fallback is for.
            [switch]$CreateIfMissing
        )
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        try {
            if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
                if (-not $CreateIfMissing) { return $false }
                New-Item -ItemType Directory -Path $Path -Force -ErrorAction Stop | Out-Null
            }
            # Anything an earlier run could not delete. Bounded self-healing: the
            # probe below cannot clean up after itself when deletion is what
            # failed, so each such run used to leave one more file behind forever.
            try {
                $cutoff = (Get-Date).AddDays(-1)
                foreach ($old in @(Get-ChildItem -LiteralPath $Path -File -Filter "unsloth-probe-*.tmp" -ErrorAction Stop)) {
                    # Shape, not prefix. This runs in the HOST's temp directory,
                    # where a name that merely starts the same way belongs to
                    # somebody else; the probe below only ever writes eight hex
                    # characters, so anything else is not ours to delete.
                    if ($old.Name -notmatch '^unsloth-probe-[0-9a-f]{8}\.tmp$') { continue }
                    if ($old.LastWriteTime -lt $cutoff) {
                        Remove-Item -LiteralPath $old.FullName -Force -ErrorAction SilentlyContinue
                    }
                }
            } catch {}
            # Write, read back, delete -- not Test-Path: the failures that matter
            # (write without read, a scanner deleting the file) pass an existence check.
            $probe = Join-Path $Path ("unsloth-probe-" + [guid]::NewGuid().ToString('N').Substring(0, 8) + ".tmp")
            [System.IO.File]::WriteAllText($probe, "unsloth")
            $readBack = [System.IO.File]::ReadAllText($probe)
            # Deleting has to work too, and be VERIFIED. csc.exe writes its source
            # and its output into this directory and then cleans up, so one that
            # accepts a file and will not give it back is the shape that produced
            # #9140; a suppressed Remove-Item said nothing either way. Retried a
            # couple of times first, because a scanner holding the file for a
            # moment is not the same as a directory that denies deletion, and only
            # the second should cost a healthy host its own temp.
            $probeGone = $false
            foreach ($attempt in 1..3) {
                Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
                if (-not [System.IO.File]::Exists($probe)) { $probeGone = $true; break }
                Start-Sleep -Milliseconds 100
            }
            if (-not $probeGone) { return $false }
            return ($readBack -eq "unsloth")
        } catch {
            return $false
        }
    }

    function Remove-StudioStalePrivateTempDirectories {
        param([Parameter(Mandatory = $true)][string]$Root)
        # These outlive the install (an autostarted Unsloth inherits one as its own
        # %TEMP%), so bound the pile by age instead of deleting one still in use.
        try {
            $cutoff = (Get-Date).AddDays(-1)
            foreach ($stale in @(Get-ChildItem -LiteralPath $Root -Directory -Filter "ust-*" -ErrorAction Stop)) {
                # SHAPE, not prefix. The delete below is recursive and this is the
                # only ownership test there is, so it has to name a directory the
                # allocator could actually have made: "ust-" + $PID + "-" + 8 hex.
                # A prefix match takes "ust-legacy" or "ust-user-cache" too, and
                # since neither has a parseable PID the liveness check is skipped
                # for exactly the names least likely to be ours. scripts/uninstall.ps1
                # already requires this shape; the two had drifted apart.
                if ($stale.Name -notmatch '^ust-[0-9]+-[0-9a-f]{8}$') { continue }
                if ($stale.LastWriteTime -ge $cutoff) { continue }
                # Before any owner logic, because the allocator never makes a link:
                # anything here that IS one is not ours, reading owner.pid out of it
                # would read through it, and unlinking is safe whatever owns the
                # target. Not Remove-Item: on 5.1 without -Recurse it throws a
                # NullReferenceException on a junction that -ErrorAction
                # SilentlyContinue does not suppress (measured on windows-latest), and
                # -Recurse has walked THROUGH the link on some 5.1 builds.
                # Directory.Delete with recursive:$false cannot follow it.
                if ($stale.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    try { [System.IO.Directory]::Delete($stale.FullName, $false) } catch {}
                    continue
                }
                # Age alone is not proof it is unused, and this sweep runs before the
                # runtime mutex, so it could delete a live process's %TEMP%. owner.pid
                # names the process that INHERITED this directory (the autostarted
                # Unsloth, which outlives the installer); the PID in the name is only
                # the installer's and is already gone. If the owner is alive, leave it.
                # PID reuse only ever costs a directory its cleanup.
                $ownerPid = 0
                $ownerFile = Join-Path $stale.FullName "owner.pid"
                $recorded = $null
                try {
                    if ([System.IO.File]::Exists($ownerFile)) {
                        $recorded = [System.IO.File]::ReadAllText($ownerFile).Trim()
                    }
                } catch { $recorded = $null }
                if (-not [string]::IsNullOrWhiteSpace($recorded)) {
                    $null = [int]::TryParse($recorded, [ref]$ownerPid)
                }
                # Whether the owner was RECORDED, or only guessed from the name.
                # The name carries the installer's PID, and an installer that was
                # killed between Start-Process and the owner.pid write leaves a dead
                # PID in the name while the Unsloth it started is very much alive on
                # that directory as its own %TEMP%. Guessing therefore proves much
                # less than reading, and the two are not treated alike below.
                $ownerRecorded = ($ownerPid -gt 0)
                if ($ownerPid -le 0) {
                    $null = [int]::TryParse(($stale.Name -split '-')[1], [ref]$ownerPid)
                }
                # No recorded owner: unknown, not abandoned. Still collected, so the
                # pile stays bounded, but only once it has gone a whole week without
                # a single entry being created in it, which an Unsloth actually using
                # it as %TEMP% would not manage.
                if (-not $ownerRecorded -and $stale.LastWriteTime -ge (Get-Date).AddDays(-7)) {
                    continue
                }
                if ($ownerPid -gt 0) {
                    $ownerLives = $true
                    try {
                        $null = Get-Process -Id $ownerPid -ErrorAction Stop
                    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
                        # The only answer that means "abandoned"; any other failure
                        # says nothing about the owner, so keep the directory.
                        $ownerLives = $false
                    } catch {
                        $ownerLives = $true
                    }
                    if ($ownerLives) { continue }
                }
                Remove-Item -LiteralPath $stale.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        } catch {}
    }

    function Set-StudioPrivateTempOwner {
        param([Parameter(Mandatory = $true)][int]$OwnerProcessId)
        # Only meaningful if this run redirected the temp; otherwise it is the host's.
        if ($null -eq $script:StudioTempOverride) { return }
        # Only a directory this run created. The other override shape just pins the
        # absolute spelling of the host's own temp, and dropping a file in there
        # would be litter in somebody else's directory.
        if (-not $script:StudioTempOverride.Owned) { return }
        $owned = $script:StudioTempOverride.Path
        if ([string]::IsNullOrWhiteSpace($owned)) { return }
        try {
            [System.IO.File]::WriteAllText((Join-Path $owned "owner.pid"), [string]$OwnerProcessId)
        } catch {}
    }

    function Get-StudioPrivateTempRoots {
        # Only under paths scripts/uninstall.ps1 already reclaims (LOCALAPPDATA\
        # "Unsloth Studio", ~\.unsloth\.cache): anywhere else would survive an
        # uninstall, and a leftover directly under ~\.unsloth would be worse, since
        # that is removed only when empty.
        $roots = @()
        if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            $roots += (Join-Path $env:LOCALAPPDATA "Unsloth Studio\temp")
        }
        try {
            $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
            if (-not [string]::IsNullOrWhiteSpace($localAppData)) {
                $roots += (Join-Path $localAppData "Unsloth Studio\temp")
            }
        } catch {}
        if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
            $roots += (Join-Path $env:USERPROFILE ".unsloth\.cache\temp")
        }
        return $roots
    }

    function New-StudioPrivateTempDirectory {
        foreach ($root in @(Get-StudioPrivateTempRoots)) {
            # Short leaf: the .NET Framework compiler 5.1 shells out to is still
            # bound by the legacy path limit.
            $leaf = "ust-" + $PID + "-" + [guid]::NewGuid().ToString('N').Substring(0, 8)
            $candidate = Join-Path $root $leaf
            # Which of these did not exist BEFORE the probe touched anything. Only
            # those may be unwound below: a pre-provisioned "Unsloth Studio\temp"
            # with custom ACLs, or an empty junction pointing somewhere else, is
            # configuration this installer did not create and must not remove
            # merely for being empty and correctly named.
            $preAbsent = @{}
            $walk = $candidate
            for ($seen = 0; $seen -lt 4; $seen++) {
                if ([string]::IsNullOrEmpty($walk)) { break }
                $preAbsent[$walk] = (-not (Test-Path -LiteralPath $walk))
                $walk = [System.IO.Path]::GetDirectoryName($walk)
            }
            if (Test-StudioDirectoryUsable -Path $candidate -CreateIfMissing) {
                Remove-StudioStalePrivateTempDirectories -Root $root
                return $candidate
            }
            # The probe creates the candidate before it tests it, and -Force builds
            # the whole chain, so a root that fails leaves "Unsloth Studio\temp\ust-x"
            # behind; on a host where every root fails that is a data directory tree
            # conjured by an install that then gave up. Walk back up, but only through
            # the directories this path is made of and only while each one is EMPTY,
            # so a tree that already held something is never touched and neither is
            # ~\.unsloth itself, which is shared and is not ours to remove.
            $ours = @("temp", "Unsloth Studio", ".cache")
            $unwind = $candidate
            for ($depth = 0; $depth -lt 4; $depth++) {
                try {
                    if (-not $preAbsent[$unwind]) { break }
                    if (-not (Test-Path -LiteralPath $unwind -PathType Container)) { break }
                    $item = Get-Item -LiteralPath $unwind -Force -ErrorAction Stop
                    # A relocation junction is somebody's configuration even when the
                    # probe created it, and unlinking it is not "taking back what we
                    # made". Leave it and stop.
                    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { break }
                    if (@(Get-ChildItem -LiteralPath $unwind -Force -ErrorAction Stop).Count -gt 0) { break }
                    [System.IO.Directory]::Delete($unwind, $false)
                } catch { break }
                $unwind = [System.IO.Path]::GetDirectoryName($unwind)
                if ([string]::IsNullOrEmpty($unwind)) { break }
                if ($ours -notcontains [System.IO.Path]::GetFileName($unwind)) { break }
            }
        }
        return $null
    }

    $script:StudioTempOverride = $null
    $script:StudioTempChecked = $false
    function Initialize-StudioTempEnvironment {
        if ($script:StudioTempChecked) { return }
        $script:StudioTempChecked = $true
        # TMP wins over TEMP, so only fall through when TMP is unset. IsNullOrEmpty,
        # not IsNullOrWhiteSpace: GetTempPath takes the first of TMP/TEMP that is
        # merely non-empty, so a whitespace-only TMP is what Windows and every child
        # will use, and treating it as unset would probe a healthy TEMP and change
        # nothing.
        $inherited = if (-not [string]::IsNullOrEmpty($env:TMP)) { $env:TMP } else { $env:TEMP }
        # Resolve BEFORE probing, not after. Test-Path is relative to PowerShell's
        # location while the .NET file APIs are relative to the process working
        # directory, and Set-Location moves only the first, so probing a relative
        # value can check one directory and write to another.
        # Whitespace-only is left exactly as it is, so the probe below rejects it.
        # Resolving it first would turn "   " or a tab into the working directory
        # plus that name, which is creatable on some filesystems, and the installer
        # would manufacture a junk directory and then trust it as the host's temp.
        $absolute = $inherited
        if (-not [string]::IsNullOrWhiteSpace($inherited)) {
            try { $absolute = [System.IO.Path]::GetFullPath($inherited) } catch { $absolute = $inherited }
        }
        if (Test-StudioDirectoryUsable -Path $absolute) {
            # A host whose temp was fixed since the last run never allocates another
            # private directory, and the allocator is the only thing that sweeps.
            # Without this, whatever an earlier degraded run left behind ages in
            # place until an uninstall. Each root that does not exist is a no-op.
            foreach ($root in @(Get-StudioPrivateTempRoots)) {
                Remove-StudioStalePrivateTempDirectories -Root $root
            }
            # Pin what was probed. A relative value (temp, or the drive-relative
            # C:temp) is resolved by whoever reads it, and the install relocates
            # out of a Windows system directory further down, so the same value
            # could later name somewhere else, or nowhere. An already-absolute
            # value normalizes to itself and is left alone.
            if (-not [string]::Equals($absolute, $inherited, [System.StringComparison]::Ordinal)) {
                $script:StudioTempOverride = [pscustomobject]@{
                    TmpSet = ($null -ne $env:TMP)
                    TmpValue = $env:TMP
                    TempSet = ($null -ne $env:TEMP)
                    TempValue = $env:TEMP
                    Path = $absolute
                    Owned = $false
                }
                $env:TMP = $absolute
                $env:TEMP = $absolute
            }
            return
        }
        $private = New-StudioPrivateTempDirectory
        if (-not $private) {
            Write-StudioLine "[WARN] No writable temporary directory was found; downloads may fail." -ForegroundColor Yellow
            return
        }
        # Absent is not empty: restoring an absent variable as "" would change how
        # every later child resolves its own temp directory.
        $script:StudioTempOverride = [pscustomobject]@{
            TmpSet = ($null -ne $env:TMP)
            TmpValue = $env:TMP
            TempSet = ($null -ne $env:TEMP)
            TempValue = $env:TEMP
            Path = $private
            Owned = $true
        }
        $env:TMP = $private
        $env:TEMP = $private
        Write-StudioLine "[WARN] The inherited temporary directory is not usable; this install will use its own." -ForegroundColor Yellow
    }

    function Restore-StudioTempEnvironment {
        $override = $script:StudioTempOverride
        if ($null -eq $override) { return }
        $script:StudioTempOverride = $null
        if ($override.TmpSet) { $env:TMP = $override.TmpValue }
        else { Remove-Item Env:\TMP -ErrorAction SilentlyContinue }
        if ($override.TempSet) { $env:TEMP = $override.TempValue }
        else { Remove-Item Env:\TEMP -ErrorAction SilentlyContinue }
        # The directory stays: an autostarted Unsloth inherited it as its own %TEMP%,
        # and the host's real one is broken. The next run sweeps the old ones.
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
                    Write-StudioLine "[ERROR] --package requires an argument." -ForegroundColor Red
                    return (Exit-InstallFailure "--package requires an argument.")
                }
                $PackageName = $argList[$i]
            }
            "--with-llama-cpp-dir" {
                $i++
                if ($i -ge $argList.Count) {
                    Write-StudioLine "[ERROR] --with-llama-cpp-dir requires a path argument." -ForegroundColor Red
                    return (Exit-InstallFailure "--with-llama-cpp-dir requires a path argument.")
                }
                $WithLlamaCppDir = $argList[$i]
            }
        }
    }

    # Env-var equivalent for web installs; an explicit flag still wins.
    if ($env:UNSLOTH_NO_TORCH -in @('1', 'true', 'yes', 'on')) { $SkipTorch = $true }
    if ($env:UNSLOTH_SKIP_AUTOSTART -in @('1', 'true', 'yes', 'on')) { $SkipAutostart = $true }

    # Propagate to child processes (process-scoped).
    if ($script:UnslothVerbose) {
        $env:UNSLOTH_VERBOSE = '1'
    }

    if ($StudioLocalInstall) {
        $RepoRoot = (Resolve-Path (Split-Path -Parent $PSCommandPath)).Path
        if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
            Write-StudioLine "[ERROR] --local must be run from the unsloth repo root (pyproject.toml not found at $RepoRoot)" -ForegroundColor Red
            return (Exit-InstallFailure "--local must be run from the unsloth repo root")
        }
    }

    # Validate --package to prevent injection into shell/Python commands
    if ($PackageName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
        Write-StudioLine "[ERROR] --package name contains invalid characters (allowed: a-z A-Z 0-9 . _ -)" -ForegroundColor Red
        return (Exit-InstallFailure "--package name contains invalid characters")
    }

    # UNSLOTH_PYTHON pins the version (mirrors install.sh --python); default 3.13.
    $PythonVersion = if ($env:UNSLOTH_PYTHON) { $env:UNSLOTH_PYTHON } else { "3.13" }
    # python.org fallback patch when winget and the live listing both fail; bump alongside $PythonVersion.
    $PythonFallbackFullVersion = "3.13.13"
    # Patch releases the stack cannot run; mirrors PYTHON_SKIP in install.sh.
    # Windows resolves an installed interpreter and hands uv its path rather
    # than a version, so uv never picks one of these -- but the machine may
    # already have it, and $PythonFallbackFullVersion above is what replaces it.
    $PythonSkip = @("3.13.8")
    # The entry above is skipped for one reason: it cannot `import torch`. A
    # -NoTorch install never imports it, so refusing the interpreter would send a
    # locked-down GGUF-only machine into winget/python.org recovery it may not be
    # able to complete, over a package it will not install.
    if ($SkipTorch) { $PythonSkip = @() }

    # Install dest priority: UNSLOTH_STUDIO_HOME, STUDIO_HOME alias, USERPROFILE-redirect, default.
    # Whitespace-only == unset (matches the Python resolvers' .strip()).
    $envOverrideVar = $null
    $envOverride = $null
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
        $envOverrideVar = "UNSLOTH_STUDIO_HOME"
        $envOverride = $env:UNSLOTH_STUDIO_HOME.Trim()
    } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
        $envOverrideVar = "STUDIO_HOME"
        $envOverride = $env:STUDIO_HOME.Trim()
    }
    $defaultProfile = $null
    try { $defaultProfile = [Environment]::GetFolderPath("UserProfile") } catch {}
    $tauriProfile = if ($defaultProfile) { $defaultProfile } else { $env:USERPROFILE }

    # GetFinalPathNameByHandleW is the only exact answer: it follows junctions,
    # symlinks and SUBST drives, expands 8.3 aliases and reports the on-disk
    # spelling, none of which GetFullPath does. It costs a C# compile, and 5.1 (the
    # interpreter the desktop app spawns) compiles by writing the source to %TEMP%
    # and running csc.exe. When that directory is unusable, or a scanner eats the
    # source, Add-Type throws CS2001, which used to abort a first launch as "Could
    # not create the Unsloth install lock" (issue #9140). Try once, retry with a
    # %TEMP% we own, cache the answer (callers resolve dozens of paths), then let
    # Get-StudioLexicalPath carry the run.
    $script:StudioFinalPathNativeState = $null
    # Reset with the rest: under `irm | iex` these are the caller's own.
    $script:StudioNativeResolveWarned = $false
    $script:StudioFinalPathWarned = $false
    function Write-StudioFinalPathDegraded {
        param([string]$Reason)
        if ($script:StudioFinalPathWarned) { return }
        $script:StudioFinalPathWarned = $true
        Write-StudioLine "[WARN] Could not load the native path resolver ($Reason)." -ForegroundColor Yellow
        Write-StudioLine "       Continuing with the PowerShell resolver; installation is unaffected." -ForegroundColor Yellow
    }

    function Initialize-StudioFinalPathNativeType {
        if ("UnslothStudioFinalPathV2" -as [type]) {
            $script:StudioFinalPathNativeState = $true
            return $true
        }
        if ($null -ne $script:StudioFinalPathNativeState) { return $script:StudioFinalPathNativeState }
        # Constrained Language Mode forbids Add-Type, so compiling would only produce
        # a second, less honest error.
        $languageMode = "FullLanguage"
        try { $languageMode = [string]$ExecutionContext.SessionState.LanguageMode } catch {}
        if ($languageMode -ne "FullLanguage") {
            $script:StudioFinalPathNativeState = $false
            Write-StudioFinalPathDegraded -Reason "PowerShell is in $languageMode"
            return $false
        }
        Initialize-StudioTempEnvironment
        $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

public static class UnslothStudioFinalPathV2
{
    private const uint FileShareRead = 0x00000001;
    private const uint FileShareWrite = 0x00000002;
    private const uint FileShareDelete = 0x00000004;
    private const uint OpenExisting = 3;
    private const uint FileFlagBackupSemantics = 0x02000000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFileW(
        string fileName,
        uint desiredAccess,
        uint shareMode,
        IntPtr securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetFinalPathNameByHandleW(
        SafeFileHandle file,
        StringBuilder path,
        uint pathLength,
        uint flags);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(
        uint desiredAccess,
        bool inheritHandle,
        int processId);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool QueryFullProcessImageNameW(
        IntPtr process,
        uint flags,
        StringBuilder path,
        ref uint pathLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    public static string Resolve(string path)
    {
        using (SafeFileHandle handle = CreateFileW(
            path,
            0,
            FileShareRead | FileShareWrite | FileShareDelete,
            IntPtr.Zero,
            OpenExisting,
            FileFlagBackupSemantics,
            IntPtr.Zero))
        {
            if (handle.IsInvalid)
                throw new Win32Exception(Marshal.GetLastWin32Error());

            StringBuilder buffer = new StringBuilder(512);
            uint length = GetFinalPathNameByHandleW(
                handle, buffer, (uint)buffer.Capacity, 0);
            if (length == 0)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            if (length >= buffer.Capacity)
            {
                buffer = new StringBuilder((int)length + 1);
                length = GetFinalPathNameByHandleW(
                    handle, buffer, (uint)buffer.Capacity, 0);
                if (length == 0)
                    throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            if (length >= buffer.Capacity)
                throw new InvalidOperationException("Final path exceeded the allocated buffer");
            return buffer.ToString();
        }
  }

    public static string GetProcessImagePath(int processId)
    {
        const uint ProcessQueryLimitedInformation = 0x1000;
        IntPtr process = OpenProcess(ProcessQueryLimitedInformation, false, processId);
        if (process == IntPtr.Zero)
        {
            return null;
        }
        try
        {
            StringBuilder path = new StringBuilder(32768);
            uint pathLength = (uint)path.Capacity;
            return QueryFullProcessImageNameW(process, 0, path, ref pathLength)
                ? path.ToString()
                : null;
        }
        finally
        {
            CloseHandle(process);
        }
  }
}
'@
        $firstError = $null
        try {
            Add-Type -TypeDefinition $source -ErrorAction Stop
        } catch {
            $firstError = $_.Exception.Message
        }
        # A compile that reports failure can still have loaded the type, and the same
        # name cannot be defined twice in one session.
        if ("UnslothStudioFinalPathV2" -as [type]) {
            $script:StudioFinalPathNativeState = $true
            return $true
        }
        $private = New-StudioPrivateTempDirectory
        if ($private) {
            $hadTmp = ($null -ne $env:TMP)
            $previousTmp = $env:TMP
            $hadTemp = ($null -ne $env:TEMP)
            $previousTemp = $env:TEMP
            try {
                # Both, because GetTempPath reads TMP first.
                $env:TMP = $private
                $env:TEMP = $private
                try { Add-Type -TypeDefinition $source -ErrorAction Stop } catch {}
            } finally {
                if ($hadTmp) { $env:TMP = $previousTmp } else { Remove-Item Env:\TMP -ErrorAction SilentlyContinue }
                if ($hadTemp) { $env:TEMP = $previousTemp } else { Remove-Item Env:\TEMP -ErrorAction SilentlyContinue }
                # Only now: deleting while csc.exe still holds it is the race being
                # worked around.
                Remove-Item -LiteralPath $private -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        if ("UnslothStudioFinalPathV2" -as [type]) {
            $script:StudioFinalPathNativeState = $true
            return $true
        }
        $script:StudioFinalPathNativeState = $false
        # First line of the compiler output, not the whole C# dump it echoes after.
        $reason = if ($firstError) { ($firstError -split "`r?`n")[0].Trim() } else { "compilation failed" }
        Write-StudioFinalPathDegraded -Reason $reason
        return $false
    }

    function Resolve-StudioLinkTarget {
        param([Parameter(Mandatory = $true)][string]$Path)
        $item = $null
        try { $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop } catch { return $null }
        $target = $null
        # PowerShell 7 walks the whole chain; 5.1 exposes only the raw reparse target,
        # relative for a relative symlink and still carrying a junction's NT prefix.
        if ($item.PSObject.Methods.Name -contains 'ResolveLinkTarget') {
            try {
                $final = $item.ResolveLinkTarget($true)
                if ($final) { $target = [string]$final.FullName }
            } catch { $target = $null }
        }
        if ([string]::IsNullOrWhiteSpace($target)) {
            $raw = $null
            try { $raw = $item.Target } catch { $raw = $null }
            # 5.1 hands this back as a COLLECTION, not a string, and not always an
            # [array], so unwrap anything that is not already a string.
            if ($null -ne $raw -and $raw -isnot [string]) {
                $raw = @($raw) | Select-Object -First 1
            }
            if (-not [string]::IsNullOrWhiteSpace($raw)) { $target = [string]$raw }
        }
        if ([string]::IsNullOrWhiteSpace($target)) { return $null }
        if ($target.StartsWith('\??\')) {
            $target = $target.Substring(4)
            # \??\UNC\server\share is the device spelling of \\server\share. Left as
            # "UNC\server\share" it reads as RELATIVE and gets combined with the
            # link's local parent, inventing a path; a wrong identity is a wrong mutex.
            if ($target.StartsWith('UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
                $target = '\\' + $target.Substring(4)
            } elseif ($target.StartsWith('Volume{', [System.StringComparison]::OrdinalIgnoreCase)) {
                # A mounted folder reports \??\Volume{GUID}\..., the same trap:
                # "Volume{...}\..." is not rooted either. \\?\ is the extended-length
                # spelling of that device path, so it keeps naming the volume.
                $target = '\\?\' + $target
            }
        }
        # "\real" is rooted as far as IsPathRooted is concerned but names no drive,
        # so GetFullPath would resolve it against the PROCESS current drive. Windows
        # resolves a drive-less target on the LINK's own volume, so anchor it there.
        if ($target.Length -ge 1 -and ($target[0] -eq '\' -or $target[0] -eq '/') -and
            -not ($target.Length -ge 2 -and ($target[1] -eq '\' -or $target[1] -eq '/'))) {
            $linkRoot = $null
            try { $linkRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path)) } catch { $linkRoot = $null }
            # Empty for a volume-GUID spelling; leaving the target alone beats guessing.
            if (-not [string]::IsNullOrEmpty($linkRoot)) {
                try { $target = [System.IO.Path]::Combine($linkRoot, $target.TrimStart('\', '/')) } catch {}
            }
        }
        try {
            if (-not [System.IO.Path]::IsPathRooted($target)) {
                $parent = [System.IO.Path]::GetDirectoryName($Path)
                if ([string]::IsNullOrEmpty($parent)) { return $null }
                $target = [System.IO.Path]::Combine($parent, $target)
            }
            $target = [System.IO.Path]::GetFullPath($target)
        } catch { return $null }
        # Compare like against like: $target went through GetFullPath, so $Path must
        # too, or a relative spelling misses the self-reference guard and loops.
        $self = $Path
        try { $self = [System.IO.Path]::GetFullPath($Path) } catch { $self = $Path }
        if ([string]::Equals(
            $target.TrimEnd('\', '/'), $self.TrimEnd('\', '/'), [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return $null
        }
        return $target
    }

    # Compiler-free stand-in for the native resolver. Normalized, not exact: it
    # cannot expand an 8.3 alias or recover stored casing, so callers are told the
    # answer is inexact. Never throws; an identity nobody can establish must not
    # stop an install.
    $script:StudioSubstMap = $null
    function Get-StudioSubstTarget {
        param([Parameter(Mandatory = $true)][string]$Path)
        # A SUBST drive is a DOS device mapping and no 5.1 API reports it: measured
        # on windows-latest, Get-PSDrive.DisplayRoot and Win32_LogicalDisk.ProviderName
        # are empty, GetFullPath and Resolve-Path hand X:\ straight back, and
        # (Get-Item X:\).Target reports the target's tail under the SUBST letter.
        # `subst` with no arguments prints the mapping in full ("X:\: => D:\real\dir"),
        # so that is what this reads, once.
        if ($null -eq $script:StudioSubstMap) {
            $script:StudioSubstMap = @{}
            try {
                foreach ($line in @(& "$env:SystemRoot\System32\subst.exe" 2>$null)) {
                    $m = [regex]::Match([string]$line, '^([A-Za-z]):\\?:\s*=>\s*(\S.*)$')
                    if ($m.Success) {
                        $script:StudioSubstMap[$m.Groups[1].Value.ToUpperInvariant()] =
                            $m.Groups[2].Value.TrimEnd()
                    }
                }
            } catch {}
        }
        if ($script:StudioSubstMap.Count -eq 0) { return $null }
        if ($Path.Length -lt 2 -or $Path[1] -ne ':') { return $null }
        $letter = ([string]$Path[0]).ToUpperInvariant()
        if (-not $script:StudioSubstMap.ContainsKey($letter)) { return $null }
        $tail = $Path.Substring(2).TrimStart('\', '/')
        $target = $script:StudioSubstMap[$letter]
        if ([string]::IsNullOrWhiteSpace($tail)) { return $target }
        try { return [System.IO.Path]::Combine($target, $tail) } catch { return $null }
    }

    function Get-StudioLexicalPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        $current = $null
        try { $current = [System.IO.Path]::GetFullPath($Path) } catch { return $Path }
        # Fold a SUBST drive BEFORE walking components: the Python runtime gate
        # resolves it, so leaving it would give the two sides different mutex names
        # for one directory and hide a running Unsloth from the in-use scan.
        $substituted = Get-StudioSubstTarget -Path $current
        if ($substituted) {
            try { $current = [System.IO.Path]::GetFullPath($substituted) } catch {}
        }
        # Hashtable, not a generic HashSet: already case-insensitive, and a
        # locked-down host (the kind that got here) can forbid generic types.
        $visited = @{}
        # A link on a PARENT component is the ordinary Windows shape (redirected
        # profiles), so walk from the root and restart at each one. Capped with a
        # visited set, because reparse points can point at each other.
        for ($hop = 0; $hop -lt 32; $hop++) {
            if ($visited.ContainsKey($current)) { break }
            $visited[$current] = $true
            $root = ""
            try { $root = [System.IO.Path]::GetPathRoot($current) } catch { break }
            if ([string]::IsNullOrEmpty($root) -or $current.Length -lt $root.Length) { break }
            $walked = $root
            $rewritten = $null
            $tail = $current.Substring($root.Length)
            # [char[]] on purpose: an untyped array binds Split's single-string
            # overload instead, which splits on nothing.
            foreach ($segment in $tail.Split([char[]]@('\', '/'), [System.StringSplitOptions]::RemoveEmptyEntries)) {
                $walked = [System.IO.Path]::Combine($walked, $segment)
                $target = Resolve-StudioLinkTarget -Path $walked
                if ($target) {
                    $remainder = $current.Substring($walked.Length).TrimStart('\', '/')
                    $rewritten = if ($remainder) { [System.IO.Path]::Combine($target, $remainder) } else { $target }
                    break
                }
            }
            if (-not $rewritten) { break }
            try { $current = [System.IO.Path]::GetFullPath($rewritten) } catch { $current = $rewritten; break }
        }
        try {
            $provider = (Resolve-Path -LiteralPath $current -ErrorAction Stop).ProviderPath
            if (-not [string]::IsNullOrWhiteSpace($provider)) { $current = $provider }
        } catch {}
        return $current
    }

    # Exact = $true means the native resolver answered, so the string is what it
    # always was. Callers keying a lock on it use that to judge an inequality.
    function Resolve-StudioFinalPathInfo {
        param([Parameter(Mandatory = $true)][string]$Path)
        $fullPath = [System.IO.Path]::GetFullPath($Path)
        $fullRoot = [System.IO.Path]::GetPathRoot($fullPath)
        if (-not $fullRoot -or $fullPath.Length -gt $fullRoot.Length) {
            $fullPath = $fullPath.TrimEnd('\', '/')
        }
        $existingPath = $fullPath
        $missingSegments = @()
        while (-not (Test-Path -LiteralPath $existingPath)) {
            $leaf = [System.IO.Path]::GetFileName($existingPath)
            $parent = [System.IO.Path]::GetDirectoryName($existingPath)
            if ([string]::IsNullOrEmpty($leaf) -or [string]::IsNullOrEmpty($parent)) {
                return [pscustomobject]@{ Path = $fullPath; Exact = $false }
            }
            $missingSegments = @($leaf) + $missingSegments
            $existingPath = $parent
        }
        $exact = $false
        $resolved = $null
        if (Initialize-StudioFinalPathNativeType) {
            try {
                $resolved = [UnslothStudioFinalPathV2]::Resolve($existingPath)
                $exact = $true
            } catch {
                # The helper COMPILED and still could not answer: a path renamed
                # between the Test-Path walk and CreateFileW, an access denial on a
                # component, a volume with no drive letter. Falling back keeps the
                # install alive, and Exact = $false already makes the runtime lock
                # fail closed, but nothing said so out loud: the degraded warning
                # below only fires when the compile itself failed. An operator was
                # left with a silently inexact identity on a host that looks fine.
                $resolved = $null
                if (-not $script:StudioNativeResolveWarned) {
                    $script:StudioNativeResolveWarned = $true
                    Write-StudioLine "[WARN] Could not resolve a path with the native helper; continuing with the PowerShell resolver." -ForegroundColor Yellow
                }
            }
        }
        if ([string]::IsNullOrEmpty($resolved)) {
            $resolved = Get-StudioLexicalPath -Path $existingPath
            $exact = $false
        }
        if ($resolved.StartsWith('\\?\UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
            $resolved = '\\' + $resolved.Substring(8)
        } elseif ($resolved.StartsWith('\\?\', [System.StringComparison]::OrdinalIgnoreCase)) {
            # Every extended spelling, INCLUDING \\?\Volume{GUID}\, because this string
            # is hashed into the runtime mutex name and unsloth_cli/_studio_runtime_gate.py
            # strips \\?\ unconditionally (_resolved_windows_path). Keeping the prefix
            # here made the installer and a running Unsloth compute different names for
            # one directory, so neither would exclude the other. Rootedness does matter
            # while a link target is being anchored, and Resolve-StudioLinkTarget keeps
            # the extended form for exactly that; nothing after this point anchors
            # anything, and Combine only drops its left side for a ROOTED right side.
            $resolved = $resolved.Substring(4)
        }
        foreach ($segment in $missingSegments) {
            $resolved = [System.IO.Path]::Combine($resolved, $segment)
        }
        $resolvedRoot = [System.IO.Path]::GetPathRoot($resolved)
        if ($resolvedRoot -and $resolved.Length -le $resolvedRoot.Length) {
            return [pscustomobject]@{ Path = $resolvedRoot; Exact = $exact }
        }
        return [pscustomobject]@{ Path = $resolved.TrimEnd('\', '/'); Exact = $exact }
    }

    function Get-StudioFinalPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        return (Resolve-StudioFinalPathInfo -Path $Path).Path
    }

    # Custom Unsloth roots are not supported with --tauri (the desktop app uses
    # the Windows profile folder). Pass through if the override is that same root.
    if ($TauriMode -and $envOverride) {
        $_tauriOverride = $envOverride
        if ($_tauriOverride -eq "~" -or $_tauriOverride -like "~/*" -or $_tauriOverride -like "~\*") {
            $_tauriOverride = (Join-Path $env:USERPROFILE $_tauriOverride.Substring(1).TrimStart('/','\'))
        }
        try {
            $_tauriOverride = Get-StudioFinalPath -Path $_tauriOverride
        } catch {}
        $_legacyTauriRoot = Join-Path $tauriProfile ".unsloth\studio"
        try {
            $_legacyTauriRoot = Get-StudioFinalPath -Path $_legacyTauriRoot
        } catch {}
        # Strip trailing separators so ".../studio\" matches ".../studio".
        $_trimSeps = @(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
        $_tauriOverride = $_tauriOverride.TrimEnd($_trimSeps)
        $_legacyTauriRoot = $_legacyTauriRoot.TrimEnd($_trimSeps)
        if ($_tauriOverride -ne $_legacyTauriRoot) {
            Write-StudioLine "ERROR: $envOverrideVar is not supported with --tauri." -ForegroundColor Red
            Write-StudioLine "       The desktop app uses the Windows profile .unsloth\studio root." -ForegroundColor Red
            Write-StudioLine "       Run install.ps1 without --tauri for custom-root shell installs," -ForegroundColor Yellow
            Write-StudioLine "       or unset the env var for default desktop installs." -ForegroundColor Yellow
            # Resolving the roots above can redirect TMP/TEMP, and this throw is well
            # before the lock try/finally. Under `irm | iex` those variables are the
            # caller's own and would stay pointed at an installer-owned directory.
            Restore-StudioTempEnvironment
            throw "$envOverrideVar is not supported with --tauri."
        }
    }


    # LOCALAPPDATA may be unset in service / CI contexts; guard Join-Path under ErrorActionPreference=Stop.
    $defaultDataDir = if ($env:LOCALAPPDATA -and -not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Join-Path $env:LOCALAPPDATA "Unsloth Studio"
    } else { $null }

    if ($envOverride) {
        # Tilde expansion: env vars aren't subject to it when quoted on assignment.
        if ($envOverride -eq "~" -or $envOverride -like "~/*" -or $envOverride -like "~\*") {
            $envOverride = (Join-Path $env:USERPROFILE $envOverride.Substring(1).TrimStart('/','\'))
        }
        try {
            # .NET API: New-Item -Path treats brackets as wildcards (no -LiteralPath in PS 5.1).
            [System.IO.Directory]::CreateDirectory($envOverride) | Out-Null
            $StudioHome = (Resolve-Path -LiteralPath $envOverride).Path
        } catch {
            Write-StudioLine "ERROR: $envOverrideVar=$envOverride cannot be created or accessed." -ForegroundColor Red
            # Same as the --tauri rejection above: still before the lock finally.
            Restore-StudioTempEnvironment
            throw "$envOverrideVar=$envOverride cannot be created or accessed."
        }
        $probe = Join-Path $StudioHome (".unsloth-write-probe-" + [guid]::NewGuid())
        try {
            [System.IO.File]::WriteAllText($probe, "")  # literal-path safe + closes handle
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
        } catch {
            Write-StudioLine "ERROR: $envOverrideVar=$StudioHome is not writable." -ForegroundColor Red
            Restore-StudioTempEnvironment
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
        # A redirected stdout is not a console and GetConsoleMode fails on a non-console handle,
        # so the block below could only return $false anyway. Answer without Add-Type, which runs
        # csc.exe and drops source in %TEMP%. install.rs spawns us with a pipe, so this is the path
        # the compile was on.
        if ($script:StudioStdoutRedirected) { return $false }
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

    Write-StudioLine ""
    if ($script:StudioVtOk -and -not $env:NO_COLOR) {
        Write-StudioLine ("  " + (Get-StudioAnsi Title) + $Sloth + " Unsloth Studio Installer (Windows)" + (Get-StudioAnsi Reset))
        Write-StudioLine ("  {0}{1}{2}" -f (Get-StudioAnsi Dim), $Rule, (Get-StudioAnsi Reset))
    } else {
        Write-StudioLine ("  {0} Unsloth Studio Installer (Windows)" -f $Sloth) -ForegroundColor DarkGreen
        Write-StudioLine "  $Rule" -ForegroundColor DarkGray
    }
    Write-StudioLine ""

    # Here so its warning lands under the banner. A no-op the second time: a --tauri
    # run with a custom root reaches it first via Initialize-StudioFinalPathNativeType.
    Initialize-StudioTempEnvironment

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
    # Direct registry access preserves REG_EXPAND_SZ (dotnet/runtime#1442). Append keeps existing tools first; Prepend for must-win.
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
                    Write-StudioLine "[WARN] User PATH is empty - initializing with $Directory" -ForegroundColor Yellow
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
            Write-StudioLine "[WARN] Could not update User PATH: $($_.Exception.Message)" -ForegroundColor Yellow
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
            Write-StudioLine ("  {0}{1}{2}{3}{4}{2}" -f $dim, $padded, $rst, $val, $Value)
        } else {
            $padded = if ($Label.Length -ge 15) { $Label.Substring(0, 15) } else { $Label.PadRight(15) }
            $fc = switch ($Color) {
                'Green' { 'DarkGreen' }
                'Yellow' { 'Yellow' }
                'Red' { 'Red' }
                'DarkGray' { 'DarkGray' }
                default { 'DarkGreen' }
            }
            # One composed record: two calls are two records, and a redirected
            # consumer splits the label from the value at the boundary.
            # Write-StudioLine picks the sink, so this stays a single line on
            # both of them.
            Write-StudioLine ("  {0}{1}" -f $padded, $Value) -ForegroundColor $fc
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
            Write-StudioLine ("  {0}{1}{2}{3}" -f $msgCol, $pad, $Message, (Get-StudioAnsi Reset))
        } else {
            $fc = switch ($Color) {
                'Yellow' { 'Yellow' }
                'Red' { 'Red' }
                default { 'DarkGray' }
            }
            Write-StudioLine ("  {0,-15}{1}" -f "", $Message) -ForegroundColor $fc
        }
    }

    # Managed llama.cpp access check. This script cannot dot-source setup.ps1, so
    # it holds byte-identical copies; test_denied_llama_cpp_preflight.py enforces that.
    # ── BEGIN SHARED WITH studio/setup.ps1 ──

    # Recognize ERROR_ACCESS_DENIED through PowerShell's wrapper exceptions.
    function Test-AccessDeniedError {
        param($ErrorRecord)

        $ex = if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) { $ErrorRecord.Exception } else { $ErrorRecord }
        while ($ex) {
            if ($ex -is [System.UnauthorizedAccessException]) { return $true }
            # IOException uses HRESULT; Win32Exception uses NativeErrorCode.
            if ($ex.HResult -eq -2147024891) { return $true }
            if ($ex -is [System.ComponentModel.Win32Exception] -and $ex.NativeErrorCode -eq 5) { return $true }
            $ex = $ex.InnerException
        }
        if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) {
            return ($ErrorRecord.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::PermissionDenied)
        }
        return $false
    }

    # Keep denied ACLs distinct from absent paths instead of letting Test-Path throw.
    function Get-PathState {
        param(
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path,
            [ValidateSet("Any", "Leaf", "Container")][string]$PathType = "Any"
        )

        if ([string]::IsNullOrWhiteSpace($Path)) { return "Absent" }
        try {
            if (Test-Path -LiteralPath $Path -PathType $PathType -ErrorAction Stop) { return "Present" }
            return "Absent"
        } catch {
            if (Test-AccessDeniedError $_) { return "Denied" }
            # Malformed path, offline drive, dangling link: nothing usable there.
            return "Absent"
        }
    }

    # Dir, marker and listing: no single probe separates readable/absent/denied.
    function Get-LlamaCppInstallReadState {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

        $dirState = Get-PathState -Path $Path -PathType Container
        if ($dirState -eq "Denied") { return "Denied" }
        if ($dirState -ne "Present") { return "Absent" }
        switch (Get-PathState -Path (Join-Path $Path "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf) {
            "Denied"  { return "Denied" }
        }
        try { $null = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1) }
        catch {
            if (Test-AccessDeniedError $_) { return "Denied" }
            # Nonfatal here: nothing has been installed yet.
        }
        return "Readable"
    }

    # Describe a denied link target without risking another reporting failure.
    function Get-PathDenialDetail {
        param([Parameter(Mandatory = $true)][AllowNull()][AllowEmptyString()][string]$Path)

        if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        if (-not $item) { return "" }
        # Non-filesystem providers do not expose FileSystemInfo attributes.
        if ($item -isnot [System.IO.FileSystemInfo]) { return "" }
        if (-not ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { return "" }
        $target = $null
        try { $target = $item.Target } catch { $target = $null }
        # PS 5.1 exposes .Target as a collection; PS 7 as a string.
        if ($target) { return " (it is a link to $(@($target) -join ', '))" }
        return " (it is a link)"
    }

    # Print guidance; returns the failure reason as its only pipeline output.
    function Write-PathAccessDenied {
        param(
            [Parameter(Mandatory = $true)][AllowNull()][AllowEmptyString()][string]$Path,
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Label,
            # Never tell users to delete a build they supplied.
            [switch]$UserSupplied,
            # Unreadable custom homes cannot be identified as managed caches.
            [switch]$OwnershipUnverified
        )

        step "permissions" "$Label at $Path cannot be read: access is denied$(Get-PathDenialDetail -Path $Path)" "Red"
        if ($UserSupplied) {
            substep "Unsloth will not touch a directory you pointed it at, so this has to be fixed at the source" "Yellow"
            substep "Restore access with these two in an elevated PowerShell, or point UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build:" "Yellow"
        } elseif ($OwnershipUnverified) {
            substep "Unsloth cannot confirm this folder is its own install while it is unreadable, so it will not tell you to remove it" "Yellow"
            substep "Restore access with these two in an elevated PowerShell, or move the folder aside and re-run setup:" "Yellow"
        } else {
            substep "This folder lives outside the app, so reinstalling Unsloth Studio, to any drive, reuses it and fails the same way" "Yellow"
            substep "Simplest fix: close Unsloth, delete or rename $Path, then re-run setup (it is a managed cache and gets reinstalled)" "Yellow"
            substep "If deleting is also denied, run these two in an elevated PowerShell, then re-run setup:" "Yellow"
        }
        substep "takeown /F `"$Path`" /R /D Y" "Yellow"
        substep "icacls `"$Path`" /reset /T" "Yellow"
        substep "Antivirus or Controlled folder access can deny this path too; allow or exclude it, then retry" "Yellow"
        if ($UserSupplied) {
            return "Access denied reading $Label at $Path. Restore access with takeown/icacls, or point UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build, then re-run setup."
        }
        if ($OwnershipUnverified) {
            return "Access denied reading $Label at $Path. Unsloth cannot confirm that folder is its own install while it is unreadable: restore access with takeown/icacls, or move it aside, then re-run setup."
        }
        return "Access denied reading the existing $Label at $Path. Delete or rename that folder (Unsloth reinstalls it) or restore access with takeown/icacls, then re-run setup. Reinstalling the app does not reset it."
    }

    # Canonicalize a directory for comparison; unresolvable ones are only normalized.
    function Get-CanonicalDir {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

        $trimmedPath = $Path.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmedPath)) { return $trimmedPath }
        $resolvedPath = $null
        if ((Get-PathState -Path $trimmedPath -PathType Container) -eq "Present") {
            try { $resolvedPath = (Resolve-Path -LiteralPath $trimmedPath).Path } catch {}
        }
        if (-not $resolvedPath) {
            try {
                $resolvedPath =
                    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($trimmedPath)
            } catch {
                $resolvedPath = [System.Environment]::ExpandEnvironmentVariables($trimmedPath)
            }
            try { $resolvedPath = [System.IO.Path]::GetFullPath($resolvedPath) } catch {}
        }
        # Resolve-Path keeps a trailing separator, so trim after both branches (never a root).
        try {
            $root = [System.IO.Path]::GetPathRoot($resolvedPath)
            if ($root -and $resolvedPath.Length -gt $root.Length) { return $resolvedPath.TrimEnd('\', '/') }
        } catch {}
        return $resolvedPath
    }

    # Compare canonical homes so path spelling does not change ownership policy.
    function Test-StudioHomeIsCustom {
        return ((Get-CanonicalDir -Path $StudioHome) -ne
            (Get-CanonicalDir -Path (Join-Path $env:USERPROFILE ".unsloth\studio")))
    }

    # Explicit staging root, shared default cache, or the custom Unsloth home's tree.
    function Get-ManagedLlamaCppDir {
        param([AllowNull()][string]$StagingRoot = $null)

        if ($StagingRoot) {
            return (Join-Path $StagingRoot "llama.cpp")
        }
        if (-not (Test-StudioHomeIsCustom)) {
            return (Join-Path $env:USERPROFILE ".unsloth\llama.cpp")
        }
        return (Join-Path (Get-CanonicalDir -Path $StudioHome) "llama.cpp")
    }

    # Failure reason when the managed tree is denied; never touches its ACLs.
    function Invoke-ManagedLlamaCppPreflight {
        param([AllowNull()][string]$StagingRoot = $null)

        # Let the existing profile validation handle a missing USERPROFILE later.
        if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }
        $dir = Get-ManagedLlamaCppDir -StagingRoot $StagingRoot
        if ((Get-LlamaCppInstallReadState -Path $dir) -ne "Denied") { return $null }
        Write-StudioLine ""
        # A denied custom home cannot be claimed as an Unsloth-managed cache.
        $homeIsCustom = Test-StudioHomeIsCustom
        # Preserve user-supplied wording when either override names this tree.
        $suppliedDir = if ($WithLlamaCppDir) { $WithLlamaCppDir } else { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR }
        $userSupplied = (-not [string]::IsNullOrWhiteSpace($suppliedDir)) -and
            ((Get-CanonicalDir -Path $suppliedDir) -eq (Get-CanonicalDir -Path $dir))
        $reason = Write-PathAccessDenied -Path $dir -Label "llama.cpp install" `
            -UserSupplied:$userSupplied -OwnershipUnverified:$homeIsCustom
        substep "Stopping here, before phase 1: nothing has been downloaded or installed" "Yellow"
        substep "Fix access, then run the same install, setup, or update command again" "Yellow"
        Write-StudioLine ""
        return "$reason Nothing was installed."
    }

    # ── END SHARED WITH studio/setup.ps1 ──

    # Redact index-URL credentials (userinfo + ?query= + #fragment) from captured installer
    # output before printing on failure; uv/pip errors echo the failing --index-url verbatim.
    # Mirrors the other installers. Verbose mode streams uncaptured, so it isn't redacted.
    function Redact-InstallOutput {
        param([string]$Text)
        if (-not $Text) { return $Text }
        $Text = $Text -replace '(https?://)[^/@\s`]+@', '$1<redacted>@'
        $Text = $Text -replace '([?&][^=\s&`]+)=[^&#\s`]+', '$1=<redacted>'
        # A #token fragment is as sensitive as a query.
        return $Text -replace '(https?://[^\s`#]+)#[^\s`]+', '$1#<redacted>'
    }

    # Run native commands quietly (mirrors install.sh); full output only with --verbose / UNSLOTH_VERBOSE=1.
    function Invoke-InstallCommand {
        param(
            [Parameter(Mandatory = $true)][ScriptBlock]$Command,
            [string]$Label = "install command"
        )
        # Installer-pinned index installs must beat an inherited uv mirror (#6898): for
        # --default-index, clear uv index env vars and set UV_NO_CONFIG=1 so a uv.toml/pyproject index can't outrank the CLI pin.
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
            Write-TauriLog "OUTPUT_CLEAR" $Label
            if ($script:UnslothVerbose) {
                # Merge stderr into stdout so progress/warning output stays visible
                # without flipping $? on successful native commands (PS 5.1 treats
                # stderr records as errors that set $? = $false even on exit code 0).
                # Redact per record: uv echoes index URLs (credentials and all) in
                # its errors, and verbose mode must not bypass the quiet path's
                # redaction. ForEach-Object/Out-Host leave $LASTEXITCODE untouched.
                & $Command 2>&1 | ForEach-Object {
                    Write-UvDownloadMarker "$_"
                    Redact-InstallOutput "$_"
                } | Out-Host
            } else {
                # Streamed, not collected, so a marker reaches the app mid-download.
                $collected = [System.Text.StringBuilder]::new()
                & $Command 2>&1 | ForEach-Object {
                    $line = "$_"
                    [void]$collected.AppendLine($line)
                    Write-UvDownloadMarker $line
                }
                $output = $collected.ToString()
                if ($LASTEXITCODE -ne 0) {
                    Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red
                }
            }
            $exitCode = [int]$LASTEXITCODE
            if ($exitCode -eq 0) {
                Clear-TauriInstallError "$Label recovered"
            } else {
                Write-TauriLog "ERROR_OUTPUT" "$Label failed (exit code $exitCode)"
            }
            return $exitCode
        } finally {
            $ErrorActionPreference = $prevEap
            if ($savedUvIndex) {
                Remove-Item "Env:UV_NO_CONFIG" -ErrorAction SilentlyContinue
                foreach ($n in $savedUvIndex.Keys) { if ($null -ne $savedUvIndex[$n]) { Set-Item "Env:$n" $savedUvIndex[$n] } }
            }
        }
    }

    # Retry Invoke-InstallCommand on transient uv download failures with backoff; returns last exit code on permanent failure.
    function Invoke-InstallCommandRetry {
        param(
            [Parameter(Mandatory = $true, Position = 0)][ScriptBlock]$Command,
            [string]$Label = "install step"
        )
        # Sanitize overrides; default 3. Bounds 1..100 retries, 0..3600s (TryParse avoids overflow throw).
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
            $code = Invoke-InstallCommand -Command $Command -Label $Label
            if ($code -eq 0) { return 0 }
            if ($attempt -ge $maxAttempts) { return $code }
            substep ("retrying ""$Label"" after transient failure (attempt $($attempt + 1)/$maxAttempts, waiting ${delay}s)...") "Yellow"
            Start-Sleep -Seconds $delay
            $attempt++
            $delay = $delay * 2
        }
    }

    # ── Managed CLI invocation, Application Control safe ──
    # Windows materializes the `unsloth` console script as a generated, unsigned .exe.
    # AppLocker, WDAC and Smart App Control deny it while the venv's python.exe, a copy
    # of the signed CPython binary, still runs, so setup died at "running unsloth studio
    # setup" (#8490). Everything here goes through the interpreter instead; the console
    # script is still installed and hardlinked, so an unaffected machine sees no change.
    #
    # Byte-identical to WINDOWS_CLI_ENTRYPOINT in studio/src-tauri/src/process.rs. Two
    # halves, both load bearing:
    #   argv[0] BEFORE the import, because unsloth_cli/__init__ decides at import time
    #   whether it is the console script (UTF-8 streams, the -np<N> rewrite) and typer
    #   reads it for the program name in usage text.
    #   sys.path[:1] drops the working directory `python -c` adds and a console script
    #   does not, so a stray `unsloth_cli` folder cannot shadow the managed package. It
    #   is a no-op under -P or PYTHONSAFEPATH. -I would drop it too, but -I implies -E,
    #   and discarding PYTHONPATH, PYTHONWARNINGS and user site-packages diverges from
    #   the console script on machines with no policy at all.
    # Written into every generated bin\unsloth.cmd and required by every ownership
    # check that accepts one. Mirrored in scripts/uninstall.ps1 and studio/setup.ps1.
    $script:UnslothCmdShimMarker = "unsloth-studio-managed-launcher"
    $script:UnslothCliTrampoline = "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"

    # Recognize ERROR_ACCESS_DISABLED_BY_POLICY through PowerShell's wrapper exceptions,
    # the same way Test-AccessDeniedError above recognizes ERROR_ACCESS_DENIED.
    # $LASTEXITCODE cannot answer this: no process was created, so it still holds the
    # exit code of whichever native command ran last.
    function Test-ApplicationControlBlock {
        param($ErrorRecord)

        $ex = if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) { $ErrorRecord.Exception } else { $ErrorRecord }
        while ($ex) {
            # Win32Exception carries NativeErrorCode; outer wrappers keep only the HRESULT
            # form of the same code (0x800704EC).
            if ($ex -is [System.ComponentModel.Win32Exception] -and $ex.NativeErrorCode -eq 1260) { return $true }
            if ($ex.HResult -eq -2147023636) { return $true }
            $ex = $ex.InnerException
        }
        return $false
    }

    # Print guidance; returns the failure reason as its only pipeline output, matching
    # Write-PathAccessDenied. Never suggests turning a security policy off: the fix is
    # for whoever owns the policy, and Unsloth already stopped needing the blocked file.
    function Write-ApplicationControlBlocked {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

        Write-StudioLine "[ERROR] Windows Application Control blocked the managed Python runtime." -ForegroundColor Red
        Write-StudioLine "        Windows error 1260 (ERROR_ACCESS_DISABLED_BY_POLICY)" -ForegroundColor Yellow
        Write-StudioLine "        Blocked program: $Path" -ForegroundColor Yellow
        Write-StudioLine "        Ask your administrator to review AppLocker `"EXE and DLL`" event 8004," -ForegroundColor Yellow
        Write-StudioLine "        or CodeIntegrity/Operational event 3077." -ForegroundColor Yellow
        return "Windows Application Control blocked the managed Python runtime at $Path (Windows error 1260)."
    }

    # One pre-quoted command line, not an argument array: Start-Process joins
    # -ArgumentList with spaces and quotes nothing, so the trampoline as an array
    # element reaches python as a dozen arguments. It holds no double quote and no
    # backslash, so wrapping it is unambiguous under CommandLineToArgvW.
    function Get-ManagedUnslothCliCommandLine {
        param([string[]]$Arguments = @())

        $quoted = '"' + $script:UnslothCliTrampoline + '"'
        # Callers pass bare subcommands today, but the signature invites a path:
        # unquoted, `--model C:\my models\a.gguf` arrives as three arguments.
        $safeArgs = @($Arguments | ForEach-Object {
            if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
        })
        return (@("-X", "utf8", "-c", $quoted) + $safeArgs) -join " "
    }

    # Run the managed `unsloth` CLI through the interpreter. The exit code is published
    # in $script:ManagedUnslothCliExit rather than returned, because the child's stdout
    # shares this function's output stream and capturing it would stop the install
    # streaming to the console and the Tauri log. $null means the interpreter never
    # started under Application Control, which is not an exit code.
    function Invoke-ManagedUnslothCli {
        param(
            [Parameter(Mandatory = $true)][string]$Python,
            [string[]]$Arguments = @()
        )

        # -X utf8, not the env var, so the child decodes whatever the console code page
        # is. No -I: it would discard PYTHONPATH and friends, which the console script
        # honors, and the trampoline already drops the -c working-directory entry.
        # Same form as the desktop's build_managed_cli_command.
        $pythonArgs = @("-X", "utf8", "-c", $script:UnslothCliTrampoline) + $Arguments
        $script:ManagedUnslothCliExit = $null
        try {
            # Otherwise a child that exits 0 inherits the last native command's code.
            $global:LASTEXITCODE = 0
            & $Python @pythonArgs
            $script:ManagedUnslothCliExit = [int]$LASTEXITCODE
        } catch {
            # $PSNativeCommandUseErrorActionPreference is off, so a nonzero exit never
            # lands here: anything caught is a failure to create the process at all.
            if (-not (Test-ApplicationControlBlock $_)) { throw }
        }
    }

    # Does Application Control deny this launcher here? Windows cannot be asked without
    # trying: AppLocker's cmdlets need the policy and an admin token, WDAC and Smart App
    # Control expose nothing. A denial is free (CreateProcess fails synchronously, no
    # child, 1260 straight back); an unaffected machine pays one `--version`, measured
    # at a quarter second. A process that started is not blocked whatever it does next,
    # so the timeout path still answers "not blocked" and just stops waiting.
    function Test-ShimLaunchBlocked {
        param([Parameter(Mandatory = $true)][string]$Path)

        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        # Both the PATH-shadow warning and the launch instructions ask, and the answer
        # cannot change between them.
        if ($null -ne $script:ShimLaunchBlockedCache -and
            $script:ShimLaunchBlockedPath -eq $Path) {
            return $script:ShimLaunchBlockedCache
        }
        $script:ShimLaunchBlockedPath = $Path
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Path
            $psi.Arguments = "--version"
            # UseShellExecute = false surfaces a denial as a Win32Exception rather than
            # a shell dialog, and is required before output can be redirected.
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            try {
                # Drain both pipes BEFORE waiting. With two redirected streams and no
                # reader, a child that fills a pipe buffer blocks writing while we block
                # waiting, and neither side moves. It is reachable here: the trampoline
                # no longer passes -I, so PYTHONPROFILEIMPORTTIME=1 in the environment
                # reaches this child and produces ~24 KB of stderr against a 4 KB buffer.
                # Reading them sequentially would not help, since the child can fill one
                # while we block on the other; async is the only correct answer.
                $stdout = $proc.StandardOutput.ReadToEndAsync()
                $stderr = $proc.StandardError.ReadToEndAsync()
                if ($proc.WaitForExit(20000)) {
                    # The timed overload does not wait for the readers to finish; the
                    # parameterless one does, and returns at once for an exited process.
                    $proc.WaitForExit()
                } else {
                    # It started, so it is not blocked, but leaving it running would
                    # hold the launcher open against the next update.
                    try { $proc.Kill() } catch { }
                }
                # Observed so a faulted read cannot surface later as an unobserved task.
                foreach ($reader in @($stdout, $stderr)) {
                    try { $null = $reader.Result } catch { }
                }
            } finally {
                $proc.Dispose()
            }
            $script:ShimLaunchBlockedCache = $false
            return $false
        } catch {
            $script:ShimLaunchBlockedCache = (Test-ApplicationControlBlock $_)
            return $script:ShimLaunchBlockedCache
        }
    }

    # Which launcher should the printed instructions name? The .cmd, when the .exe
    # cannot be used: either a policy denies it, or it is not there at all. The second
    # case is real -- antivirus quarantine takes the unsigned .exe, and the hardlink and
    # its copy fallback can both fail -- and Test-ShimLaunchBlocked answers $false for a
    # missing file, since nothing refused to start it. Naming a path that does not exist
    # is the one outcome worse than naming a blocked one.
    function Test-UnslothCmdShimPreferred {
        param(
            [Parameter(Mandatory = $true)][string]$ShimExe,
            [Parameter(Mandatory = $true)][string]$ShimCmd
        )

        if (-not (Test-Path -LiteralPath $ShimCmd -PathType Leaf)) { return $false }
        if (-not (Test-Path -LiteralPath $ShimExe -PathType Leaf)) { return $true }
        return (Test-ShimLaunchBlocked -Path $ShimExe)
    }

    # Relative path from $From (a directory) to $To, or $null with no common root
    # (different volumes, UNC vs local). Longhand because Path.GetRelativePath is .NET
    # Core only and Uri.MakeRelativeUri percent-encodes the spaces, '#' and '%' that
    # appear in real profile paths.
    function Get-RelativeShimPath {
        param(
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$From,
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$To
        )

        if ([string]::IsNullOrWhiteSpace($From) -or [string]::IsNullOrWhiteSpace($To)) { return $null }
        $fromParts = @($From.TrimEnd('\', '/') -split '[\\/]+' | Where-Object { $_ -ne "" })
        $toParts = @($To -split '[\\/]+' | Where-Object { $_ -ne "" })
        if ($fromParts.Count -eq 0 -or $toParts.Count -eq 0) { return $null }
        # Drive letter or share name: no common root means no relative path exists.
        if ($fromParts[0] -ne $toParts[0]) { return $null }
        $shared = 0
        while ($shared -lt $fromParts.Count -and $shared -lt $toParts.Count -and
               $fromParts[$shared] -eq $toParts[$shared]) {
            $shared++
        }
        # Guarded: a PowerShell range whose start passes its end counts DOWN, so
        # $toParts[2..1] returns the path reversed, not the empty tail it reads like.
        if ($shared -ge $toParts.Count) { return $null }
        $up = @("..") * ($fromParts.Count - $shared)
        $down = @($toParts[$shared..($toParts.Count - 1)])
        return (@($up + $down) -join '\')
    }

    # The `unsloth.cmd` companion to the shim .exe. A pure function of the two paths, so
    # a re-run reproduces it byte for byte, and %~dp0 keeps profile paths containing
    # spaces, '$', brackets and apostrophes out of the file. CRLF and ASCII, no BOM:
    # cmd.exe reads a BOM as part of the first command.
    #
    # Scope, stated plainly: this answers EXE-and-DLL enforcement of the unsigned
    # console script, which is issue #8490. A machine that also enforces AppLocker's
    # Script collection denies .cmd and .ps1 alike, and install.ps1 itself would not
    # have run there. The interpreter route is what carries such a machine; the shim
    # is the convenience on top of it.
    function Get-UnslothCmdShimContent {
        param(
            [Parameter(Mandatory = $true)][string]$ShimDir,
            [Parameter(Mandatory = $true)][string]$PythonPath
        )

        $relative = Get-RelativeShimPath -From $ShimDir -To $PythonPath
        $target = if ($relative) { "%~dp0$relative" } else {
            # Different volume: an absolute path is the only option left. '%' is the one
            # character cmd still expands inside double quotes, so double it.
            $PythonPath -replace '%', '%%'
        }
        $lines = @(
            "@echo off",
            "rem Runs the Unsloth CLI through the managed interpreter. The generated",
            "rem unsloth.exe beside it is unsigned, and Windows Application Control",
            "rem denies it on managed machines; this file is the way through.",
            # The ownership marker, and the only reason a .cmd may stand in for the other
            # sentinels. It gates a recursive delete, so it has to be something nobody
            # writes by accident: `from unsloth_cli import app` is a plausible line in
            # anyone's hand-rolled wrapper, this is not.
            "rem $script:UnslothCmdShimMarker",
            # With delayed expansion on (cmd /V:ON, or the machine-wide default) a '!'
            # is eaten out of every argument. exit /b ends the scope, so nothing leaks.
            "setlocal DisableDelayedExpansion",
            "`"$target`" -X utf8 -c `"$script:UnslothCliTrampoline`" %*",
            "@exit /b %errorlevel%"
        )
        return (($lines -join "`r`n") + "`r`n")
    }

    # Write bin\unsloth.cmd, but only when the bytes would change, so a re-run is a
    # no-op. Never fatal: the .exe shim is still the primary launcher and nothing
    # requires this file to exist.
    function Write-UnslothCmdShim {
        param(
            [Parameter(Mandatory = $true)][string]$ShimDir,
            [Parameter(Mandatory = $true)][string]$PythonPath
        )

        # Probes inside the try too: this runs under the venv rollback guard, so a throw
        # here would undo a successful install over an optional launcher.
        $shimCmd = Join-Path $ShimDir "unsloth.cmd"
        try {
            if (Test-Path -LiteralPath $shimCmd -PathType Container) {
                substep "cannot write $shimCmd, a directory is in the way" "Yellow"
                return
            }
            # A run killed between the write and the rename leaves its temp file behind,
            # and the next run has a different PID and would never look at it. Swept
            # before the unchanged-content return below, which is the path an interrupted
            # install takes on its way out. Only this file's own temps, only ones no
            # live process is still writing.
            Get-ChildItem -LiteralPath $ShimDir -Filter "unsloth.cmd.*.tmp" -File `
                -ErrorAction SilentlyContinue | ForEach-Object {
                $ownerPid = 0
                if ([int]::TryParse(($_.Name -replace '^unsloth\.cmd\.', '' -replace '\.tmp$', ''),
                                    [ref]$ownerPid) -and $ownerPid -eq $PID) { return }
                if ($ownerPid -gt 0 -and (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)) { return }
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }
            $content = Get-UnslothCmdShimContent -ShimDir $ShimDir -PythonPath $PythonPath
            $desired = (New-Object System.Text.UTF8Encoding($false)).GetBytes($content)
            # Bytes, not ReadAllText: that strips a BOM before comparing and PowerShell's
            # -eq is case insensitive, so a BOM-prefixed file (which cmd.exe reads as part
            # of the first command) would be declared unchanged and left broken forever.
            if (Test-Path -LiteralPath $shimCmd -PathType Leaf) {
                $existing = [System.IO.File]::ReadAllBytes($shimCmd)
                if (@(Compare-Object $existing $desired -SyncWindow 0).Count -eq 0) { return }
                # This directory is the installer's own, so the file is replaced either
                # way, but a file that carries neither our marker nor our trampoline was
                # written by something else and its owner deserves to read that it went.
                if (-not (Test-UnslothCmdShimFile -Path $shimCmd)) {
                    substep "replacing an unrecognised $shimCmd" "Yellow"
                }
            }
            # Publish by rename so a shell reading it mid-install never sees half a file.
            $tmp = "$shimCmd.$PID.tmp"
            [System.IO.File]::WriteAllBytes($tmp, $desired)
            try {
                Move-Item -LiteralPath $tmp -Destination $shimCmd -Force -ErrorAction Stop
            } finally {
                Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            }
        } catch {
            substep "could not write $shimCmd`: $($_.Exception.Message)" "Yellow"
        }
    }

    # Is this bin\unsloth.cmd one we wrote? The name alone is a plausible wrapper for
    # anything built on unsloth, and callers use the answer to decide whether a
    # user-chosen directory may be deleted. Mirrored in scripts/uninstall.ps1
    # (_IsUnslothCmdShim) and studio/setup.ps1.
    function Test-UnslothCmdShimFile {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        try {
            if ((Get-Item -LiteralPath $Path -ErrorAction Stop).Length -gt 8192) { return $false }
            $text = [System.IO.File]::ReadAllText($Path)
        } catch {
            # Unreadable proves nothing, and "proves nothing" must not mean "deletable".
            return $false
        }
        return ($text -like "*unsloth-studio-managed-launcher*" -and $text -like "*from unsloth_cli import app*")
    }

    function New-StudioShortcuts {
        param(
            [Parameter(Mandatory = $true)][string]$ManagedPythonPath
        )

        if (-not (Test-Path -LiteralPath $ManagedPythonPath)) {
            substep "cannot create shortcuts, managed Python not found at $ManagedPythonPath" "Yellow"
            return
        }
        try {
            # Persist an absolute path in launcher scripts so shortcut working
            # directory changes do not break process startup.
            $ManagedPythonPath = (Resolve-Path -LiteralPath $ManagedPythonPath).Path
            # Escape for single-quoted embedding in generated launcher script.
            # This prevents runtime variable expansion for paths containing '$'.
            $SingleQuotedPythonPath = $ManagedPythonPath -replace "'", "''"
            # Same escaping for the trampoline: it contains apostrophes of its own.
            $SingleQuotedTrampoline = $script:UnslothCliTrampoline -replace "'", "''"

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

            # Same-install discriminator: per-install opaque id read by launcher and backend
            # (/api/health); avoids leaking the install path and canonicalization drift. Lives at
            # $StudioHome\share\studio_install_id (found via _STUDIO_ROOT_RESOLVED) regardless of mode. 32 crypto bytes -> 64 hex.
            $_studioIdDir = Join-Path $StudioHome "share"
            if (-not (Test-Path -LiteralPath $_studioIdDir)) {
                [System.IO.Directory]::CreateDirectory($_studioIdDir) | Out-Null
            }
            $_studioIdFile = Join-Path $_studioIdDir "studio_install_id"
            $_studioRootId = ""
            if ((Test-Path -LiteralPath $_studioIdFile) -and `
                ((Get-Item -LiteralPath $_studioIdFile).Length -gt 0)) {
                $_studioRootId = ([System.IO.File]::ReadAllText($_studioIdFile)).Trim()
                # Same contract as the backend and the desktop app: 64
                # lowercase hex. -cnotmatch because -notmatch is case
                # insensitive and would take uppercase the backend rejects.
                # Anything else lands in a single-quoted assignment in the
                # generated launcher, so a planted quote would be code.
                if ($_studioRootId -cnotmatch '^[0-9a-f]{64}$') {
                    $_studioRootId = ""
                }
            }
            if (-not $_studioRootId) {
                $_idBytes = New-Object byte[] 32
                [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($_idBytes)
                $_studioRootId = -join ($_idBytes | ForEach-Object { $_.ToString('x2') })
                # Publish no-clobber: the desktop app mints this same id, so
                # -Force could replace one a running backend already reported.
                # Two-arg File.Move throws when the destination exists (the
                # 3-arg overwrite overload is .NET Core only), so we adopt it.
                $_idTmp = $_studioIdFile + ".$PID.tmp"
                [System.IO.File]::WriteAllText($_idTmp, $_studioRootId)
                try {
                    [System.IO.File]::Move($_idTmp, $_studioIdFile)
                } catch [System.IO.IOException] {
                    $_adoptedRootId = ""
                    try {
                        $_adoptedRootId = ([System.IO.File]::ReadAllText($_studioIdFile)).Trim()
                    } catch { }
                    if ($_adoptedRootId -cmatch '^[0-9a-f]{64}$') {
                        $_studioRootId = $_adoptedRootId
                    } else {
                        # Zero-length or malformed is an interrupted write
                        # or a planted value, not an id: replace it with one
                        # atomic rename, no unlink.
                        Move-Item -LiteralPath $_idTmp -Destination $_studioIdFile -Force
                    }
                } finally {
                    Remove-Item -LiteralPath $_idTmp -Force -ErrorAction SilentlyContinue
                }
            }

            # Env-mode: persist UNSLOTH_STUDIO_HOME (and llama path) and bake per-install
            # $portFile / $mutexName so concurrent custom-root launchers don't serialize on one global mutex. Default installs get an empty prefix.
            $studioHomeExport = if ($StudioRedirectMode -eq 'env') {
                # Reuse the preflight's managed path for legacy-home overrides.
                $_llamaPath = Get-ManagedLlamaCppDir
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
    # The managed interpreter, not the generated unsloth.exe console script: that one is
    # unsigned and Application Control denies it on managed machines (#8490).
    `$studioPython = '$SingleQuotedPythonPath'
    `$studioEntry = '$SingleQuotedTrampoline'
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
    # The entry point is single-quoted for the same reason: it carries apostrophes
    # around 'unsloth', and unquoted they would end the string mid-expression.
    `$studioCommand = "& '" + (`$studioPython -replace "'", "''") + "' -X utf8 -c '" +
        (`$studioEntry -replace "'", "''") + "' studio -p " + `$launchPort
    # RemoteSigned, not Bypass: the child runs an inline -Command against an executable, so no
    # script file is loaded and the two behave identically here. No reason to spend a scored
    # token on a launch that needs no policy relief.
    `$launchArgs = @(
        '-NoExit',
        '-NoProfile',
        '-ExecutionPolicy',
        'RemoteSigned',
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
            #
            # Content-compared first, like the .cmd shim: the launcher is regenerated by
            # every reinstall and by every `unsloth studio update` (--shortcuts-only),
            # and rewriting identical bytes moves its timestamp for no reason. A re-run
            # has to be a true no-op on disk.
            $utf8Bom = New-Object System.Text.UTF8Encoding($true)
            # Bytes including the BOM, not ReadAllText: that decodes and discards the
            # preamble, so a BOM-less launcher written by an older installer would be
            # called unchanged and never gain the BOM 5.1 needs to decode it.
            $desiredLauncher = $utf8Bom.GetPreamble() + $utf8Bom.GetBytes($launcherContent)
            $launcherUnchanged = $false
            if (Test-Path -LiteralPath $launcherPs1 -PathType Leaf) {
                try {
                    $existingLauncher = [System.IO.File]::ReadAllBytes($launcherPs1)
                    $launcherUnchanged =
                        (@(Compare-Object $existingLauncher $desiredLauncher -SyncWindow 0).Count -eq 0)
                } catch {
                    # Unreadable is not "unchanged"; fall through and rewrite it.
                    $launcherUnchanged = $false
                }
            }
            if (-not $launcherUnchanged) {
                [System.IO.File]::WriteAllBytes($launcherPs1, $desiredLauncher)
            }
            # WriteAllBytes replaces the unnamed data stream and leaves any other NTFS stream on
            # an existing file alone, so a launcher that somehow acquired a mark of the web keeps
            # it across the rewrite. The shortcut loads this under RemoteSigned, which refuses a
            # marked unsigned script, so clear the mark on the file we just authored. A no-op on
            # every ordinary install, where the stream was never there.
            #
            # Outside the content check on purpose: a launcher whose bytes already match still
            # has to lose the mark, and clearing an absent stream neither writes nor restamps
            # the file, so an unchanged re-run stays a no-op on disk either way.
            #
            # -Confirm:$false: Unblock-File declares SupportsShouldProcess at the default
            # Medium impact, so a profile setting $ConfirmPreference to Medium or Low would
            # prompt here, even for a launcher that never had the stream. ErrorAction does
            # not suppress a ShouldProcess prompt, and a noninteractive host turns it into
            # an error that skips shortcut setup entirely.
            Unblock-File -LiteralPath $launcherPs1 -Confirm:$false -ErrorAction SilentlyContinue
            # No .vbs launcher is written. A WScript.Shell .vbs that spawns a hidden
            # ExecutionPolicy-Bypass PowerShell is exactly the shape VBS-dropper
            # heuristics score (e.g. Kaspersky HEUR:Trojan.VBS.Agent.gen). The .lnk
            # shortcuts instead point straight at powershell.exe running
            # launch-studio.ps1 with a hidden window (selected below).

            # Delete any launch-studio.vbs left by a pre-hardening install (AV-flagged shape). Covers default and env-mode ($appDir).
            $legacyLauncherVbs = Join-Path $appDir "launch-studio.vbs"
            if (Test-Path -LiteralPath $legacyLauncherVbs) {
                Remove-Item -LiteralPath $legacyLauncherVbs -Force -ErrorAction SilentlyContinue
            }

            # Prefer bundled icon (local/dev), else best-effort download from raw GitHub; only attach if
            # the file has a valid ICO header. Snapshot the existing icon to gate the heavier cache refresh on a real change.
            $preIconHash = $null
            if (Test-Path -LiteralPath $iconPath) {
                try { $preIconHash = (Get-FileHash -LiteralPath $iconPath -Algorithm SHA256).Hash } catch {}
            }
            $hasValidIcon = $false
            if ($bundledIcon -and (Test-Path -LiteralPath $bundledIcon)) {
                try {
                    Copy-Item -LiteralPath $bundledIcon -Destination $iconPath -Force
                } catch {
                    Write-StudioLine "[DEBUG] Error copying bundled icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                }
            } elseif (-not (Test-Path -LiteralPath $iconPath)) {
                try {
                    Invoke-WebRequest -Uri $iconUrl -OutFile $iconPath -UseBasicParsing
                } catch {
                    Write-StudioLine "[DEBUG] Error downloading icon: $($_.Exception.Message)" -ForegroundColor DarkGray
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
                    Write-StudioLine "[DEBUG] Error validating or removing icon: $($_.Exception.Message)" -ForegroundColor DarkGray
                    Remove-Item -LiteralPath $iconPath -Force -ErrorAction SilentlyContinue
                }
            }

            # Only a real change (or a first/removed icon) triggers the heavy refresh; a no-op reinstall must not.
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

            # Env-mode: skip persistent .lnk shortcuts (may point at a deleted workspace); launcher + icon stay.
            if ($StudioRedirectMode -eq 'env') {
                substep "wrote launcher at $launcherPs1 (persistent shortcuts skipped in env-override mode)"
                return
            }

            # First install == no pre-existing .lnk; gates the heavy refresh so a no-op reinstall doesn't repeatedly clear caches (a dropper-like AV cluster).
            $firstInstall = -not (
                ($desktopLink -and (Test-Path -LiteralPath $desktopLink)) -or
                ($startMenuLink -and (Test-Path -LiteralPath $startMenuLink))
            )

            # Launch transport for the shortcuts: powershell.exe runs
            # launch-studio.ps1 with a hidden window. We deliberately avoid a
            # .vbs/WScript.Shell wrapper -- that script-engine shape is what AV
            # VBS-dropper heuristics score (Kaspersky HEUR:Trojan.VBS.Agent.gen).
            #
            # RemoteSigned, not Bypass: a hidden window beside a bypassed policy is the pair
            # Microsoft's detections key on, and install.rs makes the same call for the app's own
            # launch. This launcher is written locally, so RemoteSigned loads it either way.
            $powershellForLnk = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
            $shortcutTarget = $powershellForLnk
            $shortcutArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File `"$launcherPs1`""
            # A launcher on a share is a REMOTE script to PowerShell and RemoteSigned refuses an
            # unsigned one, so a roaming profile would get a shortcut that exits without starting
            # Unsloth. Bypass for that case only, and without the hidden window: a console beats
            # nothing launching. A mapped drive (H:, Z:) is the same share and the same zone, and
            # DriveInfo on the root reports Network for both.
            $launcherIsRemote = $launcherPs1 -like "\\*"
            if (-not $launcherIsRemote) {
                try {
                    $launcherIsRemote = ([System.IO.DriveInfo]::new(
                        [System.IO.Path]::GetPathRoot($launcherPs1))).DriveType -eq 'Network'
                } catch {}
            }
            if ($launcherIsRemote) {
                $shortcutArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$launcherPs1`""
            }

            try {
                $wshell = New-Object -ComObject WScript.Shell
                $createdShortcutCount = 0
                $createdShortcutPaths = @()
                $desiredIconLocation = if ($hasValidIcon) { "$iconPath,0" } else { $null }
                foreach ($linkPath in @($desktopLink, $startMenuLink)) {
                    if (-not $linkPath -or [string]::IsNullOrWhiteSpace($linkPath)) { continue }
                    try {
                        $shortcut = $wshell.CreateShortcut($linkPath)
                        # CreateShortcut returns an existing .lnk populated, so the
                        # incumbent's fields are readable before anything is assigned.
                        # Save() rewrites even when every field matches, moving the
                        # timestamp on a no-op reinstall. IconLocation carries the ",0"
                        # index back, so compare against the string we would set.
                        $shortcutUnchanged = (Test-Path -LiteralPath $linkPath -PathType Leaf) -and
                            ($shortcut.TargetPath -eq $shortcutTarget) -and
                            ($shortcut.Arguments -eq $shortcutArgs) -and
                            ($shortcut.WorkingDirectory -eq $appDir) -and
                            ($shortcut.WindowStyle -eq 7) -and
                            ($shortcut.Description -eq "Launch Unsloth Studio") -and
                            ((-not $desiredIconLocation) -or ($shortcut.IconLocation -eq $desiredIconLocation))
                        if (-not $shortcutUnchanged) {
                            $shortcut.TargetPath = $shortcutTarget
                            $shortcut.Arguments = $shortcutArgs
                            $shortcut.WorkingDirectory = $appDir
                            # Start minimized so the brief PowerShell console flash is muted.
                            $shortcut.WindowStyle = 7
                            $shortcut.Description = "Launch Unsloth Studio"
                            if ($hasValidIcon) {
                                $shortcut.IconLocation = $desiredIconLocation
                            }
                            $shortcut.Save()
                        }
                        $createdShortcutCount++
                        $createdShortcutPaths += $linkPath
                    } catch {
                        substep "could not create shortcut at ${linkPath}: $($_.Exception.Message)" "Yellow"
                    }
                }
                if ($createdShortcutCount -gt 0) {
                    substep "Created Unsloth Studio shortcut"
                    # Cheap per-item refresh so a rewritten same-name .lnk renders its new target/icon:
                    # per-item SHChangeNotify SHCNE_UPDATEITEM + SHCNF_PATHW (the global SHCNE_ASSOCCHANGED broadcast alone does not recover a stale item).
                    try {
                        Add-Type -Namespace UnslothShell -Name IconRefresh -MemberDefinition '[System.Runtime.InteropServices.DllImport("shell32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public static extern void SHChangeNotify(int eventId, uint flags, string item1, System.IntPtr item2);' -ErrorAction SilentlyContinue
                        # SHCNE_UPDATEITEM (0x00002000) + SHCNF_PATHW (0x0005) per shortcut
                        foreach ($scPath in $createdShortcutPaths) {
                            try { [UnslothShell.IconRefresh]::SHChangeNotify(0x00002000, 0x0005, $scPath, [System.IntPtr]::Zero) } catch {}
                        }
                        # SHCNE_ASSOCCHANGED (0x08000000) global refresh (belt-and-suspenders)
                        [UnslothShell.IconRefresh]::SHChangeNotify(0x08000000, 0, $null, [System.IntPtr]::Zero)
                    } catch {}
                    # Heavier icon-cache clear + StartMenuExperienceHost rebuild only on first install or icon change (doing it every no-op reinstall is a dropper-like cluster).
                    if ($firstInstall -or $iconChanged) {
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -ClearIconCache 2>$null } catch {}
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -show 2>$null } catch {}
                        # Win11 StartMenuExperienceHost keeps its own tile-icon cache that ie4uinit/explorer restart don't
                        # invalidate. Drop only the render caches (NEVER start2.bin, the pinned layout) and let the host rebuild. Win10 has no such host.
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
        # `unsloth studio update` reaches the installer only here, and returns before
        # the lock try/finally, so undo the temp redirection on every way out,
        # including the throw below. Under `irm | iex` these variables belong to the
        # caller's own session, so leaving them redirected outlives the install.
        try {
        if ($TauriMode) { return }
        # The launcher runs the interpreter, so that is what has to be there. Checking
        # unsloth.exe instead would refuse to regenerate shortcuts on exactly the
        # machines that need them, where the console script is present but denied.
        $ShortcutPython = Join-Path $VenvDir "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $ShortcutPython)) {
            Write-StudioLine "[ERROR] managed Python missing at $ShortcutPython; run install.ps1 first." -ForegroundColor Red
            # throw (not Exit-InstallFailure) so non-Tauri callers see rc != 0.
            throw "managed Python missing"
        }
        # `unsloth studio update` reaches the installer only through here, so without
        # this an install predating the .cmd never gets one -- and that population is
        # exactly the one that needs the escape hatch.
        # No-op when the bytes already match, and never fatal.
        # Created, not required: installers older than the shim directory never made
        # one, and those are precisely the installs that reach this branch through
        # `unsloth studio update` and never see the full installer again. Requiring
        # it here left them without the escape hatch forever. Directory creation is
        # idempotent, and a failure to create it must not fail an update, so the
        # write stays inside the same non-fatal helper.
        $ShortcutShimDir = Join-Path $StudioHome "bin"
        try {
            [System.IO.Directory]::CreateDirectory($ShortcutShimDir) | Out-Null
        } catch {
            substep "could not create $ShortcutShimDir`: $($_.Exception.Message)" "Yellow"
        }
        if (Test-Path -LiteralPath $ShortcutShimDir -PathType Container) {
            Write-UnslothCmdShim -ShimDir $ShortcutShimDir -PythonPath $ShortcutPython
            # An installer older than the shim directory put the venv's Scripts dir on
            # PATH instead, so the .cmd we just wrote would sit somewhere nothing looks.
            # Add-ToUserPath is idempotent and a no-op for every modern install, so this
            # only moves a machine that update is the sole route back into the installer
            # for. Env mode never writes the registry.
            if ($StudioRedirectMode -ne 'env') {
                if (Add-ToUserPath -Directory $ShortcutShimDir -Position 'Prepend') {
                    substep "added $ShortcutShimDir to PATH"
                }
            }
        }
        New-StudioShortcuts -ManagedPythonPath $ShortcutPython
        return
        } finally {
            Restore-StudioTempEnvironment
        }
    }

    # ── Leave Windows system directories before installing ──
    # "Run as administrator" starts in C:\Windows\System32, so a piped web run installs
    # from there and `unsloth studio setup` refuses only after PyTorch has downloaded,
    # then rolls back. Relocating is safe: nothing here reads the caller's directory
    # ($RepoRoot from $PSCommandPath, $StudioHome from the environment), so only
    # --with-llama-cpp-dir needs a rebase first. Not restored at the end, same reason as
    # the PSModulePath fix at the top: the interactive path ends running Unsloth in the
    # foreground, so a finally would not fire until it stops.
    $SystemRootDir = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
    $SystemRootDir = [System.IO.Path]::GetFullPath($SystemRootDir).TrimEnd('\')
    # Separator included, or siblings like C:\Windows.old and C:\WindowsStudio match too.
    $SystemRootPrefix = $SystemRootDir + [System.IO.Path]::DirectorySeparatorChar
    $CurrentDir = $null
    try {
        # FileSystem provider: a caller parked on HKLM:\ still has the location children inherit.
        $CurrentDir = [System.IO.Path]::GetFullPath(
            (Get-Location -PSProvider FileSystem -ErrorAction Stop).ProviderPath
        ).TrimEnd('\')
    } catch {
        $CurrentDir = $null
    }
    function Test-UnderSystemRoot {
        param([string]$Path)
        return $Path -and (
            $Path.Equals($SystemRootDir, [System.StringComparison]::OrdinalIgnoreCase) -or
            $Path.StartsWith($SystemRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
        )
    }
    $InSystemDir = Test-UnderSystemRoot $CurrentDir
    if ($InSystemDir) {
        if ($WithLlamaCppDir) {
            # Anchor to the directory the user typed it against, including the partially
            # qualified forms (C:llama.cpp, \llama.cpp). Not [System.IO.Path]::GetFullPath,
            # which resolves against [Environment]::CurrentDirectory, a separate location
            # that Set-Location never updates.
            try {
                $WithLlamaCppDir =
                    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WithLlamaCppDir)
            } catch {
                $WithLlamaCppDir = Join-Path $CurrentDir $WithLlamaCppDir
            }
        }
        # SYSTEM's profile is C:\Windows\System32\config\systemprofile, so a candidate
        # inside the Windows directory is no better than where we already are.
        $SafeDirCandidates = @($env:USERPROFILE, $HOME, $env:PUBLIC, $env:TEMP) |
            Where-Object {
                $_ -and (Test-Path -LiteralPath $_ -PathType Container) -and
                -not (Test-UnderSystemRoot ([System.IO.Path]::GetFullPath($_).TrimEnd('\')))
            }
        $SafeDir = $null
        foreach ($candidate in $SafeDirCandidates) {
            try {
                Set-Location -LiteralPath $candidate -ErrorAction Stop
                $SafeDir = (Get-Location -PSProvider FileSystem).ProviderPath
                break
            } catch {
                continue
            }
        }
        # $StudioHome came from USERPROFILE far above, so a SYSTEM account would keep
        # installing into C:\Windows\System32\config\systemprofile while we report having
        # escaped. Rebasing it would orphan the install (the runtime resolvers recompute
        # the root from USERPROFILE), so stop instead.
        $StudioHomeFull = ""
        try { $StudioHomeFull = [System.IO.Path]::GetFullPath($StudioHome).TrimEnd('\') } catch {}
        if ($SafeDir -and (Test-UnderSystemRoot $StudioHomeFull)) {
            Write-StudioLine ""
            Write-StudioLine "[ERROR] Unsloth would install into $StudioHomeFull," -ForegroundColor Red
            Write-StudioLine "        which is inside $SystemRootDir." -ForegroundColor Yellow
            Write-StudioLine "        That is where a service or the SYSTEM account keeps its profile." -ForegroundColor Yellow
            Write-StudioLine "        Sign in as a normal user, open PowerShell there, and run the" -ForegroundColor Yellow
            Write-StudioLine "        installer again:" -ForegroundColor Yellow
            Write-StudioLine "          irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Cyan
            Write-StudioLine ""
            return (Exit-InstallFailure "Refusing to install into $StudioHomeFull, which is inside $SystemRootDir. Run the installer from a normal user account.")
        }
        if ($SafeDir) {
            Write-TauriLog "STEP" "Left system directory $CurrentDir for $SafeDir"
            step "directory" "$CurrentDir is a Windows system folder" "Yellow"
            substep "Unsloth cannot install or run from there, so this install continues in:" "Yellow"
            substep "  $SafeDir" "Yellow"
            substep "This is normal: 'Run as administrator' opens PowerShell in System32." "Yellow"
        } else {
            Write-StudioLine ""
            Write-StudioLine "[ERROR] Unsloth cannot be installed from $CurrentDir." -ForegroundColor Red
            Write-StudioLine "        That is a Windows system folder, and Unsloth writes its virtual" -ForegroundColor Yellow
            Write-StudioLine "        environment caches, model downloads and build files into the" -ForegroundColor Yellow
            Write-StudioLine "        working directory, which Windows blocks there." -ForegroundColor Yellow
            Write-StudioLine "        'Run as administrator' opens PowerShell in System32, which is how" -ForegroundColor Yellow
            Write-StudioLine "        most people land here." -ForegroundColor Yellow
            # USERPROFILE was just rejected as a candidate, so naming it here would send
            # the user back into the same tree.
            Write-StudioLine "        Nothing outside $SystemRootDir was usable either (USERPROFILE," -ForegroundColor Yellow
            Write-StudioLine "        HOME, PUBLIC, TEMP), which normally means a service or the SYSTEM" -ForegroundColor Yellow
            Write-StudioLine "        account. Sign in as a normal user, open PowerShell there, and run" -ForegroundColor Yellow
            Write-StudioLine "        the installer again:" -ForegroundColor Yellow
            Write-StudioLine "          irm https://unsloth.ai/install.ps1 | iex" -ForegroundColor Cyan
            Write-StudioLine ""
            return (Exit-InstallFailure "Refusing to install from the Windows system directory $CurrentDir, and no folder outside $SystemRootDir was usable. Run the installer from a normal user account.")
        }
    }

    # ── Preflight the managed llama.cpp cache ──
    # After System32 relocation (it picks the profile), before any download.
    $llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight
    if ($llamaPreflightFailure) {
        return (Exit-InstallFailure $llamaPreflightFailure)
    }

    # ── Check winget ──
    # winget is only needed to install Python or uv. If both are
    # already on PATH (Windows ARM64 GitHub-hosted runners, manual
    # python.org + Astral uv installs, corporate locked-down hosts
    # without the Store, etc.) the script can proceed without it.
    # We defer the hard failure to the Python / uv install branches
    # below, where winget is actually invoked.
    function Enter-StudioNamedMutex {
        param([Parameter(Mandatory = $true)][string]$Name)
        $mutex = [System.Threading.Mutex]::new($false, $Name)
        $acquired = $false
        try {
            $acquired = $mutex.WaitOne(0)
        } catch [System.Threading.AbandonedMutexException] {
            $acquired = $true
        }
        if (-not $acquired) {
            $mutex.Dispose()
            return $null
        }
        return $mutex
    }

    function Get-StudioPathHash {
        param([Parameter(Mandatory = $true)][string]$Path)
        # Unchanged whenever the native resolver answered. When it could not, this
        # hashes the normalized spelling, so two ALIASES of one root (a junction and
        # its target, an 8.3 name and its long form) name different mutexes and do not
        # exclude each other. That needs two installs racing into one directory through
        # different spellings, and beats refusing to install, which is what this did.
        $canonical = (Get-StudioFinalPath -Path $Path).ToUpperInvariant()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($canonical)
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha256.ComputeHash($bytes)
        } finally {
            $sha256.Dispose()
        }
        $hex = -join ($digest | ForEach-Object { $_.ToString('x2') })
        return $hex
    }

    function Test-StudioPathEqual {
        param(
            [Parameter(Mandatory = $true)][string]$Left,
            [Parameter(Mandatory = $true)][string]$Right
        )
        try {
            $leftInfo = Resolve-StudioFinalPathInfo -Path $Left
            $rightInfo = Resolve-StudioFinalPathInfo -Path $Right
        } catch {
            Write-StudioLine "[WARN] Could not resolve Unsloth path identity; using the runtime lock." -ForegroundColor Yellow
            return $null
        }
        if ([string]::Equals(
            $leftInfo.Path, $rightInfo.Path, [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return $true
        }
        # Different spellings only prove different directories when both resolved
        # exactly; otherwise they may be aliases of one. $null is the caller's
        # "identity unresolved" signal and makes it take both runtime locks.
        if (-not $leftInfo.Exact -or -not $rightInfo.Exact) {
            Write-StudioLine "[WARN] Could not resolve Unsloth path identity; using the runtime lock." -ForegroundColor Yellow
            return $null
        }
        return $false
    }

    function Get-StudioInstallMutexName {
        param([Parameter(Mandatory = $true)][string]$Path)
        return "Global\UnslothStudioInstall-$(Get-StudioPathHash -Path $Path)"
    }

    function Get-StudioRuntimeMutexNameForSid {
        param([Parameter(Mandatory = $true)][string]$Sid)
        return "Global\UnslothStudioManagedEnvironment-$Sid"
    }

    function Get-StudioRuntimePathHash {
        param([Parameter(Mandatory = $true)][string]$Path)
        # Byte-for-byte spelling: .NET and Python disagree on Unicode case (ß).
        $canonical = Get-StudioFinalPath -Path $Path
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($canonical)
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha256.ComputeHash($bytes)
        } finally {
            $sha256.Dispose()
        }
        return (-join ($digest | ForEach-Object { $_.ToString('x2') }))
    }

    function Get-StudioRuntimeMutexNameForPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        return "Global\UnslothStudioManagedEnvironmentPath-$(Get-StudioRuntimePathHash -Path $Path)"
    }

    function Get-StudioCurrentUserSid {
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        if ($null -eq $identity) {
            throw "Could not determine the Windows user for the Unsloth runtime lock"
        }
        try {
            $sid = if ($identity.User) { $identity.User.Value } else { $null }
        } finally {
            $identity.Dispose()
        }
        if ([string]::IsNullOrWhiteSpace($sid)) {
            throw "Could not determine the Windows user SID for the Unsloth runtime lock"
        }
        return $sid
    }

    function Get-StudioRuntimeMutexName {
        return (Get-StudioRuntimeMutexNameForSid -Sid (Get-StudioCurrentUserSid))
    }

    function Get-StudioRuntimeMutexNames {
        param(
            [AllowNull()]$TauriRootMatch,
            [Parameter(Mandatory = $true)][string]$Path
        )
        $names = @()
        # true: Tauri default -> SID lock. false: custom root -> path lock.
        # null: identity unresolved -> take both and fail closed.
        if ($TauriRootMatch -ne $false) {
            $names += Get-StudioRuntimeMutexName
        }
        if ($TauriRootMatch -ne $true) {
            $names += Get-StudioRuntimeMutexNameForPath -Path $Path
        }
        return $names
    }

    function Enter-StudioInstallMutex {
        param([Parameter(Mandatory = $true)][string]$Path)
        return (Enter-StudioNamedMutex -Name (Get-StudioInstallMutexName -Path $Path))
    }

    function Exit-StudioInstallMutex {
        param([System.Threading.Mutex]$Mutex)
        if ($null -eq $Mutex) { return }
        try { $Mutex.ReleaseMutex() } catch {} finally { $Mutex.Dispose() }
    }

    function Test-StudioProtectedPathMatch {
        param(
            [Parameter(Mandatory = $true)][string]$Candidate,
            [Parameter(Mandatory = $true)][string]$ProtectedPath,
            [switch]$Exact
        )
        $candidateKey = $Candidate.TrimEnd('\', '/')
        $protectedKey = $ProtectedPath.TrimEnd('\', '/')
        if ([string]::Equals(
            $candidateKey, $protectedKey, [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return $true
        }
        if ($Exact) { return $false }
        $prefix = $protectedKey + [System.IO.Path]::DirectorySeparatorChar
        return $candidateKey.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
    }

    function Get-StudioDesktopProcessesForCurrentUser {
        $currentSid = Get-StudioCurrentUserSid
        try {
            $candidates = @(
                Get-CimInstance -ClassName Win32_Process `
                    -Filter "Name = 'unsloth-studio.exe'" -ErrorAction Stop
            )
        } catch {
            return
        }
        foreach ($candidate in $candidates) {
            try {
                $owner = Invoke-CimMethod -InputObject $candidate `
                    -MethodName GetOwnerSid -ErrorAction Stop
            } catch {
                continue
            }
            if ($null -eq $owner -or $owner.ReturnValue -ne 0) { continue }
            if ([string]::Equals(
                $owner.Sid, $currentSid, [System.StringComparison]::OrdinalIgnoreCase
            )) {
                [pscustomobject]@{
                    ProcessName = [System.IO.Path]::GetFileNameWithoutExtension($candidate.Name)
                    Id = [int]$candidate.ProcessId
                }
            }
        }
    }

    # QueryFullProcessImageNameW answers for processes whose MainModule is not
    # readable here, but needs the compiled helper. Without a fallback a host that
    # cannot compile would find NO running processes and overwrite a venv Unsloth has
    # open, so the ladder ends at Win32_Process. Every rung reports a real executable
    # image; a command line or working directory mentioning the path is never proof.
    $script:StudioProcessImageTable = $null
    $script:StudioProcessImageWarned = $false
    function Get-StudioProcessImagePath {
        param([Parameter(Mandatory = $true)][int]$ProcessId)
        if (Initialize-StudioFinalPathNativeType) {
            try {
                $native = [UnslothStudioFinalPathV2]::GetProcessImagePath($ProcessId)
                if (-not [string]::IsNullOrWhiteSpace($native)) { return $native }
            } catch {}
            return $null
        }
        if (-not $script:StudioProcessImageWarned) {
            $script:StudioProcessImageWarned = $true
            Write-StudioLine "[WARN] Scanning for running Unsloth processes without the native helper; a process this shell cannot inspect may go unnoticed." -ForegroundColor Yellow
        }
        $process = $null
        try { $process = Get-Process -Id $ProcessId -ErrorAction Stop } catch { $process = $null }
        if ($process) {
            # .Path is MainModule.FileName (an ETS ScriptProperty over it on 5.1), so
            # there is no second rung here: empty means MainModule was unreadable.
            try {
                if (-not [string]::IsNullOrWhiteSpace($process.Path)) { return $process.Path }
            } catch {}
        }
        # Queried once per run, not once per process: this is the slow rung.
        if ($null -eq $script:StudioProcessImageTable) {
            $script:StudioProcessImageTable = @{}
            try {
                foreach ($row in @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)) {
                    if (-not [string]::IsNullOrWhiteSpace($row.ExecutablePath)) {
                        $script:StudioProcessImageTable[[int]$row.ProcessId] = [string]$row.ExecutablePath
                    }
                }
            } catch {}
        }
        if ($script:StudioProcessImageTable.ContainsKey($ProcessId)) {
            return $script:StudioProcessImageTable[$ProcessId]
        }
        return $null
    }

    function Get-RunningStudioVenvProcesses {
        param(
            [Parameter(Mandatory = $true)][string]$VenvPath,
            [switch]$Exact
        )
        try {
            $resolvedPath = (Get-StudioFinalPath -Path $VenvPath).TrimEnd('\', '/')
        } catch {
            throw "Could not resolve managed Unsloth process path '$VenvPath': $($_.Exception.Message)"
        }
        # No root-relative fallback here: without the native helper EVERY path is
        # inexact, so it compared path tails across unrelated drives and an ordinary
        # D:\env\python.exe matched a protected C:\env, aborting a legitimate install
        # as "still in use". The alias it was written for, SUBST, is folded in
        # Get-StudioLexicalPath instead. A volume reached by GUID still cannot be
        # matched to the same volume by drive letter without the compiler.

        # Block only confirmed executable identities: a command line or working
        # directory that merely mentions the path is not proof of an open file.
        foreach ($process in @(Get-Process -ErrorAction SilentlyContinue)) {
            $executable = $null
            try { $executable = Get-StudioProcessImagePath -ProcessId $process.Id } catch { continue }
            if (-not $executable) { continue }
            try { $executable = Get-StudioFinalPath -Path $executable } catch { continue }
            if (Test-StudioProtectedPathMatch -Candidate $executable -ProtectedPath $resolvedPath -Exact:$Exact) {
                [pscustomobject]@{
                    ProcessName = $process.ProcessName
                    Id = $process.Id
                    Path = $executable
                }
            }
        }
    }
    try {
        $studioInstallMutex = Enter-StudioInstallMutex -Path $StudioHome
    } catch {
        Write-StudioLine "[ERROR] Could not create the Unsloth install lock: $($_.Exception.Message)" -ForegroundColor Red
        return (Exit-InstallFailure "Could not create the Unsloth install lock")
    }
    if ($null -eq $studioInstallMutex) {
        Write-StudioLine "[ERROR] Another Unsloth Studio install or repair is already running." -ForegroundColor Red
        Write-StudioLine "        Wait for it to finish, then re-run install.ps1." -ForegroundColor Yellow
        return (Exit-InstallFailure "Another Unsloth Studio install or repair is already running")
    }

    $studioRuntimeMutexes = @()
    $tauriManagedStudioHome = if ($tauriProfile) {
        Join-Path $tauriProfile ".unsloth\studio"
    } else { $null }
    $studioTauriRootMatch = if ($tauriManagedStudioHome) {
        Test-StudioPathEqual -Left $StudioHome -Right $tauriManagedStudioHome
    } else { $false }
    $studioUsesTauriManagedRoot = ($studioTauriRootMatch -eq $true)
    $studioNeedsRuntimeLock = $true
    $studioUsesLegacyLayout = ($StudioRedirectMode -ne 'env') -or $studioUsesTauriManagedRoot
    $studioAutoStartProcess = $null
    try {
        if ($studioNeedsRuntimeLock) {
            try {
                $studioRuntimeMutexNames = @(
                    Get-StudioRuntimeMutexNames -TauriRootMatch $studioTauriRootMatch -Path $StudioHome
                )
                foreach ($studioRuntimeMutexName in $studioRuntimeMutexNames) {
                    $mutex = Enter-StudioNamedMutex -Name $studioRuntimeMutexName
                    if ($null -eq $mutex) {
                        Write-StudioLine "[ERROR] Unsloth Studio is starting or installation is already running." -ForegroundColor Red
                        Write-StudioLine "        Close Unsloth Studio completely, wait for the other operation, then re-run install.ps1." -ForegroundColor Yellow
                        return (Exit-InstallFailure "The managed Unsloth environment is busy")
                    }
                    $studioRuntimeMutexes += $mutex
                }
            } catch {
                Write-StudioLine "[ERROR] Could not create the Unsloth runtime lock: $($_.Exception.Message)" -ForegroundColor Red
                return (Exit-InstallFailure "Could not create the Unsloth runtime lock")
            }
        }

        $protectedProcessPaths = @(
            [pscustomobject]@{ Path = $VenvDir; Exact = $false }
            [pscustomobject]@{ Path = (Join-Path $StudioHome "bin\unsloth.exe"); Exact = $true }
        )
        if ($studioUsesLegacyLayout) {
            $protectedProcessPaths += [pscustomobject]@{
                Path = (Join-Path $StudioHome ".venv")
                Exact = $false
            }
            $protectedProcessPaths += [pscustomobject]@{
                Path = (Join-Path $env:USERPROFILE "unsloth_studio")
                Exact = $false
            }
        }
        $runningVenvProcessesById = @{}
        foreach ($candidate in $protectedProcessPaths) {
            foreach ($process in @(
                Get-RunningStudioVenvProcesses -VenvPath $candidate.Path -Exact:$candidate.Exact
            )) {
                $processId = [string]$process.Id
                if (-not $runningVenvProcessesById.ContainsKey($processId)) {
                    $runningVenvProcessesById[$processId] = $process
                }
            }
        }
        $runningVenvProcesses = @($runningVenvProcessesById.Values)
        if ($runningVenvProcesses.Count -gt 0) {
            $runningSummary = ($runningVenvProcesses | ForEach-Object { "$($_.ProcessName) (PID $($_.Id))" }) -join ", "
            Write-StudioLine "[ERROR] Unsloth Studio is using the managed Python environment." -ForegroundColor Red
            Write-StudioLine "        Active processes: $runningSummary" -ForegroundColor Yellow
            Write-StudioLine "        Close Unsloth Studio completely, including its tray process, then re-run install.ps1." -ForegroundColor Yellow
            return (Exit-InstallFailure "The managed Python environment is still in use")
        }

        if (-not $TauriMode -and $studioUsesLegacyLayout) {
            $runningDesktopApps = @(Get-StudioDesktopProcessesForCurrentUser)
            if ($runningDesktopApps.Count -gt 0) {
                $desktopSummary = ($runningDesktopApps | ForEach-Object { "PID $($_.Id)" }) -join ", "
                Write-StudioLine "[ERROR] The Unsloth Studio desktop app is still running ($desktopSummary)." -ForegroundColor Red
                Write-StudioLine "        Close the app completely, including its tray process, then re-run install.ps1." -ForegroundColor Yellow
                return (Exit-InstallFailure "The Unsloth Studio desktop app is still running")
            }
        }

    Write-TauriLog "STEP" "Checking system dependencies"
    # -CommandType Application, as for Resolve-UvExecutable below: winget installs Python AND uv,
    # so a profile "function winget {...}" or "Set-Alias winget ..." would otherwise receive both
    # installs. The resolved path is pinned so the five call sites cannot be re-resolved later.
    $script:WingetExe = (
        Get-Command winget -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { $_.Source } | Select-Object -First 1 -ExpandProperty Source
    )
    $script:WingetAvailable = [bool]$script:WingetExe
    if ($script:WingetAvailable) {
        step "winget" "available"
    } else {
        step "winget" "not available -- will require Python + uv to be already installed" "Yellow"
        substep "Get it from https://aka.ms/getwinget if Python / uv are not already on PATH." "Yellow"
    }

    # ── Helper: detect a working Python 3.11-3.13. Skips conda: its modified DLL search paths break torch's c10.dll on Windows; check both exe path AND sys.base_prefix since a conda-derived venv inherits base_prefix even when its path lacks "conda". ──
    $script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'

    function Test-IsCondaPython {
        param([string]$Exe)
        if ($Exe -match $script:CondaSkipPattern) { return $true }
        try {
            $basePrefix = (& $Exe -S -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
            if ($basePrefix -match $script:CondaSkipPattern) { return $true }
        } catch { }
        return $false
    }

    # The interpreter's own arch, asked of it: win-amd64|win-arm64|win32|"".
    # -S: the caller compares this with -eq, so a sitecustomize banner would read as
    # "unknown" and lose the x64-over-ARM64 preference.
    function Get-PythonPlatformTag {
        param([string]$Exe)
        try {
            return (& $Exe -S -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
        } catch { return "" }
    }

    # Returns @{ Version = "3.13"; Path = "C:\...\python.exe" } or $null.
    # The resolved Path is passed to `uv venv --python` to prevent uv from
    # re-resolving the version string back to a conda interpreter.
    # Candidates on $PythonSkip are dropped as they are enumerated, so no caller
    # can be handed an interpreter that cannot import torch and the usual
    # 3.13 -> 3.12 -> 3.11 fallback still applies when the preferred minor is bad.
    function Find-CompatiblePython {
        # -X64Only: best installed x64 interpreter or $null, never ARM64. Last resort for
        # Install-X64Python, where x64 of a lower-priority minor beats ARM64.
        param([switch]$X64Only)
        # Windows on ARM: prefer x64. pyarrow (via datasets) and hf-transfer ship no
        # win_arm64 wheel, so a native ARM64 Python source-builds both and dies on CMake /
        # Rust minutes in; x64 runs fine emulated. ARM64 is still returned when it is all
        # there is, and the caller then bootstraps x64 or warns.
        $preferX64 = $X64Only -or ((Get-HostMachineArch) -eq "arm64")
        $candidates = @()
        # Try the Python Launcher first (most reliable on Windows)
        # py.exe resolves to the standard CPython install, not conda.
        # Prefer the requested $PythonVersion, then newest-first fallback.
        $minors = @($PythonVersion) + (@("3.13", "3.12", "3.11") | Where-Object { $_ -ne $PythonVersion })
        # -All: Windows PowerShell 5.1 returns only the first launcher without it.
        foreach ($pyLauncher in @(Get-Command py -All -CommandType Application -ErrorAction SilentlyContinue)) {
            if ($pyLauncher.Source -match $script:CondaSkipPattern) { continue }
            foreach ($minor in $minors) {
                try {
                    $out = & $pyLauncher.Source "-$minor" --version 2>&1 | Out-String
                    if ($out -match "Python ((3\.1[1-3])\.\d+)") {
                        # Both captures first: Test-IsCondaPython below runs -match
                        # and overwrites $Matches. Screening the patch here costs no
                        # extra subprocess (the text is already in hand) and lets the
                        # loop carry on to the next launcher and the next minor.
                        $full = $Matches[1]
                        $ver = $Matches[2]
                        if ($PythonSkip -contains $full) { continue }
                        # Resolve the actual executable path and verify it is not conda-based
                        $resolvedExe = (& $pyLauncher.Source "-$minor" -S -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
                        if ($resolvedExe -and (Test-Path -LiteralPath $resolvedExe -PathType Leaf) -and -not (Test-IsCondaPython $resolvedExe)) {
                            if (-not $preferX64) { return @{ Version = $ver; Path = $resolvedExe; Arch = "" } }
                            $candidates += @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # python3 / python via -All to look past stubs shadowing a real Python; skip WindowsApps (App Execution Alias stubs can open the Store; real Store Python is caught by the py launcher above) and conda (path + sys.base_prefix).
        foreach ($name in @("python3", "python")) {
            foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
                if (-not $cmd.Source) { continue }
                if ($cmd.Source -like "*\WindowsApps\*") { continue }
                if (Test-IsCondaPython $cmd.Source) { continue }
                try {
                    $out = & $cmd.Source --version 2>&1 | Out-String
                    if ($out -match "Python ((3\.1[1-3])\.\d+)") {
                        $full = $Matches[1]
                        $ver = $Matches[2]
                        if ($PythonSkip -contains $full) { continue }
                        # PATH entries may be wrappers (e.g. pyenv-win's python.bat).
                        # Resolve the real executable so uv bypasses wrapper re-resolution.
                        $resolvedExe = (& $cmd.Source -S -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
                        if ($resolvedExe -and (Test-Path -LiteralPath $resolvedExe -PathType Leaf) -and -not (Test-IsCondaPython $resolvedExe)) {
                            if (-not $preferX64) { return @{ Version = $ver; Path = $resolvedExe; Arch = "" } }
                            $candidates += @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # `py -3.12` runs the launcher's preferred build, normally the native ARM64 one, so
        # a same-minor x64 install that is neither preferred nor on PATH never becomes a
        # candidate. `-3.12-64` cannot disambiguate (deprecated, it only means "not
        # 32-bit"), so enumerate every registration with -0p and probe each path.
        if ($preferX64) {
            foreach ($pyLauncher in @(Get-Command py -All -CommandType Application -ErrorAction SilentlyContinue)) {
                if ($pyLauncher.Source -match $script:CondaSkipPattern) { continue }
                $listed = @()
                try { $listed = @(& $pyLauncher.Source "-0p" 2>$null) } catch {}
                foreach ($line in $listed) {
                    # " -V:3.12 *   C:\...\python.exe": tag, optional default marker, path.
                    $m = [regex]::Match([string]$line, '(?i)^\s*-\S+\s+\*?\s*"?(?<p>\S.*?\.exe)"?\s*$')
                    if (-not $m.Success) { continue }
                    $exe = $m.Groups['p'].Value.Trim()
                    if ($candidates | Where-Object { $_.Path -eq $exe }) { continue }
                    if (-not (Test-Path -LiteralPath $exe)) { continue }
                    if (Test-IsCondaPython $exe) { continue }
                    try {
                        $out = & $exe --version 2>&1 | Out-String
                        if ($out -match "Python ((3\.1[1-3])\.\d+)") {
                            $full = $Matches[1]
                            $ver = $Matches[2]
                            if ($PythonSkip -contains $full) { continue }
                            $candidates += @{ Version = $ver; Path = $exe }
                        }
                    } catch {}
                }
            }
        }
        # Prefer x64, but only within one minor: $minors is the caller's version preference,
        # so ranking on arch alone would answer UNSLOTH_PYTHON=3.12 with an x64 3.13 and
        # never bootstrap x64 3.12. Probing costs a subprocess, so non-ARM returned above.
        foreach ($c in $candidates) {
            $tag = Get-PythonPlatformTag $c.Path
            $c.Arch = if ($tag -eq "win-amd64") { "x86_64" } elseif ($tag -eq "win-arm64") { "arm64" } else { "unknown" }
        }
        foreach ($minor in $minors) {
            $sameMinor = @($candidates | Where-Object { $_.Version -eq $minor })
            if ($sameMinor.Count -eq 0) { continue }
            $x64 = $sameMinor | Where-Object { $_.Arch -eq "x86_64" } | Select-Object -First 1
            if ($x64) { return $x64 }
            if (-not $X64Only) { return $sameMinor[0] }
        }
        if (-not $X64Only -and $candidates.Count -gt 0) { return $candidates[0] }
        return $null
    }

    # ── Fallback: install CPython from python.org when winget is unavailable/fails (notably msstore cert-pinning 0x8a15005e). Silent per-user install (no UAC) puts python.exe + py launcher on PATH. Returns @{ Version; Path } or $null. ──
    function Install-PythonFromPythonOrg {
        # $Arch overrides the host arch, to pull x64 onto an ARM64 box.
        param([string]$Arch = "")
        # python.org ships one installer per architecture.
        $targetArch = if ($Arch) { $Arch } else { Get-TauriDiagArch }
        $archSuffix = switch ($targetArch) {
            "x86_64" { "-amd64" }
            "arm64"  { "-arm64" }
            "x86"    { "" }
            default  { $null }
        }
        if ($null -eq $archSuffix) {
            substep "No python.org installer is available for this architecture." "Yellow"
            return $null
        }

        # Latest $PythonVersion.x patch from the python.org listing, else same-minor fallback. Use the pinned full version only when it matches the requested minor so a non-default UNSLOTH_PYTHON (e.g. 3.12) doesn't silently install 3.13.
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

        # Same trust boundary as the VC++ runtime in studio/setup.ps1: $full moves per patch
        # release, so there is no SHA-256 to pin and the publisher is what we can check.
        # Inspection itself can fail (antivirus quarantining the download first), and the
        # script-wide 'Stop' would let that escape the function, skipping the $null fallback
        # and leaving the executable behind. Unreadable is unverified, so it takes the same
        # route as a bad signature.
        $sig = $null
        try { $sig = Get-AuthenticodeSignature -LiteralPath $dest } catch { $sig = $null }
        if ($null -eq $sig -or
            $sig.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
            $null -eq $sig.SignerCertificate -or
            $sig.SignerCertificate.Subject -notmatch '(^|,\s*)O="?Python Software Foundation"?(,|$)') {
            $sigStatus = if ($null -eq $sig) { "could not be read" } else { $sig.Status }
            substep "python.org installer is not validly signed by the Python Software Foundation (signature status: $sigStatus); not running it." "Yellow"
            Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
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
            # Launcher per-user too: Include_launcher defaults InstallLauncherAllUsers=1 (needs admin, breaks this non-admin fallback).
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

    # Backstop for the screen inside Find-CompatiblePython, for an interpreter that
    # reports one version to --version and another to sys.version_info (a wrapper or
    # a shim). Returning $null sends the caller down the install path, which pins
    # $PythonFallbackFullVersion.
    function Remove-SkippedPython {
        param($Candidate)
        if (-not $Candidate) { return $null }
        try {
            $raw = (& $Candidate.Path -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null | Select-Object -First 1)
        } catch {
            return $Candidate  # unreadable: not evidence of a bad version
        }
        if ($raw -and ($PythonSkip -contains $raw.Trim())) {
            substep "Python $($raw.Trim()) cannot import torch -- installing another." "Yellow"
            return $null
        }
        return $Candidate
    }

    # ── Windows on ARM: get an x64 CPython ──
    # --architecture x64 forces winget off the ARM64 build; python.org takes the same override.
    function Install-X64Python {
        if ($script:WingetAvailable) {
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                & $script:WingetExe install -e --id "Python.Python.$PythonVersion" --source winget --architecture x64 --accept-package-agreements --accept-source-agreements
            } catch { }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath
            $found = Find-CompatiblePython
            if ($found -and $found.Arch -eq "x86_64") { return $found }
            substep "winget could not provide an x64 Python -- trying python.org..." "Yellow"
        }
        $found = Install-PythonFromPythonOrg -Arch "x86_64"
        if ($found -and $found.Arch -eq "x86_64") { return $found }
        # Nothing installable (offline / no winget): an x64 build of another supported minor
        # still runs the wheels ARM64 cannot, so take it over the native interpreter.
        return (Find-CompatiblePython -X64Only)
    }

    # ── Install Python if no compatible version (3.11-3.13) found ──
    Write-TauriLog "STEP" "Installing Python"
    $DetectedPython = Remove-SkippedPython (Find-CompatiblePython)

    if ($DetectedPython) {
        step "python" "Python $($DetectedPython.Version) already installed"
    }
    if (-not $DetectedPython) {
        substep "installing Python ${PythonVersion}..."
        $pythonPackageId = "Python.Python.$PythonVersion"
        $wingetExit = $null

        if ($script:WingetAvailable) {
            # --source winget avoids the msstore source (cert-pinning 0x8a15005e can abort the whole install); Python and uv both live in the winget source. Lower ErrorActionPreference so winget stderr isn't a terminating error on PS 5.1.
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                & $script:WingetExe install -e --id $pythonPackageId --source winget --accept-package-agreements --accept-source-agreements
                $wingetExit = $LASTEXITCODE
            } catch { $wingetExit = 1 }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath

            # Re-detect after install (PATH may have changed)
            $DetectedPython = Remove-SkippedPython (Find-CompatiblePython)

            if (-not $DetectedPython) {
                # Still not functional after winget -- force reinstall. Handles real failures AND "already installed" codes where winget thinks Python is present but it's not on PATH.
                substep "Python not found on PATH after winget. Retrying with --force..." "Yellow"
                $ErrorActionPreference = "Continue"
                try {
                    & $script:WingetExe install -e --id $pythonPackageId --source winget --accept-package-agreements --accept-source-agreements --force
                    $wingetExit = $LASTEXITCODE
                } catch { $wingetExit = 1 }
                $ErrorActionPreference = $prevEAP
                Refresh-SessionPath
                $DetectedPython = Remove-SkippedPython (Find-CompatiblePython)
            }
        }

        # Fall back to python.org if winget is unavailable or couldn't install a working Python (msstore cert errors --source winget can't fix), keeping the install automatic.
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
            Write-StudioLine "[ERROR] Python installation failed$exitNote" -ForegroundColor Red
            Write-StudioLine "        Please install Python $PythonVersion manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-StudioLine "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-StudioLine "        Then re-run this installer." -ForegroundColor Yellow
            return (Exit-InstallFailure "Python installation failed")
        }
    }
    # ── Windows on ARM: swap a native ARM64 interpreter for x64 ──
    # pyarrow and hf-transfer publish no win_arm64 wheel, so an ARM64 Python source-builds
    # both and fails deep into the run. Warn up front if x64 is unobtainable.
    if ($DetectedPython -and (Get-HostMachineArch) -eq "arm64" -and $DetectedPython.Arch -ne "x86_64") {
        substep "windows on arm: only a native ARM64 Python $($DetectedPython.Version) was found." "Yellow"
        substep "pyarrow and hf-transfer publish no win_arm64 wheels, so installing x64 Python..." "Yellow"
        $X64Python = Install-X64Python
        if ($X64Python) {
            $DetectedPython = $X64Python
            step "python" "using x64 Python $($DetectedPython.Version) under emulation"
        } else {
            Write-StudioLine "[WARN] Could not install an x64 Python on this ARM64 machine." -ForegroundColor Yellow
            Write-StudioLine "       Continuing with ARM64 Python $($DetectedPython.Version), but the install is likely to fail:" -ForegroundColor Yellow
            Write-StudioLine "       pyarrow (via datasets) and hf-transfer ship no win_arm64 wheels and will be" -ForegroundColor Yellow
            Write-StudioLine "       built from source, which needs CMake plus the MSVC and Rust toolchains." -ForegroundColor Yellow
            Write-StudioLine "       Fix: install x64 Python from https://www.python.org/downloads/windows/" -ForegroundColor Yellow
            Write-StudioLine "       (choose 'Windows installer (64-bit)', not ARM64), then re-run this installer." -ForegroundColor Yellow
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

    # PowerShell ranks aliases and functions above anything on PATH, so a profile carrying
    # "Set-Alias uv ..." or "function uv {...}" captures every bare uv call here and the version
    # probe reads a good uv as broken. $null when nothing named uv resolves to an executable, so
    # the caller's "not installed yet" branch keeps its meaning.
    function Resolve-UvExecutable {
        # @(): a one-element return unrolls to a bare string, and indexing THAT gives its
        # first character.
        $candidates = @(Get-UvExecutableCandidates)
        if ($candidates.Count -gt 0) { return $candidates[0] }
        return $null
    }

    # Every uv this machine offers, in the order the bare token would pick them: alias first,
    # then PATH. A LIST rather than one answer, so the version gate can move past an alias
    # pointing at a stale uv and still find a current one.
    function Get-UvExecutableCandidates {
        # An ALIAS pointing at a real executable FIRST, because PowerShell resolves aliases
        # ahead of PATH. Followed only as far as an Application, returning the resolved path so
        # the rest of the script is not back in command discovery.
        $found = [System.Collections.Generic.List[string]]::new()
        $alias = Get-Command uv -CommandType Alias -ErrorAction SilentlyContinue
        while ($alias -and $alias.ResolvedCommand) {
            $target = $alias.ResolvedCommand
            if ($target.CommandType -eq 'Application' -and $target.Source) {
                $found.Add($target.Source)
                break
            }
            if ($target.CommandType -ne 'Alias') { break }
            $alias = $target
        }
        # -All, because Get-Command's choice among several matches is otherwise incidental.
        # Applications come back in PATH order, so with no profile overrides this is the uv the
        # bare token would run.
        $apps = @(
            Get-Command uv -CommandType Application -All -ErrorAction SilentlyContinue |
                Where-Object { $_.Source }
        )
        foreach ($app in $apps) {
            if (-not $found.Contains($app.Source)) { $found.Add($app.Source) }
        }
        if ($found.Count -gt 0) { return @($found) }
        # Anything else named uv is a function, a cmdlet, or an alias to one, and handing back
        # the bare token would let a wrapper answering `--version` pass the gate and then
        # receive every install command. An empty list means "not installed yet", so uv gets
        # installed and the gate re-probes.
        return @()
    }

    function Test-UvVersionOk {
        # EVERY candidate, not just the first, so an alias pointing at a stale uv cannot hide a
        # current uv on PATH. Alias first still, so a passing alias keeps winning.
        foreach ($exe in Get-UvExecutableCandidates) {
            if (Test-UvCandidateVersion $exe) { return $true }
        }
        return $false
    }

    function Test-UvCandidateVersion {
        param([string]$exe)
        if (-not $exe) { return $false }
        try {
            $raw = (& $exe --version 2>$null | Select-Object -First 1)
        } catch {
            return $false
        }
        if ($raw -notmatch 'uv\s+([0-9]+(?:\.[0-9]+)+)') { return $false }
        try {
            if ([version]$Matches[1] -ge [version]$UvMinVersion) {
                # Pin the executable that actually answered: every install command below runs
                # this path, so a later Refresh-SessionPath or a profile alias cannot swap it.
                $script:UvExe = $exe
                return $true
            }
            return $false
        } catch {
            return $false
        }
    }

    function Get-UvExecutableVerdict {
        # "ok", "failed" or "unknown". Only the binary itself answering non-zero is "failed".
        # A launch that throws or a wait that times out is "unknown", because the probe got no
        # verdict: Start-Process -NoNewWindow with redirected streams does not behave in a
        # Windows container or on the arm64 image the way it does in a desktop session, and
        # treating that as a broken binary turned three clean-machine CI legs into hard install
        # failures. The digest already proved these bytes are astral's pinned release, so no
        # verdict publishes, as the pre-pin code did. Every path says why.
        param([string]$Path)
        if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return "failed" }
        # Redirected: uv's version line is not part of this installer's output.
        $outFile = [System.IO.Path]::GetTempFileName()
        $errFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath $Path -ArgumentList "--version" -NoNewWindow -PassThru `
                -RedirectStandardOutput $outFile -RedirectStandardError $errFile -ErrorAction Stop
            if (-not $proc.WaitForExit(20000)) {
                try { $proc.Kill() } catch {}
                substep "uv did not answer --version within 20s; installing it unprobed." "Yellow"
                return "unknown"
            }
            # The timed overload can return before the exit code is cached, which is how
            # arm64 and the Windows containers reported an EMPTY code and had a working uv
            # read as broken. The parameterless wait settles it and returns at once, since
            # the process has already exited. No code at all is still no verdict.
            try { $proc.WaitForExit() } catch {}
            $code = $null
            try { $code = $proc.ExitCode } catch {}
            if ($null -eq $code -or "$code" -eq "") {
                substep "uv --version gave no exit code; installing it unprobed." "Yellow"
                return "unknown"
            }
            if ($code -eq 0) { return "ok" }
            $detail = ""
            try {
                $detail = Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue
            } catch {}
            if ($detail) { $detail = " " + (($detail.Trim()) -replace '\s+', ' ') }
            substep "uv --version exited $code.$detail" "Yellow"
            return "failed"
        } catch {
            substep "could not probe uv: $($_.Exception.Message); installing it unprobed." "Yellow"
            return "unknown"
        } finally {
            Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $errFile -Force -ErrorAction SilentlyContinue
        }
    }

    # Fallback for hosts without winget. Same archive, destination and user-PATH
    # prepend as astral's install.ps1, but it fetches a data file with a pinned
    # SHA-256 instead of script text run in-process, which is what AMSI and cloud
    # ML scanners score hardest. Bumping the version means bumping all 3 hashes:
    #   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/uv-<arch>-pc-windows-msvc.zip.sha256
    $UvPinnedVersion = "0.12.1"
    $UvPinnedAssets = @{
        "x86_64" = @{ Asset = "uv-x86_64-pc-windows-msvc.zip";  Sha256 = "8FCB0CB46E1229065E344758980924E569BEF5882EF45F46FADA8FB24E06B74A" }
        "arm64"  = @{ Asset = "uv-aarch64-pc-windows-msvc.zip"; Sha256 = "9BC7C18E616230FA2DC6FB24BC3AFDE18A95C2B5C9433DE747E9502C66041568" }
        "x86"    = @{ Asset = "uv-i686-pc-windows-msvc.zip";    Sha256 = "9B51C33D307A8AB9E9DFD88D4AE1491761F63DE0BFFA3CEC96BEC536491C9B97" }
    }

    function Install-UvFromRelease {
        $arch = Get-HostMachineArch
        if (-not $UvPinnedAssets.ContainsKey($arch)) {
            substep "No uv build is published for this architecture ($arch)." "Yellow"
            return $false
        }
        $asset  = $UvPinnedAssets[$arch].Asset
        $wanted = $UvPinnedAssets[$arch].Sha256

        # Same destination priority as astral's installer, so an existing uv is
        # replaced in place and the PATH probe further below still finds it.
        $destDir = $null
        foreach ($candidate in @($env:UV_INSTALL_DIR, $env:UV_UNMANAGED_INSTALL, $env:XDG_BIN_HOME)) {
            if ($candidate) { $destDir = $candidate; break }
        }
        if (-not $destDir -and $env:XDG_DATA_HOME) { $destDir = Join-Path $env:XDG_DATA_HOME "../bin" }
        if (-not $destDir) {
            $userHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
            if (-not $userHome) {
                substep "Could not determine a home directory to install uv into." "Yellow"
                return $false
            }
            $destDir = Join-Path $userHome ".local\bin"
        }

        # astral's sources in astral's order, each exclusive when set: a host that sets one
        # usually cannot reach the public endpoints at all, so trying those first would stall.
        # The pin still applies, so a source serving a different build fails the digest.
        $uvBase = if ($env:UV_DOWNLOAD_URL) {
            @("$($env:UV_DOWNLOAD_URL.TrimEnd('/'))")
        } elseif ($env:INSTALLER_DOWNLOAD_URL) {
            @("$($env:INSTALLER_DOWNLOAD_URL.TrimEnd('/'))")
        } elseif ($env:UV_INSTALLER_GHE_BASE_URL) {
            @("$($env:UV_INSTALLER_GHE_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
        } elseif ($env:UV_INSTALLER_GITHUB_BASE_URL) {
            @("$($env:UV_INSTALLER_GITHUB_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
        } else {
            @("https://releases.astral.sh/github/uv/releases/download/$UvPinnedVersion",
              "https://github.com/astral-sh/uv/releases/download/$UvPinnedVersion")
        }

        $work = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-uv-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
        $zip  = Join-Path $work $asset
        try {
            [System.IO.Directory]::CreateDirectory($work) | Out-Null
            # Digest per mirror, not once after the loop: a proxy answering 200 with its own
            # body is a successful download by every measure Invoke-WebRequest has, and checking
            # afterwards spends the only attempt on it.
            $downloaded = $false
            foreach ($base in $uvBase) {
                substep "downloading uv $UvPinnedVersion ($arch) from $base..." "Yellow"
                try {
                    Invoke-WebRequest -UseBasicParsing -OutFile $zip -Uri "$base/$asset"
                } catch {
                    substep "uv download failed: $($_.Exception.Message)" "Yellow"
                    continue
                }
                $actual = ""
                try { $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash } catch {}
                if ($actual -eq $wanted) {
                    $downloaded = $true
                    break
                }
                substep "uv download failed checksum verification -- discarding it." "Red"
                substep "expected $wanted, got $actual" "Red"
                Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
            }
            if (-not $downloaded) { return $false }

            # The Windows archives are flat: uv.exe, uvx.exe, uvw.exe at the root.
            Expand-Archive -LiteralPath $zip -DestinationPath $work -Force
            [System.IO.Directory]::CreateDirectory($destDir) | Out-Null

            $stagedUv = Join-Path $work "uv.exe"
            if (-not (Test-Path -LiteralPath $stagedUv)) {
                substep "uv.exe was not present in $asset." "Yellow"
                return $false
            }
            # Run it where it landed, before the destination is touched. A host can have a
            # working older uv while AppLocker, WDAC or endpoint protection refuses this one, and
            # copying first would leave the user with neither. A policy scoped to the destination
            # path is not covered here: the caller's fallback handles it.
            if ((Get-UvExecutableVerdict -Path $stagedUv) -eq "failed") {
                substep "the downloaded uv $UvPinnedVersion could not run on this machine." "Yellow"
                return $false
            }

            # uvw.exe is the windowless launcher and has no console to answer a probe on, so
            # the staged uv.exe above stands for the set: it came from the same verified
            # archive. Copy-Item under Stop so a locked or ACL-denied destination fails the
            # install rather than leaving half a set behind quietly.
            $ok = $true
            foreach ($exe in @("uv.exe", "uvx.exe", "uvw.exe")) {
                $src = Join-Path $work $exe
                if (-not (Test-Path -LiteralPath $src)) { continue }
                $dst = Join-Path $destDir $exe
                try {
                    Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
                } catch {
                    $ok = $false
                    break
                }
                if ($exe -eq "uv.exe") {
                    # Copy-Item is non-terminating under some callers preference, so compare
                    # against the archive we verified: a stale uv.exe must not pass for ours.
                    $copied = $false
                    try {
                        $copied = (Test-Path -LiteralPath $dst) -and
                            (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash -eq
                            (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
                    } catch { $copied = $false }
                    if (-not $copied) { $ok = $false; break }
                }
            }
            if (-not $ok) {
                substep "the downloaded uv $UvPinnedVersion could not run on this machine." "Yellow"
                return $false
            }
        } finally {
            Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
        }

        # Same PATH treatment and opt-outs astral's installer uses: an unmanaged
        # install forces no-modify-path there, so it must here too.
        if (-not $env:UV_NO_MODIFY_PATH -and -not $env:UV_UNMANAGED_INSTALL) {
            Add-ToUserPath -Directory $destDir -Position Prepend | Out-Null
        }
        $env:PATH = "$destDir;$env:PATH"
        # Refresh-SessionPath rebuilds PATH machine-first and drops that prepend,
        # so record where uv actually landed for the probe below.
        $script:UvInstallDestDir = $destDir
        return $true
    }

    if (-not (Test-UvVersionOk)) {
        # Resolve-UvExecutable, not a bare Get-Command: a profile alias named uv would
        # otherwise report "updating" on what is in fact a first install.
        if (Resolve-UvExecutable) {
            substep "updating uv package manager..."
        } else {
            substep "installing uv package manager..."
        }
        if ($script:WingetAvailable) {
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try { & $script:WingetExe upgrade --id=astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements } catch {}
            if (-not (Test-UvVersionOk)) {
                try { & $script:WingetExe install --id=astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements } catch {}
            }
            $ErrorActionPreference = $prevEAP
            Refresh-SessionPath
        }
        # winget unavailable or it didn't put uv on PATH: install the pinned
        # release directly (ARM64 runners, machines without the Store).
        if (-not (Test-UvVersionOk)) {
            Install-UvFromRelease | Out-Null
            Refresh-SessionPath
        }
    }

    # A freshly installed uv can sit later on PATH than an older one; prefer a just-installed uv from a known location.
    if (-not (Test-UvVersionOk)) {
        $origPath = $env:PATH
        foreach ($d in @($script:UvInstallDestDir, $env:UV_INSTALL_DIR, $env:XDG_BIN_HOME,
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

    # Bytecode compilation can exceed uv's 60s default on slow machines; default 180s, preserving overrides ("0" disables).
    if (-not $env:UV_COMPILE_BYTECODE_TIMEOUT) {
        $env:UV_COMPILE_BYTECODE_TIMEOUT = "180"
    }

    # Raise uv HTTP retries + read timeout for large wheel downloads (preserves user values).
    if (-not $env:UV_HTTP_RETRIES) {
        $env:UV_HTTP_RETRIES = "5"
    }
    if (-not $env:UV_HTTP_TIMEOUT) {
        $env:UV_HTTP_TIMEOUT = "180"
    }

    # ── Create venv (migrate old layout if possible, otherwise fresh); pass the resolved exe path to uv so it doesn't re-resolve back to conda. ──
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
    $script:StudioVenvRollbackPartial = $false
    # Release-preservation state, reset per run: with `irm | iex` the script scope IS the
    # caller's session, so a second invocation (e.g. a different UNSLOTH_STUDIO_HOME with no
    # existing venv) must not inherit the previous run's release or pin.
    $script:PrevTorchVer = ""
    $script:PrevTorchPin = $null

    function Test-VenvPythonReady {
        param([Parameter(Mandatory = $true)][string]$PythonExe)
        if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { return $false }

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $global:LASTEXITCODE = -1
            $null = & $PythonExe -c "import sys; sys.exit(0)" 2>$null
            return ($LASTEXITCODE -eq 0)
        } catch {
            return $false
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
    }

    # uv creates only into a path that is absent or an empty directory. The .NET
    # API counts hidden entries and reads wildcards in the path literally.
    function Test-DirectoryHasEntries {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
            # Still an existing path to CreateDirectory, which answers
            # ERROR_ALREADY_EXISTS for a file or a link whose target is gone, so
            # uv refuses it too. -PathType Container follows the link and cannot
            # see a dangling one; Get-Item sees the link itself.
            return ($null -ne (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue))
        }
        try {
            foreach ($entry in [System.IO.Directory]::EnumerateFileSystemEntries($Path)) {
                if ($entry) { return $true }
            }
        } catch {
            # Present but unreadable: report it occupied rather than let uv fail on it.
            return $true
        }
        return $false
    }

    # Move-Item into an existing directory nests the source inside it rather than
    # renaming it, and uv then refuses that target as in #9479. A migration branch
    # already means $VenvDir is absent or empty, so clear it: Directory.Delete is
    # non-recursive, and on a reparse point it unlinks without following.
    function Clear-MigrationTargetDirectory {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not (Test-Path -LiteralPath $Path)) { return }
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            # Not Remove-Item: on Windows PowerShell 5.1 it trips a reparse-tag
            # mismatch on a directory symlink (PowerShell/PowerShell#621).
            # Directory.Delete throws on a link to a file or a dangling one,
            # hence the File.Delete fallback.
            try { [System.IO.Directory]::Delete($Path) }
            catch {
                try { [System.IO.File]::Delete($Path) }
                catch { throw "$Path is in the way of the environment migration. Move it aside and re-run." }
            }
            return
        }
        if (-not $item.PSIsContainer) {
            throw "$Path is a file and is in the way of the environment migration. Move it aside and re-run."
        }
        try {
            [System.IO.Directory]::Delete($Path)
        } catch {
            throw "$Path is in the way of the environment migration. Move it aside and re-run."
        }
    }

    function Get-VenvBaseHome {
        param([Parameter(Mandatory = $true)][string]$VenvRoot)
        $configPath = Join-Path $VenvRoot "pyvenv.cfg"
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { return $null }

        try {
            foreach ($line in [System.IO.File]::ReadAllLines($configPath)) {
                if ($line -match '^\s*home\s*=\s*(.*?)\s*$') {
                    return $Matches[1].Trim()
                }
            }
        } catch {}
        return $null
    }

    # Test-Path follows a link, so a dangling one reads as absent. A rollback holds
    # whatever Test-DirectoryHasEntries called occupied, so ask the path itself.
    function Test-StudioPathPresent {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)
        if (-not $Path) { return $false }
        return ($null -ne (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue))
    }

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
        $script:StudioVenvRollbackDir = $candidate
        $script:StudioVenvRollbackTarget = $ExistingDir
        $script:StudioVenvRollbackActive = $true
        $script:StudioVenvRollbackPartial = $false
        # Publish the rollback state before the atomic rename so interruption
        # cannot land after Move-Item but before cleanup knows where the old venv went.
        try {
            Move-Item -LiteralPath $ExistingDir -Destination $candidate -ErrorAction Stop
        } catch {
            # A collision or ordinary rename failure leaves the original in place.
            # On Windows an open handle inside the tree fails the rename *partway*
            # instead: entries walked before the locked one land at $candidate while
            # the rest stay at $ExistingDir. Reading only $ExistingDir scores that
            # split tree as "the rename never happened" and drops the sole reference
            # to where the other half went, so the caller can neither restore nor
            # report it. Clear the state only when the destination is genuinely
            # absent; when both paths exist keep it active so Restore-StudioVenvRollback
            # can reverse the partial move, and name both locations either way.
            $candidateExists = Test-Path -LiteralPath $candidate
            if ((Test-Path -LiteralPath $ExistingDir) -and (-not $candidateExists)) {
                $script:StudioVenvRollbackActive = $false
                $script:StudioVenvRollbackDir = $null
            } elseif ($candidateExists -and (Test-Path -LiteralPath $ExistingDir)) {
                # Flag the split: both paths now hold halves of the *same* previous
                # environment, so restoration must merge them rather than clear the
                # destination first the way the committed-replacement path does.
                $script:StudioVenvRollbackPartial = $true
                Write-StudioLine "[WARN] Moving the existing environment aside stopped partway -- files are in both places." -ForegroundColor Yellow
                Write-StudioLine "       still in place: $ExistingDir" -ForegroundColor Yellow
                Write-StudioLine "       moved aside:    $candidate" -ForegroundColor Yellow
                Write-StudioLine "       A running 'unsloth studio' process usually holds a file open here." -ForegroundColor Yellow
                Write-StudioLine "       Close Unsloth Studio and re-run the installer to reverse the move." -ForegroundColor Yellow
            }
            throw
        }
        substep "previous environment preserved for rollback"
    }

    function Remove-StudioVenvTreeWithRetry {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Label
        )
        $lastError = $null
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            } catch {
                $lastError = $_.Exception.Message
            }
            if (-not (Test-Path -LiteralPath $Path)) { return $true }
            if ($attempt -lt 3) { Start-Sleep -Milliseconds (250 * $attempt) }
        }
        Write-StudioLine "[WARN] Could not remove $Label at $Path" -ForegroundColor Yellow
        if ($lastError) { Write-StudioLine "       $lastError" -ForegroundColor Yellow }
        return $false
    }

    function Test-StudioVenvRollbackMustBePreserved {
        param([Parameter(Mandatory = $true)][System.IO.FileSystemInfo]$Rollback)
        # Preserve anything outside the installer's timestamp.PID[.suffix] format.
        if ($Rollback.Name -notmatch '^unsloth_studio\.rollback\.[0-9]{14}\.([0-9]+)(?:\.[0-9]+)?$') {
            return $true
        }
        $ownerPid = 0
        if (-not [int]::TryParse($Matches[1], [ref]$ownerPid)) { return $true }
        if ($ownerPid -eq $PID) { return $true }
        return $null -ne (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)
    }

    function Remove-StaleStudioVenvRollbacks {
        try {
            $rollbacks = @(
                Get-ChildItem -LiteralPath $StudioHome -Directory -Force -ErrorAction Stop |
                    Where-Object { $_.Name -like 'unsloth_studio.rollback.*' }
            )
        } catch {
            Write-StudioLine "[WARN] Could not inspect stale environment rollbacks in $StudioHome" -ForegroundColor Yellow
            Write-StudioLine "       $($_.Exception.Message)" -ForegroundColor Yellow
            return
        }
        foreach ($rollback in $rollbacks) {
            if (($rollback.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                Write-StudioLine "[WARN] Refusing to remove rollback reparse point $($rollback.FullName)" -ForegroundColor Yellow
                continue
            }
            # A concurrent installer may have moved its live venv aside. The PID
            # in the generated name keeps this run from deleting its rescue copy.
            if (Test-StudioVenvRollbackMustBePreserved -Rollback $rollback) { continue }
            if (Remove-StudioVenvTreeWithRetry -Path $rollback.FullName -Label "stale environment rollback") {
                substep "removed stale environment rollback $($rollback.Name)"
            }
        }
    }

    function Merge-StudioVenvRollbackTree {
        # Moves every entry of $Source into $Destination without ever overwriting or
        # deleting what is already there. Returns $true only when $Source ends up
        # empty and removed, i.e. the two halves were fully reunited.
        param(
            [Parameter(Mandatory = $true)][string]$Source,
            [Parameter(Mandatory = $true)][string]$Destination
        )
        if (-not (Test-Path -LiteralPath $Destination)) {
            [System.IO.Directory]::CreateDirectory($Destination) | Out-Null
        }
        $complete = $true
        foreach ($entry in @(Get-ChildItem -LiteralPath $Source -Force -ErrorAction Stop)) {
            # Not $destination: PowerShell names are case-insensitive, so that would
            # reassign the $Destination parameter and nest every later sibling under
            # the previous entry's name.
            $entryTarget = Join-Path $Destination $entry.Name
            if (-not (Test-Path -LiteralPath $entryTarget)) {
                Move-Item -LiteralPath $entry.FullName -Destination $entryTarget -ErrorAction Stop
                continue
            }
            # Same relative path on both sides: the move stopped inside this subtree.
            # Recurse so the halves reunite -- Move-Item -Force would overwrite the
            # half that never moved, which is exactly the data this path protects.
            # A junction on either side is a leaf, not a subtree: recursing through one
            # moves files to wherever it points, outside $StudioHome. Keep both instead.
            # Get-Item both sides: Get-ChildItem has reported Attributes inconsistently.
            $entryItem = Get-Item -LiteralPath $entry.FullName -Force -ErrorAction Stop
            $targetItem = Get-Item -LiteralPath $entryTarget -Force -ErrorAction Stop
            $linked = (($entryItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) -or
                      (($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
            if ($entryItem.PSIsContainer -and $targetItem.PSIsContainer -and -not $linked) {
                if (-not (Merge-StudioVenvRollbackTree -Source $entry.FullName -Destination $entryTarget)) {
                    $complete = $false
                }
                continue
            }
            Write-StudioLine "[WARN] Kept both copies of $($entry.Name)" -ForegroundColor Yellow
            Write-StudioLine "       $($entry.FullName)" -ForegroundColor Yellow
            Write-StudioLine "       $entryTarget" -ForegroundColor Yellow
            $complete = $false
        }
        if ($complete) {
            Remove-Item -LiteralPath $Source -Force -ErrorAction SilentlyContinue
            return (-not (Test-Path -LiteralPath $Source))
        }
        return $false
    }

    function Restore-StudioVenvRollback {
        if (-not $script:StudioVenvRollbackActive) { return }
        $backup = $script:StudioVenvRollbackDir
        $target = $script:StudioVenvRollbackTarget
        if (-not (Test-StudioPathPresent -Path $backup)) {
            $script:StudioVenvRollbackActive = $false
            $script:StudioVenvRollbackPartial = $false
            return
        }
        substep "restoring previous environment after failed install..." "Yellow"
        if ($script:StudioVenvRollbackPartial) {
            # $target is not an incomplete *new* environment here -- it is the half of
            # the previous one the interrupted move left behind. Removing it first,
            # the way the branch below does, would delete files that exist nowhere
            # else and "restore" a corrupted venv. Merge the halves instead.
            $merged = $false
            try {
                $merged = Merge-StudioVenvRollbackTree -Source $backup -Destination $target
            } catch {
                Write-StudioLine "[WARN] Could not merge $backup back into $target" -ForegroundColor Yellow
                Write-StudioLine "       $($_.Exception.Message)" -ForegroundColor Yellow
            }
            if ($merged) {
                substep "restored previous environment"
                $script:StudioVenvRollbackActive = $false
                $script:StudioVenvRollbackDir = $null
                $script:StudioVenvRollbackPartial = $false
            } else {
                Write-StudioLine "[WARN] The previous environment is still split in two." -ForegroundColor Yellow
                Write-StudioLine "       still in place: $target" -ForegroundColor Yellow
                Write-StudioLine "       moved aside:    $backup" -ForegroundColor Yellow
                Write-StudioLine "       Close Unsloth Studio and re-run the installer to finish reversing the move." -ForegroundColor Yellow
            }
            return
        }
        try {
            if (Test-Path -LiteralPath $target) {
                if (-not (Remove-StudioVenvTreeWithRetry -Path $target -Label "incomplete environment")) {
                    throw "Could not remove incomplete environment at $target"
                }
            }
            Move-Item -LiteralPath $backup -Destination $target -Force -ErrorAction Stop
            substep "restored previous environment"
            $script:StudioVenvRollbackActive = $false
            $script:StudioVenvRollbackDir = $null
        } catch {
            Write-StudioLine "[WARN] Could not restore previous environment from $backup to $target" -ForegroundColor Yellow
            Write-StudioLine "       $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    function Complete-StudioVenvRollback {
        if (-not $script:StudioVenvRollbackActive) { return }
        $backup = $script:StudioVenvRollbackDir
        # The replacement is committed. Disable restoration before deleting the
        # backup so interruption cannot restore a partially deleted environment.
        $script:StudioVenvRollbackActive = $false
        $script:StudioVenvRollbackDir = $null
        $script:StudioVenvRollbackPartial = $false
        if (Test-StudioPathPresent -Path $backup) {
            Remove-StudioVenvTreeWithRetry -Path $backup -Label "environment rollback" | Out-Null
        }
    }

    # Raw torch.__version__ from $PythonExe's venv (last non-empty stdout line), or $null.
    # Bounded ProcessStartInfo probe (async-drain both streams, 30s timeout, kill on hang) so a
    # wedged "import torch" can't stall the installer; feeds Get-InstalledTorchTag and the torch
    # release-preservation decision (twin of install.sh's _PREV_TORCH_VER probe / _previous_torch_pin).
    function Get-InstalledTorchVersionRaw {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $PythonExe
            # Dist metadata, not "import torch": a broken CUDA/ROCm DLL or a slow native
            # import would yield no release and silently drop the pin (parity with install.sh).
            $psi.Arguments = '-c "import importlib.metadata as m; print(m.version(''torch''))"'
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            # Drain BOTH streams async before WaitForExit: a synchronous ReadToEnd() would block on a wedged "import torch", and an undrained stderr would deadlock a child flooding the pipe buffer. A truly hung probe still hits the 30s timeout.
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            $finished = $proc.WaitForExit(30000)
            if (-not $finished) { try { $proc.Kill() } catch {}; return $null }
            $out = $outTask.GetAwaiter().GetResult()
            [void]$errTask.GetAwaiter().GetResult()
            if ($proc.ExitCode -ne 0) { return $null }
            # Last non-empty line only, so stdout noise before the version can't corrupt the pin.
            $lines = @($out -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
            if ($lines.Count -eq 0) { return $null }
            return $lines[-1]
        } catch { return $null }
    }

    $studioVenvReplacementCommitted = $false
    try {
    # Replace occupied venvs even when python.exe is missing, as in #9479.
    if ((Test-Path -LiteralPath $VenvPython) -or (Test-DirectoryHasEntries -Path $VenvDir)) {
        # why: matching guard to the .venv branch below -- in env-mode
        # $StudioHome is a user-chosen workspace, so refuse to nuke an
        # existing $StudioHome\unsloth_studio that lacks Unsloth sentinels.
        # -PathType Leaf rejects a directory at the sentinel path. Accept the
        # in-VENV ownership marker so partial-install retries are not blocked.
        # The .cmd counts too, and for the same reason the uninstaller accepts it: a
        # policy's quarantine can take the unsigned .exe and leave a root that is still
        # ours. Content-checked, never by name -- this guard gates a recursive delete.
        if (
            $StudioRedirectMode -eq 'env' -and
            -not (Test-Path -LiteralPath (Join-Path $VenvDir ".unsloth-studio-owned") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "share\studio.conf") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe") -PathType Leaf) -and
            -not (Test-UnslothCmdShimFile (Join-Path $StudioHome "bin\unsloth.cmd"))
        ) {
            Write-StudioLine "[ERROR] $VenvDir already exists but does not look like an Unsloth Studio install." -ForegroundColor Red
            Write-StudioLine "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME." -ForegroundColor Yellow
            throw "Refusing to delete non-Unsloth venv at $VenvDir"
        }
        # Record the existing venv's torch RELEASE BEFORE the rollback move (see Get-PreviousTorchPin);
        # a re-run then keeps that release rather than silently jumping torch versions. Opt out with
        # UNSLOTH_TORCH_UPGRADE=1. Only the new-layout replace probes; the legacy-migration branches
        # reuse the venv, so torch survives naturally and needs no pin.
        if (-not $SkipTorch) {
            $script:PrevTorchVer = Get-InstalledTorchVersionRaw -PythonExe $VenvPython
        }
        # New layout already exists -- replace only after preserving rollback copy.
        substep "preserving existing environment for rollback..."
        try {
            Start-StudioVenvRollback -ExistingDir $VenvDir
        } catch {
            Write-StudioLine "[ERROR] Could not prepare existing environment for reinstall: $($_.Exception.Message)" -ForegroundColor Red
            return (Exit-InstallFailure "Could not prepare existing environment for reinstall")
        }
    } elseif (
        $studioUsesLegacyLayout `
        -and (Test-Path -LiteralPath (Join-Path $StudioHome ".venv\Scripts\python.exe"))
    ) {
        # Old layout (~/.unsloth/studio/.venv) exists -- validate before migrating.
        # Skip custom-root env-mode installs so we do not replace an unrelated
        # project .venv; an override of the managed default root still migrates.
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
            try {
                Clear-MigrationTargetDirectory -Path $VenvDir
            } catch {
                Write-StudioLine "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
                return (Exit-InstallFailure "Could not clear $VenvDir for the environment migration")
            }
            Move-Item -LiteralPath $OldVenv -Destination $VenvDir -Force
            substep "moved .venv -> unsloth_studio"
            $_Migrated = $true
        } else {
            substep "legacy environment failed validation -- creating fresh environment" "Yellow"
            $invalidVenv = Join-Path $StudioHome (".venv.invalid.{0}.{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), $PID)
            Move-Item -LiteralPath $OldVenv -Destination $invalidVenv -Force -ErrorAction SilentlyContinue
        }
    } elseif (
        $studioUsesLegacyLayout `
        -and (Test-Path -LiteralPath (Join-Path $env:USERPROFILE "unsloth_studio\Scripts\python.exe"))
    ) {
        # CWD-relative venv from old install.ps1 -> migrate to absolute path.
        # Skip custom-root env-mode so it is not relocated into a workspace root.
        $CwdVenv = Join-Path $env:USERPROFILE "unsloth_studio"
        substep "found CWD-relative Unsloth environment, migrating to $VenvDir..."
        try {
            Clear-MigrationTargetDirectory -Path $VenvDir
        } catch {
            Write-StudioLine "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
            return (Exit-InstallFailure "Could not clear $VenvDir for the environment migration")
        }
        Move-Item -LiteralPath $CwdVenv -Destination $VenvDir -Force
        substep "moved ~/unsloth_studio -> ~/.unsloth/studio/unsloth_studio"
        $_Migrated = $true
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        step "venv" "creating Python $($DetectedPython.Version) virtual environment"
        substep "$VenvDir"
        $venvExit = Invoke-InstallCommand -Label "create virtual environment" { & $script:UvExe venv $VenvDir --python "$($DetectedPython.Path)" }
        if ($venvExit -ne 0) {
            Write-StudioLine "[ERROR] Failed to create virtual environment (exit code $venvExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to create virtual environment (exit code $venvExit)" $venvExit)
        }
    } else {
        step "venv" "using migrated environment"
        substep "$VenvDir"
    }

    # Mark the managed venv before probing so failed installs can be replaced on rerun.
    if (Test-Path -LiteralPath $VenvDir -PathType Container) {
        try { [System.IO.File]::WriteAllText((Join-Path $VenvDir ".unsloth-studio-owned"), "") } catch {}
    }

    if (-not (Test-VenvPythonReady -PythonExe $VenvPython)) {
        $recordedBaseHome = Get-VenvBaseHome -VenvRoot $VenvDir
        Write-StudioLine "[ERROR] The managed Python interpreter is missing or cannot be launched." -ForegroundColor Red
        Write-StudioLine "        Managed Python: $VenvPython" -ForegroundColor Yellow
        if (-not $recordedBaseHome) { $recordedBaseHome = "unavailable" }
        Write-StudioLine "        Recorded base Python home: $recordedBaseHome" -ForegroundColor Yellow
        # The occupied-directory branch makes this venv replaceable on a plain re-run.
        Write-StudioLine "        Restore that Python installation, or just re-run install.ps1." -ForegroundColor Yellow
        return (Exit-InstallFailure "Managed Python is unavailable at $VenvPython (recorded base home: $recordedBaseHome)")
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
        # RunAsInvoker blocks the auto-elevation prompt; the timeout bounds a flaky amd-smi (30s mirrors amd.py).
        $prevCompat = [Environment]::GetEnvironmentVariable('__COMPAT_LAYER', 'Process')
        $env:__COMPAT_LAYER = 'RunAsInvoker'
        try {
            # [Process]::Start, NOT Start-Process -PassThru: the latter leaves .ExitCode $null after WaitForExit on PS 5.1, so $LASTEXITCODE reads non-zero and kills detection. Async reads drain the pipes; amd-smi args have no spaces so a plain join is safe.
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

    # ── Helper: run nvidia-smi under a timeout so a wedged driver can't hang the installer (no RunAsInvoker: nvidia-smi doesn't auto-elevate). Returns combined stdout+stderr; "" on timeout/failure. ──
    function Invoke-NvidiaSmiBounded {
        param(
            [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
            [Parameter(Position = 1)][string[]]$SmiArgs = @(),
            [int]$TimeoutSec = 10,
            # Driver warnings on stderr would corrupt machine-readable --query-gpu
            # output; the human-readable probes keep the default merge.
            [switch]$StdoutOnly
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
            if ($StdoutOnly) { return $outTask.Result }
            return ($outTask.Result + "`n" + $errTask.Result)
        } catch {
            $global:LASTEXITCODE = 1
            return ""
        }
    }

    # ── Helper: nvidia-smi -L lists at least one real GPU. Exit 0 alone isn't enough (a stale/driverless nvidia-smi can exit 0 with no GPU, marking an AMD host NVIDIA and suppressing ROCm) -- require a "GPU <n>:" data row. ──
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
    # Declared with its neighbours, not inside the block below: the arms that read
    # it are outside that gate, so an NVIDIA host would leave it undefined.
    $ROCmUnsupportedGfxArch = $null
    if (-not $HasNvidiaSmi) {
        # hipinfo: PATH first, then HIP_PATH/ROCM_PATH bin fallback (HIP SDK sets HIP_PATH but may not add bin to PATH). Ignore the venv hipInfo.exe (AMD wheel, not a HIP SDK, so amd-smi would still auto-elevate). Cf. _path_inside_venv().
        function Test-HipinfoIsVenvInternal {
            param([AllowNull()][string]$HipinfoPath)
            if ([string]::IsNullOrWhiteSpace($HipinfoPath)) { return $false }
            # Also derive the venv from the setup python + default Unsloth home, so the venv hipInfo is caught when VenvDir/VIRTUAL_ENV are unset.
            $venvRoots = @()
            if ($env:VIRTUAL_ENV) { $venvRoots += $env:VIRTUAL_ENV }
            $vd = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue
            if ($vd) { $venvRoots += $vd }
            if ($env:UNSLOTH_SETUP_PYTHON) {
                try { $venvRoots += (Split-Path -Parent (Split-Path -Parent $env:UNSLOTH_SETUP_PYTHON)) } catch {}
            }
            if ($env:USERPROFILE) { $venvRoots += (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio") }
            # A custom Unsloth home moves the venv off the default path; seed it too or its hipInfo escapes the filter.
            $studioHomeEnv = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() } else { $null }
            if ($studioHomeEnv) {
                # Expand a leading ~ like the canonical resolver; else GetFullPath keeps the literal ~ and the hipInfo escapes the filter.
                if (($studioHomeEnv -eq "~" -or $studioHomeEnv -like "~/*" -or $studioHomeEnv -like "~\*") -and -not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                    # A bare "~" leaves an empty child path (Join-Path rejects that on PS 5.1), so use USERPROFILE directly and only join a real remainder.
                    $studioHomeRest = $studioHomeEnv.Substring(1).TrimStart('/', '\')
                    $studioHomeEnv = if ($studioHomeRest) { Join-Path $env:USERPROFILE $studioHomeRest } else { $env:USERPROFILE }
                }
                $venvRoots += (Join-Path $studioHomeEnv "unsloth_studio")
            }
            try { $hip = [System.IO.Path]::GetFullPath($HipinfoPath).TrimEnd('\', '/') } catch { return $false }
            foreach ($root in $venvRoots) {
                if ([string]::IsNullOrWhiteSpace($root)) { continue }
                try { $r = [System.IO.Path]::GetFullPath($root).TrimEnd('\', '/') } catch { continue }
                # Skip a bare drive root (e.g. a non-venv UNSLOTH_SETUP_PYTHON yields C:) -- it would match every path on that drive.
                if ($r -match '^[a-zA-Z]:$') { continue }
                if ($hip.Equals($r, [System.StringComparison]::OrdinalIgnoreCase) -or
                    $hip.StartsWith($r + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                    return $true
                }
            }
            return $false
        }
        # Scan all hipinfo and keep the first non-venv one (the venv copy could shadow a real HIP SDK); -CommandType Application matches only real executables, not an alias/function named hipinfo.
        $hipinfoExe = Get-Command hipinfo -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-HipinfoIsVenvInternal $_.Source) } |
            Select-Object -First 1
        if (-not $hipinfoExe) {
            # Iterate the env roots and take the first non-venv bin\hipinfo.exe, so a venv-internal HIP_PATH can't mask a real SDK in ROCM_PATH.
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
                Write-StudioLine "  [WARN] hipinfo not on PATH -- located via ${hipEnvLabel}: $hipinfoCandidate" -ForegroundColor Yellow
                Write-StudioLine "         Add '$(Join-Path $hipRoot 'bin')' to your PATH to suppress this warning" -ForegroundColor Yellow
                Write-StudioLine "         Quick fix: [Environment]::SetEnvironmentVariable('PATH',`$env:PATH+';$(Join-Path $hipRoot 'bin')','User')" -ForegroundColor Yellow
                $hipinfoExe = [PSCustomObject]@{ Source = $hipinfoCandidate }
                break
            }
            if ((-not $hipinfoExe) -and $hipMissingLabel) {
                Write-StudioLine "  [WARN] ${hipMissingLabel}=$hipMissingRoot is set but hipinfo.exe not found at $hipMissingCandidate" -ForegroundColor Yellow
                Write-StudioLine "         HIP SDK install may be incomplete -- re-install from:" -ForegroundColor Yellow
                Write-StudioLine "         https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" -ForegroundColor Yellow
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
                        Write-StudioLine "  [INFO] hipinfo exited with code $LASTEXITCODE but reported gcnArchName -- treating as ROCm-capable (see #6043)" -ForegroundColor Cyan
                    }
                } elseif ($LASTEXITCODE -ne 0) {
                    # hipinfo ran but returned a HIP runtime error without any gcnArchName
                    # output (e.g. "no ROCm-capable device detected"), or crashed before
                    # printing device info.
                    $firstLine = ($hipOut -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -First 1)
                    Write-StudioLine "  [WARN] hipinfo returned a HIP runtime error (exit $LASTEXITCODE)" -ForegroundColor Yellow
                    Write-StudioLine "         $firstLine" -ForegroundColor Yellow
                    Write-StudioLine "         Ensure ROCm drivers are installed: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" -ForegroundColor Yellow
                }
            } catch {}
        }
        # Without a working HIP runtime amd-smi elevates a child at runtime (UAC/DiskPart prompt RunAsInvoker can't suppress), so only probe when a HIP SDK is present or the user opts in; else fall through to WMI name inference (enough for ROCm wheels + llama.cpp prebuilt). An explicit opt-out (UNSLOTH_ENABLE_AMD_SMI=0/false/no/off) wins over the HIP-SDK heuristic, since a broken runtime can still pop the prompt.
        $amdSmiOptOut = $env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(0|false|no|off)$'
        $amdSmiAllowed = (-not $amdSmiOptOut) -and ($HipSdkInstalled -or ($env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(1|true|yes|on)$'))
        if (-not $HasROCm -and $amdSmiAllowed) {
            $amdSmiExe = Get-Command "amd-smi" -ErrorAction SilentlyContinue
            if ($amdSmiExe) {
                try {
                    $smiOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('list')
                    if ($LASTEXITCODE -eq 0 -and $smiOut -match "(?im)^GPU\s*[:\[]\s*\d") {
                        $HasROCm = $true
                        # Mirror the hipinfo path: collect all gfx tokens in enumeration order and pick the runtime-visible one via HIP_VISIBLE_DEVICES.
                        $_smiVisIdx = if ($env:HIP_VISIBLE_DEVICES -match '^\d') { [int]($env:HIP_VISIBLE_DEVICES -split ',')[0] } elseif ($env:ROCR_VISIBLE_DEVICES -match '^\d') { [int]($env:ROCR_VISIBLE_DEVICES -split ',')[0] } else { 0 }
                        # Attempt 1: newer amd-smi versions embed the gfx arch in list output.
                        $_smiGfxTokens = @([regex]::Matches($smiOut, "(?i)\b(gfx\d+[a-z]?)\b") | ForEach-Object { $_.Groups[1].Value.ToLower() })
                        if ($_smiGfxTokens.Count -gt 0) {
                            $ROCmGfxArch = if ($_smiVisIdx -lt $_smiGfxTokens.Count) { $_smiGfxTokens[$_smiVisIdx] } else { $_smiGfxTokens[0] }
                            $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                        } else {
                            # Attempt 2: 'static --asic' exposes the GFX target (ROCm 6+) needed for wheel index selection.
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
        # This scan fills the LABEL the arch inference reads, so its gate is load-bearing:
        # widening it turned a CPU install into a ROCm one on hosts amd-smi had already
        # claimed. CIM, not Get-WmiObject, which PowerShell 7 removed: the catch below is
        # silent, so a supported Radeon named only by Windows took the CPU path there, and
        # the report-only peer scan cannot undo that. Same class and fields as every other
        # adapter scan in this file, and identical output under Windows PowerShell 5.1.
        if (-not $HasROCm) {
            try {
                # ConfigManagerErrorCode 0 is "working properly". Filter on it exactly as
                # setup.ps1's scan does: taking a card setup discards names an arch for a GPU
                # that is not the active one, and since a mapped arch installs ROCm wheels right
                # here, a disabled Radeon listed ahead of a healthy one bought wheels for the
                # dead card while the live one went unserved.
                # If the filter leaves none, keep the full list: code 45 ("not connected") is
                # routine on a muxless laptop with a parked dGPU, and there is no healthy peer
                # to prefer. @() wraps the WHOLE if so a one-element branch stays indexable.
                $amdAdapters = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match "AMD|Radeon" })
                $healthyAdapters = @($amdAdapters | Where-Object {
                    ($null -eq $_.ConfigManagerErrorCode) -or ($_.ConfigManagerErrorCode -eq 0) })
                $wmiGpu = @(if ($healthyAdapters.Count -gt 0) { $healthyAdapters } else { $amdAdapters })[0]
                if ($wmiGpu) { $ROCmGpuLabel = $wmiGpu.Name }
            } catch {}
        }
        # Peer names for the REPORT ONLY, kept apart from the scan above: that one feeds the
        # inference and runs only without a runtime, this one only the uncovered-card verdict.
        $wmiAmdNames = @()
        if (-not $ROCmGfxArch) {
            try {
                $peerAdapters = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match "AMD|Radeon" })
                $healthyPeers = @($peerAdapters | Where-Object {
                    ($null -eq $_.ConfigManagerErrorCode) -or ($_.ConfigManagerErrorCode -eq 0) })
                $usePeers = @(if ($healthyPeers.Count -gt 0) { $healthyPeers } else { $peerAdapters })
                $wmiAmdNames = @($usePeers | ForEach-Object { $_.Name })
            } catch {}
        }
        # GPU name -> gfx arch for AMD generations Unsloth's ROCm wheels do NOT cover:
        # RDNA 1 and Polaris 10/20/30 (unslothai#8529). Kept apart from $nameArchTable on
        # purpose: it only WORDS a message, never selects a wheel index. AMD's TheRock
        # ships RDNA 1 wheels, but not on the repo.amd.com indexes routed here, and never
        # gfx803. The (?!0) guards stop "RX 570" swallowing an "RX 5700". Names from
        # LLVM's AMDGPU tables plus libdrm amdgpu.ids/pci.ids for the Navi 10/14
        # professional parts LLVM omits; nothing is guessed, so Polaris 11/12 (RX
        # 460/550/560, a different die) is left out.
        $unsupportedNameArchTable = @(
            @{ P = "Radeon Pro V520|Radeon Pro 5600M";        A = "gfx1011" }  # RDNA 1
            @{ P = "RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700";     A = "gfx1010" }  # RDNA 1 (Navi 10)
            @{ P = "RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300";        A = "gfx1012" }  # RDNA 1 (Navi 14)
            @{ P = "RX 4[78]0(?!0)|RX 5[789]0(?!0)|Radeon Pro WX 7100|Radeon Pro WX 5100"; A = "gfx803"  }  # Polaris 10/20/30
        )
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
            # 2. Best-effort name → arch lookup from marketing name (amd-smi / WMI); targets only arches the ROCm prebuilts cover (gfx120X/110X/1151/1150/103X), unknown names fall back to CPU.
            elseif ($ROCmGpuLabel) {
                $nameArchTable = @(
                    @{ P = "9070|9080|R9700";                                     A = "gfx1201" }  # RDNA 4 (Navi 48: RX 9070 XT / 9070 GRE / 9070 / 9080, Radeon AI PRO R9700)
                    @{ P = "9060";                                                A = "gfx1200" }  # RDNA 4 (Navi 44: RX 9060 XT / 9060)
                    @{ P = "8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max"; A = "gfx1151" }  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
                    @{ P = "890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]"; A = "gfx1150" }  # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
                    @{ P = "860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33"; A = "gfx1152" }  # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
                    @{ P = "RX 7900|PRO W7900|PRO W7800";                         A = "gfx1100" }  # RDNA 3 desktop/workstation (Navi 31)
                    @{ P = "RX 7800|RX 7700(?!S)|PRO W7700|PRO V710";             A = "gfx1101" }  # RDNA 3 (Navi 32)
                    @{ P = "RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500";        A = "gfx1102" }  # RDNA 3 (Navi 33)
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
                # 3. Still nothing: the card may be a generation ROCm never covered rather
                #    than one we failed to recognise (unslothai#8529). Reporting only --
                #    $ROCmGfxArch stays null, so CPU fallback is reached by the same path.
                #    Gated on NO adapter being covered: $wmiGpu takes index 0, so on a host
                #    pairing an RX 5700 with an RX 7900 the label is the 5700 and the "no
                #    SDK or override can help" wording below would be false (masking to the
                #    7900 and setting gfx1100 installs). Such a host keeps the arch-unknown
                #    arm, which says exactly that. studio/setup.ps1 already scores every
                #    adapter before it reaches its own lookup.
                if (-not $ROCmGfxArch) {
                    # HIP/ROCR only: CUDA_VISIBLE_DEVICES masks NVIDIA devices and says
                    # nothing about which Radeon was chosen.
                    $unsupMasked = @($env:HIP_VISIBLE_DEVICES, $env:ROCR_VISIBLE_DEVICES) |
                        Where-Object { $null -ne $_ }
                    $coveredPeer = $false
                    if ($unsupMasked) {
                        # A masked-out peer cannot answer for the card the user named, but the
                        # label is only that card when there is nothing else to select: both
                        # sources above keep adapter 0, so HIP_VISIBLE_DEVICES=1 beside a
                        # supported peer would blame the wrong GPU. Stay quiet, rather than
                        # guess an order Win32_VideoController does not promise matches HIP's.
                        $coveredPeer = ($wmiAmdNames.Count -gt 1)
                    } else {
                        foreach ($peerName in $wmiAmdNames) {
                            foreach ($row in $nameArchTable) {
                                if ($peerName -match $row.P) { $coveredPeer = $true; break }
                            }
                            if ($coveredPeer) { break }
                        }
                    }
                    if (-not $coveredPeer) {
                        foreach ($row in $unsupportedNameArchTable) {
                            if ($ROCmGpuLabel -match $row.P) {
                                $ROCmUnsupportedGfxArch = $row.A
                                break
                            }
                        }
                    }
                }
            }
        }
        # Capture ROCm version for wheel selection (hipconfig, then amd-smi). Run whenever the HIP SDK binary is present, since hipconfig --version works even when hipinfo reports no ROCm device (driver issue).
        if ($HasROCm -or $HipSdkInstalled) {
            $hipConfigExe = Get-Command hipconfig -ErrorAction SilentlyContinue
            if (-not $hipConfigExe) {
                $hipRoot = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { $null }
                if ($hipRoot) {
                    $hipConfigCandidate = Join-Path $hipRoot "bin\hipconfig.exe"
                    if (Test-Path $hipConfigCandidate) {
                        $hipConfigEnvLabel = if ($env:HIP_PATH) { "HIP_PATH" } else { "ROCM_PATH" }
                        Write-StudioLine "  [WARN] hipconfig not on PATH -- located via ${hipConfigEnvLabel}: $hipConfigCandidate" -ForegroundColor Yellow
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

    # ── Optional WSL-ROCm driver hint: WSL2 needs AMD Adrenalin >= 26.2.2 (native Windows GPU works with any recent driver). Can't auto-install (AMD referrer-gates downloads, no winget package), so just point at AMD's page; shown only when the installed driver predates 26.2.2 (Feb 2026). Suppress with UNSLOTH_SKIP_AMD_DRIVER_HINT=1. ──
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
            # Older than 26.2.2 (Feb 2026) => can't expose the GPU to WSL ROCm; unreadable date => still show the hint (informational, suppressible).
            if ($drvDate -and $drvDate -ge (Get-Date '2026-02-01')) { return }
            substep "Tip: to use this GPU inside WSL too, install AMD Adrenalin 26.2.2+ (for WSL2)." "Cyan"
            substep "  Your current driver predates it; native Windows GPU is unaffected. Get it from AMD:" "Cyan"
            substep "    https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html" "Cyan"
            substep "  Then reboot and run this installer inside an Ubuntu-24.04 WSL distro." "Cyan"
            # If WSL isn't installed yet, point at the command that provisions it (wsl.exe absent => no WSL).
            $hasWsl = $false
            try { $hasWsl = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue) } catch {}
            if (-not $hasWsl) {
                substep "  No WSL yet? Install it first:  wsl --install -d Ubuntu-24.04" "Cyan"
            }
            substep "  (suppress: set UNSLOTH_SKIP_AMD_DRIVER_HINT=1)" "Cyan"
        } catch {}
    }

    # ── AMD gfx arch → AMD pip index family (repo.amd.com/rocm/whl/<family>) ──
    # Hoisted above the Intel scan, which needs it: an arch missing here gets CPU torch, so it
    # must not outrank a usable Arc card. The AMD reroute below consumes it too.
    $archFamilyMap = @{
        "gfx1201" = "gfx120X-all"; "gfx1200" = "gfx120X-all"  # RDNA 4
        "gfx1151" = "gfx1151";     "gfx1150" = "gfx1150"       # RDNA 3.5 (Strix Halo/Point)
        "gfx1152" = "gfx1152"                                  # RDNA 3.5 (Krackan Point)
        "gfx1103" = "gfx110X-all"; "gfx1102" = "gfx110X-all"   # RDNA 3
        "gfx1101" = "gfx110X-all"; "gfx1100" = "gfx110X-all"
        "gfx1036" = "gfx103X-all"; "gfx1035" = "gfx103X-all"   # RDNA 2 (RX 6000)
        "gfx1034" = "gfx103X-all"; "gfx1033" = "gfx103X-all"
        "gfx1032" = "gfx103X-all"; "gfx1031" = "gfx103X-all"
        "gfx1030" = "gfx103X-all"
        "gfx90a"  = "gfx90a";      "gfx908"  = "gfx908"        # MI200/MI100
    }
    # "AMD gets GPU wheels here", NOT "an AMD GPU is present": $HasROCm / $ROCmGfxArch are
    # true on unmapped arches too, and those install CPU torch.
    $AmdHasGpuWheels = [bool]($ROCmGfxArch -and $archFamilyMap.ContainsKey($ROCmGfxArch))

    # ── Bounded "ask the venv python" probe ──
    # A wedged torch import or a hanging Intel driver init -- what the XPU probes below exist to
    # detect -- would block a bare `& python -c ...` forever. ProcessStartInfo, not &, so stderr
    # cannot trip $ErrorActionPreference; BOTH streams drain async so a noisy import cannot
    # deadlock on a full pipe; WaitForExit bounds the wait and kills the child. Every failure
    # (timeout, crash, exception) reads as .Ok = $false; .Error carries WHICH one, since stderr
    # used to be drained and discarded, leaving a driver-level DLL load error and a missing torch
    # indistinguishable. Defined above the Intel scan: PowerShell binds a function when it runs.
    function Invoke-BoundedPythonProbe {
        param([string]$PythonExe, [string]$Code, [int]$TimeoutSec = 30)
        $result = [pscustomobject]@{ Ok = $false; Output = ""; Error = "" }
        if (-not $PythonExe -or -not $Code) { return $result }
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $PythonExe
            $psi.Arguments = "-c `"$Code`""
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
                try { $proc.Kill() } catch {}
                # Synthesised, not read back: waiting on the reader tasks of a wedged child
                # would reintroduce the hang this helper exists to bound.
                $result.Error = "python did not answer within $TimeoutSec seconds"
                return $result
            }
            $result.Output = $outTask.GetAwaiter().GetResult()
            # Kept, not discarded: the only place a failed probe's OSError / WinError text exists.
            $result.Error = $errTask.GetAwaiter().GetResult()
            $result.Ok = ($proc.ExitCode -eq 0)
            return $result
        } catch {
            $result.Error = $_.Exception.Message
            return $result
        }
    }

    # Bounded Win32_VideoController scan: the query can block forever on a degraded WMI
    # repository, -ErrorAction only suppresses reported errors, and -OperationTimeoutSec is not
    # enforced for the local COM session this uses, so out of process with a wall-clock kill is
    # the only bound that holds. Ok = $false on an empty answer too, since a Windows host always
    # has an adapter. Mirrors setup.ps1's copy.
    function Invoke-BoundedVideoControllerScan {
        param([int]$TimeoutSec = 15)
        $result = [pscustomobject]@{ Ok = $false; Names = @() }
        $job = $null
        try {
            $job = Start-Job -ScriptBlock {
                Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty Name
            }
            if (Wait-Job -Job $job -Timeout $TimeoutSec) {
                $names = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
                $result.Names = @($names | Where-Object { $_ })
                $result.Ok = ($result.Names.Count -gt 0)
            } else {
                Stop-Job -Job $job -ErrorAction SilentlyContinue
            }
        } catch {
        } finally {
            if ($job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue }
        }
        return $result
    }

    # Registry fallback for the scan above, mirroring install_llama_prebuilt.py's
    # windows_intel_gpu_in_registry(): the display-adapter class key, one NNNN subkey per driver
    # config. Weaker than WMI (a config can outlive removed hardware), so it is the fallback here
    # while that function is registry-first -- there a false positive only picks a different
    # llama.cpp bundle, here it would install XPU torch on a host with no Arc. Mirrors setup.ps1.
    function Get-IntelRegistryAdapterNames {
        $names = @()
        $classKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try {
            $subs = @(Get-ChildItem -LiteralPath $classKey -ErrorAction SilentlyContinue)
        } catch { return @() }
        foreach ($sub in $subs) {
            # Guarded per subkey, not around the loop: one unreadable entry must not discard the
            # adapters found after it. Matches windows_intel_gpu_in_registry()'s per-key skip.
            try {
                # Numeric subkeys only: "Properties" is ACL-restricted and not an adapter.
                if ("$($sub.PSChildName)" -match '^\d+$') {
                    $props = Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue
                    if ($props) {
                        $desc = "$($props.DriverDesc)"
                        # Callers re-filter on "Intel", so a vendor-ID hit with a localized or
                        # OEM-branded DriverDesc would be found here and dropped there. Tag it
                        # instead; still only XPU-capable if the name says Arc / Data Center GPU.
                        if ("$($props.MatchingDeviceId)" -match '(?i)ven_8086') {
                            $names += if ($desc -match '(?i)intel') { $desc }
                                      elseif ($desc) { "Intel $desc" }
                                      else { "Intel Graphics" }
                        } elseif ($desc -match '(?i)intel') {
                            $names += $desc
                        }
                    }
                }
            } catch { }
        }
        return $names
    }

    # ── Intel GPU detection (Arc / Data Center GPU Max / Flex) ──
    # Runs BEFORE the report chain, not inside its final else: a WMI-named-only AMD adapter
    # would take that chain and hide a discrete Arc card. $HasIntelGpu is "an Intel adapter is
    # present"; $script:IsIntelXpu is "XPU wheels suit it" -- only Arc / Data Center parts
    # qualify, UHD / HD / Iris Xe do not.
    $HasIntelGpu = $false
    $IntelGpuLabel = $null
    # Reset every invocation: under a piped web run $script: is the caller's session scope, so a
    # second run would inherit a stale $true and reroute a now-NVIDIA host to the xpu index.
    $script:IsIntelXpu = $false
    # $AmdHasGpuWheels keeps a wheel-served AMD host out of the XPU reroute below; an AMD host
    # with no wheels is heading for CPU torch, so a neighbouring Arc card wins.
    if (-not $HasNvidiaSmi -and -not $AmdHasGpuWheels) {
        try {
            # Bounded, registry as the fallback when WMI does not answer; same match either way.
            $_gpuScan = Invoke-BoundedVideoControllerScan
            # @() wraps the WHOLE if, not each branch: a one-element array unrolls on its way
            # out, making $_gpuNames a String on a single-adapter host and turning the += below
            # into string concatenation.
            $_gpuNames = @(if ($_gpuScan.Ok) { $_gpuScan.Names } else { Get-IntelRegistryAdapterNames })
            # One definition for both the reconciliation gate and the classification below.
            $_xpuNameRe = "(?i)Intel.*(Arc|Data Center GPU)"
            # On non-English Windows the WMI name carries no ASCII "Intel" for the
            # classification below. The registry helper resolves the PCI vendor id, so use it to
            # RE-LABEL an adapter WMI already reported, never to add one (an unmatched entry is a
            # driver record outliving its card). Gated on the absence of an XPU match, not of any
            # Intel name: a hybrid laptop reports "Intel UHD" next to a localized Arc.
            if ($_gpuScan.Ok -and -not ($_gpuNames | Where-Object { $_ -match $_xpuNameRe })) {
                foreach ($_reg in @(Get-IntelRegistryAdapterNames)) {
                    foreach ($_wmiName in $_gpuNames) {
                        if ($_wmiName -and $_reg.Contains($_wmiName)) { $_gpuNames += $_reg; break }
                    }
                }
            }
            $intelGpus = @($_gpuNames | Where-Object { $_ -match "(?i)Intel" })
            if ($intelGpus.Count -gt 0) {
                $HasIntelGpu = $true
                $xpuGpu = $intelGpus | Where-Object { $_ -match $_xpuNameRe } | Select-Object -First 1
                $IntelGpuLabel = if ($xpuGpu) { $xpuGpu } else { $intelGpus[0] }
                if ($xpuGpu) { $script:IsIntelXpu = $true }
            }
        } catch {}
        # A migrated env's torch can only CONFIRM the match, never veto it: an unavailable
        # runtime means a stale driver, not unsuitable hardware, and the /cpu fallback could not
        # displace the installed +xpu wheel anyway (PEP 440 ignores the local label). Bounded,
        # and a timeout reads as "no XPU" -- the WMI verdict stands.
        # A rerun has already moved the old venv to $script:StudioVenvRollbackDir and put an
        # empty one in its place, so probing $VenvPython would ask an interpreter with no torch.
        # Ask the preserved environment instead -- that is the migrated runtime this rescues.
        $_xpuProbePy = $VenvPython
        if ($script:StudioVenvRollbackDir) {
            $_rollbackPy = Join-Path $script:StudioVenvRollbackDir "Scripts\python.exe"
            if (Test-Path -LiteralPath $_rollbackPy) { $_xpuProbePy = $_rollbackPy }
        }
        if (Test-Path -LiteralPath $_xpuProbePy) {
            $xpuCheck = Invoke-BoundedPythonProbe -PythonExe $_xpuProbePy -Code 'import torch; print(torch.xpu.is_available())'
            if ($xpuCheck.Ok -and $xpuCheck.Output -match '(?m)^\s*True\s*$') {
                $HasIntelGpu = $true
                $script:IsIntelXpu = $true
                if (-not $IntelGpuLabel) { $IntelGpuLabel = "Intel GPU (detected by PyTorch XPU)" }
            }
        }
    }

    if ($HasNvidiaSmi) {
        step "gpu" "NVIDIA GPU detected"
    } elseif ($script:IsIntelXpu) {
        # Ranks above every AMD branch: only true when AMD gets no GPU wheel ($AmdHasGpuWheels
        # gates the scan above), so those branches would all end on CPU.
        step "gpu" "Intel GPU detected" "Green"
        substep "$IntelGpuLabel"
        # The reroute below prints the index: only it knows the mirror URL and any pin.
    } elseif ($HasROCm -and -not $ROCmUnsupportedGfxArch) {
        # Guarded like the HIP SDK arm below: amd-smi can report a GPU with no gfx token
        # and only a market name, setting $HasROCm without an arch. Calling that card
        # "AMD ROCm" contradicts the wheel note this run also prints.
        step "gpu" $ROCmGpuLabel
        $hipSdkPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "on system PATH" }
        substep "HIP SDK: $hipSdkPath"
        if ($ROCmVersionFull) { substep "hipconfig: $ROCmVersionFull" }
    } elseif ($HipSdkInstalled -and $ROCmGpuLabel -and -not $ROCmUnsupportedGfxArch) {
        # HIP SDK installed but ROCm can't see the device (driver issue, not SDK issue).
        # Excludes cards already known to be out of scope: the #8529 reporters installed
        # the HIP SDK BECAUSE this arm said to, so unguarded it hides the arm below from
        # exactly the users it is for.
        $sdkVer = if ($ROCmVersionFull) { " (HIP $ROCmVersionFull)" } else { "" }
        step "gpu" "AMD GPU detected -- not ROCm-accessible$sdkVer" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        substep "[WARN] HIP SDK is installed but hipinfo reports no ROCm-capable device." "Yellow"
        substep "       This is a driver issue, not an SDK issue." "Yellow"
        substep "       Ensure the ROCm compute driver is installed alongside the display driver:" "Yellow"
        substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    } elseif ($ROCmGfxArch) {
        # Known arch: Unsloth setup installs AMD's bundled-runtime ROCm PyTorch wheels (repo.amd.com), which ship their own runtime -- HIP SDK optional.
        step "gpu" "AMD ROCm ($ROCmGfxArch)" "Cyan"
        substep "Detected: $ROCmGpuLabel" "Cyan"
        substep "GPU PyTorch uses AMD's bundled-runtime ROCm wheels -- HIP SDK not required (optional)." "Cyan"
    } elseif ($ROCmUnsupportedGfxArch) {
        # Detected, identified, out of scope for ROCm PyTorch. Ranks above the "arch
        # unknown" arm below: the arch is known here, and that arm's advice cannot
        # succeed on this GPU (unslothai#8529).
        # Not "training runs on CPU": with no CUDA/XPU visible, unsloth raises
        # NotImplementedError at import (unsloth/device_type.py). The Vulkan setter is
        # single-quoted so PowerShell prints $env:... rather than expanding it; a pasted
        # VAR=value resolves as a command name here and sets nothing.
        # Both claims below are conditional: an explicit index pin reaches the ROCm install
        # path further down even for an arch Unsloth has no wheels for, and studio/setup.ps1
        # THROWS on the Vulkan variable on Windows ARM64, where no bundle is published.
        $unsupPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) -or `
                       (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY))
        $unsupArm64 = (Get-HostMachineArch) -eq "arm64"
        step "gpu" "AMD GPU detected ($ROCmUnsupportedGfxArch) -- no ROCm PyTorch wheels Unsloth installs" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        if ($unsupPinned) {
            substep "Unsloth ships no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch, but the torch index" "Yellow"
            substep "you pinned is used as given, so torch is whatever that index publishes." "Yellow"
        } else {
            substep "Unsloth installs no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch, so torch stays" "Yellow"
            substep "CPU-only: Unsloth training and GPU inference are unavailable. Installing the" "Yellow"
            substep "HIP SDK or setting UNSLOTH_ROCM_GFX_ARCH will not change that for it." "Yellow"
        }
        if ($unsupArm64) {
            substep "GGUF chat would need Vulkan on this GPU, and no Windows ARM64 Vulkan bundle is published: build llama.cpp from source, or run this on x64." "Yellow"
        } else {
            substep "GGUF chat can still use this GPU through Vulkan: set" "Yellow"
            substep '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan" and re-run this installer. It' "Yellow"
            substep "selects the llama.cpp bundle at install time, so setting it afterwards has" "Yellow"
            substep "no effect until you install or update again." "Yellow"
        }
    } elseif ($ROCmGpuLabel) {
        step "gpu" "AMD GPU detected -- arch unknown" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        substep "Could not determine the GPU arch -- install the HIP SDK or set" "Yellow"
        substep "UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm PyTorch:" "Yellow"
        substep "https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    } else {
        step "gpu" "none (chat-only / GGUF)" "Yellow"
        if ($HasIntelGpu) { substep "Detected: $IntelGpuLabel (not XPU-capable)" "Yellow" }
        substep "Training and GPU inference require an NVIDIA, AMD ROCm, or Intel Arc GPU." "Yellow"
    }
    # On an AMD GPU (no NVIDIA), surface the optional WSL-ROCm driver hint.
    if (-not $HasNvidiaSmi -and ($ROCmGfxArch -or $ROCmGpuLabel)) { Show-AmdWslDriverHint }

    # Trim trailing slashes from the URL PATH only, preserving ?query / #fragment (a whole-URL TrimEnd corrupts a token ending in "/"). Shared.
    function Trim-IndexPathSlashes {
        param([string]$Url)
        $value = $Url.Trim()
        $idx = $value.IndexOfAny([char[]]@('?', '#'))
        if ($idx -lt 0) {
            return $value.TrimEnd('/')
        }
        return $value.Substring(0, $idx).TrimEnd('/') + $value.Substring($idx)
    }

    # Index leaf (cpu / cu128 / xpu / gfx1201), ?query and #fragment stripped so a
    # token-authenticated mirror still classifies by family.
    function Get-TorchIndexLeafName {
        param([string]$Url)
        if ([string]::IsNullOrWhiteSpace($Url)) { return "" }
        return ((($Url -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1]).ToLowerInvariant()
    }

    # Classify the physical NVIDIA inventory for a cu126 fallback: "cu126" when it
    # covers every GPU, "uncovered" for an incompatible mix, empty when no fallback is
    # needed or the inventory is unreadable. CUDA_VISIBLE_DEVICES is ignored because
    # the wheel must support the host. Mirrors _nvidia_cu126_verdict in install.sh.
    function Get-NvidiaCu126Verdict {
        # Floor is per-release, not fixed: only 2.11 dropped sm_70 from cu128.
        param([string]$SmiExe, [int]$LegacyFloorSm = 75)
        if (-not $SmiExe) { return '' }
        $raw = Invoke-NvidiaSmiBounded $SmiExe @('--query-gpu=compute_cap', '--format=csv,noheader,nounits') -StdoutOnly
        if ($LASTEXITCODE -ne 0 -or -not $raw) { return '' }
        $legacy = $false
        $outsideCu126 = $false
        $seen = $false
        foreach ($line in ($raw -split "`n")) {
            $value = $line.Trim()
            if (-not $value) { continue }
            if ($value -notmatch '^(\d+)\.(\d+)$') { return '' }
            $sm = ([int]$Matches[1] * 10) + [int]$Matches[2]
            if ($sm -lt $LegacyFloorSm) { $legacy = $true }
            if ($sm -lt 50 -or $sm -gt 90) { $outsideCu126 = $true }
            $seen = $true
        }
        if (-not $seen -or -not $legacy) { return '' }
        if ($outsideCu126) { return 'uncovered' }
        return 'cu126'
    }

    function Get-CudaFamilyCappedForPreTuring {
        param([string]$Family, [string]$SmiExe)
        if ($Family -notin @('cu128', 'cu130')) { return $Family }
        # torch 2.11.0+cu128 dropped Volta (its arch list runs sm_75..sm_120; 2.10.0+cu128
        # still had sm_70), so now that the pin reaches 2.11 a pre-Turing host is stranded
        # on cu128 exactly as it is on cu130. Both families take the 75 floor.
        $legacyFloorSm = 75
        switch (Get-NvidiaCu126Verdict $SmiExe $legacyFloorSm) {
            'cu126' {
                substep "pre-Turing NVIDIA GPUs (sm_<75) are present -- selecting cu126, because PyTorch 2.11's $Family wheels start at sm_75" "Yellow"
                return 'cu126'
            }
            'uncovered' {
                substep "this host mixes pre-Turing NVIDIA GPUs with GPUs that cu126 cannot serve; no PyTorch 2.11 CUDA family covers both" "Yellow"
                substep "keeping $Family, so the pre-Turing GPUs will be unusable; set UNSLOTH_TORCH_INDEX_FAMILY=cu126 to choose the other way" "Yellow"
            }
        }
        return $Family
    }

    # ── Choose the correct PyTorch index URL based on driver CUDA version ──
    # Mirrors Get-PytorchCudaTag in setup.ps1.
    function Get-TorchIndexUrl {
        $baseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        # Explicit pin skips ALL GPU probing: UNSLOTH_TORCH_INDEX_URL wins (full URL, verbatim); _FAMILY is the leaf appended to the mirror base. Matches install.sh / install_python_stack.py.
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
            return (Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL)
        }
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY)) {
            return "$baseUrl/$($env:UNSLOTH_TORCH_INDEX_FAMILY.Trim().Trim('/'))"
        }
        if (-not $NvidiaSmiExe) { return "$baseUrl/cpu" }
        try {
            $output = Invoke-NvidiaSmiBounded $NvidiaSmiExe
            # Newer NVIDIA drivers print "CUDA UMD Version: X.Y" instead of the legacy "CUDA Version: X.Y"; accept both so we don't fall through to the cu126 default.
            if ($output -match 'CUDA(?: UMD)? Version:\s+(\d+)\.(\d+)') {
                $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                if ($major -ge 13)                        { $family = "cu130" }
                elseif ($major -eq 12 -and $minor -ge 8)  { $family = "cu128" }
                elseif ($major -eq 12 -and $minor -ge 6)  { $family = "cu126" }
                elseif ($major -ge 12) { $family = "cu124" }
                elseif ($major -ge 11) { $family = "cu118" }
                else { return "$baseUrl/cpu" }
                return "$baseUrl/$(Get-CudaFamilyCappedForPreTuring $family $NvidiaSmiExe)"
            }
        } catch {}
        substep "could not determine CUDA version from nvidia-smi, defaulting to cu126" "Yellow"
        return "$baseUrl/cu126"
    }

    # Strip userinfo AND query/fragment so an authenticated pin never leaks. Shared with _strip_index_url_credentials (install.sh / py / setup.ps1).
    function Remove-IndexUrlCredentials {
        param([string]$Url)
        # Ordinal, not culture-aware: on non-English locales (e.g. th-TH) linguistic
        # IndexOf treats "://" as ignorable, mis-locates it, and crashes Substring (issue #7279).
        $sep = $Url.IndexOf('://', [System.StringComparison]::Ordinal)
        if ($sep -lt 0) { return $Url }
        $scheme = $Url.Substring(0, $sep)
        $rest = $Url.Substring($sep + 3)
        # Drop query / fragment (may hold auth tokens).
        $q = $rest.IndexOfAny([char[]]('?', '#'))
        if ($q -ge 0) { $rest = $rest.Substring(0, $q) }
        $slash = $rest.IndexOf('/', [System.StringComparison]::Ordinal)
        $authority = if ($slash -ge 0) { $rest.Substring(0, $slash) } else { $rest }
        $at = $authority.LastIndexOf('@', [System.StringComparison]::Ordinal)
        $host_ = if ($at -ge 0) { $authority.Substring($at + 1) } else { $authority }
        if ($slash -ge 0) { return "${scheme}://${host_}$($rest.Substring($slash))" }
        return "${scheme}://${host_}"
    }

    # Append a path to a URL that may carry ?query / #fragment auth. A private mirror is
    # allowed to be "https://mirror/whl?token=abc", and a naive "$base/$leaf" put the leaf
    # INSIDE the token value, leaving the path still /whl -- so the tokenized mirror this
    # exists to honour was the one case that could not resolve a wheel.
    function Join-UrlPath {
        param([string]$Base, [string]$Path)
        if ([string]::IsNullOrWhiteSpace($Base)) { return $Path }
        $cut = $Base.IndexOfAny([char[]]('?', '#'))
        if ($cut -lt 0) { return "$($Base.TrimEnd('/'))/$Path" }
        $head = $Base.Substring(0, $cut).TrimEnd('/')
        $tail = $Base.Substring($cut)
        return "$head/$Path$tail"
    }

    # ── Torch flavor helpers (to repair a stale CPU / wrong-CUDA wheel) ──
    # torch.__version__ -> flavor tag (cuXXX / rocm / cpu); untagged wheel = cpu,
    # matching setup.ps1's stale-venv parse.
    function ConvertTo-TorchFlavorTag {
        param([string]$TorchVersion)
        if (-not $TorchVersion) { return $null }
        if ($TorchVersion -match '\+(cu\d+)') { return $Matches[1] }
        if ($TorchVersion -match '\+rocm')    { return 'rocm' }
        if ($TorchVersion -match '\+xpu')     { return 'xpu' }
        if ($TorchVersion -match '\+cpu')     { return 'cpu' }
        return 'cpu'
    }

    # Expected tag from the index leaf: cuXXX / cpu / rocm ($ROCmIndexUrl or a gfx* leaf -> rocm); $null on an unknown leaf so repair no-ops.
    function Get-ExpectedTorchFlavorTag {
        param([string]$TorchIndexUrl, [string]$ROCmIndexUrl)
        if (-not [string]::IsNullOrWhiteSpace($ROCmIndexUrl)) { return 'rocm' }
        if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) { return $null }
        # Drop query/fragment first so .../cu128?token=x classifies as cu128 (else it reinstalls every run).
        $leaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
        if ($leaf -match '^cu\d+$') { return $leaf }
        if ($leaf -eq 'cpu')        { return 'cpu' }
        if ($leaf -eq 'xpu')        { return 'xpu' }
        if ($leaf -match '^rocm')   { return 'rocm' }
        # gfx must be followed by a digit (an architecture leaf); gfx-private is custom.
        if ($leaf -match '^gfx[0-9]') { return 'rocm' }
        return $null
    }

    # sysconfig platform of $PythonExe's venv, lowercased ("" when it cannot be asked). Windows
    # on ARM has no torchaudio wheel on any index, so every XPU spec list is built from this.
    function Get-VenvPlatformTag {
        param([string]$PythonExe)
        if (-not $PythonExe) { return "" }
        try {
            return (& $PythonExe -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
        } catch { return "" }
    }

    # The XPU trio for $Platform: floor 2.6 (unsloth/models/_utils.py raises at import for an
    # XPU device below it) and no torchaudio on win-arm64.
    function Get-XpuTorchSpecs {
        param([string]$Platform)
        $specs = @("torch>=2.6,<2.11.0", "torchvision>=0.21,<0.26.0")
        if ($Platform -eq "win-arm64") { return $specs }
        return $specs + @("torchaudio>=2.6,<2.11.0")
    }

    # Installed torch flavor tag in $PythonExe's venv, or $null if absent. Bounded, so a wedged
    # "import torch" cannot stall the flavor repair.
    function Get-InstalledTorchTag {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.__version__)'
        if (-not $probe.Ok) { return $null }
        $torchVer = $probe.Output.Trim()
        if (-not $torchVer) { return $null }
        return ConvertTo-TorchFlavorTag $torchVer
    }

    # Full installed torch version in $PythonExe's venv ("2.10.0+cu130"), or $null. Separate
    # from Get-InstalledTorchTag because the xFormers pin below needs the RELEASE as well as
    # the flavor: xFormers publishes one wheel per exact torch patch (2.9.0 -> 0.0.33.post1,
    # 2.9.1 -> 0.0.33.post2, 2.10.0 -> 0.0.34), not per minor.
    function Get-InstalledTorchVersion {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.__version__)'
        if (-not $probe.Ok) { return $null }
        $torchVer = $probe.Output.Trim()
        if (-not $torchVer) { return $null }
        return $torchVer
    }

    # ── xFormers must match the torch BUILD, not just the torch version ──
    # xformers/_C.pyd is linked against one exact (torch, CUDA) pair. Loaded beside any
    # other pair torch.ops.load_library raises, and xformers/_cpp_lib.py turns that into a
    # log warning rather than an error -- so the import "succeeds" and memory-efficient
    # attention, SwiGLU and the sparse ops are all silently gone. PyPI publishes only the
    # CUDA-12.8 flavour, which is why a cu130 install that lets pip resolve xformers ends up
    # reporting "xFormers was built for PyTorch 2.10.0+cu128 (you have 2.10.0+cu130)".
    #
    # Resolve from the same index the torch install used, so UNSLOTH_TORCH_INDEX_URL /
    # UNSLOTH_TORCH_INDEX_FAMILY / UNSLOTH_PYTORCH_MIRROR keep working unchanged. Every row
    # below was HEAD-verified live on download.pytorch.org and its xformers/cpp_lib.json read
    # back, e.g. cu130/xformers-0.0.34 reports {"torch": "2.10.0+cu130"}. Keep in step with
    # _XFORMERS_WHEEL_VERSIONS in studio/backend/utils/wheel_utils.py and the matrix in
    # tests/python/test_windows_xformers_wheel_match.py.
    #
    # Deliberately NOT a floor-and-let-pip-pick: the cu130 index also serves
    # xformers-0.0.35, whose extension is built for torch 2.10.0 while its metadata only
    # asks for torch>=2.10, so a resolver is free to pair it with a torch it cannot load.
    #
    # torch 2.7.0 (xFormers 0.0.30) is deliberately absent: it predates the stable-ABI
    # switch, publishes one wheel per interpreter and stops at cp312, and this installer
    # defaults to Python 3.13 -- so the row would resolve to a wheel that does not exist.
    $script:XformersWheelVersions = @{
        "2.7.1"  = @{ "cu126" = "0.0.31.post1"; "cu128" = "0.0.31.post1" }
        "2.8.0"  = @{ "cu126" = "0.0.32.post2"; "cu128" = "0.0.32.post2"; "cu129" = "0.0.32.post2" }
        "2.9.0"  = @{ "cu126" = "0.0.33.post1"; "cu128" = "0.0.33.post1"; "cu130" = "0.0.33.post1" }
        "2.9.1"  = @{ "cu126" = "0.0.33.post2"; "cu128" = "0.0.33.post2"; "cu130" = "0.0.33.post2" }
        "2.10.0" = @{ "cu126" = "0.0.34";       "cu128" = "0.0.34";       "cu130" = "0.0.34" }
        # Stable-ABI era: 0.0.35 targets torch 2.10+ and upstream guarantees it works on
        # any later release, so one row per torch covers 2.11 onward with the same wheel.
        "2.11.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
        "2.12.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
        "2.13.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
    }

    # The stable-ABI era in one place: every torch STRICTLY ABOVE the floor resolves to this
    # wheel, which is compiled against the floor release and works on any later one by
    # upstream's own guarantee. The exact rows above still win.
    #
    # An exact-key table alone refuses the PATCH releases -- 2.10.1, 2.11.1, 2.12.1 are all
    # supported builds, and none of them can be listed here, because they are published
    # after this script ships. Each one silently left the machine with no xFormers.
    $script:XformersStableAbiFloor = "2.10.0"
    $script:XformersStableAbiVersion = "0.0.35"
    $script:XformersStableAbiFamilies = @("cu126", "cu128", "cu130")

    # "2.12.1" -> [version]2.12.1, or $null for a dev/rc/nightly string, which names no
    # released torch and must not be swept into the era comparison.
    function ConvertTo-TorchReleaseVersion {
        param([string]$Release)
        if (-not $Release -or -not [regex]::IsMatch($Release, '^\d+(\.\d+)*$')) { return $null }
        try { return [version]$Release } catch { return $null }
    }

    # xFormers version built for exactly ($TorchVersion, $CudaTag), or $null when that pair
    # has no published wheel -- in which case we install nothing rather than a mismatch.
    function Get-XformersWheelVersion {
        param([string]$TorchVersion, [string]$CudaTag)
        if (-not $TorchVersion -or -not $CudaTag) { return $null }
        # "2.10.0+cu130" -> "2.10.0"; a dev/rc suffix has no wheel and must miss the table.
        $release = ($TorchVersion -split '\+', 2)[0].Trim()
        if ($script:XformersWheelVersions.ContainsKey($release)) {
            $byFamily = $script:XformersWheelVersions[$release]
            if ($byFamily.ContainsKey($CudaTag)) { return $byFamily[$CudaTag] }
            return $null
        }
        # Not listed: above the stable-ABI floor the answer is known without a row.
        $parsed = ConvertTo-TorchReleaseVersion $release
        if ($null -eq $parsed) { return $null }
        if ($parsed -gt [version]$script:XformersStableAbiFloor -and $script:XformersStableAbiFamilies -contains $CudaTag) {
            return $script:XformersStableAbiVersion
        }
        return $null
    }

    # The torch build the SELECTED wheel records in its cpp_lib.json, which is what
    # Get-InstalledXformersBuild reads back. For an exact-era wheel that is the resident
    # torch; for the stable-ABI wheel it is the floor release it was compiled against, so
    # comparing against the resident torch marked a perfectly good 0.0.35 as mismatched and
    # force-reinstalled it on every run.
    function Get-XformersExpectedTorchBuild {
        param([string]$Version, [string]$TorchVersion, [string]$CudaTag)
        if ($Version -eq $script:XformersStableAbiVersion) {
            return "$($script:XformersStableAbiFloor)+$CudaTag"
        }
        return $TorchVersion
    }

    # The interpreter tag in the wheel FILENAME: 0.0.31..0.0.34 ship one cp39-abi3 wheel,
    # 0.0.35 switched to py39-none (a packaging change -- the extension never bound the
    # CPython ABI). Unknown releases return $null so a direct URL is never guessed.
    # Mirrors _XFORMERS_FILENAME_PYTHON_TAGS in studio/backend/utils/wheel_utils.py.
    function Get-XformersFilenamePythonTag {
        param([string]$Version)
        $parsed = ConvertTo-TorchReleaseVersion (($Version -split '[^0-9.]', 2)[0].TrimEnd('.'))
        if ($null -eq $parsed) { return $null }
        if ($parsed -ge [version]"0.0.31" -and $parsed -le [version]"0.0.34") { return "cp39-abi3" }
        if ($parsed -eq [version]"0.0.35") { return "py39-none" }
        return $null
    }

    # What the RESIDENT xformers actually is: "<version> <torch-it-was-built-for>", e.g.
    # "0.0.34 2.10.0+cu128", or $null when xformers is absent or carries no build metadata.
    # BOTH halves are needed. The version alone cannot tell cu126 from cu128 from cu130 --
    # all three publish the same version string -- and the build tag alone cannot tell
    # 0.0.34 from 0.0.35, which report the same torch. Read from disk rather than
    # "import xformers" so a mismatched .pyd cannot log its own warning into the probe.
    function Get-InstalledXformersBuild {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        $code = 'import importlib.metadata as m,importlib.util,json,os;s=importlib.util.find_spec(''xformers'');l=(list(s.submodule_search_locations) if s and s.submodule_search_locations else []);p=(os.path.join(l[0],''cpp_lib.json'') if l else '''');t=(json.load(open(p))[''version''][''torch''] if p and os.path.isfile(p) else '''');v=m.version(''xformers'') if l else '''';print((v + '' '' + t) if (v and t) else '''')'
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code $code
        if (-not $probe.Ok) { return $null }
        $build = $probe.Output.Trim()
        if (-not $build) { return $null }
        return $build
    }

    # Post-install XPU runtime check. A +xpu wheel installing is not proof the GPU is usable: on
    # an old Intel compute driver torch.xpu.is_available() is False and Unsloth then dies at
    # import. Warn, never fall back -- a driver update fixes it.
    function Assert-XpuRuntimeReady {
        param([string]$PythonExe)
        # No interpreter to ask means something bigger already broke; do not blame the driver.
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $true }
        # Line-anchored so a stdout banner ahead of the answer hides nothing. Every failure,
        # timeout included, reads as not-ready and warns rather than hanging the installer.
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.xpu.is_available())'
        if ($probe.Ok -and $probe.Output -match '(?m)^\s*True\s*$') { return $true }
        substep "[WARN] PyTorch XPU is installed but torch.xpu.is_available() is False." "Yellow"
        substep "       The Intel GPU driver is most likely too old -- PyTorch XPU on Windows" "Yellow"
        substep "       needs Intel Graphics Driver 32.0.101.6739 (WHQL) or newer." "Yellow"
        substep "       Update the driver, then re-run. See:" "Yellow"
        substep "       https://unsloth.ai/docs/get-started/install/intel" "Yellow"
        return $false
    }

    # ── Torch release preservation (twin of install.sh's _previous_torch_pin, PR 7250): keep the
    # previous venv's torch RELEASE across a re-run when it falls inside the freshly chosen constraint
    # window; flavor follows the new index. Opt out with UNSLOTH_TORCH_UPGRADE=1. ──

    # Parse a probed torch.__version__ into a normalized stable release, or $null.
    # Strips ONLY the +local tag; anchored numeric match so dev/rc/alpha/garbage never pin.
    function ConvertTo-TorchNumericRelease {
        param([string]$TorchVersion)
        if ([string]::IsNullOrWhiteSpace($TorchVersion)) { return $null }
        $publicBase = ($TorchVersion.Trim() -split '\+', 2)[0]
        if ($publicBase -notmatch '^(\d+)\.(\d+)(?:\.(\d+))?$') { return $null }
        try {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($Matches[3]) { $patch = [int]$Matches[3] } else { $patch = 0 }
            $normalized = New-Object System.Version($major, $minor, $patch)
        } catch { return $null }
        return [pscustomobject]@{
            PublicBase = $publicBase; Major = $major; Minor = $minor; Patch = $patch; Version = $normalized
        }
    }

    # True when a release falls inside a "torch>=A,<B" range. Fails closed on any
    # other constraint shape (exact pins, bare torch) -- preservation then defers.
    function Test-TorchReleaseInWindow {
        param(
            [Parameter(Mandatory = $true)]$Release,
            [Parameter(Mandatory = $true)][string]$Constraint
        )
        if ($Constraint -notmatch '^torch>=(\d+(?:\.\d+){0,2}),<(\d+(?:\.\d+){0,2})$') { return $false }
        $floor = ConvertTo-TorchNumericRelease $Matches[1]
        $ceiling = ConvertTo-TorchNumericRelease $Matches[2]
        if (-not $floor -or -not $ceiling) { return $false }
        return ($Release.Version -ge $floor.Version -and $Release.Version -lt $ceiling.Version)
    }

    # The kept-release trio for a previously installed torch, or $null when nothing
    # should be kept (no/unstable version, UNSLOTH_TORCH_UPGRADE=1, outside the final
    # route window -- a raised ROCm floor correctly rejects an older release).
    # Exact-release pin, matching install.sh's _previous_torch_pin; companions pair
    # to the kept minor (torchaudio no longer exact-pins torch).
    function Get-PreviousTorchPin {
        param(
            [string]$TorchVersion,
            # Named -Constraint (like Test-TorchReleaseInWindow): the Windows port keeps no shell-style
            # constraint variable, and its structural tests forbid that token appearing in install.ps1.
            [Parameter(Mandatory = $true)][string]$Constraint
        )
        if ($env:UNSLOTH_TORCH_UPGRADE -eq '1') { return $null }
        $release = ConvertTo-TorchNumericRelease $TorchVersion
        if (-not $release) { return $null }
        if (-not (Test-TorchReleaseInWindow -Release $release -Constraint $Constraint)) { return $null }
        if ($release.Major -ne 2) { return $null }
        $visionMinor = $release.Minor + 15
        return [pscustomobject]@{
            Release    = $release
            TorchSpec  = "torch==$($release.PublicBase)"
            VisionSpec = "torchvision==0.$visionMinor.*"
            AudioSpec  = "torchaudio==2.$($release.Minor).*"
        }
    }

    # An explicit pin is authoritative: the AMD ROCm reroute below must not rewrite it (e.g. a deliberate cpu pin on an AMD host).
    $TorchIndexPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) -or `
                        (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY))
    $TorchIndexUrl = Get-TorchIndexUrl

    # Intel XPU reroute. Must run AFTER the Get-TorchIndexUrl call above, or that call overwrites
    # it. An explicit pin still wins, like the AMD ROCm reroute below.
    if ($script:IsIntelXpu -and -not $TorchIndexPinned -and -not $SkipTorch) {
        $XpuBaseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        $TorchIndexUrl = "$XpuBaseUrl/xpu"
        substep "PyTorch XPU (SYCL) wheels will be installed from $(Remove-IndexUrlCredentials $TorchIndexUrl)"
    }

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
        # $archFamilyMap is defined above the Intel scan (the scan needs it too).
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
            "gfx1152" = "torch>=2.11.0,<2.12.0"
        }
        # Companion ranges track the torch ceiling so pip resolves a consistent trio on AMD's per-arch index (each published independently). Mirrors setup.ps1 / install_python_stack.py; bump all three together for 2.12.x.
        $torchvisionFloorMap = @{
            "gfx1201" = "torchvision>=0.26.0,<0.27.0"; "gfx1200" = "torchvision>=0.26.0,<0.27.0"
            "gfx1151" = "torchvision>=0.26.0,<0.27.0"; "gfx1150" = "torchvision>=0.26.0,<0.27.0"
            "gfx1152" = "torchvision>=0.26.0,<0.27.0"
        }
        $torchaudioFloorMap = @{
            "gfx1201" = "torchaudio>=2.11.0,<2.12.0"; "gfx1200" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1151" = "torchaudio>=2.11.0,<2.12.0"; "gfx1150" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1152" = "torchaudio>=2.11.0,<2.12.0"
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
        } elseif ($ROCmUnsupportedGfxArch) {
            substep "AMD GPU ($ROCmUnsupportedGfxArch) has no ROCm PyTorch wheels Unsloth installs -- falling back to CPU-only PyTorch" "Yellow"
        } else {
            substep "AMD GPU detected but arch unknown -- falling back to CPU-only PyTorch" "Yellow"
        }
    }

    # A gfx*/rocm pin skips the auto-reroute above, but the generic CPU/CUDA install below (torch>=2.4,<2.11) would pull a known-bad wheel on the gfx115x/gfx120x/rocm>=7.2 indexes (the _grouped_mm bug). Route a pinned ROCm index through the ROCm path.
    if ($TorchIndexPinned -and -not $ROCmIndexUrl -and -not $SkipTorch) {
        $_pinLeaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLower()
        $_pinRocm211 = $false
        # Anchor ($) so a suffixed custom leaf (rocm7.2-private) falls through to verbatim.
        if ($_pinLeaf -match '^rocm(\d+)\.(\d+)$') {
            # Only KNOWN-2.11 rocm (rocm7.2) gets the floor. Matches Test-RocmKnown211Version.
            $_pinRocm211 = ([int]$Matches[1] -eq 7 -and [int]$Matches[2] -eq 2)
        }
        # Only the 2.11-allowlist gfx arches need the floor; others publish <2.11 and stay bare.
        $_pinGfx211 = @('gfx120x-all', 'gfx1151', 'gfx1150', 'gfx1152') -contains $_pinLeaf
        if ($_pinGfx211 -or $_pinRocm211) {
            $ROCmIndexUrl = $TorchIndexUrl
            $ROCmTorchFloor = "torch>=2.11.0,<2.12.0"
            $PinnedRocmVisionSpec = "torchvision>=0.26.0,<0.27.0"
            $PinnedRocmAudioSpec = "torchaudio>=2.11.0,<2.12.0"
            substep "pinned ROCm index ($_pinLeaf) -- enforcing $ROCmTorchFloor" "Cyan"
        } elseif ($_pinLeaf -match '^gfx[0-9]' -or $_pinLeaf -match '^rocm[0-9]+(\.[0-9]+)?$') {
            # Other gfx / older rocm (<=7.1) ship torch <2.11; route via the ROCm path with bare specs. Only EXACT rocm<digits>/gfx* are families; a suffixed leaf is verbatim.
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
        Write-StudioLine ""
        if ($ROCmGfxArch) {
            # Only an unmapped arch reaches here (a mapped one set $ROCmIndexUrl above): no ROCm torch wheels for this arch (e.g. RDNA2 gfx103X) -> CPU.
            substep "Installing CPU PyTorch -- no ROCm PyTorch wheels are available for $ROCmGfxArch." "Yellow"
            substep "PyTorch (training and Transformers inference) runs on CPU on this GPU." "Yellow"
        } else {
            if ($HipSdkInstalled -and -not $HasROCm -and -not $ROCmUnsupportedGfxArch) {
                # Guarded like the gpu step above: an installed HIP SDK is the SYMPTOM on
                # these cards, so it must not outrank the arm saying why it cannot help.
                substep "Installing CPU-only PyTorch (HIP SDK found but GPU not ROCm-accessible)." "Yellow"
            } elseif ($ROCmUnsupportedGfxArch) {
                # Same words as the $ROCmGfxArch arm above, for a card whose arch we know
                # from its name rather than from a probe (unslothai#8529).
                substep "Installing CPU PyTorch -- Unsloth has no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch." "Yellow"
                substep "Unsloth training and GPU inference are unavailable on CPU torch." "Yellow"
                substep "Neither the HIP SDK nor UNSLOTH_ROCM_GFX_ARCH can give this GPU ROCm." "Yellow"
                if ((Get-HostMachineArch) -eq "arm64") {
                    # No Windows ARM64 Vulkan bundle exists, and studio/setup.ps1 throws on the
                    # variable, so the usual advice would abort the next update instead of helping.
                    substep "GGUF chat would need Vulkan here, and no Windows ARM64 Vulkan bundle is published: build llama.cpp from source, or run this on x64." "Yellow"
                } else {
                    substep 'For GPU GGUF chat through Vulkan, set $env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan"' "Yellow"
                    substep "and re-run this installer; the bundle is chosen at install time, not at launch." "Yellow"
                }
            } elseif ($ROCmGpuLabel) {
                substep "Installing CPU-only PyTorch (AMD GPU arch unknown -- install the HIP SDK" "Yellow"
                substep "or set UNSLOTH_ROCM_GFX_ARCH to enable GPU ROCm)." "Yellow"
            } elseif ($HasIntelGpu -and -not $script:IsIntelXpu) {
                substep "Intel GPU detected but not XPU-capable. Installing CPU-only PyTorch." "Yellow"
                substep "PyTorch XPU needs Intel Arc or Data Center GPU plus a current driver." "Yellow"
                substep "See: https://unsloth.ai/docs/get-started/install/intel" "Yellow"
            } else {
                substep "No NVIDIA GPU detected." "Yellow"
            }
            substep "Installing CPU-only PyTorch. If you only need GGUF chat/inference," "Yellow"
            substep "re-run with --no-torch for a faster, lighter install:" "Yellow"
            substep ".\install.ps1 --no-torch" "Yellow"
        }
        Write-StudioLine ""
    }

    # ── Install PyTorch first, then unsloth separately ──
    # Two steps because `uv pip install unsloth --torch-backend=cpu` on Windows resolves to the pre-CLI unsloth==2024.8 (no unsloth.exe); installing torch from the explicit index first avoids that solver dead-end.
    # --upgrade-package (not --upgrade) so upgrading unsloth doesn't re-resolve torch from PyPI and strip the +cuXXX suffix step 1 pinned; new deps (transformers, trl, peft) are still pulled in.
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

    # ── Freeze the installed torch trio for the with-deps unsloth install (twin of install.sh's
    # _build_unsloth_torch_overrides): a released unsloth wheel can pin an older torch, and a
    # with-deps resolve then downgrades the pinned +cuXXX/+rocm trio. Return a temp uv --overrides
    # file pinning torch/torchvision/torchaudio to their installed versions, or $null when torch is
    # absent (--no-torch) so the caller installs unchanged. Caller removes the file afterwards. ──
    function New-UnslothTorchOverridesFile {
        param([string]$PythonExe)
        if ($SkipTorch) { return $null }
        $pins = & $PythonExe -c "from importlib.metadata import version, PackageNotFoundError`nfor _p in ('torch', 'torchvision', 'torchaudio'):`n    try:`n        print(_p + '==' + version(_p))`n    except PackageNotFoundError:`n        pass" 2>$null
        $lines = @($pins | Where-Object { $_ -match '^torch' })
        if ($lines.Count -eq 0 -or $lines[0] -notmatch '^torch==') { return $null }
        # --overrides replaces any UV_OVERRIDE env file, so fold caller-supplied
        # override files in (minus their torch-trio lines) like install.sh does.
        $ovDirs = @()
        if ($env:UV_OVERRIDE) {
            foreach ($ovFile in ($env:UV_OVERRIDE -split '\s+' | Where-Object { $_ })) {
                if (Test-Path -LiteralPath $ovFile -PathType Leaf) {
                    $lines += @(Get-Content -LiteralPath $ovFile | Where-Object {
                        $_ -notmatch '^\s*torch(vision|audio)?([\s<>=!~;@[]|$)'
                    })
                    $ovDirs += (Split-Path -Parent (Convert-Path -LiteralPath $ovFile))
                }
            }
        }
        # uv resolves an override file's relative references (-r nested.txt, ./pkg.whl)
        # against THAT file's own directory, so a merged copy in %TEMP% makes uv look for
        # them beside the temp file and the install fails. Write the merge next to the
        # caller's override file instead. Falls back to %TEMP% when there is no caller
        # file (nothing relative to preserve), when several span different directories,
        # or when that directory is not writable.
        $f = $null
        $ovDirs = @($ovDirs | Sort-Object -Unique)
        if ($ovDirs.Count -eq 1) {
            try {
                $candidate = Join-Path $ovDirs[0] ("unsloth-torch-overrides-" + [guid]::NewGuid().ToString("N") + ".txt")
                New-Item -ItemType File -Path $candidate -ErrorAction Stop | Out-Null
                $f = $candidate
            } catch { $f = $null }
        }
        if (-not $f) { $f = [System.IO.Path]::GetTempFileName() }
        # UTF-8 without a BOM, not -Encoding ascii: a non-ASCII path or marker in the
        # caller's override file would otherwise be rewritten as "?" and uv would fail to
        # find the requirement. WriteAllText adds no trailing newline, so terminate the
        # last line explicitly (an unterminated file joins two requirements into one).
        [System.IO.File]::WriteAllText($f, (($lines -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
        return $f
    }

    $_desktopMinVer = if ($env:UNSLOTH_DESKTOP_BACKEND_VERSION) { $env:UNSLOTH_DESKTOP_BACKEND_VERSION.Trim() } else { "" }
    $_unslothDesktopInstallSpec = if ($_desktopMinVer) { "unsloth>=$_desktopMinVer" } else { $null }
    $_unslothReleaseInstallSpec = if ($_unslothDesktopInstallSpec) { $_unslothDesktopInstallSpec } else { "unsloth>=2026.9.2" }

    if ($_Migrated) {
        # Migrated env: force-reinstall unsloth+unsloth-zoo for a clean state, preserving existing torch/CUDA unless the flavor repair below re-lands it.
        Write-TauriLog "STEP" "Installing unsloth"
        substep "upgrading unsloth in migrated environment..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated no-torch)" { & $script:UvExe pip install --python $VenvPython --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.9.1" }
            if ($baseInstallExit -eq 0) {
                # Resolve pydantic WITH deps so pip pins pydantic-core
                # to the matching version (no-torch-runtime.txt below
                # is --no-deps). All transitive deps are torch-free.
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { & $script:UvExe pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { & $script:UvExe pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } else {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated)" { & $script:UvExe pip install --python $VenvPython --reinstall-package unsloth --reinstall-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.9.1" }
        }
        if ($baseInstallExit -ne 0) {
            Write-StudioLine "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
        }
        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand -Label "overlay local repo" { & $script:UvExe pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { & $script:UvExe pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        }
    } elseif ($TorchIndexUrl -or $ROCmIndexUrl) {
        # Bounded default trio (torch 2.11 line; wheels verified on cpu + cu126/cu128/cu130 for
        # win_amd64 with paired triton-windows 3.6), HOISTED so the release-preservation decision
        # below can use it as the default route window. torchaudio 2.11 dropped its torch pin, so a
        # bare companion can drift from a capped torch. Bump the three ceilings together with
        # install.sh's _TORCH_CEILING trio and the repair/fallback sites below (2 more literals).
        $_pinTorchSpec = "torch>=2.4,<2.12.0"
        $_pinVisionSpec = "torchvision>=0.19,<0.27.0"
        $_pinAudioSpec = "torchaudio>=2.4,<2.12.0"
        # Release preservation (twin of install.sh's _PREV_TORCH_PIN decision): evaluated after every
        # index/floor choice, incl. the ROCm reroute, so a raised floor rejects an older release. The
        # kept release is exported for setup.ps1 (UNSLOTH_KEPT_TORCH) and cleared after setup runs.
        $script:PrevTorchPin = $null
        # Internal handoff variable: always clear an inherited value first (an interrupted
        # earlier run can leak a stale pin into setup.ps1) and set it only on a fresh decision.
        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
        if (-not $SkipTorch -and $script:PrevTorchVer) {
            $_routeWindow = $_pinTorchSpec
            # The Intel XPU branch below installs its own trio (floor 2.6, its own ceiling), so
            # vet the kept release against THAT window or a kept 2.5 would be exported as a pin
            # the XPU index can never satisfy. ROCm is evaluated last so its floor still wins.
            if ((Get-TorchIndexLeafName $TorchIndexUrl) -eq "xpu") { $_routeWindow = (Get-XpuTorchSpecs -Platform (Get-VenvPlatformTag -PythonExe $VenvPython))[0] }
            if ($ROCmIndexUrl -and $ROCmTorchFloor) { $_routeWindow = $ROCmTorchFloor }
            $script:PrevTorchPin = Get-PreviousTorchPin -TorchVersion $script:PrevTorchVer -Constraint $_routeWindow
            if ($script:PrevTorchPin) {
                $env:UNSLOTH_KEPT_TORCH = $script:PrevTorchPin.Release.PublicBase
                substep "existing install has torch $script:PrevTorchVer -- keeping it (set UNSLOTH_TORCH_UPGRADE=1 to get the newest release)"
            }
        }
        if ($SkipTorch) {
            substep "skipping PyTorch (--no-torch flag set)." "Yellow"
        } elseif ($ROCmIndexUrl) {
            Write-TauriLog "STEP" "Installing PyTorch (AMD ROCm Windows)"
            substep "installing PyTorch from $(Remove-IndexUrlCredentials $ROCmIndexUrl)..."
            $torchSpec = if ($ROCmTorchFloor) { $ROCmTorchFloor } else { "torch" }
            # Pin companions to match $torchSpec; bare names can resolve an ABI-incompatible torchvision/torchaudio on AMD's per-arch index.
            $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
            $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
            # Kept-release attempt first (pin already vetted against the ROCm floor); companions follow the kept minor.
            if ($script:PrevTorchPin) {
                $_keptTorch = $script:PrevTorchPin.TorchSpec; $_keptVision = $script:PrevTorchPin.VisionSpec; $_keptAudio = $script:PrevTorchPin.AudioSpec
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (kept release)" { & $script:UvExe pip install --python $VenvPython --force-reinstall $_keptTorch $_keptVision $_keptAudio --default-index $ROCmIndexUrl }
                if ($torchInstallExit -ne 0) {
                    substep "[WARN] $_keptTorch is not installable from $(Remove-IndexUrlCredentials $ROCmIndexUrl) -- installing the newest supported release instead" "Yellow"
                    $script:PrevTorchPin = $null
                    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
                }
            }
            if (-not $script:PrevTorchPin) {
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (AMD ROCm)" { & $script:UvExe pip install --python $VenvPython --force-reinstall --default-index $ROCmIndexUrl $torchSpec $visionSpec $audioSpec }
            }
            if ($torchInstallExit -ne 0) {
                # Transient AMD-index failure: fall back to a CPU base (Unsloth setup retries ROCm). Explicit CPU index -- for a pinned ROCm index $TorchIndexUrl IS the ROCm mirror, so reusing it would just retry it.
                $CpuFallbackIndexUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { "$($env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/'))/cpu" } else { "https://download.pytorch.org/whl/cpu" }
                substep "ROCm PyTorch install failed (exit $torchInstallExit); using a CPU base, Unsloth setup retries ROCm." "Yellow"
                # --force-reinstall: a failed ROCm install can leave an unpinned ROCm
                # torch (e.g. 2.10.0+rocm on gfx110X/gfx90a) that still satisfies the CPU
                # torch>= range, so without it uv would keep the ROCm build and only swap
                # the companions -- a mismatched venv the flavor-repair block won't fix.
                # No kept-release attempt: the ROCm attempts above always resolve or clear
                # $script:PrevTorchPin first, so a pin never reaches this CPU base.
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (CPU fallback)" { & $script:UvExe pip install --python $VenvPython --force-reinstall "torch>=2.4,<2.12.0" "torchvision>=0.19,<0.27.0" "torchaudio>=2.4,<2.12.0" --default-index $CpuFallbackIndexUrl }
                if ($torchInstallExit -ne 0) {
                    Write-StudioLine "[ERROR] Failed to install PyTorch (ROCm and CPU base both failed, exit code $torchInstallExit)" -ForegroundColor Red
                    return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
                }
                # CPU base is in; drop the ROCm expectation so the flavor-repair block below won't retry the just-failed index and abort. setup.ps1 reinstalls ROCm afterwards (recomputes its own index URL).
                $ROCmIndexUrl = $null
                $ROCmTorchFloor = $null
            }
        # Keyed off the index leaf alone, like the bitsandbytes gate below: a FAMILY=xpu or URL
        # pin lands here on a host whose Intel scan never ran (a mixed NVIDIA box), and requiring
        # $script:IsIntelXpu sent it to the generic branch and its 2.4 floor.
        } elseif ((Get-TorchIndexLeafName $TorchIndexUrl) -eq "xpu") {
            # ── Intel Arc / XPU PyTorch install ──
            # XPU wheels carry their own oneAPI runtime (intel-sycl-rt et al.), published under
            # PEP 503 at https://download.pytorch.org/whl/xpu.
            Write-TauriLog "STEP" "Installing PyTorch (Intel XPU)"
            substep "installing PyTorch from $(Remove-IndexUrlCredentials $TorchIndexUrl)..."
            # Bound the trio like every other index: the xpu index serves torch past our ceiling
            # (up to 2.13.0), and torchaudio dropped its exact torch pin. The floor is 2.6, not
            # the usual 2.4: unsloth/models/_utils.py raises at import for an XPU device below
            # that. Windows on ARM has no torchaudio wheel on any index, so drop that pin here
            # rather than abort -- the pin-only route reaches this branch on an arm64 interpreter.
            $VenvPlatform = Get-VenvPlatformTag -PythonExe $VenvPython
            $_xpuSpecs = Get-XpuTorchSpecs -Platform $VenvPlatform
            $_xpuCpuSpecs = @("torch>=2.4,<2.12.0", "torchvision>=0.19,<0.27.0", "torchaudio>=2.4,<2.12.0")
            if ($VenvPlatform -eq "win-arm64") {
                substep "windows on arm: skipping torchaudio (upstream publishes no win_arm64 wheel)."
                $_xpuCpuSpecs = @("torch>=2.4,<2.12.0", "torchvision>=0.19,<0.27.0")
            }
            # Kept-release attempt first, as on the ROCm and CUDA routes: the pin was vetted
            # against the XPU window in the release-preservation decision above, and the kept
            # trio follows $_xpuSpecs' shape so win-arm64 still asks for no torchaudio.
            if ($script:PrevTorchPin) {
                $_keptTorch = $script:PrevTorchPin.TorchSpec; $_keptVision = $script:PrevTorchPin.VisionSpec; $_keptAudio = $script:PrevTorchPin.AudioSpec
                $_keptXpuSpecs = @($_keptTorch, $_keptVision)
                if ($_xpuSpecs.Count -ge 3) { $_keptXpuSpecs += $_keptAudio }
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (kept release)" { & $script:UvExe pip install --python $VenvPython --force-reinstall @_keptXpuSpecs --default-index $TorchIndexUrl }
                if ($torchInstallExit -ne 0) {
                    substep "[WARN] $_keptTorch is not installable from $(Remove-IndexUrlCredentials $TorchIndexUrl) -- installing the newest supported release instead" "Yellow"
                    $script:PrevTorchPin = $null
                    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
                }
            }
            if (-not $script:PrevTorchPin) {
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (Intel XPU)" { & $script:UvExe pip install --python $VenvPython --force-reinstall @_xpuSpecs --default-index $TorchIndexUrl }
            }
            if ($torchInstallExit -ne 0) {
                # Transient XPU-index failure: fall back to CPU base.
                $CpuFallbackIndexUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { "$($env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/'))/cpu" } else { "https://download.pytorch.org/whl/cpu" }
                substep "XPU PyTorch install failed (exit $torchInstallExit); using a CPU base." "Yellow"
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (CPU fallback)" { & $script:UvExe pip install --python $VenvPython --force-reinstall @_xpuCpuSpecs --default-index $CpuFallbackIndexUrl }
                if ($torchInstallExit -ne 0) {
                    Write-StudioLine "[ERROR] Failed to install PyTorch (XPU and CPU base both failed, exit code $torchInstallExit)" -ForegroundColor Red
                    return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
                }
                # Drop the XPU expectation so flavor-repair below skips the failed index (as ROCm does).
                $script:IsIntelXpu = $false
                $TorchIndexUrl = $CpuFallbackIndexUrl
            } else {
                # A WMI name match says XPU-capable, not that the runtime initializes: on a stale
                # driver unsloth/device_type.py raises NotImplementedError at import. The helper
                # warns and deliberately does NOT fall back to CPU.
                Assert-XpuRuntimeReady -PythonExe $VenvPython | Out-Null
            }
        } else {
            Write-TauriLog "STEP" "Installing PyTorch"
            # Windows on ARM lacks only torchaudio (whl/cpu win_arm64: torch 42,
            # torchvision 60, torchaudio 0), so drop that pin instead of aborting. Ask the
            # interpreter, not PROCESSOR_ARCHITECTURE; reached when no x64 Python exists.
            $VenvPlatform = ""
            try {
                $VenvPlatform = (& $VenvPython -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
            } catch { $VenvPlatform = "" }
            substep "installing PyTorch ($(Remove-IndexUrlCredentials $TorchIndexUrl))..."
            # Bound the companions to the capped torch on EVERY index, cu<digits>
            # families included: torchaudio 2.11 dropped its exact torch pin from
            # the wheel metadata, so a bare companion next to torch<2.11 can
            # resolve a mismatched 2.11.0 build. Mirrors install.sh.
            # The trio is the one hoisted above ($_pinTorchSpec / $_pinVisionSpec /
            # $_pinAudioSpec -- the torch 2.11 line), so the route window the
            # release-preservation decision vetted against and the specs installed here
            # can never drift apart.
            $_torchSpecs = @($_pinTorchSpec, $_pinVisionSpec, $_pinAudioSpec)
            if ($VenvPlatform -eq "win-arm64") {
                substep "windows on arm: skipping torchaudio (upstream publishes no"
                substep "win_arm64 wheel); torch and torchvision install normally."
                $_torchSpecs = @($_pinTorchSpec, $_pinVisionSpec)
            }
            # Kept-release attempt first (pin vetted against the leaf-gated route window); companions follow the kept minor and the win-arm64 no-torchaudio shape. Range install (the hoisted bounded trio) runs when there is no pin or the kept attempt failed.
            if ($script:PrevTorchPin) {
                $_keptTorch = $script:PrevTorchPin.TorchSpec; $_keptVision = $script:PrevTorchPin.VisionSpec; $_keptAudio = $script:PrevTorchPin.AudioSpec
                $_keptSpecs = @($_keptTorch, $_keptVision)
                if ($_torchSpecs.Count -ge 3) { $_keptSpecs += $_keptAudio }
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (kept release)" { & $script:UvExe pip install --python $VenvPython @_keptSpecs --default-index $TorchIndexUrl }
                if ($torchInstallExit -ne 0) {
                    substep "[WARN] $_keptTorch is not installable from $(Remove-IndexUrlCredentials $TorchIndexUrl) -- installing the newest supported release instead" "Yellow"
                    $script:PrevTorchPin = $null
                    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
                }
            }
            if (-not $script:PrevTorchPin) {
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch" { & $script:UvExe pip install --python $VenvPython @_torchSpecs --default-index $TorchIndexUrl }
            }
            if ($torchInstallExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to install PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
            }
        }

        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (no-torch)" { & $script:UvExe pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.9.1" }
            if ($baseInstallExit -eq 0) {
                # Same pydantic-with-deps trick as the migrated branch.
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { & $script:UvExe pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { & $script:UvExe pip install --python $VenvPython --no-deps -r $NoTorchReq }
                }
            }
        } elseif ($StudioLocalInstall) {
            # Freeze the installed torch trio so this with-deps resolve can't downgrade the pinned +cuXXX/+rocm build (twin of install.sh's _build_unsloth_torch_overrides).
            $script:TorchOverridesFile = New-UnslothTorchOverridesFile -PythonExe $VenvPython
            if ($script:TorchOverridesFile) {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (local)" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth --overrides $script:TorchOverridesFile "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.9.1" }
                Remove-Item -LiteralPath $script:TorchOverridesFile -Force -ErrorAction SilentlyContinue
                $script:TorchOverridesFile = $null
            } else {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (local)" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.9.1" }
            }
        } else {
            $_unslothPkg = if ($PackageName -eq "unsloth" -and $_unslothDesktopInstallSpec) { $_unslothDesktopInstallSpec } else { $PackageName }
            # Freeze the installed torch trio (see above) so the with-deps unsloth resolve can't strip the +cuXXX/+rocm suffix.
            $script:TorchOverridesFile = New-UnslothTorchOverridesFile -PythonExe $VenvPython
            if ($script:TorchOverridesFile) {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth --overrides $script:TorchOverridesFile -- "$_unslothPkg" }
                Remove-Item -LiteralPath $script:TorchOverridesFile -Force -ErrorAction SilentlyContinue
                $script:TorchOverridesFile = $null
            } else {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth -- "$_unslothPkg" }
            }
        }
        if ($baseInstallExit -ne 0) {
            Write-StudioLine "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
            return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
        }

        if ($StudioLocalInstall) {
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand -Label "overlay local repo" { & $script:UvExe pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { & $script:UvExe pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        }
    } else {
        # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (auto torch backend)" { & $script:UvExe pip install --python $VenvPython "unsloth-zoo>=2026.9.1" "$_unslothReleaseInstallSpec" --torch-backend=auto }
            if ($baseInstallExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
            }
            substep "overlaying local repo (editable)..."
            $overlayExit = Invoke-InstallCommand -Label "overlay local repo" { & $script:UvExe pip install --python $VenvPython -e $RepoRoot --no-deps }
            if ($overlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay local repo (exit code $overlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay local repo (exit code $overlayExit)" $overlayExit)
            }
            substep "overlaying unsloth-zoo from git main..."
            $zooOverlayExit = Invoke-InstallCommandRetry -Label "overlay unsloth-zoo (git main)" { & $script:UvExe pip install --python $VenvPython --no-deps --reinstall-package unsloth-zoo "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo" }
            if ($zooOverlayExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to overlay unsloth-zoo (exit code $zooOverlayExit)" $zooOverlayExit)
            }
        } else {
            $_unslothPkg = if ($PackageName -eq "unsloth" -and $_unslothDesktopInstallSpec) { $_unslothDesktopInstallSpec } else { $PackageName }
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (auto torch backend)" { & $script:UvExe pip install --python $VenvPython --torch-backend=auto -- "$_unslothPkg" }
            if ($baseInstallExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to install unsloth (exit code $baseInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install unsloth (exit code $baseInstallExit)" $baseInstallExit)
            }
        }
    }

    # ── Intel XPU: bitsandbytes must carry XPU kernels ──
    # unsloth/bnb_availability.py binds cgemv_4bit_inference_fp16/bf16 for device_type "xpu" and
    # only bitsandbytes' XPU library exports those, so a wheel without it turns 4-bit QLoRA off.
    # 0.48.2 is the first win_amd64 build carrying that library, but the floor is 0.50.0, the AMD
    # paths' floor: <=0.49.2 NaNs at 4-bit decode on AMD, and an Arc card can sit next to a
    # Radeon. unsloth's own floor (>=0.45.5) lets a MIGRATED venv keep an older wheel, so run
    # after that install for the last word. --no-deps (torch/numpy are in), and never the curated
    # unsloth[intel-gpu-torch*] extra: it pins torch to a single +xpu wheel URL, unpinning the
    # bounded trio, and carries a preview bitsandbytes wheel uv refuses.
    # Keyed off the index leaf, not $script:IsIntelXpu: a FAMILY=xpu pin on a non-Intel host
    # skips the XPU branch above yet still installs +xpu torch. The CPU fallback there rewrites
    # $TorchIndexUrl, so a failed XPU install reads as "cpu" here. Best-effort: a failure warns.
    if (-not $SkipTorch -and (Get-TorchIndexLeafName $TorchIndexUrl) -eq "xpu") {
        substep "installing bitsandbytes with Intel XPU kernels..."
        $bnbXpuExit = Invoke-InstallCommandRetry -Label "install bitsandbytes (Intel XPU)" { & $script:UvExe pip install --python $VenvPython --no-deps "bitsandbytes>=0.50.0" }
        if ($bnbXpuExit -ne 0) {
            substep "[WARN] could not install an XPU-capable bitsandbytes (exit $bnbXpuExit); 4-bit QLoRA may be unavailable." "Yellow"
        }
    }

    $installedPackageVersion = (& $VenvPython -c "
import sys
try:
    from studio.install_manifest import installed_version_probe
except Exception:
    # --package installs something that does not ship studio/. Report what the
    # old probe would have, rather than claiming the version is unknown.
    from importlib.metadata import PackageNotFoundError, version
    try:
        print(version(sys.argv[1]))
    except PackageNotFoundError:
        sys.exit(1)
    sys.exit(0)
installed, conflict = installed_version_probe(sys.argv[1])
print(installed)
sys.exit(2 if conflict else (0 if installed else 1))
" $PackageName 2>$null | Out-String).Trim()
    $_installedPackageVersionExit = $LASTEXITCODE
    if ($_installedPackageVersionExit -eq 2) {
        substep "duplicate metadata found for $PackageName; the dependency pass will repair it" "Cyan"
    } elseif ($_installedPackageVersionExit -eq 0 -and $installedPackageVersion) {
        step $PackageName "$installedPackageVersion installed"
    } else {
        substep "[WARN] installed $PackageName version could not be determined" "Yellow"
    }

    # ── Enforce the installed torch flavor matches the detected GPU build. PEP 440 ignores the +cpu/+cuXXX/+rocm local label in a version range, so uv keeps a stale torch+cpu against a CUDA index and setup.ps1 loops on "cpu != required cuXXX". Reinstall the right triplet when a GPU build is expected: CUDA from $TorchIndexUrl, ROCm from $ROCmIndexUrl (a PEP 503 index uv resolves via --default-index). --no-torch / CPU-only hosts are no-ops. ──
    if (-not $SkipTorch) {
        $expectedTorchTag = Get-ExpectedTorchFlavorTag -TorchIndexUrl $TorchIndexUrl -ROCmIndexUrl $ROCmIndexUrl
        if ($expectedTorchTag -and $expectedTorchTag -ne 'cpu') {
            $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
            if ($installedTorchTag -and $installedTorchTag -ne $expectedTorchTag) {
                if ($expectedTorchTag -eq 'rocm' -and $ROCmIndexUrl) {
                    # AMD: a migrated venv can keep a stale CPU torch the fresh ROCm path would have force-reinstalled. Repair from the same repo.amd.com index.
                    $rocmSpec = if ($ROCmTorchFloor) { $ROCmTorchFloor } else { "torch" }
                    # Pin companions like the fresh ROCm path (bare names can pull an ABI-incompatible torchvision/torchaudio from the per-arch index).
                    $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
                    $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
                    # Kept-release substitution (twin of install.sh's _install_torch_default_index honoring _PREV_TORCH_PIN): honor the preserved torch when the pin survived the E-decision (already floor-vetted); restore the range specs and retry if it isn't installable here.
                    $_rocmKept = $false
                    if ($script:PrevTorchPin) {
                        $_origRocmSpec = $rocmSpec; $_origVisionSpec = $visionSpec; $_origAudioSpec = $audioSpec
                        $rocmSpec = $script:PrevTorchPin.TorchSpec; $visionSpec = $script:PrevTorchPin.VisionSpec; $audioSpec = $script:PrevTorchPin.AudioSpec
                        $_rocmKept = $true
                    }
                    substep "PyTorch flavor mismatch (installed $installedTorchTag, need ROCm) -- reinstalling correct build..." "Yellow"
                    $torchFixExit = Invoke-InstallCommand -Label "reinstall PyTorch (ROCm)" { & $script:UvExe pip install --python $VenvPython --force-reinstall --default-index $ROCmIndexUrl $rocmSpec $visionSpec $audioSpec }
                    if ($torchFixExit -ne 0 -and $_rocmKept) {
                        substep "[WARN] $rocmSpec is not installable from $(Remove-IndexUrlCredentials $ROCmIndexUrl) -- installing the newest supported release instead" "Yellow"
                        $rocmSpec = $_origRocmSpec; $visionSpec = $_origVisionSpec; $audioSpec = $_origAudioSpec
                        $script:PrevTorchPin = $null
                        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
                        $torchFixExit = Invoke-InstallCommand -Label "reinstall PyTorch (ROCm)" { & $script:UvExe pip install --python $VenvPython --force-reinstall --default-index $ROCmIndexUrl $rocmSpec $visionSpec $audioSpec }
                    }
                    if ($torchFixExit -ne 0) {
                        Write-StudioLine "[ERROR] Failed to reinstall PyTorch with the correct ROCm build (exit code $torchFixExit)" -ForegroundColor Red
                        return (Exit-InstallFailure "Failed to reinstall PyTorch (ROCm) (exit code $torchFixExit)" $torchFixExit)
                    }
                    $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
                } elseif ($expectedTorchTag -ne 'rocm') {
                    # CUDA: stale +cpu (or wrong cuXXX) against a CUDA index -> reinstall triplet with the default 2.11-line trio (ceiling bump site, see the hoisted trio above).
                    $_fixTorchSpec = "torch>=2.4,<2.12.0"; $_fixVisionSpec = "torchvision>=0.19,<0.27.0"; $_fixAudioSpec = "torchaudio>=2.4,<2.12.0"
                    # Kept-release substitution (twin of install.sh's _install_torch_default_index honoring _PREV_TORCH_PIN): honor the preserved torch when the pin survived the E-decision; restore the range specs and retry if it isn't installable here. The --reinstall-package triplet stays on both attempts.
                    $_cudaKept = $false
                    if ($script:PrevTorchPin) {
                        $_origFixTorchSpec = $_fixTorchSpec; $_origFixVisionSpec = $_fixVisionSpec; $_origFixAudioSpec = $_fixAudioSpec
                        $_fixTorchSpec = $script:PrevTorchPin.TorchSpec; $_fixVisionSpec = $script:PrevTorchPin.VisionSpec; $_fixAudioSpec = $script:PrevTorchPin.AudioSpec
                        $_cudaKept = $true
                    }
                    substep "PyTorch flavor mismatch (installed $installedTorchTag, need $expectedTorchTag) -- reinstalling correct build..." "Yellow"
                    # Same trio builder as the XPU install above: a migrated win-arm64 venv
                    # reaches THIS path, and asking for a torchaudio the index has no wheel for
                    # fails the repair before setup.ps1's ARM-aware fallback. When a kept pin is
                    # active the scalars above already hold it, so the range trio is read back
                    # from $_origFix*Spec -- that is what the restore path below returns to.
                    $_fixSpecs = @($_fixTorchSpec, $_fixVisionSpec, $_fixAudioSpec)
                    $_origFixSpecs = if ($_cudaKept) { @($_origFixTorchSpec, $_origFixVisionSpec, $_origFixAudioSpec) } else { $_fixSpecs }
                    if ($expectedTorchTag -eq 'xpu') {
                        $_origFixSpecs = Get-XpuTorchSpecs -Platform (Get-VenvPlatformTag -PythonExe $VenvPython)
                        # Keep the kept trio in the XPU trio's shape (no torchaudio on win-arm64).
                        if ($_cudaKept) {
                            $_fixSpecs = @($_fixTorchSpec, $_fixVisionSpec)
                            if ($_origFixSpecs.Count -ge 3) { $_fixSpecs += $_fixAudioSpec }
                        } else {
                            $_fixSpecs = $_origFixSpecs
                        }
                    }
                    $torchFixExit = Invoke-InstallCommand -Label "reinstall PyTorch ($expectedTorchTag)" { & $script:UvExe pip install --python $VenvPython @_fixSpecs --default-index $TorchIndexUrl --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio }
                    if ($torchFixExit -ne 0 -and $_cudaKept) {
                        substep "[WARN] $_fixTorchSpec is not installable from $(Remove-IndexUrlCredentials $TorchIndexUrl) -- installing the newest supported release instead" "Yellow"
                        $_fixTorchSpec = $_origFixTorchSpec; $_fixVisionSpec = $_origFixVisionSpec; $_fixAudioSpec = $_origFixAudioSpec
                        $_fixSpecs = $_origFixSpecs
                        $script:PrevTorchPin = $null
                        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
                        $torchFixExit = Invoke-InstallCommand -Label "reinstall PyTorch ($expectedTorchTag)" { & $script:UvExe pip install --python $VenvPython @_fixSpecs --default-index $TorchIndexUrl --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio }
                    }
                    if ($torchFixExit -ne 0) {
                        Write-StudioLine "[ERROR] Failed to reinstall PyTorch with the correct CUDA build (exit code $torchFixExit)" -ForegroundColor Red
                        return (Exit-InstallFailure "Failed to reinstall PyTorch ($expectedTorchTag) (exit code $torchFixExit)" $torchFixExit)
                    }
                    $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
                }
            }
            # Safety net (incl. AMD): GPU build expected but still CPU -> warn loudly.
            if ($installedTorchTag -eq 'cpu') {
                Write-StudioLine ""
                Write-StudioLine "  [WARN] PyTorch is CPU-only but a $expectedTorchTag GPU build was expected for this machine." -ForegroundColor Yellow
                Write-StudioLine "  [WARN] Training and GPU inference will run on CPU until this is fixed." -ForegroundColor Yellow
                Write-StudioLine "  [WARN] Re-run this installer, or reinstall the GPU build manually for your GPU." -ForegroundColor Yellow
            }
        }
    }

    # ── Pin xFormers to the wheel built for the torch that is actually installed ──
    # See $script:XformersWheelVersions above for why a version floor is not enough.
    # Runs AFTER the flavor repair so it reads the final torch, and keys off THAT torch
    # rather than off $TorchIndexUrl, which the repair does not always reconcile (it is
    # skipped when the expected tag is 'cpu' or unrecognised, so a migrated venv can hold
    # a +cu128 torch while the index leaf says /cpu, and /cpu serves no usable xFormers).
    # xFormers is an optional accelerator, so every failure here warns and the install
    # continues on torch SDPA; and when no wheel matches we install NOTHING, because
    # installing a mismatched one is the bug being fixed. UNSLOTH_SKIP_XFORMERS=1 opts out.
    if (-not $SkipTorch -and $env:UNSLOTH_SKIP_XFORMERS -ne "1") {
        $_xfTorchVersion = Get-InstalledTorchVersion -PythonExe $VenvPython
        $_xfCudaTag = ConvertTo-TorchFlavorTag $_xfTorchVersion
        # cu<digits> only: cpu / rocm / xpu torch has no xFormers wheel on any index.
        # IsMatch, not -match, so this does not clobber $Matches for the enclosing scope.
        if ($_xfTorchVersion -and [regex]::IsMatch([string]$_xfCudaTag, '^cu\d+$')) {
            $_xfVersion = Get-XformersWheelVersion -TorchVersion $_xfTorchVersion -CudaTag $_xfCudaTag
            if (-not $_xfVersion) {
                # "not in the table", which also covers families we have no row for (cu118,
                # cu124) as well as torch releases upstream never built against.
                substep "no matching xFormers wheel for torch $_xfTorchVersion -- skipping it (attention falls back to torch SDPA)."
            } elseif ((Get-InstalledXformersBuild -PythonExe $VenvPython) -eq "$_xfVersion $(Get-XformersExpectedTorchBuild -Version $_xfVersion -TorchVersion $_xfTorchVersion -CudaTag $_xfCudaTag)") {
                substep "xFormers $_xfVersion already matches torch $_xfTorchVersion."
            } else {
                # How to fetch it, in two cases.
                #
                # An EXPLICIT index pin is authoritative and stays whole. Its final component
                # is not required to be the CUDA family: a documented full-URL override can be
                # an authenticated PEP 503 mirror, and comparing its leaf threw exactly that
                # index away -- the one that had just supplied the resident CUDA torch --
                # leaving an air-gapped host unable to reach any wheel at all.
                #
                # Otherwise install the DIRECT wheel URL rather than resolving a version off
                # an index. --default-index does not make an index exclusive (uv's --index /
                # UV_INDEX are used "in addition to" it), and cu126 / cu128 / cu130 publish
                # the SAME version string, so a machine with UV_INDEX set -- PyPI's CUDA-12.8
                # build, say -- could satisfy a pinned 0.0.35 from the wrong family and
                # recreate the silent extension failure this whole step exists to prevent.
                # A URL cannot be resolved anywhere else. UNSLOTH_PYTORCH_MIRROR is still
                # honoured, as it is everywhere else in this script.
                $_xfIndexUrl = $null
                $_xfWheelUrl = $null
                $_xfPyTag = Get-XformersFilenamePythonTag $_xfVersion
                $_xfWheelName = if ($_xfPyTag) { "xformers-$_xfVersion-$_xfPyTag-win_amd64.whl" } else { $null }
                # A FULL-URL override only. UNSLOTH_TORCH_INDEX_FAMILY also sets
                # $TorchIndexPinned, and routing that through the index reintroduced the very
                # hole above: cu126 / cu128 / cu130 share a version string, so a machine-level
                # UV_INDEX could satisfy the pin with the wrong family. A family pin names a
                # leaf we can build a direct URL from, so it takes the URL path.
                if ($TorchIndexUrl -and -not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
                    # Even here, prefer a URL. When the override already names a CUDA leaf
                    # (.../cu130, the documented shape) the wheel can be addressed under it
                    # directly, which no UV_INDEX can substitute for. Only an override whose
                    # leaf is not a family -- a bare PEP 503 mirror root -- has to be resolved,
                    # and that one gets the machine-level indexes cleared below.
                    $_xfOverrideLeaf = Get-TorchIndexLeafName $TorchIndexUrl
                    if ($_xfWheelName -and [regex]::IsMatch($_xfOverrideLeaf, '^cu\d+$')) {
                        $_xfWheelUrl = Join-UrlPath $TorchIndexUrl $_xfWheelName
                    } else {
                        $_xfIndexUrl = $TorchIndexUrl
                    }
                } else {
                    $_xfBase = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR } else { "https://download.pytorch.org/whl" }
                    if ($_xfWheelName) {
                        $_xfWheelUrl = Join-UrlPath $_xfBase "$_xfCudaTag/$_xfWheelName"
                    } else {
                        # Unknown filename shape: fall back to the index rather than guessing
                        # a URL that 404s.
                        $_xfIndexUrl = Join-UrlPath $_xfBase $_xfCudaTag
                    }
                }
                $_xfSource = if ($_xfWheelUrl) { $_xfWheelUrl } else { $_xfIndexUrl }
                substep "installing xFormers $_xfVersion for torch $_xfTorchVersion ($(Remove-IndexUrlCredentials $_xfSource))..."
                # --no-deps: the wheel declares torch==<exact release>, and acting on that can
                #   pull a PyPI (CUDA 12.8) torch over the CUDA build just installed.
                # --reinstall-package: cu126 / cu128 / cu130 all publish the SAME xformers
                #   version string, so a wrong-CUDA wheel is invisible to a version check and
                #   would otherwise be left in place on an upgrade of a broken install.
                # Go through $script:UvExe when something has resolved one, so this call
                # cannot be captured by a profile alias named uv. That variable is not set
                # on this branch; Get-Variable rather than a bare read so the lookup is
                # also safe under a profile's Set-StrictMode.
                $_xfUv = Get-Variable -Name 'UvExe' -Scope Script -ValueOnly -ErrorAction SilentlyContinue
                if (-not $_xfUv) { $_xfUv = 'uv' }
                $_xfExit = if ($_xfWheelUrl) {
                    Invoke-InstallCommandRetry -Label "install xFormers" { & $_xfUv pip install --python $VenvPython --no-deps --reinstall-package xformers $_xfWheelUrl }
                } else {
                    # --default-index does not make an index exclusive: uv reads UV_INDEX and
                    # UV_EXTRA_INDEX_URL "in addition to" it, and every CUDA family publishes the
                    # SAME xformers version, so a machine-level index could satisfy the pin from
                    # the wrong one. Cleared for this call only, and restored after.
                    $_xfSavedIndex = $env:UV_INDEX
                    $_xfSavedExtra = $env:UV_EXTRA_INDEX_URL
                    try {
                        $env:UV_INDEX = $null
                        $env:UV_EXTRA_INDEX_URL = $null
                        Invoke-InstallCommandRetry -Label "install xFormers" { & $_xfUv pip install --python $VenvPython --no-deps --reinstall-package xformers "xformers==$_xfVersion" --default-index $_xfIndexUrl }
                    } finally {
                        $env:UV_INDEX = $_xfSavedIndex
                        $env:UV_EXTRA_INDEX_URL = $_xfSavedExtra
                    }
                }
                if ($_xfExit -ne 0) {
                    substep "[WARN] could not install xFormers $_xfVersion (exit $_xfExit); attention falls back to torch SDPA." "Yellow"
                }
            }
        }
    }

    # ── CI only: overlay a source checkout over the package just installed ──
    # Mirrors install.sh. Not a consumer knob: no switch, absent from the usage text,
    # ignored unless UNSLOTH_CI_SOURCE_OVERLAY names a directory with a pyproject.toml.
    #
    # The clean-machine legs run THIS script from a branch but install unsloth from
    # PyPI, the consumer path, so everything Python-side (studio/setup.ps1,
    # install_python_stack.py and every requirements/constraints file they reach via
    # Path(__file__)) would be the released wheel's and a branch could not be
    # validated. The `studio setup` handoff below goes through the CLI, and an
    # editable overlay makes _PACKAGE_ROOT in unsloth_cli/commands/studio.py resolve to
    # the working tree by PEP 660 __file__, so setup.ps1 comes from this ref. NOT
    # --local: that also installs `unsloth-zoo @ git+https://...`, which genuinely needs
    # the git these legs remove; editable + --no-deps resolves and clones nothing, so it
    # survives git, cmake and MSVC all missing.
    if ($env:UNSLOTH_CI_SOURCE_OVERLAY) {
        $CiOverlayRoot = $env:UNSLOTH_CI_SOURCE_OVERLAY
        if (-not (Test-Path -LiteralPath (Join-Path $CiOverlayRoot "pyproject.toml"))) {
            Write-StudioLine "[ERROR] UNSLOTH_CI_SOURCE_OVERLAY is set to '$CiOverlayRoot' but there is no pyproject.toml there." -ForegroundColor Red
            return (Exit-InstallFailure "UNSLOTH_CI_SOURCE_OVERLAY has no pyproject.toml: $CiOverlayRoot")
        }
        substep "CI: overlaying source checkout (editable, no deps): $CiOverlayRoot"
        # Retry: the editable build fetches its backend from PyPI, same network risk.
        $CiOverlayExit = Invoke-InstallCommandRetry -Label "overlay CI source checkout" -Command { & $script:UvExe pip install --python $VenvPython --no-deps -e $CiOverlayRoot }
        if ($CiOverlayExit -ne 0) {
            return (Exit-InstallFailure "Failed to overlay the CI source checkout (exit code $CiOverlayExit)" $CiOverlayExit)
        }
    }

    # ── Run studio setup ──
    # setup.ps1 will handle installing Git, CMake, Visual Studio Build Tools,
    # CUDA Toolkit, and other dependencies automatically via winget. Node.js is
    # NOT installed via winget -- setup.ps1 uses an isolated Node it manages and
    # never touches the system Node/npm.
    Write-TauriLog "STEP" "Running studio setup"
    step "setup" "running unsloth studio setup..."
    # Tested, never executed: pip generates the console script whatever the policy says
    # about running it, so it stays the cheapest "this wheel shipped the CLI" signal.
    #
    # Cheapest, but no longer required. A policy DENIES this file and leaves it on disk,
    # so it answers there, but antivirus QUARANTINES it, deleting it out of a venv that
    # is otherwise perfectly able to run. Nothing after this point executes it -- the
    # setup handoff, the shortcuts and bin\unsloth.cmd all go through $VenvPython -- so
    # failing here would refuse to install or repair Unsloth for exactly the machines
    # this change is for. Ask the interpreter instead, and only then give up.
    $UnslothExe = Join-Path $VenvDir "Scripts\unsloth.exe"
    if (-not (Test-Path -LiteralPath $UnslothExe)) {
        $cliImportable = $false
        if (Test-Path -LiteralPath $VenvPython) {
            # The trampoline's own import, so this answers for what actually launches.
            Invoke-ManagedUnslothCli -Python $VenvPython -Arguments @("--version")
            $cliImportable = ($script:ManagedUnslothCliExit -eq 0)
        }
        if ($cliImportable) {
            substep "the generated unsloth.exe is gone (antivirus quarantine); the managed CLI still runs." "Yellow"
        } else {
            Write-TauriLog "ERROR" "unsloth CLI was not installed correctly"
            Write-StudioLine "[ERROR] unsloth CLI was not installed correctly." -ForegroundColor Red
            Write-StudioLine "        Expected: $UnslothExe" -ForegroundColor Yellow
            Write-StudioLine "        This usually means an older unsloth version was installed that does not include the Unsloth CLI." -ForegroundColor Yellow
            Write-StudioLine "        Try re-running the installer or see: https://github.com/unslothai/unsloth?tab=readme-ov-file#-quickstart" -ForegroundColor Yellow
            return (Exit-InstallFailure "unsloth CLI was not installed correctly")
        }
    }
    # This is the file the handoff actually starts, so it gets its own check rather
    # than being discovered as a launch failure halfway through setup.
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-TauriLog "ERROR" "managed Python is missing"
        Write-StudioLine "[ERROR] The managed Python interpreter is missing." -ForegroundColor Red
        Write-StudioLine "        Expected: $VenvPython" -ForegroundColor Yellow
        Write-StudioLine "        Re-run the installer to rebuild the environment." -ForegroundColor Yellow
        return (Exit-InstallFailure "managed Python is missing at $VenvPython")
    }
    # Tell setup.ps1 to skip base package installation (install.ps1 already did it)
    $env:SKIP_STUDIO_BASE = "1"
    $env:STUDIO_PACKAGE_NAME = $PackageName
    $env:UNSLOTH_NO_TORCH = if ($SkipTorch) { "true" } else { "false" }
    # The torch family THIS run settled on, for setup.ps1's preserve guard (full rationale there,
    # at $InstallerTorchTag): "a GPU wheel is in the venv" is not on its own evidence that this
    # installer put it there -- the migrated-venv arm above installs unsloth only and never
    # touches torch. Empty means "no answer": --no-torch, or a custom index whose leaf names no
    # flavor. Always assigned so a previous run in the same session cannot leak a value; 7.5+
    # keeps it present and blank, 5.1 / 7.0-7.4 remove it, and setup.ps1 treats both as unknown.
    $env:UNSLOTH_INSTALLER_TORCH_TAG = if ($SkipTorch) { "" } else {
        [string](Get-ExpectedTorchFlavorTag -TorchIndexUrl $TorchIndexUrl -ROCmIndexUrl $ROCmIndexUrl)
    }
    # Tauri desktop app bundles its own frontend — skip Node/npm/frontend build
    $env:SKIP_STUDIO_FRONTEND = if ($TauriMode) { "1" } else { "0" }
    # Always set STUDIO_LOCAL_INSTALL explicitly to avoid a stale value from a previous --local run in the same session.
    if ($StudioLocalInstall) {
        $env:STUDIO_LOCAL_INSTALL = "1"
        $env:STUDIO_LOCAL_REPO = $RepoRoot
    } else {
        $env:STUDIO_LOCAL_INSTALL = "0"
        Remove-Item Env:STUDIO_LOCAL_REPO -ErrorAction SilentlyContinue
    }
    # 'studio setup' (not 'update'): 'update' pops SKIP_STUDIO_BASE -> redundant reinstall + bypasses the PR #4667 fast-path version check. Propagate UNSLOTH_STUDIO_HOME only for env-override installs (else an inherited value misplaces llama.cpp).
    $previousUnslothStudioHome = $env:UNSLOTH_STUDIO_HOME
    $hadPreviousUnslothStudioHome = ($null -ne $previousUnslothStudioHome)
    $previousTauriMode = $env:UNSLOTH_TAURI_MODE
    $hadPreviousTauriMode = ($null -ne $previousTauriMode)
    $env:UNSLOTH_TAURI_MODE = if ($TauriMode) { "1" } else { "0" }
    if ($StudioRedirectMode -eq 'env') {
        $env:UNSLOTH_STUDIO_HOME = $StudioHome
    } else {
        Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
    }
    $studioArgs = @('studio', 'setup')
    if ($script:UnslothVerbose) { $studioArgs += '--verbose' }
    if ($WithLlamaCppDir) {
        if (-not (Test-Path -LiteralPath $WithLlamaCppDir -PathType Container)) {
            Write-StudioLine "[ERROR] --with-llama-cpp-dir path does not exist: $WithLlamaCppDir" -ForegroundColor Red
            return (Exit-InstallFailure "--with-llama-cpp-dir path does not exist.")
        }
        $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR = (Resolve-Path -LiteralPath $WithLlamaCppDir).Path
    }
    $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED = "1"
    # Hand the venv interpreter to setup.ps1 so it reuses the Python we resolved instead of re-probing the system (which can trip over an unsupported `python` 3.14 or a Store stub on PATH). setup.ps1 Test-Path-guards this before use.
    $env:UNSLOTH_SETUP_PYTHON = Join-Path $VenvDir "Scripts\python.exe"
    # Installer already owns the runtime mutex; the child inherits it rather
    # than deadlocking trying to reacquire it.
    $previousSetupRuntimeGateHandoff = $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF
    $hadPreviousSetupRuntimeGateHandoff = ($null -ne $previousSetupRuntimeGateHandoff)
    $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"
    # The proxy defaults kept out of the discarded profile table, for the duration of the child
    # only. setup.ps1 runs with -NoProfile and downloads on its own; see the prologue.
    $previousProxyHandoff = $env:_UNSLOTH_PS_PROXY_DEFAULTS
    $hadPreviousProxyHandoff = ($null -ne $previousProxyHandoff)
    # Set even when there is nothing to hand over: its ABSENCE is how the CLI recognises a
    # standalone update and goes looking through the user's profiles. An empty object says "the
    # installer looked, and there is none".
    $env:_UNSLOTH_PS_PROXY_DEFAULTS =
        if ($UnslothProxyHandoffJson) { $UnslothProxyHandoffJson } else { '{}' }
    # Forward the arch this run resolved. Both scripts scan WMI, so a scan that answers here but
    # not there leaves setup expecting cpu torch against the ROCm wheels just installed: it
    # reports "needs repair", the installer rolls back, and the app retries that forever.
    #
    # PRIVATE, not UNSLOTH_ROCM_GFX_ARCH: install_llama_prebuilt.py reads that one back as
    # _manual to decide whether a forwarded --rocm-gfx outranks its own probe, and this scan is
    # the weaker of the two anyway (first AMD adapter, no visible-device mask, no shadowing-iGPU
    # repick, all of which setup.ps1 applies). So setup consumes it only after its own probes
    # come up empty, and nested installers never see it.
    $previousRocmGfxHandoff = $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF
    $hadPreviousRocmGfxHandoff = ($null -ne $previousRocmGfxHandoff)
    if ($ROCmGfxArch) {
        $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF = $ROCmGfxArch
    } else {
        # Cleared, not left alone: an inherited value from an outer process is not this run's
        # answer, and handing it down would forward an arch nothing here detected.
        Remove-Item Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF -ErrorAction SilentlyContinue
    }
    try {
        Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs
        $setupExit = $script:ManagedUnslothCliExit
    } finally {
        if ($hadPreviousUnslothStudioHome) {
            $env:UNSLOTH_STUDIO_HOME = $previousUnslothStudioHome
        } else {
            Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
        }
        if ($hadPreviousTauriMode) {
            $env:UNSLOTH_TAURI_MODE = $previousTauriMode
        } else {
            Remove-Item Env:UNSLOTH_TAURI_MODE -ErrorAction SilentlyContinue
        }
        if ($hadPreviousSetupRuntimeGateHandoff) {
            $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = $previousSetupRuntimeGateHandoff
        } else {
            Remove-Item Env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF -ErrorAction SilentlyContinue
        }
        if ($hadPreviousRocmGfxHandoff) {
            $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF = $previousRocmGfxHandoff
        } else {
            Remove-Item Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF -ErrorAction SilentlyContinue
        }
        if ($hadPreviousProxyHandoff) {
            $env:_UNSLOTH_PS_PROXY_DEFAULTS = $previousProxyHandoff
        } else {
            Remove-Item Env:_UNSLOTH_PS_PROXY_DEFAULTS -ErrorAction SilentlyContinue
        }
        # ...and the copy this function holds goes with it, rather than sitting in the frame for
        # the rest of a long install.
        $UnslothProxyHandoffJson = $null
        Remove-Item Env:UNSLOTH_LOCAL_LLAMA_CPP_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_SETUP_PYTHON -ErrorAction SilentlyContinue
    }
    # $null, not a code: Application Control refused to create the process, so there is
    # no exit code to report. Checked first because in PowerShell $null -ne 0 is true,
    # and the branch below would print "exit code " with nothing after it.
    if ($null -eq $setupExit) {
        return (Exit-InstallFailure (Write-ApplicationControlBlocked -Path $VenvPython))
    }
    # Release-preservation handoff done: setup.ps1 has consumed UNSLOTH_KEPT_TORCH (if any). Clear it so a
    # later 'studio update' in the same session doesn't re-pin an old release, and warn (never abort) if the
    # kept torch series changed out from under us during setup.
    if ($script:PrevTorchPin) {
        $_keptSeries = "$($script:PrevTorchPin.Release.Major).$($script:PrevTorchPin.Release.Minor)"
        $_nowVer = Get-InstalledTorchVersionRaw -PythonExe $VenvPython
        $_nowRelease = ConvertTo-TorchNumericRelease $_nowVer
        if ($_nowRelease -and "$($_nowRelease.Major).$($_nowRelease.Minor)" -ne $_keptSeries) {
            Write-StudioLine "[WARN] kept torch $($script:PrevTorchVer) but the environment now has torch $_nowVer" -ForegroundColor Red
        }
    }
    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
    if ($setupExit -ne 0) {
        if (-not $TauriMode) {
            Write-StudioLine "[ERROR] unsloth studio setup failed (exit code $setupExit)" -ForegroundColor Red
        }
        return (Exit-InstallFailure "unsloth studio setup failed (exit code $setupExit)" $setupExit)
    }
    Clear-TauriInstallError "studio setup completed"

    # ── Expose `unsloth` via a shim dir containing only unsloth.exe (NOT the venv Scripts dir, which also holds python.exe/pip.exe and would hijack the system interpreter). Hardlink preferred, copy fallback if cross-volume/non-NTFS. ──
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
    # Fatal preflight outside the lock-handling try/catch -- a directory at the shim path must not be downgraded to "Continuing with the existing launcher", or the install finishes with no usable shim.
    if (Test-Path -LiteralPath $ShimExe -PathType Container) {
        Write-StudioLine "[ERROR] Cannot create unsloth launcher: $ShimExe is a directory." -ForegroundColor Red
        Write-StudioLine "        Move or remove it manually, then re-run the installer." -ForegroundColor Yellow
        throw "Cannot create unsloth launcher: $ShimExe is a directory."
    }
    # try/catch: if unsloth.exe is locked (Unsloth running), keep the old shim.
    $shimUpdated = $false
    try {
        if (Test-Path -LiteralPath $ShimExe) { Remove-Item -LiteralPath $ShimExe -Force -ErrorAction Stop }
        try {
            # New-Item -ItemType HardLink doesn't accept -LiteralPath in any PowerShell version, so use -Path; wildcards in $ShimExe (brackets in custom roots) glob-expand here and fall through to the Copy-Item -LiteralPath fallback below.
            New-Item -ItemType HardLink -Path $ShimExe -Target $UnslothExe -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -LiteralPath $UnslothExe -Destination $ShimExe -Force -ErrorAction Stop # fallback: copy
        }
        $shimUpdated = $true
    } catch {
        if (Test-Path -LiteralPath $ShimExe) {
            Write-StudioLine "[WARN] Could not refresh unsloth launcher at $ShimExe." -ForegroundColor Yellow
            Write-StudioLine "       This usually means a running 'unsloth studio' process still holds the file open." -ForegroundColor Yellow
            Write-StudioLine "       Close Unsloth and re-run the installer to pick up the latest launcher." -ForegroundColor Yellow
            Write-StudioLine "       Continuing with the existing launcher." -ForegroundColor Yellow
        } else {
            Write-StudioLine "[WARN] Could not create unsloth launcher at $ShimExe" -ForegroundColor Yellow
            Write-StudioLine "       $($_.Exception.Message)" -ForegroundColor Yellow
            # The interpreter, not $UnslothExe: this arm is reached on machines whose
            # policy denies the generated console script, where that advice cannot work.
            Write-StudioLine "       Until the next successful install, start Unsloth with:" -ForegroundColor Yellow
            Write-StudioLine "       & '$VenvPython' -I -m unsloth_cli studio -p 8888" -ForegroundColor Yellow
        }
    }
    # Companion launcher for machines whose policy denies the generated .exe. PATHEXT
    # resolves .EXE ahead of .CMD, so bare `unsloth` still picks the .exe wherever it
    # runs; where it is denied, `unsloth.cmd` is what the user has left.
    Write-UnslothCmdShim -ShimDir $ShimDir -PythonPath $VenvPython

    # Add to PATH only when a launcher exists. Either one counts: if the .exe could not
    # be hardlinked or copied, or a policy's quarantine removed it, the .cmd beside it
    # is a working launcher and its directory still belongs on PATH. Env-mode:
    # session-only export, no registry change (workspace path may be deleted later).
    # Test-UnslothCmdShimFile, not Test-Path: Write-UnslothCmdShim warns and leaves an
    # unwritable file alone, so a foreign bin\unsloth.cmd in a custom root can survive
    # this run. Counting it would put its directory on PATH and advertise someone
    # else's command as the policy-safe launcher.
    $ShimCmd = Join-Path $ShimDir "unsloth.cmd"
    $ShimUsable = (Test-Path -LiteralPath $ShimExe -PathType Leaf) -or
                  (Test-UnslothCmdShimFile -Path $ShimCmd)
    $pathAdded = $false
    if ($ShimUsable) {
        if ($StudioRedirectMode -ne 'env') {
            $pathAdded = Add-ToUserPath -Directory $ShimDir -Position 'Prepend'
        }
    }
    if ($shimUpdated -and $pathAdded) {
        step "path" "added unsloth launcher to PATH"
    }
    Refresh-SessionPath  # sync current session with registry
    Complete-StudioVenvRollback
    $studioVenvReplacementCommitted = $true
    Remove-StaleStudioVenvRollbacks
    } finally {
        if (-not $studioVenvReplacementCommitted) {
            Restore-StudioVenvRollback
        }
    }

    # Env-mode session export AFTER Refresh-SessionPath; otherwise a legacy
    # User PATH entry (Machine > User > current $env:Path) would win.
    if ($StudioRedirectMode -eq 'env' -and $ShimUsable) {
        $env:Path = "$ShimDir;$env:Path"
        step "path" "exported $ShimDir for this session (no registry PATH change in env-override mode)"
    }

    # ── Tauri mode: done, skip shortcuts and auto-launch ──
    if ($TauriMode) {
        Write-TauriLog "DONE" ""
        return
    }

    # New-StudioShortcuts gates the .lnk shortcuts on env-mode internally.
    New-StudioShortcuts -ManagedPythonPath $VenvPython

    # Warn if another 'unsloth' wins on PATH (different venv, system pip). Content-hash equality (Get-FileHash) so hardlinks/symlinks/identical copies of the shim don't false-trigger; CommandType Application restricts the probe to real executables.
    try {
        $_pathCmd = Get-Command unsloth -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($_pathCmd) {
            $_pathExe = $_pathCmd.Source
            # This installer's own unsloth.cmd is not a foreign 'unsloth'. It only wins
            # when the .exe beside it is gone, and hashing it against a PE would always
            # differ and always warn.
            $_ourCmdShim = Join-Path $ShimDir "unsloth.cmd"
            $_isOurCmdShim = $_pathExe -and
                ($_pathExe.TrimEnd('\', '/') -ieq $_ourCmdShim.TrimEnd('\', '/'))
            $_installedHash = (Get-FileHash -LiteralPath $UnslothExe -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
            $_pathHash      = (Get-FileHash -LiteralPath $_pathExe   -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
            if ((-not $_isOurCmdShim) -and $_installedHash -and $_pathHash -and ($_installedHash -ne $_pathHash)) {
                Write-StudioLine ""
                step "warning" "another 'unsloth' wins on PATH:" "Yellow"
                substep $_pathExe
                substep "this installer's binary is at:"
                substep $UnslothExe
                substep "to use this install, call the absolute path above,"
                substep "or put its dir earlier on PATH."
                # That absolute path is the generated console script, which is exactly
                # what an Application Control policy denies; name the launcher that works.
                if (Test-ShimLaunchBlocked -Path $ShimExe) {
                    substep "(Windows blocks that file here; use $_ourCmdShim instead)"
                }
                Write-StudioLine ""
            }
        }
    } catch {
        # Diagnostic only; never block install on a probe failure.
    }

    # Interactive terminals: prompt before starting Unsloth (unless the caller disabled it); non-interactive (CI, Docker): just print instructions.
    $IsInteractive = (-not $SkipAutostart) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        Write-StudioLine ""
        $reply = Read-Host "  Start Unsloth Studio now? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($reply) -or $reply -match '^[Yy]') {
            # Keep both locks until the process exists: a second installer can
            # then take them, but its scan sees Unsloth before it mutates.
            $_runtimeGateHandoff = $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF
            try {
                $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"
                # Through the interpreter, not the generated console script: the
                # autostart must not be the one step an Application Control policy
                # can still refuse after a clean install.
                $studioAutoStartProcess = Start-Process -FilePath $VenvPython `
                    -ArgumentList (Get-ManagedUnslothCliCommandLine -Arguments @("studio", "-p", "8888")) `
                    -NoNewWindow -PassThru
                # This inherits the private %TEMP% and outlives the installer, so it,
                # not $PID, is the owner the next sweep must see.
                if ($null -ne $studioAutoStartProcess) {
                    Set-StudioPrivateTempOwner -OwnerProcessId $studioAutoStartProcess.Id
                }
            } finally {
                if ($null -eq $_runtimeGateHandoff) {
                    Remove-Item Env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF -ErrorAction SilentlyContinue
                } else {
                    $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = $_runtimeGateHandoff
                }
            }
        } else {
            step "launch" "to start later, run:"
            # PATHEXT resolves .EXE before .CMD, so bare `unsloth` picks the generated
            # console script, which a denying machine cannot run. Probed, not assumed:
            # an unaffected machine must see the line it has always seen.
            if (Test-UnslothCmdShimPreferred -ShimExe $ShimExe -ShimCmd $ShimCmd) {
                substep "unsloth.cmd studio -p 8888"
                substep "(the generated unsloth.exe is not usable on this machine;"
                substep " unsloth.cmd beside it runs the same CLI through the managed Python)"
            } else {
                substep "unsloth studio -p 8888"
            }
            substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
            substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
            Write-StudioLine ""
        }
    } else {
        step "launch" "manual commands:"
        # Single-quote the printed paths so $-vars / backticks in custom roots don't reparse when the user pastes the command.
        $_actLiteral = "'" + ((Join-Path $VenvDir "Scripts\Activate.ps1") -replace "'", "''") + "'"
        # Activating the venv puts Scripts\unsloth.exe first and PATHEXT prefers .EXE,
        # so every bare `unsloth` below is unusable where the policy denies it. Same
        # probe, same reason: unaffected machines must see unchanged text.
        $_shimBlocked = Test-UnslothCmdShimPreferred -ShimExe $ShimExe -ShimCmd $ShimCmd
        $_bareLaunch = if ($_shimBlocked) { "unsloth.cmd studio -p 8888" } else { "unsloth studio -p 8888" }
        if ($StudioRedirectMode -eq 'env') {
            # Env-mode skips registry PATH; print the absolute shim path. The .cmd
            # beside it goes through the interpreter, so it is the one that works where
            # the policy denies the generated .exe -- named only when it is needed.
            $_shimLeaf = if ($_shimBlocked) { "bin\unsloth.cmd" } else { "bin\unsloth.exe" }
            $_shim = Join-Path $StudioHome $_shimLeaf
            $_shimLiteral = "'" + ($_shim -replace "'", "''") + "'"
            substep "& $_shimLiteral studio -p 8888"
            substep "or activate env first:"
            substep "& $_actLiteral"
            substep $_bareLaunch
        } else {
            substep "& $_actLiteral"
            substep $_bareLaunch
        }
        if ($_shimBlocked) {
            substep "(the generated unsloth.exe is not usable on this machine;"
            substep " unsloth.cmd runs the same CLI through the managed Python)"
        }
        substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
        substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
        Write-StudioLine ""
    }
    } finally {
        for ($i = $studioRuntimeMutexes.Count - 1; $i -ge 0; $i--) {
            Exit-StudioInstallMutex -Mutex $studioRuntimeMutexes[$i]
        }
        Exit-StudioInstallMutex -Mutex $studioInstallMutex
        # Matters for `irm | iex`, where these are the user's own session variables.
        Restore-StudioTempEnvironment
    }
    if ($null -ne $studioAutoStartProcess) {
        $studioAutoStartProcess.WaitForExit()
    }
}

# $null so a value left by an earlier run in the same session never reaches the finally
# below: under `irm | iex` the script scope IS the caller's session. Twin of install.sh's
# `_UNSLOTH_TORCH_OVERRIDES=""` ahead of its traps -- only a path this run created is removed.
$script:TorchOverridesFile = $null
try {
    Install-UnslothStudio @args
} finally {
    # UNSLOTH_KEPT_TORCH is a process-scoped handoff to setup.ps1. Under `irm | iex` the
    # session outlives the installer, so a terminating exception that bypasses the in-flow
    # clears must not leave an abandoned exact pin for a later `studio setup` / `update`.
    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
    # Same for the generated uv overrides temp file (twin of install.sh's
    # _cleanup_install_temporaries): it copies the caller's inherited UV_OVERRIDE contents, so a
    # terminating error between its creation and the in-flow removal would leave those
    # requirements sitting in %TEMP%. Only paths this script created are ever set here.
    if ($script:TorchOverridesFile) {
        Remove-Item -LiteralPath $script:TorchOverridesFile -Force -ErrorAction SilentlyContinue
        $script:TorchOverridesFile = $null
    }
}
