# Unsloth Studio Installer for Windows PowerShell
#
# Usage, options and the web one-liner: see "Unsloth Studio (web UI)" in the README
# (https://github.com/unslothai/unsloth#unsloth-studio-web-ui). Not repeated here: nothing reads
# this header from inside the script, and the whole file is scanned before any of it runs.
# Why several things below are written the long way: tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
#
# The web entry point cannot forward arguments, so it takes options as environment variables set
# beforehand (UNSLOTH_NO_TORCH, UNSLOTH_SKIP_AUTOSTART, UNSLOTH_ISOLATE_UV_CACHE,
# UNSLOTH_INSTALL_NO_ROLLBACK, UNSLOTH_PYTHON, UNSLOTH_STUDIO_HOME); a local run takes the
# equivalent flags (--no-torch, --skip-autostart, --isolated-uv-cache, --no-rollback,
# --python, --local).
#
# Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME (alias) > $USERPROFILE\.unsloth\studio
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Remove-Item can throw a terminating error for an unresolvable path that -ErrorAction
# cannot suppress, so guard and catch everything (temp names can be 8.3 aliases).
# Top level, NOT inside Install-UnslothStudio: the outer finally runs after that function
# returns (tests/studio/test_unsloth_torch_override.ps1 pins the placement).
function Remove-UnslothTempFileQuietly {
    param([string]$Path)
    if (-not $Path) { return }
    try {
        if (-not (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) { return }
        Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch { }
}

function Install-UnslothStudio {
    $ErrorActionPreference = "Stop"

    # First: under `irm | iex` the script scope is the caller's session.
    $script:StudioUvMarkerSaved = $false
    $script:StudioUvMarkerExisted = $false
    $script:StudioUvMarkerPrevious = $null
    # One flag for both rollbacks, so an interruption cannot restore only half.
    $script:StudioInstallCommitted = $false

    # The user's PowerShell profile has already run, and the piped web entry cannot re-launch
    # with -NoProfile, so each way a profile can reach in is cut below. Strict mode Off: this
    # script reads legitimately unset variables. Scoped to here and below.
    Set-StrictMode -Off

    # A profile setting 'None' is fatal on PowerShell 7, which preloads no modules. First,
    # because the proxy handoff below needs ConvertTo-Json.
    $PSModuleAutoLoadingPreference = 'All'

    # Keep proxy keys: on locked-down hosts they may be the only route out. IsMatch, not
    # -match, so no $Matches leaks.
    $_UnslothKeptDefaults = @{}
    foreach ($_UnslothDefaultKey in @($PSDefaultParameterValues.Keys)) {
        # IgnoreCase: 'invoke-webrequest:proxy' binds the same parameter.
        if ($_UnslothDefaultKey -is [string] -and
            [regex]::IsMatch(
                $_UnslothDefaultKey,
                ':Proxy(Credential|UseDefaultCredentials)?$',
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            $_UnslothKeptDefaults[$_UnslothDefaultKey] = $PSDefaultParameterValues[$_UnslothDefaultKey]
        }
    }
    # A profile default like 'Start-Process:WindowStyle' would rebind every launch here.
    # No scope qualifier: shadows the caller's table for this scope only.
    $PSDefaultParameterValues = $_UnslothKeptDefaults

    # 5.1 redraws the Invoke-WebRequest progress bar on every read, which dominates download
    # time (41s vs 0.08s for python.org's installer). Unqualified, so the caller keeps theirs.
    $ProgressPreference = 'SilentlyContinue'

    # Kept proxies go to studio/setup.ps1 as JSON in _UNSLOTH_PS_PROXY_DEFAULTS (variables do
    # not cross processes). Credentials do not travel.
    $_UnslothProxyHandoff = @{}
    foreach ($_UnslothDefaultKey in @($_UnslothKeptDefaults.Keys)) {
        $_UnslothDefaultValue = $_UnslothKeptDefaults[$_UnslothDefaultKey]
        # [uri] is the form the parameter actually takes and serializes to its own string.
        if ($_UnslothDefaultValue -is [uri]) {
            $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultValue.AbsoluteUri
        } elseif ($_UnslothDefaultValue -is [string] -or $_UnslothDefaultValue -is [bool]) {
            $_UnslothProxyHandoff[$_UnslothDefaultKey] = $_UnslothDefaultValue
        } elseif ($_UnslothDefaultValue -is [scriptblock]) {
            # Evaluate a script-block default here and hand over the result, never the code.
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
    # Function-local: under a piped run this is the caller's session and the value may carry
    # credentials. Module-qualified so a profile alias cannot replace ConvertTo-Json.
    $UnslothProxyHandoffJson =
        if ($_UnslothProxyHandoff.Count -gt 0) {
            $_UnslothProxyHandoff | Microsoft.PowerShell.Utility\ConvertTo-Json -Compress
        }
        else { $null }

    # PowerShell 7 only: if a profile enables it, native non-zero exits would throw past
    # Exit-InstallFailure.
    $PSNativeCommandUseErrorActionPreference = $false

    # Reset per invocation: under `irm | iex` $script: is the caller's session.
    $script:UvExe = 'uv'
    $script:UvInstallDestDir = $null
    # Same reason; this caches a machine policy decision.
    $script:ShimLaunchBlockedCache = $null
    $script:ShimLaunchBlockedPath = $null

    $script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq "1")

    # Same fix as studio/setup.ps1: via Tauri -> Rust -> powershell.exe, 5.1 inherits
    # PowerShell 7's module paths first and cannot load its own modules. Deliberately not
    # restored: an interactive run ends with Unsloth in the foreground.
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

    # Same UTF-8 invariant as studio/setup.ps1; must precede the first write.
    $_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false
    try {
        [Console]::OutputEncoding = $_UnslothUtf8NoBom
    } catch {
        # No console: the setter drops the cached writer before throwing, so bind UTF-8 writers.
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

    # Resolved once: it picks the sink in Write-StudioLine. Same probe as studio/setup.ps1.
    $script:StudioStdoutRedirected = $false
    try { $script:StudioStdoutRedirected = [Console]::IsOutputRedirected } catch { }

    # 5.1's Write-Host uses the OEM code page, and the desktop app decodes the pipe as UTF-8,
    # so write to the console handle when redirected and Write-Host (colour) when interactive.
    # Defined before the first write.
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
        # Machine scope first: emulated x64 says AMD64 per-process (ARCHITEW6432 is WOW64-only).
        $machineArch = ""
        try { $machineArch = [string][Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE", "Machine") } catch { $machineArch = "" }
        $signals = @($machineArch, [string]$env:PROCESSOR_ARCHITEW6432, [string]$env:PROCESSOR_ARCHITECTURE, $osArch)
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

    # WoA + NVIDIA: emulated triton cannot target the Blackwell SM. Probed for a win_arm64 CUDA wheel, never by arch.
    $script:WoaNativeCudaTorch = $false
    $script:WoaTorchIndexUrl = $null

    $script:WoaNvidiaTorchIndexUrls = if ($env:UNSLOTH_WOA_TORCH_INDEX_URL) {
        @($env:UNSLOTH_WOA_TORCH_INDEX_URL.Trim().TrimEnd('/'))
    } else {
        @(
            "https://pypi.nvidia.com/nvtorch_oot",
            "https://pypi.nvidia.com/nvtorch_oot_nightly"
        )
    }
    $script:WoaTorchAudio = $false

    # Wheels PyPI lacks for win_arm64 (pyarrow, sqlite-vec), built and signed by
    # .github/workflows/woa-wheelhouse.yml. UNSLOTH_WOA_WHEELHOUSE overrides.
    $script:WoaWheelhouse = if ($env:UNSLOTH_WOA_WHEELHOUSE) {
        $env:UNSLOTH_WOA_WHEELHOUSE.Trim().TrimEnd('/')
    } else {
        "https://github.com/unslothai/unsloth/releases/download/Windows-ARM64"
    }
    # AVAILABLE, not missing: a name absent here is EXCLUDED on ARM64 by the drop list below.
    $script:WoaPyPIProvided = @{}
    $script:WoaPyPIMatchedVersion = $null
    $script:WoaPyarrowWheel = $null
    $script:WoaPyarrowSource = $null     # "pypi" | "wheelhouse" | "local"
    $script:WoaPyarrowWheelName = $null  # the wheel the pin is written from

    function Test-ZipArchiveReadable {
        param([string]$Path)
        try {
            Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
            $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        } catch { return $false }
        try { return ($zip.Entries.Count -gt 0) } catch { return $false } finally { $zip.Dispose() }
    }

    # NVIDIA ships bare .bin artifacts; uv/pip need the tags in the filename, so rebuild it.
    function Get-WheelFileNameFromArchive {
        param([string]$Path)
        try {
            Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
            $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        } catch { return $null }
        try {
            $wheelEntry = $zip.Entries | Where-Object { $_.FullName -match '^[^/]+\.dist-info/WHEEL$' } | Select-Object -First 1
            if (-not $wheelEntry) { return $null }
            $distInfo = ($wheelEntry.FullName -split '/')[0] -replace '\.dist-info$', ''
            $reader = New-Object System.IO.StreamReader($wheelEntry.Open())
            try { $content = $reader.ReadToEnd() } finally { $reader.Dispose() }
            if ($content -notmatch '(?m)^Tag:\s*(\S+)\s*$') { return $null }
            return "$distInfo-$($Matches[1]).whl"
        } catch {
            return $null
        } finally {
            $zip.Dispose()
        }
    }

    # "$base/$leaf" would put the leaf inside a mirror's ?token=.
    function Join-UrlPath {
        param([string]$Base, [string]$Path)
        if ([string]::IsNullOrWhiteSpace($Base)) { return $Path }
        $cut = $Base.IndexOfAny([char[]]('?', '#'))
        if ($cut -lt 0) { return "$($Base.TrimEnd('/'))/$Path" }
        $head = $Base.Substring(0, $cut).TrimEnd('/')
        $tail = $Base.Substring($cut)
        return "$head/$Path$tail"
    }

    # Userinfo AND query/fragment, so an authenticated pin never leaks. Mirrors the Python side.
    function Remove-IndexUrlCredentials {
        param([string]$Url)
        # Ordinal, not culture-aware: th-TH treats "://" as ignorable and crashes Substring (#7279).
        $sep = $Url.IndexOf('://', [System.StringComparison]::Ordinal)
        if ($sep -lt 0) { return $Url }
        $scheme = $Url.Substring(0, $sep)
        $rest = $Url.Substring($sep + 3)
        $q = $rest.IndexOfAny([char[]]('?', '#'))
        if ($q -ge 0) { $rest = $rest.Substring(0, $q) }
        $slash = $rest.IndexOf('/', [System.StringComparison]::Ordinal)
        $authority = if ($slash -ge 0) { $rest.Substring(0, $slash) } else { $rest }
        $at = $authority.LastIndexOf('@', [System.StringComparison]::Ordinal)
        $host_ = if ($at -ge 0) { $authority.Substring($at + 1) } else { $authority }
        if ($slash -ge 0) { return "${scheme}://${host_}$($rest.Substring($slash))" }
        return "${scheme}://${host_}"
    }

    function Test-WoaWheelhouseIsLocal {
        param([string]$Value)
        if (-not $Value) { return $false }
        if ($Value -match '^[A-Za-z][A-Za-z0-9+.-]*://') { return $false }
        try { return (Test-Path -LiteralPath $Value -PathType Container) } catch { return $false }
    }

    # uv splits UV_OVERRIDE on whitespace, so a spaced path breaks every later uv call (#6503).
    function Get-UvSafePath {
        param([string]$Path)
        if (-not $Path -or -not $Path.Contains(" ")) { return $Path }
        try {
            $fso = New-Object -ComObject Scripting.FileSystemObject
            $short = if (Test-Path -LiteralPath $Path -PathType Container) {
                $fso.GetFolder($Path).ShortPath
            } else {
                $fso.GetFile($Path).ShortPath
            }
            # A space-free alias is not necessarily a name that resolves (#11290). This value
            # reaches UV_OVERRIDE and --find-links, so a bogus one breaks every later uv call.
            if ($short -and -not $short.Contains(" ") -and (Test-Path -LiteralPath $short -ErrorAction SilentlyContinue)) { return $short }
        } catch {}
        return $Path
    }

    # uv accepts no quoting in UV_OVERRIDE (0.10.7), so 8.3 is the only mitigation, and a volume can disable it.
    $script:WoaResolverPathsOk = $null
    function Test-WoaResolverPathsUsable {
        if ($null -ne $script:WoaResolverPathsOk) { return $script:WoaResolverPathsOk }
        try { New-Item -ItemType Directory -Force -Path $StudioHome -ErrorAction Stop | Out-Null } catch {}
        $script:WoaResolverPathsOk = -not ((Get-UvSafePath $StudioHome) -match '\s')
        if (-not $script:WoaResolverPathsOk) {
            substep "windows on arm: $StudioHome contains a space and this volume has no 8.3" "Yellow"
            substep "short name for it, which uv cannot read -- using the x64 stack instead." "Yellow"
            substep "Set UNSLOTH_STUDIO_HOME to a path without spaces for the native install." "Yellow"
        }
        return $script:WoaResolverPathsOk
    }

    # PEP 425 fields, not a substring: "*cp313-cp313*" also matches cp313-cp313t. Fields are "."-sets.
    function Test-WoaWheelTags {
        param([string]$Name, [string]$PyTag, [string]$AbiTag)
        if (-not $Name) { return $false }
        $stem = $Name -replace '(?i)\.whl$', ''
        $fields = $stem -split '-'
        if ($fields.Count -lt 5) { return $false }
        $pyTags = $fields[$fields.Count - 3] -split '\.'
        $abiTags = $fields[$fields.Count - 2] -split '\.'
        return (($pyTags -contains $PyTag) -and ($abiTags -contains $AbiTag))
    }

    # PEP 440 to the depth these floors use; a pre-release sorts BELOW its release (21.0.0rc1 fails >=21.0.0).
    function Test-WoaVersionAtLeast {
        param([string]$Version, [string]$Floor)
        $parse = {
            param([string]$v)
            if ($v -notmatch '^\s*v?(\d+(\.\d+)*)') { return $null }
            $release = $Matches[1] -split '\.' | ForEach-Object { [int]$_ }
            $post = 0
            if ($v -match '\.post(\d+)') { $post = [int]$Matches[1] }
            # a/b/rc only, and only off the release, so ".post1rc1" is not a pre-release of it.
            $pre = [bool]($v -match '(?i)^\s*v?\d+(\.\d+)*(a|b|rc)\d')
            # PEP 440 hangs .devN off what precedes it, so 0.0.22.post7.dev0 is below 0.0.22.post7.
            $dev = [int]::MaxValue
            if ($v -match '(?i)\.dev(\d+)') { $dev = [int]$Matches[1] }
            return @{ Release = @($release); Post = $post; Pre = $pre; Dev = $dev }
        }
        $a = & $parse $Version
        $b = & $parse $Floor
        if ($null -eq $a -or $null -eq $b) { return $false }
        $width = [Math]::Max($a.Release.Count, $b.Release.Count)
        for ($i = 0; $i -lt $width; $i++) {
            $x = if ($i -lt $a.Release.Count) { $a.Release[$i] } else { 0 }
            $y = if ($i -lt $b.Release.Count) { $b.Release[$i] } else { 0 }
            if ($x -ne $y) { return ($x -gt $y) }
        }
        if ($a.Pre -ne $b.Pre) { return $b.Pre }
        if ($a.Post -ne $b.Post) { return ($a.Post -gt $b.Post) }
        if ($a.Dev -ne $b.Dev) { return ($a.Dev -gt $b.Dev) }
        return $true
    }

    # The wheelhouse may BE the managed directory (offline reuse), and Copy-Item refuses a self-copy.
    function Test-WoaSamePath {
        param([string]$A, [string]$B)
        if (-not $A -or -not $B) { return $false }
        try {
            $a = [System.IO.Path]::GetFullPath($A).TrimEnd('\', '/')
            $b = [System.IO.Path]::GetFullPath($B).TrimEnd('\', '/')
        } catch { return $false }
        return $a.Equals($b, [System.StringComparison]::OrdinalIgnoreCase)
    }

    # uv and pip resolve -r/-c/-f against the file CONTAINING the line, so a moved line must rebase.
    function Resolve-WoaOverrideLine {
        param([string]$Line, [string]$BaseDir)
        if (-not $BaseDir -or $Line -match '^\s*(#|$)') { return $Line }
        # pip's inline comment is whitespace then "#": split off, or the rebase reads it as part of the path.
        $comment = ""
        if ($Line -match '^(.*?)(\s+#.*)$') { $Line = $Matches[1]; $comment = $Matches[2] }
        $abs = {
            param([string]$p)
            if (-not $p -or $p -match '^[A-Za-z][A-Za-z0-9+.-]*://' -or [System.IO.Path]::IsPathRooted($p)) { return $p }
            try { return [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($BaseDir, $p)) } catch { return $p }
        }
        if ($Line -match '^(\s*)(-r|--requirement|-c|--constraint|-f|--find-links)([=\s]+)(.+?)(\s*)$') {
            # Copied out FIRST: the -match below replaces $Matches, dropping the option token.
            $lead = $Matches[1]; $opt = $Matches[2]; $sep = $Matches[3]
            $bare = $Matches[4].Trim('"').Trim("'"); $tail = $Matches[5]
            $rebased = & $abs $bare
            if ($rebased -match '\s') { $rebased = '"' + $rebased + '"' }
            return "$lead$opt$sep$rebased$tail$comment"
        }
        # -e / --editable names a path too. Extras split off first, or GetFullPath folds ".[dev]" into the parent and leaves a directory nobody has.
        if ($Line -match '^(\s*)(-e|--editable)([=\s]+)(.+?)(\s*)$') {
            $lead = $Matches[1]; $opt = $Matches[2]; $sep = $Matches[3]
            $bare = $Matches[4].Trim('"').Trim("'"); $tail = $Matches[5]
            $extras = ""
            if ($bare -match '^(.*?)(\[[^\]]*\])$') { $bare = $Matches[1]; $extras = $Matches[2] }
            $rebased = (& $abs $bare) + $extras
            if ($rebased -match '\s') { $rebased = '"' + $rebased + '"' }
            return "$lead$opt$sep$rebased$tail$comment"
        }
        if ($Line -match '^(\s*[^\s@]+\s*@\s*)(.+?)(\s*)$') {
            $head = $Matches[1]; $target = $Matches[2]; $tail = $Matches[3]
            # PEP 508 separates a marker from a URL with whitespace before the ";": kept aside, or it would be rebased as part of the path.
            $marker = ""
            if ($target -match '^(.*?)(\s+;.*)$') { $target = $Matches[1]; $marker = $Matches[2] }
            if ($target -match '^file:(?!//)(.*)$') {
                $rebasedPath = & $abs $Matches[1]
                $uri = try { (New-Object System.Uri -ArgumentList @($rebasedPath, [System.UriKind]::Absolute)).AbsoluteUri } catch { "file:" + $rebasedPath }
                return "$head$uri$marker$tail$comment"
            }
            return "$Line$comment"
        }
        if ($Line -match '^(\s*)([^\s#;]+\.(?:whl|tar\.gz|zip))(\s*.*)$') {
            $lead = $Matches[1]; $path = $Matches[2]; $rest = $Matches[3]
            if ($path -match '[\\/]') { return "$lead" + (& $abs $path) + "$rest$comment" }
        }
        # A bare local directory is a requirement to pip and uv both; the leading dot segment is what tells it from a package name, which may not start with one.
        if ($Line -match '^(\s*)(\.{1,2}[^\s#;]*)(\s*(?:[;#].*)?)$') {
            $lead = $Matches[1]; $path = $Matches[2]; $rest = $Matches[3]
            $extras = ""
            if ($path -match '^(.*?)(\[[^\]]*\])$') { $path = $Matches[1]; $extras = $Matches[2] }
            return "$lead" + (& $abs $path) + "$extras$rest$comment"
        }
        return "$Line$comment"
    }

    # Which index a uv resolve uses, from uv.toml / pyproject.toml. Quote-aware, and a subset parser because PS 5.1 has no TOML reader.
    function Remove-WoaTomlComment {
        param([string]$Line)
        $inD = $false; $inS = $false
        for ($i = 0; $i -lt $Line.Length; $i++) {
            $c = $Line[$i]
            if ($inD) {
                if ($c -eq '\') { $i++; continue }
                if ($c -eq '"') { $inD = $false }
                continue
            }
            if ($inS) { if ($c -eq "'") { $inS = $false }; continue }
            if ($c -eq '"') { $inD = $true; continue }
            if ($c -eq "'") { $inS = $true; continue }
            if ($c -eq '#') { return $Line.Substring(0, $i) }
        }
        return $Line
    }

    function Split-WoaTomlKey {
        param([string]$Text)
        $parts = @()
        $i = 0
        while ($i -lt $Text.Length) {
            while ($i -lt $Text.Length -and [char]::IsWhiteSpace($Text[$i])) { $i++ }
            if ($i -ge $Text.Length) { return $null }
            $c = $Text[$i]
            if ($c -eq '"' -or $c -eq "'") {
                $q = $c; $i++; $sb = ""
                while ($i -lt $Text.Length -and $Text[$i] -ne $q) {
                    if ($q -eq '"' -and $Text[$i] -eq '\' -and ($i + 1) -lt $Text.Length) { $i++ }
                    $sb += $Text[$i]; $i++
                }
                if ($i -ge $Text.Length) { return $null }
                $i++
                $parts += $sb
            } else {
                $sb = ""
                while ($i -lt $Text.Length -and ([string]$Text[$i]) -match '[A-Za-z0-9_-]') { $sb += $Text[$i]; $i++ }
                if (-not $sb) { return $null }
                $parts += $sb
            }
            while ($i -lt $Text.Length -and [char]::IsWhiteSpace($Text[$i])) { $i++ }
            if ($i -lt $Text.Length) {
                if ($Text[$i] -ne '.') { return $null }
                $i++
            }
        }
        if ($parts.Count -eq 0) { return $null }
        return ,$parts
    }

    # uv's inline spelling of [[index]]: `index = [{ url = "...", default = true }]`. Returns @{ DefaultUrl; Extras }, or $null on ANY doubt; only a single-line, brace-balanced array of FLAT inline tables is read.
    # The brace characters below are built from code points: the parity tests extract a function by counting braces, so a literal one inside a string would make this body un-tested.
    function Read-WoaUvInlineIndexArray {
        param([string]$Value)
        $lb = [char]0x7B; $rb = [char]0x7D
        $v = ([string]$Value).Trim()
        if ($v -notmatch '^\[(.*)\]$') { return $null }
        $inner = $Matches[1].Trim()
        $result = @{ DefaultUrl = $null; Extras = @() }
        if (-not $inner) { return $result }
        $groups = @()
        $depth = 0; $start = -1; $inD = $false; $inS = $false
        for ($i = 0; $i -lt $inner.Length; $i++) {
            $c = $inner[$i]
            if ($inD) { if ($c -eq '\') { $i++ } elseif ($c -eq '"') { $inD = $false }; continue }
            if ($inS) { if ($c -eq "'") { $inS = $false }; continue }
            if (($c -eq '"' -or $c -eq "'") -and $depth -eq 0) { return $null }
            if ($c -eq '"') { $inD = $true; continue }
            if ($c -eq "'") { $inS = $true; continue }
            if ($c -eq $lb) { if ($depth -eq 0) { $start = $i + 1 }; $depth++; continue }
            if ($c -eq $rb) {
                $depth--
                if ($depth -lt 0) { return $null }
                if ($depth -eq 0) { $groups += $inner.Substring($start, $i - $start); $start = -1 }
                continue
            }
            if ($depth -eq 0 -and ([string]$c) -notmatch '[\s,]') { return $null }
        }
        if ($depth -ne 0 -or $inD -or $inS -or -not $groups.Count) { return $null }
        foreach ($g in $groups) {
            if ($g.IndexOf($lb) -ge 0 -or $g.IndexOf('[') -ge 0) { return $null }
            $url = $null; $isDefault = $false; $isExplicit = $false
            foreach ($pair in ($g -split ',')) {
                $pair = $pair.Trim()
                if (-not $pair) { continue }
                $eq = $pair.IndexOf('=')
                if ($eq -lt 1) { return $null }
                $k = $pair.Substring(0, $eq).Trim().Trim('"').Trim("'").ToLowerInvariant()
                $raw = $pair.Substring($eq + 1).Trim()
                $str = if ($raw -match '^"(.*)"$' -or $raw -match "^'(.*)'$") { $Matches[1] } else { $null }
                $bool = if ($raw -match '^(true|false)$') { $raw -eq 'true' } else { $null }
                if ($k -eq 'url') { if (-not $str) { return $null }; $url = $str }
                elseif ($k -eq 'default') { if ($null -eq $bool) { return $null }; $isDefault = $bool }
                # uv: an explicit index serves only packages pinned to it via [tool.uv.sources], so it is neither the default nor an extra.
                elseif ($k -eq 'explicit') { if ($null -eq $bool) { return $null }; $isExplicit = $bool }
            }
            if (-not $url) { return $null }
            # explicit AND default also removes PyPI as the default (uv docs): not modelled, so doubt.
            if ($isExplicit) { if ($isDefault) { return $null }; continue }
            if ($isDefault) { if (-not $result.DefaultUrl) { $result.DefaultUrl = $url } }
            else { $result.Extras += $url }
        }
        return $result
    }

    function Read-WoaUvTomlIndexKeys {
        param([string]$Path, [string]$Top)
        try { $lines = [System.IO.File]::ReadAllLines($Path) } catch { return $null }
        # uv pip (0.10.7): [pip] scalars beat top-level, [[index]] default = true beats both. Ranked at the end.
        $topScope = @{ NoIndex = $null; IndexUrl = $null; Extras = @() }
        $pipScope = @{ NoIndex = $null; IndexUrl = $null; Extras = @() }
        $section = ""
        # A hashtable, because an assignment inside the $flush script block would be local to it.
        $inIndex = $false; $idxUrl = $null; $idxDefault = $false; $idxExplicit = $false; $entry = @{ DefaultUrl = $null; Extras = @(); Doubt = $false }
        $indexTable = if ($Top) { "$Top.index" } else { "index" }
        $pipTable = if ($Top) { "$Top.pip" } else { "pip" }
        $flush = {
            if ($inIndex -and $idxUrl) {
                # Explicit entries are skipped, as in the inline reader; explicit AND default is doubt.
                if ($idxExplicit) { if ($idxDefault) { $entry.Doubt = $true } }
                elseif ($idxDefault) { if (-not $entry.DefaultUrl) { $entry.DefaultUrl = $idxUrl } }
                else { $entry.Extras += $idxUrl }
            }
        }
        foreach ($raw in $lines) {
            $line = (Remove-WoaTomlComment $raw).Trim()
            if (-not $line) { continue }
            if ($line -match '^\[\[(.+?)\]\]$') {
                & $flush
                $section = $Matches[1].Trim(); $inIndex = ($section -eq $indexTable); $idxUrl = $null; $idxDefault = $false; $idxExplicit = $false
                continue
            }
            if ($line -match '^\[(.+?)\]$') { & $flush; $section = $Matches[1].Trim(); $inIndex = $false; continue }
            $eq = -1; $inD = $false; $inS = $false
            for ($i = 0; $i -lt $line.Length; $i++) {
                $c = $line[$i]
                if ($inD) { if ($c -eq '\') { $i++ } elseif ($c -eq '"') { $inD = $false }; continue }
                if ($inS) { if ($c -eq "'") { $inS = $false }; continue }
                if ($c -eq '"') { $inD = $true; continue }
                if ($c -eq "'") { $inS = $true; continue }
                if ($c -eq '=') { $eq = $i; break }
            }
            if ($eq -lt 1) { continue }
            $parts = Split-WoaTomlKey $line.Substring(0, $eq)
            if (-not $parts) { continue }
            $key = $parts[-1]
            $prefix = if ($parts.Count -gt 1) { ($parts[0..($parts.Count - 2)] -join '.') } else { "" }
            $val = $line.Substring($eq + 1).Trim()
            if (-not $val) { continue }
            $str = if ($val -match '^"(.*)"$' -or $val -match "^'(.*)'$") { $Matches[1] } else { $null }
            $bool = if ($val -match '^(true|false)$') { $val -eq 'true' } else { $null }
            if ($inIndex) {
                if (-not $prefix) {
                    if ($key -eq 'url' -and $str) { $idxUrl = $str }
                    if ($key -eq 'default' -and $null -ne $bool) { $idxDefault = $bool }
                    if ($key -eq 'explicit' -and $null -ne $bool) { $idxExplicit = $bool }
                }
                continue
            }
            $scopeSection = if ($prefix) { if ($section) { "$section.$prefix" } else { $prefix } } else { $section }
            $scope = if ($scopeSection -eq $Top) { $topScope } elseif ($scopeSection -eq $pipTable) { $pipScope } else { $null }
            if ($null -eq $scope) { continue }
            if ($key -eq 'no-index' -and $null -ne $bool -and $null -eq $scope.NoIndex) { $scope.NoIndex = $bool }
            if (($key -eq 'default-index' -or $key -eq 'index-url') -and $str -and -not $scope.IndexUrl) { $scope.IndexUrl = $str }
            if ($key -eq 'extra-index-url') {
                if ($val -match '^\[(.*)\]$') {
                    foreach ($m in [regex]::Matches($Matches[1], ('"([^"]*)"' + "|'([^']*)'"))) {
                        $scope.Extras += $(if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value })
                    }
                } elseif ($str) { $scope.Extras += $str }
            }
            if ($key -eq 'index') {
                $inline = Read-WoaUvInlineIndexArray -Value $val
                if ($null -eq $inline) { return $null }
                if ($inline.DefaultUrl -and -not $entry.DefaultUrl) { $entry.DefaultUrl = $inline.DefaultUrl }
                $entry.Extras += @($inline.Extras)
            }
        }
        & $flush
        if ($entry.Doubt) { return $null }
        $noIndex = if ($null -ne $pipScope.NoIndex) { $pipScope.NoIndex } else { $topScope.NoIndex }
        $defaultIndex = if ($entry.DefaultUrl) { $entry.DefaultUrl } elseif ($pipScope.IndexUrl) { $pipScope.IndexUrl } else { $topScope.IndexUrl }
        $extras = @($entry.Extras) + @($pipScope.Extras) + @($topScope.Extras)
        return @{ NoIndex = $noIndex; DefaultIndex = $defaultIndex; ExtraIndexes = @($extras | Where-Object { $_ }) }
    }

    # uv's boolish set (uv 0.10.7 parse_boolish_environment_variable): y/yes/t/true/on/1 vs
    # n/no/f/false/off/0, case-insensitive. Trimmed to match setup.sh and
    # install_python_stack.py. ToLowerInvariant for Turkish locales.
    function Test-UvEnvFlag {
        param([string]$Name)
        $value = [string][Environment]::GetEnvironmentVariable($Name)
        return (@("1", "t", "true", "y", "yes", "on") -contains $value.Trim().ToLowerInvariant())
    }

    # pip's own rule for PIP_NO_INDEX (strtobool; empty means unset). Same literals as uv's
    # today, restated so either can change independently.
    function Test-PipEnvFlag {
        param([string]$Name)
        $value = [string][Environment]::GetEnvironmentVariable($Name)
        return (@("1", "t", "true", "y", "yes", "on") -contains $value.Trim().ToLowerInvariant())
    }

    # UV_NO_INDEX is OURS: uv defines no such environment variable (--no-index is a
    # command-line flag only). Read with uv's boolish set for consistency with UV_OFFLINE/UV_NO_CONFIG;
    # not turned into --no-index, which would change behaviour for existing users.
    function Test-NoIndexRequested {
        return (Test-UvEnvFlag "UV_NO_INDEX")
    }

    function Get-WoaUvConfigIndexPolicy {
        $result = @{ NoIndex = $false; DefaultIndex = $null; Unreadable = $false; UnreadablePath = $null; ExtraIndexes = @() }
        if (Test-UvEnvFlag "UV_NO_CONFIG") { return $result }
        $files = @()
        $cfgFile = [string][Environment]::GetEnvironmentVariable("UV_CONFIG_FILE")
        if ($cfgFile) {
            $files += @{ Path = $cfgFile; Top = "" }
        } else {
            $dir = (Get-Location).Path
            while ($dir) {
                $u = Join-Path $dir "uv.toml"; $pp = Join-Path $dir "pyproject.toml"
                if (Test-Path -LiteralPath $u -PathType Leaf) { $files += @{ Path = $u; Top = "" }; break }
                if (Test-Path -LiteralPath $pp -PathType Leaf) {
                    $txt = try { [System.IO.File]::ReadAllText($pp) } catch { "" }
                    if ($txt -match '(?m)^\s*\[+tool\.uv(\.|\])') { $files += @{ Path = $pp; Top = "tool.uv" }; break }
                }
                $parent = Split-Path -Parent $dir
                if (-not $parent -or $parent -eq $dir) { break }
                $dir = $parent
            }
            if ($env:APPDATA) { $files += @{ Path = (Join-Path $env:APPDATA "uv\uv.toml"); Top = "" } }
            if ($env:ProgramData) { $files += @{ Path = (Join-Path $env:ProgramData "uv\uv.toml"); Top = "" } }
        }
        $noIndexSet = $false
        foreach ($f in $files) {
            if (-not (Test-Path -LiteralPath $f.Path -PathType Leaf)) { continue }
            $policy = Read-WoaUvTomlIndexKeys -Path $f.Path -Top $f.Top
            if ($null -eq $policy) {
                $result.Unreadable = $true
                if (-not $result.UnreadablePath) { $result.UnreadablePath = $f.Path }
                continue
            }
            if (-not $noIndexSet -and $null -ne $policy.NoIndex) { $result.NoIndex = $policy.NoIndex; $noIndexSet = $true }
            if (-not $result.DefaultIndex -and $policy.DefaultIndex) { $result.DefaultIndex = $policy.DefaultIndex }
            $result.ExtraIndexes = @($result.ExtraIndexes) + @($policy.ExtraIndexes)
        }
        return $result
    }

    # The host, not a substring: "https://pypi.org.corp.example/simple" is not public PyPI.
    function Test-WoaUrlIsPublicPyPI {
        param([string]$Url)
        if (-not $Url) { return $false }
        try { $u = [System.Uri]$Url.Trim() } catch { return $false }
        if (-not $u.IsAbsoluteUri) { return $false }
        return ($u.Host.ToLowerInvariant() -eq "pypi.org")
    }

    # This script resolves with uv, so uv's policy is the one that counts: the UV_* variables uv actually defines, and its configuration files. pip's PIP_* variables are pip's alone; uv never reads them. UV_NO_INDEX is neither -- see Test-NoIndexRequested.
    function Test-WoaResolveReachesPyPI {
        # UV_OFFLINE really stops uv; UV_NO_INDEX is our convention and uv ignores it.
        if (Test-UvEnvFlag "UV_OFFLINE") { return $false }
        if (Test-NoIndexRequested) { return $false }
        $extraIsPyPI = $false
        foreach ($name in @("UV_INDEX", "UV_EXTRA_INDEX_URL")) {
            $list = [string][Environment]::GetEnvironmentVariable($name)
            foreach ($u in ($list -split '\s+' | Where-Object { $_ })) {
                if (Test-WoaUrlIsPublicPyPI $u) { $extraIsPyPI = $true }
            }
        }
        foreach ($name in @("UV_DEFAULT_INDEX", "UV_INDEX_URL")) {
            $url = [string][Environment]::GetEnvironmentVariable($name)
            if ($url -and ($url.Trim())) { return ($extraIsPyPI -or (Test-WoaUrlIsPublicPyPI $url)) }
        }
        # Doubt resolves to "not PyPI": that answer keeps a wheel, the other loses the package.
        $cfg = Get-WoaUvConfigIndexPolicy
        if ($cfg.Unreadable -or $cfg.NoIndex) { return $false }
        foreach ($u in @($cfg.ExtraIndexes)) { if (Test-WoaUrlIsPublicPyPI $u) { $extraIsPyPI = $true } }
        if ($cfg.DefaultIndex -and -not (Test-WoaUrlIsPublicPyPI $cfg.DefaultIndex)) { return $extraIsPyPI }
        return $true
    }

    # True when uv's own config decides the indexes and we could not read it: fatal where the caller scrubs UV_* and sets UV_NO_CONFIG, because the trio index would be the only source left.
    function Test-WoaUvIndexPolicyUnreadable {
        foreach ($name in @("UV_NO_INDEX", "UV_DEFAULT_INDEX", "UV_INDEX_URL")) {
            $v = [string][Environment]::GetEnvironmentVariable($name)
            if ($v -and $v.Trim()) { return $false }
        }
        return [bool](Get-WoaUvConfigIndexPolicy).Unreadable
    }

    # The index beside the CUDA one for torch's shared dependencies. Invoke-InstallCommand clears the inherited index settings whenever --default-index is passed, so the caller's policy is restated here.
    function Get-WoaDependencyIndexArgs {
        param([string]$Resolver = "uv")
        $pip = ($Resolver -eq "pip")
        $defaultNames = if ($pip) { @("PIP_INDEX_URL") } else { @("UV_DEFAULT_INDEX", "UV_INDEX_URL") }
        $extraNames = if ($pip) { @("PIP_EXTRA_INDEX_URL") } else { @("UV_INDEX", "UV_EXTRA_INDEX_URL") }
        # Each variable by its owner's rule, and they are not symmetric. PIP_NO_INDEX is
        # pip's and pip really reads it, so naming no index here matches what pip will
        # then do. UV_NO_INDEX is ours alone, so that arm is us honouring the operator.
        $noIndexRequested = if ($pip) { Test-PipEnvFlag "PIP_NO_INDEX" } else { Test-NoIndexRequested }
        if ($noIndexRequested) { return @() }
        $default = $null
        foreach ($name in $defaultNames) {
            $url = [string][Environment]::GetEnvironmentVariable($name)
            if ($url -and $url.Trim()) { $default = $url.Trim(); break }
        }
        $extras = @()
        foreach ($name in $extraNames) {
            $list = [string][Environment]::GetEnvironmentVariable($name)
            foreach ($u in ($list -split '\s+' | Where-Object { $_ })) { $extras += $u }
        }
        if (-not $pip -and (-not $default -or -not $extras)) {
            $cfg = Get-WoaUvConfigIndexPolicy
            if ($cfg.NoIndex) { return @() }
            # An unreadable policy with no env default: name nothing, or the PyPI fallback below would silently override whatever mirror the file configured.
            if ($cfg.Unreadable -and -not $default) { return @() }
            if (-not $default -and $cfg.DefaultIndex) { $default = $cfg.DefaultIndex }
            if (-not $extras) { $extras = @($cfg.ExtraIndexes) }
        }
        if (-not $default) { $default = "https://pypi.org/simple" }
        $indexArgs = @()
        foreach ($u in @(@($default) + @($extras) | Where-Object { $_ } | Select-Object -Unique)) {
            $indexArgs += @("--extra-index-url", $u)
        }
        return $indexArgs
    }

    # Exact tags, or a wheel that does not care (cp38-abi3, py3-none). Free-threaded has no abi3.
    function Test-WoaWheelTagsUsable {
        param([string]$Name, [string]$PyTag, [string]$AbiTag = "")
        if (-not $AbiTag) { $AbiTag = $PyTag }
        if (Test-WoaWheelTags -Name $Name -PyTag $PyTag -AbiTag $AbiTag) { return $true }
        $fields = ($Name -replace '(?i)\.whl$', '') -split '-'
        if ($fields.Count -lt 5) { return $false }
        # Free-threaded CPython has no stable ABI (CPython #111506), so abi3 is not an option there.
        if ($AbiTag -like "*t") { return $false }
        $abiTags = $fields[$fields.Count - 2] -split '\.'
        $minor = 0
        if ($PyTag -match '^cp3(\d+)$') { $minor = [int]$Matches[1] }
        foreach ($pyTag in ($fields[$fields.Count - 3] -split '\.')) {
            # abi3 is forward compatible only from what it was built against: cp314-abi3 fails on cp313.
            if (($abiTags -contains "abi3") -and ($pyTag -match '^cp3(\d+)$')) {
                if ([int]$Matches[1] -le $minor) { return $true }
            }
            if ($abiTags -contains "none") {
                if ($pyTag -match '^py3(\d*)$') {
                    if ([string]::IsNullOrEmpty($Matches[1]) -or ([int]$Matches[1] -le $minor)) { return $true }
                }
                if ($pyTag -match '^cp3(\d+)$') {
                    if ([int]$Matches[1] -le $minor) { return $true }
                }
            }
        }
        return $false
    }

    function Test-WoaPyPIWheel {
        param([string]$Project, [string]$PyTag, [string]$AbiTag = "", [string]$Floor = "", [switch]$AllowAgnostic, [switch]$Exact)
        if (-not $Project -or -not $PyTag) { return $false }
        if (-not $AbiTag) { $AbiTag = $PyTag }
        # pypi.org/simple normalises the project name; a wheel filename carries the underscored form.
        $slug = ($Project -replace '_', '-').ToLowerInvariant()
        try {
            $body = [string](Invoke-RestMethod -Uri "https://pypi.org/simple/$slug/" -UseBasicParsing -TimeoutSec 20)
        } catch { return $false }
        foreach ($match in [regex]::Matches($body, "[A-Za-z0-9_.+-]+win_arm64\.whl")) {
            $name = $match.Value
            $tagsOk = if ($AllowAgnostic) { Test-WoaWheelTagsUsable -Name $name -PyTag $PyTag -AbiTag $AbiTag }
                      else { Test-WoaWheelTags -Name $name -PyTag $PyTag -AbiTag $AbiTag }
            if (-not $tagsOk) { continue }
            $fields = ($name -replace '(?i)\.whl$', '') -split '-'
            if ($fields.Count -lt 5) { continue }
            if (($fields[0] -replace '_', '-').ToLowerInvariant() -ne $slug) { continue }
            if (-not $Floor) { $script:WoaPyPIMatchedVersion = $fields[1]; return $true }
            # -Exact: the same version, so an exact pin the wheelhouse copy satisfies is never lost to a newer upstream release it does not.
            $ok = if ($Exact) {
                (Test-WoaVersionAtLeast -Version $fields[1] -Floor $Floor) -and (Test-WoaVersionAtLeast -Version $Floor -Floor $fields[1])
            } else { Test-WoaVersionAtLeast -Version $fields[1] -Floor $Floor }
            if ($ok) {
                $script:WoaPyPIMatchedVersion = $fields[1]
                return $true
            }
        }
        return $false
    }

    # Ours is first in UV_FIND_LINKS and wins the tie; an upstream BEHIND it leaves ours staged.
    function Test-WoaWheelhouseWheelIsRedundant {
        param([string]$Name, [string]$PyTag, [string]$AbiTag = "")
        if ($Name -notlike "*win_arm64*") { return $false }
        $fields = ($Name -replace '(?i)\.whl$', '') -split '-'
        if ($fields.Count -lt 5) { return $false }
        if (-not (Test-WoaResolveReachesPyPI)) { return $false }
        if (-not (Test-WoaWheelTagsUsable -Name $Name -PyTag $PyTag -AbiTag $AbiTag)) { return $false }
        return (Test-WoaPyPIWheel -Project $fields[0] -PyTag $PyTag -AbiTag $AbiTag -Floor $fields[1] -AllowAgnostic -Exact)
    }

    function Test-WoaWheelAvailable {
        param([string]$Project, [string]$PythonMinor, [string]$AbiTag = "")
        $tag = "cp" + ($PythonMinor -replace '\.', '')
        if (-not $AbiTag) { $AbiTag = $tag }
        $pattern = "$Project-[^`"'<>\s]*?win_arm64\.whl"
        if ((Test-WoaResolveReachesPyPI) -and (Test-WoaPyPIWheel -Project $Project -PyTag $tag -AbiTag $AbiTag)) { return $true }
        if (-not $script:WoaWheelhouse) { return $false }
        if (Test-WoaWheelhouseIsLocal $script:WoaWheelhouse) {
            $local = Get-ChildItem -LiteralPath $script:WoaWheelhouse -Filter "$Project-*win_arm64.whl" -ErrorAction SilentlyContinue |
                Where-Object { Test-WoaWheelTags -Name $_.Name -PyTag $tag -AbiTag $AbiTag } | Select-Object -First 1
            return [bool]$local
        }
        try {
            $index = [string](Invoke-RestMethod -Uri (Join-UrlPath $script:WoaWheelhouse "index.txt") -UseBasicParsing -TimeoutSec 20)
            foreach ($match in [regex]::Matches($index, $pattern)) {
                if (Test-WoaWheelTags -Name $match.Value -PyTag $tag -AbiTag $AbiTag) { return $true }
            }
        } catch {}
        return $false
    }

    # constraints.txt floors ARM64 pyarrow at 21.0.0 and staging pins exactly, so a 19.x cannot resolve.
    $script:WoaPyarrowFloor = "21.0.0"
    function Test-WoaPyarrowWheelUsable {
        param([string]$Name, [string]$PyTag, [string]$AbiTag)
        # Usable, not exact: pyarrow's coming win_arm64 wheel is cp311-abi3 (apache/arrow#48539).
        if (-not (Test-WoaWheelTagsUsable -Name $Name -PyTag $PyTag -AbiTag $AbiTag)) { return $false }
        $fields = ($Name -replace '(?i)\.whl$', '') -split '-'
        if ($fields.Count -lt 5) { return $false }
        return (Test-WoaVersionAtLeast -Version $fields[1] -Floor $script:WoaPyarrowFloor)
    }

    # uv's --overrides beat a spec named on the command line (0.10.7): drop the trio only, here only.
    function New-WoaTorchStepOverrideValue {
        param([string]$Value, [string]$Dir = "")
        $result = @{ Value = $null; Temps = @() }
        if (-not $Value) { return $result }
        $files = @()
        foreach ($entry in ($Value -split '\s+' | Where-Object { $_ })) {
            $path = $entry.Trim('"').Trim("'")
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
            $kept = @()
            $dropped = $false
            foreach ($line in (Get-WoaRequirementEntries -Path $path)) {
                $name = ((($line.Line -split '[\s<>=!~;@\[]', 2)[0]).Trim() -replace '[-_.]+', '-').ToLowerInvariant()
                if (@("torch", "torchvision", "torchaudio") -contains $name) { $dropped = $true; continue }
                $kept += (Resolve-WoaOverrideLine -Line $line.Line -BaseDir $line.BaseDir)
            }
            if (-not $dropped) { $files += $path; continue }
            # Flattened, because a line dropped from an include cannot come along by reference. Recorded for deletion: it can carry an authenticated URL.
            $tmp = if ($Dir -and (Test-Path -LiteralPath $Dir -PathType Container)) {
                Join-Path $Dir ("torch-step-" + [System.IO.Path]::GetRandomFileName() + ".txt")
            } else { [System.IO.Path]::GetTempFileName() }
            [System.IO.File]::WriteAllLines($tmp, [string[]]$kept, (New-Object System.Text.UTF8Encoding($false)))
            $result.Temps += $tmp
            $files += $tmp
        }
        if (-not $files) { $result.Value = ""; return $result }
        $safe = foreach ($f in $files) { Get-UvSafePath $f }
        $result.Value = (($safe | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " ")
        return $result
    }

    function Get-WoaRequirementEntries {
        param([string]$Path, $Seen = $null, [int]$Depth = 0)
        if ($Depth -gt 8) { return @() }
        if ($null -eq $Seen) { $Seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase) }
        if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return @() }
        try { $full = Convert-Path -LiteralPath $Path } catch { return @() }
        if (-not $Seen.Add($full)) { return @() }
        try { $lines = [System.IO.File]::ReadAllLines($full) } catch { return @() }
        $dir = [System.IO.Path]::GetDirectoryName($full)
        $entries = @()
        foreach ($line in $lines) {
            if ($line -match '^\s*(?:-r|--requirement)[=\s]+(.+?)\s*$') {
                # pip needs whitespace before an inline "#": "-r a#b.txt" keeps its hash.
                $nested = ($Matches[1] -replace '\s+#.*$', '').Trim().Trim('"', "'")
                if (-not [System.IO.Path]::IsPathRooted($nested)) { $nested = Join-Path $dir $nested }
                $entries += @(Get-WoaRequirementEntries -Path $nested -Seen $Seen -Depth ($Depth + 1))
                continue
            }
            $entries += , @{ Line = $line; BaseDir = $dir }
        }
        return $entries
    }

    # UNSLOTH_PYARROW_WHEEL, then PyPI, then the wheelhouse. "" leaves native unselected.
    function Get-WoaPyarrowSource {
        param([string]$PythonMinor, [string]$AbiTag = "")
        $tag = "cp" + ($PythonMinor -replace '\.', '')
        if (-not $AbiTag) { $AbiTag = $tag }
        $script:WoaPyarrowWheelName = $null
        if ($env:UNSLOTH_PYARROW_WHEEL) {
            $_paWheel = $env:UNSLOTH_PYARROW_WHEEL
            if (Test-Path -LiteralPath $_paWheel -PathType Leaf) {
                $_paName = Split-Path -Leaf $_paWheel
                if ($_paName -notlike "*.whl") {
                    $_paFromArchive = Get-WheelFileNameFromArchive -Path $_paWheel
                    if ($_paFromArchive) { $_paName = $_paFromArchive }
                }
                if (($_paName -like "pyarrow-*") -and (Test-WoaPyarrowWheelUsable -Name $_paName -PyTag $tag -AbiTag $AbiTag) -and
                    ($_paName -like "*win_arm64.whl")) {
                    # The whole archive: a truncated download still starts with "PK".
                    if (Test-ZipArchiveReadable -Path $_paWheel) { return "local" }
                    substep "windows on arm: UNSLOTH_PYARROW_WHEEL is not a readable wheel archive -- ignoring it." "Yellow"
                } else {
                    substep "windows on arm: UNSLOTH_PYARROW_WHEEL is not a $tag win_arm64 pyarrow >= $($script:WoaPyarrowFloor) wheel -- ignoring it." "Yellow"
                }
            } else {
                substep "windows on arm: UNSLOTH_PYARROW_WHEEL does not exist -- ignoring it." "Yellow"
            }
        }
        if (Test-WoaResolveReachesPyPI) { try {
            $body = [string](Invoke-RestMethod -Uri "https://pypi.org/simple/pyarrow/" -UseBasicParsing -TimeoutSec 20)
            foreach ($match in [regex]::Matches($body, 'pyarrow-[^"''<>\s]*?win_arm64\.whl')) {
                if (Test-WoaPyarrowWheelUsable -Name $match.Value -PyTag $tag -AbiTag $AbiTag) {
                    # Recorded so the override pins THIS wheel: a newer sdist satisfies >=21.0.0.
                    $script:WoaPyarrowWheelName = $match.Value
                    return "pypi"
                }
            }
        } catch {} }
        if (Test-WoaWheelhouseIsLocal $script:WoaWheelhouse) {
            $local = Get-ChildItem -LiteralPath $script:WoaWheelhouse -Filter "pyarrow-*win_arm64.whl" -ErrorAction SilentlyContinue |
                Where-Object {
                    (Test-WoaPyarrowWheelUsable -Name $_.Name -PyTag $tag -AbiTag $AbiTag) -and
                    (Test-ZipArchiveReadable -Path $_.FullName)
                } | Select-Object -First 1
            if ($local) { return "wheelhouse" }
            return ""
        }
        try {
            $index = [string](Invoke-RestMethod -Uri (Join-UrlPath $script:WoaWheelhouse "index.txt") -UseBasicParsing -TimeoutSec 20)
            foreach ($match in [regex]::Matches($index, 'pyarrow-[^"''<>\s]*?win_arm64\.whl')) {
                if (Test-WoaPyarrowWheelUsable -Name $match.Value -PyTag $tag -AbiTag $AbiTag) { return "wheelhouse" }
            }
        } catch {}
        return ""
    }

    # ── Bound nvidia-smi so a wedged driver cannot hang the installer. ──
    function Invoke-NvidiaSmiBounded {
        param(
            [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
            [Parameter(Position = 1)][string[]]$SmiArgs = @(),
            [int]$TimeoutSec = 10,
            # Driver warnings on stderr would corrupt machine-readable --query-gpu output; the human-readable probes keep the default merge.
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

    # ── BEGIN SHARED WITH studio/setup.ps1 (Get-NvidiaSmiCandidatePaths) ──
    # Every directory nvidia-smi might be in, in the order worth trying, filtered to the ones that
    # actually hold the binary. Callers still run their own Test-NvidiaSmiHasGpu over the result:
    # existing is not the same as answering.
    #
    # Only two locations were searched before, and each addition below is a host where a real NVIDIA
    # GPU was reported absent, which sends the installer to CPU-only PyTorch:
    #
    #   SysNative       From a 32-bit process "System32" is redirected to SysWOW64, which has no
    #                   nvidia-smi. SysNative is the alias that reaches the real System32.
    #   ProgramW64      In that same process $env:ProgramFiles is "Program Files (x86)".
    #                   ProgramW6432 is the 64-bit Program Files whatever the process bitness.
    #   x86             Not the driver's own location, but a machine that once had a 32-bit
    #                   toolkit can carry a working copy there.
    #   DriverStore     Where the driver package itself lives. Present on hosts where the copy
    #                   into System32 did not happen, which is the #9255 population.
    #
    # Nothing is removed. One ordering does change, and deliberately: the two call sites disagreed
    # with each other before, one trying System32 first and the other the NVSMI directory first, and
    # a single list cannot keep both. System32 wins, because that is where the current driver puts
    # its copy; NVSMI is the legacy location and a machine carrying both can have an older binary
    # there reporting an older CUDA version. This only affects a host that has both, and only in
    # which of two working binaries answers.
    function Get-NvidiaSmiCandidatePaths {
        $dirs = @()
        # Current driver locations first, in BOTH process bitnesses, before any legacy one.
        # Both bitness spellings before $env:ProgramFiles, which in a 32-bit process is the x86 tree
        # where a stale toolkit copy can sit and answer first.
        if ($env:SystemRoot) { $dirs += (Join-Path $env:SystemRoot "System32") }
        if ($env:SystemRoot) { $dirs += (Join-Path $env:SystemRoot "SysNative") }
        if ($env:ProgramW6432) { $dirs += (Join-Path $env:ProgramW6432 "NVIDIA Corporation\NVSMI") }
        if ($env:ProgramFiles) { $dirs += (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI") }
        $pfx86 = ${env:ProgramFiles(x86)}
        if ($pfx86) { $dirs += (Join-Path $pfx86 "NVIDIA Corporation\NVSMI") }
        # Bounded on purpose. FileRepository holds every driver package the machine has ever had.
        #
        # The bound is applied to directories that ACTUALLY HOLD the binary, not to the nv* matches.
        # NVIDIA ships several packages whose names start nv (HD audio, the virtual audio device,
        # the network service), so a host with eight of those newer than the display driver would
        # otherwise spend the whole allowance on directories with no nvidia-smi in them and report
        # the machine as having none. Test-Path is a file existence check, not a process spawn; the
        # bound that matters is on the candidates handed back, since each of those costs a probe.
        try {
            $repos = @()
            if ($env:SystemRoot) {
                # Both spellings, same WOW64 reason as above. At most one resolves in any given process, so
                # this is not a doubled scan.
                $repos += (Join-Path $env:SystemRoot "System32\DriverStore\FileRepository")
                $repos += (Join-Path $env:SystemRoot "SysNative\DriverStore\FileRepository")
            }
            $packages = @()
            foreach ($repo in $repos) {
                if (-not (Test-Path -LiteralPath $repo -PathType Container)) { continue }
                $packages += @(Get-ChildItem -LiteralPath $repo -Directory -Filter "nv*" -ErrorAction SilentlyContinue |
                    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "nvidia-smi.exe") -PathType Leaf })
            }
            # The bound is on the whole DriverStore contribution rather than per root, so adding
            # the second spelling cannot double the number of probes this hands back.
            foreach ($dir in @($packages | Sort-Object LastWriteTime -Descending | Select-Object -First 8)) {
                $dirs += $dir.FullName
            }
        } catch {}
        $paths = @()
        $seen = @{}
        foreach ($dir in $dirs) {
            if ([string]::IsNullOrWhiteSpace($dir)) { continue }
            $candidate = Join-Path $dir "nvidia-smi.exe"
            # Case-insensitive, because the same directory reached two ways must not be probed twice:
            # each probe is a bounded process spawn with a wall-clock timeout behind it.
            $key = $candidate.ToLowerInvariant()
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $paths += $candidate }
        }
        return $paths
    }
    # ── END SHARED WITH studio/setup.ps1 (Get-NvidiaSmiCandidatePaths) ──

    # One list, because the presence probe and the driver-version probe must agree: a host only the first finds reports a GPU with no driver version, and the CUDA-major guard is then skipped.
    # Reset per run: under `irm | iex` the script scope IS the caller's session, so a second
    # invocation in one shell would otherwise reuse the first run's answer.
    $script:WoaNvidiaSmiProbed = $false
    $script:WoaNvidiaSmiPath = $null

    function Get-WoaNvidiaSmiPath {
        # Memoised: both callers probe, and each candidate costs a bounded process spawn.
        if ($script:WoaNvidiaSmiProbed) { return $script:WoaNvidiaSmiPath }
        $script:WoaNvidiaSmiProbed = $true
        $exe = $null
        try { $exe = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source } catch { $exe = $null }
        if ($exe) { $script:WoaNvidiaSmiPath = $exe; return $exe }
        $found = @(Get-NvidiaSmiCandidatePaths)
        if ($found.Count -eq 0) { return $null }
        # The first candidate that ANSWERS: a broken copy would decline the whole ARM64 CUDA route.
        # Soft and hard budgets as in the main detection loop; only the hard one may end empty.
        $woaDeadline = (Get-Date).AddSeconds(30)
        $woaHardDeadline = (Get-Date).AddSeconds(60)
        $woaListing = $null
        foreach ($p in $found) {
            if ($null -ne $woaListing -and (Get-Date) -gt $woaDeadline) { break }
            if ((Get-Date) -gt $woaHardDeadline) { break }
            try {
                $listing = Invoke-NvidiaSmiBounded $p @('-L')
                if (-not ($LASTEXITCODE -eq 0 -and $listing -match '(?m)^\s*GPU\s+\d+')) { continue }
                if (-not $woaListing) { $woaListing = $p }
                # Prefer a candidate that also reports the CUDA version: without it the CUDA-major guard
                # in Initialize-WoaNativeCudaTorch is skipped.
                if ((Get-Date) -gt $woaDeadline) { break }
                $banner = Invoke-NvidiaSmiBounded $p
                if ($banner -match 'CUDA(?: UMD)? Version:\s+\d+\.\d+') {
                    $script:WoaNvidiaSmiPath = $p
                    return $p
                }
            } catch {}
        }
        # A GPU listing without a version still beats no answer.
        if ($woaListing) {
            $script:WoaNvidiaSmiPath = $woaListing
            return $woaListing
        }
        # Nothing answered: fall back to the first that exists.
        $script:WoaNvidiaSmiPath = $found[0]
        return $found[0]
    }

    function Test-WoaNvidiaPresent {
        $exe = Get-WoaNvidiaSmiPath
        if (-not $exe) { return $false }
        $listing = Invoke-NvidiaSmiBounded $exe @('-L')
        return ($LASTEXITCODE -eq 0 -and $listing -match '(?m)^\s*GPU\s+\d+')
    }

    # The driver's CUDA version as @(major, minor), or $null when nvidia-smi does not say.
    function Get-WoaDriverCudaVersion {
        $exe = Get-WoaNvidiaSmiPath
        if (-not $exe) { return $null }
        $out = Invoke-NvidiaSmiBounded $exe
        # Newer drivers print "CUDA UMD Version"; accept both spellings.
        if ($out -notmatch 'CUDA(?: UMD)? Version:\s+(\d+)\.(\d+)') { return $null }
        return @([int]$Matches[1], [int]$Matches[2])
    }

    function Get-WoaDriverCudaLeaf {
        $v = Get-WoaDriverCudaVersion
        if (-not $v) { return $null }
        $major = [int]$v[0]; $minor = [int]$v[1]
        if ($major -ge 13) { return "cu130" }
        if ($major -eq 12 -and $minor -ge 8) { return "cu128" }
        if ($major -eq 12 -and $minor -ge 6) { return "cu126" }
        if ($major -ge 12) { return "cu124" }
        if ($major -ge 11) { return "cu118" }
        return $null
    }

    # A free-threaded build installs cp313t, not cp313. Unknown answers $false, keeping the GIL path.
    function Test-PythonFreeThreaded {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { return $false }
        $out = ""
        try {
            $out = (& $PythonExe -c "import sysconfig; print(1 if sysconfig.get_config_var('Py_GIL_DISABLED') else 0)" 2>$null | Out-String).Trim()
        } catch { return $false }
        return ($out -eq "1")
    }

    function Get-WoaAbiTag {
        param([string]$PythonMinor, [bool]$FreeThreaded)
        $tag = "cp" + ($PythonMinor -replace '\.', '')
        if ($FreeThreaded) { return "${tag}t" }
        return $tag
    }

    # CUDA is established POSITIVELY by requiring +cuNNN: PyPI's own win_arm64 torch carries no local version, and the stamp and +cuXXX tag identify the torch BUILD a companion belongs to.
    function Test-WoaWheelPairsWithTorch {
        param([string]$TorchVersion, [string]$OtherVersion, [string]$Project = "torchvision")
        if (-not $TorchVersion -or -not $OtherVersion) { return $false }
        $stamp = {
            param($v)
            $m = [regex]::Match($v, '(?i)\.dev(\d+)')
            if ($m.Success) { return $m.Groups[1].Value } else { return "" }
        }
        $local = {
            param($v)
            $parts = $v -split '\+', 2
            if ($parts.Count -eq 2) { return $parts[1].ToLowerInvariant() } else { return "" }
        }
        if ((& $stamp $TorchVersion) -ne (& $stamp $OtherVersion)) { return $false }
        if ((& $local $TorchVersion) -ne (& $local $OtherVersion)) { return $false }
        if (& $stamp $TorchVersion) { return $true }
        # Stable releases share an empty stamp, so they pair by offset: torchvision 0.(M+15) to 2.M.
        $rel = {
            param($v)
            $m = [regex]::Match($v, '^(\d+)\.(\d+)')
            if ($m.Success) { return @([int]$m.Groups[1].Value, [int]$m.Groups[2].Value) } else { return $null }
        }
        $t = & $rel $TorchVersion; $o = & $rel $OtherVersion
        if (-not $t -or -not $o) { return $false }
        switch (($Project -replace '[-_.]+', '-').ToLowerInvariant()) {
            "torchvision" { return (($t[0] -eq 2) -and ($o[0] -eq 0) -and ($o[1] -eq ($t[1] + 15))) }
            "torchaudio"  { return (($o[0] -eq $t[0]) -and ($o[1] -eq $t[1])) }
            default       { return $true }
        }
    }

    function Get-WoaCudaWheelVersion {
        param([string]$IndexUrl, [string]$PythonMinor, [string]$Project = "torch", [string]$AbiTag = "",
              [string]$PairWith = "", [string]$Below = "")
        if ([string]::IsNullOrWhiteSpace($IndexUrl) -or [string]::IsNullOrWhiteSpace($PythonMinor)) { return $null }
        $tag = "cp" + ($PythonMinor -replace '\.', '')
        if (-not $AbiTag) { $AbiTag = $tag }
        $projectUrl = Join-UrlPath $IndexUrl "$Project/"
        try {
            $body = [string](Invoke-RestMethod -Uri $projectUrl -UseBasicParsing -TimeoutSec 20)
        } catch {
            return $null
        }
        if ([string]::IsNullOrWhiteSpace($body)) { return $null }
        $best = $null
        $bestKey = $null
        # PEP 440 order within one numeric release: dev < a < b < rc < final, then by number.
        $prerelease = {
            param([string]$v)
            $r = ($v -split '\+', 2)[0]
            if ($r -match '\.dev(\d+)') { return @(0, [long]$Matches[1]) }
            if ($r -match '(?i)(a|b|rc)(\d+)') { return @(@{ a = 1; b = 2; rc = 3 }[$Matches[1].ToLowerInvariant()], [long]$Matches[2]) }
            return @(4, [long]0)
        }
        # -Below: only versions ranked strictly under it, so a caller can walk back from an unpaired newest.
        $belowKey = $null; $belowRank = $null
        if ($Below) {
            $belowNumeric = [regex]::Match((($Below -split '\+', 2)[0]), '^\d+(\.\d+){0,2}').Value
            try { $belowKey = [version]$belowNumeric } catch { return $null }
            $belowRank = & $prerelease $Below
        }
        foreach ($match in [regex]::Matches($body, "$Project-[^`"'<>\s]*?win_arm64\.whl")) {
            $name = $match.Value
            try { $name = [System.Uri]::UnescapeDataString($name) } catch {}
            if (-not (Test-WoaWheelTags -Name $name -PyTag $tag -AbiTag $AbiTag)) { continue }
            if ($name -notmatch '\+cu[0-9]+') { continue }
            $version = ($name -split '-')[1]
            if ($PairWith -and -not (Test-WoaWheelPairsWithTorch -TorchVersion $PairWith -OtherVersion $version -Project $Project)) { continue }
            # Numeric release only: a .dev suffix must not make a newer line look older.
            $release = ($version -split '\+', 2)[0]
            $numeric = [regex]::Match($release, '^\d+(\.\d+){0,2}').Value
            $key = $null
            try { $key = [version]$numeric } catch { continue }
            if ($belowKey) {
                if ($key -gt $belowKey) { continue }
                if ($key -eq $belowKey) {
                    $rank = & $prerelease $version
                    if (($rank[0] -gt $belowRank[0]) -or (($rank[0] -eq $belowRank[0]) -and ($rank[1] -ge $belowRank[1]))) { continue }
                }
            }
            if ($null -eq $bestKey -or $key -gt $bestKey) {
                $bestKey = $key; $best = $version
            } elseif ($key -eq $bestKey) {
                $rankNew = & $prerelease $version; $rankBest = & $prerelease $best
                if (($rankNew[0] -gt $rankBest[0]) -or (($rankNew[0] -eq $rankBest[0]) -and ($rankNew[1] -gt $rankBest[1]))) { $best = $version }
            }
        }
        return $best
    }

    function Test-WoaCudaWheel {
        param([string]$IndexUrl, [string]$PythonMinor, [string]$Project = "torch", [string]$AbiTag = "")
        return [bool](Get-WoaCudaWheelVersion -IndexUrl $IndexUrl -PythonMinor $PythonMinor -Project $Project -AbiTag $AbiTag)
    }

    function Test-WoaAudioMatchesTorch {
        param([string]$TorchVersion, [string]$AudioVersion)
        if (-not $TorchVersion -or -not $AudioVersion) { return $false }
        $shorten = {
            param($v)
            [regex]::Match((($v -split '\+', 2)[0]), '^\d+\.\d+').Value
        }
        $a = & $shorten $TorchVersion
        $b = & $shorten $AudioVersion
        return ($a -and $b -and $a -eq $b)
    }

    function Initialize-WoaNativeCudaTorch {
        param([string]$PythonMinor, [bool]$FreeThreaded = $false)
        # ABI, not just minor: cp313-cp313 for a 3.13t interpreter finds GIL wheels it cannot install.
        $_woaAbiTag = Get-WoaAbiTag -PythonMinor $PythonMinor -FreeThreaded $FreeThreaded
        $script:WoaFreeThreaded = $FreeThreaded
        $script:WoaNativeCudaTorch = $false
        $script:WoaTorchIndexUrl = $null
        $script:WoaTorchAudio = $false
        if ((Get-HostMachineArch) -ne "arm64") { return }
        if ($SkipTorch) { return }
        if ($env:UNSLOTH_WOA_NATIVE -and $env:UNSLOTH_WOA_NATIVE.Trim().ToLowerInvariant() -in @("0", "false", "no", "off")) {
            substep "windows on arm: UNSLOTH_WOA_NATIVE is off -- using the x64 path." "Yellow"
            return
        }
        if (-not (Test-WoaNvidiaPresent)) { return }
        if (-not (Test-WoaResolverPathsUsable)) { return }
        $pinned = @()
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
            $pinned += $env:UNSLOTH_TORCH_INDEX_URL.Trim().TrimEnd('/')
        } elseif (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY)) {
            $mirror = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
            $pinned += "$mirror/$($env:UNSLOTH_TORCH_INDEX_FAMILY.Trim().Trim('/'))"
        }
        $candidates = @()
        if ($pinned.Count -gt 0) {
            $candidates = $pinned
        } else {
            $mirror = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
            $driverLeaf = Get-WoaDriverCudaLeaf
            if ($driverLeaf) { $candidates += "$mirror/$driverLeaf" }
            $candidates += $script:WoaNvidiaTorchIndexUrls
        }
        $torchIndex = $null
        $_woaTorchVersion = $null
        $_woaVisionVersion = $null
        $_woaUnpairedIndex = $null
        foreach ($candidate in $candidates) {
            if (-not (Test-WoaCudaWheel -IndexUrl $candidate -PythonMinor $PythonMinor -AbiTag $_woaAbiTag -Project "torch")) { continue }
            $_woaTorchVersion = Get-WoaCudaWheelVersion -IndexUrl $candidate -PythonMinor $PythonMinor -AbiTag $_woaAbiTag
            # An index qualifies only with a torchvision build PAIRED to that torch, or a lagging nightly or partial mirror leaves torchvision resolving against a torch it was not built for.
            # The newest torch is not the only candidate: a nightly torchvision lags by days, so the newest COMPLETE pair is searched for, a few versions deep.
            $_woaVisionVersion = $null
            $_woaBacktracks = 0
            while ($_woaTorchVersion) {
                $_woaVisionVersion = Get-WoaCudaWheelVersion -IndexUrl $candidate -PythonMinor $PythonMinor -Project "torchvision" -AbiTag $_woaAbiTag -PairWith $_woaTorchVersion
                if ($_woaVisionVersion -or $_woaBacktracks -ge 5) { break }
                $_woaOlderTorch = Get-WoaCudaWheelVersion -IndexUrl $candidate -PythonMinor $PythonMinor -AbiTag $_woaAbiTag -Below $_woaTorchVersion
                if (-not $_woaOlderTorch) { break }
                substep "windows on arm: $(Remove-IndexUrlCredentials $candidate) publishes torch $_woaTorchVersion but no torchvision paired with it; trying torch $_woaOlderTorch." "Yellow"
                $_woaTorchVersion = $_woaOlderTorch
                $_woaBacktracks++
            }
            if (-not $_woaVisionVersion) {
                substep "windows on arm: $(Remove-IndexUrlCredentials $candidate) publishes torch $_woaTorchVersion but no torchvision paired with it; trying the next index." "Yellow"
                if (-not $_woaUnpairedIndex) { $_woaUnpairedIndex = $candidate }
                continue
            }
            $torchIndex = $candidate
            break
        }
        if (-not $torchIndex) {
            if ($_woaUnpairedIndex) {
                substep "windows on arm: no index pairs a torchvision with its win_arm64 CUDA torch; using the x64 stack instead." "Yellow"
            }
            return
        }
        # A CUDA 13 runtime does not run on a CUDA 12 driver (no forward compatibility on Windows), so the wheel's major is checked against the driver's.
        $_woaDriver = Get-WoaDriverCudaVersion
        if ($_woaDriver -and $_woaTorchVersion -match '\+cu(\d+)') {
            $_woaWheelMajor = [int]($Matches[1].Substring(0, $Matches[1].Length - 1))
            if ($_woaWheelMajor -gt [int]$_woaDriver[0]) {
                substep "windows on arm: $(Remove-IndexUrlCredentials $torchIndex) publishes torch $_woaTorchVersion (CUDA $_woaWheelMajor), but this driver supports CUDA $($_woaDriver[0]).$($_woaDriver[1])." "Yellow"
                substep "using the x64 stack instead. Update the NVIDIA driver for the native install." "Yellow"
                return
            }
        }
        # torch alone is not enough: datasets needs pyarrow, unbuildable here. Both decided up front.
        $pyarrowSource = Get-WoaPyarrowSource -PythonMinor $PythonMinor -AbiTag $_woaAbiTag
        if (-not $pyarrowSource) {
            substep "windows on arm: native CUDA torch is available, but no win_arm64 pyarrow wheel could be found." "Yellow"
            substep "using the x64 stack instead. Set UNSLOTH_PYARROW_WHEEL to a win_arm64 pyarrow wheel for the native install." "Yellow"
            return
        }
        # Free-threaded only: constraints.txt's win_arm64 av wheel is cp311-abi3 (CPython #111506).
        if ($FreeThreaded -and -not (Test-WoaWheelAvailable -Project "av" -PythonMinor $PythonMinor -AbiTag $_woaAbiTag)) {
            substep "windows on arm: no win_arm64 av (PyAV) wheel for this free-threaded interpreter ($_woaAbiTag)." "Yellow"
            substep "using the x64 stack instead. A GIL build of the same Python takes the native path." "Yellow"
            return
        }
        $script:WoaNativeCudaTorch = $true
        $script:WoaTorchIndexUrl = $torchIndex
        $script:WoaPyarrowSource = $pyarrowSource
        # Only when the audio wheel PAIRS with the torch one: 2.11 dropped its exact torch pin.
        $_woaAudioVersion = Get-WoaCudaWheelVersion -IndexUrl $torchIndex -PythonMinor $PythonMinor -Project "torchaudio" -AbiTag $_woaAbiTag -PairWith $_woaTorchVersion
        $script:WoaTorchAudio = Test-WoaAudioMatchesTorch -TorchVersion $_woaTorchVersion -AudioVersion $_woaAudioVersion
        # Kept so the install pins what the probe SELECTED: unsafe-best-match spans ALL indexes.
        $script:WoaTorchWheelVersion = $_woaTorchVersion
        $script:WoaAudioWheelVersion = $_woaAudioVersion
        $script:WoaVisionWheelVersion = $_woaVisionVersion
        # Read off the wheel, not the URL: a mirror of the prerelease channel need not say "nightly".
        $script:WoaTorchIsPrerelease = [bool]($_woaTorchVersion -match '(?i)\d(a|b|rc)\d|\.dev\d')
        if ($_woaAudioVersion -and -not $script:WoaTorchAudio) {
            substep "windows on arm: this index has torchaudio $_woaAudioVersion but torch $_woaTorchVersion; skipping torchaudio."
        }
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
        # Name a full disk here, since big writes exit through this function (#11313); below
        # 64 MB is the cause. Folded into $Message because --tauri shows only that line.
        $_diskSuffix = ""
        try {
            # Probed: early exits happen before the helper is defined.
            if ($StudioHome -and (Get-Command Get-StudioFreeSpaceBytes -CommandType Function -ErrorAction SilentlyContinue)) {
                $_diskFree = Get-StudioFreeSpaceBytes -Path $StudioHome
                if ($null -ne $_diskFree -and $_diskFree -lt 64MB) {
                    $_diskMb = [math]::Round($_diskFree / 1MB)
                    # Advice depends on what happened to the old environment (same four cases as install.sh).
                    $_diskRemedy = if ($script:StudioVenvDiscardLeftover) {
                        "Free some space and re-run. The previous environment could not be removed and is still at $($script:StudioVenvDiscardLeftover); deleting it will reclaim that space."
                    } elseif ($script:StudioVenvDiscardSucceeded) {
                        "Free some space and re-run. The previous environment was already discarded by --no-rollback, so the installer has nothing further of its own to reclaim."
                    } elseif ($script:StudioNoRollback) {
                        "Free some space and re-run."
                    } else {
                        "Free some space and re-run. --no-rollback (UNSLOTH_INSTALL_NO_ROLLBACK=1) drops the previous environment instead of keeping a copy of it during the install."
                    }
                    $_diskSuffix = ": $StudioHome has only $_diskMb MB free, so the disk is full, which is very likely the cause. $_diskRemedy"
                    if (-not $TauriMode) {
                        Write-StudioLine "        $StudioHome has only $_diskMb MB free -- the disk is full, which is very likely the cause." -ForegroundColor Red
                        Write-StudioLine "        $_diskRemedy" -ForegroundColor Red
                    }
                }
            }
        } catch { $_diskSuffix = "" }
        $Message = "$Message$_diskSuffix"
        Write-TauriLog "ERROR_DEFAULT" $Message
        # Under `irm | iex` the session survives a failure; a leaked handoff would re-pin it.
        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
        if (Get-Command Restore-StudioVenvRollback -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioVenvRollback
        }
        # Separate from the venv rollback: an install can fail before one is in flight.
        if (Get-Command Restore-StudioUvCacheMarker -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioUvCacheMarker -StudioRoot $StudioHome
        }
        # Most failures return before the lock try/finally, and under `irm | iex` these are the
        # caller's variables. Probed like above.
        if (Get-Command Restore-StudioTempEnvironment -CommandType Function -ErrorAction SilentlyContinue) {
            Restore-StudioTempEnvironment
        }
        if ($TauriMode) {
            exit $Code
        }
        throw $Message
    }

    # Usable temporary storage. Windows never checks TMP/TEMP exist or are writable
    # (#9140), and downloads stage there. If it cannot hold a file, point BOTH at a
    # directory we own; children and GetTempPath() read the process environment.
    function Test-StudioDirectoryUsable {
        param(
            [string]$Path,
            # Only for a directory we OWN: -Force would materialize a stale inherited TMP path.
            [switch]$CreateIfMissing
        )
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        try {
            if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
                if (-not $CreateIfMissing) { return $false }
                New-Item -ItemType Directory -Path $Path -Force -ErrorAction Stop | Out-Null
            }
            # Remove probes earlier runs could not delete.
            try {
                $cutoff = (Get-Date).AddDays(-1)
                foreach ($old in @(Get-ChildItem -LiteralPath $Path -File -Filter "unsloth-probe-*.tmp" -ErrorAction Stop)) {
                    # Shape, not prefix: this is the host's temp directory.
                    if ($old.Name -notmatch '^unsloth-probe-[0-9a-f]{8}\.tmp$') { continue }
                    if ($old.LastWriteTime -lt $cutoff) {
                        Remove-Item -LiteralPath $old.FullName -Force -ErrorAction SilentlyContinue
                    }
                }
            } catch {}
            # Write, read back, delete: Test-Path misses the failures that matter.
            $probe = Join-Path $Path ("unsloth-probe-" + [guid]::NewGuid().ToString('N').Substring(0, 8) + ".tmp")
            [System.IO.File]::WriteAllText($probe, "unsloth")
            $readBack = [System.IO.File]::ReadAllText($probe)
            # Deletion must work and be verified (the #9140 shape); retry briefly for scanners.
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
        # These outlive the install (autostarted Unsloth's %TEMP%), so prune by age.
        try {
            $cutoff = (Get-Date).AddDays(-1)
            foreach ($stale in @(Get-ChildItem -LiteralPath $Root -Directory -Filter "ust-*" -ErrorAction Stop)) {
                # SHAPE, not prefix, before a recursive delete: "ust-" + PID + "-" + 8 hex, as
                # scripts/uninstall.ps1 requires.
                if ($stale.Name -notmatch '^ust-[0-9]+-[0-9a-f]{8}$') { continue }
                if ($stale.LastWriteTime -ge $cutoff) { continue }
                # Reparse points are never ours: unlink without following. Not Remove-Item, which can
                # throw unsuppressibly or walk through a junction on 5.1.
                if ($stale.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    try { [System.IO.Directory]::Delete($stale.FullName, $false) } catch {}
                    continue
                }
                # owner.pid names the process that inherited this dir; if it is alive, keep the dir.
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
                # A recorded owner proves more than the installer PID in the name.
                $ownerRecorded = ($ownerPid -gt 0)
                if ($ownerPid -le 0) {
                    $null = [int]::TryParse(($stale.Name -split '-')[1], [ref]$ownerPid)
                }
                # No recorded owner: collect only after a week with no new entries.
                if (-not $ownerRecorded -and $stale.LastWriteTime -ge (Get-Date).AddDays(-7)) {
                    continue
                }
                if ($ownerPid -gt 0) {
                    $ownerLives = $true
                    try {
                        $null = Get-Process -Id $ownerPid -ErrorAction Stop
                    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
                        # Only this failure means "abandoned"; others keep the directory.
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
        # Only a directory this run created; never litter the host's temp.
        if (-not $script:StudioTempOverride.Owned) { return }
        $owned = $script:StudioTempOverride.Path
        if ([string]::IsNullOrWhiteSpace($owned)) { return }
        try {
            [System.IO.File]::WriteAllText((Join-Path $owned "owner.pid"), [string]$OwnerProcessId)
        } catch {}
    }

    function Get-StudioPrivateTempRoots {
        # Only under paths scripts/uninstall.ps1 already reclaims.
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
            # Short leaf: 5.1's .NET Framework compiler is bound by the legacy path limit.
            $leaf = "ust-" + $PID + "-" + [guid]::NewGuid().ToString('N').Substring(0, 8)
            $candidate = Join-Path $root $leaf
            # Only directories absent BEFORE the probe may be unwound; pre-provisioned ones are
            # configuration we did not create.
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
            # A failed root leaves the -Force chain behind; walk back up through our own empty
            # components only, never ~\.unsloth itself.
            $ours = @("temp", "Unsloth Studio", ".cache")
            $unwind = $candidate
            for ($depth = 0; $depth -lt 4; $depth++) {
                try {
                    if (-not $preAbsent[$unwind]) { break }
                    if (-not (Test-Path -LiteralPath $unwind -PathType Container)) { break }
                    $item = Get-Item -LiteralPath $unwind -Force -ErrorAction Stop
                    # A relocation junction is somebody's configuration; leave it and stop.
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
        # TMP wins over TEMP. IsNullOrEmpty, not IsNullOrWhiteSpace: GetTempPath takes the first
        # non-empty one, so a whitespace TMP is what Windows uses.
        $inherited = if (-not [string]::IsNullOrEmpty($env:TMP)) { $env:TMP } else { $env:TEMP }
        # Resolve BEFORE probing: Test-Path and .NET resolve relative paths differently. Leave
        # whitespace-only as is so the probe rejects it.
        $absolute = $inherited
        if (-not [string]::IsNullOrWhiteSpace($inherited)) {
            try { $absolute = [System.IO.Path]::GetFullPath($inherited) } catch { $absolute = $inherited }
        }
        if (Test-StudioDirectoryUsable -Path $absolute) {
            # The allocator is the only sweeper, so sweep here too when the temp is healthy.
            foreach ($root in @(Get-StudioPrivateTempRoots)) {
                Remove-StudioStalePrivateTempDirectories -Root $root
            }
            # Pin the absolute spelling: a relative value could later name somewhere else.
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
        # Absent is not empty: restoring an absent variable as "" changes children's temp.
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
        # The directory stays: an autostarted Unsloth inherited it as %TEMP%.
    }

    # ── Parse flags ──
    $StudioLocalInstall = $false
    $PackageName = "unsloth"
    $RepoRoot = ""
    $TauriMode = $false
    $SkipTorch = $false
    $SkipAutostart = $false
    $IsolateUvCache = $false
    # Script-scoped: tests/studio/test_install_rollback_lifecycle.ps1 extracts
    # Start-StudioVenvRollback on its own.
    $script:StudioNoRollback = $false
    # Initialised here, before the cache notice that sets it (the later reset would clear it).
    # Reset per run: under `irm | iex` the script scope is the caller's session.
    $script:StudioVolumeList = $null
    # Set by the --no-rollback discard when the environment it is about to delete reports XPU,
    # read by the Intel scan once that tree is gone.
    $script:StudioPreservedXpuVerdict = $false
    $ShortcutsOnly = $false
    $WithLlamaCppDir = ""
    $argList = $args
    for ($i = 0; $i -lt $argList.Count; $i++) {
        switch ($argList[$i]) {
            "--local"    { $StudioLocalInstall = $true }
            "--tauri"    { $TauriMode = $true }
            "--no-torch" { $SkipTorch = $true }
            "--isolated-uv-cache" { $IsolateUvCache = $true }
            "--no-rollback" { $script:StudioNoRollback = $true }
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
    if ($env:UNSLOTH_ISOLATE_UV_CACHE -in @('1', 'true', 'yes', 'on')) { $IsolateUvCache = $true }
    if ($env:UNSLOTH_INSTALL_NO_ROLLBACK -in @('1', 'true', 'yes', 'on')) { $script:StudioNoRollback = $true }

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
    # python.org fallback patch when winget and the live listing fail; bump with $PythonVersion.
    $PythonFallbackFullVersion = "3.13.13"
    # Patch releases the stack cannot run; mirrors PYTHON_SKIP in install.sh.
    $PythonSkip = @("3.13.8")
    # Only skipped because it cannot `import torch`, so a -NoTorch install accepts it.
    if ($SkipTorch) { $PythonSkip = @() }

    # An elevated run writes the install root as Administrators and the same account cannot
    # read it back afterwards. Twin in studio/setup.ps1.
    function Get-ElevationState {
        try {
            $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
            $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
            if ($principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
                return "true"
            }
            return "false"
        } catch {
            return "unknown"
        }
    }

    # Recorded either way; install.rs record_diag_marker puts it in the support report.
    function Write-ElevationNotice {
        param(
            [Parameter(Mandatory = $true)][string]$State,
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Root,
            # UNSLOTH_TAURI_MODE is not assigned until the setup call far below; --tauri is.
            [switch]$Tauri
        )

        if ($Tauri) {
            [Console]::Out.WriteLine("[TAURI:DIAG] elevated=$State")
            [Console]::Out.Flush()
        }
        if ($State -ne "true") { return }
        Write-StudioLine "  [WARNING] Running as administrator. Unsloth does not need this." -ForegroundColor Yellow
        Write-StudioLine "            Anything written to $Root will be owned by Administrators," -ForegroundColor Yellow
        Write-StudioLine "            and your normal account will not be able to read it afterwards." -ForegroundColor Yellow
        Write-StudioLine "            That folder outlives an uninstall, so a later install, setup or" -ForegroundColor Yellow
        Write-StudioLine "            update run normally fails on it." -ForegroundColor Yellow
        Write-StudioLine "            Stop and start again without 'Run as administrator'." -ForegroundColor Yellow
        Write-StudioLine ""
    }

    function Get-CanonicalRootPath {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
        $_p = $Path.Trim()
        if (($_p -eq "~" -or $_p -like "~/*" -or $_p -like "~\*") -and
            -not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
            # A bare "~" leaves an empty child path, which Join-Path rejects on PS 5.1.
            $_rest = $_p.Substring(1).TrimStart('/', '\')
            $_p = if ($_rest) { Join-Path $env:USERPROFILE $_rest } else { $env:USERPROFILE }
        }
        # GetFullPath anchors a relative path to [Environment]::CurrentDirectory, which
        # Set-Location does not move, so it disagreed with the Resolve-Path based resolver
        # (PowerShell#10278, by design). It stays as the fallback, never the first answer.
        try {
            $_p = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($_p)
        } catch {
            try { $_p = [System.IO.Path]::GetFullPath($_p) } catch { }
        }
        return $_p.TrimEnd('\', '/')
    }

    # Warn before the resolver below creates and probes the root, so mirror its precedence
    # rather than reuse the unassigned $StudioHome. USERPROFILE can be unset (service, CI).
    $UnslothRoot = if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        '%USERPROFILE%\.unsloth'
    } else {
        Join-Path $env:USERPROFILE ".unsloth"
    }
    $ElevationRoot = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) {
        $env:UNSLOTH_STUDIO_HOME.Trim()
    } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) {
        $env:STUDIO_HOME.Trim()
    } else {
        $UnslothRoot
    }
    # A legacy-equal override is not a custom root: llama.cpp and node stay siblings of
    # studio under ~/.unsloth, so name that parent.
    if ((Get-CanonicalRootPath $ElevationRoot) -ieq
        (Get-CanonicalRootPath (Join-Path $UnslothRoot "studio"))) {
        $ElevationRoot = $UnslothRoot
    }
    Write-ElevationNotice -State (Get-ElevationState) -Root $ElevationRoot -Tauri:$TauriMode

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

    # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)

    $script:StudioFinalPathWarned = $false
    $script:StudioInstallIsFresh = $null
    function Write-StudioFinalPathDegraded {
        param([string]$Reason)
        if ($script:StudioFinalPathWarned) { return }
        $script:StudioFinalPathWarned = $true
        if ($script:StudioInstallIsFresh -eq $true) {
            Write-Verbose "Unsloth: path identity is inexact ($Reason); first install, taking every runtime lock."
            return
        }
        # Only claim that the installer can continue.
        Write-StudioLine "[WARN] Could not resolve a path exactly ($Reason)." -ForegroundColor Yellow
        Write-StudioLine "       Continuing with the lexical resolver, which cannot recover a path's" -ForegroundColor Yellow
        Write-StudioLine "       stored casing or expand an 8.3 name, so paths are compared as written." -ForegroundColor Yellow
        Write-StudioLine "       Unsloth will take every runtime lock it might need rather than assume" -ForegroundColor Yellow
        Write-StudioLine "       two spellings are different directories." -ForegroundColor Yellow
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
            # \??\UNC\server\share is the device spelling of \\server\share; left as is it reads as
            # relative and yields a wrong mutex identity.
            if ($target.StartsWith('UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
                $target = '\\' + $target.Substring(4)
            } elseif ($target.StartsWith('Volume{', [System.StringComparison]::OrdinalIgnoreCase)) {
                # Same for \??\Volume{GUID}\...: \\?\ keeps naming the volume.
                $target = '\\?\' + $target
            }
        }
        # A drive-less "\real" resolves on the LINK's volume, not the process's current drive.
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

    # Compiler-free stand-in for the native resolver: normalized, not exact (no 8.3
    # expansion or casing). Never throws.
    $script:StudioSubstMap = $null
    function Get-StudioSubstTarget {
        param([Parameter(Mandatory = $true)][string]$Path)
        # No 5.1 API reports SUBST mappings; `subst` with no arguments prints them, read once.
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
        # Fold SUBST first: the Python runtime gate resolves it, and both must name one mutex.
        $substituted = Get-StudioSubstTarget -Path $current
        if ($substituted) {
            try { $current = [System.IO.Path]::GetFullPath($substituted) } catch {}
        }
        # Hashtable, not a generic HashSet: already case-insensitive, and a
        # locked-down host (the kind that got here) can forbid generic types.
        $visited = @{}
        # Links on parent components are ordinary; walk from the root, capped against loops.
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

    # Runs before the install lock: find an existing Python only. Path.resolve matches what
    # unsloth_cli/_studio_runtime_gate.py hashes, so both sides name the same lock.
    $script:StudioEarlyPythonProbed = $false
    $script:StudioEarlyPython = $null
    $script:StudioEarlyPythonProbedWithoutVenv = $false

    function Test-StudioSddlRightsAreWrite {
        param([string]$Rights)
        if ([string]::IsNullOrWhiteSpace($Rights)) { return $true }
        $text = "$Rights".Trim().ToUpper()
        if ($text -like "0X*") {
            $mask = [long]0
            foreach ($ch in $text.Substring(2).ToCharArray()) {
                $digit = "0123456789ABCDEF".IndexOf($ch)
                if ($digit -lt 0) { return $true }
                $mask = ($mask * 16) + $digit
            }
            return (($mask -band 0x500D0156) -ne 0)
        }
        foreach ($alias in @("GA", "GW", "WD", "WO", "SD", "DT", "FA", "FW", "KA", "KW",
                             "CC", "DC", "WP", "SW")) {
            if ($text.Contains($alias)) { return $true }
        }
        return $false
    }

    function Test-StudioSddlPrincipalIsAdminOnly {
        param([string]$Principal)
        if ([string]::IsNullOrWhiteSpace($Principal)) { return $false }
        $who = "$Principal".Trim().ToUpper()
        return (@(
            "BA", "SY", "S-1-5-32-544", "S-1-5-18",
            "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
        ) -contains $who)
    }

    function Test-StudioSddlWritableByNonAdmin {
        param([string]$Sddl)
        if ([string]::IsNullOrWhiteSpace($Sddl)) { return $true }
        $text = "$Sddl".Trim()
        if (-not ($text -match '^O:([A-Za-z0-9\-]+?)(?:G:|D:|S:|$)')) { return $true }
        if (-not (Test-StudioSddlPrincipalIsAdminOnly -Principal $Matches[1])) { return $true }
        $daclAt = $text.IndexOf("D:")
        if ($daclAt -lt 0) { return $true }
        $dacl = $text.Substring($daclAt)
        $saclAt = $dacl.IndexOf("S:")
        if ($saclAt -ge 0) { $dacl = $dacl.Substring(0, $saclAt) }
        $seen = 0
        foreach ($chunk in ($dacl -split '\)')) {
            $open = $chunk.IndexOf("(")
            if ($open -lt 0) { continue }
            $seen++
            $fields = $chunk.Substring($open + 1) -split ';'
            if ($fields.Count -lt 6) { return $true }
            $type = "$($fields[0])".Trim().ToUpper()
            $flags = "$($fields[1])".Trim().ToUpper()
            if (@("D", "OD", "XD") -contains $type) { continue }
            if ($flags.Contains("IO")) { continue }
            if (-not (Test-StudioSddlRightsAreWrite -Rights $fields[2])) { continue }
            if (-not (Test-StudioSddlPrincipalIsAdminOnly -Principal $fields[5])) { return $true }
        }
        # A DACL with no ACEs in it is not evidence of anything, and neither is one this did not
        # manage to split.
        if ($seen -eq 0) { return $true }
        return $false
    }

    function Test-StudioDirectoryIsAdminOnly {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        $sddl = ""
        try {
            # Select-Object -ExpandProperty, not a property read: Constrained Language Mode refuses
            # property access on types outside its allowed list, and the security descriptor is one.
            $sddl = "$(Get-Acl -LiteralPath $Path -ErrorAction Stop |
                Select-Object -ExpandProperty Sddl)"
        } catch { return $false }
        if ([string]::IsNullOrWhiteSpace($sddl)) { return $false }
        return (-not (Test-StudioSddlWritableByNonAdmin -Sddl $sddl))
    }

    function Get-StudioLexicalParent {
        param([string]$Path)
        $trimmed = "$Path".TrimEnd('\', '/')
        $cut = $trimmed.LastIndexOfAny(@([char]92, [char]47))
        if ($cut -lt 0) { return "" }
        return $trimmed.Substring(0, $cut)
    }

    function Test-StudioPathUnderAdminRoot {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        # $env: rather than [Environment]::GetEnvironmentVariable: System.Environment is not on
        # Constrained Language Mode's allowed type list, and this helper runs on hosts under it.
        $roots = @()
        foreach ($value in @($env:SystemRoot, $env:ProgramFiles, $env:ProgramW6432, ${env:ProgramFiles(x86)})) {
            if (-not [string]::IsNullOrWhiteSpace($value)) { $roots += "$value".TrimEnd('\', '/') }
        }
        $matchedRoot = ""
        foreach ($root in $roots) {
            $escapedRoot = $root -replace '([\[\]\*\?])', '`$1'
            if (("$Path" -like ($escapedRoot + "\*")) -or ("$Path" -like ($escapedRoot + "/*"))) {
                $matchedRoot = $root
                break
            }
        }
        if ([string]::IsNullOrWhiteSpace($matchedRoot)) { return $false }
        $windowsRoot = "$($env:SystemRoot)".TrimEnd('\', '/')
        if (-not [string]::IsNullOrWhiteSpace($windowsRoot)) {
            foreach ($leaf in @(
                "Temp", "Tasks", "Tracing", "Registration\CRMLog", "debug\WIA",
                "System32\Tasks", "System32\spool\drivers\color", "System32\spool\PRINTERS",
                "System32\spool\SERVERS", "System32\com\dmp", "System32\FxsTmp",
                "SysWOW64\Tasks", "SysWOW64\com\dmp", "SysWOW64\FxsTmp"
            )) {
                $writable = $windowsRoot + "\" + $leaf
                $escapedWritable = $writable -replace '([\[\]\*\?])', '`$1'
                if (("$Path" -like ($escapedWritable + "\*")) -or
                    ("$Path" -like ($escapedWritable + "/*"))) { return $false }
            }
        }
        $current = Get-StudioLexicalParent -Path "$Path"
        $guard = 0
        while (-not [string]::IsNullOrWhiteSpace($current)) {
            $guard++
            if ($guard -gt 64) { return $false }
            if (-not (Test-StudioDirectoryIsAdminOnly -Path $current)) { return $false }
            if ("$current".TrimEnd('\', '/') -eq $matchedRoot) { return $true }
            $next = Get-StudioLexicalParent -Path "$current"
            if ("$next" -eq "$current") { return $false }
            $current = $next
        }
        return $false
    }

    function Test-StudioInterpreterFileIsAdminOnly {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        try {
            $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop |
                Select-Object -Property Attributes, LinkType
            if ($null -eq $item) { return $false }
            if ("$($item.Attributes)" -match 'ReparsePoint') { return $false }
            if (-not [string]::IsNullOrWhiteSpace("$($item.LinkType)")) { return $false }
        } catch { return $false }
        return (Test-StudioDirectoryIsAdminOnly -Path $Path)
    }

    function Get-StudioEarlyPython {
        # A miss taken before $VenvDir existed (--tauri) is re-probed once it does.
        $venvDirValue = $null
        try { $venvDirValue = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue } catch {}
        $venvKnown = -not [string]::IsNullOrWhiteSpace($venvDirValue)
        if ($script:StudioEarlyPythonProbed) {
            if (-not ($script:StudioEarlyPythonProbedWithoutVenv -and $venvKnown -and
                      [string]::IsNullOrWhiteSpace($script:StudioEarlyPython))) {
                return $script:StudioEarlyPython
            }
        }
        $script:StudioEarlyPythonProbed = $true
        $script:StudioEarlyPythonProbedWithoutVenv = (-not $venvKnown)
        if ("$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0") { return $null }
        $candidates = @()
        $venvCandidates = @()
        if ($venvKnown) {
            $venvCandidates = @((Join-Path $venvDirValue "Scripts\python.exe"), (Join-Path $venvDirValue "bin/python3"))
            $candidates += $venvCandidates
        }
        foreach ($name in @("python3", "python")) {
            try {
                foreach ($cmd in @(Get-Command $name -All -CommandType Application -ErrorAction SilentlyContinue)) {
                    if ($cmd -and $cmd.Source) { $candidates += $cmd.Source }
                }
            } catch {}
        }
        $requireAdminRoot = $false
        if ($env:OS -eq "Windows_NT") {
            try { $requireAdminRoot = Test-StudioChildScriptDirectoryElevated } catch { $requireAdminRoot = $true }
        }
        $rejectedForWritability = $false
        foreach ($candidate in $candidates) {
            if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
            # Test-Path throws on an unreadable dir under Stop: skip this candidate only.
            $isFile = $false
            try { $isFile = Test-Path -LiteralPath $candidate -PathType Leaf } catch {}
            if (-not $isFile) { continue }
            # The venv interpreter runs here before the destination guard does, so only where that
            # guard would let it run: the default layout, or a custom home that is already Unsloth's.
            if ($venvCandidates -contains $candidate) {
                $ours = $false
                try {
                    $mode = Get-Variable -Name StudioRedirectMode -ValueOnly -ErrorAction Stop
                    $studioHomeValue = Get-Variable -Name StudioHome -ValueOnly -ErrorAction Stop
                    # The guard's own predicates; one not yet defined this early throws, which declines.
                    $ours = ($mode -ne 'env') -or
                        (Test-Path -LiteralPath (Join-Path $venvDirValue ".unsloth-studio-owned") -PathType Leaf) -or
                        (Test-Path -LiteralPath (Join-Path $studioHomeValue "share\studio.conf") -PathType Leaf) -or
                        (Test-Path -LiteralPath (Join-Path $studioHomeValue "bin\unsloth.exe") -PathType Leaf) -or
                        (Test-StudioPlainFile -Path (Join-Path $studioHomeValue ".unsloth-studio-owned")) -or
                        (Test-UnslothCmdShimFile (Join-Path $studioHomeValue "bin\unsloth.cmd"))
                } catch { $ours = $false }
                if (-not $ours) {
                    if ($requireAdminRoot) { $rejectedForWritability = $true }
                    continue
                }
            }
            if ($requireAdminRoot -and ($venvCandidates -notcontains $candidate)) {
                $adminOnly = $false
                try {
                    $adminOnly = (Test-StudioPathUnderAdminRoot -Path $candidate) -and
                                 (Test-StudioInterpreterFileIsAdminOnly -Path $candidate)
                } catch { $adminOnly = $false }
                if (-not $adminOnly) {
                    $rejectedForWritability = $true
                    continue
                }
            }
            if ("$candidate" -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]') { continue }
            # Probe the interpreter's own directory: $PSScriptRoot is empty when the script runs from memory.
            $probeDir = $null
            try { $probeDir = [System.IO.Path]::GetDirectoryName($candidate) } catch {}
            if ([string]::IsNullOrWhiteSpace($probeDir)) { continue }
            $probe = Invoke-StudioEarlyPython -Exe $candidate -Path $probeDir
            if (-not [string]::IsNullOrWhiteSpace($probe)) {
                $script:StudioEarlyPython = $candidate
                return $candidate
            }
        }
        if ($rejectedForWritability) {
            Write-StudioFinalPathDegraded -Reason ("this run is elevated and the only interpreters found " +
                "are in directories a standard user can write, which an elevated run must not launch")
        }
        return $null
    }

    function Invoke-StudioEarlyPython {
        param(
            [Parameter(Mandatory = $true)][string]$Exe,
            [Parameter(Mandatory = $true)][string]$Path,
            [int]$TimeoutMs = 10000
        )
        # The gate's _resolved_windows_path, strict so loops/dangling links raise. <3.8 does not follow links.
        $script = "import pathlib,sys" + [char]10 +
                  "sys.exit(2) if sys.version_info < (3,8) else None" + [char]10 +
                  "sys.stdout.buffer.write(str(pathlib.Path(sys.argv[1]).resolve(strict=True)).encode('utf-8'))"
        # Any error, incl. from Test-Path, returns $null (lexical rung).
        try {
            # Verbatim: Trim() would drop a trailing U+00A0, which NTFS names keep.
            $answer = "$(Invoke-StudioEarlyPythonScript -Exe $Exe -Script $script -ScriptArgs @($Path) -TimeoutMs $TimeoutMs)"
            if ([string]::IsNullOrWhiteSpace($answer)) { return $null }
            if (-not [System.IO.Path]::IsPathRooted($answer)) { return $null }
            if (-not (Test-Path -LiteralPath $answer)) { return $null }
            return $answer
        } catch {
            return $null
        }
    }

    function Invoke-StudioEarlyPythonScript {
        param(
            [Parameter(Mandatory = $true)][string]$Exe,
            [Parameter(Mandatory = $true)][string]$Script,
            [string[]]$ScriptArgs = @(),
            [int]$TimeoutMs = 10000
        )
        $proc = $null
        $clock = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Exe
            # -S as well as -I: -I still imports site, so a sitecustomize could print or hang.
            # -B: these probes can run before the install lock, so they must write no .pyc.
            # ArgumentList is .NET Core only; Windows PowerShell 5.1 lacks it.
            $argv = @("-I", "-S", "-B", "-c", $Script) + $ScriptArgs
            if ($null -ne $psi.PSObject.Properties["ArgumentList"]) {
                foreach ($a in $argv) { $null = $psi.ArgumentList.Add($a) }
            } else {
                # Quote for CommandLineToArgvW, doubling trailing backslashes so "C:\dir\" keeps its quote.
                $psi.Arguments = (@($argv | ForEach-Object {
                    '"' + ($_ -replace '(\\+)$', '$1$1') + '"'
                }) -join ' ')
            }
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            # Otherwise 5.1 decodes with the console codepage and corrupts non-ASCII paths.
            $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $stdout = $proc.StandardOutput.ReadToEndAsync()
            $null = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit($TimeoutMs)) {
                try { $proc.Kill() } catch {}
                return $null
            }
            if ($proc.ExitCode -ne 0) { return $null }
            # A leftover child can hold stdout open, so the deadline bounds the read too.
            $left = [Math]::Max(0, $TimeoutMs - [int]$clock.ElapsedMilliseconds)
            if (-not $stdout.Wait($left)) { return $null }
            return "$($stdout.Result)"
        } catch {
            return $null
        } finally {
            if ($proc) { try { $proc.Dispose() } catch {} }
        }
    }


    function Get-StudioSystem32Tool {
        param([Parameter(Mandatory = $true)][string]$Name)
        if ([string]::IsNullOrWhiteSpace($env:SystemRoot)) { return "" }
        $candidate = Join-Path (Join-Path $env:SystemRoot "System32") $Name
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { return "" }
        return $candidate
    }

    function Invoke-StudioSystem32ToolBounded {
        param([string]$Exe, [string[]]$Arguments = @(), [int]$TimeoutMs = 3000)
        # CLM cannot construct ProcessStartInfo. A job captures the trusted system tool's
        # output through PowerShell's pipes, without redirect files in an unlabelled directory.
        if ("$($ExecutionContext.SessionState.LanguageMode)" -ne "FullLanguage") {
            $job = $null
            try {
                $seconds = ($TimeoutMs - ($TimeoutMs % 1000)) / 1000
                if (($TimeoutMs % 1000) -ne 0) { $seconds++ }
                if ($seconds -lt 1) { $seconds = 1 }
                $job = Start-Job -ScriptBlock {
                    param($tool, $toolArgs)
                    $lines = @(& $tool @toolArgs 2>$null)
                    return @{ Output = ($lines -join "`n"); ExitCode = $LASTEXITCODE }
                } -ArgumentList $Exe, (,$Arguments)
                if (Wait-Job -Job $job -Timeout $seconds) {
                    return (Receive-Job -Job $job -ErrorAction SilentlyContinue | Select-Object -Last 1)
                }
                Stop-Job -Job $job -ErrorAction SilentlyContinue
            } catch { return $null }
            finally { if ($job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } }
            return $null
        }
        $proc = $null
        try {
            # If policy forbids these types, decline. The caller treats an unreadable
            # token as elevated and uses the inline probe when no private directory is safe.
            $clock = [System.Diagnostics.Stopwatch]::StartNew()
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Exe
            if ($null -ne $psi.PSObject.Properties['ArgumentList']) {
                foreach ($arg in $Arguments) { $null = $psi.ArgumentList.Add($arg) }
            } else {
                # The arguments here are paths and fixed switches; paths cannot contain quotes.
                $psi.Arguments = (@($Arguments | ForEach-Object {
                    '"' + ($_ -replace '(\\+)$', '$1$1') + '"'
                }) -join ' ')
            }
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $stdout = $proc.StandardOutput.ReadToEndAsync()
            $null = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit($TimeoutMs)) {
                try { $proc.Kill() } catch { }
                return $null
            }
            $left = [Math]::Max(0, $TimeoutMs - [int]$clock.ElapsedMilliseconds)
            if (-not $stdout.Wait($left)) { return $null }
            return @{ Output = "$($stdout.Result)"; ExitCode = $proc.ExitCode }
        } catch { return $null }
        finally { if ($proc) { try { $proc.Dispose() } catch { } } }
    }

    function Test-StudioChildScriptDirectoryElevated {
        $groups = ""
        $whoami = Get-StudioSystem32Tool -Name "whoami.exe"
        if (-not $whoami) { return $true }
        try {
            $result = Invoke-StudioSystem32ToolBounded -Exe $whoami -Arguments @('/groups')
            if ($null -eq $result) { return $true }
            $groups = "$($result.Output)"
        } catch { return $true }
        if ([string]::IsNullOrWhiteSpace($groups)) { return $true }
        if ($groups -match "S-1-16-(12288|16384)") { return $true }
        if ($groups -match "S-1-16-\d+") { return $false }
        return $true
    }

    function New-StudioChildScriptDirectory {
        $tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
        # Join-Path throws on a TEMP naming a missing drive: decline, like any other unusable root.
        try { $dir = Join-Path $tempRoot ("unsloth-child-" + [guid]::NewGuid().ToString("N")) -ErrorAction Stop } catch { return "" }
        $made = $false
        try {
            $createdPath = "$(New-Item -ItemType Directory -Path $dir -ErrorAction Stop |
                Select-Object -ExpandProperty FullName)"
            $made = (Test-Path -LiteralPath $dir -PathType Container)
            if ((-not $made) -and $createdPath) {
                Remove-Item -LiteralPath $createdPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        } catch { }
        if ((-not $made) -and $dir -match '[\[\]]') {
            $escaped = $dir -replace '([\[\]])', '`$1'
            try {
                $null = New-Item -ItemType Directory -Path $escaped -ErrorAction Stop
                $made = (Test-Path -LiteralPath $dir -PathType Container)
            } catch { }
        }
        # Created by THIS call. New-Item without -Force throws on a directory that already exists,
        # which is the point: a pre-created one carrying an attacker's ACL is refused, not adopted.
        if (-not $made) { return "" }
        if ($env:OS -eq "Windows_NT") {
            # Until it is labelled a standard user can swap this directory for a junction: a link is
            # never adopted, and cleanup never recurses through one into its target.
            $isLink = { try { "$((Get-Item -LiteralPath $dir -Force -ErrorAction Stop).Attributes)" -match 'ReparsePoint' } catch { $true } }
            $labelled = $false
            $icacls = Get-StudioSystem32Tool -Name "icacls.exe"
            if (-not $icacls) {
                if (-not (& $isLink)) { try { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue } catch { } }
                return ""
            }
            try {
                $result = Invoke-StudioSystem32ToolBounded -Exe $icacls -Arguments @($dir, '/setintegritylevel', '(OI)(CI)H')
                $labelled = ($null -ne $result -and $result.ExitCode -eq 0)
                if (-not $labelled -and $null -ne $result) {
                    $readback = Invoke-StudioSystem32ToolBounded -Exe $icacls -Arguments @($dir)
                    $labelled = ($null -ne $readback -and "$($readback.Output)" -match "S-1-16-12288|High Mandatory Level")
                }
            } catch { $labelled = $false }
            if ((-not $labelled) -and (Test-StudioChildScriptDirectoryElevated)) {
                if (-not (& $isLink)) { try { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue } catch { } }
                return ""
            }
            if (& $isLink) { return "" }
            if ($labelled) {
                $planted = $true
                try {
                    $planted = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop).Count -ne 0
                } catch { $planted = $true }
                if ($planted) {
                    if (-not (& $isLink)) { try { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue } catch { } }
                    return ""
                }
            }
        }
        return $dir
    }

    # One child per distinct path: the process scan asks for the same image paths repeatedly.
    $script:StudioPythonFinalPathCache = $null

    function Get-StudioPythonFinalPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        if ($null -eq $script:StudioPythonFinalPathCache) { $script:StudioPythonFinalPathCache = @{} }
        if ($script:StudioPythonFinalPathCache.ContainsKey($Path)) {
            return $script:StudioPythonFinalPathCache[$Path]
        }
        $exe = $null
        try { $exe = Get-StudioEarlyPython } catch {}
        # Not cached: the re-probe once $VenvDir is known may still find an interpreter.
        if (-not $exe) { return $null }
        $answer = Invoke-StudioEarlyPython -Exe $exe -Path $Path
        $script:StudioPythonFinalPathCache[$Path] = $answer
        return $answer
    }

    function Resolve-StudioFinalPathsInOneChild {
        param([string[]]$Paths = @())
        if ($null -eq $script:StudioPythonFinalPathCache) { $script:StudioPythonFinalPathCache = @{} }
        $wanted = @()
        foreach ($candidate in $Paths) {
            if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
            if ($script:StudioPythonFinalPathCache.ContainsKey($candidate)) { continue }
            if ($wanted -notcontains $candidate) { $wanted += $candidate }
        }
        if ($wanted.Count -eq 0) { return }
        $exe = $null
        try { $exe = Get-StudioEarlyPython } catch {}
        if (-not $exe) { return }
        $listDir = New-StudioChildScriptDirectory
        $listFile = $null
        if ($listDir) {
            $listFile = Join-Path $listDir "paths.txt"
            try { Set-Content -LiteralPath $listFile -Value $wanted -Encoding UTF8 -ErrorAction Stop }
            catch { Remove-Item -LiteralPath $listDir -Recurse -Force -ErrorAction SilentlyContinue; $listDir = $null; $listFile = $null }
        }
        # Without a private directory the list rides the environment, in chunks under its 32767-character
        # block, so a declined directory does not fall back to one child per process.
        $batches = @()
        if ($listFile) { $batches += ,@{ Args = @($listFile); Env = $null } }
        else {
            $chunk = ""
            foreach ($candidate in $wanted) {
                if ($chunk -and ($chunk.Length + $candidate.Length + 1) -gt 30000) { $batches += ,@{ Args = @(); Env = $chunk }; $chunk = "" }
                $chunk = if ($chunk) { $chunk + "`n" + $candidate } else { $candidate }
            }
            if ($chunk) { $batches += ,@{ Args = @(); Env = $chunk } }
        }
        $probe = "import os,pathlib,sys" + [char]10 +
            "sys.exit(2) if sys.version_info < (3,8) else None" + [char]10 +
            "out=[]" + [char]10 +
            "if len(sys.argv)>1:" + [char]10 +
            "    with open(sys.argv[1],'r',encoding='utf-8-sig') as fh: lines=fh.read().split('\n')" + [char]10 +
            "else:" + [char]10 +
            "    lines=os.environ.get('UNSLOTH_FINAL_PATHS','').split('\n')" + [char]10 +
            "for line in lines:" + [char]10 +
            "    p=line.rstrip('\r\n')" + [char]10 +
            "    if not p: continue" + [char]10 +
            "    try:" + [char]10 +
            "        out.append(p+'|'+str(pathlib.Path(p).resolve(strict=True)))" + [char]10 +
            "    except Exception:" + [char]10 +
            "        out.append(p+'|')" + [char]10 +
            "sys.stdout.buffer.write('\n'.join(out).encode('utf-8'))"
        $raw = ""
        $savedPaths = $env:UNSLOTH_FINAL_PATHS
        try {
            foreach ($batch in $batches) {
                if ($null -ne $batch.Env) { $env:UNSLOTH_FINAL_PATHS = $batch.Env }
                try { $raw += "`n" + (Invoke-StudioEarlyPythonScript -Exe $exe -Script $probe -ScriptArgs $batch.Args -TimeoutMs 30000) } catch { }
            }
        } finally {
            if ($null -eq $savedPaths) { Remove-Item Env:UNSLOTH_FINAL_PATHS -ErrorAction SilentlyContinue }
            else { $env:UNSLOTH_FINAL_PATHS = $savedPaths }
            if ($listDir) { Remove-Item -LiteralPath $listDir -Recurse -Force -ErrorAction SilentlyContinue }
        }
        $answers = @{}
        $reported = @{}
        foreach ($line in ("$raw" -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            # The first separator, because the pipe is not legal in a Windows path so the left
            # half cannot contain one. $requested, not $input: $input is an automatic variable.
            $split = $line.IndexOf('|')
            if ($split -lt 1) { continue }
            $requested = $line.Substring(0, $split)
            $answer = $line.Substring($split + 1)
            $reported[$requested] = $true
            if ([string]::IsNullOrWhiteSpace($answer)) { continue }
            if (-not (Split-Path -IsAbsolute $answer)) { continue }
            if (-not (Test-Path -LiteralPath $answer)) { continue }
            $answers[$requested] = $answer
        }
        foreach ($candidate in $wanted) {
            if ($answers.ContainsKey($candidate)) {
                $script:StudioPythonFinalPathCache[$candidate] = $answers[$candidate]
            } elseif ($reported.ContainsKey($candidate)) {
                # Asked, and the child said it cannot be resolved. Recorded as $null rather than
                # left absent, so the miss is not re-asked once per protected root.
                $script:StudioPythonFinalPathCache[$candidate] = $null
            }
            # Neither: the child answered, but not about this path. Left uncached, so the
            # single-path rung still gets its turn.
        }
    }

    # Tell Explorer a shortcut changed via a child interpreter, where the type cannot be
    # defined in this shell. Cosmetic; never fails the install.
    function Invoke-StudioPythonShellIconRefresh {
        param([string[]]$Paths = @(), [string]$Exe = "", [switch]$DestinationValidated)
        if (-not ($env:OS -eq "Windows_NT")) { return $false }
        # Elevated, a venv interpreter runs only after the destination guard; --shortcuts-only never reaches it.
        if (-not $DestinationValidated) {
            try { if ((Get-ElevationState) -ne "false") { return $false } } catch { return $false }
        }
        # Kill switch before $Exe too: it forbids any child on this host, however the path was found.
        if ("$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0") { return $false }
        $exe = $Exe
        if ([string]::IsNullOrWhiteSpace($exe)) {
            try {
                $exe = Get-StudioEarlyPython
            } catch { return $false }
        }
        if (-not $exe) { return $false }
        # Per-item SHCNE_UPDATEITEM is required: the global broadcast misses in-place .lnk rewrites.
        # SHCNF_FLUSH (0x1000) because the child exits at once and a queued notification is lost.
        $script = "import ctypes,sys" + [char]10 +
            "from ctypes import wintypes" + [char]10 +
            "s32=ctypes.WinDLL('shell32',use_last_error=True)" + [char]10 +
            "s32.SHChangeNotify.restype=None" + [char]10 +
            "s32.SHChangeNotify.argtypes=[wintypes.LONG,wintypes.UINT,wintypes.LPCWSTR,wintypes.LPCWSTR]" + [char]10 +
            "for p in sys.argv[1:]:" + [char]10 +
            "    s32.SHChangeNotify(0x00002000,0x1005,p,None)" + [char]10 +
            "s32.SHChangeNotify(0x08000000,0x1000,None,None)" + [char]10 +
            "sys.stdout.write('ok')"
        try {
            $answer = Invoke-StudioEarlyPythonScript -Exe $exe -Script $script -ScriptArgs $Paths -TimeoutMs 10000
        } catch { return $false }
        return ("$answer".Trim() -eq "ok")
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
            # A dangling link/loop reads as missing; a reparse point in the stripped tail must stay inexact.
            $strippedAttributes = $null
            try { $strippedAttributes = [System.IO.File]::GetAttributes($existingPath) } catch { }
            if ($null -ne $strippedAttributes -and
                ($strippedAttributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                return [pscustomobject]@{ Path = $fullPath; Exact = $false }
            }
            $missingSegments = @($leaf) + $missingSegments
            $existingPath = $parent
        }
        $exact = $false
        $resolved = Get-StudioPythonFinalPath -Path $existingPath
        if (-not [string]::IsNullOrWhiteSpace($resolved)) { $exact = $true }
        if ([string]::IsNullOrEmpty($resolved)) {
            $resolved = Get-StudioLexicalPath -Path $existingPath
            $exact = $false
            Write-StudioFinalPathDegraded -Reason "no Python interpreter was available to resolve it"
        }
        if ($resolved.StartsWith('\\?\UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
            $resolved = '\\' + $resolved.Substring(8)
        } elseif ($resolved.StartsWith('\\?\', [System.StringComparison]::OrdinalIgnoreCase)) {
            # Strip every extended prefix, including \\?\Volume{GUID}\: this is hashed into the
            # runtime mutex name and unsloth_cli/_studio_runtime_gate.py strips \\?\ unconditionally.
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

    function Test-StudioSameDirectoryByProbe {
        param([string]$Left, [string]$Right)
        if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) { return $false }
        if (-not (Test-Path -LiteralPath $Left -PathType Container)) { return $false }
        if (-not (Test-Path -LiteralPath $Right -PathType Container)) { return $false }
        $name = ".unsloth-path-probe-" + [guid]::NewGuid().ToString("N")
        $probe = Join-Path $Left $name
        try { Set-Content -LiteralPath $probe -Value "" -ErrorAction Stop } catch { return $false }
        try {
            return (Test-Path -LiteralPath (Join-Path $Right $name))
        } finally {
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
        }
    }

    function Split-StudioExistingAncestor {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
        $current = $Path
        $tail = ""
        # Bounded: a malformed path can leave Split-Path returning something that never shortens.
        for ($hop = 0; $hop -lt 64; $hop++) {
            if ([string]::IsNullOrWhiteSpace($current)) { break }
            if (Test-Path -LiteralPath $current -PathType Container) {
                return [pscustomobject]@{ Root = $current; Tail = $tail }
            }
            $leaf = Split-Path -Leaf $current
            $parent = Split-Path -Parent $current
            if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) { break }
            if ([string]::IsNullOrWhiteSpace($leaf)) { break }
            $tail = if ($tail) { Join-Path $leaf $tail } else { $leaf }
            $current = $parent
        }
        return $null
    }

    # Free space for the disk-full diagnosis (#11313), or $null. A volume can be mounted at a
    # directory, so take the longest Win32_Volume Name prefixing the path; otherwise callers
    # fall back to the drive root.
    function Select-StudioVolumeForPath {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path,
              [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Volumes)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
        # Unrooted would anchor to the current directory and invent a match.
        if (-not [System.IO.Path]::IsPathRooted($Path)) { return $null }
        $full = [System.IO.Path]::GetFullPath($Path)
        # Append a separator so the mount point matches itself and C:\studiofoo does not match
        # C:\studio.
        if (-not $full.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
            $full += [System.IO.Path]::DirectorySeparatorChar
        }
        $best = $null
        foreach ($vol in $Volumes) {
            if (-not $vol -or -not $vol.Name) { continue }
            if (-not $full.StartsWith($vol.Name, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            if ((-not $best) -or $vol.Name.Length -gt $best.Name.Length) { $best = $vol }
        }
        return $best
    }

    # Junctions lie about which volume a path is on; resolve first, lexical fallback.
    function Resolve-StudioVolumeQueryPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        try {
            $final = Get-StudioFinalPath -Path $Path
            # Volume{GUID}\... is not rooted, so keep the caller's path then.
            if ($final -and [System.IO.Path]::IsPathRooted($final)) { return $final }
            return $Path
        } catch { return $Path }
    }

    # Out of process and bounded: a degraded WMI repository blocks CIM queries forever.
    # Cached (timeouts too); -Fresh for the free-space caller.
    function Get-StudioVolumeList {
        param([switch]$Fresh)
        if (-not $Fresh -and $null -ne $script:StudioVolumeList) { return $script:StudioVolumeList }
        $script:StudioVolumeList = @()
        $job = $null
        try {
            $job = Start-Job -ScriptBlock {
                Get-CimInstance -ClassName Win32_Volume -ErrorAction SilentlyContinue |
                    Select-Object Name, DeviceID, FreeSpace
            }
            if (Wait-Job -Job $job -Timeout 15) {
                $script:StudioVolumeList = @(Receive-Job -Job $job -ErrorAction SilentlyContinue |
                    Where-Object { $_ -and $_.Name })
            } else {
                Stop-Job -Job $job -ErrorAction SilentlyContinue
            }
        } catch {
        } finally {
            if ($job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue }
        }
        return $script:StudioVolumeList
    }

    function Get-StudioMountedVolume {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path,
              [switch]$Fresh)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
        if (-not ($IsWindows -or $env:OS -eq "Windows_NT")) { return $null }
        try {
            return (Select-StudioVolumeForPath -Path (Resolve-StudioVolumeQueryPath -Path $Path) `
                -Volumes (Get-StudioVolumeList -Fresh:$Fresh))
        } catch { return $null }
    }

    # GetPathRoot on a path that does not exist yet still names the volume it would be created on.
    function Get-StudioFreeSpaceBytes {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
        $queryPath = Resolve-StudioVolumeQueryPath -Path $Path
        # -Fresh: this is the caller that wants a number, and a stale one is worse than none.
        $mounted = Get-StudioMountedVolume -Path $queryPath -Fresh
        if ($mounted -and $null -ne $mounted.FreeSpace) { return [int64]$mounted.FreeSpace }
        try {
            # The fallback needs the resolved path too: DriveInfo on the lexical one answers for
            # the drive the junction lives on, not the drive it points at.
            $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($queryPath))
            if (-not $root) { return $null }
            return ([System.IO.DriveInfo]::new($root)).AvailableFreeSpace
        } catch { return $null }
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
        $_tauriSameRoot = $false
        if ($_tauriOverride -eq $_legacyTauriRoot) {
            $_tauriSameRoot = $true
        } else {
            $_tauriSameRoot = Test-StudioSameDirectoryByProbe -Left $_tauriOverride -Right $_legacyTauriRoot
        }
        if (-not $_tauriSameRoot) {
            $_tauriLeft = Split-StudioExistingAncestor -Path $_tauriOverride
            $_tauriRight = Split-StudioExistingAncestor -Path $_legacyTauriRoot
            if ($_tauriLeft -and $_tauriRight -and [string]::Equals(
                    $_tauriLeft.Tail, $_tauriRight.Tail, [System.StringComparison]::OrdinalIgnoreCase)) {
                $_tauriSameRoot = Test-StudioSameDirectoryByProbe -Left $_tauriLeft.Root -Right $_tauriRight.Root
            }
        }
        if (-not $_tauriSameRoot) {
            Write-StudioLine "ERROR: $envOverrideVar is not supported with --tauri." -ForegroundColor Red
            Write-StudioLine "       The desktop app uses the Windows profile .unsloth\studio root." -ForegroundColor Red
            Write-StudioLine "       Run install.ps1 without --tauri for custom-root shell installs," -ForegroundColor Yellow
            Write-StudioLine "       or unset the env var for default desktop installs." -ForegroundColor Yellow
            # No-op today (nothing redirected TMP/TEMP yet), kept so this stays correct if that changes.
            Restore-StudioTempEnvironment
            throw "$envOverrideVar is not supported with --tauri."
        }
    }


    # LOCALAPPDATA may be unset in service / CI contexts; guard Join-Path under Stop.
    $defaultDataDir = if ($env:LOCALAPPDATA -and -not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Join-Path $env:LOCALAPPDATA "Unsloth Studio"
    } else { $null }

    # Cleared per invocation: script scope outlives a run, and under `irm | iex` one session can
    # install twice. A stale $true makes the lock skip claiming a root it just created.
    $script:StudioEnvRootPath = $null
    $script:StudioEnvRootExisted = $false

    if ($envOverride) {
        # Tilde expansion: env vars aren't subject to it when quoted on assignment.
        if ($envOverride -eq "~" -or $envOverride -like "~/*" -or $envOverride -like "~\*") {
            $envOverride = (Join-Path $env:USERPROFILE $envOverride.Substring(1).TrimStart('/','\'))
        }
        try {
            # Recorded BEFORE the create: the lock later needs to know whether the root pre-existed.
            $script:StudioEnvRootPath = $envOverride
            $script:StudioEnvRootExisted = [System.IO.Directory]::Exists($envOverride)
            # .NET API: New-Item -Path treats brackets as wildcards (no -LiteralPath on PS 5.1).
            [System.IO.Directory]::CreateDirectory($envOverride) | Out-Null
            $StudioHome = (Resolve-Path -LiteralPath $envOverride).Path
            # Re-recorded against the resolved spelling, which is what the lock is handed.
            $script:StudioEnvRootPath = $StudioHome
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
    # Read before anything below creates them; see Write-StudioFinalPathDegraded. Every root
    # the process scan protects counts, so a legacy-layout install is not taken for a fresh one.
    $script:StudioInstallIsFresh = -not (
        (Test-Path -LiteralPath $VenvDir) -or
        (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe")) -or
        (Test-Path -LiteralPath (Join-Path $StudioHome ".venv")) -or
        ($env:USERPROFILE -and (Test-Path -LiteralPath (Join-Path $env:USERPROFILE "unsloth_studio"))))

    # Records which uv cache this install used so an update reuses it. Absolutized against
    # $PWD.Path (not the .NET process dir), as _absolutize_uv_cache_dir in install.sh.
    function Resolve-StudioUvCachePath {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Cache)
        if ([string]::IsNullOrEmpty($Cache)) { return $Cache }
        if (-not [System.IO.Path]::IsPathRooted($Cache)) {
            try {
                $base = if (-not [string]::IsNullOrWhiteSpace($env:UV_WORKING_DIR)) {
                    if ([System.IO.Path]::IsPathRooted($env:UV_WORKING_DIR)) {
                        $env:UV_WORKING_DIR
                    } else { Join-Path $PWD.Path $env:UV_WORKING_DIR }
                } else { $PWD.Path }
                $Cache = [System.IO.Path]::GetFullPath((Join-Path $base $Cache))
            } catch { return $Cache }
        }
        # Trim trailing separators (never past the root), as in _absolutize_uv_cache_dir.
        try {
            $root = [System.IO.Path]::GetPathRoot($Cache)
            while ($Cache.Length -gt $root.Length -and
                   ($Cache.EndsWith("\") -or $Cache.EndsWith("/"))) {
                $Cache = $Cache.Substring(0, $Cache.Length - 1)
            }
        } catch { }
        return $Cache
    }

    # Claim the root before anything lands in it, so the uninstaller need not guess. Never
    # fatal. A trusted sentinel must be a regular file, never a link.
    function Test-StudioPlainFile {
        param([string]$Path, [string]$Container)
        try {
            # Check the container too: a linked parent holding a genuine marker proves nothing.
            if (-not [string]::IsNullOrWhiteSpace($Container)) {
                $dir = Get-Item -LiteralPath $Container -Force -ErrorAction SilentlyContinue
                if ($dir -and (($dir.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) { return $false }
            }
            $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
            return (-not $item.PSIsContainer -and
                (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0))
        } catch { return $false }
    }

    # Only a root this run may take over (empty, or with an unambiguous marker); this
    # authorizes a delete. Emptiness test inline (the helper is defined later); unreadable
    # counts as occupied. Defined above its earliest reader.
    $script:StudioInstallLockFileName = ".unsloth-install.lock"

    function Write-StudioRootOwnerMarker {
        param([Parameter(Mandatory = $true)][string]$Root)
        try {
            $marker = Join-Path $Root ".unsloth-studio-owned"
            # Already ours and the right shape: leave it. This runs twice per install, and a run
            # killed between the delete and the write would lose the only proof this root is ours.
            if (Test-StudioPlainFile -Path $marker) { return }
            if (Test-Path -LiteralPath $Root) {
                $occupied = $true
                try {
                    # The lock's own file is not occupancy: it is created in this root before
                    # anything else, so counting it makes a fresh root look like somebody else's.
                    $occupied = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction Stop |
                        Where-Object { $_.Name -ne $script:StudioInstallLockFileName } |
                        Select-Object -First 1).Count -gt 0
                } catch { $occupied = $true }
                $claimable = (
                    $StudioRedirectMode -ne 'env' -or
                    (Test-StudioPlainFile -Path (Join-Path $Root "unsloth_studio\.unsloth-studio-owned") `
                        -Container (Join-Path $Root "unsloth_studio")) -or
                    (Test-StudioPlainFile -Path (Join-Path $Root "share\studio.conf") `
                        -Container (Join-Path $Root "share")) -or
                    -not $occupied
                )
                if (-not $claimable) { return }
            } else {
                # .NET API: New-Item -Path treats brackets as wildcards.
                [System.IO.Directory]::CreateDirectory($Root) | Out-Null
            }
            # Delete first, then confirm with Get-Item -Force: WriteAllText follows a file link and
            # Test-Path follows a dangling one.
            Remove-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue
            # CreateNew closes the check-then-write race. A dangling link is still followed (it can
            # only create an empty file).
            $markerStream = $null
            try {
                $markerStream = [System.IO.File]::Open($marker,
                    [System.IO.FileMode]::CreateNew,
                    [System.IO.FileAccess]::Write,
                    [System.IO.FileShare]::None)
            } catch {
                # No marker is fine; the venv writes its own later.
                return
            } finally {
                if ($markerStream) { try { $markerStream.Dispose() } catch {} }
            }
        } catch { }
    }

    function Write-StudioUvCacheMarker {
        param(
            [Parameter(Mandatory = $true)][string]$StudioRoot,
            [Parameter(Mandatory = $true)][string]$Cache
        )
        $markerDir = Join-Path $StudioRoot "cache"
        $markerFile = Join-Path $markerDir "uv-cache-dir"
        # Absolute: the update resolves this against ITS working directory.
        $Cache = Resolve-StudioUvCachePath -Cache $Cache
        # Remembered so a rollback can put it back.
        if (-not $script:StudioUvMarkerSaved) {
            # Under "Stop", Test-Path inside an ACL-denied directory throws.
            $existing = Test-Path -LiteralPath $markerFile -ErrorAction SilentlyContinue
            if ($existing) {
                # UTF8: 5.1 decodes BOM-less files with the ANSI code page.
                $previous = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 `
                    -ErrorAction SilentlyContinue
                # One we cannot read is one we cannot put back, so leave it alone.
                if ($null -eq $previous) { return }
                $script:StudioUvMarkerPrevious = $previous
                $script:StudioUvMarkerExisted = $true
            } else {
                $script:StudioUvMarkerPrevious = $null
                $script:StudioUvMarkerExisted = $false
            }
            $script:StudioUvMarkerSaved = $true
        }
        # CreateDirectory, not New-Item -Path: -Path treats [] in the root as a wildcard.
        if (-not (Test-Path -LiteralPath $markerDir -PathType Container -ErrorAction SilentlyContinue)) {
            try { [System.IO.Directory]::CreateDirectory($markerDir) | Out-Null } catch { }
        }
        # Removed first: Set-Content follows a symlink and would truncate its target.
        Remove-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue
        # And only once gone, since that removal fails non-terminatingly. Get-Item -Force
        # reports the link itself; Test-Path would follow it.
        if ($null -ne (Get-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue)) {
            return
        }
        Set-Content -LiteralPath $markerFile -Value $Cache `
            -Encoding utf8 -ErrorAction SilentlyContinue
    }

    function Restore-StudioUvCacheMarker {
        # AllowEmptyString and the blank guard: a throw here would be reported as a
        # failed environment restore.
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$StudioRoot)
        if ($script:StudioInstallCommitted) { return }
        if (-not $script:StudioUvMarkerSaved) { return }
        if ([string]::IsNullOrWhiteSpace($StudioRoot)) { return }
        $markerFile = Join-Path (Join-Path $StudioRoot "cache") "uv-cache-dir"
        Remove-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue
        $stillThere = $null -ne (Get-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue)
        if (-not $stillThere -and $script:StudioUvMarkerExisted -and $null -ne $script:StudioUvMarkerPrevious) {
            Set-Content -LiteralPath $markerFile -Value ([string]$script:StudioUvMarkerPrevious).TrimEnd("`r", "`n") `
                -Encoding utf8 -ErrorAction SilentlyContinue
        }
        $script:StudioUvMarkerSaved = $false
    }

    # uv --no-cache neither reads nor writes a cache, so nothing belongs in the marker.
    # Lowercased, not trimmed; clap's boolish set. Mirrors _uv_no_cache_requested.
    function Test-StudioUvNoCache {
        return (([string]$env:UV_NO_CACHE).ToLowerInvariant() -in @("1", "y", "yes", "t", "true", "on"))
    }

    # True for a name uv itself creates: <kind>-v<N>, whole suffix numeric. `archive-v0.backup`
    # is not uv's. Mirrors _uv_is_bucket_name, suffix from the LAST `-v` included.
    function Test-StudioUvBucketName {
        param(
            [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Name,
            # Off by default: warmth counting bytes under `Archive-V0` that uv looks for at
            # `archive-v0` picks a cache that cannot serve them. Only the write probe folds.
            [switch]$Fold
        )
        # Whole name, since the `-v` marker varies with the kind. Invariant: a Turkish
        # locale dots the I in `flat-Index-v4`.
        $lower = if ($Fold) { $Name.ToLowerInvariant() } else { $Name }
        $at = $lower.LastIndexOf("-v")
        if ($at -lt 0) { return $false }
        $suffix = $lower.Substring($at + 2)
        if ([string]::IsNullOrEmpty($suffix)) { return $false }
        # \A and \z, not ^ and $: in .NET `$` also matches before a final newline, so
        # `archive-v1<LF>` passed here while the sh helper rejected it.
        if (-not ($suffix -match '\A[0-9]+\z')) { return $false }
        # Every CacheBucket in uv 0.12.1 ($UvPinnedVersion) plus built-wheels; keep in step
        # with install.sh on a pin bump.
        return ($lower.Substring(0, $at) -in @(
            "archive", "binaries", "builds", "built-wheels", "environments", "flat-index",
            "git", "interpreter", "osv", "python", "sdists", "simple", "wheels"))
    }

    # Readable is not usable: create and delete for real in the root and in EVERY
    # <kind>-v<N> bucket. Mirrors the probe loop in _configure_uv_cache.
    function Test-StudioUvCacheWritable {
        param([Parameter(Mandatory = $true)][string]$Cache)
        $probeDirs = [System.Collections.Generic.List[string]]::new()
        $probeDirs.Add($Cache)
        # Measured like _uv_cache_is_writable does: NTFS folds unless fsutil
        # setCaseSensitiveInfo says otherwise, which is how a WSL-created tree behaves.
        $fold = $false
        $probeRoot = Join-Path $Cache (".unsloth-case-probe." +
            [guid]::NewGuid().ToString("N").Substring(0, 8) + "-A")
        try {
            [System.IO.Directory]::CreateDirectory($probeRoot) | Out-Null
            $fold = [System.IO.Directory]::Exists($probeRoot.Substring(0, $probeRoot.Length - 1) + "a")
        } catch { }
        Remove-Item -LiteralPath $probeRoot -Force -Recurse -ErrorAction SilentlyContinue
        try {
            foreach ($entry in [System.IO.Directory]::GetFileSystemEntries($Cache)) {
                $name = [System.IO.Path]::GetFileName($entry)
                # Only where a BUCKET should be. Anything else up here is not uv's to write,
                # and a cache-dir on a mount point has a root-owned lost+found that must not
                # condemn it.
                if (-not (Test-StudioUvBucketName -Name $name -Fold:$fold)) { continue }
                # A file, or a link dangling or not, is an existing path to uv's own create,
                # which answers "already exists", so uv refuses it.
                if (-not [System.IO.Directory]::Exists($entry)) { return $false }
                $probeDirs.Add($entry)
            }
        } catch { return $false }
        foreach ($dir in $probeDirs) {
            # A generated name, not a fixed one: a predictable path can be pre-created as a link
            # for the write to follow.
            $probe = Join-Path $dir (".unsloth-write-probe." + [guid]::NewGuid().ToString("N").Substring(0, 8))
            try { [System.IO.File]::WriteAllText($probe, "") } catch { return $false }
            try { Remove-Item -LiteralPath $probe -Force -ErrorAction Stop } catch { return $false }
        }
        return $true
    }

    # Warm means package BYTES in a bucket kind uv fills (wheels-* is metadata only on
    # uv 0.10). [ref] $Blocked: unreadable is not empty.
    function Test-StudioUvCachePopulated {
        param(
            [Parameter(Mandatory = $true)][string]$Cache,
            [ref]$Blocked
        )
        try {
            $buckets = Get-ChildItem -LiteralPath $Cache -Directory -Force -ErrorAction Stop |
                Where-Object {
                    # No -Fold: warmth is exact-case, so every name here is lowercase.
                    (Test-StudioUvBucketName -Name $_.Name) -and
                    ($_.Name.Substring(0, $_.Name.LastIndexOf("-v")) -in
                        @("archive", "builds", "built-wheels", "wheels", "sdists"))
                }
        } catch {
            $Blocked.Value = $true
            return $false
        }
        foreach ($bucket in $buckets) {
            # SilentlyContinue, not Stop: one denied subdirectory must not make a populated cache
            # read as empty. `find` also skips and continues.
            $scanErrors = $null
            $entry = Get-ChildItem -LiteralPath $bucket.FullName -File -Recurse -Force `
                    -ErrorAction SilentlyContinue -ErrorVariable scanErrors |
                Where-Object {
                    $_.Name -notin @("CACHEDIR.TAG", ".git", ".gitignore") -and
                    -not $_.Name.StartsWith(".unsloth-write-probe.") -and
                    $_.Extension -notin @(".lock", ".msgpack", ".http", ".rev")
                } |
                Select-Object -First 1
            if ($null -ne $entry) { return $true }
            if ($scanErrors -and $scanErrors.Count -gt 0) { $Blocked.Value = $true }
        }
        return $false
    }

    # Can we create AND fill this directory? A real create, since an existing unwritable one
    # satisfies CreateDirectory. For the Studio cache, which may not exist yet, where the bucket
    # probe above has nothing to walk.
    function Test-StudioUvCacheRootWritable {
        param([Parameter(Mandatory = $true)][string]$Cache)
        try {
            [System.IO.Directory]::CreateDirectory($Cache) | Out-Null
            $probe = Join-Path $Cache (".unsloth-write-probe." + [guid]::NewGuid().ToString("N").Substring(0, 8))
            [System.IO.File]::WriteAllText($probe, "")
        } catch { return $false }
        # A root can grant create and deny delete; retry, then ask whether the probe is GONE.
        for ($attempt = 0; $attempt -lt 3; $attempt++) {
            if ($attempt -gt 0) { Start-Sleep -Seconds 1 }
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
            if (-not [System.IO.File]::Exists($probe)) { return $true }
        }
        return $false
    }

    # Creatable AND fillable: the root helper answers for a cache that may not exist yet, the
    # bucket probe for one uv has already made buckets in. The fallback and the launch repoint
    # both need both, or they hand back a cache the candidate probe just rejected.
    function Test-StudioUvCacheUsable {
        param([Parameter(Mandatory = $true)][string]$Cache)
        if (-not (Test-StudioUvCacheRootWritable -Cache $Cache)) { return $false }
        return (Test-StudioUvCacheWritable -Cache $Cache)
    }

    # The cache THIS install last recorded, absolute, or "" when there is none. Read before the
    # selector writes its own. BOM and all: PowerShell 5.1 writes -Encoding utf8 WITH a BOM and
    # the update writes the same file BOM-less.
    function Read-StudioUvCacheMarker {
        param([Parameter(Mandatory = $true)][string]$StudioRoot)
        $markerFile = Join-Path (Join-Path $StudioRoot "cache") "uv-cache-dir"
        try {
            $recorded = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 -ErrorAction Stop
        } catch { return "" }
        if ($null -eq $recorded) { return "" }
        $recorded = ([string]$recorded).Trim([char]0xFEFF).Trim()
        if ([string]::IsNullOrWhiteSpace($recorded)) { return "" }
        return (Resolve-StudioUvCachePath -Cache $recorded)
    }

    function Set-StudioUvCacheEnvironment {
        param(
            [Parameter(Mandatory = $true)][string]$StudioRoot,
            [bool]$Isolated = $false,
            [string]$UvExecutable = ""
        )
        $studioCache = Join-Path (Join-Path $StudioRoot "cache") "uv"
        if (-not [string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
            $script:StudioUvCacheMode = "custom"
            # Absolute before anything uses it, so every phase of one install and the
            # marker name the same directory (see _absolutize_uv_cache_dir in install.sh).
            $env:UV_CACHE_DIR = Resolve-StudioUvCachePath -Cache $env:UV_CACHE_DIR
            if (Test-StudioUvNoCache) {
                step "uv cache" "preserving custom UV_CACHE_DIR ($env:UV_CACHE_DIR); uv caching is off (UV_NO_CACHE), so nothing is recorded"
                return
            }
            # Recorded like any other choice; a caller still outranks the marker.
            Write-StudioUvCacheMarker -StudioRoot $StudioRoot -Cache $env:UV_CACHE_DIR
            step "uv cache" "preserving custom UV_CACHE_DIR ($env:UV_CACHE_DIR)"
            return
        }

        if ($Isolated) {
            Set-Item -LiteralPath Env:UV_CACHE_DIR -Value $studioCache
            $script:StudioUvCacheMode = "isolated"
            if (-not (Test-StudioUvNoCache)) {
                Write-StudioUvCacheMarker -StudioRoot $StudioRoot -Cache $studioCache
            }
            step "uv cache" "forced Unsloth Studio cache isolation ($studioCache); already-cached packages may download again" "Yellow"
            return
        }

        # Nothing to select either: probing would touch a cache the caller told uv to leave alone.
        if (Test-StudioUvNoCache) {
            Set-Item -LiteralPath Env:UV_CACHE_DIR -Value $studioCache
            $script:StudioUvCacheMode = "studio"
            step "uv cache" "uv caching is off (UV_NO_CACHE); nothing to select or record"
            return
        }

        $defaultCache = $null
        $scanBlocked = $false
        $blockedCache = ""
        $warnCache = ""
        $chosenCache = ""
        try {
            # Ask uv so uv.toml / UV_CONFIG_FILE / the default count; remove a
            # blank inherited value so it cannot override them.
            Remove-Item -LiteralPath Env:UV_CACHE_DIR -ErrorAction SilentlyContinue
            if ($UvExecutable) {
                $resolvedCache = @(& $UvExecutable cache dir 2>$null)
                $uvCacheExit = $LASTEXITCODE
                # Last nonblank line, not [0]: a notice would become the path.
                $resolvedLine = $resolvedCache |
                    Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
                    Select-Object -Last 1
                if ($uvCacheExit -eq 0 -and $null -ne $resolvedLine) {
                    $defaultCache = ([string]$resolvedLine).Trim()
                }
            }
            if (-not $defaultCache -and -not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
                $defaultCache = Join-Path (Join-Path $env:LOCALAPPDATA "uv") "cache"
            }
            # A relative cache-dir comes back verbatim and uv resolves it against its working
            # directory, so scanning it as written inspects a same-named directory beside us.
            if ($defaultCache) { $defaultCache = Resolve-StudioUvCachePath -Cache $defaultCache }

            # The recorded cache outranks uv's default while still warm. An unmarked (pre-marker)
            # Studio cache goes LAST. Same order as install.sh and unsloth_cli/commands/studio.py.
            $recorded = Read-StudioUvCacheMarker -StudioRoot $StudioRoot
            # A marker naming a directory that is gone is a stale pointer, not a decision, so
            # it lands in the same place.
            $unmarkedStudio = if ($recorded -and (Test-Path -LiteralPath $recorded -PathType Container -ErrorAction SilentlyContinue)) {
                $null
            } else { $studioCache }
            foreach ($candidate in @($recorded, $defaultCache, $unmarkedStudio)) {
                if ([string]::IsNullOrWhiteSpace([string]$candidate)) { continue }
                if ($chosenCache) { continue }
                # SilentlyContinue: under "Stop" Test-Path in an ACL-denied dir throws and kills the loop.
                if (-not (Test-Path -LiteralPath $candidate -PathType Container -ErrorAction SilentlyContinue)) { continue }
                # Per candidate, so the message names the cache actually refused.
                $candidateBlocked = $false
                $populated = Test-StudioUvCachePopulated -Cache $candidate -Blocked ([ref]$candidateBlocked)
                if ($candidateBlocked) {
                    $scanBlocked = $true
                    if (-not $blockedCache) { $blockedCache = $candidate }
                }
                if (-not $populated) { continue }
                if (Test-StudioUvCacheWritable -Cache $candidate) {
                    $chosenCache = $candidate
                } elseif (-not $warnCache) {
                    $warnCache = $candidate
                }
            }
        } catch {
            # An uninspectable cache is not an install error; isolation is safe.
            $chosenCache = ""
            $scanBlocked = $true
        }

        if ($chosenCache) {
            $selectedCache = $chosenCache
            # studio, not shared, when the choice IS the Studio cache: the launch repoint below
            # only has to move a cache that is not already ours.
            $script:StudioUvCacheMode = if ($chosenCache -eq $studioCache) { "studio" } else { "shared" }
        } else {
            $selectedCache = $studioCache
            $script:StudioUvCacheMode = "studio"
            # The fallback must be usable (root AND buckets); otherwise prefer the populated cache
            # that only failed the probe, then uv's default.
            if (-not (Test-StudioUvCacheUsable -Cache $studioCache)) {
                # A warm cache that merely failed the probe beats a cold one.
                if ($warnCache) {
                    $selectedCache = $warnCache
                    if ($warnCache -ne $studioCache) {
                        $script:StudioUvCacheMode = "shared"
                    }
                } elseif ($defaultCache -and $defaultCache -ne $studioCache -and
                          (Test-StudioUvCacheUsable -Cache $defaultCache)) {
                    $selectedCache = $defaultCache
                    $script:StudioUvCacheMode = "shared"
                }
            }
        }
        Set-Item -LiteralPath Env:UV_CACHE_DIR -Value $selectedCache
        Write-StudioUvCacheMarker -StudioRoot $StudioRoot -Cache $selectedCache

        switch ($script:StudioUvCacheMode) {
            "shared" {
                step "uv cache" "reusing existing shared cache ($selectedCache) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate"
            }
            "studio" {
                if ($chosenCache) {
                    step "uv cache" "reusing this install's Unsloth Studio cache ($selectedCache)"
                # Never about the directory we are falling back TO: the Studio cache is itself
                # a candidate now, so it can be the one refused, and naming it claims a fallback
                # that did not happen.
                } elseif ($scanBlocked -and -not [string]::IsNullOrWhiteSpace([string]$blockedCache) -and $blockedCache -ne $selectedCache) {
                    step "uv cache" "using new Unsloth Studio-owned cache ($selectedCache); part of $blockedCache could not be read, so cached packages may download again" "Yellow"
                } elseif ($scanBlocked -and [string]::IsNullOrWhiteSpace([string]$blockedCache)) {
                    step "uv cache" "using new Unsloth Studio-owned cache ($selectedCache); the existing uv cache could not be inspected, so cached packages may download again" "Yellow"
                # Warm and still here means the write probe refused it.
                } elseif ($warnCache -and $warnCache -ne $selectedCache) {
                    step "uv cache" "using new Unsloth Studio-owned cache ($selectedCache); $warnCache is populated but not writable, so cached packages may download again" "Yellow"
                } else {
                    step "uv cache" "using new Unsloth Studio-owned cache ($selectedCache)"
                }
            }
        }
    }

    function Set-StudioUvCacheForLaunch {
        param([Parameter(Mandatory = $true)][string]$StudioRoot)
        if ($script:StudioUvCacheMode -ne "shared") { return }
        # Only repoint the launch to a cache the backend can fill (root AND buckets). Mirrors
        # _prepare_studio_uv_cache_for_launch.
        $launchCache = Join-Path (Join-Path $StudioRoot "cache") "uv"
        if (-not (Test-StudioUvCacheUsable -Cache $launchCache)) { return }
        Set-Item -LiteralPath Env:UV_CACHE_DIR -Value $launchCache
    }

    function Restore-StudioUvCacheEnvironment {
        param(
            [bool]$WasPresent,
            [AllowNull()][string]$PreviousValue
        )
        # The provider, not the .NET API: that maps empty to deletion on Windows.
        if ($WasPresent) {
            Set-Item -LiteralPath Env:UV_CACHE_DIR -Value $PreviousValue
        } else {
            Remove-Item -LiteralPath Env:UV_CACHE_DIR -ErrorAction SilentlyContinue
        }
    }

    $Rule = [string]::new([char]0x2500, 52)
    $Sloth = [char]::ConvertFromUtf32(0x1F9A5)

    function Enable-StudioVirtualTerminal {
        if ($env:NO_COLOR) { return $false }
        # Redirected stdout (the desktop app's pipe) has no virtual terminal.
        if ($script:StudioStdoutRedirected) { return $false }

        # The console host already enables VT at startup and verifies it, so this property cannot
        # read True while VT is off (measured in PR #10984; see
        # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)). [bool] maps a no-UI host's
        # $null to $false; try/catch covers a caller's Set-StrictMode.
        try { return [bool]$Host.UI.SupportsVirtualTerminal } catch { return $false }
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

    # Here so its warning lands under the banner, before the temp dir's first user.
    Initialize-StudioTempEnvironment

    # ── Helper: refresh PATH from registry (deduplicating entries) ──
    # Merge order: venv Scripts (if active) > Machine > User > current $env:Path.
    # Active conda prefixes, deepest first: CONDA_PREFIX, CONDA_PREFIX_1.., and CONDA_EXE's root.
    function Get-ActiveCondaPrefixes {
        $prefixes = New-Object System.Collections.Generic.List[string]
        if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) { $prefixes.Add($env:CONDA_PREFIX) }
        # Enumerated from CONDA_SHLVL with a ceiling, not a fixed list.
        $levels = 0
        if (-not [string]::IsNullOrWhiteSpace($env:CONDA_SHLVL)) {
            [void][int]::TryParse($env:CONDA_SHLVL, [ref]$levels)
        }
        if ($levels -lt 1) { $levels = 1 }
        if ($levels -gt 64) { $levels = 64 }
        for ($level = 1; $level -le $levels; $level++) {
            $value = [Environment]::GetEnvironmentVariable("CONDA_PREFIX_$level")
            if (-not [string]::IsNullOrWhiteSpace($value)) { $prefixes.Add($value) }
        }
        if (-not [string]::IsNullOrWhiteSpace($env:_CONDA_ROOT)) { $prefixes.Add($env:_CONDA_ROOT) }
        if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
            # Regex, not Split-Path, which is provider-aware on non-Windows test hosts.
            $scripts = $env:CONDA_EXE -replace '[\\/][^\\/]*$', ''
            $root = $scripts -replace '[\\/][^\\/]*$', ''
            if ($root -and $root -ne $env:CONDA_EXE) { $prefixes.Add($root) }
        }
        return $prefixes
    }

    # Is $Path inside one of $Prefixes? On a directory boundary, so "C:\conda-backup" is not
    # dragged to the front along with "C:\conda".
    function Test-PathUnderCondaPrefix {
        param([string]$Path, $Prefixes)
        if ([string]::IsNullOrWhiteSpace($Path) -or -not $Prefixes) { return $false }
        $candidate = [Environment]::ExpandEnvironmentVariables($Path).Trim().Trim('"').TrimEnd('\')
        if (-not $candidate) { return $false }
        foreach ($prefix in $Prefixes) {
            $normalized = [Environment]::ExpandEnvironmentVariables($prefix).Trim().Trim('"').TrimEnd('\')
            if (-not $normalized) { continue }
            if ($candidate -ieq $normalized) { return $true }
            if ($candidate.StartsWith($normalized + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
        return $false
    }

    function Refresh-SessionPath {
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $venvScripts = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts" } else { $null }
        # An activated conda env lives only in the process PATH, so keep it in front.
        $condaFront = @()
        if (Test-ActiveCondaEnvironment) {
            $prefixes = Get-ActiveCondaPrefixes
            if ($prefixes) {
                foreach ($entry in ($env:Path -split ";")) {
                    if (Test-PathUnderCondaPrefix -Path $entry -Prefixes $prefixes) {
                        $condaFront += $entry
                    }
                }
            } else {
                # Only CONDA_DEFAULT_ENV, no directory: keep the caller's PATH whole and in front.
                $condaFront = @($env:Path)
            }
        }
        $sources = @()
        # Still first: this is the environment the installer is driving, and it is ours.
        if ($venvScripts) { $sources += $venvScripts }
        $sources += $condaFront
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

    # ── Helper: is a conda environment ACTIVE in this session? ──
    # CONDA_PREFIX or CONDA_DEFAULT_ENV, not merely conda being installed.
    function Test-ActiveCondaEnvironment {
        foreach ($condaVar in @($env:CONDA_PREFIX, $env:CONDA_DEFAULT_ENV)) {
            if (-not [string]::IsNullOrWhiteSpace($condaVar)) { return $true }
        }
        return $false
    }

    # Persistent writes must append under an active conda env: a prepend outlives the
    # activation and corrupts Anaconda base (#5871).
    function Resolve-UserPathPosition {
        param(
            [ValidateSet('Append','Prepend')]
            [string]$Position = 'Append'
        )
        if ($Position -eq 'Prepend' -and (Test-ActiveCondaEnvironment)) { return 'Append' }
        return $Position
    }

    # ── Helper: safely add a directory to the persistent User PATH ──
    # Direct registry access preserves REG_EXPAND_SZ (dotnet/runtime#1442).
    function Add-ToUserPath {
        param(
            [Parameter(Mandatory = $true)][string]$Directory,
            [ValidateSet('Append','Prepend')]
            [string]$Position = 'Append'
        )
        # Every persistent PATH write goes through Resolve-UserPathPosition, so an active
        # conda environment cannot be demoted by any call site. Announced once per run: the
        # installer makes several of these writes and one note is information, five is noise.
        $resolvedPosition = Resolve-UserPathPosition -Position $Position
        # A downgraded request must MOVE an existing front entry, or the #5871 demotion stays armed.
        $positionDowngraded = ($resolvedPosition -ne $Position)
        if ($positionDowngraded) {
            if (-not $script:CondaPathPositionNoted) {
                $script:CondaPathPositionNoted = $true
                substep "conda environment active ($env:CONDA_PREFIX) -- appending to PATH instead of prepending, so conda keeps priority." "Yellow"
            }
            $Position = $resolvedPosition
        }
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
                # Not $positionDowngraded: that request has to reposition an existing front entry.
                # Already at the back is still a no-op, caught by the $newPath -ceq $rawPath check.
                if ($alreadyPresent -and $Position -eq 'Append' -and -not $positionDowngraded) {
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
            # One composed record so a redirected consumer cannot split label from value.
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

        # Read attributes through the filesystem API rather than Get-Item. Getting
        # attributes needs only FILE_READ_ATTRIBUTES, which survives the ACLs that
        # deny reading the directory itself, so Get-Item returns nothing in exactly
        # the case this detail is meant to describe and the caller silently loses
        # every hint below.
        $attrs = $null
        try { $attrs = [System.IO.File]::GetAttributes($Path) } catch { $attrs = $null }
        if ($null -eq $attrs) { return "" }

        # Ordered by how much each one changes the fix. takeown/icacls cannot help
        # with any of the first three, so name them before falling back to ACLs.
        # On a directory this attribute only means new descendants are encrypted
        # by default, so listing it never needs the key and a denial there is an
        # ACL. Only a file's own streams are unreadable without the certificate.
        $isDirectory = ([int]$attrs -band [int][System.IO.FileAttributes]::Directory) -ne 0
        if (-not $isDirectory -and ($attrs -band [System.IO.FileAttributes]::Encrypted)) {
            return " (it is EFS-encrypted, so it stays unreadable even elevated unless the encrypting account or its recovery certificate is available)"
        }
        # Offline plus either recall attribute is a cloud placeholder, typically
        # OneDrive Files On-Demand that cannot hydrate.
        # RECALL_ON_OPEN (0x40000) and RECALL_ON_DATA_ACCESS (0x400000) are absent
        # from the FileAttributes enum on Windows PowerShell 5.1, so test the bits.
        $offline = ([int]$attrs -band [int][System.IO.FileAttributes]::Offline) -ne 0
        $recall = ([int]$attrs -band (0x00040000 -bor 0x00400000)) -ne 0
        if ($offline -or $recall) {
            return " (it is a cloud placeholder, e.g. OneDrive Files On-Demand, that cannot be hydrated right now)"
        }
        if ($attrs -band [System.IO.FileAttributes]::ReparsePoint) {
            $target = $null
            try {
                $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
                if ($item -is [System.IO.FileSystemInfo]) { $target = $item.Target }
            } catch { $target = $null }
            # PS 5.1 exposes .Target as a collection; PS 7 as a string.
            if ($target) { return " (it is a link to $(@($target) -join ', '))" }
            return " (it is a link)"
        }
        return ""
    }

    # Which security software could be denying this path, as a possibility.
    #
    # Worth naming because takeown and icacls cannot clear a filter-driver block
    # and elevation does not either, so a user whose antivirus is holding the
    # folder is otherwise sent round the takeown loop for as long as they are
    # willing. Nothing readable from here attributes the specific denial though,
    # so this names the candidate and the log that settles it and never
    # contradicts the ACL advice it follows.
    #
    # Defender's Controlled folder access modes are 0 Disabled, 1 Enabled,
    # 2 AuditMode, 3 BlockDiskModificationOnly, 4 AuditDiskModificationOnly. Only
    # 1 gates file access; 3 and 4 are direct disk-sector writes rather than
    # files, so neither explains a denied folder.
    #
    # When Defender is not it, name whichever antivirus is registered and running
    # instead: third-party suites ship the same protected-folders feature under
    # their own product names, and the user cannot act on advice that does not say
    # which product to open. Which suite ships what is recorded in
    # tests/studio/test_installer_av_shapes.py and deliberately not repeated here,
    # because this file is scanned in full before a line of it runs and a comment
    # listing security products raises the score of the very file explaining it.
    # Nothing below hard-codes a product: it reads what SecurityCenter2 registered.
    #
    # Answers "" whenever it cannot tell, so a machine with no Defender module
    # and no SecurityCenter registration reads the same as one that says no.
    function Get-SecuritySoftwareNote {
        $mode = $null
        try {
            if (Get-Command Get-MpPreference -ErrorAction SilentlyContinue) {
                $mode = [int](Get-MpPreference -ErrorAction Stop).EnableControlledFolderAccess
            }
        } catch { $mode = $null }
        if ($mode -eq 1) {
            return "Controlled folder access is ON here, and it gates writes to protected folders whatever your privileges are, so where it is the cause takeown and icacls will not clear it: Windows Defender Operational events 1123 and 1124 say whether it stopped this path, and Virus & threat protection > Ransomware protection > Allow an app is where to allow Unsloth"
        }
        # SecurityCenter2 is the registration every consumer antivirus makes, and
        # it is absent on Server SKUs, so this stays best-effort. productState
        # packs the running state in 0xF000: 0x1000 on, 0x2000 snoozed, 0 off.
        # A product that is not running cannot be holding the folder and naming it
        # sends the user to the wrong console; a state we cannot read proves
        # nothing either way, so it is kept.
        $others = @()
        try {
            $others = @(Get-CimInstance -Namespace "root/SecurityCenter2" -ClassName AntiVirusProduct -ErrorAction Stop |
                Where-Object { $state = $_.productState -as [uint32]; ($null -eq $state) -or (($state -band 0xF000) -eq 0x1000) } |
                ForEach-Object { [string]$_.displayName } |
                Where-Object { $_ -and $_ -notmatch "Windows Defender" -and $_ -notmatch "Microsoft Defender" })
        } catch { $others = @() }
        if ($others.Count -gt 0) {
            $names = ($others | Select-Object -Unique) -join ", "
            return "$names is running here, and its ransomware or protected-folder feature can deny a path whatever your privileges are. If takeown and icacls do not clear this, look in $names for a block on this folder, and add an exclusion for it and for Unsloth"
        }
        if ($mode -eq 2) {
            return "Controlled folder access is in audit mode, so it is logging rather than blocking and is not the cause here; Windows Defender Operational events 1123 and 1124 name whatever it did stop"
        }
        return ""
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
        # After the generic line, since this one either confirms it or rules it
        # out, and an empty answer must leave the generic advice standing.
        $securitySoftware = Get-SecuritySoftwareNote
        if ($securitySoftware) { substep $securitySoftware "Yellow" }
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

    # The master root storage_roots.unsloth_home() reads. llama.cpp, node and whisper.cpp sit
    # BESIDE studio\ under it, so deriving them from $StudioHome would put them one level below
    # where every runtime resolver looks.
    function Get-MasterRootOverride {
        if ([string]::IsNullOrWhiteSpace($env:UNSLOTH_HOME)) { return $null }
        $value = $env:UNSLOTH_HOME.Trim()
        if ($value -eq "~") {
            $value = $env:USERPROFILE
        } elseif ($value -like "~/*" -or $value -like "~\*") {
            $value = (Join-Path $env:USERPROFILE $value.Substring(1).TrimStart('/', '\'))
        }
        return (Get-CanonicalDir -Path $value)
    }

    # Explicit staging root, the master root, the shared default cache, or the custom Unsloth
    # home's tree.
    function Get-ManagedLlamaCppDir {
        param([AllowNull()][string]$StagingRoot = $null)

        if ($StagingRoot) {
            return (Join-Path $StagingRoot "llama.cpp")
        }
        $masterRoot = Get-MasterRootOverride
        if ($masterRoot) {
            return (Join-Path $masterRoot "llama.cpp")
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
        # A denied custom home cannot be claimed as an Unsloth-managed cache. Computed rather
        # than read off $RuntimeRootIsCustom: this runs beside the line that defines it.
        $homeIsCustom = (Test-StudioHomeIsCustom) -or [bool](Get-MasterRootOverride)
        # Preserve user-supplied wording when either override names this tree, or
        # names a build inside it: moving or deleting this folder takes that build
        # with it, and the later --with-llama-cpp-dir check then aborts on a path
        # we made disappear.
        $suppliedDir = if ($WithLlamaCppDir) { $WithLlamaCppDir } else { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR }
        # Walking the ancestors beats comparing the two canonical strings: an
        # override that does not exist yet cannot be resolved, so a prefix test
        # would compare a resolved path against an unresolved one and miss.
        $userSupplied = $false
        if (-not [string]::IsNullOrWhiteSpace($suppliedDir)) {
            $canonicalDir = [string](Get-CanonicalDir -Path $dir)
            $probe = [string](Get-CanonicalDir -Path $suppliedDir)
            while (-not [string]::IsNullOrWhiteSpace($probe)) {
                if ([string](Get-CanonicalDir -Path $probe) -eq $canonicalDir) {
                    $userSupplied = $true
                    break
                }
                $parent = ""
                try { $parent = [string](Split-Path -Parent $probe) } catch { $parent = "" }
                if ($parent -eq $probe) { break }
                $probe = $parent
            }
        }

        # Only the default branch below tells the user to delete this folder, so
        # only that case may move it. A user-supplied build is not ours to touch,
        # and an unreadable custom home cannot be confirmed as a managed cache.
        # Renaming needs DELETE on the folder plus write on its parent, neither of
        # which is read access, so this recovers denials that takeown and icacls
        # do not: the folder is a managed cache that setup reinstalls anyway.
        # Setup never makes this a link, so a link here is something the user
        # arranged, pointing at a build we were not told about. Moving it would
        # silently change which tree they run without touching the one they were
        # protecting, so it is left alone and named in the guidance instead.
        $isLink = $false
        try {
            $linkAttrs = [System.IO.File]::GetAttributes($dir)
            $isLink = ([int]$linkAttrs -band [int][System.IO.FileAttributes]::ReparsePoint) -ne 0
        } catch {
            # Unreadable attributes prove nothing, and "proves nothing" must not
            # mean "movable": fall through to the guidance rather than guess.
            $isLink = $true
        }
        if (-not $userSupplied -and -not $homeIsCustom -and -not $isLink) {
            $asideDir = "$dir.denied-$(Get-Date -Format 'yyyyMMddHHmmss')"
            $moved = $false
            try {
                # [System.IO.Directory]::Move, not Move-Item. Move-Item falls back to
                # copy-then-delete when the rename fails, which creates $asideDir and then
                # dies on the unreadable contents, leaving a stray llama.cpp.denied-* folder
                # beside the original on every run. Directory.Move is a bare rename: it
                # either moves the tree or throws having created nothing.
                # Measured on windows-latest, denying each shape on the folder itself:
                #   (OI)(CI)(RX)  rename refused, Move-Item left a stray folder
                #   (OI)(CI)(R)   rename refused, Move-Item left a stray folder
                #   (RX)          rename refused, Move-Item left a stray folder
                #   (DE)          rename SUCCEEDED, both ways
                # So on Windows a read denial always refuses the rename (the open asks for
                # SYNCHRONIZE, which every read deny removes) and this recovery cannot fire;
                # denying DELETE, which sounds like the blocker, does not stop it. On POSIX
                # the rename needs only write+execute on the parent, so the recovery is real
                # there and is why this stays rather than being deleted.
                [System.IO.Directory]::Move($dir, $asideDir)
                $moved = $true
            } catch {
                # Expected when the denial covers the rename; fall through to guidance.
                # Nothing to clean up: Directory.Move creates nothing when it throws.
            }
            if ($moved) {
                step "permissions" "llama.cpp install at $dir could not be read, so it was moved aside" "Yellow"
                substep "Moved to $asideDir; it is a managed cache and setup reinstalls it" "Yellow"
                substep "Delete the moved folder once access is restored, it is no longer used" "Yellow"
                Write-StudioLine ""
                return $null
            }
            # This runs before the install lock, so a second run can have moved
            # the folder in between. Re-probe rather than report a denial for a
            # path that is no longer there and stop an install that can proceed.
            if ((Get-LlamaCppInstallReadState -Path $dir) -ne "Denied") { return $null }
        }

        $reason = Write-PathAccessDenied -Path $dir -Label "llama.cpp install" `
            -UserSupplied:$userSupplied -OwnershipUnverified:$homeIsCustom
        substep "Stopping here, before phase 1: nothing has been downloaded or installed" "Yellow"
        substep "Fix access, then run the same install, setup, or update command again" "Yellow"
        Write-StudioLine ""
        return "$reason Nothing was installed."
    }

    function Test-MirrorConfigured {
        param([ValidateSet('uv', 'pip')][string]$Tool)
        if ($Tool -eq 'uv') {
            if ("$env:UV_DEFAULT_INDEX$env:UV_INDEX_URL$env:UV_INDEX$env:UV_EXTRA_INDEX_URL") { return $true }
            $pattern = '^\s*(\[\[(tool\.uv\.)?index\]\]|(pip\.)?(index|index-url|default-index|extra-index-url|no-index)\s*=)'
            $files = @($env:UV_CONFIG_FILE, "$env:APPDATA\uv\uv.toml", "$env:ProgramData\uv\uv.toml")
            $dir = (Get-Location -PSProvider FileSystem).ProviderPath
            while ($dir) {
                $pyproject = Join-Path $dir 'pyproject.toml'
                if (Test-Path -LiteralPath (Join-Path $dir 'uv.toml') -PathType Leaf) { $files += Join-Path $dir 'uv.toml'; break }
                if ((Test-Path -LiteralPath $pyproject -PathType Leaf) -and
                    (Select-String -LiteralPath $pyproject -Pattern '^\s*\[+tool\.uv(\.|\])' -Quiet -ErrorAction SilentlyContinue)) { $files += $pyproject; break }
                $dir = Split-Path -Parent $dir
            }
        } else {
            if ("$env:PIP_INDEX_URL$env:PIP_EXTRA_INDEX_URL$env:PIP_NO_INDEX") { return $true }
            $pattern = '^\s*(index[-_]url|extra[-_]index[-_]url|no[-_]index)\s*[=:]'
            $files = @($env:PIP_CONFIG_FILE, $(if ($venv = Get-Variable VenvDir -ValueOnly -ErrorAction SilentlyContinue) { Join-Path $venv 'pip.ini' }), "$env:APPDATA\pip\pip.ini", "$env:USERPROFILE\pip\pip.ini", "$env:ProgramData\pip\pip.ini")
        }
        foreach ($file in $files) {
            if ($file -and (Test-Path -LiteralPath $file -PathType Leaf) -and
                (Select-String -LiteralPath $file -Pattern $pattern -Quiet -ErrorAction SilentlyContinue)) {
                return $true
            }
        }
        return $false
    }

    function Start-MirrorProbe {
        param([string[]]$Urls, [double]$Seconds, [long]$LastByte)
        $state = @{ Urls = $Urls; Seconds = $Seconds; LastByte = $LastByte; Clock = [System.Diagnostics.Stopwatch]::StartNew(); Requests = @{}; Heads = @{}; InTime = @{}; Bodies = @{}; Buffers = @{} }
        $deadline = [System.Threading.Tasks.Task]::Delay([int]($Seconds * 1000))
        foreach ($url in $Urls) {
            $state.Requests[$url] = [System.Net.WebRequest]::Create($url)
            # The CERNET mirrors answer 403 to a request without a User-Agent; a fresh connection keeps every probe cold, as curl's are.
            $state.Requests[$url].UserAgent = 'unsloth-installer'
            $state.Requests[$url].KeepAlive = $false
            $state.Requests[$url].AddRange(0, $LastByte)
            $state.Heads[$url] = $state.Requests[$url].GetResponseAsync()
            $state.InTime[$url] = [System.Threading.Tasks.Task]::WhenAny([System.Threading.Tasks.Task[]]@($state.Heads[$url], $deadline))
        }
        return $state
    }

    function Wait-MirrorProbe {
        param($Probe)
        $results = @{}
        $measure = { param($url) @([int]$Probe.Heads[$url].Result.StatusCode, [long]($Probe.Buffers[$url].Position / $Probe.Clock.Elapsed.TotalSeconds)) }
        try {
            while ($true) {
                foreach ($url in $Probe.Urls) {
                    if ($results.ContainsKey($url) -or -not $Probe.InTime[$url].IsCompleted) { continue }
                    if (-not [object]::ReferenceEquals($Probe.InTime[$url].Result, $Probe.Heads[$url])) {
                        $results[$url] = @(0, [long]0)
                    } elseif ($Probe.Heads[$url].Status -ne 'RanToCompletion') {
                        $failure = $Probe.Heads[$url].Exception.InnerException -as [System.Net.WebException]
                        $results[$url] = @($(if ($failure -and $failure.Response) { [int]$failure.Response.StatusCode } else { 0 }), [long]0)
                    } elseif (-not $Probe.Bodies.ContainsKey($url)) {
                        $Probe.Buffers[$url] = [System.IO.MemoryStream]::new([byte[]]::new($Probe.LastByte + 65537))
                        $Probe.Bodies[$url] = $Probe.Heads[$url].Result.GetResponseStream().CopyToAsync($Probe.Buffers[$url], 65536)
                    } elseif ($Probe.Bodies[$url].IsCompleted) {
                        $results[$url] = & $measure $url
                    }
                }
                $pending = @($Probe.Urls | Where-Object { -not $results.ContainsKey($_) } | ForEach-Object { if ($Probe.Bodies.ContainsKey($_)) { $Probe.Bodies[$_] } else { $Probe.InTime[$_] } })
                $left = [int](($Probe.Seconds - $Probe.Clock.Elapsed.TotalSeconds) * 1000)
                if ($pending.Count -eq 0 -or $left -le 0) { break }
                [void][System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]$pending, $left)
            }
            foreach ($url in $Probe.Urls) {
                if ($results.ContainsKey($url)) { continue }
                $results[$url] = if ($Probe.Buffers.ContainsKey($url)) { & $measure $url } else { @(0, [long]0) }
            }
        } finally {
            foreach ($url in @($Probe.Requests.Keys)) {
                $Probe.Requests[$url].Abort()
                if ($Probe.Heads[$url].Status -eq 'RanToCompletion') { $Probe.Heads[$url].Result.Dispose() }
            }
        }
        return $results
    }

    function Get-MirrorDnsServers {
        try {
            [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | Where-Object { $_.OperationalStatus -eq 'Up' } |
                ForEach-Object { $_.GetIPProperties().DnsAddresses } | ForEach-Object { "$_" }
        } catch {}
    }

    # No network call: a mainland China time zone, or a resolver from a mainland public DNS or cloud, as _mirror_in_china in install.sh.
    function Test-MirrorInChina {
        try { if ((Get-TimeZone).Id -in 'China Standard Time', 'Asia/Shanghai', 'Asia/Chongqing', 'Asia/Chungking', 'Asia/Harbin', 'Asia/Urumqi', 'Asia/Kashgar', 'PRC') { return $true } } catch {}
        return [bool](@(Get-MirrorDnsServers) -match '^(223\.5\.5\.5|223\.6\.6\.6|119\.29\.29\.29|114\.114\.11[45]\.11[0459]|182\.254\.116\.116|119\.28\.28\.28|180\.76\.76\.76|1\.2\.4\.8|210\.2\.4\.8|100\.100\.2\.13[68]|183\.60\.8[23]\.(19|98))$')
    }

    function Invoke-MirrorFallback {
        param([switch]$SpareOnly)
        $optIn = "$env:UNSLOTH_MIRROR_FALLBACK".Trim()
        # When off, a retry state inherited from a parent is dropped so nothing downstream acts on it.
        if ($optIn -match '^(0|false|no|off)$' -or ($optIn -notmatch '^(1|true|yes|on)$' -and -not (Test-MirrorInChina))) {
            Remove-Item Env:_UNSLOTH_MIRROR_SPARE -ErrorAction SilentlyContinue
            return
        }
        if ($env:_UNSLOTH_MIRROR_PROBED) { return }
        if (-not $SpareOnly) { $env:_UNSLOTH_MIRROR_PROBED = '1' }
        $cernet = 'https://tuna.mirrors.cernet.edu.cn'
        $npmMirror = 'https://registry.npmmirror.com'
        $pypiMirror = "$cernet/pypi/web/simple"
        $minBps = 1MB
        $useUv = -not (Test-MirrorConfigured -Tool uv)
        $usePip = -not (Test-MirrorConfigured -Tool pip)
        $uvWheel = 'packages/72/d6/207945fe69903b9794e2ef3e42608c91a59972567343a6719078d99c71f7/uv-0.12.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
        $torchWheel = 'whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
        $nodeTarball = 'v24.18.0/node-v24.18.0-linux-x64.tar.gz'
        $artifact = @{
            'pypi' = "https://files.pythonhosted.org/$uvWheel"; 'cernet-pypi' = "$cernet/pypi/web/$uvWheel"
            'torch' = "https://download-r2.pytorch.org/$torchWheel"; 'cernet-torch' = "$cernet/pytorch/$torchWheel"
            'node' = "https://nodejs.org/dist/$nodeTarball"; 'npmmirror-node' = "$npmMirror/-/binary/node/$nodeTarball"
            'npm' = 'https://registry.npmjs.org/typescript/-/typescript-5.9.3.tgz'; 'npmmirror' = "$npmMirror/typescript/-/typescript-5.9.3.tgz"
            'astral' = 'https://releases.astral.sh/github/uv/releases/download/0.12.1/uv-x86_64-unknown-linux-gnu.tar.gz'
        }
        $hosts = [ordered]@{}
        if ($useUv -or $usePip) { $hosts['pypi'] = @('pypi', 'cernet-pypi', $pypiMirror, 'https://pypi.org/simple/uv/', "$pypiMirror/uv/") }
        if (-not "$env:UNSLOTH_PYTORCH_MIRROR$env:UNSLOTH_TORCH_INDEX_URL") {
            $hosts['torch'] = @('torch', 'cernet-torch', "$cernet/pytorch/whl", 'https://download.pytorch.org/whl/cpu/torch/', "$cernet/pytorch/whl/cpu/torch/")
        }
        if (-not $env:UNSLOTH_NODE_MIRROR) { $hosts['node'] = @('node', 'npmmirror-node', "$npmMirror/-/binary/node", $null, $null) }
        if (-not "$env:UNSLOTH_NPM_REGISTRY$env:NPM_CONFIG_REGISTRY") { $hosts['npm'] = @('npm', 'npmmirror', $npmMirror, $null, $null) }
        if (-not "$env:UNSLOTH_UV_WHEEL_MIRROR$env:UV_DOWNLOAD_URL$env:INSTALLER_DOWNLOAD_URL$env:UV_INSTALLER_GHE_BASE_URL$env:UV_INSTALLER_GITHUB_BASE_URL") {
            $hosts['uvbin'] = @('astral', 'cernet-pypi', "$cernet/pypi/web", $null, $null)
        }
        if ($hosts.Count -eq 0) { return }
        $varsOf = {
            param($name)
            $to = if ($hosts.Contains($name)) { $hosts[$name][2] }
            switch ($name) {
                'pypi' {
                    if ($useUv) { "UV_DEFAULT_INDEX=$to" }
                    if ($usePip) { "PIP_INDEX_URL=$to" }
                }
                'unsynced' {
                    # Only for one rerun: uv's unsafe-first-match fetches every package from every index, and fails outright when one is unreachable.
                    if ($useUv) {
                        'UV_DEFAULT_INDEX=https://pypi.org/simple'; "UV_INDEX=$pypiMirror"
                        "UV_INDEX_STRATEGY=$(if ($env:UV_INDEX_STRATEGY) { $env:UV_INDEX_STRATEGY } else { 'unsafe-first-match' })"
                    }
                    if ($usePip) { 'PIP_EXTRA_INDEX_URL=https://pypi.org/simple'; "PIP_INDEX_URL=$pypiMirror" }
                }
                'torch' { "UNSLOTH_PYTORCH_MIRROR=$to" }
                'node' { "UNSLOTH_NODE_MIRROR=$to" }
                'npm' { "UNSLOTH_NPM_REGISTRY=$to" }
                'uvbin' { "UNSLOTH_UV_WHEEL_MIRROR=$to" }
            }
        }
        $env:_UNSLOTH_MIRROR_SPARE = @($hosts.Keys | ForEach-Object { (@($_) + @(& $varsOf $_)) -join '|' }) -join ' '
        if ($SpareOnly -or (Test-UvEnvFlag 'UV_OFFLINE')) { return }
        $answered = @{}
        $codeOf = { param($index, $result) if ($index -and "$($answered[$index][0])" -notmatch '^2\d\d$') { 0 } else { $result[0] } }
        # PS 5.1 may pin TLS 1.0/1.1 (every probed host refuses it; Tls|Tls12 still fails) and queues past 2 connections per host.
        $savedProtocol = [System.Net.ServicePointManager]::SecurityProtocol
        $savedLimit = [System.Net.ServicePointManager]::DefaultConnectionLimit
        if ([int]$savedProtocol -ne 0) { [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12 }
        [System.Net.ServicePointManager]::DefaultConnectionLimit = [Math]::Max($savedLimit, 16)
        try {
            $defaults = @($hosts.Values | ForEach-Object { $_[0] } | Select-Object -Unique)
            $indexes = Start-MirrorProbe -Urls @($hosts.Values | ForEach-Object { $_[3] } | Where-Object { $_ }) -Seconds 4 -LastByte 1023
            $timed = @{}
            foreach ($name in $defaults) { $timed[$name] = (Wait-MirrorProbe (Start-MirrorProbe -Urls $artifact[$name] -Seconds 1.5 -LastByte 1048575))[$artifact[$name]] }
            $answered = Wait-MirrorProbe $indexes
            $slow = @($hosts.Keys | Where-Object {
                $code = & $codeOf $hosts[$_][3] $timed[$hosts[$_][0]]
                $code -eq 0 -or ($code -ge 300 -and $code -lt 400) -or ($code -ge 200 -and $code -lt 300 -and $timed[$hosts[$_][0]][1] -lt $minBps)
            })
            if ($slow.Count -eq 0) { return }
            $sources = @($slow | ForEach-Object { $hosts[$_][1] } | Select-Object -Unique)
            $indexes = Start-MirrorProbe -Urls @($slow | ForEach-Object { $hosts[$_][3]; $hosts[$_][4] } | Where-Object { $_ }) -Seconds 4 -LastByte 1023
            $race = Wait-MirrorProbe (Start-MirrorProbe -Urls @($slow | ForEach-Object { $artifact[$hosts[$_][0]] }; $sources | ForEach-Object { $artifact[$_] }) -Seconds 4 -LastByte 1048575)
            $sourceIndexes = Wait-MirrorProbe $indexes
            foreach ($url in $sourceIndexes.Keys) { $answered[$url] = $sourceIndexes[$url] }
        } finally {
            [System.Net.ServicePointManager]::SecurityProtocol = $savedProtocol
            [System.Net.ServicePointManager]::DefaultConnectionLimit = $savedLimit
        }
        $used = $false
        foreach ($name in $slow) {
            $default = $race[$artifact[$hosts[$name][0]]]
            $mirror = $race[$artifact[$hosts[$name][1]]]
            $how = if ("$(& $codeOf $hosts[$name][3] $default)" -match '^2\d\d$') { 'slow' } else { 'blocked' }
            $defaultBps = if ($how -eq 'slow') { $default[1] } else { [long]0 }
            if ("$(& $codeOf $hosts[$name][4] $mirror)" -notmatch '^2\d\d$' -or $defaultBps -ge $minBps -or $mirror[1] -le $defaultBps) { continue }
            Set-MirrorEnv @(& $varsOf $name)
            $env:_UNSLOTH_MIRROR_SPARE = @(-split $env:_UNSLOTH_MIRROR_SPARE | Where-Object { $_ -notlike "$name|*" }) -join ' '
            if ($name -eq 'pypi' -and $how -eq 'slow') { $env:_UNSLOTH_MIRROR_SPARE = (@(-split $env:_UNSLOTH_MIRROR_SPARE) + ((@('unsynced') + @(& $varsOf 'unsynced')) -join '|')) -join ' ' }
            step "mirror" "$(Get-MirrorName $name) is $how ($($defaultBps -shr 10) KB/s, mirror $($mirror[1] -shr 10) KB/s); using $($hosts[$name][2])" "Yellow"
            $used = $true
        }
        if ($used) { substep "Set UNSLOTH_MIRROR_FALLBACK=0 to always use the default hosts." }
    }

    function Get-MirrorName {
        param([string]$Name)
        @{ pypi = 'PyPI'; unsynced = 'The PyPI mirror'; torch = 'download.pytorch.org'; node = 'nodejs.org'; npm = 'registry.npmjs.org'; uvbin = 'releases.astral.sh (uv)' }[$Name]
    }

    function Set-MirrorEnv {
        param([string[]]$Pairs)
        foreach ($pair in $Pairs) { Set-Item "Env:$($pair.Split('=', 2)[0])" $pair.Split('=', 2)[1] }
    }

    function Pop-MirrorSpare {
        param([string]$Name)
        $entry = @(-split $env:_UNSLOTH_MIRROR_SPARE | Where-Object { $_ -like "$Name|*" })
        if (-not $entry) { return }
        $env:_UNSLOTH_MIRROR_SPARE = @(-split $env:_UNSLOTH_MIRROR_SPARE | Where-Object { $_ -notlike "$Name|*" }) -join ' '
        $pairs = @($entry[0].Split('|') | Select-Object -Skip 1)
        step "mirror" "$(Get-MirrorName $Name) failed; retrying through $($pairs[0].Split('=', 2)[1])" "Yellow"
        return $pairs
    }

    function Use-MirrorSpare {
        param([string]$Name)
        $pairs = @(Pop-MirrorSpare $Name)
        Set-MirrorEnv $pairs
        return $pairs.Count -gt 0
    }

    function Get-MirrorFailedHost {
        param([string]$Output, [string]$Ran)
        if ($Output -notmatch 'error sending request|timed out|network timeout|idle timeout|connection (reset|refused|closed|aborted)|network aborted|broken pipe|dns error|failed to lookup address|name resolution|nodename nor servname|network is unreachable|error decoding response body|end of file before message length|unexpected eof|tls handshake|sslerror|certificate verify failed|server error|service unavailable|bad gateway|gateway time-?out|too many requests|max retries exceeded|remotedisconnected|incompleteread|econnreset|etimedout|eidletimeout|eai_again|enotfound|econnrefused|socket hang up') {
            if ($Output -match 'only \S+ (.* )?(is|are) available|no versions? of|not found in the package registry|could not find a version that satisfies|no matching distribution found') { 'unsynced' }
            return
        }
        if ($Output -match 'download(-r2)?\.pytorch\.org') { 'torch' }
        elseif ($Output -match 'registry\.npmjs\.org') { 'npm' }
        elseif ($Output -match 'pypi\.org|pythonhosted\.org') { 'pypi' }
        elseif ($Ran -and $Output -notmatch 'https?://') { $Ran }
    }

    # ── END SHARED WITH studio/setup.ps1 ──

    # Redact index-URL credentials (userinfo, query, fragment) from captured output before
    # printing on failure. Verbose mode streams uncaptured, so it isn't redacted.
    function Redact-InstallOutput {
        param([string]$Text)
        if (-not $Text) { return $Text }
        $Text = $Text -replace '(https?://)[^/@\s`]+@', '$1<redacted>@'
        $Text = $Text -replace '([?&][^=\s&`]+)=[^&#\s`]+', '$1=<redacted>'
        return $Text -replace '(https?://[^\s`#]+)#[^\s`]+', '$1#<redacted>'
    }

    function Invoke-InstallCommand {
        param(
            [Parameter(Mandatory = $true)][ScriptBlock]$Command,
            [string]$Label = "install command",
            [switch]$NoMirror
        )
        if ($script:InstallTorchMirror) {
            foreach ($v in $Command.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.VariablePath.IsUnqualified }, $true)) {
                $value = $ExecutionContext.SessionState.PSVariable.GetValue($v.VariablePath.UserPath)
                if (@($value) -like 'https://download.pytorch.org/whl*') {
                    Set-Variable -Name $v.VariablePath.UserPath -Value ($value -replace '^https://download\.pytorch\.org/whl', $script:InstallTorchMirror.TrimEnd('/'))
                }
            }
        }
        # A pinned index must beat an inherited uv mirror (#6898); UV_NO_CONFIG=1 blocks uv.toml.
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
            $collected = [System.Text.StringBuilder]::new()
            if ($script:UnslothVerbose) {
                # Merge stderr into stdout without flipping $? (5.1 treats stderr records as errors).
                # Redact per record, verbose mode included.
                if ($NoMirror -or -not $env:_UNSLOTH_MIRROR_SPARE) {
                    & $Command 2>&1 | ForEach-Object {
                        Write-UvDownloadMarker "$_"
                        Redact-InstallOutput "$_"
                    } | Out-Host
                } else {
                    & $Command 2>&1 | ForEach-Object {
                        [void]$collected.AppendLine("$_")
                        Write-UvDownloadMarker "$_"
                        Redact-InstallOutput "$_"
                    } | Out-Host
                }
            } else {
                # Streamed, not collected, so a marker reaches the app mid-download.
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
        } finally {
            $ErrorActionPreference = $prevEap
            if ($savedUvIndex) {
                Remove-Item "Env:UV_NO_CONFIG" -ErrorAction SilentlyContinue
                foreach ($n in $savedUvIndex.Keys) { if ($null -ne $savedUvIndex[$n]) { Set-Item "Env:$n" $savedUvIndex[$n] } }
            }
        }
        if ($exitCode -eq 0 -or $NoMirror -or -not $env:_UNSLOTH_MIRROR_SPARE) { return $exitCode }
        return (Invoke-InstallMirrorRetry -Code $exitCode -Command $Command -Label $Label -Output $collected.ToString())
    }

    function Invoke-InstallMirrorRetry {
        param([int]$Code, [ScriptBlock]$Command, [string]$Label, [string]$Output)
        $words = @(foreach ($n in $Command.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -or $n -is [System.Management.Automation.Language.VariableExpressionAst] }, $true)) {
            if ($n -is [System.Management.Automation.Language.StringConstantExpressionAst]) { $n.Value } else { $ExecutionContext.SessionState.PSVariable.GetValue($n.VariablePath.UserPath) }
        }) | ForEach-Object { "$_" }
        $torchArg = [bool](@($words) -like 'https://download.pytorch.org/whl*')
        $pinned = [bool](@($words) -match '^--(index-url|default-index)$')
        $ran = if ($torchArg) { 'torch' } elseif ($pinned -or @($words) -match '^(--find-links|--no-index|--torch-backend.*|venv)$|://') { '' } else { 'pypi' }
        $failed = Get-MirrorFailedHost -Output $Output -Ran $ran
        $ownIndex = $pinned -or @($words) -contains '--no-index'
        if (-not (($failed -eq 'torch' -and $torchArg) -or ($failed -eq 'pypi' -and -not $pinned) -or ($failed -eq 'unsynced' -and -not $ownIndex))) { return $Code }
        $pairs = @(Pop-MirrorSpare $failed)
        if (-not $pairs) { return $Code }
        $saved = @{}
        foreach ($pair in $pairs) { $saved[$pair.Split('=', 2)[0]] = [Environment]::GetEnvironmentVariable($pair.Split('=', 2)[0]) }
        $savedTorch = $script:InstallTorchMirror
        Set-MirrorEnv $pairs
        if ($failed -eq 'torch') { $script:InstallTorchMirror = $env:UNSLOTH_PYTORCH_MIRROR }
        $code = Invoke-InstallCommand -Command $Command -Label $Label -NoMirror
        if ($code -ne 0) {
            foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) }
            $script:InstallTorchMirror = $savedTorch
        }
        return $code
    }

    function Invoke-InstallCommandRetry {
        param(
            [Parameter(Mandatory = $true, Position = 0)][ScriptBlock]$Command,
            [string]$Label = "install step"
        )
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
            $code = Invoke-InstallCommand -Command $Command -Label $Label -NoMirror:($attempt -lt $maxAttempts)
            if ($code -eq 0) { return 0 }
            if ($attempt -ge $maxAttempts) { return $code }
            substep ("retrying ""$Label"" after transient failure (attempt $($attempt + 1)/$maxAttempts, waiting ${delay}s)...") "Yellow"
            Start-Sleep -Seconds $delay
            $attempt++
            $delay = $delay * 2
        }
    }

    # ── Managed CLI invocation, Application Control safe ──
    # The generated unsigned unsloth.exe is denied by AppLocker/WDAC/Smart App Control while
    # the venv's signed python.exe runs (#8490), so go through the interpreter.
    # Byte-identical to WINDOWS_CLI_ENTRYPOINT in studio/src-tauri/src/process.rs: argv[0] is
    # set BEFORE the import (unsloth_cli reads it at import time), and sys.path[:1] drops the
    # cwd entry `python -c` adds. Not -I: it implies -E.
    # Marker written into every bin\unsloth.cmd; mirrored in scripts/uninstall.ps1 and
    # studio/setup.ps1.
    $script:UnslothCmdShimMarker = "unsloth-studio-managed-launcher"
    $script:UnslothCliTrampoline = "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"

    # Policy refusal through PowerShell's wrapper exceptions: AppLocker 1260, App Control
    # for Business and Smart App Control 4551 (ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION).
    # $LASTEXITCODE cannot answer this: no process was created, so it still holds the
    # exit code of whichever native command ran last.
    function Test-ApplicationControlBlock {
        param($ErrorRecord)

        $ex = if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) { $ErrorRecord.Exception } else { $ErrorRecord }
        while ($ex) {
            # Win32Exception carries NativeErrorCode; outer wrappers keep only the HRESULT
            # form of the same code (0x800704EC, 0x800711C7).
            if ($ex -is [System.ComponentModel.Win32Exception] -and $ex.NativeErrorCode -in @(1260, 4551)) { return $true }
            if ($ex.HResult -in @(-2147023636, -2147020345)) { return $true }
            $ex = $ex.InnerException
        }
        return $false
    }

    # Never suggests turning a security policy off.
    function Write-ApplicationControlBlocked {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)

        Write-StudioLine "[ERROR] Windows Application Control blocked the managed Python runtime." -ForegroundColor Red
        Write-StudioLine "        AppLocker, App Control for Business or Smart App Control refused to start it." -ForegroundColor Yellow
        Write-StudioLine "        Blocked program: $Path" -ForegroundColor Yellow
        Write-StudioLine "        Review AppLocker `"EXE and DLL`" event 8004," -ForegroundColor Yellow
        Write-StudioLine "        or CodeIntegrity/Operational event 3077." -ForegroundColor Yellow
        return "Windows Application Control blocked the managed Python runtime at $Path."
    }

    # One pre-quoted command line: Start-Process joins -ArgumentList unquoted.
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

    # Exit code goes to $script:ManagedUnslothCliExit, since stdout must keep streaming.
    # $null means the interpreter never started under Application Control.
    function Invoke-ManagedUnslothCli {
        param(
            [Parameter(Mandatory = $true)][string]$Python,
            [string[]]$Arguments = @()
        )

        # -X utf8, not the env var. No -I (see the trampoline). Same as
        # build_managed_cli_command in the desktop app.
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
    # child, the policy code straight back); an unaffected machine pays one `--version`,
    # measured at a quarter second. A process that started is not blocked whatever it does next,
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
                # Drain both pipes asynchronously BEFORE waiting, or a child filling one buffer deadlocks
                # (PYTHONPROFILEIMPORTTIME can produce ~24 KB of stderr).
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

    # Name the .cmd when the .exe is blocked OR missing (removed after install); a missing file is
    # not "blocked".
    function Test-UnslothCmdShimPreferred {
        param(
            [Parameter(Mandatory = $true)][string]$ShimExe,
            [Parameter(Mandatory = $true)][string]$ShimCmd
        )

        if (-not (Test-Path -LiteralPath $ShimCmd -PathType Leaf)) { return $false }
        if (-not (Test-Path -LiteralPath $ShimExe -PathType Leaf)) { return $true }
        return (Test-ShimLaunchBlocked -Path $ShimExe)
    }

    # Relative path or $null across roots. Longhand: GetRelativePath is .NET Core only and
    # MakeRelativeUri percent-encodes.
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

    # Pure function of two paths, so re-runs are byte-identical. %~dp0 keeps special path
    # characters out of the file. CRLF, ASCII, no BOM (cmd reads a BOM as a command).
    # Covers EXE/DLL enforcement (#8490); AppLocker Script rules block .cmd too.
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
            # The ownership marker; it gates a recursive delete, so it must be unmistakable.
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

        # Probes inside the try: a throw under the rollback guard would undo the install.
        $shimCmd = Join-Path $ShimDir "unsloth.cmd"
        try {
            if (Test-Path -LiteralPath $shimCmd -PathType Container) {
                substep "cannot write $shimCmd, a directory is in the way" "Yellow"
                return
            }
            # Sweep temps from killed runs (not live ones) before the unchanged-content return.
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
            # Bytes, not ReadAllText: that strips a BOM and -eq is case insensitive.
            if (Test-Path -LiteralPath $shimCmd -PathType Leaf) {
                $existing = [System.IO.File]::ReadAllBytes($shimCmd)
                if (@(Compare-Object $existing $desired -SyncWindow 0).Count -eq 0) { return }
                # Warn when replacing a file that is not ours.
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

    # Ownership check that gates deleting a user-chosen directory. Mirrored in
    # scripts/uninstall.ps1 (_IsUnslothCmdShim) and studio/setup.ps1.
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
            [Parameter(Mandatory = $true)][string]$ManagedPythonPath,
            [switch]$DestinationValidated
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
            # The packaged .ico (studio\frontend\dist) serves irm|iex installs with no $PSScriptRoot.
            $packagedIcon = $null
            try {
                $venvRoot = Split-Path -Parent (Split-Path -Parent $ManagedPythonPath)
                $packagedIconCandidate = Join-Path $venvRoot "Lib\site-packages\studio\frontend\dist\unsloth.ico"
                if (Test-Path -LiteralPath $packagedIconCandidate) { $packagedIcon = $packagedIconCandidate }
            } catch {}
            $iconUrl = "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth.ico"

            if (-not (Test-Path -LiteralPath $appDir)) {
                [System.IO.Directory]::CreateDirectory($appDir) | Out-Null
            }

            # Per-install opaque id for launcher and backend, so neither compares install paths.
            $_studioIdDir = Join-Path $StudioHome "share"
            if (-not (Test-Path -LiteralPath $_studioIdDir)) {
                [System.IO.Directory]::CreateDirectory($_studioIdDir) | Out-Null
            }
            $_studioIdFile = Join-Path $_studioIdDir "studio_install_id"
            $_studioRootId = ""
            if ((Test-Path -LiteralPath $_studioIdFile) -and `
                ((Get-Item -LiteralPath $_studioIdFile).Length -gt 0)) {
                $_studioRootId = ([System.IO.File]::ReadAllText($_studioIdFile)).Trim()
                # 64 lowercase hex, as the backend and desktop app require. -cnotmatch (case sensitive);
                # anything else would be code in the single-quoted launcher assignment.
                if ($_studioRootId -cnotmatch '^[0-9a-f]{64}$') {
                    $_studioRootId = ""
                }
            }
            if (-not $_studioRootId) {
                $_idBytes = New-Object byte[] 32
                [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($_idBytes)
                $_studioRootId = -join ($_idBytes | ForEach-Object { $_.ToString('x2') })
                # No-clobber: the desktop app mints the same id. Two-arg File.Move throws on an existing
                # destination, so adopt it.
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
                        # Malformed is an interrupted write or a planted value: replace atomically.
                        Move-Item -LiteralPath $_idTmp -Destination $_studioIdFile -Force
                    }
                } finally {
                    Remove-Item -LiteralPath $_idTmp -Force -ErrorAction SilentlyContinue
                }
            }

            # Bake per-install $portFile / $mutexName so custom-root launchers do not share a mutex.
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
`$_StudioInstallIdFile = '$($_studioIdFile -replace "'", "''")'

# A missing or malformed id makes /api/health report "", which the baked id never matches: restore ours no-clobber (the desktop app mints it too), leaving a different valid id or an unreadable file alone.
function Repair-StudioInstallId {
    `$idTmp = `$null
    try {
        if (Test-Path -LiteralPath `$_StudioInstallIdFile -PathType Container) { return }
        if (Test-Path -LiteralPath `$_StudioInstallIdFile -PathType Leaf) {
            `$current = ([System.IO.File]::ReadAllText(`$_StudioInstallIdFile)).Trim()
            if (`$current -cmatch '^[0-9a-f]{64}$') { return }
        }
        [System.IO.Directory]::CreateDirectory((Split-Path -Parent `$_StudioInstallIdFile)) | Out-Null
        `$idTmp = "`$_StudioInstallIdFile.`$([System.IO.Path]::GetRandomFileName()).tmp"
        [System.IO.File]::WriteAllText(`$idTmp, `$_ExpectedStudioRootId)
        try {
            [System.IO.File]::Move(`$idTmp, `$_StudioInstallIdFile)
        } catch [System.IO.IOException] {
            `$incumbent = ''
            try { `$incumbent = ([System.IO.File]::ReadAllText(`$_StudioInstallIdFile)).Trim() } catch { return }
            if (`$incumbent -cnotmatch '^[0-9a-f]{64}$') {
                Move-Item -LiteralPath `$idTmp -Destination `$_StudioInstallIdFile -Force
            }
        }
    } catch {
    } finally {
        if (`$idTmp) { Remove-Item -LiteralPath `$idTmp -Force -ErrorAction SilentlyContinue }
    }
}
Repair-StudioInstallId

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

            # UTF-8 with BOM for 5.1. Content-compared first so a re-run is a no-op on disk.
            $utf8Bom = New-Object System.Text.UTF8Encoding($true)
            # Compare bytes including the BOM, so an older BOM-less launcher gets rewritten.
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
            # Clear any mark-of-the-web (RemoteSigned refuses marked unsigned scripts), outside the
            # content check. -Confirm:$false: a profile's $ConfirmPreference would otherwise prompt.
            Unblock-File -LiteralPath $launcherPs1 -Confirm:$false -ErrorAction SilentlyContinue
            # No .vbs launcher: the .lnk shortcuts run powershell.exe on launch-studio.ps1 hidden.
            # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)

            # Drop any launch-studio.vbs left by an install that predates that change.
            $legacyLauncherVbs = Join-Path $appDir "launch-studio.vbs"
            if (Test-Path -LiteralPath $legacyLauncherVbs) {
                Remove-Item -LiteralPath $legacyLauncherVbs -Force -ErrorAction SilentlyContinue
            }

            # Attach only on a valid ICO header; snapshot the icon so the refresh runs on a change.
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
            } elseif ($packagedIcon) {
                try {
                    Copy-Item -LiteralPath $packagedIcon -Destination $iconPath -Force
                } catch {
                    Write-StudioLine "[DEBUG] Error copying packaged icon: $($_.Exception.Message)" -ForegroundColor DarkGray
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

            # Env-mode: skip .lnk shortcuts (may point at a deleted workspace); launcher stays.
            if ($StudioRedirectMode -eq 'env') {
                substep "wrote launcher at $launcherPs1 (persistent shortcuts skipped in env-override mode)"
                return
            }

            # Gates the heavy refresh below.
            # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
            $firstInstall = -not (
                ($desktopLink -and (Test-Path -LiteralPath $desktopLink)) -or
                ($startMenuLink -and (Test-Path -LiteralPath $startMenuLink))
            )

            # Shortcuts run powershell.exe on launch-studio.ps1 hidden, without a .vbs wrapper.
            # RemoteSigned, not Bypass, as install.rs does; a locally written launcher loads either way.
            # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
            $powershellForLnk = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
            $shortcutTarget = $powershellForLnk
            $shortcutArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File `"$launcherPs1`""
            # A launcher on a share (UNC or mapped drive) is remote to RemoteSigned, so use Bypass
            # without the hidden window for that case only.
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
                        # Save() rewrites even when unchanged, so compare first (IconLocation carries ",0").
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
                    # Per-item SHChangeNotify: the global broadcast misses a rewritten same-name .lnk.
                    try {
                        $null = Invoke-StudioPythonShellIconRefresh `
                            -Paths $createdShortcutPaths -Exe $ManagedPythonPath -DestinationValidated:$DestinationValidated
                    } catch {}
                    if ($firstInstall -or $iconChanged) {
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -ClearIconCache 2>$null } catch {}
                        try { & "$env:SystemRoot\System32\ie4uinit.exe" -show 2>$null } catch {}
                        # Win11 caches tile icons past ie4uinit; never drop start2.bin (the layout).
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
        # Returns before the lock try/finally, so undo the temp redirection on every way out.
        try {
        if ($TauriMode) { return }
        # Check the interpreter, which the launcher runs, not the possibly denied unsloth.exe.
        $ShortcutPython = Join-Path $VenvDir "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $ShortcutPython)) {
            Write-StudioLine "[ERROR] managed Python missing at $ShortcutPython; run install.ps1 first." -ForegroundColor Red
            # throw (not Exit-InstallFailure) so non-Tauri callers see rc != 0.
            throw "managed Python missing"
        }
        # Older installs only reach the installer through `unsloth studio update`, so create the
        # shim directory and .cmd here too. Idempotent and never fatal.
        $ShortcutShimDir = Join-Path $StudioHome "bin"
        try {
            [System.IO.Directory]::CreateDirectory($ShortcutShimDir) | Out-Null
        } catch {
            substep "could not create $ShortcutShimDir`: $($_.Exception.Message)" "Yellow"
        }
        if (Test-Path -LiteralPath $ShortcutShimDir -PathType Container) {
            Write-UnslothCmdShim -ShimDir $ShortcutShimDir -PythonPath $ShortcutPython
            # Older installers put Scripts on PATH instead; add the shim dir (idempotent). Env mode
            # never writes the registry.
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
    # "Run as administrator" starts in System32, where setup refuses after the PyTorch
    # download. Relocating is safe (nothing reads the caller's directory except
    # --with-llama-cpp-dir, rebased first). Not restored, like the PSModulePath fix.
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
            # Anchor to the directory the user typed against; GetFullPath uses the .NET current
            # directory, which Set-Location never updates.
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
        # A SYSTEM account's $StudioHome is under System32; rebasing would orphan the install, so
        # stop.
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
    # Only needed to install Python or uv, so the hard failure is deferred to those branches.
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
        # Hashes the normalized spelling when the native resolver could not answer, so two aliases
        # of one root may not exclude each other; better than refusing to install.
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
        # Different spellings prove different directories only when both resolved exactly.
        if (-not $leftInfo.Exact -or -not $rightInfo.Exact) {
            if ($script:StudioInstallIsFresh -ne $true) {
                Write-StudioLine "[WARN] Could not resolve Unsloth path identity; using the runtime lock." -ForegroundColor Yellow
            }
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

    # Mutex AND file: the mutex misses aliases when the resolver is inexact, the file does not;
    # released installers only know the mutex.

    function Enter-StudioInstallLock {
        param([Parameter(Mandatory = $true)][string]$Path)
        $mutex = Enter-StudioInstallMutex -Path $Path
        if ($null -eq $mutex) { return $null }
        $stream = $null
        try {
            # Race-safe: concurrent creators both succeed, and only one opens the file exclusively
            # below. A throw here surfaces as the caller's install-lock error.
            $existedBefore = [System.IO.Directory]::Exists($Path)
            # Prefer the env-mode root's own record of whether it pre-existed; only ever in the
            # "created" direction.
            if ($script:StudioEnvRootPath -and
                $script:StudioEnvRootPath.Equals($Path, [System.StringComparison]::OrdinalIgnoreCase) -and
                -not $script:StudioEnvRootExisted) {
                $existedBefore = $false
            }
            $null = [System.IO.Directory]::CreateDirectory($Path)
            $lockPath = Join-Path $Path $script:StudioInstallLockFileName
            # A link at the lock path would point our exclusive handle at someone else's file: remove
            # the link itself (checked with Get-Item -Force), never its target.
            $existingLock = $null
            try {
                $existingLock = Get-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
            } catch { $existingLock = $null }
            if ($existingLock -and
                ($existingLock.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                # A link that will not go stops the install with a lock-creation error.
                try {
                    Remove-Item -LiteralPath $lockPath -Force -ErrorAction Stop
                } catch {
                    throw "A link is planted at the install lock path $lockPath and cannot be removed."
                }
            }
            # Hard links carry no reparse attribute: ours is always zero bytes, so check the opened
            # stream's Length. One repair at most, via CreateNew.
            $repaired = $false
            while ($true) {
                if ($repaired) { $mode = [System.IO.FileMode]::CreateNew }
                else { $mode = [System.IO.FileMode]::OpenOrCreate }
                try {
                    $stream = [System.IO.File]::Open(
                        $lockPath,
                        $mode,
                        [System.IO.FileAccess]::ReadWrite,
                        [System.IO.FileShare]::None)
                } catch [System.IO.IOException] {
                    # Only sharing/lock violations (32, 33; EAGAIN 11 off Windows; 80/183/17 for a lost
                    # CreateNew race) mean another installer holds it.
                    $code = $_.Exception.HResult -band 0xFFFF
                    $lost = $repaired -and ($code -eq 80 -or $code -eq 183 -or $code -eq 17)
                    if (-not $lost -and $code -ne 32 -and $code -ne 33 -and $code -ne 11) { throw }
                    if ($stream) { $stream.Dispose() }
                    Exit-StudioInstallMutex -Mutex $mutex
                    return $null
                }
                # Re-check after opening: narrows, not closes, the swap window (that needs
                # FILE_FLAG_OPEN_REPARSE_POINT).
                if ($stream.Length -eq 0) {
                    $entry = $null
                    try { $entry = Get-Item -LiteralPath $lockPath -Force -ErrorAction Stop } catch { $entry = $null }
                    if ($entry -and (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
                        $stream.Dispose()
                        $stream = $null
                        throw "A link was planted at the install lock path $lockPath while it was being opened."
                    }
                    break
                }
                if ($repaired) {
                    # A file this run just created cannot be non-empty. Fail loudly, not in a loop.
                    throw "The install lock file at $lockPath is not empty after being replaced."
                }
                # Move the entry ASIDE rather than delete it: an ordinary non-empty file keeps its bytes,
                # and a planted hard link still leaves the lock path.
                $stream.Dispose()
                $stream = $null
                $displaced = "$lockPath.displaced-" + (Get-Date -Format "yyyyMMddHHmmss") +
                    "-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
                try {
                    Move-Item -LiteralPath $lockPath -Destination $displaced -Force -ErrorAction Stop
                } catch {
                    # Same codes as the open below; dug out of the wrapped exception.
                    $mvCode = 0
                    $mvEx = $_.Exception
                    while ($mvEx) {
                        if ($mvEx -is [System.IO.IOException]) {
                            $mvCode = $mvEx.HResult -band 0xFFFF
                            break
                        }
                        $mvEx = $mvEx.InnerException
                    }
                    Exit-StudioInstallMutex -Mutex $mutex
                    if ($mvCode -ne 32 -and $mvCode -ne 33 -and $mvCode -ne 11) { throw }
                    return $null
                }
                $repaired = $true
            }
            # Nothing is written: holding the handle IS the lock. Mark a root this call created, via
            # the link-safe helper, so scripts/uninstall.ps1's _IsStudioRoot can remove it; never a
            # pre-existing UNSLOTH_STUDIO_HOME.
            if (-not $existedBefore) {
                Write-StudioRootOwnerMarker -Root $Path
            }
            return [pscustomobject]@{ Mutex = $mutex; Stream = $stream; Path = $lockPath }
        } catch {
            if ($stream) { try { $stream.Dispose() } catch {} }
            Exit-StudioInstallMutex -Mutex $mutex
            throw
        }
    }

    function Exit-StudioInstallLock {
        param($Lock)
        if ($null -eq $Lock) { return }
        # Closing the handle IS the release, so a dead process leaves nothing to clean up. The
        # file stays: deleting it would race another installer that has just opened it.
        if ($Lock.Stream) { try { $Lock.Stream.Dispose() } catch {} }
        Exit-StudioInstallMutex -Mutex $Lock.Mutex
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

    # QueryFullProcessImageNameW works where MainModule is unreadable; without it a venv in
    # use could be overwritten. Only real executable images count as proof.
    $script:StudioProcessImageTable = $null
    # PID -> image path for every visible process, via ctypes in one child, so no type is defined
    # in this script. $null leaves the WMI rung exactly as it was.
    $script:StudioPythonProcessImageTable = $null
    $script:StudioPythonProcessImageProbed = $false
    $script:StudioProcessImageWarned = $false

    function Get-StudioPythonProcessImageTable {
        $exe = Get-StudioEarlyPython
        if (-not $exe) { return $null }
        if (-not ($env:OS -eq "Windows_NT")) { return $null }
        $probe = "import ctypes,sys" + [char]10 +
            "from ctypes import wintypes" + [char]10 +
            "k32=ctypes.WinDLL('kernel32',use_last_error=True)" + [char]10 +
            "psapi=ctypes.WinDLL('psapi',use_last_error=True)" + [char]10 +
            "k32.OpenProcess.restype=wintypes.HANDLE" + [char]10 +
            "k32.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]" + [char]10 +
            "k32.CloseHandle.argtypes=[wintypes.HANDLE]" + [char]10 +
            "k32.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]" + [char]10 +
            "n=1024" + [char]10 +
            "while True:" + [char]10 +
            "    a=(wintypes.DWORD*n)();b=wintypes.DWORD()" + [char]10 +
            "    if not psapi.EnumProcesses(ctypes.byref(a),ctypes.sizeof(a),ctypes.byref(b)): sys.exit(3)" + [char]10 +
            "    if b.value < ctypes.sizeof(a): break" + [char]10 +
            "    n*=2" + [char]10 +
            "out=[]" + [char]10 +
            "for pid in a[:b.value//ctypes.sizeof(wintypes.DWORD)]:" + [char]10 +
            "    if not pid: continue" + [char]10 +
            "    h=k32.OpenProcess(0x1000,False,pid)" + [char]10 +
            "    if not h: continue" + [char]10 +
            "    try:" + [char]10 +
            "        buf=ctypes.create_unicode_buffer(32768);sz=wintypes.DWORD(32768)" + [char]10 +
            "        if k32.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(sz)): out.append(str(pid)+'|'+buf.value)" + [char]10 +
            "    finally:" + [char]10 +
            "        k32.CloseHandle(h)" + [char]10 +
            "sys.stdout.buffer.write('\n'.join(out).encode('utf-8'))"
        $raw = Invoke-StudioEarlyPythonScript -Exe $exe -Script $probe -TimeoutMs 20000
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        $table = @{}
        foreach ($line in ($raw -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $split = $line.IndexOf('|')
            if ($split -lt 1) { continue }
            $pidText = $line.Substring(0, $split)
            $path = $line.Substring($split + 1)
            $parsed = 0
            if (-not [int]::TryParse($pidText, [ref]$parsed)) { continue }
            if ($parsed -le 0) { continue }
            if (-not [string]::IsNullOrWhiteSpace($path)) { $table[$parsed] = $path }
        }
        if ($table.Count -eq 0) { return $null }
        return $table
    }

    # Bounded out of process like Get-StudioVolumeList: without Python this rung is the common one,
    # and a degraded WMI repository never returns.
    function Get-StudioWmiProcessImageRows {
        param(
            [scriptblock]$Query = {
                Get-CimInstance -ClassName Win32_Process -ErrorAction Stop | Select-Object ProcessId, ExecutablePath
            },
            [int]$TimeoutSeconds = 15
        )
        $job = $null
        try {
            $job = Start-Job -ScriptBlock $Query
            if (Wait-Job -Job $job -Timeout $TimeoutSeconds) {
                return @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
            }
            Stop-Job -Job $job -ErrorAction SilentlyContinue
        } catch {
        } finally {
            if ($job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue }
        }
        return @()
    }

    function Get-StudioProcessImagePath {
        param([Parameter(Mandatory = $true)][int]$ProcessId)
        $process = $null
        try { $process = Get-Process -Id $ProcessId -ErrorAction Stop } catch { $process = $null }
        if ($process) {
            # .Path is MainModule.FileName (an ETS ScriptProperty over it on 5.1), so
            # there is no second rung here: empty means MainModule was unreadable.
            try {
                if (-not [string]::IsNullOrWhiteSpace($process.Path)) { return $process.Path }
            } catch {}
        }
        # PROCESS_QUERY_LIMITED_INFORMATION works where PROCESS_VM_READ does not; one child per run.
        if (-not $script:StudioPythonProcessImageProbed) {
            $script:StudioPythonProcessImageProbed = $true
            # A failure falls through to the WMI rung, never past it.
            try { $script:StudioPythonProcessImageTable = Get-StudioPythonProcessImageTable } catch {
                $script:StudioPythonProcessImageTable = $null
            }
            if ($env:OS -eq "Windows_NT" -and $null -eq $script:StudioPythonProcessImageTable -and
                -not $script:StudioProcessImageWarned) {
                $script:StudioProcessImageWarned = $true
                Write-StudioLine "[WARN] Scanning for running Unsloth processes without the process-image helper; a process this shell cannot inspect may go unnoticed." -ForegroundColor Yellow
            }
        }
        if ($script:StudioPythonProcessImageTable -and
            $script:StudioPythonProcessImageTable.ContainsKey($ProcessId)) {
            return $script:StudioPythonProcessImageTable[$ProcessId]
        }
        # Per-run PID snapshots: a reused PID reads stale; accepted.
        # Queried once per run, not once per process: this is the slow rung.
        if ($null -eq $script:StudioProcessImageTable) {
            $script:StudioProcessImageTable = @{}
            foreach ($row in @(Get-StudioWmiProcessImageRows)) {
                if ($row -and -not [string]::IsNullOrWhiteSpace($row.ExecutablePath)) {
                    $script:StudioProcessImageTable[[int]$row.ProcessId] = [string]$row.ExecutablePath
                }
            }
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
        # No root-relative fallback: with every path inexact it matched tails across drives.
        # SUBST is folded in Get-StudioLexicalPath instead.

        # Block only confirmed executable identities: a command line or working
        # directory that merely mentions the path is not proof of an open file.
        $studioScanProcesses = @(Get-Process -ErrorAction SilentlyContinue |
            Select-Object -Property Id, ProcessName)
        $studioScanImages = @{}
        foreach ($process in $studioScanProcesses) {
            $image = $null
            try { $image = Get-StudioProcessImagePath -ProcessId $process.Id } catch { continue }
            if ([string]::IsNullOrWhiteSpace($image)) { continue }
            $studioScanImages[[string]$process.Id] = $image
        }
        try { Resolve-StudioFinalPathsInOneChild -Paths @($studioScanImages.Values) } catch { }
        foreach ($process in $studioScanProcesses) {
            $executable = $studioScanImages[[string]$process.Id]
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
        $studioInstallLock = Enter-StudioInstallLock -Path $StudioHome
    } catch {
        Write-StudioLine "[ERROR] Could not create the Unsloth install lock: $($_.Exception.Message)" -ForegroundColor Red
        return (Exit-InstallFailure "Could not create the Unsloth install lock")
    }
    if ($null -eq $studioInstallLock) {
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
    $hadPreviousUvCacheDir = [Environment]::GetEnvironmentVariables().ContainsKey("UV_CACHE_DIR")
    $previousUvCacheDir = [Environment]::GetEnvironmentVariable("UV_CACHE_DIR", "Process")
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
    # -CommandType Application so a profile alias or function named winget cannot intercept.
    # Pinned so later call sites cannot re-resolve.
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

    # ── Detect Python 3.11-3.13, skipping conda and venvs whose base_prefix is conda. ──
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

    # Returns @{ Version; Path } or $null. The resolved Path stops uv re-resolving to conda.
    # $PythonSkip versions are dropped during enumeration.
    function Find-CompatiblePython {
        # -X64Only: best installed x64 interpreter or $null, never ARM64. Last resort for
        # Install-X64Python, where x64 of a lower-priority minor beats ARM64.
        param([switch]$X64Only)
        # Windows on ARM prefers x64: pyarrow and hf-transfer ship no win_arm64 wheel. ARM64 is
        # preferred on a native CUDA host or under UNSLOTH_ALLOW_ARM64_PYTHON (never for
        # -X64Only).
        $preferArm64 = (-not $X64Only) -and (
            $script:WoaNativeCudaTorch -or
            ((Get-HostMachineArch) -eq "arm64" -and (Test-Arm64PythonOptOut))
        )
        $preferX64 = $X64Only -or ((Get-HostMachineArch) -eq "arm64" -and -not $preferArm64)
        $rankByArch = $preferX64 -or $preferArm64
        $candidates = @()
        # Python Launcher first; it resolves to standard CPython, not conda.
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
                            if (-not $rankByArch) { return @{ Version = $ver; Path = $resolvedExe; Arch = "" } }
                            $candidates += @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # -All looks past shadowing stubs. Skip WindowsApps (alias stubs) and conda.
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
                            if (-not $rankByArch) { return @{ Version = $ver; Path = $resolvedExe; Arch = "" } }
                            $candidates += @{ Version = $ver; Path = $resolvedExe }
                        }
                    }
                } catch {}
            }
        }
        # `py -3.12` runs the preferred (usually ARM64) build and `-3.12-64` only means 64-bit,
        # so enumerate registrations with -0p.
        if ($rankByArch) {
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
        # Prefer x64 only within one minor, respecting the caller's version preference.
        foreach ($c in $candidates) {
            $tag = Get-PythonPlatformTag $c.Path
            $c.Arch = if ($tag -eq "win-amd64") { "x86_64" } elseif ($tag -eq "win-arm64") { "arm64" } else { "unknown" }
        }
        # Sweep every minor for ARM64 first, or the opt-out still yields x64.
        if ($preferArm64) {
            foreach ($minor in $minors) {
                $arm = $candidates |
                    Where-Object { $_.Version -eq $minor -and $_.Arch -eq "arm64" } |
                    Select-Object -First 1
                if ($arm) { return $arm }
            }
        }
        foreach ($minor in $minors) {
            $sameMinor = @($candidates | Where-Object { $_.Version -eq $minor })
            if ($sameMinor.Count -eq 0) { continue }
            if ($preferArm64) {
                # x64 is still returned when no ARM64 build exists; the torch step re-checks the tag.
                $arm = $sameMinor | Where-Object { $_.Arch -eq "arm64" } | Select-Object -First 1
                if ($arm) { return $arm }
                return $sameMinor[0]
            }
            $x64 = $sameMinor | Where-Object { $_.Arch -eq "x86_64" } | Select-Object -First 1
            if ($x64) { return $x64 }
            if (-not $X64Only) { return $sameMinor[0] }
        }
        if (-not $X64Only -and $candidates.Count -gt 0) { return $candidates[0] }
        return $null
    }

    # PrependPath=1 inside active conda demotes conda's entries (#5871); use AppendPath.
    function Get-PythonInstallerPathSwitch {
        if (Test-ActiveCondaEnvironment) { return "AppendPath=1" }
        return "PrependPath=1"
    }

    # ── Fallback when winget fails (msstore 0x8a15005e): silent per-user python.org, no UAC. ──
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

        # The pinned full version applies only to a matching minor, so 3.12 cannot install 3.13.
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

        # No SHA-256 to pin per patch release, so verify the publisher. Inspection failure is
        # treated as unverified.
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

        # Per-user install => no UAC. The PATH switch puts python + py on PATH;
        # Include_launcher installs py.exe (preferred by Find-CompatiblePython).
        substep "installing Python $full (silent, per-user)..."
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=0",
            (Get-PythonInstallerPathSwitch),
            "Include_launcher=1",
            # Launcher per-user too: Include_launcher defaults to all-users, which needs admin.
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

    # Backstop for interpreters whose --version disagrees with sys.version_info (shims).
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
        # Inside conda try python.org first: winget's package hard-codes PrependPath=1 (#5871).
        $pythonOrgTried = $false
        if (Test-ActiveCondaEnvironment) {
            substep "conda environment active ($env:CONDA_PREFIX) -- installing x64 Python from python.org, which can be installed without taking PATH priority." "Yellow"
            $pythonOrgTried = $true
            $found = Install-PythonFromPythonOrg -Arch "x86_64"
            if ($found -and $found.Arch -eq "x86_64") { return $found }
            if ($script:WingetAvailable) {
                substep "python.org could not provide an x64 Python -- falling back to winget, whose package always puts Python ahead of conda on PATH." "Yellow"
            }
        }
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
        # $pythonOrgTried, so an offline box inside conda does not sit through the same
        # failing download twice.
        if (-not $pythonOrgTried) {
            $found = Install-PythonFromPythonOrg -Arch "x86_64"
            if ($found -and $found.Arch -eq "x86_64") { return $found }
        }
        # Nothing installable (offline / no winget): an x64 build of another supported minor
        # still runs the wheels ARM64 cannot, so take it over the native interpreter.
        return (Find-CompatiblePython -X64Only)
    }

    # Single reader for UNSLOTH_ALLOW_ARM64_PYTHON, used by both decision sites.
    function Test-Arm64PythonOptOut {
        return ($env:UNSLOTH_ALLOW_ARM64_PYTHON -in @('1', 'true', 'yes', 'on'))
    }

    # ── Windows on ARM: the interpreter handed to uv has to be x64 ──
    # No win_arm64 pyarrow/hf-transfer (#8495); $null is a STOP (#10875).
    # UNSLOTH_ALLOW_ARM64_PYTHON=1 opts out.
    function Resolve-WindowsOnArmX64Python {
        param($SelectedPython)
        if (-not $SelectedPython) { return $null }
        # Read BEFORE attempting the x64 install, or the opt-out does nothing where it succeeds.
        if ((Test-Arm64PythonOptOut) -and ($SelectedPython.Arch -eq "arm64")) {
            Write-StudioLine "[WARN] UNSLOTH_ALLOW_ARM64_PYTHON is set, so keeping native ARM64 Python $($SelectedPython.Version)." -ForegroundColor Yellow
            Write-StudioLine "       pyarrow (via datasets) and hf-transfer publish no win_arm64 wheels, so they will be" -ForegroundColor Yellow
            Write-StudioLine "       built from source, which needs CMake plus the MSVC and Rust toolchains." -ForegroundColor Yellow
            Write-StudioLine "       Unset UNSLOTH_ALLOW_ARM64_PYTHON to have x64 Python installed and used instead." -ForegroundColor Yellow
            return $SelectedPython
        }
        substep "windows on arm: only a native ARM64 Python $($SelectedPython.Version) was found." "Yellow"
        substep "pyarrow and hf-transfer publish no win_arm64 wheels, so installing x64 Python..." "Yellow"
        $X64Python = Install-X64Python
        if ($X64Python) {
            step "python" "using x64 Python $($X64Python.Version) under emulation"
            return $X64Python
        }
        Write-StudioLine "[ERROR] Could not install an x64 Python on this ARM64 machine." -ForegroundColor Red
        Write-StudioLine "        pyarrow (via datasets) and hf-transfer ship no win_arm64 wheels, so an install" -ForegroundColor Yellow
        Write-StudioLine "        on ARM64 Python $($SelectedPython.Version) fails while building them from source." -ForegroundColor Yellow
        Write-StudioLine "        Fix: install x64 Python from https://www.python.org/downloads/windows/" -ForegroundColor Yellow
        Write-StudioLine "        (choose 'Windows installer (64-bit)', not ARM64), then re-run this installer." -ForegroundColor Yellow
        Write-StudioLine "        To attempt the install on ARM64 Python anyway, with CMake, MSVC and Rust" -ForegroundColor Yellow
        Write-StudioLine "        already installed, set UNSLOTH_ALLOW_ARM64_PYTHON=1." -ForegroundColor Yellow
        return $null
    }

    function Get-StudioVenvBasePython {
        param([Parameter(Mandatory = $true)][string]$VenvDir)
        # Fall back to the existing venv's base interpreter (from pyvenv.cfg) when discovery sees
        # no ARM64 one; read before the reinstall moves the venv.
        $cfg = Join-Path $VenvDir "pyvenv.cfg"
        if (-not (Test-Path -LiteralPath $cfg -PathType Leaf)) { return $null }
        $venvHome = $null
        $baseExe = $null
        foreach ($line in @(Get-Content -LiteralPath $cfg -ErrorAction SilentlyContinue)) {
            if ($line -match '^\s*home\s*=\s*(.+?)\s*$') { $venvHome = $Matches[1] }
            elseif ($line -match '^\s*base-executable\s*=\s*(.+?)\s*$') { $baseExe = $Matches[1] }
        }
        # base-executable names the interpreter exactly; home is its directory, which is what
        # a venv built by an older Python records on its own.
        $candidates = @()
        if ($baseExe) { $candidates += $baseExe }
        if ($venvHome) { $candidates += (Join-Path $venvHome "python.exe") }
        foreach ($exe in $candidates) {
            if (-not $exe) { continue }
            if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { continue }
            if (Test-IsCondaPython $exe) { continue }
            # ARM64 and a supported minor only: anything not provably native is no better
            # than the x64 selection it would replace.
            if ((Get-PythonPlatformTag $exe) -ne "win-arm64") { continue }
            $out = (& $exe --version 2>&1 | Out-String)
            if ($out -notmatch "Python ((3\.1[1-3])\.\d+)") { continue }
            $full = $Matches[1]
            $ver = $Matches[2]
            if ($PythonSkip -contains $full) { continue }
            return @{ Version = $ver; Path = $exe; Arch = "arm64" }
        }
        return $null
    }

    # Does a reused venv still match the chosen interpreter? A migrated venv keeps its
    # builder, which on Windows on ARM is native ARM64 (#8495, #10875).
    function Test-StudioVenvArchMismatch {
        param(
            [Parameter(Mandatory = $true)][string]$VenvPython,
            $SelectedPython,
            [string]$RecordedTag = $null
        )
        if ((Get-HostMachineArch) -ne "arm64") { return $false }
        # Use $RecordedTag: the venv may already have moved aside for rollback.
        $tag = $null
        if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
            $tag = Get-PythonPlatformTag $VenvPython
        } elseif ($RecordedTag) {
            $tag = $RecordedTag
        }
        # Asked before the opt-out, so the message below is only printed when there really is
        # a native ARM64 environment to keep.
        if ($tag -ne "win-arm64") { return $false }
        # Honoured here too: this path runs when the ordinary probe already found x64.
        if (Test-Arm64PythonOptOut) {
            # Two outcomes, so two sentences: with an ARM64 interpreter the environment is
            # really not rebuilt on x64, and with none the reinstall has already moved the
            # tree aside and the new one IS on x64.
            if ($SelectedPython -and $SelectedPython.Arch -eq "x86_64") {
                substep "windows on arm: UNSLOTH_ALLOW_ARM64_PYTHON is set, but no ARM64 Python is left on" "Yellow"
                substep "this machine, so the new environment is built on x64; the ARM64 one is kept." "Yellow"
            } else {
                substep "windows on arm: UNSLOTH_ALLOW_ARM64_PYTHON is set, so the environment is not rebuilt on x64." "Yellow"
            }
            return $false
        }
        if (-not $SelectedPython -or $SelectedPython.Arch -ne "x86_64") { return $false }
        return $true
    }

    # ── Install Python if no compatible version (3.11-3.13) found ──
    Write-TauriLog "STEP" "Installing Python"
    Initialize-WoaNativeCudaTorch -PythonMinor $PythonVersion
    if ($script:WoaNativeCudaTorch) {
        step "gpu" "Windows on ARM + NVIDIA: native CUDA wheels available -- installing the ARM64 stack" "Green"
        substep "torch index: $(Remove-IndexUrlCredentials $script:WoaTorchIndexUrl)"
    }
    $DetectedPython = Remove-SkippedPython (Find-CompatiblePython)

    # No usable interpreter: uv provides one instead of a system install (#7802), except on
    # Windows on ARM.
    $PythonFromUv = (-not $DetectedPython) -and ((Get-HostMachineArch) -ne "arm64")
    if ($DetectedPython) {
        step "python" "Python $($DetectedPython.Version) already installed"
    } elseif ($PythonFromUv) {
        step "python" "no Python 3.11-3.13 found; uv will provide Python $PythonVersion"
    }
    # Dot-sourced so $DetectedPython lands in this scope; also the fallback when uv cannot.
    $InstallSystemPython = {
        substep "installing Python ${PythonVersion}..."
        $pythonPackageId = "Python.Python.$PythonVersion"
        $wingetExit = $null

        # Inside conda try python.org first: winget's package hard-codes PrependPath=1 (#5871).
        $pythonOrgTried = $false
        if (Test-ActiveCondaEnvironment) {
            substep "conda environment active ($env:CONDA_PREFIX) -- installing Python from python.org, which can be installed without taking PATH priority." "Yellow"
            $pythonOrgTried = $true
            $DetectedPython = Install-PythonFromPythonOrg
            if (-not $DetectedPython -and $script:WingetAvailable) {
                substep "python.org could not provide Python -- falling back to winget, whose package always puts Python ahead of conda on PATH." "Yellow"
            }
        }

        if ($script:WingetAvailable -and -not $DetectedPython) {
            # --source winget avoids msstore, whose cert-pinning (0x8a15005e) can abort the install.
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
                # Covers real failures and "already installed" codes where Python is not on PATH.
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

        # $pythonOrgTried, so an offline box inside conda does not sit through the same
        # failing download twice and then read "falling back to python.org" for a fallback
        # that already ran.
        if (-not $DetectedPython -and -not $pythonOrgTried) {
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
        }
    }
    if (-not $DetectedPython -and -not $PythonFromUv) {
        . $InstallSystemPython
        if (-not $DetectedPython) { return (Exit-InstallFailure "Python installation failed") }
    }
    # Re-probe for the interpreter actually selected: every native decision is keyed to a cp3XX tag.
    $WoaProbedMinor = $PythonVersion
    $WoaProbedFreeThreaded = $false
    $WoaDetectedFreeThreaded = $false
    if ($DetectedPython -and (Get-HostMachineArch) -eq "arm64") {
        $WoaDetectedFreeThreaded = Test-PythonFreeThreaded -PythonExe $DetectedPython.Path
    }
    if ($DetectedPython -and (
            ($DetectedPython.Version -ne $WoaProbedMinor) -or
            ($WoaDetectedFreeThreaded -ne $WoaProbedFreeThreaded))) {
        $WoaNativeBeforeReprobe = $script:WoaNativeCudaTorch
        $WoaProbedMinor = $DetectedPython.Version
        $WoaProbedFreeThreaded = $WoaDetectedFreeThreaded
        Initialize-WoaNativeCudaTorch -PythonMinor $DetectedPython.Version `
            -FreeThreaded $WoaDetectedFreeThreaded
        if ($script:WoaNativeCudaTorch -ne $WoaNativeBeforeReprobe) {
            $ReselectedPython = Remove-SkippedPython (Find-CompatiblePython)
            if ($ReselectedPython) {
                # The reselection can hand back an interpreter the probe has not answered for, so that one is probed too; when it has no stack the probe's interpreter is kept and its answer restored.
                $_woaReselectedFreeThreaded = $false
                if ((Get-HostMachineArch) -eq "arm64") {
                    $_woaReselectedFreeThreaded = Test-PythonFreeThreaded -PythonExe $ReselectedPython.Path
                }
                if (($ReselectedPython.Version -ne $WoaProbedMinor) -or
                    ($_woaReselectedFreeThreaded -ne $WoaProbedFreeThreaded)) {
                    Initialize-WoaNativeCudaTorch -PythonMinor $ReselectedPython.Version `
                        -FreeThreaded $_woaReselectedFreeThreaded
                    if ($script:WoaNativeCudaTorch) {
                        $WoaProbedMinor = $ReselectedPython.Version
                        $WoaProbedFreeThreaded = $_woaReselectedFreeThreaded
                        $DetectedPython = $ReselectedPython
                    } else {
                        substep "windows on arm: Python $($ReselectedPython.Version) has no win_arm64 CUDA stack; keeping Python $($DetectedPython.Version), which has one." "Yellow"
                        Initialize-WoaNativeCudaTorch -PythonMinor $WoaProbedMinor `
                            -FreeThreaded $WoaProbedFreeThreaded
                    }
                } else {
                    $DetectedPython = $ReselectedPython
                }
            }
            if ($script:WoaNativeCudaTorch) {
                step "gpu" "Windows on ARM + NVIDIA: native CUDA wheels available -- installing the ARM64 stack" "Green"
                substep "torch index: $(Remove-IndexUrlCredentials $script:WoaTorchIndexUrl)"
            } else {
                substep "windows on arm: no win_arm64 CUDA stack for Python $($DetectedPython.Version) -- using the x64 stack." "Yellow"
            }
        }
    }
    # An emulated x64 python cannot load the win_arm64 wheels the probe just found.
    if ($script:WoaNativeCudaTorch -and $DetectedPython -and $DetectedPython.Arch -eq "x86_64") {
        substep "windows on arm: found an x64 Python $($DetectedPython.Version); installing the native ARM64 build..." "Yellow"
        $Arm64Python = Install-PythonFromPythonOrg -Arch "arm64"
        if ($Arm64Python -and $Arm64Python.Arch -ne "x86_64") {
            $DetectedPython = $Arm64Python
            step "python" "using native ARM64 Python $($DetectedPython.Version)"
            $_woaNewFreeThreaded = Test-PythonFreeThreaded -PythonExe $DetectedPython.Path
            if (($DetectedPython.Version -ne $WoaProbedMinor) -or
                ($_woaNewFreeThreaded -ne $WoaProbedFreeThreaded)) {
                $WoaProbedMinor = $DetectedPython.Version
                $WoaProbedFreeThreaded = $_woaNewFreeThreaded
                Initialize-WoaNativeCudaTorch -PythonMinor $DetectedPython.Version `
                    -FreeThreaded $_woaNewFreeThreaded
                if (-not $script:WoaNativeCudaTorch) {
                    substep "windows on arm: no win_arm64 CUDA stack for Python $($DetectedPython.Version) -- using the x64 stack." "Yellow"
                }
            }
        } else {
            substep "could not install a native ARM64 Python; falling back to the x64 stack." "Yellow"
            $script:WoaNativeCudaTorch = $false
            $script:WoaTorchIndexUrl = $null
        }
    }
    # ── Windows on ARM: the opt-out's last ARM64 interpreter ──
    # With no ARM64 interpreter visible, try the venv's own base interpreter from pyvenv.cfg.
    if ((Get-HostMachineArch) -eq "arm64" -and (Test-Arm64PythonOptOut) -and
            (-not $DetectedPython -or $DetectedPython.Arch -ne "arm64")) {
        $ArmBasePython = Get-StudioVenvBasePython -VenvDir $VenvDir
        if ($ArmBasePython) {
            step "python" "windows on arm: UNSLOTH_ALLOW_ARM64_PYTHON is set and no other ARM64 Python is registered; using the ARM64 Python $($ArmBasePython.Version) the existing environment was built from"
            $DetectedPython = $ArmBasePython
        }
    }

    # ── Windows on ARM: swap a native ARM64 interpreter for x64 ──
    # Stops without x64 (see Resolve-WindowsOnArmX64Python). Not on a native CUDA host.
    if (-not $script:WoaNativeCudaTorch -and
            $DetectedPython -and (Get-HostMachineArch) -eq "arm64" -and $DetectedPython.Arch -ne "x86_64") {
        $DetectedPython = Resolve-WindowsOnArmX64Python -SelectedPython $DetectedPython
        if (-not $DetectedPython) {
            return (Exit-InstallFailure "windows on arm: no x64 Python could be installed, and ARM64 Python cannot resolve pyarrow or hf-transfer")
        }
    }

    $DiagPythonVersion = $PythonVersion
    if ($DetectedPython) { $DiagPythonVersion = $DetectedPython.Version }
    $InitialGpuBranch = "unknown"
    if ($SkipTorch) { $InitialGpuBranch = "no_torch" }
    Write-TauriDiag -GpuBranch $InitialGpuBranch -TorchIndexFamily "none" -PythonVersionForDiag $DiagPythonVersion

    foreach ($_mirrorEnvName in @('_UNSLOTH_MIRROR_PROBED', '_UNSLOTH_MIRROR_SPARE', 'UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_INDEX_STRATEGY', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'UNSLOTH_PYTORCH_MIRROR', 'UNSLOTH_NODE_MIRROR', 'UNSLOTH_NPM_REGISTRY', 'UNSLOTH_UV_WHEEL_MIRROR')) {
        $script:MirrorEnvSaved[$_mirrorEnvName] = [Environment]::GetEnvironmentVariable($_mirrorEnvName)
    }
    Invoke-MirrorFallback

    # ── Install uv ──
    Write-TauriLog "STEP" "Installing uv package manager"
    $UvMinVersion = "0.8.16"

    # Profile aliases/functions outrank PATH, so resolve a real executable. $null means none.
    function Resolve-UvExecutable {
        # @(): a one-element return unrolls to a bare string, and indexing THAT gives its
        # first character.
        $candidates = @(Get-UvExecutableCandidates)
        if ($candidates.Count -gt 0) { return $candidates[0] }
        return $null
    }

    # Every uv in bare-token order (alias, then PATH), so the version gate can skip a stale one.
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
        # Never hand back the bare token: a wrapper could pass the gate and receive every command.
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
        # "ok", "failed" or "unknown". Only a non-zero exit is "failed"; a throw or timeout
        # (seen in Windows containers and on arm64) is no verdict, and the digest already
        # verified the bytes.
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
            # The timed overload can return before the exit code is cached; the parameterless wait
            # settles it.
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

    # Fallback for hosts without winget: same archive and destination as astral's install.ps1,
    # but a pinned SHA-256 instead of running remote script text.
    # tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
    # Bumping the version means bumping all 3 hashes, and each Wheel/WheelSha256 from https://pypi.org/pypi/uv/<ver>/json:
    #   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/uv-<arch>-pc-windows-msvc.zip.sha256
    $UvPinnedVersion = "0.12.1"
    $UvPinnedAssets = @{
        "x86_64" = @{ Asset = "uv-x86_64-pc-windows-msvc.zip";  Sha256 = "8FCB0CB46E1229065E344758980924E569BEF5882EF45F46FADA8FB24E06B74A"
                      Wheel = "packages/0d/a4/467c99c76fefa8b1259a1d382a5e49f73068f38a2d58db401504a783ed2c/uv-0.12.1-py3-none-win_amd64.whl"; WheelSha256 = "BD02F2DA212E6A983115DC64A6FC94E9256C2D60E056D6B669DE0A6025AAEC05" }
        "arm64"  = @{ Asset = "uv-aarch64-pc-windows-msvc.zip"; Sha256 = "9BC7C18E616230FA2DC6FB24BC3AFDE18A95C2B5C9433DE747E9502C66041568"
                      Wheel = "packages/68/80/ec1acbf8e22dc4866f9070c30b064728cc0da73bedc30f2fbfdc0c5901a7/uv-0.12.1-py3-none-win_arm64.whl"; WheelSha256 = "EAD7AD064F291A5DF358C3FFA8FFAB347A32BD5A75A6A068CA22254C2539A829" }
        "x86"    = @{ Asset = "uv-i686-pc-windows-msvc.zip";    Sha256 = "9B51C33D307A8AB9E9DFD88D4AE1491761F63DE0BFFA3CEC96BEC536491C9B97"
                      Wheel = "packages/fd/02/f73e4867c0748eaa3dea90cdfeb73d15bab0f04802c5c20bb37fc14918fe/uv-0.12.1-py3-none-win32.whl"; WheelSha256 = "173EE216F17D89FC39F65339D311A53584FC7DE4918D27C0F3C7EDAFABC6B54D" }
    }

    function Install-UvFromRelease {
        $script:UvReleaseUnfetched = $false
        $arch = Get-HostMachineArch
        if (-not $UvPinnedAssets.ContainsKey($arch)) {
            substep "No uv build is published for this architecture ($arch)." "Yellow"
            return $false
        }
        $asset  = $UvPinnedAssets[$arch].Asset
        $wanted = $UvPinnedAssets[$arch].Sha256
        $remote = $asset
        $wheelDir = $null

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

        # astral's sources in astral's order, each exclusive when set; the pin still applies.
        $uvBase = if ($env:UV_DOWNLOAD_URL) {
            @("$($env:UV_DOWNLOAD_URL.TrimEnd('/'))")
        } elseif ($env:INSTALLER_DOWNLOAD_URL) {
            @("$($env:INSTALLER_DOWNLOAD_URL.TrimEnd('/'))")
        } elseif ($env:UV_INSTALLER_GHE_BASE_URL) {
            @("$($env:UV_INSTALLER_GHE_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
        } elseif ($env:UV_INSTALLER_GITHUB_BASE_URL) {
            @("$($env:UV_INSTALLER_GITHUB_BASE_URL.TrimEnd('/'))/astral-sh/uv/releases/download/$UvPinnedVersion")
        } elseif ($env:UNSLOTH_UV_WHEEL_MIRROR) {
            $remote = $UvPinnedAssets[$arch].Wheel
            $wanted = $UvPinnedAssets[$arch].WheelSha256
            $wheelDir = "uv-$UvPinnedVersion.data/scripts"
            @("$($env:UNSLOTH_UV_WHEEL_MIRROR.TrimEnd('/'))")
        } else {
            @("https://releases.astral.sh/github/uv/releases/download/$UvPinnedVersion",
              "https://github.com/astral-sh/uv/releases/download/$UvPinnedVersion")
        }

        $work = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-uv-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
        $zip  = Join-Path $work $asset
        try {
            [System.IO.Directory]::CreateDirectory($work) | Out-Null
            # Digest per mirror: a proxy can answer 200 with its own body.
            $downloaded = $false
            $script:UvReleaseUnfetched = $true
            foreach ($base in $uvBase) {
                substep "downloading uv $UvPinnedVersion ($arch) from $base..." "Yellow"
                try {
                    Invoke-WebRequest -UseBasicParsing -OutFile $zip -Uri "$base/$remote"
                } catch {
                    substep "uv download failed: $($_.Exception.Message)" "Yellow"
                    continue
                }
                $script:UvReleaseUnfetched = $false
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

            Expand-Archive -LiteralPath $zip -DestinationPath $work -Force
            [System.IO.Directory]::CreateDirectory($destDir) | Out-Null
            $srcRoot = if ($wheelDir) { Join-Path $work $wheelDir } else { $work }

            $stagedUv = Join-Path $srcRoot "uv.exe"
            if (-not (Test-Path -LiteralPath $stagedUv)) {
                substep "uv.exe was not present in $asset." "Yellow"
                return $false
            }
            # Run the staged uv before touching the destination, so a policy-blocked new uv cannot
            # replace a working old one.
            if ((Get-UvExecutableVerdict -Path $stagedUv) -eq "failed") {
                substep "the downloaded uv $UvPinnedVersion could not run on this machine." "Yellow"
                return $false
            }

            # uvw.exe has no console to probe; the staged uv.exe stands for the verified set. Copy
            # under Stop so a half-copied set fails the install.
            $ok = $true
            foreach ($exe in @("uv.exe", "uvx.exe", "uvw.exe")) {
                $src = Join-Path $srcRoot $exe
                if (-not (Test-Path -LiteralPath $src)) { continue }
                $dst = Join-Path $destDir $exe
                try {
                    Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
                } catch {
                    $ok = $false
                    break
                }
                if ($exe -eq "uv.exe") {
                    # Compare against the verified archive: a stale uv.exe must not pass for ours.
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
            if (@(Install-UvFromRelease)[-1] -ne $true -and $script:UvReleaseUnfetched -and (Use-MirrorSpare uvbin)) { Install-UvFromRelease | Out-Null }
            Refresh-SessionPath
        }
    }

    # A freshly installed uv can sit behind an older one on PATH; prefer known install locations.
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

    # Ahead of the cache setup (which returns early for a preset UV_CACHE_DIR) and outside
    # it: tests/python/test_windows_python_venv_hardening.py runs that function standalone.
    Write-StudioRootOwnerMarker -Root $StudioHome
    Set-StudioUvCacheEnvironment -StudioRoot $StudioHome -Isolated $IsolateUvCache -UvExecutable $script:UvExe
    # Bytecode compilation can exceed uv's 60s default on slow machines ("0" disables).
    if (-not $env:UV_COMPILE_BYTECODE_TIMEOUT) {
        $env:UV_COMPILE_BYTECODE_TIMEOUT = "180"
    }

    if (-not $env:UV_HTTP_RETRIES) {
        $env:UV_HTTP_RETRIES = "5"
    }
    if (-not $env:UV_HTTP_TIMEOUT) {
        $env:UV_HTTP_TIMEOUT = "180"
    }

    # --no-bin / --no-registry: only uv's own Python store changes. --system: `find` would
    # otherwise answer with an active venv. $null falls back to the system install.
    function Resolve-UvManagedPython {
        # A range excluding $PythonSkip, as install.sh's _python_request.
        $request = $PythonVersion
        $bad = @($PythonSkip | Where-Object { $_ -like "$PythonVersion.*" })
        if ($bad.Count -gt 0 -and $PythonVersion -match '^(\d+)\.(\d+)$') {
            $request = ">=$PythonVersion,<$($Matches[1]).$([int]$Matches[2] + 1)" + (($bad | ForEach-Object { ",!=$_" }) -join "")
        }
        $installExit = Invoke-InstallCommand -NoMirror -Label "install uv-managed Python $request" {
            & $script:UvExe python install --no-bin --no-registry $request
        }
        if ($installExit -ne 0) { return $null }
        $exe = ""
        try {
            $exe = (& $script:UvExe python find --system --managed-python $request 2>$null | Select-Object -First 1)
        } catch {}
        $exe = "$exe".Trim()
        if (-not $exe -or -not (Test-Path -LiteralPath $exe -PathType Leaf)) { return $null }
        # -S: a sitecustomize banner would otherwise be read as the version.
        $full = ""
        try {
            $full = (& $exe -S -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null | Select-Object -First 1)
        } catch {}
        $full = "$full".Trim()
        if ($full -notmatch '^(3\.1[1-3])\.\d+$') { return $null }
        $minor = $Matches[1]
        # find answers through uv's per-minor link, which can still name a skipped patch.
        if ($PythonSkip -contains $full) { return $null }
        return @{ Version = $minor; Path = $exe; Arch = "" }
    }

    if ($PythonFromUv) {
        $DetectedPython = Resolve-UvManagedPython
        if ($DetectedPython) {
            step "python" "using uv-managed Python $($DetectedPython.Version)"
        } else {
            substep "uv could not provide Python $PythonVersion -- installing it system-wide instead." "Yellow"
            . $InstallSystemPython
            if (-not $DetectedPython) { return (Exit-InstallFailure "Python installation failed") }
        }
    }

    # ── Create the venv; hand uv the resolved exe path so it does not re-resolve back to conda. ──
    Write-TauriLog "STEP" "Creating virtual environment"
    Write-StudioRootOwnerMarker -Root $StudioHome

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $_Migrated = $false
    $script:StudioVenvRollbackDir = $null
    $script:StudioVenvRollbackTarget = $VenvDir
    # Only the Windows-on-ARM rebuild preserves its rollback; others are deleted on commit.
    $script:StudioVenvRollbackPreserve = $false
    # The existing interpreter's platform tag, read before any rollback move takes it away.
    $script:PrevVenvPlatformTag = $null
    $script:StudioVenvRollbackActive = $false
    # Distinguishes "deleted on purpose" (--no-rollback) from "not moved aside yet".
    $script:StudioVenvRollbackDiscarded = $false
    # Set by the discard itself, so the disk-full remedy describes what happened rather than what
    # was requested.
    $script:StudioVenvDiscardSucceeded = $false
    $script:StudioVenvDiscardLeftover = $null
    $script:StudioVenvRollbackPartial = $false
    # Reset per run: under `irm | iex` the script scope IS the caller's session.
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

    # uv creates only into an absent or empty directory; the .NET API counts hidden entries.
    function Test-DirectoryHasEntries {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
            # Get-Item sees a file or dangling link, which uv also refuses.
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

    # Move-Item into an existing dir nests the source (#9479), so clear the empty target first:
    # Directory.Delete is non-recursive and unlinks reparse points without following.
    function Clear-MigrationTargetDirectory {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not (Test-Path -LiteralPath $Path)) { return }
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            # Not Remove-Item (5.1 reparse-tag bug, PowerShell#621); File.Delete for file/dangling links.
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

    # Test-Path follows links, so a dangling one reads as absent; ask the path itself.
    function Test-StudioPathPresent {
        param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Path)
        if (-not $Path) { return $false }
        return ($null -ne (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue))
    }

    # Bounded probe: a wedged torch import or Intel driver init would hang `& python`.
    # ProcessStartInfo, both streams drained async, kill at the deadline. .Error says which
    # failure. Defined above the Intel scan (functions bind when they run).
    function Invoke-BoundedPythonProbe {
        param([string]$PythonExe, [string]$Code, [int]$TimeoutSec = 30)
        $result = [pscustomobject]@{ Ok = $false; Output = ""; Error = "" }
        if (-not $PythonExe -or -not $Code) { return $result }
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $PythonExe
            $psi.Arguments = "-I -c `"$Code`""
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
        # Publish the rollback state before the rename so an interruption cannot lose the old venv.
        try {
            Move-Item -LiteralPath $ExistingDir -Destination $candidate -ErrorAction Stop
        } catch {
            # An open handle can fail the rename partway, splitting the tree. Clear the state only when
            # the destination is absent; otherwise keep it so Restore-StudioVenvRollback can merge.
            $candidateExists = Test-Path -LiteralPath $candidate
            if ((Test-Path -LiteralPath $ExistingDir) -and (-not $candidateExists)) {
                $script:StudioVenvRollbackActive = $false
                $script:StudioVenvRollbackDir = $null
            } elseif ($candidateExists -and (Test-Path -LiteralPath $ExistingDir)) {
                # Both paths hold halves of the same environment: restore must merge.
                $script:StudioVenvRollbackPartial = $true
                Write-StudioLine "[WARN] Moving the existing environment aside stopped partway -- files are in both places." -ForegroundColor Yellow
                Write-StudioLine "       still in place: $ExistingDir" -ForegroundColor Yellow
                Write-StudioLine "       moved aside:    $candidate" -ForegroundColor Yellow
                Write-StudioLine "       A running 'unsloth studio' process usually holds a file open here." -ForegroundColor Yellow
                Write-StudioLine "       Close Unsloth Studio and re-run the installer to reverse the move." -ForegroundColor Yellow
            }
            throw
        }
        # --no-rollback: drop the old environment now (disk-constrained case). Rename first so uv
        # never builds into an occupied path; clear state before deleting.
        if ($script:StudioNoRollback) {
            $discard = $script:StudioVenvRollbackDir
            # Take the XPU verdict from the old environment's torch before deleting it. Wrapped so a
            # failed probe cannot cost the rename.
            try {
                $_discardPy = Join-Path $discard "Scripts\python.exe"
                if (Test-Path -LiteralPath $_discardPy -ErrorAction SilentlyContinue) {
                    $_discardXpu = Invoke-BoundedPythonProbe -PythonExe $_discardPy `
                        -Code 'import torch; print(torch.xpu.is_available())'
                    if ($_discardXpu.Ok -and $_discardXpu.Output -match '(?m)^\s*True\s*$') {
                        $script:StudioPreservedXpuVerdict = $true
                    }
                }
            } catch { }
            $script:StudioVenvRollbackActive = $false
            $script:StudioVenvRollbackDir = $null
            # Distinct from "never started". Inactive alone cannot tell the two apart, and the
            # ARM64 migration below reads it to decide whether it still has a tree to move.
            $script:StudioVenvRollbackDiscarded = $true
            # Never report success when the tree survived.
            if (Remove-StudioVenvTreeWithRetry -Path $discard -Label "previous environment" -LinkAware) {
                $script:StudioVenvDiscardSucceeded = $true
                substep "previous environment discarded (--no-rollback); a failed install cannot be undone"
            } else {
                $script:StudioVenvDiscardLeftover = $discard
                substep "it is no longer used for rollback; remove $discard by hand to reclaim the space." "Yellow"
            }
            return
        }
        substep "previous environment preserved for rollback"
    }

    function Remove-StudioVenvTreeWithRetry {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Label,
            # Opt-in so installs without --no-rollback report exactly as before.
            [switch]$LinkAware
        )
        $lastError = $null
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            } catch {
                $lastError = $_.Exception.Message
            }
            # Test-Path follows a dangling reparse point and would report a removal that did not happen.
            $stillThere = if ($LinkAware) { Test-StudioPathPresent -Path $Path } else { Test-Path -LiteralPath $Path }
            if (-not $stillThere) { return $true }
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
            # Same relative path on both sides: recurse to reunite halves instead of overwriting.
            # Junctions are leaves, never recursed through. Get-Item both sides for reliable Attributes.
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
        # The same flag the marker restore consults, so an interruption mid-commit cannot
        # put one half of a committed install back and keep the other.
        if ($script:StudioInstallCommitted) { return }
        if (-not $script:StudioVenvRollbackActive) { return }
        # A restore puts the tree back at $VenvDir, so there is nothing left to preserve and
        # the flag must not survive into any later rollback in the same run.
        $script:StudioVenvRollbackPreserve = $false
        $backup = $script:StudioVenvRollbackDir
        $target = $script:StudioVenvRollbackTarget
        if (-not (Test-StudioPathPresent -Path $backup)) {
            $script:StudioVenvRollbackActive = $false
            $script:StudioVenvRollbackPartial = $false
            return
        }
        substep "restoring previous environment after failed install..." "Yellow"
        if ($script:StudioVenvRollbackPartial) {
            # $target is the other half of the previous environment here: merge, never delete.
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
        # First and alone: this one assignment is what both restores consult, so an
        # interruption cannot find them disagreeing. A first install rolls nothing back
        # and still commits.
        $script:StudioInstallCommitted = $true
        $script:StudioUvMarkerSaved = $false
        if (-not $script:StudioVenvRollbackActive) { return }
        $backup = $script:StudioVenvRollbackDir
        # The replacement is committed. Disable restoration before deleting the
        # backup so interruption cannot restore a partially deleted environment.
        $script:StudioVenvRollbackActive = $false
        $script:StudioVenvRollbackDir = $null
        $script:StudioVenvRollbackPartial = $false
        $preserve = $script:StudioVenvRollbackPreserve
        $script:StudioVenvRollbackPreserve = $false
        if (-not (Test-StudioPathPresent -Path $backup)) { return }
        if ($preserve) {
            # Keep the ARM64 tree as promised, renamed out of unsloth_studio.rollback.* so the next
            # run's sweep does not delete it.
            $stamp = Get-Date -Format "yyyyMMddHHmmss"
            $kept = Join-Path $StudioHome "unsloth_studio.arm64.$stamp"
            $suffix = 0
            while (Test-Path -LiteralPath $kept) {
                $suffix++
                $kept = Join-Path $StudioHome "unsloth_studio.arm64.$stamp.$suffix"
            }
            try {
                Move-Item -LiteralPath $backup -Destination $kept -ErrorAction Stop
                substep "previous ARM64 environment kept at $kept"
                substep "delete it once you have re-installed any extra packages you had added to it."
                return
            } catch {
                # Naming it is the whole point, so a failed rename reports where it actually
                # is rather than silently deleting the copy the user was told to expect.
                Write-StudioLine "[WARN] Could not rename the previous ARM64 environment: $($_.Exception.Message)" -ForegroundColor Yellow
                Write-StudioLine "       It is still at $backup" -ForegroundColor Yellow
                return
            }
        }
        Remove-StudioVenvTreeWithRetry -Path $backup -Label "environment rollback" | Out-Null
    }

    # Bounded probe (async-drained, 30s) so a wedged "import torch" cannot stall the installer.
    function Get-InstalledTorchVersionRaw {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $PythonExe
            # Dist metadata, not "import torch": a broken DLL would drop the pin (as in install.sh).
            $psi.Arguments = '-I -c "import importlib.metadata as m; print(m.version(''torch''))"'
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            # Drain BOTH streams async before WaitForExit, or a wedged import deadlocks.
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            $finished = $proc.WaitForExit(30000)
            if (-not $finished) { try { $proc.Kill() } catch {}; return $null }
            $out = $outTask.GetAwaiter().GetResult()
            [void]$errTask.GetAwaiter().GetResult()
            if ($proc.ExitCode -ne 0) { return $null }
            # Last non-empty line only, so stdout noise cannot corrupt the pin.
            $lines = @($out -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
            if ($lines.Count -eq 0) { return $null }
            return $lines[-1]
        } catch { return $null }
    }

    $studioVenvReplacementCommitted = $false
    try {
    # Replace occupied venvs even when python.exe is missing, as in #9479.
    if ((Test-Path -LiteralPath $VenvPython) -or (Test-DirectoryHasEntries -Path $VenvDir)) {
        # why: matching guard to the .venv branch below. Env mode: refuse to replace an existing
        # $StudioHome\unsloth_studio without Unsloth sentinels (the in-venv marker and a
        # content-checked .cmd count). This gates a recursive delete.
        if (
            $StudioRedirectMode -eq 'env' -and
            # Test-StudioPlainFile, not Test-Path: the claim refuses to write a marker through a
            # link, so reading one through a link here would undo that decision.
            -not (Test-StudioPlainFile -Path (Join-Path $StudioHome ".unsloth-studio-owned")) -and
            -not (Test-Path -LiteralPath (Join-Path $VenvDir ".unsloth-studio-owned") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "share\studio.conf") -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $StudioHome "bin\unsloth.exe") -PathType Leaf) -and
            -not (Test-UnslothCmdShimFile (Join-Path $StudioHome "bin\unsloth.cmd"))
        ) {
            Write-StudioLine "[ERROR] $VenvDir already exists but does not look like an Unsloth Studio install." -ForegroundColor Red
            Write-StudioLine "        Move it aside or choose an empty UNSLOTH_STUDIO_HOME." -ForegroundColor Yellow
            throw "Refusing to delete non-Unsloth venv at $VenvDir"
        }
        # Record the release before the rollback move; only the new-layout replace needs it.
        if (-not $SkipTorch) {
            $script:PrevTorchVer = Get-InstalledTorchVersionRaw -PythonExe $VenvPython
        }
        # Record the interpreter's arch now, before the move takes $VenvPython away.
        if ((Get-HostMachineArch) -eq "arm64" -and (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
            $script:PrevVenvPlatformTag = Get-PythonPlatformTag $VenvPython
        }
        # Replace the new layout after preserving a rollback copy (unless --no-rollback).
        if ($script:StudioNoRollback) {
            substep "moving the existing environment aside..."
        } else {
            substep "preserving existing environment for rollback..."
        }
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
                & $OldPy -I -c "import torch; A = torch.ones((2,2)); B = A + A" 2>$null | Out-Null
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

    # Re-check the venv about to be REUSED (Test-StudioVenvArchMismatch); preserved through the
    # rollback helper and cleared so a fresh x64 venv is built.
    if (Test-StudioVenvArchMismatch -VenvPython $VenvPython -SelectedPython $DetectedPython `
            -RecordedTag $script:PrevVenvPlatformTag) {
        substep "windows on arm: the existing environment runs native ARM64 Python, which cannot" "Yellow"
        substep "resolve pyarrow or hf-transfer -- rebuilding it on x64 Python $($DetectedPython.Version)." "Yellow"
        # Tell the user packages are not carried over and the old tree is kept as
        # unsloth_studio.arm64.*, unless --no-rollback already discarded it.
        if (-not ($script:StudioVenvRollbackDiscarded -or $script:StudioNoRollback)) {
            substep "the ARM64 environment is kept under $StudioHome as unsloth_studio.arm64.*;" "Yellow"
            substep "re-install any extra packages you had added to it, or set UNSLOTH_ALLOW_ARM64_PYTHON=1 to keep it." "Yellow"
        }
        try {
            if ($script:StudioVenvRollbackDiscarded) {
                # --no-rollback already deleted it; a second rollback would throw.
                substep "the previous ARM64 environment was discarded by --no-rollback, so there is" "Yellow"
                substep "nothing to copy packages out of; the x64 environment is built fresh." "Yellow"
            } elseif ($script:StudioVenvRollbackActive) {
                # Already moved aside by the ordinary reinstall; just mark it kept.
                $script:StudioVenvRollbackPreserve = $true
            } else {
                Start-StudioVenvRollback -ExistingDir $VenvDir
                if ($script:StudioVenvRollbackDiscarded) {
                    # Discarded under --no-rollback, so there is nothing to preserve.
                    substep "the previous ARM64 environment was discarded by --no-rollback rather than kept;" "Yellow"
                    substep "the x64 environment is built fresh, and extra packages are not recoverable." "Yellow"
                } else {
                    # After the move, so a rollback that never started cannot leave the flag set
                    # for some later unrelated rollback to act on.
                    $script:StudioVenvRollbackPreserve = $true
                }
            }
        } catch {
            Write-StudioLine "[ERROR] Could not move the ARM64 environment aside: $($_.Exception.Message)" -ForegroundColor Red
            Write-StudioLine "        Close Unsloth Studio, including its tray process, then re-run install.ps1." -ForegroundColor Yellow
            return (Exit-InstallFailure "Could not replace the native ARM64 environment at $VenvDir")
        }
        # Left set, the migrated-environment branch below would --no-deps into a fresh venv.
        $_Migrated = $false
    }

    # The opt-out must actually keep the ARM64 tree (as unsloth_studio.arm64.*), or the
    # rollback would delete it on success.
    if ($script:PrevVenvPlatformTag -eq "win-arm64" -and (Test-Arm64PythonOptOut) -and
        $script:StudioVenvRollbackActive) {
        $script:StudioVenvRollbackPreserve = $true
        substep "windows on arm: keeping the existing native ARM64 environment under" "Yellow"
        substep "$StudioHome as unsloth_studio.arm64.*; the new environment is built on" "Yellow"
        substep "Python $($DetectedPython.Version) ($($DetectedPython.Arch))." "Yellow"
    }

    # A migrated environment has to be ARM64 too: an x64 one has a Triton that cannot do sm_121.
    if ($script:WoaNativeCudaTorch -and $_Migrated -and (Test-Path -LiteralPath $VenvPython)) {
        $_woaMigPlatform = ""
        try {
            $_woaMigPlatform = (& $VenvPython -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
        } catch { $_woaMigPlatform = "" }
        if ($_woaMigPlatform -ne "win-arm64") {
            $_woaMigLabel = if ($_woaMigPlatform) { $_woaMigPlatform } else { "unknown" }
            substep "windows on arm: the migrated environment is $_woaMigLabel, not win-arm64 --" "Yellow"
            # Under --no-rollback do not claim the previous environment is recoverable.
            if ($script:StudioNoRollback) {
                substep "rebuilding it as ARM64; --no-rollback discards the previous one rather than keeping it." "Yellow"
            } else {
                substep "rebuilding it as ARM64; the previous one is kept for rollback." "Yellow"
            }
            try {
                Start-StudioVenvRollback -ExistingDir $VenvDir
                # Left set, the install step below would --no-deps into an empty venv.
                $_Migrated = $false
            } catch {
                substep "could not preserve it for rollback -- using the x64 stack instead." "Yellow"
            }
        }
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

    # The venv itself has to be ARM64 and nothing downstream re-checks: an x64 one aborts here.
    $WoaVenvMinor = if ($DetectedPython) { $DetectedPython.Version } else { $PythonVersion }
    if ($script:WoaNativeCudaTorch) {
        $_woaVenvPlatform = ""
        try {
            $_woaVenvPlatform = (& $VenvPython -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
        } catch { $_woaVenvPlatform = "" }
        if ($_woaVenvPlatform -ne "win-arm64") {
            $_woaVenvLabel = if ($_woaVenvPlatform) { $_woaVenvPlatform } else { "unknown" }
            substep "windows on arm: the existing environment is $_woaVenvLabel, not win-arm64 -- using the x64 stack." "Yellow"
            $script:WoaNativeCudaTorch = $false
            $script:WoaTorchIndexUrl = $null
        } else {
            $_woaVenvVersion = ""
            try {
                $_woaVenvVersion = (& $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null | Out-String).Trim()
            } catch { $_woaVenvVersion = "" }
            if ($_woaVenvVersion -match '^\d+\.\d+$') { $WoaVenvMinor = $_woaVenvVersion }
            $script:WoaVenvFreeThreaded = Test-PythonFreeThreaded -PythonExe $VenvPython
        }
    }

    # Drop what a PREVIOUS run left and nothing else, or an air-gapped user's mirror goes with it.
    $_woaOwnedPrefix = Join-Path $StudioHome "woa"
    $_woaOwnedPrefixes = @($_woaOwnedPrefix, (Get-UvSafePath $_woaOwnedPrefix)) |
        Where-Object { $_ } | Select-Object -Unique
    # Split with the separator it is joined with: a whitespace split tears "C:\a b" in two.
    $_woaJoinWith = @{ "UV_OVERRIDE" = " "; "UV_FIND_LINKS" = ","; "PIP_FIND_LINKS" = " " }
    $_woaSplitOn = @{ "UV_OVERRIDE" = '\s+'; "UV_FIND_LINKS" = ','; "PIP_FIND_LINKS" = '\s+' }
    foreach ($_woaResolverVar in 'UV_OVERRIDE', 'UV_FIND_LINKS', 'PIP_FIND_LINKS') {
        $_woaInherited = [Environment]::GetEnvironmentVariable($_woaResolverVar)
        if (-not $_woaInherited) { continue }
        $_woaSep = $_woaSplitOn[$_woaResolverVar]
        $_woaKept = @()
        foreach ($_woaEntry in ($_woaInherited -split $_woaSep | ForEach-Object { $_.Trim() })) {
            if (-not $_woaEntry) { continue }
            $_woaOwned = $false
            foreach ($_woaPrefix in $_woaOwnedPrefixes) {
                if ($_woaEntry.Equals($_woaPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
                    $_woaEntry.StartsWith("$_woaPrefix\", [System.StringComparison]::OrdinalIgnoreCase) -or
                    $_woaEntry.StartsWith("$_woaPrefix/", [System.StringComparison]::OrdinalIgnoreCase)) {
                    $_woaOwned = $true
                    break
                }
            }
            if (-not $_woaOwned) { $_woaKept += $_woaEntry }
        }
        if ($_woaKept.Count -eq 0) {
            Remove-Item "Env:$_woaResolverVar" -ErrorAction SilentlyContinue
        } elseif ($_woaKept.Count -ne @($_woaInherited -split $_woaSep | ForEach-Object { $_.Trim() } | Where-Object { $_ }).Count) {
            Set-Item "Env:$_woaResolverVar" -Value ($_woaKept -join $_woaJoinWith[$_woaResolverVar])
        }
    }
    if ($script:WoaNativeCudaTorch) {
        $WoaDir = Join-Path $StudioHome "woa"
        $script:WoaDir = $WoaDir
        $WoaWheelDir = Join-Path $WoaDir "wheels"
        try {
            New-Item -ItemType Directory -Force -Path $WoaWheelDir -ErrorAction Stop | Out-Null
            # Checked, not assumed: with a file in the way New-Item can return without creating anything.
            if (-not (Test-Path -LiteralPath $WoaWheelDir -PathType Container)) { throw "$WoaWheelDir was not created; a file may occupy part of the path" }
        } catch {
            # The venv is already native ARM64: standing down here would send the torch step to an index with no win_arm64 wheel, so stop with the reason.
            Write-StudioLine "[ERROR] windows on arm: could not create $WoaWheelDir ($($_.Exception.Message)), and the native ARM64 environment cannot resolve without it." -ForegroundColor Red
            Write-StudioLine "        Move aside whatever occupies that path, or set UNSLOTH_WOA_NATIVE=0 for the x64 stack." -ForegroundColor Yellow
            return (Exit-InstallFailure "windows on arm: could not create $WoaWheelDir")
        }
    }
    if ($script:WoaNativeCudaTorch) {
        if ($script:WoaPyarrowSource -eq "local") {
            $srcWheel = $env:UNSLOTH_PYARROW_WHEEL
            $wheelName = [System.IO.Path]::GetFileName($srcWheel)
            if ($wheelName -notlike "*.whl") {
                $wheelName = Get-WheelFileNameFromArchive -Path $srcWheel
            }
            if (-not $wheelName) {
                substep "UNSLOTH_PYARROW_WHEEL is not a readable wheel: $srcWheel" "Yellow"
                $script:WoaNativeCudaTorch = $false
            } else {
                try {
                    $_woaLocalDest = Join-Path $WoaWheelDir $wheelName
                    if (-not (Test-WoaSamePath $srcWheel $_woaLocalDest)) {
                        Copy-Item -LiteralPath $srcWheel -Destination $_woaLocalDest -Force -ErrorAction Stop
                    }
                    $script:WoaPyarrowWheelName = $wheelName
                    substep "windows on arm: using the supplied pyarrow wheel $wheelName"
                } catch {
                    substep "could not stage $srcWheel ($($_.Exception.Message))." "Yellow"
                    $script:WoaNativeCudaTorch = $false
                }
            }
        } elseif ($script:WoaPyarrowSource -eq "wheelhouse") {
            $tag = "cp" + ($WoaVenvMinor -replace '\.', '')
            $AbiTag = Get-WoaAbiTag -PythonMinor $WoaVenvMinor -FreeThreaded ([bool]$script:WoaVenvFreeThreaded)
            if (Test-WoaWheelhouseIsLocal $script:WoaWheelhouse) {
                $found = Get-ChildItem -LiteralPath $script:WoaWheelhouse -Filter "pyarrow-*win_arm64.whl" -ErrorAction SilentlyContinue |
                    Where-Object {
                        (Test-WoaPyarrowWheelUsable -Name $_.Name -PyTag $tag -AbiTag $AbiTag) -and
                        (Test-ZipArchiveReadable -Path $_.FullName)
                    } | Select-Object -First 1
                if ($found) {
                    $_woaDest = Join-Path $WoaWheelDir $found.Name
                    if (-not (Test-WoaSamePath $found.FullName $_woaDest)) {
                        Copy-Item -LiteralPath $found.FullName -Destination $_woaDest -Force
                    }
                    $script:WoaPyarrowWheelName = $found.Name
                    substep "windows on arm: using pyarrow wheel $($found.Name) from the wheelhouse"
                } else {
                    $script:WoaNativeCudaTorch = $false
                }
            } else {
                $wheelName = $null
                try {
                    $index = [string](Invoke-RestMethod -Uri (Join-UrlPath $script:WoaWheelhouse "index.txt") -UseBasicParsing -TimeoutSec 20)
                    foreach ($match in [regex]::Matches($index, 'pyarrow-[^"''<>\s]*?win_arm64\.whl')) {
                        if (Test-WoaPyarrowWheelUsable -Name $match.Value -PyTag $tag -AbiTag $AbiTag) { $wheelName = $match.Value; break }
                    }
                } catch {}
                if ($wheelName) {
                    try {
                        $_woaPaDest = Join-Path $WoaWheelDir $wheelName
                        Invoke-WebRequest -Uri (Join-UrlPath $script:WoaWheelhouse $wheelName) -OutFile $_woaPaDest -UseBasicParsing -TimeoutSec 300 -ErrorAction Stop
                        # A mirror can serve a truncated body with a 200.
                        if (-not (Test-ZipArchiveReadable -Path $_woaPaDest)) {
                            Remove-Item -LiteralPath $_woaPaDest -Force -ErrorAction SilentlyContinue
                            throw "the downloaded wheel is not a readable archive"
                        }
                        $script:WoaPyarrowWheelName = $wheelName
                        substep "windows on arm: downloaded pyarrow wheel $wheelName"
                    } catch {
                        substep "could not download $wheelName ($($_.Exception.Message))." "Yellow"
                        $script:WoaNativeCudaTorch = $false
                    }
                } else {
                    $script:WoaNativeCudaTorch = $false
                }
            }
        }
        if (-not $script:WoaNativeCudaTorch) {
            # The venv is already native ARM64, and without pyarrow neither the CUDA index nor the x64 one can resolve for it.
            Write-StudioLine "[ERROR] windows on arm: pyarrow could not be staged, and the native ARM64 environment cannot resolve without it." -ForegroundColor Red
            Write-StudioLine "        Re-run the installer (a transient download is the usual cause), or set UNSLOTH_WOA_NATIVE=0 for the x64 stack." -ForegroundColor Yellow
            return (Exit-InstallFailure "windows on arm: pyarrow could not be staged")
        }
    }
    if ($script:WoaNativeCudaTorch) {
        $WoaExtraStaged = 0
        $_woaListing = $null
        $_woaExtraTag = "cp" + ($WoaVenvMinor -replace '\.', '')
        $_woaExtraAbi = Get-WoaAbiTag -PythonMinor $WoaVenvMinor -FreeThreaded ([bool]$script:WoaVenvFreeThreaded)
        if (Test-WoaWheelhouseIsLocal $script:WoaWheelhouse) {
            $_woaListing = @(Get-ChildItem -LiteralPath $script:WoaWheelhouse -Filter "*.whl" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
            foreach ($wheel in @(Get-ChildItem -LiteralPath $script:WoaWheelhouse -Filter "*.whl" -ErrorAction SilentlyContinue)) {
                if ($wheel.Name -like "pyarrow-*") { continue }
                if ($wheel.Name -notlike "*win_arm64*" -and $wheel.Name -notlike "*-any.whl") { continue }
                if (Test-WoaWheelhouseWheelIsRedundant -Name $wheel.Name -PyTag $_woaExtraTag -AbiTag $_woaExtraAbi) {
                    $_woaRedundantKey = (($wheel.Name -split '-')[0] -replace '[-_.]+', '-').ToLowerInvariant()
                    if (-not $script:WoaPyPIProvided.ContainsKey($_woaRedundantKey)) { $script:WoaPyPIProvided[$_woaRedundantKey] = @() }
                    $script:WoaPyPIProvided[$_woaRedundantKey] += [string]$script:WoaPyPIMatchedVersion
                    substep "windows on arm: PyPI publishes $($wheel.Name) itself -- taking it from there, not the wheelhouse."
                    # The managed copy too, or UV_FIND_LINKS still offers it and wins the tie.
                    Remove-Item -LiteralPath (Join-Path $WoaWheelDir $wheel.Name) -Force -ErrorAction SilentlyContinue
                    continue
                }
                # Opened, not just named: _find_links_wheel_versions reads only the filename.
                if (-not (Test-ZipArchiveReadable -Path $wheel.FullName)) {
                    substep "windows on arm: $($wheel.Name) in the wheelhouse is not a readable wheel -- skipping it." "Yellow"
                    continue
                }
                try {
                    $_woaExtraDest = Join-Path $WoaWheelDir $wheel.Name
                    if (-not (Test-WoaSamePath $wheel.FullName $_woaExtraDest)) {
                        Copy-Item -LiteralPath $wheel.FullName -Destination $_woaExtraDest -Force -ErrorAction Stop
                    }
                    $WoaExtraStaged++
                } catch {}
            }
        } else {
            try {
                $index = [string](Invoke-RestMethod -Uri (Join-UrlPath $script:WoaWheelhouse "index.txt") -UseBasicParsing -TimeoutSec 20)
                $_woaListing = @([regex]::Matches($index, '[A-Za-z0-9_.+-]+\.whl') | ForEach-Object { $_.Value })
                foreach ($match in [regex]::Matches($index, '[A-Za-z0-9_.+-]+\.whl')) {
                    $name = $match.Value
                    if ($name -like "pyarrow-*") { continue }
                    if ($name -notlike "*win_arm64*" -and $name -notlike "*-any.whl") { continue }
                    if (Test-WoaWheelhouseWheelIsRedundant -Name $name -PyTag $_woaExtraTag -AbiTag $_woaExtraAbi) {
                        $_woaRedundantKey = (($name -split '-')[0] -replace '[-_.]+', '-').ToLowerInvariant()
                        if (-not $script:WoaPyPIProvided.ContainsKey($_woaRedundantKey)) { $script:WoaPyPIProvided[$_woaRedundantKey] = @() }
                        $script:WoaPyPIProvided[$_woaRedundantKey] += [string]$script:WoaPyPIMatchedVersion
                        substep "windows on arm: PyPI publishes $name itself -- taking it from there, not the wheelhouse."
                        Remove-Item -LiteralPath (Join-Path $WoaWheelDir $name) -Force -ErrorAction SilentlyContinue
                        continue
                    }
                    $dest = Join-Path $WoaWheelDir $name
                    if (Test-Path -LiteralPath $dest) {
                        if (Test-ZipArchiveReadable -Path $dest) { $WoaExtraStaged++; continue }
                        Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
                    }
                    try {
                        Invoke-WebRequest -Uri (Join-UrlPath $script:WoaWheelhouse $name) -OutFile $dest -UseBasicParsing -TimeoutSec 300 -ErrorAction Stop
                        if (-not (Test-ZipArchiveReadable -Path $dest)) {
                            Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
                            continue
                        }
                        $WoaExtraStaged++
                    } catch {
                        Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
                    }
                }
            } catch {}
        }
        # Wheels an EARLIER wheelhouse staged read as hosted below; kept only when the listing could not be read (offline reuse).
        if ($null -ne $_woaListing) {
            $_woaKeep = @(@($_woaListing) + @($script:WoaPyarrowWheelName) | Where-Object { $_ } | ForEach-Object { $_.ToLowerInvariant() })
            $_woaPruned = 0
            foreach ($_woaOld in @(Get-ChildItem -LiteralPath $WoaWheelDir -Filter "*.whl" -ErrorAction SilentlyContinue)) {
                if ($_woaKeep -notcontains $_woaOld.Name.ToLowerInvariant()) {
                    Remove-Item -LiteralPath $_woaOld.FullName -Force -ErrorAction SilentlyContinue
                    $_woaPruned++
                }
            }
            if ($_woaPruned -gt 0) { substep "windows on arm: removed $_woaPruned wheel(s) an earlier wheelhouse left." }
        }
        if ($WoaExtraStaged -gt 0) {
            substep "windows on arm: staged $WoaExtraStaged more wheel(s) from the wheelhouse."
        }
    }
    if ($script:WoaNativeCudaTorch) {
        $WoaOverrides = Join-Path $WoaDir "overrides.txt"
        $WoaWheelTag = "cp" + ($WoaVenvMinor -replace '\.', '')
        $WoaWheelAbi = Get-WoaAbiTag -PythonMinor $WoaVenvMinor -FreeThreaded ([bool]$script:WoaVenvFreeThreaded)
        $WoaWheelStable = -not $script:WoaVenvFreeThreaded
        $WoaWheelMinor = 0
        if ($WoaVenvMinor -match '^\d+\.(\d+)') { $WoaWheelMinor = [int]$Matches[1] }
        $WoaWheelNames = @{}
        foreach ($wheel in @(Get-ChildItem -LiteralPath $WoaWheelDir -Filter "*.whl" -ErrorAction SilentlyContinue)) {
            $parts = [System.IO.Path]::GetFileNameWithoutExtension($wheel.Name) -split '-'
            if ($parts.Count -lt 5) { continue }
            $platTags = $parts[-1] -split '\.'
            if (-not (($platTags -contains 'win_arm64') -or ($platTags -contains 'any'))) { continue }
            $abiTags = $parts[-2] -split '\.'
            $compatible = $false
            foreach ($pyTag in ($parts[-3] -split '\.')) {
                # ABI too: uv rejects a cp313t wheel on a GIL interpreter.
                if ($pyTag -eq $WoaWheelTag -and (
                        ($abiTags -contains $WoaWheelAbi) -or
                        ($WoaWheelStable -and ($abiTags -contains 'abi3')) -or
                        ($abiTags -contains 'none'))) {
                    $compatible = $true; break
                }
                if (($abiTags -contains 'none') -and ($pyTag -match '^py3(\d*)$')) {
                    if ([string]::IsNullOrEmpty($Matches[1]) -or ([int]$Matches[1] -le $WoaWheelMinor)) {
                        $compatible = $true; break
                    }
                }
                if ($WoaWheelStable -and ($abiTags -contains 'abi3') -and ($pyTag -match '^cp3(\d+)$')) {
                    if ([int]$Matches[1] -le $WoaWheelMinor) { $compatible = $true; break }
                }
            }
            if (-not $compatible) { continue }
            $_woaWheelKey = ($parts[0] -replace '[-_.]+', '-').ToLowerInvariant()
            if (-not $WoaWheelNames.ContainsKey($_woaWheelKey)) { $WoaWheelNames[$_woaWheelKey] = @() }
            $WoaWheelNames[$_woaWheelKey] += $parts[1]
        }
        # Still available for win_arm64, or the drop list below excludes it on ARM64 entirely.
        foreach ($_woaProvided in $script:WoaPyPIProvided.Keys) {
            if (-not $WoaWheelNames.ContainsKey($_woaProvided)) { $WoaWheelNames[$_woaProvided] = @() }
            $WoaWheelNames[$_woaProvided] += @($script:WoaPyPIProvided[$_woaProvided] | Where-Object { $_ })
        }
        # None of these has a win_arm64 wheel or a buildable sdist; httpx negotiates around brotli.
        $WoaDropCandidates = @(
            "hf-transfer", "hf_transfer", "xformers", "torchcodec",
            "brotli", "brotlicffi"
        )
        if (-not $script:WoaTorchAudio) { $WoaDropCandidates += "torchaudio" }
        # Floors the RELEASED metadata puts on a drop candidate: below one, keep the drop.
        $WoaDropFloors = @{ "xformers" = "0.0.22.post7" }
        $WoaOverrideLines = @(
            '# Generated by install.ps1 for Windows on ARM (win_arm64). See the WoA block there.'
        )
        $WoaReported = @{}
        foreach ($candidate in $WoaDropCandidates) {
            $key = ($candidate -replace '[-_.]+', '-').ToLowerInvariant()
            $_woaHosted = $false
            if ($WoaWheelNames.ContainsKey($key)) {
                $_woaFloor = $WoaDropFloors[$key]
                if (-not $_woaFloor) { $_woaHosted = $true }
                else {
                    foreach ($_woaHostedVer in $WoaWheelNames[$key]) {
                        if (Test-WoaVersionAtLeast -Version $_woaHostedVer -Floor $_woaFloor) { $_woaHosted = $true; break }
                    }
                    if (-not $_woaHosted -and -not $WoaReported.ContainsKey($key)) {
                        $WoaReported[$key] = $true
                        substep "windows on arm: the wheelhouse $key is below the required $_woaFloor -- keeping the drop." "Yellow"
                    }
                }
            }
            if ($_woaHosted) {
                if (-not $WoaReported.ContainsKey($key)) {
                    $WoaReported[$key] = $true
                    substep "windows on arm: keeping $key (the wheelhouse provides a win_arm64 wheel)."
                }
                continue
            }
            $WoaOverrideLines += "$candidate ; platform_machine == `"AMD64`""
        }
        # Released unsloth metadata caps torch below the only win_arm64 CUDA build (a 2.15 nightly).
        $WoaOverrideLines += 'torch>=2.4'
        $WoaOverrideLines += 'torchvision>=0.19'
        # Without this uv takes the newest pyarrow on PyPI, an sdist here. For the PyPI route too.
        if ($script:WoaPyarrowWheelName -and $script:WoaPyarrowWheelName -match '^pyarrow-([^-]+)-') {
            $WoaOverrideLines += "pyarrow==$($Matches[1])"
        }
        # uv COMBINES override files and two naming one package is an error: no overwrite, no append.
        $_woaOwnNames = @{}
        foreach ($_woaLine in $WoaOverrideLines) {
            if ($_woaLine -match '^\s*(#|$)') { continue }
            $_woaName = (($_woaLine -split '[\s<>=!~;@\[]', 2)[0]).Trim()
            if ($_woaName) { $_woaOwnNames[($_woaName -replace '[-_.]+', '-').ToLowerInvariant()] = $true }
        }
        # Folding is the LAST resort: copying a line moves the base its relative paths resolve on.
        $_woaKeepFiles = @()
        $_woaSessionLines = @()
        if ($env:UV_OVERRIDE) {
            foreach ($_woaOvFile in ($env:UV_OVERRIDE -split '\s+' | Where-Object { $_ })) {
                if (-not (Test-Path -LiteralPath $_woaOvFile -PathType Leaf)) { continue }
                try { $_woaOvFull = Convert-Path -LiteralPath $_woaOvFile } catch { continue }
                $_woaOvEntries = @(Get-WoaRequirementEntries -Path $_woaOvFull)
                $_woaOvConflicts = $false
                foreach ($_woaOvEntry in $_woaOvEntries) {
                    $_woaOvName = ((($_woaOvEntry.Line -split '[\s<>=!~;@\[]', 2)[0]).Trim() -replace '[-_.]+', '-').ToLowerInvariant()
                    if ($_woaOvName -and $_woaOwnNames.ContainsKey($_woaOvName)) { $_woaOvConflicts = $true; break }
                }
                if (-not $_woaOvConflicts) {
                    $_woaKeepFiles += $_woaOvFull
                    continue
                }
                foreach ($_woaOvEntry in $_woaOvEntries) {
                    $_woaOvName = ((($_woaOvEntry.Line -split '[\s<>=!~;@\[]', 2)[0]).Trim() -replace '[-_.]+', '-').ToLowerInvariant()
                    if ($_woaOvName -and $_woaOwnNames.ContainsKey($_woaOvName)) { continue }
                    $_woaSessionLines += (Resolve-WoaOverrideLine -Line $_woaOvEntry.Line -BaseDir $_woaOvEntry.BaseDir)
                }
            }
        }
        # No BOM: uv reads these as plain requirements files.
        [System.IO.File]::WriteAllLines($WoaOverrides, [string[]]$WoaOverrideLines, (New-Object System.Text.UTF8Encoding($false)))
        $_woaOverrideValue = @(Get-UvSafePath $WoaOverrides)
        foreach ($_woaKeepFile in $_woaKeepFiles) { $_woaOverrideValue += (Get-UvSafePath $_woaKeepFile) }
        # Folded caller lines go to a per-run file, never the persistent one setup.ps1 restores: their credentials and policy must not outlive this run.
        $_woaSessionFile = Join-Path (Split-Path -Parent $WoaOverrides) "overrides.session.txt"
        Remove-Item -LiteralPath $_woaSessionFile -Force -ErrorAction SilentlyContinue
        if ($_woaSessionLines.Count -gt 0) {
            [System.IO.File]::WriteAllLines($_woaSessionFile, [string[]]$_woaSessionLines, (New-Object System.Text.UTF8Encoding($false)))
            $script:WoaSessionOverrides = $_woaSessionFile
            $_woaOverrideValue += (Get-UvSafePath $_woaSessionFile)
        }
        # Under `irm | iex` these are the caller's own session variables: snapshotted once here and put back when the installer returns.
        if ($null -eq $script:WoaResolverEnvSaved) {
            $script:WoaResolverEnvSaved = @{
                UV_OVERRIDE = $env:UV_OVERRIDE; UV_FIND_LINKS = $env:UV_FIND_LINKS; PIP_FIND_LINKS = $env:PIP_FIND_LINKS
            }
        }
        $env:UV_OVERRIDE = ($_woaOverrideValue -join " ")
        # Additive, ours first to win a tie. UV_FIND_LINKS is comma-separated, PIP_FIND_LINKS not.
        $_woaCallerUvLinks = $env:UV_FIND_LINKS
        $_woaCallerPipLinks = $env:PIP_FIND_LINKS
        $env:UV_FIND_LINKS = if ($_woaCallerUvLinks) { "$WoaWheelDir,$_woaCallerUvLinks" } else { $WoaWheelDir }
        $_woaSafeWheelDir = Get-UvSafePath $WoaWheelDir
        $env:PIP_FIND_LINKS = if ($_woaCallerPipLinks) { "$_woaSafeWheelDir $_woaCallerPipLinks" } else { $_woaSafeWheelDir }
        # install_python_stack.py falls back to pip when `uv` is missing, and pip has no UV_OVERRIDE.
        $uvDir = $null
        try { $uvDir = [System.IO.Path]::GetDirectoryName($script:UvExe) } catch {}
        if ($uvDir -and (Test-Path -LiteralPath $uvDir -PathType Container) -and ($env:PATH -notlike "*$uvDir*")) {
            $env:PATH = "$uvDir;$env:PATH"
        }
        step "windows on arm" "native ARM64 CUDA stack selected"
        substep "overrides: $WoaOverrides"
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
    # __COMPAT_LAYER=RunAsInvoker stops amd-smi auto-elevating; WMI fallback still resolves
    # the arch.
    function Invoke-AmdSmiNoElevate {
        param(
            [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
            [Parameter(Position = 1)][string[]]$SmiArgs = @(),
            [int]$TimeoutSec = 30
        )
        # RunAsInvoker blocks auto-elevation; the timeout bounds a flaky amd-smi (30s, as amd.py).
        $prevCompat = [Environment]::GetEnvironmentVariable('__COMPAT_LAYER', 'Process')
        $env:__COMPAT_LAYER = 'RunAsInvoker'
        try {
            # [Process]::Start, not Start-Process: PS 5.1 leaves .ExitCode $null and detection dies.
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

    # ── A driverless nvidia-smi exits 0 listing no GPU, so require a "GPU <n>:" row. ──
    # 124 means Invoke-NvidiaSmiBounded killed the probe; recorded so the banner skips a
    # second query.
    $script:NvidiaSmiWedged = $false
    # Reset per run: streamed execution shares the caller's script scope.
    $script:NvidiaPresenceOnly = $false
    $script:NvidiaPresenceCudaFloor = $null
    $script:NvidiaPresenceDriverRelease = $null
    $script:VideoControllerScanResult = $null

    function Test-NvidiaSmiHasGpu {
        param([Parameter(Mandatory = $true)][string]$Exe)
        $out = Invoke-NvidiaSmiBounded $Exe @('-L')
        # Assigned, not OR-ed: the fallback loop tries several paths, and what matters is
        # whether the binary it settled on answered, not whether an earlier one hung.
        $script:NvidiaSmiWedged = ($LASTEXITCODE -eq 124)
        return ($LASTEXITCODE -eq 0 -and $out -match '(?m)^GPU\s+\d+:')
    }

    # Deliberately NOT in the shared region: the two files find Python differently.
    function Get-NvidiaProbePythonExe {
        if ("$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0") { return "" }
        # This run's venv interpreter comes FIRST: Get-StudioEarlyPython may have cached a miss
        # before the venv existed.
        if (-not [string]::IsNullOrWhiteSpace($VenvPython) -and
            (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
            return $VenvPython
        }
        try { return "$(Get-StudioEarlyPython)" } catch { return "" }
    }

    # ── BEGIN SHARED WITH studio/setup.ps1 (Get-NvidiaLibraryInventory) ──
    # Full paths only: a bare name searches PATH, where ZLUDA's nvcuda.dll and nvml.dll pass for NVIDIA.
    function Get-NvidiaNvmlLibraryPath {
        $dirs = @()
        if ($env:SystemRoot) { $dirs += (Join-Path $env:SystemRoot "System32") }
        if ($env:ProgramFiles) { $dirs += (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI") }
        foreach ($dir in $dirs) {
            $candidate = Join-Path $dir "nvml.dll"
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
        return (Join-Path (Get-NvidiaSystem32Dir) "nvml.dll")
    }

    function Get-NvidiaSystem32Dir {
        $root = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
        return (Join-Path $root "System32")
    }


    # The same inventory with nothing emitted: CPython's ctypes makes the identical NVML and CUDA
    # driver calls, and the interop leaves the scanned surface rather than moving within it.
    # available or the probe itself says nothing.
    # Get-NvidiaProbePythonExe is deliberately per-file. The installer has its early read-only
    # interpreter ladder; setup.ps1 has the venv a previous run already built.
    function Read-NvidiaLibraryRawViaPython {
        param([int]$TimeoutMs = 10000, [switch]$SkipNvml)
        # Set only when the child is killed at the deadline, so a caller can tell a hung driver from
        # an empty answer. Reset first: every early return below is an answer, not a timeout.
        $script:NvidiaPythonProbeTimedOut = $false
        if ("$($env:UNSLOTH_NVIDIA_PYTHON_PROBE)".Trim() -eq "0") { return "" }
        $exe = ""
        try { $exe = "$(Get-NvidiaProbePythonExe)" } catch { return "" }
        if (-not $exe) { return "" }
        $windows = ($env:OS -eq "Windows_NT")
        $nvmlHint = if ($windows) { Get-NvidiaNvmlLibraryPath } else { "libnvidia-ml.so.1" }
        $cudaHint = if ($windows) { Join-Path (Get-NvidiaSystem32Dir) "nvcuda.dll" } else { "libcuda.so.1" }
        # Kept byte-identical with studio/nvidia_probe.py's readers by
        # tests/studio/test_nvidia_python_probe_parity.ps1. Column 0 on purpose: this is Python.
        $probeSource = @'
import ctypes, os, sys


def _names(kind, hint):
    if os.name == "nt":
        # The driver's full path only: a bare name also finds a CUDA stand-in such as ZLUDA
        # beside the interpreter, and that is not an NVIDIA GPU (#11736).
        return [hint] if hint and os.path.isabs(hint) else []
    if kind == "nvml":
        return ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    return ["libcuda.so.1", "libcuda.so"]


def _load(kind, hint):
    for name in _names(kind, hint):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except Exception:
            continue
    return None


def _unpack(packed):
    return "%d;%d" % (packed // 1000, (packed % 1000) // 10)


def read_nvml(hint):
    lib = _load("nvml", hint)
    if lib is None:
        return ""
    try:
        if lib.nvmlInit_v2() != 0:
            return ""
    except Exception:
        return ""
    try:
        count = ctypes.c_uint(0)
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
            return ""
        packed = ctypes.c_int(0)
        if lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(packed)) != 0 or packed.value < 1000:
            return ""
        # ctypes defaults every return and every pointer argument to a C int, which truncates a
        # 64-bit nvmlDevice_t handle. Declare both before the first call, not after.
        handle_of = lib.nvmlDeviceGetHandleByIndex_v2
        handle_of.restype = ctypes.c_int
        handle_of.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        cap_of = lib.nvmlDeviceGetCudaComputeCapability
        cap_of.restype = ctypes.c_int
        cap_of.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        caps = []
        for index in range(count.value):
            device = ctypes.c_void_p()
            # One unreadable GPU voids the source: a partial list misleads the pre-Turing cap.
            if handle_of(index, ctypes.byref(device)) != 0:
                return ""
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            if cap_of(device, ctypes.byref(major), ctypes.byref(minor)) != 0:
                return ""
            caps.append("%d.%d" % (major.value, minor.value))
        return "nvml;%s;%s" % (_unpack(packed.value), ",".join(caps))
    finally:
        try:
            lib.nvmlShutdown()
        except Exception:
            pass


def read_cuda(hint):
    lib = _load("cuda", hint)
    if lib is None:
        return ""
    # The driver API honours CUDA_VISIBLE_DEVICES; the inventory must be the physical one, so a
    # hidden pre-Turing card still caps the family. cuInit reads the mask once.
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        init = lib.cuInit(0)
    except Exception:
        return ""
    finally:
        if saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    if init != 0:
        return ""
    count = ctypes.c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return ""
    packed = ctypes.c_int(0)
    if lib.cuDriverGetVersion(ctypes.byref(packed)) != 0 or packed.value < 1000:
        return ""
    caps = []
    for index in range(count.value):
        device = ctypes.c_int(0)
        if lib.cuDeviceGet(ctypes.byref(device), index) != 0:
            return ""
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76.
        if lib.cuDeviceGetAttribute(ctypes.byref(major), 75, device) != 0:
            return ""
        if lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, device) != 0:
            return ""
        caps.append("%d.%d" % (major.value, minor.value))
    return "cuda;%s;%s" % (_unpack(packed.value), ",".join(caps))


def main():
    nvml_hint = os.environ.get("UNSLOTH_NVML_HINT", "")
    cuda_hint = os.environ.get("UNSLOTH_CUDA_HINT", "")
    answer = ""
    # Set only for the CUDA-only retry that follows a child NVML held past its deadline.
    if os.environ.get("UNSLOTH_NVIDIA_PROBE_SKIP_NVML", "") != "1":
        try:
            answer = read_nvml(nvml_hint)
        except Exception:
            answer = ""
    if not answer:
        try:
            answer = read_cuda(cuda_hint)
        except Exception:
            answer = ""
    sys.stdout.write(answer)


main()
'@
        # Cmdlets only: Constrained Language Mode refuses New-Object ProcessStartInfo and
        # [Process]::Start, and this launcher has to work there.
        $probeDir = New-StudioChildScriptDirectory
        $inline = (-not $probeDir)
        if ($inline) {
            # Elevated, a shared root lets a standard user plant or swap the redirect files: decline.
            try { if (Test-StudioChildScriptDirectoryElevated) { return "" } } catch { return "" }
            # The first root that takes a file: a TEMP that declined the directory may refuse this too.
            $stem = $null
            foreach ($root in @($env:TEMP, $env:TMP, $env:LOCALAPPDATA, $env:TMPDIR, "/tmp")) {
                if (-not $root) { continue }
                try {
                    $candidateStem = Join-Path $root ("unsloth-nvprobe-" + [guid]::NewGuid().ToString("N")) -ErrorAction Stop
                    $null = New-Item -ItemType File -Path "$candidateStem.out" -ErrorAction Stop; $stem = $candidateStem; break
                } catch {}
            }
            if (-not $stem) { return "" }
        } else {
            $stem = Join-Path $probeDir "nvprobe"
        }
        $scriptFile = "$stem.py"
        $outFile = "$stem.out"
        $errFile = "$stem.err"
        $raw = ""
        try {
            if (-not $inline) { Set-Content -LiteralPath $scriptFile -Value $probeSource -Encoding UTF8 -ErrorAction Stop }
            # Whole seconds, rounded up, without [math]::Ceiling: CLM blocks it. PowerShell's / is
            # floating point and [int] rounds to nearest, so 10000ms must not become 11s.
            $seconds = ($TimeoutMs - ($TimeoutMs % 1000)) / 1000
            if (($TimeoutMs % 1000) -ne 0) { $seconds = $seconds + 1 }
            $seconds = [int]$seconds
            if ($seconds -lt 1) { $seconds = 1 }
            # Nothing with a space in it reaches the command line. Windows PowerShell 5.1 appends
            # each native argument verbatim, so a script under "C:\Users\First Last\AppData\Local\
            # Temp" or a hint under "C:\Program Files\NVIDIA Corporation\NVSMI" would split on its
            # spaces and the child would run something else. The script arrives on stdin and the two
            $savedNvml = $env:UNSLOTH_NVML_HINT
            $savedCuda = $env:UNSLOTH_CUDA_HINT
            $savedSkip = $env:UNSLOTH_NVIDIA_PROBE_SKIP_NVML
            $savedSource = $env:UNSLOTH_NVIDIA_PROBE_SOURCE
            $env:UNSLOTH_NVML_HINT = $nvmlHint
            $env:UNSLOTH_CUDA_HINT = $cudaHint
            # The switch reaches this child only. An inherited value must not make a first child skip NVML.
            if ($SkipNvml) { $env:UNSLOTH_NVIDIA_PROBE_SKIP_NVML = "1" }
            else { Remove-Item Env:UNSLOTH_NVIDIA_PROBE_SKIP_NVML -ErrorAction SilentlyContinue }
            try {
                if ($inline) {
                    $env:UNSLOTH_NVIDIA_PROBE_SOURCE = $probeSource
                    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
                        -ArgumentList @("-I", "-S", "-B", "-c", "exec(__import__('os').environ['UNSLOTH_NVIDIA_PROBE_SOURCE'])") `
                        -RedirectStandardOutput $outFile -RedirectStandardError $errFile -ErrorAction Stop
                } else {
                    $proc = Start-Process -FilePath $exe -ArgumentList @("-I", "-S", "-B", "-") -NoNewWindow -PassThru `
                        -RedirectStandardInput $scriptFile -RedirectStandardOutput $outFile -RedirectStandardError $errFile -ErrorAction Stop
                }
            } finally {
                if ($null -eq $savedSource) { Remove-Item Env:UNSLOTH_NVIDIA_PROBE_SOURCE -ErrorAction SilentlyContinue }
                else { $env:UNSLOTH_NVIDIA_PROBE_SOURCE = $savedSource }
                if ($null -eq $savedNvml) { Remove-Item Env:UNSLOTH_NVML_HINT -ErrorAction SilentlyContinue }
                else { $env:UNSLOTH_NVML_HINT = $savedNvml }
                if ($null -eq $savedCuda) { Remove-Item Env:UNSLOTH_CUDA_HINT -ErrorAction SilentlyContinue }
                else { $env:UNSLOTH_CUDA_HINT = $savedCuda }
                if ($null -eq $savedSkip) { Remove-Item Env:UNSLOTH_NVIDIA_PROBE_SKIP_NVML -ErrorAction SilentlyContinue }
                else { $env:UNSLOTH_NVIDIA_PROBE_SKIP_NVML = $savedSkip }
            }
            if (-not $proc) { return "" }
            # -InputObject and an error variable, never $proc.Id or $proc.HasExited. Constrained
            # Language Mode permits property reads only on its allowed type list and
            # System.Diagnostics.Process is not on it, so reading either one throws on exactly the
            # hosts this rung exists for. Handing the object to a cmdlet keeps the access inside
            # compiled code, where the language mode does not reach.
            $waitError = $null
            Wait-Process -InputObject $proc -Timeout $seconds -ErrorAction SilentlyContinue -ErrorVariable waitError
            if ($waitError) {
                $script:NvidiaPythonProbeTimedOut = $true
                try { Stop-Process -InputObject $proc -Force -ErrorAction SilentlyContinue } catch { }
                # Let the killed child release its redirected files before the finally deletes them.
                Wait-Process -InputObject $proc -Timeout 2 -ErrorAction SilentlyContinue
                return ""
            }
            $raw = "$(Get-Content -LiteralPath $outFile -Raw -ErrorAction SilentlyContinue)"
        } catch { return "" }
        finally {
            # The directory, not just the three files: it is ours, nothing else may be in it, and
            # leaving an empty one behind per probe would litter %TEMP% on every run.
            if ($probeDir) { Remove-Item -LiteralPath $probeDir -Recurse -Force -ErrorAction SilentlyContinue }
            foreach ($stale in @($scriptFile, $outFile, $errFile)) {
                Remove-Item -LiteralPath $stale -Force -ErrorAction SilentlyContinue
            }
        }
        return "$raw".Trim()
    }

    # "source;cudaMajor;cudaMinor;cap,cap" from NVML, else the CUDA driver API; "" when neither
    function Read-NvidiaLibraryRaw {
        param([int]$TimeoutMs = 30000)
        # One deadline for both children, as the two in-process readers had: the first gets a whole
        # bound, and only a hung NVML earns a CUDA-only child with what is left, less the 2 s kept back
        # to reap the child it may have to kill. Compared by hand: CLM refuses [math].
        $deadline = (Get-Date).AddMilliseconds($TimeoutMs * 2)
        $raw = ""
        try { $raw = Read-NvidiaLibraryRawViaPython -TimeoutMs $TimeoutMs } catch { return "" }
        if (-not $script:NvidiaPythonProbeTimedOut) { return $raw }
        $remainingMs = [int]($deadline - (Get-Date)).TotalMilliseconds
        if ($remainingMs -lt 3000) { return "" }
        $childMs = $remainingMs - 2000; if ($childMs -gt $TimeoutMs) { $childMs = $TimeoutMs }
        try { return (Read-NvidiaLibraryRawViaPython -TimeoutMs $childMs -SkipNvml) } catch { return "" }
    }

    # NVIDIA inventory from the driver's own libraries (NVML, then the CUDA driver API), for a
    # host whose nvidia-smi is absent, stale or hangs (#9255). Twin of studio/nvidia_probe.py.
    # Cached. $null, or @{ Source; CudaMajor; CudaMinor; ComputeCaps ("8.9" strings); Count }.
    function Get-NvidiaLibraryInventory {
        param([int]$TimeoutSec = 30)
        if ($script:NvidiaLibraryInventoryProbed) { return $script:NvidiaLibraryInventory }
        $script:NvidiaLibraryInventoryProbed = $true
        $script:NvidiaLibraryInventory = $null
        if ("$($env:UNSLOTH_NVIDIA_LIBRARY_PROBE)".Trim() -eq "0") { return $null }
        try { $raw = Read-NvidiaLibraryRaw -TimeoutMs ($TimeoutSec * 1000) } catch { return $null }
        # The Python probe is the only library reader, so its opt-outs leave nvidia-smi as the only source.
        if (-not "$raw" -and ("$($env:UNSLOTH_NVIDIA_PYTHON_PROBE)".Trim() -eq "0" -or "$($env:UNSLOTH_EARLY_PYTHON_PROBE)".Trim() -eq "0")) {
            Write-StudioLine "   NVIDIA driver libraries not read (Python probe disabled): a GPU without a working nvidia-smi is not detected. Set UNSLOTH_TORCH_INDEX_FAMILY=cu128 (or your CUDA wheel) to choose one." -ForegroundColor Yellow
        }
        $parts = "$raw".Split(";")
        if ($parts.Count -ne 4 -or -not $parts[3] -or [int]$parts[1] -lt 1) { return $null }
        $caps = @($parts[3].Split(","))
        if (@($caps | Where-Object { $_ -notmatch '^\d+\.\d+$' }).Count -gt 0) { return $null }
        $script:NvidiaLibraryInventory = @{
            Source      = $parts[0]
            CudaMajor   = [int]$parts[1]
            CudaMinor   = [int]$parts[2]
            ComputeCaps = $caps
            Count       = $caps.Count
        }
        return $script:NvidiaLibraryInventory
    }
    # ── END SHARED WITH studio/setup.ps1 (Get-NvidiaLibraryInventory) ──
    # Under `irm | iex` the script scope is the caller's session: probe once per invocation.
    $script:NvidiaLibraryInventoryProbed = $false
    $script:NvidiaLibraryInventory = $null
    # ── Detect GPU (robust: PATH + hardcoded fallback paths, mirrors setup.ps1) ──
    $HasNvidiaSmi = $false
    $NvidiaSmiExe = $null
    $NvidiaGpuName = $null
    $NvidiaSmArch = $null
    $NvidiaDriverVersion = $null
    try {
        $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvSmiCmd -and (Test-NvidiaSmiHasGpu $nvSmiCmd.Source)) {
            $HasNvidiaSmi = $true; $NvidiaSmiExe = $nvSmiCmd.Source
        }
    } catch {}
    if (-not $HasNvidiaSmi) {
        # Prefer a candidate that also prints a parseable CUDA banner, or a stale binary could
        # pick cu126 on a cu128 host. One time budget across the loop (Get-Date: CLM refuses
        # Stopwatch). Candidates are enumerated before the clock starts.
        $smiCandidates = @(Get-NvidiaSmiCandidatePaths)
        # Two budgets: the soft one only once a usable answer exists; only the hard one may end
        # with nothing, since that means CPU-only PyTorch on a GPU host.
        $probeDeadline = (Get-Date).AddSeconds(30)
        $probeHardDeadline = (Get-Date).AddSeconds(60)
        $firstListing = $null
        # Captured with the path: $script:NvidiaSmiWedged describes the LAST binary probed.
        $firstListingWedged = $false
        foreach ($p in $smiCandidates) {
            # Only give up early when there is already something to fall back on.
            if ($null -ne $firstListing -and (Get-Date) -gt $probeDeadline) { break }
            # With nothing found, keep going to the hard bound. Reporting no GPU is the expensive
            # answer, so it has to be the one that costs the most before it is reached.
            if ((Get-Date) -gt $probeHardDeadline) { break }
            try {
                if (-not (Test-NvidiaSmiHasGpu $p)) { continue }
                if (-not $firstListing) {
                    $firstListing = $p
                    $firstListingWedged = $script:NvidiaSmiWedged
                }
                # Rechecked here so a late candidate cannot nearly double the budget.
                if ((Get-Date) -gt $probeDeadline) { break }
                $banner = Invoke-NvidiaSmiBounded $p
                if ($banner -match 'CUDA(?: UMD)? Version:\s+\d+\.\d+') {
                    $HasNvidiaSmi = $true; $NvidiaSmiExe = $p; break
                }
            } catch {}
        }
        if (-not $HasNvidiaSmi -and $firstListing) {
            $HasNvidiaSmi = $true; $NvidiaSmiExe = $firstListing
            # Restore the wedged flag for the selected binary, not the last one probed.
            $script:NvidiaSmiWedged = $firstListingWedged
        }
    }
    # Must stay above its callers (no hoisting). WMI can hang and the timeout is not enforced
    # for local COM, so query out of process with a wall-clock kill.
    function Invoke-BoundedVideoControllerScan {
        param([int]$TimeoutSec = 15)
        # Cached even when Ok = $false: that is the hang case, and a retry would hang again.
        if ($null -ne $script:VideoControllerScanResult) { return $script:VideoControllerScanResult }
        $result = [pscustomobject]@{ Ok = $false; Names = @(); Adapters = @() }
        $job = $null
        try {
            $job = Start-Job -ScriptBlock {
                Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Select-Object -Property Name, PNPDeviceID, ConfigManagerErrorCode, DriverVersion
            }
            if (Wait-Job -Job $job -Timeout $TimeoutSec) {
                $rows = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
                $result.Adapters = @($rows | Where-Object { $_ })
                $names = @($rows | ForEach-Object { $_.Name })
                $result.Names = @($names | Where-Object { $_ })
                $result.Ok = ($result.Names.Count -gt 0)
            } else {
                Stop-Job -Job $job -ErrorAction SilentlyContinue
            }
        } catch {
        } finally {
            # -ErrorAction does not stop a terminating error, which would abort the run.
            if ($job) { try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch {} }
        }
        $script:VideoControllerScanResult = $result
        return $result
    }

    # Mirrors windows_intel_gpu_in_registry(), but as a fallback: stale keys could install XPU
    # torch without an Arc.
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

    # Returns the entry, not a bool: the version ladder needs its DriverVersion.
    function Get-NvidiaRegistryAdapter {
        $classKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try {
            $subs = @(Get-ChildItem -LiteralPath $classKey -ErrorAction SilentlyContinue)
        } catch { return $null }
        # These keys outlive hardware, so only a UNANIMOUS release is used; disagreement reads as unknown.
        $present = $false
        $versions = @()
        foreach ($sub in $subs) {
            try {
                if ("$($sub.PSChildName)" -notmatch '^\d+$') { continue }
                $props = Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue
                if (-not $props) { continue }
                if ("$($props.MatchingDeviceId)" -notmatch '(?i)ven_10de') { continue }
                $present = $true
                $version = "$($props.DriverVersion)"
                if ([string]::IsNullOrWhiteSpace($version)) { continue }
                $release = Get-NvidiaDriverRelease -DriverVersion $version
                if ($null -eq $release) { continue }
                if ($versions -notcontains $release) { $versions += $release }
            } catch {}
        }
        if (-not $present) { return $null }
        if ($versions.Count -ne 1) { return [pscustomobject]@{ Release = $null } }
        return [pscustomobject]@{ Release = $versions[0] }
    }

    # True when no healthy NVIDIA adapter reports an NVIDIA-shaped driver version (e.g. Microsoft Basic
    # Display): no CUDA driver, so the host must keep CPU wheels and AMD / Intel detection.
    function Test-NvidiaAdapterWithoutNvidiaDriver {
        param($Scan = $null)
        if ($null -eq $Scan) { $Scan = Invoke-BoundedVideoControllerScan }
        $foreign = $false
        foreach ($adapter in @($Scan.Adapters)) {
            if ("$($adapter.PNPDeviceID)" -notmatch '(?i)ven_10de') { continue }
            if ($null -eq $adapter.ConfigManagerErrorCode) { continue }
            if ([int]$adapter.ConfigManagerErrorCode -ne 0) { continue }
            $version = "$($adapter.DriverVersion)"
            if ([string]::IsNullOrWhiteSpace($version)) { $foreign = $true; continue }
            if ($null -ne (Get-NvidiaDriverRelease -DriverVersion $version)) { return $false }
            $foreign = $true
        }
        return $foreign
    }

    # VEN_10DE and ConfigManagerErrorCode 0 only, never names: a false yes sets $HasNvidiaSmi and
    # switches off AMD and Intel detection.
 
    function Test-NvidiaAdapterPresent {
        param($Scan = $null)
        if ($null -eq $Scan) { $Scan = Invoke-BoundedVideoControllerScan }
        foreach ($adapter in @($Scan.Adapters)) {
            if ("$($adapter.PNPDeviceID)" -notmatch '(?i)ven_10de') { continue }
            # $null is not evidence of health.
            if ($null -eq $adapter.ConfigManagerErrorCode) { continue }
            if ([int]$adapter.ConfigManagerErrorCode -ne 0) { continue }
            return $true
        }
        # No registry fallback: stale class keys would promote a GPU that is gone.
        return $false
    }

    # Must match the Intel route's pattern exactly (test_nvidia_adapter_presence.ps1); a
    # function, since an unset $null pattern matches everything.
    function Get-XpuCapableNameRegex { return "(?i)Intel.*(Arc|Data Center GPU)" }

    function Test-IntelXpuRuntimeProven {
        param($Scan = $null, [string]$PythonExe = "")
        # The Intel route also serves non-Arc Intel GPUs once the existing torch proves XPU works, with no
        # WMI precondition, so this asks the same thing. Any failure reads as not proven.
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $false }
        try {
            $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.xpu.is_available())'
            return [bool]($probe.Ok -and $probe.Output -match '(?m)^\s*True\s*$')
        } catch {
            return $false
        }
    }

    function Test-OtherVendorAdapterPresent {
        param($Scan = $null)
        if ($null -eq $Scan) { $Scan = Invoke-BoundedVideoControllerScan }
        # AMD evidence the AMD route finds without WMI (override, the installer's forwarded arch, HIP SDK,
        # opted-in amd-smi) vetoes too.
        if ("$env:UNSLOTH_ROCM_GFX_ARCH".Trim() -or "$env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF".Trim()) { return $true }
        try {
            if (Get-Command hipinfo -CommandType Application -ErrorAction SilentlyContinue) { return $true }
            foreach ($hipEnv in @($env:HIP_PATH, $env:HIP_PATH_57, $env:ROCM_PATH)) {
                if ($hipEnv -and (Test-Path -LiteralPath (Join-Path $hipEnv "bin\hipinfo.exe"))) { return $true }
            }
            if ($env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(1|true|yes|on)$' -and
                (Get-Command amd-smi -ErrorAction SilentlyContinue)) { return $true }
        } catch {}
        # Same classification as the Intel route, which ignores PNP ID and status.
        if ($Scan.Ok) {
            $scanNames = @($Scan.Names | Where-Object { $_ } | ForEach-Object { "$_" })
            if (@($scanNames | Where-Object { $_ -match (Get-XpuCapableNameRegex) }).Count -gt 0) { return $true }
            try {
                foreach ($regName in @(Get-IntelRegistryAdapterNames)) {
                    if ("$regName" -notmatch (Get-XpuCapableNameRegex)) { continue }
                    foreach ($wmiName in $scanNames) {
                        if ("$regName".Contains($wmiName)) { return $true }
                    }
                }
            } catch {}
        }
        foreach ($adapter in @($Scan.Adapters)) {
            # AMD vetoes on the AMD route's own test (name or 1002, any status: it keeps parked and
            # PNP-less cards). Intel was answered above; UHD/Iris do not block.
            if ("$($adapter.Name)" -match "AMD|Radeon" -or "$($adapter.PNPDeviceID)" -match '(?i)ven_1002') { return $true }
        }
        # Registry fallback here is deliberate: staleness only DECLINES a promotion. Without it, broken WMI
        # hid the Arc on a hybrid host and cost it its XPU wheels.
        if ($Scan.Ok) { return $false }
        # Same normalized names as the Intel route (it prefixes "Intel" to a VEN_8086 DriverDesc).
        try {
            if (@(Get-IntelRegistryAdapterNames | Where-Object { "$_" -match (Get-XpuCapableNameRegex) }).Count -gt 0) { return $true }
        } catch {}
        $classKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try {
            $subs = @(Get-ChildItem -LiteralPath $classKey -ErrorAction SilentlyContinue)
        } catch { return $false }
        foreach ($sub in $subs) {
            try {
                if ("$($sub.PSChildName)" -notmatch '^\d+$') { continue }
                $props = Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue
                if (-not $props) { continue }
                $match = "$($props.MatchingDeviceId)"
                if ($match -match '(?i)ven_1002') { return $true }
            } catch {}
        }
        return $false
    }

    # Ported from studio/nvidia_probe.py's cuda_version_for_driver; the parity test generates the table.
    # Kept apart from the floor so "no version" differs from "too old for every row".
    function Get-NvidiaDriverRelease {
        param([string]$DriverVersion)
        if ([string]::IsNullOrWhiteSpace($DriverVersion)) { return $null }
        # NVIDIA's scheme only (32.0.15.6094 is 560.94). Basic Display versions like 10.0.19041.3636 must
        # not read as releases, or a working CUDA venv gets replaced with CPU wheels.
        $m = [regex]::Match($DriverVersion, '^\s*\d+\.\d+\.(1\d)\.(\d{4})\s*$')
        if ($m.Success) {
            $digits = ($m.Groups[1].Value + $m.Groups[2].Value)
            if ($digits.Length -ge 5) {
                $tail = $digits.Substring($digits.Length - 5)
                return [int]$tail.Substring(0, 3)
            }
            return $null
        }
        # At least three digits: "1.2.3" is malformed, and a low release now selects CPU wheels.
        $m2 = [regex]::Match($DriverVersion, '^\s*(\d+)\.')
        if ($m2.Success) {
            $parsed = [int]$m2.Groups[1].Value
            if ($parsed -ge 100) { return $parsed }
        }
        return $null
    }

    function Get-NvidiaCudaFloorForRelease {
        param($Release)
        if ($null -eq $Release) { return $null }
        # Same order and values as _DRIVER_MAJOR_CUDA.
        foreach ($row in @(
            @(580, 13, 0), @(570, 12, 8), @(560, 12, 6), @(555, 12, 5), @(550, 12, 4),
            @(545, 12, 3), @(535, 12, 2), @(525, 12, 0), @(450, 11, 0)
        )) {
            if ([int]$Release -ge $row[0]) { return @($row[1], $row[2]) }
        }
        return $null
    }

    function Get-NvidiaDriverCudaFloor {
        param([string]$DriverVersion)
        return (Get-NvidiaCudaFloorForRelease -Release (Get-NvidiaDriverRelease -DriverVersion $DriverVersion))
    }

    function Get-NvidiaAdapterDriverRelease {
        param($Scan = $null)
        if ($null -eq $Scan) { $Scan = Invoke-BoundedVideoControllerScan }
        foreach ($adapter in @($Scan.Adapters)) {
            if ("$($adapter.PNPDeviceID)" -notmatch '(?i)ven_10de') { continue }
            if ($null -eq $adapter.ConfigManagerErrorCode) { continue }
            if ([int]$adapter.ConfigManagerErrorCode -ne 0) { continue }
            $release = Get-NvidiaDriverRelease -DriverVersion "$($adapter.DriverVersion)"
            if ($null -ne $release) { return $release }
        }
        if (-not $Scan.Ok) {
            $registryAdapter = Get-NvidiaRegistryAdapter
            if ($registryAdapter) { return $registryAdapter.Release }
        }
        return $null
    }

    function Get-NvidiaAdapterCudaFloor {
        param($Scan = $null)
        if ($null -eq $Scan) { $Scan = Invoke-BoundedVideoControllerScan }
        foreach ($adapter in @($Scan.Adapters)) {
            if ("$($adapter.PNPDeviceID)" -notmatch '(?i)ven_10de') { continue }
            if ($null -eq $adapter.ConfigManagerErrorCode) { continue }
            if ([int]$adapter.ConfigManagerErrorCode -ne 0) { continue }
            $floor = Get-NvidiaDriverCudaFloor -DriverVersion "$($adapter.DriverVersion)"
            if ($floor) { return $floor }
        }
        if (-not $Scan.Ok) {
            $registryAdapter = Get-NvidiaRegistryAdapter
            if ($registryAdapter) { return (Get-NvidiaCudaFloorForRelease -Release $registryAdapter.Release) }
        }
        return $null
    }

    if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory)) {
        # A driver too old for any CUDA wheel (below 11) is no GPU this route can serve: the Intel and
        # AMD routes still get their turn instead of the install falling to CPU.
        if ((Get-NvidiaLibraryInventory).CudaMajor -ge 11) {
            # Same promotion as setup.ps1: the gates below read $HasNvidiaSmi as "NVIDIA GPU present".
            $HasNvidiaSmi = $true
            Write-StudioLine "   NVIDIA GPU found through the driver library; nvidia-smi is unavailable" -ForegroundColor Gray
        }
    }
    # Last resort for #9255: presence only, gated on no healthy AMD or Intel adapter. The CUDA
    # version is decided separately by the version ladder.
    if (-not $HasNvidiaSmi) {
        $presenceScan = Invoke-BoundedVideoControllerScan
        $_presenceXpuPy = $VenvPython
        if ($script:StudioVenvRollbackDir) {
            $_presenceRollbackPy = Join-Path $script:StudioVenvRollbackDir "Scripts\python.exe"
            if (Test-Path -LiteralPath $_presenceRollbackPy) { $_presenceXpuPy = $_presenceRollbackPy }
        }
        # Any throw declines the promotion; flags are set only once every answer is in hand.
        try {
            if ((Test-NvidiaAdapterPresent -Scan $presenceScan) -and
                (Test-NvidiaAdapterWithoutNvidiaDriver -Scan $presenceScan)) {
                Write-StudioLine "   NVIDIA GPU found without the NVIDIA driver; install the NVIDIA driver and re-run for GPU support" -ForegroundColor Yellow
            } elseif ((Test-NvidiaAdapterPresent -Scan $presenceScan) -and
                -not (Test-OtherVendorAdapterPresent -Scan $presenceScan) -and
                -not $script:StudioPreservedXpuVerdict -and
                -not (Test-IntelXpuRuntimeProven -Scan $presenceScan -PythonExe $_presenceXpuPy)) {
                $_presenceFloor = Get-NvidiaAdapterCudaFloor -Scan $presenceScan
                # The floor is $null both for too-old and unreadable, which want opposite answers.
                $_presenceRelease = Get-NvidiaAdapterDriverRelease -Scan $presenceScan
                $HasNvidiaSmi = $true
                $script:NvidiaPresenceOnly = $true
                $script:NvidiaPresenceCudaFloor = $_presenceFloor
                $script:NvidiaPresenceDriverRelease = $_presenceRelease
                Write-StudioLine "   NVIDIA GPU found on the PCI bus; nvidia-smi and the driver library are both unavailable" -ForegroundColor Gray
            }
        } catch {}
    }
    # Name the card in the banner (name, compute_cap, driver in one query), honouring the
    # visible-device mask. A mask of "" or -1 hides every device: keep vendor-only wording.
    $nvMaskHidesAll = $false
    if ($null -ne $env:CUDA_VISIBLE_DEVICES) {
        $nvMask = ($env:CUDA_VISIBLE_DEVICES -replace '\s', '')
        $nvMaskHidesAll = ($nvMask -eq '' -or $nvMask -eq '-1')
    }
    if ($HasNvidiaSmi -and $NvidiaSmiExe -and -not $script:NvidiaSmiWedged -and -not $nvMaskHidesAll) {
        try {
            # Bounded runner; -StdoutOnly so stderr warnings cannot corrupt the CSV.
            $nvOut = Invoke-NvidiaSmiBounded $NvidiaSmiExe @('--query-gpu=name,compute_cap,driver_version', '--format=csv,noheader') -StdoutOnly
            $nvRows = @($nvOut -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
            if ($nvRows.Count -gt 0) {
                # nvidia-smi ignores CUDA_VISIBLE_DEVICES. Non-numeric tokens are (possibly abbreviated)
                # GPU UUIDs or MIG ids, matched by prefix.
                $nvIdx = 0
                $nvTok = if ($env:CUDA_VISIBLE_DEVICES) { ($env:CUDA_VISIBLE_DEVICES -split ',')[0].Trim() } else { '' }
                # True while nothing has IDENTIFIED a device: a plain ordinal.
                $nvByOrdinal = $true
                # Set when an identity mask was given but did not resolve, which means CUDA
                # selected NO device. Row 0 is not a fallback for that.
                $nvUnresolved = $false
                if ($nvTok -match '^\d+$') {
                    $nvIdx = [int]$nvTok
                } elseif ($nvTok -like 'MIG-*' -and $nvTok -notlike 'MIG-GPU-*') {
                    # R470+ MIG instances have opaque UUIDs; `nvidia-smi -L` nests them under their GPU.
                    $nvListOut = Invoke-NvidiaSmiBounded $NvidiaSmiExe @('-L') -StdoutOnly
                    $cur = 0
                    $nvByOrdinal = $false
                    $nvFound = $false
                    foreach ($ln in ($nvListOut -split '\r?\n')) {
                        if ($ln -match '^GPU\s+(\d+):') { $cur = [int]$Matches[1] }
                        if ($ln -match [regex]::Escape($nvTok)) { $nvIdx = $cur; $nvFound = $true; break }
                    }
                    if (-not $nvFound) { $nvUnresolved = $true }
                } elseif ($nvTok) {
                    # Pre-R470 MIG names embed the parent UUID: MIG-<GPU-UUID>/<gi>/<ci>.
                    if ($nvTok -like 'MIG-GPU-*') { $nvTok = ($nvTok.Substring(4) -split '/')[0] }
                    $nvUuidOut = Invoke-NvidiaSmiBounded $NvidiaSmiExe @('--query-gpu=uuid', '--format=csv,noheader') -StdoutOnly
                    $nvUuids = @($nvUuidOut -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
                    $nvByOrdinal = $false
                    $nvFound = $false
                    # A prefix matching two cards selects NO device, so collect every match.
                    $nvMatches = @()
                    for ($i = 0; $i -lt $nvUuids.Count; $i++) {
                        if ($nvUuids[$i].StartsWith($nvTok, [System.StringComparison]::OrdinalIgnoreCase)) { $nvMatches += $i }
                    }
                    if ($nvMatches.Count -eq 1) { $nvIdx = $nvMatches[0]; $nvFound = $true }
                    if (-not $nvFound) { $nvUnresolved = $true }
                }
                $nvRow = if ($nvIdx -lt $nvRows.Count) { $nvRows[$nvIdx] } else { $nvRows[0] }
                # CUDA's default FASTEST_FIRST order differs from nvidia-smi's PCI order, so an ordinal
                # names a row only under PCI_BUS_ID or when all rows agree. An ordinal past the last row
                # exposes no device.
                if ($nvByOrdinal -and $nvIdx -ge $nvRows.Count) { $nvUnresolved = $true }
                $nvAmbiguous = $nvUnresolved
                if ($nvByOrdinal -and -not $nvAmbiguous) {
                    $nvOrder = (("$env:CUDA_DEVICE_ORDER") -replace '\s', '').ToUpperInvariant()
                    $nvModels = @($nvRows | ForEach-Object { $_ -replace ',[^,]*$', '' } |
                                  Sort-Object -Unique).Count
                    $nvAmbiguous = ($nvOrder -ne 'PCI_BUS_ID' -and $nvModels -gt 1)
                }
                # Split from the right: nvidia-smi does not quote, so a comma in a device
                # name would otherwise shift every field.
                $nvParts = $nvRow -split ','
                if ($nvParts.Count -ge 3) {
                    $NvidiaDriverVersion = $nvParts[-1].Trim()
                    $nvComputeCap        = $nvParts[-2].Trim()
                    $NvidiaGpuName       = ($nvParts[0..($nvParts.Count - 3)] -join ',').Trim()
                    if ($nvComputeCap -match '^(\d+)\.(\d+)$') {
                        $NvidiaSmArch = "sm_" + (([int]$Matches[1] * 10) + [int]$Matches[2])
                    }
                } else {
                    # Short row: field 1 only. Taking the whole row would print the
                    # compute capability as part of the name ("RTX 4090, 8.9").
                    $NvidiaGpuName = $nvParts[0].Trim()
                }
                # An nvidia-smi too old for a field answers with a placeholder rather
                # than failing (the 470 branch has no compute_cap at all).
                $nvPlaceholders = @('[N/A]', '[Not Supported]', '[Unknown Error]')
                if ($nvPlaceholders -contains $NvidiaGpuName)       { $NvidiaGpuName = $null }
                if ($nvPlaceholders -contains $NvidiaDriverVersion) { $NvidiaDriverVersion = $null }
                # Keep the driver, drop the identity: the banner falls back to the vendor-only
                # wording rather than claiming a card that may not be the one CUDA will use.
                if ($nvAmbiguous) { $NvidiaGpuName = $null; $NvidiaSmArch = $null }
            }
        } catch {}
    }
    # ── AMD ROCm detection (Windows) — mirrors setup.ps1 ──
    $HasROCm = $false
    $HipSdkInstalled = $false   # HIP SDK binary found (independent of device accessibility)
    $ROCmGpuLabel = $null
    # Banner-only name, kept apart from $ROCmGpuLabel, which feeds the arch tables.
    $ROCmGpuName = $null
    $ROCmVersion = $null
    $ROCmGfxArch = $null
    # Declared with its neighbours, not inside the block below: the arms that read
    # it are outside that gate, so an NVIDIA host would leave it undefined.
    $ROCmUnsupportedGfxArch = $null
    if (-not $HasNvidiaSmi) {
        # Ignore the venv hipInfo.exe: an AMD wheel, not a HIP SDK, so amd-smi still auto-elevates.
        function Test-HipinfoIsVenvInternal {
            param([AllowNull()][string]$HipinfoPath)
            if ([string]::IsNullOrWhiteSpace($HipinfoPath)) { return $false }
            $venvRoots = @()
            if ($env:VIRTUAL_ENV) { $venvRoots += $env:VIRTUAL_ENV }
            $vd = Get-Variable -Name VenvDir -ValueOnly -ErrorAction SilentlyContinue
            if ($vd) { $venvRoots += $vd }
            if ($env:UNSLOTH_SETUP_PYTHON) {
                try { $venvRoots += (Split-Path -Parent (Split-Path -Parent $env:UNSLOTH_SETUP_PYTHON)) } catch {}
            }
            if ($env:USERPROFILE) { $venvRoots += (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio") }
            $studioHomeEnv = if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { $env:UNSLOTH_STUDIO_HOME.Trim() } elseif (-not [string]::IsNullOrWhiteSpace($env:STUDIO_HOME)) { $env:STUDIO_HOME.Trim() } else { $null }
            if ($studioHomeEnv) {
                # Expand a leading ~ like the canonical resolver; GetFullPath keeps it literal.
                if (($studioHomeEnv -eq "~" -or $studioHomeEnv -like "~/*" -or $studioHomeEnv -like "~\*") -and -not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                    # A bare "~" leaves an empty child path, which Join-Path rejects on PS 5.1.
                    $studioHomeRest = $studioHomeEnv.Substring(1).TrimStart('/', '\')
                    $studioHomeEnv = if ($studioHomeRest) { Join-Path $env:USERPROFILE $studioHomeRest } else { $env:USERPROFILE }
                }
                $venvRoots += (Join-Path $studioHomeEnv "unsloth_studio")
            }
            try { $hip = [System.IO.Path]::GetFullPath($HipinfoPath).TrimEnd('\', '/') } catch { return $false }
            foreach ($root in $venvRoots) {
                if ([string]::IsNullOrWhiteSpace($root)) { continue }
                try { $r = [System.IO.Path]::GetFullPath($root).TrimEnd('\', '/') } catch { continue }
                # Skip a bare drive root; it would match every path on that drive.
                if ($r -match '^[a-zA-Z]:$') { continue }
                if ($hip.Equals($r, [System.StringComparison]::OrdinalIgnoreCase) -or
                    $hip.StartsWith($r + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                    return $true
                }
            }
            return $false
        }
        # -CommandType Application skips an alias or function named hipinfo.
        $hipinfoExe = Get-Command hipinfo -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-HipinfoIsVenvInternal $_.Source) } |
            Select-Object -First 1
        if (-not $hipinfoExe) {
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
                    # Names are trusted only when they line up with the arch list; the arch never reads them.
                    $_hipAllNames = @([regex]::Matches($hipOut, "(?im)^\s*Name\s*:\s*(.+?)\s*$") | ForEach-Object { $_.Groups[1].Value.Trim() })
                    if ($_hipAllArches.Count -gt 0) {
                        # Entry 0, as setup.ps1 does: hipinfo already applied HIP/ROCR_VISIBLE_DEVICES.
                        $ROCmGfxArch  = $_hipAllArches[0]
                        $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                        if ($_hipAllNames.Count -eq $_hipAllArches.Count) {
                            $ROCmGpuName = $_hipAllNames[0]
                        }
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
        # amd-smi pops an unsuppressable UAC prompt without HIP, so probe only with a SDK or opt-in.
        $amdSmiOptOut = $env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(0|false|no|off)$'
        $amdSmiAllowed = (-not $amdSmiOptOut) -and ($HipSdkInstalled -or ($env:UNSLOTH_ENABLE_AMD_SMI -match '^(?i)(1|true|yes|on)$'))
        if (-not $HasROCm -and $amdSmiAllowed) {
            $amdSmiExe = Get-Command "amd-smi" -ErrorAction SilentlyContinue
            if ($amdSmiExe) {
                try {
                    $smiOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('list')
                    if ($LASTEXITCODE -eq 0 -and $smiOut -match "(?im)^GPU\s*[:\[]\s*\d") {
                        $HasROCm = $true
                        # gfx tokens in enumeration order, picked via HIP_VISIBLE_DEVICES.
                        $_smiVisIdx = if ($env:HIP_VISIBLE_DEVICES -match '^\d') { [int]($env:HIP_VISIBLE_DEVICES -split ',')[0] } elseif ($env:ROCR_VISIBLE_DEVICES -match '^\d') { [int]($env:ROCR_VISIBLE_DEVICES -split ',')[0] } else { 0 }
                        # Attempt 1: newer amd-smi versions embed the gfx arch in list output.
                        $_smiGfxTokens = @([regex]::Matches($smiOut, "(?i)\b(gfx\d+[a-z]?)\b") | ForEach-Object { $_.Groups[1].Value.ToLower() })
                        # Market names, when this amd-smi prints them, indexed like the gfx
                        # tokens above. Banner only; the arch that selects the wheel is never
                        # taken from here.
                        $_smiNames = @([regex]::Matches($smiOut, "(?im)Market.?Name\s*[:\|]\s*([^\r\n]+)") | ForEach-Object { $_.Groups[1].Value.Trim() })
                        if ($_smiGfxTokens.Count -gt 0) {
                            $ROCmGfxArch = if ($_smiVisIdx -lt $_smiGfxTokens.Count) { $_smiGfxTokens[$_smiVisIdx] } else { $_smiGfxTokens[0] }
                            $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                            if ($_smiNames.Count -eq $_smiGfxTokens.Count) {
                                $ROCmGpuName = if ($_smiVisIdx -lt $_smiNames.Count) { $_smiNames[$_smiVisIdx] } else { $_smiNames[0] }
                            }
                        } else {
                            # Attempt 2: 'static --asic' exposes the GFX target on ROCm 6+.
                            $smiAsicOut = ""
                            try { $smiAsicOut = Invoke-AmdSmiNoElevate $amdSmiExe.Source @('static','--asic') } catch {}
                            $_asicGfxTokens = @([regex]::Matches($smiAsicOut, "(?i)\b(gfx\d+[a-z]?)\b") | ForEach-Object { $_.Groups[1].Value.ToLower() })
                            $_asicNames = @([regex]::Matches($smiAsicOut, "(?im)Market.?Name\s*[:\|]\s*([^\r\n]+)") | ForEach-Object { $_.Groups[1].Value.Trim() })
                            if ($_asicGfxTokens.Count -gt 0) {
                                $ROCmGfxArch = if ($_smiVisIdx -lt $_asicGfxTokens.Count) { $_asicGfxTokens[$_smiVisIdx] } else { $_asicGfxTokens[0] }
                                $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                                if ($_asicNames.Count -eq $_asicGfxTokens.Count) {
                                    $ROCmGpuName = if ($_smiVisIdx -lt $_asicNames.Count) { $_asicNames[$_smiVisIdx] } else { $_asicNames[0] }
                                }
                            } elseif ($_asicNames.Count -gt 0) {
                                $ROCmGpuName = if ($_smiVisIdx -lt $_asicNames.Count) { $_asicNames[$_smiVisIdx] } else { $_asicNames[0] }
                                $ROCmGpuLabel = "AMD ROCm ($ROCmGpuName)"
                            } else {
                                $ROCmGpuLabel = "AMD ROCm"
                            }
                        }
                    }
                } catch {}
            }
        }
        # This scan fills the label the arch inference reads, so its gate is load-bearing. CIM,
        # not Get-WmiObject (removed in PowerShell 7).
        if (-not $HasROCm) {
            try {
                # Only ConfigManagerErrorCode 0 adapters, as setup.ps1 does, so a disabled Radeon does not
                # pick the wheels; if none qualify keep all (parked dGPUs report 45). @() keeps it indexable.
                $amdAdapters = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match "AMD|Radeon" })
                $healthyAdapters = @($amdAdapters | Where-Object {
                    ($null -eq $_.ConfigManagerErrorCode) -or ($_.ConfigManagerErrorCode -eq 0) })
                $wmiGpu = @(if ($healthyAdapters.Count -gt 0) { $healthyAdapters } else { $amdAdapters })[0]
                if ($wmiGpu) { $ROCmGpuLabel = $wmiGpu.Name; $ROCmGpuName = $wmiGpu.Name }
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
        # Name -> gfx for generations no ROCm wheel covers (Polaris, unslothai#8529). Wording
        # only, never wheel selection. (?!0) stops "RX 570" matching "RX 5700".
        $unsupportedNameArchTable = @(
            @{ P = "RX 4[78]0(?!0)|RX 5[789]0(?!0)|Radeon Pro WX 7100|Radeon Pro WX 5100"; A = "gfx803"  }  # Polaris 10/20/30
        )
        # ── Arch resolution: env-var override → name inference ──────────────
        # Runs even without a confirmed runtime: repo.amd.com wheels bundle their own runtime,
        # so a mapped arch installs ROCm torch directly.
        if (-not $ROCmGfxArch) {
            # 1. Manual override: set UNSLOTH_ROCM_GFX_ARCH=gfx1151 before running.
            if ($env:UNSLOTH_ROCM_GFX_ARCH) {
                $ROCmGfxArch = ($env:UNSLOTH_ROCM_GFX_ARCH.Trim().ToLower() -split ':')[0]  # drop HIP feature suffixes (gfx1010:xnack-)
                $ROCmGpuLabel = "AMD ROCm ($ROCmGfxArch)"
                substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $ROCmGfxArch" "Cyan"
            }
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
                    @{ P = "RX 6950|RX 6900|RX 6850|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900"; A = "gfx1030" }  # RDNA 2 (Navi 21) -- gfx103X family
                    @{ P = "RX 6650|RX 6600|PRO W6600|PRO W6650";                  A = "gfx1032" }  # RDNA 2 (Navi 23) -- gfx103X family
                    @{ P = "RX 6550|RX 6500|RX 6450|RX 6400|RX 6300|PRO W6400|PRO W6500|PRO W6300";  A = "gfx1034" }  # RDNA 2 (Navi 24) -- gfx103X family
                    @{ P = "Radeon Pro V520|Radeon Pro 5600M"; A = "gfx1011" }  # RDNA 1 (Navi 12) -- multi-arch index
                    @{ P = "RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700"; A = "gfx1010" }  # RDNA 1 (Navi 10) -- multi-arch index
                    @{ P = "RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300"; A = "gfx1012" }  # RDNA 1 (Navi 14) -- multi-arch index
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
                # 3. Still nothing: the card may be a generation ROCm never covered (unslothai#8529).
                # Reporting only; gated on NO adapter being covered, since $wmiGpu is adapter 0 and a mixed host
                # (RX 5700 + RX 7900) would get false "nothing can help" wording. setup.ps1 scores every adapter.
                if (-not $ROCmGfxArch) {
                    # HIP/ROCR only: CUDA_VISIBLE_DEVICES masks NVIDIA devices, not Radeons.
                    $unsupMasked = @($env:HIP_VISIBLE_DEVICES, $env:ROCR_VISIBLE_DEVICES) |
                        Where-Object { $null -ne $_ }
                    $coveredPeer = $false
                    if ($unsupMasked) {
                        # Both sources keep adapter 0, so a mask beside a supported peer would blame the wrong GPU;
                        # stay quiet rather than assume WMI order matches HIP's.
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
        # hipconfig --version works without a ROCm device.
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

    # ── WSL2 needs AMD Adrenalin >= 26.2.2; AMD referrer-gates it, so only link their page. ──
    function Show-AmdWslDriverHint {
        if ($env:UNSLOTH_SKIP_AMD_DRIVER_HINT) { return }
        try {
            $amd = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1
            if (-not $amd) { return }
            $drvDate = $null
            try {
                if ($amd.DriverDate -is [datetime]) {
                    $drvDate = $amd.DriverDate
                } elseif ($amd.DriverDate) {
                    $drvDate = [Management.ManagementDateTimeConverter]::ToDateTime([string]$amd.DriverDate)
                }
            } catch {}
            if ($drvDate -and $drvDate -ge (Get-Date '2026-02-01')) { return }
            substep "Tip: to use this GPU inside WSL too, install AMD Adrenalin 26.2.2+ (for WSL2)." "Cyan"
            substep "  Your current driver predates it; native Windows GPU is unaffected. Get it from AMD:" "Cyan"
            substep "    https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html" "Cyan"
            substep "  Then reboot and run this installer inside an Ubuntu-24.04 WSL distro." "Cyan"
            # No wsl.exe: point at the command that provisions WSL.
            $hasWsl = $false
            try { $hasWsl = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue) } catch {}
            if (-not $hasWsl) {
                substep "  No WSL yet? Install it first:  wsl --install -d Ubuntu-24.04" "Cyan"
            }
            substep "  (suppress: set UNSLOTH_SKIP_AMD_DRIVER_HINT=1)" "Cyan"
        } catch {}
    }

    # ── AMD gfx arch → AMD pip index family (repo.amd.com/rocm/whl/<family>) ──
    # Hoisted above the Intel scan: an unmapped arch gets CPU torch and must not outrank a usable Arc.
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
    # RDNA arches use AMD's multi-arch index (unslothai#11815, #11614, #11814); gfx1033 (miscomputes)
    # and CDNA stay per-family. In sync with _WINDOWS_MULTIARCH_GFX in studio/install_python_stack.py.
    $multiArchGfx = @(
        "gfx1010", "gfx1011", "gfx1012",                                        # RDNA 1
        "gfx1030", "gfx1031", "gfx1032", "gfx1034", "gfx1035", "gfx1036",       # RDNA 2, gfx1033 stays per-family
        "gfx1100", "gfx1101", "gfx1102", "gfx1103",                             # RDNA 3
        "gfx1150", "gfx1151", "gfx1152", "gfx1153",                             # RDNA 3.5
        "gfx1200", "gfx1201"                                                    # RDNA 4
    )
    $MultiArchIndexBase = if ($env:UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR) { $env:UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR.TrimEnd('/') } else { "https://repo.amd.com/rocm/whl-multi-arch" }
    # Not rocm7.14.1: its Windows wheels ship a mismatched AOTriton runtime, so fused SDPA fails (ROCm/TheRock#7992).
    $MultiArchTag = "rocm7.14.0"
    $MultiArchTorchVersion = "2.11.0"
    $MultiArchTorchvisionVersion = "0.26.0"
    $MultiArchTorchaudioVersion = "2.11.0"
    # "AMD gets GPU wheels here", not "AMD GPU present": unmapped arches install CPU torch.
    $AmdHasGpuWheels = [bool]($ROCmGfxArch -and ($archFamilyMap.ContainsKey($ROCmGfxArch) -or $multiArchGfx -contains $ROCmGfxArch))

    # ── Intel GPU detection (Arc / Data Center GPU Max / Flex) ──
    # Before the report chain: a WMI-named-only AMD adapter would otherwise hide a discrete Arc.
    # $script:IsIntelXpu = XPU wheels suit it (Arc / Data Center only, not UHD / Iris Xe).
    $HasIntelGpu = $false
    $IntelGpuLabel = $null
    # Reset each run: under `irm | iex` $script: is the caller's session and could carry a stale $true.
    $script:IsIntelXpu = $false
    # A wheel-served AMD host skips the XPU reroute; one headed for CPU torch lets Arc win.
    if (-not $HasNvidiaSmi -and -not $AmdHasGpuWheels) {
        try {
            $_gpuScan = Invoke-BoundedVideoControllerScan
            # @() wraps the WHOLE if: a one-element array unrolls to a String and += would concatenate.
            $_gpuNames = @(if ($_gpuScan.Ok) { $_gpuScan.Names } else { Get-IntelRegistryAdapterNames })
            $_xpuNameRe = "(?i)Intel.*(Arc|Data Center GPU)"
            # Non-English Windows names carry no ASCII "Intel": use the registry's PCI vendor id to RE-LABEL
            # a WMI-reported adapter, never add one. Gated on no XPU match (hybrid: "Intel UHD" + localized Arc).
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
        # A migrated env's torch can only CONFIRM the match: an unavailable runtime means a stale driver,
        # and /cpu cannot displace an installed +xpu wheel anyway. A rerun already moved the old venv aside
        # (or --no-rollback deleted it), so ask the preserved environment's verdict, not $VenvPython.
        if ($script:StudioPreservedXpuVerdict) {
            $HasIntelGpu = $true
            $script:IsIntelXpu = $true
            if (-not $IntelGpuLabel) { $IntelGpuLabel = "Intel GPU (detected by PyTorch XPU)" }
        }
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

    # Banner includes the marketing name: the bare arch ("gfx1200") means nothing to most users.
    $ROCmGpuDisplay =
        if ($ROCmGpuName -and $ROCmGfxArch) { "$ROCmGpuName ($ROCmGfxArch)" }
        elseif ($ROCmGpuName)               { $ROCmGpuName }
        elseif ($ROCmGfxArch)               { "AMD ROCm ($ROCmGfxArch)" }
        else                                { $ROCmGpuLabel }

    # Same shape per vendor: device, compute target in parentheses. Intel gets the name alone.
    $NvidiaGpuDisplay =
        if ($NvidiaGpuName -and $NvidiaSmArch) { "$NvidiaGpuName ($NvidiaSmArch)" }
        elseif ($NvidiaGpuName)                { $NvidiaGpuName }
        else                                   { "NVIDIA GPU detected" }
    $IntelGpuDisplay  = if ($IntelGpuLabel)  { $IntelGpuLabel }  else { "Intel GPU detected" }

    if ($HasNvidiaSmi) {
        step "gpu" $NvidiaGpuDisplay
        if ($NvidiaDriverVersion) { substep "Driver: $NvidiaDriverVersion" }
    } elseif ($script:IsIntelXpu) {
        # Above every AMD branch: true only when AMD gets no GPU wheel, so those would end on CPU.
        step "gpu" $IntelGpuDisplay "Green"
    } elseif ($HasROCm -and -not $ROCmUnsupportedGfxArch) {
        # amd-smi can set $HasROCm with only a market name and no gfx arch.
        step "gpu" $ROCmGpuDisplay
        $hipSdkPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "on system PATH" }
        substep "HIP SDK: $hipSdkPath"
        if ($ROCmVersionFull) { substep "hipconfig: $ROCmVersionFull" }
    } elseif ($HipSdkInstalled -and $ROCmGpuLabel -and -not $ROCmUnsupportedGfxArch) {
        # HIP SDK present but ROCm cannot see the device. Excludes known out-of-scope cards (#8529),
        # whose owners installed the SDK because this arm told them to.
        $sdkVer = if ($ROCmVersionFull) { " (HIP $ROCmVersionFull)" } else { "" }
        step "gpu" "AMD GPU detected -- not ROCm-accessible$sdkVer" "Yellow"
        substep "Detected: $ROCmGpuLabel" "Yellow"
        substep "[WARN] HIP SDK is installed but hipinfo reports no ROCm-capable device." "Yellow"
        substep "       This is a driver issue, not an SDK issue." "Yellow"
        substep "       Ensure the ROCm compute driver is installed alongside the display driver:" "Yellow"
        substep "       https://rocm.docs.amd.com/en/latest/deploy/windows/index.html" "Yellow"
    } elseif ($ROCmGfxArch) {
        # Known arch: AMD's ROCm wheels bundle their runtime, so the HIP SDK is optional.
        step "gpu" $ROCmGpuDisplay "Cyan"
        substep "Detected: $ROCmGpuLabel" "Cyan"
        substep "GPU PyTorch uses AMD's bundled-runtime ROCm wheels -- HIP SDK not required (optional)." "Cyan"
    } elseif ($ROCmUnsupportedGfxArch) {
        # Detected, identified, out of scope for ROCm PyTorch (unslothai#8529); ranks above "arch
        # unknown". Not "training runs on CPU": unsloth raises at import with no CUDA/XPU. The Vulkan
        # setter is single-quoted so $env: prints literally. Conditional: an index pin still reaches
        # the ROCm path, and setup.ps1 throws on the Vulkan variable on Windows ARM64.
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
    if (-not $HasNvidiaSmi -and ($ROCmGfxArch -or $ROCmGpuLabel)) { Show-AmdWslDriverHint }

    # Trim the URL PATH only: a whole-URL TrimEnd corrupts a token ending in "/".
    function Trim-IndexPathSlashes {
        param([string]$Url)
        $value = $Url.Trim()
        $idx = $value.IndexOfAny([char[]]@('?', '#'))
        if ($idx -lt 0) {
            return $value.TrimEnd('/')
        }
        return $value.Substring(0, $idx).TrimEnd('/') + $value.Substring($idx)
    }

    # Index leaf (cpu / cu128 / xpu / gfx1201), ?query/#fragment stripped for token mirrors.
    function Get-TorchIndexLeafName {
        param([string]$Url)
        if ([string]::IsNullOrWhiteSpace($Url)) { return "" }
        return ((($Url -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1]).ToLowerInvariant()
    }

    # "cu126" when cu126 covers every physical GPU, "uncovered" for an incompatible mix, empty
    # when unneeded/unreadable. Ignores CUDA_VISIBLE_DEVICES. Mirrors _nvidia_cu126_verdict in install.sh.
    function Get-NvidiaCu126Verdict {
        # Floor is per-release, not fixed: only 2.11 dropped sm_70 from cu128.
        param([string]$SmiExe, [int]$LegacyFloorSm = 75, [string[]]$ComputeCaps = @())
        if ($ComputeCaps.Count -gt 0) {
            $raw = $ComputeCaps -join "`n"
        } elseif ($SmiExe) {
            $raw = Invoke-NvidiaSmiBounded $SmiExe @('--query-gpu=compute_cap', '--format=csv,noheader,nounits') -StdoutOnly
            if ($LASTEXITCODE -ne 0 -or -not $raw) { return '' }
        } else { return '' }
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
        param([string]$Family, [string]$SmiExe, [string[]]$ComputeCaps = @())
        if ($Family -notin @('cu128', 'cu130')) { return $Family }
        # torch 2.11.0+cu128 dropped Volta, so cu128 now strands a pre-Turing host as cu130 does.
        $legacyFloorSm = 75
        switch (Get-NvidiaCu126Verdict $SmiExe $legacyFloorSm $ComputeCaps) {
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

    # ── Choose the PyTorch index URL from the driver CUDA version (mirrors setup.ps1) ──
    function Get-TorchIndexUrl {
        $baseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        # An explicit pin skips ALL GPU probing, as in install.sh / install_python_stack.py.
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
            return (Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL)
        }
        if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY)) {
            return "$baseUrl/$($env:UNSLOTH_TORCH_INDEX_FAMILY.Trim().Trim('/'))"
        }
        if ($script:WoaNativeCudaTorch -and $script:WoaTorchIndexUrl) {
            return $script:WoaTorchIndexUrl
        }
        $major = $null; $minor = $null; $caps = @()
        if ($NvidiaSmiExe) {
            try {
                $output = Invoke-NvidiaSmiBounded $NvidiaSmiExe
                if ($LASTEXITCODE -eq 124) {
                    substep "nvidia-smi did not answer within 10s; retrying with a 45s limit..." "Yellow"
                    $output = Invoke-NvidiaSmiBounded $NvidiaSmiExe -TimeoutSec 45
                }
                if ($output -match 'CUDA(?: UMD)? Version:\s+(\d+)\.(\d+)') {
                    $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                }
            } catch {}
        }
        if ($null -eq $major) {
            # nvidia-smi absent, stale or hung: the driver library still names the version.
            $inventory = Get-NvidiaLibraryInventory
            if ($inventory) {
                $major = $inventory.CudaMajor; $minor = $inventory.CudaMinor; $caps = $inventory.ComputeCaps
            } elseif ($script:NvidiaPresenceOnly) {
                # Reached only after the bus promotion, so warning here cannot reach a host with no NVIDIA GPU.
                if ($script:NvidiaPresenceCudaFloor) {
                    $major = [int]$script:NvidiaPresenceCudaFloor[0]
                    $minor = [int]$script:NvidiaPresenceCudaFloor[1]
                    substep "nvidia-smi and the NVIDIA driver library are both unavailable; the display driver supports at least CUDA $major.$minor, selecting wheels for it" "Yellow"
                    # Unknown compute capability: cap at cu126, the widest family still built for sm_70 (pre-Turing).
                    if ($major -gt 12 -or ($major -eq 12 -and $minor -gt 6)) {
                        $major = 12; $minor = 6
                        substep "no source could name the GPU's compute capability, so the family stops at cu126; set UNSLOTH_TORCH_INDEX_FAMILY to override" "Yellow"
                    }
                } elseif ($null -ne $script:NvidiaPresenceDriverRelease -and
                          $script:NvidiaPresenceDriverRelease -lt 450) {
                    substep "an NVIDIA GPU is present but its driver ($script:NvidiaPresenceDriverRelease series) predates R450 and carries no usable CUDA runtime; installing CPU wheels. Update the NVIDIA driver and re-run to get CUDA" "Yellow"
                    return "$baseUrl/cpu"
                } else {
                    $major = 12; $minor = 6
                    substep "an NVIDIA GPU is present but no source could name its CUDA version; defaulting to cu126. Set UNSLOTH_TORCH_INDEX_FAMILY to override" "Yellow"
                }
            } elseif (-not $NvidiaSmiExe) {
                return "$baseUrl/cpu"
            } else {
                substep "could not determine CUDA version from nvidia-smi, defaulting to cu126" "Yellow"
                $pinHint = if ($env:UNSLOTH_PYTORCH_MIRROR) { "UNSLOTH_TORCH_INDEX_FAMILY=" } else { "UNSLOTH_TORCH_INDEX_URL=$baseUrl/" }
                substep "cu126 has no kernels for Blackwell (sm_100 / sm_120). To choose the wheel yourself, re-run with" "Yellow"
                substep "  ${pinHint}cu128   (or cu130 on a driver that supports CUDA 13)" "Yellow"
                return "$baseUrl/cu126"
            }
        }
        if ($major -ge 13)                        { $family = "cu130" }
        elseif ($major -eq 12 -and $minor -ge 8)  { $family = "cu128" }
        elseif ($major -eq 12 -and $minor -ge 6)  { $family = "cu126" }
        elseif ($major -ge 12) { $family = "cu124" }
        elseif ($major -ge 11) { $family = "cu118" }
        else { return "$baseUrl/cpu" }
        return "$baseUrl/$(Get-CudaFamilyCappedForPreTuring $family $NvidiaSmiExe $caps)"
    }

    # ── Torch flavor helpers (repair a stale CPU / wrong-CUDA wheel) ──
    # Untagged wheel = cpu, matching setup.ps1's stale-venv parse.
    function ConvertTo-TorchFlavorTag {
        param([string]$TorchVersion)
        if (-not $TorchVersion) { return $null }
        if ($TorchVersion -match '\+(cu\d+)') { return $Matches[1] }
        if ($TorchVersion -match '\+rocm')    { return 'rocm' }
        if ($TorchVersion -match '\+xpu')     { return 'xpu' }
        if ($TorchVersion -match '\+cpu')     { return 'cpu' }
        return 'cpu'
    }

    function Get-ExpectedTorchFlavorTag {
        param([string]$TorchIndexUrl, [string]$ROCmIndexUrl)
        if (-not [string]::IsNullOrWhiteSpace($ROCmIndexUrl)) { return 'rocm' }
        if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) { return $null }
        # Drop query/fragment first, else .../cu128?token=x reinstalls every run.
        $leaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLowerInvariant()
        if ($leaf -match '^cu\d+$') { return $leaf }
        if ($leaf -eq 'cpu')        { return 'cpu' }
        if ($leaf -eq 'xpu')        { return 'xpu' }
        if ($leaf -match '^rocm')   { return 'rocm' }
        # gfx must be followed by a digit (an architecture leaf); gfx-private is custom.
        if ($leaf -match '^gfx[0-9]') { return 'rocm' }
        return $null
    }

    # Windows on ARM has no torchaudio wheel, so XPU spec lists key off the venv platform.
    function Get-VenvPlatformTag {
        param([string]$PythonExe)
        if (-not $PythonExe) { return "" }
        try {
            return (& $PythonExe -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
        } catch { return "" }
    }

    # XPU floor 2.6: unsloth/models/_utils.py raises at import below it.
    function Get-XpuTorchSpecs {
        param([string]$Platform)
        $specs = @("torch>=2.6,<2.11.0", "torchvision>=0.21,<0.26.0")
        if ($Platform -eq "win-arm64") { return $specs }
        return $specs + @("torchaudio>=2.6,<2.11.0")
    }

    # Bounded so a wedged "import torch" cannot stall the flavor repair.
    function Get-InstalledTorchTag {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.__version__)'
        if (-not $probe.Ok) { return $null }
        $torchVer = $probe.Output.Trim()
        if (-not $torchVer) { return $null }
        return ConvertTo-TorchFlavorTag $torchVer
    }

    # Full version: xFormers publishes one wheel per exact torch patch, not per minor.
    function Get-InstalledTorchVersion {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $null }
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.__version__)'
        if (-not $probe.Ok) { return $null }
        $torchVer = $probe.Output.Trim()
        if (-not $torchVer) { return $null }
        return $torchVer
    }

    # ── xFormers must match the torch BUILD, not just the version ──
    # _C.pyd links one exact (torch, CUDA) pair; a mismatch only logs a warning and silently drops
    # memory-efficient attention. PyPI ships only cu128, so resolve from the torch index. Rows were
    # verified live on download.pytorch.org; keep in step with _XFORMERS_WHEEL_VERSIONS in
    # studio/backend/utils/wheel_utils.py and tests/python/test_windows_xformers_wheel_match.py.
    # Not a floor: cu130 serves 0.0.35 built for 2.10.0 but declaring torch>=2.10. torch 2.7.0
    # (0.0.30) is absent: its per-interpreter wheels stop at cp312 and we default to Python 3.13.
    $script:XformersWheelVersions = @{
        "2.7.1"  = @{ "cu126" = "0.0.31.post1"; "cu128" = "0.0.31.post1" }
        "2.8.0"  = @{ "cu126" = "0.0.32.post2"; "cu128" = "0.0.32.post2"; "cu129" = "0.0.32.post2" }
        "2.9.0"  = @{ "cu126" = "0.0.33.post1"; "cu128" = "0.0.33.post1"; "cu130" = "0.0.33.post1" }
        "2.9.1"  = @{ "cu126" = "0.0.33.post2"; "cu128" = "0.0.33.post2"; "cu130" = "0.0.33.post2" }
        "2.10.0" = @{ "cu126" = "0.0.34";       "cu128" = "0.0.34";       "cu130" = "0.0.34" }
        # Stable-ABI era: 0.0.35 works on any later torch release.
        "2.11.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
        "2.12.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
        "2.13.0" = @{ "cu126" = "0.0.35";       "cu128" = "0.0.35";       "cu130" = "0.0.35" }
    }

    # Every torch STRICTLY above the floor uses the stable-ABI wheel; exact rows above still win.
    # Needed for patch releases (2.10.1, ...) published after this script ships.
    $script:XformersStableAbiFloor = "2.10.0"
    $script:XformersStableAbiVersion = "0.0.35"
    $script:XformersStableAbiFamilies = @("cu126", "cu128", "cu130")

    # $null for dev/rc/nightly strings, which name no released torch.
    function ConvertTo-TorchReleaseVersion {
        param([string]$Release)
        if (-not $Release -or -not [regex]::IsMatch($Release, '^\d+(\.\d+)*$')) { return $null }
        try { return [version]$Release } catch { return $null }
    }

    # $null when the pair has no wheel: install nothing rather than a mismatch.
    function Get-XformersWheelVersion {
        param([string]$TorchVersion, [string]$CudaTag)
        if (-not $TorchVersion -or -not $CudaTag) { return $null }
        $release = ($TorchVersion -split '\+', 2)[0].Trim()
        if ($script:XformersWheelVersions.ContainsKey($release)) {
            $byFamily = $script:XformersWheelVersions[$release]
            if ($byFamily.ContainsKey($CudaTag)) { return $byFamily[$CudaTag] }
            return $null
        }
        $parsed = ConvertTo-TorchReleaseVersion $release
        if ($null -eq $parsed) { return $null }
        if ($parsed -gt [version]$script:XformersStableAbiFloor -and $script:XformersStableAbiFamilies -contains $CudaTag) {
            return $script:XformersStableAbiVersion
        }
        return $null
    }

    # The torch build recorded in the selected wheel's cpp_lib.json: the resident torch for exact
    # rows, the floor for the stable-ABI wheel (else a good 0.0.35 reinstalls every run).
    function Get-XformersExpectedTorchBuild {
        param([string]$Version, [string]$TorchVersion, [string]$CudaTag)
        if ($Version -eq $script:XformersStableAbiVersion) {
            return "$($script:XformersStableAbiFloor)+$CudaTag"
        }
        return $TorchVersion
    }

    # Filename interpreter tag: cp39-abi3 for 0.0.31..0.0.34, py39-none for 0.0.35; unknown returns
    # $null so no URL is guessed. Mirrors _XFORMERS_FILENAME_PYTHON_TAGS in wheel_utils.py.
    function Get-XformersFilenamePythonTag {
        param([string]$Version)
        $parsed = ConvertTo-TorchReleaseVersion (($Version -split '[^0-9.]', 2)[0].TrimEnd('.'))
        if ($null -eq $parsed) { return $null }
        if ($parsed -ge [version]"0.0.31" -and $parsed -le [version]"0.0.34") { return "cp39-abi3" }
        if ($parsed -eq [version]"0.0.35") { return "py39-none" }
        return $null
    }

    # "<version> <torch-it-was-built-for>", or $null. Both halves: cu126/128/130 share a version
    # and 0.0.34/0.0.35 share a torch. Read from disk so a bad .pyd cannot log into the probe.
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

    # A +xpu wheel is not proof: on an old compute driver xpu.is_available() is False and Unsloth
    # dies at import. Warn, never fall back; a driver update fixes it.
    function Assert-XpuRuntimeReady {
        param([string]$PythonExe)
        if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { return $true }
        # Line-anchored; every failure, timeout included, warns rather than hanging.
        $probe = Invoke-BoundedPythonProbe -PythonExe $PythonExe -Code 'import torch; print(torch.xpu.is_available())'
        if ($probe.Ok -and $probe.Output -match '(?m)^\s*True\s*$') { return $true }
        substep "[WARN] PyTorch XPU is installed but torch.xpu.is_available() is False." "Yellow"
        substep "       The Intel GPU driver is most likely too old -- PyTorch XPU on Windows" "Yellow"
        substep "       needs Intel Graphics Driver 32.0.101.6739 (WHQL) or newer." "Yellow"
        substep "       Update the driver, then re-run. See:" "Yellow"
        substep "       https://unsloth.ai/docs/get-started/install/intel" "Yellow"
        return $false
    }

    # ── Torch release preservation (twin of _previous_torch_pin, PR 7250); UPGRADE=1 opts out. ──

    # Strips ONLY the +local tag; the anchored numeric match keeps dev/rc/alpha from pinning.
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

    # Fails closed on any other shape (exact pins, bare torch), so preservation defers.
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

    # $null when nothing is kept: unstable version, UPGRADE=1, or outside the final window.
    function Get-PreviousTorchPin {
        param(
            [string]$TorchVersion,
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

    $TorchIndexPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) -or `
                        (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_FAMILY))
    $TorchIndexUrl = Get-TorchIndexUrl

    # Must run AFTER Get-TorchIndexUrl above, which would overwrite it. A pin still wins.
    if ($script:IsIntelXpu -and -not $TorchIndexPinned -and -not $SkipTorch) {
        $XpuBaseUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { $env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/') } else { "https://download.pytorch.org/whl" }
        $TorchIndexUrl = "$XpuBaseUrl/xpu"
        substep "PyTorch XPU (SYCL) wheels will be installed from $(Remove-IndexUrlCredentials $TorchIndexUrl)"
    }

    # ── AMD Windows ROCm: arch-aware pip index (repo.amd.com) ──
    # Wheels bundle their own runtime, so the HIP SDK version does not matter; newest release for the
    # arch wins. UNSLOTH_ROCM_WINDOWS_MIRROR overrides for air-gapped installs.
    $ROCmIndexUrl = $null
    $ROCmTorchFloor = $null
    $PinnedRocmVisionSpec = $null
    $PinnedRocmAudioSpec = $null
    $ROCmMultiArch = $false
    if (-not $TorchIndexPinned -and ($HasROCm -or $ROCmGfxArch) -and $TorchIndexUrl -like "*/cpu" -and -not $SkipTorch) {
        $amdIndexBase = if ($env:UNSLOTH_ROCM_WINDOWS_MIRROR) { $env:UNSLOTH_ROCM_WINDOWS_MIRROR.TrimEnd('/') } else { "https://repo.amd.com/rocm/whl" }
        # gfx120X and gfx1151/gfx1150 have a null-pointer bug in torch._C._grouped_mm below 2.11.0
        # (TheRock #5284, #3284), so floor 2.11. The <2.12.0 ceiling matches install_python_stack.py;
        # bump both once 2.12.x is validated on these arches.
        $torchFloorMap = @{
            "gfx1201" = "torch>=2.11.0,<2.12.0"; "gfx1200" = "torch>=2.11.0,<2.12.0"
            "gfx1151" = "torch>=2.11.0,<2.12.0"; "gfx1150" = "torch>=2.11.0,<2.12.0"
            "gfx1152" = "torch>=2.11.0,<2.12.0"
            "gfx1030" = "torch>=2.11.0,<2.12.0"; "gfx1031" = "torch>=2.11.0,<2.12.0"
            "gfx1032" = "torch>=2.11.0,<2.12.0"; "gfx1033" = "torch>=2.11.0,<2.12.0"
            "gfx1034" = "torch>=2.11.0,<2.12.0"; "gfx1035" = "torch>=2.11.0,<2.12.0"
            "gfx1036" = "torch>=2.11.0,<2.12.0"; "gfx1100" = "torch>=2.11.0,<2.12.0"
            "gfx1101" = "torch>=2.11.0,<2.12.0"; "gfx1102" = "torch>=2.11.0,<2.12.0"
            "gfx1103" = "torch>=2.11.0,<2.12.0"
        }
        # Companions track the torch ceiling for a consistent trio on AMD's per-arch index.
        $torchvisionFloorMap = @{
            "gfx1201" = "torchvision>=0.26.0,<0.27.0"; "gfx1200" = "torchvision>=0.26.0,<0.27.0"
            "gfx1151" = "torchvision>=0.26.0,<0.27.0"; "gfx1150" = "torchvision>=0.26.0,<0.27.0"
            "gfx1152" = "torchvision>=0.26.0,<0.27.0"
            "gfx1030" = "torchvision>=0.26.0,<0.27.0"; "gfx1031" = "torchvision>=0.26.0,<0.27.0"
            "gfx1032" = "torchvision>=0.26.0,<0.27.0"; "gfx1033" = "torchvision>=0.26.0,<0.27.0"
            "gfx1034" = "torchvision>=0.26.0,<0.27.0"; "gfx1035" = "torchvision>=0.26.0,<0.27.0"
            "gfx1036" = "torchvision>=0.26.0,<0.27.0"; "gfx1100" = "torchvision>=0.26.0,<0.27.0"
            "gfx1101" = "torchvision>=0.26.0,<0.27.0"; "gfx1102" = "torchvision>=0.26.0,<0.27.0"
            "gfx1103" = "torchvision>=0.26.0,<0.27.0"
        }
        $torchaudioFloorMap = @{
            "gfx1201" = "torchaudio>=2.11.0,<2.12.0"; "gfx1200" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1151" = "torchaudio>=2.11.0,<2.12.0"; "gfx1150" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1152" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1030" = "torchaudio>=2.11.0,<2.12.0"; "gfx1031" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1032" = "torchaudio>=2.11.0,<2.12.0"; "gfx1033" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1034" = "torchaudio>=2.11.0,<2.12.0"; "gfx1035" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1036" = "torchaudio>=2.11.0,<2.12.0"; "gfx1100" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1101" = "torchaudio>=2.11.0,<2.12.0"; "gfx1102" = "torchaudio>=2.11.0,<2.12.0"
            "gfx1103" = "torchaudio>=2.11.0,<2.12.0"
        }
        $archFamily = if ($ROCmGfxArch -and $archFamilyMap.ContainsKey($ROCmGfxArch)) { $archFamilyMap[$ROCmGfxArch] } else { $null }
        # A family-layout mirror (and no multi-arch mirror) keeps the family route for arches that have one.
        $_familyMirrorPinned = [bool]($env:UNSLOTH_ROCM_WINDOWS_MIRROR) -and -not [bool]($env:UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR)
        $ROCmMultiArch = [bool]($ROCmGfxArch -and $multiArchGfx -contains $ROCmGfxArch -and -not ($archFamily -and $_familyMirrorPinned))
        if ($ROCmMultiArch) {
            $ROCmIndexUrl = "$MultiArchIndexBase/"
            $ROCmTorchFloor = "torch[device-$ROCmGfxArch]==$MultiArchTorchVersion+$MultiArchTag"
            $PinnedRocmVisionSpec = "torchvision[device-$ROCmGfxArch]==$MultiArchTorchvisionVersion+$MultiArchTag"
            $PinnedRocmAudioSpec = "torchaudio==$MultiArchTorchaudioVersion+$MultiArchTag"
            substep "$ROCmGfxArch -- AMD multi-arch index, pinned to $MultiArchTorchVersion+$MultiArchTag (torch, torchvision, torchaudio)" "Cyan"
        } elseif ($archFamily) {
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

    # A gfx*/rocm pin skips the reroute; CPU/CUDA would pull a _grouped_mm-broken wheel there.
    if ($TorchIndexPinned -and -not $ROCmIndexUrl -and -not $SkipTorch) {
        $_pinLeaf = (($TorchIndexUrl -split '[?#]', 2)[0].TrimEnd('/') -split '/')[-1].ToLower()
        $_pinRocm211 = $false
        # Anchor ($) so a suffixed custom leaf (rocm7.2-private) falls through to verbatim.
        if ($_pinLeaf -match '^rocm(\d+)\.(\d+)$') {
            # Only KNOWN-2.11 rocm (rocm7.2) gets the floor. Matches Test-RocmKnown211Version.
            $_pinRocm211 = ([int]$Matches[1] -eq 7 -and [int]$Matches[2] -eq 2)
        }
        # Only the 2.11-allowlist gfx arches need the floor (gfx908 / gfx90a stay bare).
        $_pinGfx211 = @('gfx120x-all', 'gfx1151', 'gfx1150', 'gfx1152', 'gfx103x-all', 'gfx110x-all') -contains $_pinLeaf
        if ($_pinGfx211 -or $_pinRocm211) {
            $ROCmIndexUrl = $TorchIndexUrl
            $ROCmTorchFloor = "torch>=2.11.0,<2.12.0"
            $PinnedRocmVisionSpec = "torchvision>=0.26.0,<0.27.0"
            $PinnedRocmAudioSpec = "torchaudio>=2.11.0,<2.12.0"
            substep "pinned ROCm index ($_pinLeaf) -- enforcing $ROCmTorchFloor" "Cyan"
        } elseif ($_pinLeaf -match '^gfx[0-9]' -or $_pinLeaf -match '^rocm[0-9]+(\.[0-9]+)?$') {
            # Other gfx / older rocm (<=7.1) ship torch <2.11: bare specs. Only EXACT leaves count.
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

    if (-not $SkipTorch -and -not $ROCmIndexUrl -and $TorchIndexUrl -like "*/cpu") {
        Write-StudioLine ""
        if ($ROCmGfxArch) {
            substep "Installing CPU PyTorch -- no ROCm PyTorch wheels are available for $ROCmGfxArch." "Yellow"
            substep "PyTorch (training and Transformers inference) runs on CPU on this GPU." "Yellow"
        } else {
            if ($HipSdkInstalled -and -not $HasROCm -and -not $ROCmUnsupportedGfxArch) {
                # Guarded like the gpu step: on these cards an installed HIP SDK is the symptom.
                substep "Installing CPU-only PyTorch (HIP SDK found but GPU not ROCm-accessible)." "Yellow"
            } elseif ($ROCmUnsupportedGfxArch) {
                substep "Installing CPU PyTorch -- Unsloth has no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch." "Yellow"
                substep "Unsloth training and GPU inference are unavailable on CPU torch." "Yellow"
                substep "Neither the HIP SDK nor UNSLOTH_ROCM_GFX_ARCH can give this GPU ROCm." "Yellow"
                if ((Get-HostMachineArch) -eq "arm64") {
                    # No Windows ARM64 Vulkan bundle exists and setup.ps1 throws on the variable.
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
    # `uv pip install unsloth --torch-backend=cpu` resolves to the pre-CLI unsloth==2024.8. Use
    # --upgrade-package so upgrading unsloth cannot re-resolve torch from PyPI and drop +cuXXX.
    # uv splits -r/-c/--overrides on whitespace (#6503, #10722, #11012), so the requirements path
    # must be space-free. Mirrors uv_safe_path in uv_path_safety.py. Returns whether it copied.
    function Get-UvSafeRequirementsPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not $Path.Contains(" ")) { return @{ Path = $Path; Temporary = $false } }
        $short = $null
        try { $short = (New-Object -ComObject Scripting.FileSystemObject).GetFile($Path).ShortPath } catch { }
        # An 8.3 name may not resolve, and uv then cannot open it (#11290).
        if ($short -and -not $short.Contains(" ") -and (Test-Path -LiteralPath $short -PathType Leaf -ErrorAction SilentlyContinue)) {
            return @{ Path = $short; Temporary = $false }
        }

        # No 8.3 name: copy to a space-free, writable directory (%TEMP% itself may hold the space).
        # Built one at a time: under "Stop" a throwing element in an array literal aborts the install.
        $candidates = @()
        foreach ($produce in @(
            { [System.IO.Path]::GetTempPath() },
            { $env:TEMP },
            { $env:TMP },
            { $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetTempPath())
              if ($root) { Join-Path $root "Windows\Temp" } },
            { [System.IO.Path]::GetDirectoryName($Path) }
        )) {
            try { $candidates += & $produce } catch { }
        }
        foreach ($candidate in $candidates) {
            if (-not $candidate) { continue }
            $dir = $candidate
            if ($dir.Contains(" ")) {
                $dirShort = $null
                try { $dirShort = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($dir).ShortPath } catch { }
                # An 8.3 name that does not resolve is unusable (#11290).
                if (-not $dirShort -or $dirShort.Contains(" ") -or -not (Test-Path -LiteralPath $dirShort -PathType Container -ErrorAction SilentlyContinue)) { continue }
                $dir = $dirShort
            }
            # Declared outside the try for cleanup; built inside, since Join-Path on an unmapped drive throws.
            $tmp = $null
            try {
                $tmp = Join-Path $dir ("unsloth-reqs-" + [guid]::NewGuid().ToString("N") + ".txt")
                Copy-Item -LiteralPath $Path -Destination $tmp -Force -ErrorAction Stop
                if ($tmp.Contains(" ")) {
                    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
                    continue
                }
                return @{ Path = $tmp; Temporary = $true }
            } catch {
                if ($tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
                continue
            }
        }

        # uv's split-path error reads as a missing requirements file (#11012), so warn; return the
        # original so the install behaves as before.
        substep "[WARN] the requirements path contains a space and no space-free location was" "Yellow"
        substep "available; uv splits such a path, so this install may fail. Set TMP and TEMP" "Yellow"
        substep "to a path without spaces and run the installer again." "Yellow"
        return @{ Path = $Path; Temporary = $false }
    }

    function Find-NoTorchRuntimeFile {
        if ($StudioLocalInstall -and (Test-Path (Join-Path $RepoRoot "studio\backend\requirements\no-torch-runtime.txt"))) {
            return Join-Path $RepoRoot "studio\backend\requirements\no-torch-runtime.txt"
        }
        $installed = Get-ChildItem -LiteralPath $VenvDir -Recurse -Filter "no-torch-runtime.txt" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*studio*backend*requirements*no-torch-runtime.txt" } |
            Select-Object -ExpandProperty FullName -First 1
        return $installed
    }

    # ── Freeze the trio: a released unsloth wheel can downgrade it. The caller removes it. ──
    function New-UnslothTorchOverridesFile {
        param([string]$PythonExe)
        if ($SkipTorch) { return $null }
        $pins = & $PythonExe -I -c "from importlib.metadata import version, PackageNotFoundError`nfor _p in ('torch', 'torchvision', 'torchaudio'):`n    try:`n        print(_p + '==' + version(_p))`n    except PackageNotFoundError:`n        pass" 2>$null
        $lines = @($pins | Where-Object { $_ -match '^torch' })
        if ($lines.Count -eq 0 -or $lines[0] -notmatch '^torch==') { return $null }
        # --overrides replaces any UV_OVERRIDE file, so fold caller files in, minus their trio,
        # REBASED since uv resolves relative references against the containing file.
        if ($env:UV_OVERRIDE) {
            foreach ($ovFile in ($env:UV_OVERRIDE -split '\s+' | Where-Object { $_ })) {
                if (-not (Test-Path -LiteralPath $ovFile -PathType Leaf)) { continue }
                foreach ($ovEntry in (Get-WoaRequirementEntries -Path $ovFile)) {
                    if ($ovEntry.Line -match '^\s*torch(vision|audio)?([\s<>=!~;@[]|$)') { continue }
                    $lines += (Resolve-WoaOverrideLine -Line $ovEntry.Line -BaseDir $ovEntry.BaseDir)
                }
            }
        }
        $f = [System.IO.Path]::GetTempFileName()
        # Track it at once: a throwing WriteAllText would otherwise leave credentials on disk.
        $script:TorchOverridesFile = $f
        # May hold credentials, and New-Item inherits the directory ACL; twin of install.sh's chmod 600.
        try {
            $_acl = Get-Acl -LiteralPath $f
            $_acl.SetAccessRuleProtection($true, $false)
            $_me = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
            $_acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
                $_me, "FullControl", "Allow")))
            Set-Acl -LiteralPath $f -AclObject $_acl
        } catch { }   # best effort, and a no-op off Windows
        # UTF-8 without BOM (ascii rewrites non-ASCII as "?"), newline-terminated.
        try {
            [System.IO.File]::WriteAllText($f, (($lines -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
        } catch {
            Remove-UnslothTempFileQuietly -Path $f
            $script:TorchOverridesFile = $null
            throw
        }
        # uv splits --overrides on spaces (#10722); the 8.3 name keeps relative includes resolving.
        if ($f.Contains(" ")) {
            $short = $null
            try { $short = (New-Object -ComObject Scripting.FileSystemObject).GetFile($f).ShortPath } catch { }
            # An unresolvable 8.3 name makes uv fail and Remove-Item raise a terminating error;
            # require a real file.
        # -ErrorAction SilentlyContinue is load-bearing: under "Stop", Test-Path in an ACL-denied
        # directory throws, and none of these guards sits inside a try.
            if (-not $short -or $short.Contains(" ") -or -not (Test-Path -LiteralPath $short -PathType Leaf -ErrorAction SilentlyContinue)) {
                substep "[WARN] the torch overrides path has a space and no usable 8.3 short name;" "Yellow"
                substep "installing unsloth without freezing the installed PyTorch." "Yellow"
                Remove-UnslothTempFileQuietly -Path $f
                # Already deleted: clear it so the outer sweep skips it.
                $script:TorchOverridesFile = $null
                return $null
            }
            $f = $short
        }
        return $f
    }

    $_desktopMinVer = if ($env:UNSLOTH_DESKTOP_BACKEND_VERSION) { $env:UNSLOTH_DESKTOP_BACKEND_VERSION.Trim() } else { "" }
    $_unslothDesktopInstallSpec = if ($_desktopMinVer) { "unsloth>=$_desktopMinVer" } else { $null }
    $_unslothReleaseInstallSpec = if ($_unslothDesktopInstallSpec) { $_unslothDesktopInstallSpec } else { "unsloth>=2026.10.3" }

    if ($_Migrated) {
        Write-TauriLog "STEP" "Installing unsloth"
        substep "upgrading unsloth in migrated environment..."
        if ($SkipTorch) {
            # No-torch: install unsloth + unsloth-zoo with --no-deps, then
            # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
            # --no-deps means unsloth's own metadata is never read, so this spec IS the zoo
            # floor for this path. Keep it equal to the unsloth_zoo floor in pyproject.toml
            # (tests/test_installer_zoo_floor_parity.py enforces that).
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated no-torch)" { & $script:UvExe pip install --python $VenvPython --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.10.3" }
            if ($baseInstallExit -eq 0) {
                # pydantic WITH deps so pydantic-core matches; no-torch-runtime.txt is --no-deps.
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { & $script:UvExe pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $NoTorchReqSafe = Get-UvSafeRequirementsPath -Path $NoTorchReq
                    $NoTorchReqArg = $NoTorchReqSafe.Path
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { & $script:UvExe pip install --python $VenvPython --no-deps -r $NoTorchReqArg }
                    if ($NoTorchReqSafe.Temporary) {
                        Remove-Item -LiteralPath $NoTorchReqArg -Force -ErrorAction SilentlyContinue
                    }
                }
            }
        } else {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (migrated)" { & $script:UvExe pip install --python $VenvPython --reinstall-package unsloth --reinstall-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.10.3" }
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
        # Bounded default trio, hoisted so release preservation uses it as the route window.
        # torchaudio 2.11 dropped its torch pin, so a bare companion can drift.
        $_pinTorchSpec = "torch>=2.4,<2.12.0"
        $_pinVisionSpec = "torchvision>=0.19,<0.27.0"
        $_pinAudioSpec = "torchaudio>=2.4,<2.12.0"
        # Release preservation, after every floor choice. Exported as UNSLOTH_KEPT_TORCH.
        $script:PrevTorchPin = $null
        # An interrupted run leaks a stale pin, so clear before deciding.
        Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
        if (-not $SkipTorch -and $script:PrevTorchVer -and -not $ROCmMultiArch) {
            $_routeWindow = $_pinTorchSpec
            # Vet the kept release against the XPU window, or a kept 2.5 becomes an unsatisfiable pin.
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
            # Pin companions to $torchSpec: bare names can resolve an ABI-incompatible pair.
            $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
            $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
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
                # Explicit CPU index: under a ROCm pin $TorchIndexUrl IS the mirror that failed.
                $CpuFallbackIndexUrl = if ($env:UNSLOTH_PYTORCH_MIRROR) { "$($env:UNSLOTH_PYTORCH_MIRROR.TrimEnd('/'))/cpu" } else { "https://download.pytorch.org/whl/cpu" }
                substep "ROCm PyTorch install failed (exit $torchInstallExit); using a CPU base, Unsloth setup retries ROCm." "Yellow"
                # --force-reinstall: a failed ROCm install can leave a ROCm torch inside the CPU range, which
                # uv would keep while swapping companions.
                $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch (CPU fallback)" { & $script:UvExe pip install --python $VenvPython --force-reinstall "torch>=2.4,<2.12.0" "torchvision>=0.19,<0.27.0" "torchaudio>=2.4,<2.12.0" --default-index $CpuFallbackIndexUrl }
                if ($torchInstallExit -ne 0) {
                    Write-StudioLine "[ERROR] Failed to install PyTorch (ROCm and CPU base both failed, exit code $torchInstallExit)" -ForegroundColor Red
                    return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
                }
                # Drop the ROCm expectation so the flavor repair does not retry the failed index.
                $ROCmIndexUrl = $null
                $ROCmTorchFloor = $null
            }
        # Keyed off the index leaf: a FAMILY=xpu pin on a mixed NVIDIA box never ran the Intel scan.
        } elseif ((Get-TorchIndexLeafName $TorchIndexUrl) -eq "xpu") {
            # ── Intel Arc / XPU PyTorch install (wheels carry their own oneAPI runtime) ──
            Write-TauriLog "STEP" "Installing PyTorch (Intel XPU)"
            substep "installing PyTorch from $(Remove-IndexUrlCredentials $TorchIndexUrl)..."
            # Bound the trio: the xpu index serves torch past our ceiling. Floor 2.6 (unsloth raises below it
            # on XPU); win-arm64 has no torchaudio, so drop that pin.
            $VenvPlatform = Get-VenvPlatformTag -PythonExe $VenvPython
            $_xpuSpecs = Get-XpuTorchSpecs -Platform $VenvPlatform
            $_xpuCpuSpecs = @("torch>=2.4,<2.12.0", "torchvision>=0.19,<0.27.0", "torchaudio>=2.4,<2.12.0")
            if ($VenvPlatform -eq "win-arm64") {
                substep "windows on arm: skipping torchaudio (upstream publishes no win_arm64 wheel)."
                $_xpuCpuSpecs = @("torch>=2.4,<2.12.0", "torchvision>=0.19,<0.27.0")
            }
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
                # A name match is not a working runtime: warn on a stale driver, never fall back to CPU.
                Assert-XpuRuntimeReady -PythonExe $VenvPython | Out-Null
            }
        } else {
            Write-TauriLog "STEP" "Installing PyTorch"
            # Windows on ARM lacks only torchaudio; ask the interpreter, not PROCESSOR_ARCHITECTURE.
            $VenvPlatform = ""
            try {
                $VenvPlatform = (& $VenvPython -c "import sysconfig; print(sysconfig.get_platform())" 2>$null | Out-String).Trim().ToLowerInvariant()
            } catch { $VenvPlatform = "" }
            substep "installing PyTorch ($(Remove-IndexUrlCredentials $TorchIndexUrl))..."
            # Bound companions on EVERY index: torchaudio 2.11 dropped its exact torch pin. Mirrors install.sh.
            $_torchSpecs = @($_pinTorchSpec, $_pinVisionSpec, $_pinAudioSpec)
            $_torchExtraArgs = @()
            if ($VenvPlatform -eq "win-arm64") {
                if (-not $script:WoaTorchAudio) {
                    substep "windows on arm: skipping torchaudio (upstream publishes no"
                    substep "win_arm64 wheel); torch and torchvision install normally."
                }
                $_torchSpecs = @($_pinTorchSpec, $_pinVisionSpec)
            }
            # NVIDIA's out-of-tree builds sit above the usual ceiling, so lift it. Floors stay.
            if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {
                # Pinned exactly with its local +cu tag: that version is on no other index.
                $_torchSpecs = @()
                $_torchSpecs += if ($script:WoaTorchWheelVersion) { "torch==$($script:WoaTorchWheelVersion)" } else { "torch>=2.4" }
                $_torchSpecs += if ($script:WoaVisionWheelVersion) { "torchvision==$($script:WoaVisionWheelVersion)" } else { "torchvision>=0.19" }
                if ($script:WoaTorchAudio) {
                    $_torchSpecs += if ($script:WoaAudioWheelVersion) { "torchaudio==$($script:WoaAudioWheelVersion)" } else { "torchaudio>=2.4" }
                    substep "windows on arm: this index publishes torchaudio; installing the full trio."
                }
                # NVIDIA's index publishes only the trio; dependencies come from the caller's index policy.
                $_woaDependencyIndexArgs = @(Get-WoaDependencyIndexArgs)
                $_torchExtraArgs = @("--index-strategy", "unsafe-best-match") + $_woaDependencyIndexArgs
                # Invoke-InstallCommand clears UV_FIND_LINKS, so name the wheelhouse on the command line.
                if ($script:WoaDir) {
                    $_torchExtraArgs += @("--find-links", (Get-UvSafePath (Join-Path $script:WoaDir "wheels")))
                }
                if ($_woaDependencyIndexArgs.Count -eq 0) {
                    substep "windows on arm: no-index is set, so torch's dependencies must come from the find-links wheelhouse."
                } elseif ($_woaDependencyIndexArgs -notcontains "https://pypi.org/simple") {
                    substep "windows on arm: torch's dependencies resolve from the configured index, not public PyPI."
                }
                if ($script:WoaTorchIsPrerelease -or ($script:WoaTorchIndexUrl -match 'nightly')) {
                    $_torchExtraArgs = @("--prerelease=allow") + $_torchExtraArgs
                    substep "windows on arm: allowing prerelease torch (this index publishes win_arm64 CUDA wheels as nightlies)."
                }
            }
            # Release preservation cannot run here: its kept specs have no win_arm64 build.
            if ($script:PrevTorchPin -and $script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {
                $script:PrevTorchPin = $null
                Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
            }
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
                $_woaOverrideSaved = $null
                $_woaOverrideSwapped = $false
                $_woaOverrideTemps = @()
                $_woaCutoffSaved = @{}
                if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64" -and $env:UV_OVERRIDE) {
                    $_woaOverrideSaved = $env:UV_OVERRIDE
                    $_woaStep = New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved -Dir $script:WoaDir
                    $env:UV_OVERRIDE = $_woaStep.Value
                    $_woaOverrideTemps = @($_woaStep.Temps)
                    $_woaOverrideSwapped = $true
                }
                if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {
                    # The probe read the index page, which has no upload dates, and the pin is
                    # exact, so drop upload cutoffs for this one command.
                    foreach ($_woaCutoffName in @("UV_EXCLUDE_NEWER", "UV_EXCLUDE_NEWER_PACKAGE")) {
                        $_woaCutoffValue = [string][Environment]::GetEnvironmentVariable($_woaCutoffName)
                        if ($_woaCutoffValue) {
                            $_woaCutoffSaved[$_woaCutoffName] = $_woaCutoffValue
                            Remove-Item "Env:$_woaCutoffName" -ErrorAction SilentlyContinue
                            substep "windows on arm: $_woaCutoffName is not applied to the exact CUDA pin (the index carries no upload dates)."
                        }
                    }
                    # UV_NO_INDEX is our convention; the CUDA trio lives only on NVIDIA's index, so it yields for
                    # this command (deps still from the wheelhouse), then is restored.
                    $_woaNoIndexValue = [string][Environment]::GetEnvironmentVariable("UV_NO_INDEX")
                    if (Test-NoIndexRequested) {
                        $_woaCutoffSaved["UV_NO_INDEX"] = $_woaNoIndexValue
                        Remove-Item "Env:UV_NO_INDEX" -ErrorAction SilentlyContinue
                        substep "windows on arm: UV_NO_INDEX yields for the CUDA trio, which only the selected index carries; its dependencies still come from the wheelhouse."
                    }
                }
                try {
                    $torchInstallExit = Invoke-InstallCommandRetry -Label "install PyTorch" { & $script:UvExe pip install --python $VenvPython @_torchSpecs --default-index $TorchIndexUrl @_torchExtraArgs }
                } finally {
                    if ($_woaOverrideSwapped) { $env:UV_OVERRIDE = $_woaOverrideSaved }
                    foreach ($_woaTmp in $_woaOverrideTemps) { Remove-Item -LiteralPath $_woaTmp -Force -ErrorAction SilentlyContinue }
                    foreach ($_woaCutoffName in @($_woaCutoffSaved.Keys)) { Set-Item "Env:$_woaCutoffName" $_woaCutoffSaved[$_woaCutoffName] }
                }
            }
            if ($torchInstallExit -ne 0) {
                Write-StudioLine "[ERROR] Failed to install PyTorch (exit code $torchInstallExit)" -ForegroundColor Red
                return (Exit-InstallFailure "Failed to install PyTorch (exit code $torchInstallExit)" $torchInstallExit)
            }
        }

        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($SkipTorch) {
            # --no-deps: this spec IS the zoo floor here. Kept equal to pyproject.toml's.
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (no-torch)" { & $script:UvExe pip install --python $VenvPython --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.10.3" }
            if ($baseInstallExit -eq 0) {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install pydantic" { & $script:UvExe pip install --python $VenvPython pydantic }
            }
            if ($baseInstallExit -eq 0) {
                $NoTorchReq = Find-NoTorchRuntimeFile
                if ($NoTorchReq) {
                    $NoTorchReqSafe = Get-UvSafeRequirementsPath -Path $NoTorchReq
                    $NoTorchReqArg = $NoTorchReqSafe.Path
                    $baseInstallExit = Invoke-InstallCommandRetry -Label "install no-torch runtime deps" { & $script:UvExe pip install --python $VenvPython --no-deps -r $NoTorchReqArg }
                    if ($NoTorchReqSafe.Temporary) {
                        Remove-Item -LiteralPath $NoTorchReqArg -Force -ErrorAction SilentlyContinue
                    }
                }
            }
        } elseif ($StudioLocalInstall) {
            # Freeze the trio so this with-deps resolve cannot downgrade the pinned build.
            $script:TorchOverridesFile = New-UnslothTorchOverridesFile -PythonExe $VenvPython
            if ($script:TorchOverridesFile) {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (local)" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth --overrides $script:TorchOverridesFile "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.10.3" }
                Remove-UnslothTempFileQuietly -Path $script:TorchOverridesFile
                $script:TorchOverridesFile = $null
            } else {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (local)" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth "$_unslothReleaseInstallSpec" "unsloth-zoo>=2026.10.3" }
            }
        } else {
            $_unslothPkg = if ($PackageName -eq "unsloth" -and $_unslothDesktopInstallSpec) { $_unslothDesktopInstallSpec } else { $PackageName }
            $script:TorchOverridesFile = New-UnslothTorchOverridesFile -PythonExe $VenvPython
            if ($script:TorchOverridesFile) {
                $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth" { & $script:UvExe pip install --python $VenvPython --upgrade-package unsloth --overrides $script:TorchOverridesFile -- "$_unslothPkg" }
                Remove-UnslothTempFileQuietly -Path $script:TorchOverridesFile
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
        # Fallback: GPU detection produced no URL, so uv resolves torch.
        Write-TauriLog "STEP" "Installing unsloth"
        substep "installing unsloth (this may take a few minutes)..."
        if ($StudioLocalInstall) {
            $baseInstallExit = Invoke-InstallCommandRetry -Label "install unsloth (auto torch backend)" { & $script:UvExe pip install --python $VenvPython "unsloth-zoo>=2026.10.3" "$_unslothReleaseInstallSpec" --torch-backend=auto }
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

    # ── Intel XPU: bitsandbytes must carry XPU kernels or 4-bit QLoRA is off ──
    # Floor 0.50.0 (AMD's: <=0.49.2 NaNs at 4-bit decode, and an Arc can sit beside a Radeon); runs
    # after the unsloth install so a migrated venv cannot keep an older wheel. --no-deps, and not the
    # intel-gpu-torch extra (pins a single +xpu URL). Keyed off the index leaf; best-effort.
    if (-not $SkipTorch -and (Get-TorchIndexLeafName $TorchIndexUrl) -eq "xpu") {
        substep "installing bitsandbytes with Intel XPU kernels..."
        $bnbXpuExit = Invoke-InstallCommandRetry -Label "install bitsandbytes (Intel XPU)" { & $script:UvExe pip install --python $VenvPython --no-deps "bitsandbytes>=0.50.0" }
        if ($bnbXpuExit -ne 0) {
            substep "[WARN] could not install an XPU-capable bitsandbytes (exit $bnbXpuExit); 4-bit QLoRA may be unavailable." "Yellow"
        }
    }

    $installedPackageVersion = (& $VenvPython -I -c "
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

    # ── PEP 440 ignores +cpu/+cuXXX, so uv keeps a stale torch+cpu against a CUDA index:
    # reinstall the triplet when a GPU build is expected. ──
    if (-not $SkipTorch) {
        $expectedTorchTag = Get-ExpectedTorchFlavorTag -TorchIndexUrl $TorchIndexUrl -ROCmIndexUrl $ROCmIndexUrl
        if ($expectedTorchTag -and $expectedTorchTag -ne 'cpu') {
            $installedTorchTag = Get-InstalledTorchTag -PythonExe $VenvPython
            if ($installedTorchTag -and $installedTorchTag -ne $expectedTorchTag) {
                if ($expectedTorchTag -eq 'rocm' -and $ROCmIndexUrl) {
                    $rocmSpec = if ($ROCmTorchFloor) { $ROCmTorchFloor } else { "torch" }
                    $visionSpec = if ($PinnedRocmVisionSpec) { $PinnedRocmVisionSpec } elseif ($ROCmGfxArch -and $torchvisionFloorMap -and $torchvisionFloorMap.ContainsKey($ROCmGfxArch)) { $torchvisionFloorMap[$ROCmGfxArch] } else { "torchvision" }
                    $audioSpec = if ($PinnedRocmAudioSpec) { $PinnedRocmAudioSpec } elseif ($ROCmGfxArch -and $torchaudioFloorMap -and $torchaudioFloorMap.ContainsKey($ROCmGfxArch)) { $torchaudioFloorMap[$ROCmGfxArch] } else { "torchaudio" }
                    # Kept-release substitution, as install.sh's _install_torch_default_index does.
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
                    $_fixTorchSpec = "torch>=2.4,<2.12.0"; $_fixVisionSpec = "torchvision>=0.19,<0.27.0"; $_fixAudioSpec = "torchaudio>=2.4,<2.12.0"
                    $_cudaKept = $false
                    if ($script:PrevTorchPin) {
                        $_origFixTorchSpec = $_fixTorchSpec; $_origFixVisionSpec = $_fixVisionSpec; $_origFixAudioSpec = $_fixAudioSpec
                        $_fixTorchSpec = $script:PrevTorchPin.TorchSpec; $_fixVisionSpec = $script:PrevTorchPin.VisionSpec; $_fixAudioSpec = $script:PrevTorchPin.AudioSpec
                        $_cudaKept = $true
                    }
                    substep "PyTorch flavor mismatch (installed $installedTorchTag, need $expectedTorchTag) -- reinstalling correct build..." "Yellow"
                    $_fixSpecs = @($_fixTorchSpec, $_fixVisionSpec, $_fixAudioSpec)
                    $_origFixSpecs = if ($_cudaKept) { @($_origFixTorchSpec, $_origFixVisionSpec, $_origFixAudioSpec) } else { $_fixSpecs }
                    if ($expectedTorchTag -eq 'xpu') {
                        $_origFixSpecs = Get-XpuTorchSpecs -Platform (Get-VenvPlatformTag -PythonExe $VenvPython)
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
            if ($installedTorchTag -eq 'cpu') {
                Write-StudioLine ""
                Write-StudioLine "  [WARN] PyTorch is CPU-only but a $expectedTorchTag GPU build was expected for this machine." -ForegroundColor Yellow
                Write-StudioLine "  [WARN] Training and GPU inference will run on CPU until this is fixed." -ForegroundColor Yellow
                Write-StudioLine "  [WARN] Re-run this installer, or reinstall the GPU build manually for your GPU." -ForegroundColor Yellow
            }
        }
    }

    # ── Pin xFormers to the wheel built for the torch actually installed ──
    # After the flavor repair, keyed off the installed torch, not $TorchIndexUrl (a migrated venv can
    # hold +cu128 under a /cpu leaf). Optional: failures warn, no match installs nothing.
    # UNSLOTH_SKIP_XFORMERS=1 opts out.
    if (-not $SkipTorch -and $env:UNSLOTH_SKIP_XFORMERS -ne "1") {
        $_xfTorchVersion = Get-InstalledTorchVersion -PythonExe $VenvPython
        $_xfCudaTag = ConvertTo-TorchFlavorTag $_xfTorchVersion
        # cu<digits> only. IsMatch, not -match, so $Matches is not clobbered.
        if ($_xfTorchVersion -and [regex]::IsMatch([string]$_xfCudaTag, '^cu\d+$')) {
            $_xfVersion = Get-XformersWheelVersion -TorchVersion $_xfTorchVersion -CudaTag $_xfCudaTag
            if (-not $_xfVersion) {
                substep "no matching xFormers wheel for torch $_xfTorchVersion -- skipping it (attention falls back to torch SDPA)."
            } elseif ((Get-InstalledXformersBuild -PythonExe $VenvPython) -eq "$_xfVersion $(Get-XformersExpectedTorchBuild -Version $_xfVersion -TorchVersion $_xfTorchVersion -CudaTag $_xfCudaTag)") {
                substep "xFormers $_xfVersion already matches torch $_xfTorchVersion."
            } else {
                # An explicit full-URL pin is authoritative, even a non-family mirror root (air-gapped hosts).
                # Otherwise use the DIRECT wheel URL: --default-index is not exclusive (UV_INDEX adds to it) and
                # cu126/128/130 share a version, so an index could serve the wrong family.
                $_xfIndexUrl = $null
                $_xfWheelUrl = $null
                $_xfPyTag = Get-XformersFilenamePythonTag $_xfVersion
                $_xfWheelName = if ($_xfPyTag) { "xformers-$_xfVersion-$_xfPyTag-win_amd64.whl" } else { $null }
                # Full-URL override only: a FAMILY pin can build a direct URL, so it takes that path.
                if ($TorchIndexUrl -and -not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL)) {
                    # A CUDA leaf still gets a direct URL; only a bare mirror root resolves, with machine indexes cleared.
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
                        # Unknown filename shape: use the index rather than guess a URL that 404s.
                        $_xfIndexUrl = Join-UrlPath $_xfBase $_xfCudaTag
                    }
                }
                $_xfSource = if ($_xfWheelUrl) { $_xfWheelUrl } else { $_xfIndexUrl }
                substep "installing xFormers $_xfVersion for torch $_xfTorchVersion ($(Remove-IndexUrlCredentials $_xfSource))..."
                # --no-deps: the wheel's torch==<release> could pull a PyPI torch over the CUDA build.
                # --reinstall-package: wrong-CUDA wheels share the version string.
                # $script:UvExe (via Get-Variable, StrictMode-safe) avoids a profile alias named uv.
                $_xfUv = Get-Variable -Name 'UvExe' -Scope Script -ValueOnly -ErrorAction SilentlyContinue
                if (-not $_xfUv) { $_xfUv = 'uv' }
                $_xfExit = if ($_xfWheelUrl) {
                    Invoke-InstallCommandRetry -Label "install xFormers" { & $_xfUv pip install --python $VenvPython --no-deps --reinstall-package xformers $_xfWheelUrl }
                } else {
                    # UV_INDEX / UV_EXTRA_INDEX_URL add to --default-index; clear them for this call only.
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

    # ── CI only: editable overlay of a source checkout (UNSLOTH_CI_SOURCE_OVERLAY) ──
    # Clean-machine legs install unsloth from PyPI; the overlay makes setup.ps1 and friends come from
    # this ref. Not --local: it git-clones unsloth-zoo, and these legs remove git.
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

    # setup.ps1 comes from PyPI and may predate ARM64 support: test by capability, not version.
    if ($script:WoaNativeCudaTorch) {
        $WoaStudioSetup = $null
        try {
            $WoaStudioSetup = (& $VenvPython -I -c "import pathlib, studio; print(pathlib.Path(studio.__file__).parent / 'setup.ps1')" 2>$null | Select-Object -First 1)
        } catch {}
        $WoaStudioAware = $false
        if ($WoaStudioSetup -and (Test-Path -LiteralPath $WoaStudioSetup -PathType Leaf)) {
            $WoaStudioAware = [bool](Select-String -LiteralPath $WoaStudioSetup -Pattern 'WinArm64Venv' -Quiet)
        }
        if (-not $WoaStudioAware) {
            Write-StudioLine "[ERROR] This unsloth release predates Windows-on-ARM support." -ForegroundColor Red
            Write-StudioLine "        The installed studio setup would replace the ARM64 CUDA PyTorch just installed" -ForegroundColor Yellow
            Write-StudioLine "        with a build that has no win_arm64 wheel, leaving the environment without torch." -ForegroundColor Yellow
            Write-StudioLine "        Upgrade to a newer unsloth, or re-run with UNSLOTH_WOA_NATIVE=0 for the x64 path." -ForegroundColor Yellow
            return (Exit-InstallFailure "installed unsloth has no Windows-on-ARM support")
        }
    }

    # ── Run studio setup ──
    # setup.ps1 installs Git, CMake, VS Build Tools and CUDA via winget; Node is isolated, not winget.
    Write-TauriLog "STEP" "Running studio setup"
    step "setup" "running unsloth studio setup..."
    # Tested, never executed. Not required: it can be removed from an otherwise working venv, and
    # nothing later runs it (everything uses $VenvPython), so fall back to asking the interpreter.
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
    # The file the handoff starts gets its own check, not a mid-setup launch failure.
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-TauriLog "ERROR" "managed Python is missing"
        Write-StudioLine "[ERROR] The managed Python interpreter is missing." -ForegroundColor Red
        Write-StudioLine "        Expected: $VenvPython" -ForegroundColor Yellow
        Write-StudioLine "        Re-run the installer to rebuild the environment." -ForegroundColor Yellow
        return (Exit-InstallFailure "managed Python is missing at $VenvPython")
    }
    # `irm | iex` runs in the caller's shell, so save/restore every handoff variable. Captures sit
    # above the try so a finally never clears a value it never set.
    $previousSkipStudioBase = $env:SKIP_STUDIO_BASE
    $hadPreviousSkipStudioBase = ($null -ne $previousSkipStudioBase)
    # UNSLOTH_STUDIO_HOME only for env-override installs, or llama.cpp lands in the wrong place.
    $previousUnslothStudioHome = $env:UNSLOTH_STUDIO_HOME
    $hadPreviousUnslothStudioHome = ($null -ne $previousUnslothStudioHome)
    $previousTauriMode = $env:UNSLOTH_TAURI_MODE
    $hadPreviousTauriMode = ($null -ne $previousTauriMode)
    $previousSetupRuntimeGateHandoff = $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF
    $hadPreviousSetupRuntimeGateHandoff = ($null -ne $previousSetupRuntimeGateHandoff)
    $previousProxyHandoff = $env:_UNSLOTH_PS_PROXY_DEFAULTS
    $hadPreviousProxyHandoff = ($null -ne $previousProxyHandoff)
    $previousRocmGfxHandoff = $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF
    $hadPreviousRocmGfxHandoff = ($null -ne $previousRocmGfxHandoff)
    # A leaked SKIP_STUDIO_FRONTEND=1 would skip the next source setup's frontend build.
    $previousStudioPackageName = $env:STUDIO_PACKAGE_NAME
    $hadPreviousStudioPackageName = ($null -ne $previousStudioPackageName)
    $previousNoTorch = $env:UNSLOTH_NO_TORCH
    $hadPreviousNoTorch = ($null -ne $previousNoTorch)
    $previousInstallerTorchTag = $env:UNSLOTH_INSTALLER_TORCH_TAG
    $hadPreviousInstallerTorchTag = ($null -ne $previousInstallerTorchTag)
    $previousSkipStudioFrontend = $env:SKIP_STUDIO_FRONTEND
    $hadPreviousSkipStudioFrontend = ($null -ne $previousSkipStudioFrontend)
    $previousStudioLocalInstall = $env:STUDIO_LOCAL_INSTALL
    $hadPreviousStudioLocalInstall = ($null -ne $previousStudioLocalInstall)
    $previousStudioLocalRepo = $env:STUDIO_LOCAL_REPO
    $hadPreviousStudioLocalRepo = ($null -ne $previousStudioLocalRepo)
    $previousLocalLlamaCppDir = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR
    $hadPreviousLocalLlamaCppDir = ($null -ne $previousLocalLlamaCppDir)
    $previousInstallRollbackManaged = $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED
    $hadPreviousInstallRollbackManaged = ($null -ne $previousInstallRollbackManaged)
    $previousSetupPython = $env:UNSLOTH_SETUP_PYTHON
    $hadPreviousSetupPython = ($null -ne $previousSetupPython)
    $previousWoaHasTorchaudio = $env:UNSLOTH_WOA_HAS_TORCHAUDIO
    $hadPreviousWoaHasTorchaudio = ($null -ne $previousWoaHasTorchaudio)
    $previousWoaTorchPrerelease = $env:UNSLOTH_WOA_TORCH_PRERELEASE
    $hadPreviousWoaTorchPrerelease = ($null -ne $previousWoaTorchPrerelease)
    $previousWoaSelectedTorchIndex = $env:UNSLOTH_WOA_SELECTED_TORCH_INDEX
    $hadPreviousWoaSelectedTorchIndex = ($null -ne $previousWoaSelectedTorchIndex)
    $previousWoaPyPIProvided = $env:UNSLOTH_WOA_PYPI_PROVIDED
    $hadPreviousWoaPyPIProvided = ($null -ne $previousWoaPyPIProvided)
    try {
        $env:SKIP_STUDIO_BASE = "1"
        $env:STUDIO_PACKAGE_NAME = $PackageName
        $env:UNSLOTH_NO_TORCH = if ($SkipTorch) { "true" } else { "false" }
        # The torch family THIS run chose, for setup.ps1's preserve guard: the migrated-venv arm never
        # touches torch. Empty = unknown; always assigned so a previous session value cannot leak.
        $env:UNSLOTH_INSTALLER_TORCH_TAG = if ($SkipTorch) { "" } else {
            [string](Get-ExpectedTorchFlavorTag -TorchIndexUrl $TorchIndexUrl -ROCmIndexUrl $ROCmIndexUrl)
        }
        # Windows on ARM: torchaudio pairing, prerelease flag and index. Not UNSLOTH_WOA_TORCH_INDEX_URL,
        # which a later run would read as a pin.
        $env:UNSLOTH_WOA_HAS_TORCHAUDIO = if ($script:WoaNativeCudaTorch -and $script:WoaTorchAudio) { "1" } else { "0" }
        $env:UNSLOTH_WOA_TORCH_PRERELEASE = if ($script:WoaNativeCudaTorch -and $script:WoaTorchIsPrerelease) { "1" } else { "0" }
        if ($script:WoaNativeCudaTorch -and $script:WoaTorchIndexUrl) {
            $env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $script:WoaTorchIndexUrl
        } else {
            Remove-Item Env:UNSLOTH_WOA_SELECTED_TORCH_INDEX -ErrorAction SilentlyContinue
        }
        # Discarded wheelhouse wheels PyPI also serves, recorded for install_python_stack.py.
        $_woaProvidedPairs = @()
        foreach ($_woaProvidedKey in @($script:WoaPyPIProvided.Keys | Sort-Object)) {
            foreach ($_woaProvidedVer in @($script:WoaPyPIProvided[$_woaProvidedKey] | Where-Object { $_ } | Select-Object -Unique)) {
                $_woaProvidedPairs += "$_woaProvidedKey==$_woaProvidedVer"
            }
        }
        $env:UNSLOTH_WOA_PYPI_PROVIDED = ($_woaProvidedPairs -join " ")
        # Tauri bundles its own frontend: skip Node/npm/frontend build.
        $env:SKIP_STUDIO_FRONTEND = if ($TauriMode) { "1" } else { "0" }
        # Always set, so a previous --local run in this session cannot leak.
        if ($StudioLocalInstall) {
            $env:STUDIO_LOCAL_INSTALL = "1"
            $env:STUDIO_LOCAL_REPO = $RepoRoot
        } else {
            $env:STUDIO_LOCAL_INSTALL = "0"
            Remove-Item Env:STUDIO_LOCAL_REPO -ErrorAction SilentlyContinue
        }
        # 'studio setup', not 'update': update pops SKIP_STUDIO_BASE and bypasses the fast path (#4667).
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
        # Reuse the venv Python rather than re-probe (a 3.14 or Store stub on PATH can trip it).
        $env:UNSLOTH_SETUP_PYTHON = Join-Path $VenvDir "Scripts\python.exe"
        # The installer holds the runtime mutex; the child inherits it rather than deadlock.
        $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"
        # Proxy defaults for the -NoProfile child only. Always set: its absence means a standalone
        # update, and an empty object means "none".
        $env:_UNSLOTH_PS_PROXY_DEFAULTS =
            if ($UnslothProxyHandoffJson) { $UnslothProxyHandoffJson } else { '{}' }
        # Forward the resolved arch, or setup expects cpu torch against ROCm wheels and the app retries
        # rollback forever. PRIVATE, not UNSLOTH_ROCM_GFX_ARCH (install_llama_prebuilt.py reads that as a
        # manual override); setup uses it only when its own probes come up empty.
        if ($ROCmGfxArch) {
            $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF = $ROCmGfxArch
        } else {
            # Clear an inherited value: nothing here detected it.
            Remove-Item Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF -ErrorAction SilentlyContinue
        }
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
        if ($hadPreviousSkipStudioBase) {
            $env:SKIP_STUDIO_BASE = $previousSkipStudioBase
        } else {
            Remove-Item Env:SKIP_STUDIO_BASE -ErrorAction SilentlyContinue
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
        if ($hadPreviousStudioPackageName) {
            $env:STUDIO_PACKAGE_NAME = $previousStudioPackageName
        } else {
            Remove-Item Env:STUDIO_PACKAGE_NAME -ErrorAction SilentlyContinue
        }
        if ($hadPreviousNoTorch) {
            $env:UNSLOTH_NO_TORCH = $previousNoTorch
        } else {
            Remove-Item Env:UNSLOTH_NO_TORCH -ErrorAction SilentlyContinue
        }
        if ($hadPreviousInstallerTorchTag) {
            $env:UNSLOTH_INSTALLER_TORCH_TAG = $previousInstallerTorchTag
        } else {
            Remove-Item Env:UNSLOTH_INSTALLER_TORCH_TAG -ErrorAction SilentlyContinue
        }
        if ($hadPreviousSkipStudioFrontend) {
            $env:SKIP_STUDIO_FRONTEND = $previousSkipStudioFrontend
        } else {
            Remove-Item Env:SKIP_STUDIO_FRONTEND -ErrorAction SilentlyContinue
        }
        if ($hadPreviousStudioLocalInstall) {
            $env:STUDIO_LOCAL_INSTALL = $previousStudioLocalInstall
        } else {
            Remove-Item Env:STUDIO_LOCAL_INSTALL -ErrorAction SilentlyContinue
        }
        if ($hadPreviousStudioLocalRepo) {
            $env:STUDIO_LOCAL_REPO = $previousStudioLocalRepo
        } else {
            Remove-Item Env:STUDIO_LOCAL_REPO -ErrorAction SilentlyContinue
        }
        $UnslothProxyHandoffJson = $null
        if ($hadPreviousLocalLlamaCppDir) {
            $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR = $previousLocalLlamaCppDir
        } else {
            Remove-Item Env:UNSLOTH_LOCAL_LLAMA_CPP_DIR -ErrorAction SilentlyContinue
        }
        if ($hadPreviousInstallRollbackManaged) {
            $env:UNSLOTH_INSTALL_ROLLBACK_MANAGED = $previousInstallRollbackManaged
        } else {
            Remove-Item Env:UNSLOTH_INSTALL_ROLLBACK_MANAGED -ErrorAction SilentlyContinue
        }
        if ($hadPreviousSetupPython) {
            $env:UNSLOTH_SETUP_PYTHON = $previousSetupPython
        } else {
            Remove-Item Env:UNSLOTH_SETUP_PYTHON -ErrorAction SilentlyContinue
        }
        foreach ($_woaPair in @(
                @("UNSLOTH_WOA_HAS_TORCHAUDIO", $hadPreviousWoaHasTorchaudio, $previousWoaHasTorchaudio),
                @("UNSLOTH_WOA_TORCH_PRERELEASE", $hadPreviousWoaTorchPrerelease, $previousWoaTorchPrerelease),
                @("UNSLOTH_WOA_SELECTED_TORCH_INDEX", $hadPreviousWoaSelectedTorchIndex, $previousWoaSelectedTorchIndex),
                @("UNSLOTH_WOA_PYPI_PROVIDED", $hadPreviousWoaPyPIProvided, $previousWoaPyPIProvided))) {
            if ($_woaPair[1]) { Set-Item "Env:$($_woaPair[0])" $_woaPair[2] }
            else { Remove-Item "Env:$($_woaPair[0])" -ErrorAction SilentlyContinue }
        }
    }
    # $null: Application Control refused the process. Checked first since $null -ne 0 is true.
    if ($null -eq $setupExit) {
        return (Exit-InstallFailure (Write-ApplicationControlBlocked -Path $VenvPython))
    }
    # Clear the handoff so a later update cannot re-pin; warn, never abort, on a series change.
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

    # ── Shim dir holds only unsloth.exe; venv Scripts\python.exe on PATH would hijack the system. ──
    # Remove the legacy Scripts PATH entry older installers wrote.
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
    # Outside the lock try/catch: a directory here must not degrade to "existing launcher".
    if (Test-Path -LiteralPath $ShimExe -PathType Container) {
        Write-StudioLine "[ERROR] Cannot create unsloth launcher: $ShimExe is a directory." -ForegroundColor Red
        Write-StudioLine "        Move or remove it manually, then re-run the installer." -ForegroundColor Yellow
        throw "Cannot create unsloth launcher: $ShimExe is a directory."
    }
    # Locked unsloth.exe (Unsloth running): keep the old shim.
    $shimUpdated = $false
    try {
        if (Test-Path -LiteralPath $ShimExe) { Remove-Item -LiteralPath $ShimExe -Force -ErrorAction Stop }
        try {
            # New-Item -ItemType HardLink takes no -LiteralPath, so brackets in $ShimExe glob-expand.
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
            # The interpreter: this arm runs where policy denies the console script.
            Write-StudioLine "       Until the next successful install, start Unsloth with:" -ForegroundColor Yellow
            Write-StudioLine "       & '$VenvPython' -I -m unsloth_cli studio -p 8888" -ForegroundColor Yellow
        }
    }
    # For machines whose policy denies the .exe; PATHEXT still prefers .EXE elsewhere.
    Write-UnslothCmdShim -ShimDir $ShimDir -PythonPath $VenvPython

    # PATH only when a launcher exists; the .cmd alone counts (the .exe may have been removed).
    # Env-mode: session-only. Test-UnslothCmdShimFile, not Test-Path, so a foreign unsloth.cmd that
    # survived this run is not advertised.
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
            Restore-StudioUvCacheMarker -StudioRoot $StudioHome
        }
    }

    # After Refresh-SessionPath, or a legacy User PATH entry would win.
    if ($StudioRedirectMode -eq 'env' -and $ShimUsable) {
        $env:Path = "$ShimDir;$env:Path"
        step "path" "exported $ShimDir for this session (no registry PATH change in env-override mode)"
    }

    if ($TauriMode) {
        Write-TauriLog "DONE" ""
        return
    }

    New-StudioShortcuts -ManagedPythonPath $VenvPython -DestinationValidated

    try {
        $_pathCmd = Get-Command unsloth -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($_pathCmd) {
            $_pathExe = $_pathCmd.Source
            # Our own unsloth.cmd is not foreign; hashing it against a PE would always warn.
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
                # That path is the console script a policy denies; name the launcher that works.
                if (Test-ShimLaunchBlocked -Path $ShimExe) {
                    substep "(Windows blocks that file here; use $_ourCmdShim instead)"
                }
                Write-StudioLine ""
            }
        }
    } catch {
        # Diagnostic only; never block install on a probe failure.
    }

    $IsInteractive = (-not $SkipAutostart) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)
    if ($IsInteractive) {
        Write-StudioLine ""
        $reply = Read-Host "  Start Unsloth Studio now? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($reply) -or $reply -match '^[Yy]') {
            # Keep both locks until the process exists, so a second installer's scan sees Unsloth.
            $_runtimeGateHandoff = $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF
            try {
                $env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"
                # Through the interpreter, so autostart is not refused by Application Control.
                Set-StudioUvCacheForLaunch -StudioRoot $StudioHome

                $studioAutoStartProcess = Start-Process -FilePath $VenvPython `
                    -ArgumentList (Get-ManagedUnslothCliCommandLine -Arguments @("studio", "-p", "8888")) `
                    -NoNewWindow -PassThru
                # It inherits the private %TEMP% and outlives the installer, so it is the owner the next sweep sees.
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
            # PATHEXT prefers the .exe a denying machine cannot run. Probed, so others see unchanged text.
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
        # Single-quote printed paths so $-vars in custom roots survive a copy-paste.
        $_actLiteral = "'" + ((Join-Path $VenvDir "Scripts\Activate.ps1") -replace "'", "''") + "'"
        # Activation puts Scripts\unsloth.exe first; same probe so unaffected machines see unchanged text.
        $_shimBlocked = Test-UnslothCmdShimPreferred -ShimExe $ShimExe -ShimCmd $ShimCmd
        $_bareLaunch = if ($_shimBlocked) { "unsloth.cmd studio -p 8888" } else { "unsloth studio -p 8888" }
        if ($StudioRedirectMode -eq 'env') {
            # Env-mode skips registry PATH: print the absolute shim path, the .cmd only when needed.
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
        Restore-StudioUvCacheEnvironment -WasPresent $hadPreviousUvCacheDir -PreviousValue $previousUvCacheDir
        for ($i = $studioRuntimeMutexes.Count - 1; $i -ge 0; $i--) {
            Exit-StudioInstallMutex -Mutex $studioRuntimeMutexes[$i]
        }
        Exit-StudioInstallLock -Lock $studioInstallLock
        # Matters for `irm | iex`, where these are the user's own session variables.
        Restore-StudioTempEnvironment
    }
    if ($null -ne $studioAutoStartProcess) {
        $studioAutoStartProcess.WaitForExit()
    }
}

# Under `irm | iex` the script scope IS the caller's session; an earlier value must not leak.
$script:WoaResolverEnvSaved = $null
$script:MirrorEnvSaved = @{}
$script:InstallTorchMirror = $null
$script:WoaSessionOverrides = $null
$script:TorchOverridesFile = $null
try {
    Install-UnslothStudio @args
} finally {
    # Restore the caller's resolver variables, or their own `uv pip` keeps Studio's overrides.
    if ($script:WoaResolverEnvSaved) {
        foreach ($_woaEnvName in @($script:WoaResolverEnvSaved.Keys)) {
            $_woaEnvValue = $script:WoaResolverEnvSaved[$_woaEnvName]
            if ($null -eq $_woaEnvValue) { Remove-Item "Env:$_woaEnvName" -ErrorAction SilentlyContinue }
            else { Set-Item "Env:$_woaEnvName" $_woaEnvValue }
        }
        $script:WoaResolverEnvSaved = $null
    }
    # Guarded like the block above: @($null.Keys) is one $null item, and indexing it would stop this finally before the cleanup below.
    if ($script:MirrorEnvSaved) {
        foreach ($_mirrorEnvName in @($script:MirrorEnvSaved.Keys)) {
            $_mirrorEnvValue = $script:MirrorEnvSaved[$_mirrorEnvName]
            if ($null -eq $_mirrorEnvValue) { Remove-Item "Env:$_mirrorEnvName" -ErrorAction SilentlyContinue }
            else { Set-Item "Env:$_mirrorEnvName" $_mirrorEnvValue }
        }
    }
    # UNSLOTH_KEPT_TORCH is a process-scoped handoff, and the session outlives the installer.
    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
    # The generated overrides file copies the caller's UV_OVERRIDE contents; never leave it.
    if ($script:TorchOverridesFile) {
        Remove-UnslothTempFileQuietly -Path $script:TorchOverridesFile
        $script:TorchOverridesFile = $null
    }
    if ($script:WoaSessionOverrides) {
        Remove-UnslothTempFileQuietly -Path $script:WoaSessionOverrides
        $script:WoaSessionOverrides = $null
    }
}
