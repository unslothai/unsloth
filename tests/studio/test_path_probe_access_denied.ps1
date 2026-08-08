# Regression tests for denied setup.ps1 install trees.
# Run the real probes against chmod on POSIX and icacls deny on Windows.
$ErrorActionPreference = "Stop"
$script:failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")
. ([System.IO.Path]::Combine($repoRoot, "tests", "studio_setup_ps1", "Get-FunctionSource.ps1"))

foreach ($fn in @("Test-AccessDeniedError", "Get-PathState", "Test-PathQuiet",
                  "Get-LlamaCppInstallReadState", "Get-PathDenialDetail",
                  "Get-StudioAdoptableState", "Test-StudioOwnedAdoptable")) {
    $src = Get-FunctionSource -Path $setupPath -Name $fn
    Check "setup.ps1 defines $fn" ($null -ne $src)
    if ($src) { . ([scriptblock]::Create($src)) }
}

# ── Source contract: the crash site must probe state, not bare Test-Path ──
$setupText = Get-Content -Raw -LiteralPath $setupPath
Check "prebuilt metadata probe no longer uses a bare Test-Path" (
    $setupText -notmatch '\n\s*if \(Test-Path \$existingMetaPath\)')
Check "prebuilt metadata probe goes through a non-throwing helper" (
    $setupText -match 'Test-PathQuiet -Path \$existingMetaPath -PathType Leaf')
# Get-LlamaCppInstallReadState decides denial before this marker probe.
Check "a denied llama.cpp install fails with an actionable message" (
    $setupText -match '\$llamaDirState -eq "Denied"' -and
    $setupText -match 'return "Access denied reading the existing \$Label')
# Shared reporting returns a reason instead of exiting directly.
Check "Exit-PathAccessDenied delegates the wording to Write-PathAccessDenied" (
    $setupText -match 'Exit-SetupFailure \(Write-PathAccessDenied -Path \$Path -Label \$Label')
# Every denial route must report instead of proceeding.
Check "the prebuilt phase stops on a denied llama.cpp dir" (
    $setupText -match '\$llamaDirState = Get-LlamaCppInstallReadState -Path \$LlamaCppDir' -and
    $setupText -match '\$llamaDirState -eq "Denied"')
$setupPreflightCall = '$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight'
$setupPreflightAt = $setupText.IndexOf($setupPreflightCall)
$setupPhaseOneAt = $setupText.IndexOf('PHASE 1: System-level prerequisites')
Check "direct setup/update preflight runs exactly once before phase 1" (
    $setupPreflightAt -ge 0 -and
    $setupPreflightAt -lt $setupPhaseOneAt -and
    $setupText.IndexOf($setupPreflightCall, $setupPreflightAt + 1) -eq -1)
Check "direct setup/update resolves one managed path and reuses it in phase 3.4" (
    $setupText -match '\$LlamaCppDir = Get-ManagedLlamaCppDir' -and
    $setupText -match '\$UnslothHome = Split-Path -Parent \$LlamaCppDir' -and
    $setupText -notmatch '\$LlamaCppDir = Join-Path \$UnslothHome')
Check "the source-build .git probe stops on a denied checkout" (
    $setupText -match '\$llamaGitState = Get-PathState -Path \(Join-Path \$LlamaCppDir "\.git"\)' -and
    $setupText -match '\$llamaGitState -eq "Denied"')
Check "the ownership guard stops on a denied root instead of returning" (
    $setupText -match '\$pathState = Get-PathState -Path \$Path -PathType Container' -and
    $setupText -match '\$StudioHomeIsCustom -and \$pathState -eq "Denied"')
Check "guidance says an app reinstall does not reset the folder" (
    $setupText -match 'reinstalling Unsloth Studio, to any drive, reuses it' -and
    $setupText -match 'Reinstalling the app does not reset it\.')
Check "guidance names the concrete recovery commands" (
    $setupText -match 'takeown /F' -and $setupText -match 'icacls .* /reset /T')
Check "whisper marker probe cannot terminate the non-fatal whisper phase" (
    $setupText -match 'if \(Test-PathQuiet \$llamaMarker "Leaf"\)')

# ── Behaviour against a real unreadable directory ──
$root = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_acl_" + [guid]::NewGuid().ToString("N"))
$locked = Join-Path $root "llama.cpp"
New-Item -ItemType Directory -Force -Path $locked | Out-Null
$meta = Join-Path $locked "UNSLOTH_PREBUILT_INFO.json"
Set-Content -LiteralPath $meta -Value '{"release_tag":"app-1","published_repo":"unslothai/llama.cpp"}'

$onWindows = ($env:OS -eq "Windows_NT")
function Set-Denied([bool]$on) {
    if ($onWindows) {
        $who = "$env:USERDOMAIN\$env:USERNAME"
        if ($on) { icacls $locked /deny "${who}:(OI)(CI)(RX)" *>$null }
        else { icacls $locked /remove:d "$who" *>$null }
    } else {
        if ($on) { chmod 000 $locked } else { chmod 755 $locked }
    }
}

try {
    Check "readable metadata reports Present" ((Get-PathState -Path $meta -PathType Leaf) -eq "Present")
    Check "readable install is adoptable" (Test-StudioOwnedAdoptable $locked)
    Check "absent path reports Absent" (
        (Get-PathState -Path (Join-Path $root "missing.json") -PathType Leaf) -eq "Absent")
    Check "a readable llama.cpp install reads Readable" (
        (Get-LlamaCppInstallReadState -Path $locked) -eq "Readable")
    Check "a missing llama.cpp install reads Absent" (
        (Get-LlamaCppInstallReadState -Path (Join-Path $root "missing")) -eq "Absent")

    Set-Denied $true

    # Require a real denial so the assertions cannot pass vacuously as root.
    $oldFormTerminated = $false
    try { $null = Test-Path $meta } catch { $oldFormTerminated = $true }

    if (-not $oldFormTerminated) {
        Write-Host "  SKIP  cannot deny access on this host (running as root/admin?) -- behaviour checks skipped" -ForegroundColor Yellow
    } else {
        Check "bare Test-Path still terminates on a denied path (negative control)" $oldFormTerminated
        $state = $null
        $threw = $false
        try { $state = Get-PathState -Path $meta -PathType Leaf } catch { $threw = $true }
        Check "Get-PathState does not terminate on a denied path" (-not $threw)
        Check "Get-PathState reports Denied (not Absent)" ($state -eq "Denied")

        $quiet = $null
        $threw = $false
        try { $quiet = Test-PathQuiet $meta } catch { $threw = $true }
        Check "Test-PathQuiet does not terminate on a denied path" (-not $threw)
        Check "Test-PathQuiet reports the path as unusable" ($quiet -eq $false)

        $threw = $false
        $adoptable = $null
        try { $adoptable = Test-StudioOwnedAdoptable $locked } catch { $threw = $true }
        Check "Test-StudioOwnedAdoptable does not terminate on a denied tree" (-not $threw)
        Check "Test-StudioOwnedAdoptable cannot adopt an unreadable tree" ($adoptable -eq $false)

        # This is the state the early installer preflight must recognize.
        $treeState = $null
        $threw = $false
        try { $treeState = Get-LlamaCppInstallReadState -Path $locked } catch { $threw = $true }
        Check "Get-LlamaCppInstallReadState does not terminate on a denied tree" (-not $threw)
        Check "Get-LlamaCppInstallReadState reports Denied" ($treeState -eq "Denied")
    }
} finally {
    Set-Denied $false
    Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
}

# ── A denied tree with NO metadata inside it ──
# A missing child of a denied Windows dir looks absent, so the listing must catch
# it. POSIX mode 111 is the same shape.
$bareRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_bare_" + [guid]::NewGuid().ToString("N"))
$bareLocked = Join-Path $bareRoot "llama.cpp"
New-Item -ItemType Directory -Force -Path (Join-Path $bareLocked "build") | Out-Null
$bareModes = if ($onWindows) { @("acl") } else { @("000", "111") }
foreach ($mode in $bareModes) {
    if ($mode -eq "acl") { icacls $bareLocked /deny "$env:USERDOMAIN\${env:USERNAME}:(OI)(CI)(RX)" *>$null }
    else { chmod $mode $bareLocked }
    try {
        # Ensure the listing, not the absent marker, decides this case.
        $metaProbe = Get-PathState -Path (Join-Path $bareLocked "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf
        $state = $null
        $threw = $false
        try { $state = Get-LlamaCppInstallReadState -Path $bareLocked } catch { $threw = $true }
        Check "a denied tree with no metadata does not terminate the probe ($mode)" (-not $threw)
        if ($metaProbe -eq "Absent") {
            Check "the listing catches a denied tree the metadata probe calls absent ($mode)" (
                $state -eq "Denied")
        } else {
            Check "a denied tree with no metadata still reports Denied ($mode)" ($state -eq "Denied")
        }
    } finally {
        if ($mode -eq "acl") { icacls $bareLocked /remove:d "$env:USERDOMAIN\$env:USERNAME" *>$null }
        else { chmod 755 $bareLocked }
    }
}
Remove-Item -Recurse -Force -LiteralPath $bareRoot -ErrorAction SilentlyContinue

# ── A readable marker inside a tree that cannot be listed ──
# What a marker-only probe would call Readable. Windows denies ReadData; POSIX
# mode 111 keeps the named child stat-able.
$listRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_list_" + [guid]::NewGuid().ToString("N"))
$listLocked = Join-Path $listRoot "llama.cpp"
New-Item -ItemType Directory -Force -Path $listLocked | Out-Null
Set-Content -LiteralPath (Join-Path $listLocked "UNSLOTH_PREBUILT_INFO.json") -Value '{"release_tag":"app-1"}'
if ($onWindows) { icacls $listLocked /deny "$env:USERDOMAIN\${env:USERNAME}:(RD)" *>$null }
else { chmod 111 $listLocked }
try {
    $markerReadable = ((Get-PathState -Path (Join-Path $listLocked "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf) -eq "Present")
    $listDenied = $false
    try { $null = Get-ChildItem -LiteralPath $listLocked -Force -ErrorAction Stop } catch { $listDenied = $true }
    if ($markerReadable -and $listDenied) {
        Check "a readable marker does not excuse an unlistable tree" (
            (Get-LlamaCppInstallReadState -Path $listLocked) -eq "Denied")
    } else {
        Write-Host "  SKIP  host cannot deny listing while keeping the marker readable"
    }
} finally {
    if ($onWindows) { icacls $listLocked /remove:d "$env:USERDOMAIN\$env:USERNAME" *>$null }
    else { chmod 755 $listLocked }
    Remove-Item -Recurse -Force -LiteralPath $listRoot -ErrorAction SilentlyContinue
}

# ── Test-Path parity: the regression-safety invariant ──
# Preserve every non-throwing Test-Path result; only thrown probes may be denied.
$parityRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_par_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path (Join-Path $parityRoot "tree/sub") | Out-Null
Set-Content -LiteralPath (Join-Path $parityRoot "tree/UNSLOTH_PREBUILT_INFO.json") -Value "{}"
Set-Content -LiteralPath (Join-Path $parityRoot "tree/sub/file.txt") -Value "x"
$parityProbes = @($parityRoot, (Join-Path $parityRoot "tree"), (Join-Path $parityRoot "tree/sub"),
    (Join-Path $parityRoot "tree/UNSLOTH_PREBUILT_INFO.json"), (Join-Path $parityRoot "tree/sub/file.txt"),
    (Join-Path $parityRoot "missing"), (Join-Path $parityRoot "missing/deeper.json"))
$mismatch = 0; $deniedWithoutThrow = 0; $probed = 0
foreach ($p in $parityProbes) {
    foreach ($t in @("Any", "Leaf", "Container")) {
        $old = $null; $threw = $false
        try { $old = [bool](Test-Path -LiteralPath $p -PathType $t -ErrorAction Stop) } catch { $threw = $true }
        $new = Get-PathState -Path $p -PathType $t
        $probed++
        if ($threw) { if ($new -notin @("Denied", "Absent")) { $mismatch++ } }
        elseif ($new -eq "Denied") { $deniedWithoutThrow++ }
        elseif ($old -ne ($new -eq "Present")) { $mismatch++ }
    }
}
Remove-Item -Recurse -Force -LiteralPath $parityRoot -ErrorAction SilentlyContinue
Check "Get-PathState matches bare Test-Path on every non-throwing probe ($probed)" ($mismatch -eq 0)
Check "Denied never appears where the old probe did not throw" ($deniedWithoutThrow -eq 0)

# ── A denied marker FILE under a readable directory ──
# Windows only (POSIX keeps mode-000 files stat-able): a denied marker must not be
# mistaken for an unrelated directory.
$adoptRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_adopt_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $adoptRoot | Out-Null
$adoptMarker = Join-Path $adoptRoot "UNSLOTH_PREBUILT_INFO.json"
Set-Content -LiteralPath $adoptMarker -Value '{"release_tag":"app-1"}'
Check "a readable marker reports Yes" ((Get-StudioAdoptableState -Path $adoptRoot) -eq "Yes")
if ($onWindows) {
    $who = "$env:USERDOMAIN\$env:USERNAME"
    icacls $adoptMarker /deny "${who}:(R)" *>$null
    try {
        $markerThrew = $false
        try { $null = Test-Path -LiteralPath $adoptMarker -PathType Leaf -ErrorAction Stop } catch { $markerThrew = $true }
        if ($markerThrew) {
            Check "a denied marker reports Denied, not No" ((Get-StudioAdoptableState -Path $adoptRoot) -eq "Denied")
            Check "the boolean view still refuses to adopt it" (-not (Test-StudioOwnedAdoptable $adoptRoot))
        } else {
            Write-Host "  SKIP  this host would not deny the marker file" -ForegroundColor Yellow
        }
    } finally { icacls $adoptMarker /remove:d "$who" *>$null }
} else {
    Write-Host "  SKIP  denied marker file is a Windows-ACL-only state (POSIX keeps mode-000 files stat-able)" -ForegroundColor Yellow
}
Remove-Item -Recurse -Force -LiteralPath $adoptRoot -ErrorAction SilentlyContinue
Check "a missing marker reports No" ((Get-StudioAdoptableState -Path ([System.IO.Path]::GetTempPath())) -eq "No")

# ── The reporting path must not itself fail ──
# Reporting must tolerate an empty path without masking the original failure.
foreach ($edge in @($null, "", "   ")) {
    $edgeOk = $true
    try { $null = Get-PathDenialDetail -Path $edge } catch { $edgeOk = $false }
    Check "Get-PathDenialDetail tolerates an empty/null path" $edgeOk
}

# ── The desktop app must receive the reason, not just "exit code 1" ──
# Exercise the real Tauri failure path, which prefers [TAURI:ERROR] details.
$exitDeniedSrc = Get-FunctionSource -Path $setupPath -Name Exit-PathAccessDenied
$writeDeniedSrc = Get-FunctionSource -Path $setupPath -Name Write-PathAccessDenied
$exitSetupSrc = Get-FunctionSource -Path $setupPath -Name Exit-SetupFailure
Check "setup.ps1 defines Exit-PathAccessDenied" ($null -ne $exitDeniedSrc)
Check "setup.ps1 defines Write-PathAccessDenied" ($null -ne $writeDeniedSrc)
if ($exitDeniedSrc -and $writeDeniedSrc) {
    $harness = @"
`$ErrorActionPreference = "Stop"
function step { param([string]`$Label, [string]`$Value, [string]`$Color = "Green") Write-Host "  `$Label  `$Value" }
function substep { param([string]`$Message, [string]`$Color = "DarkGray") Write-Host "    `$Message" }
function Get-PathDenialDetail { param([string]`$Path) return "" }
$exitSetupSrc
$writeDeniedSrc
$exitDeniedSrc
Exit-PathAccessDenied -Path "C:\Users\test\.unsloth\llama.cpp" -Label "llama.cpp install"
Write-Host "REACHED_UNREACHABLE"
"@
    $harnessFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_denied_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $harnessFile -Value $harness -Encoding utf8
    $pwshExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $pwshExe) { $pwshExe = (Get-Command powershell).Source }
    $savedMode = $env:UNSLOTH_TAURI_MODE
    try {
        $env:UNSLOTH_TAURI_MODE = "1"
        $out = & $pwshExe -NoProfile -File $harnessFile 2>&1 | Out-String
        $code = $LASTEXITCODE
    } finally {
        if ($null -eq $savedMode) { Remove-Item Env:UNSLOTH_TAURI_MODE -ErrorAction SilentlyContinue }
        else { $env:UNSLOTH_TAURI_MODE = $savedMode }
        Remove-Item -LiteralPath $harnessFile -ErrorAction SilentlyContinue
    }
    Check "the denial stops setup (exit 1)" ($code -eq 1)
    Check "the denial does not fall through to the install" ($out -notmatch "REACHED_UNREACHABLE")
    Check "the desktop app gets a [TAURI:ERROR] reason, not a bare exit code" (
        $out -match '\[TAURI:ERROR\] Access denied reading the existing llama\.cpp install')
    Check "the reason names the folder to remove" ($out -match [regex]::Escape('C:\Users\test\.unsloth\llama.cpp'))
    Check "the reason says a reinstall will not help" ($out -match 'Reinstalling the app does not reset it')
    # Keep takeown and icacls on separate copy-pasteable lines.
    $takeownLines = @($out -split "`r?`n" | Where-Object { $_ -match 'takeown /F' })
    Check "takeown is printed on its own line" ($takeownLines.Count -eq 1)
    Check "icacls is not appended to the takeown line" (
        $takeownLines.Count -eq 1 -and $takeownLines[0] -notmatch 'icacls')
    Check "icacls is printed on its own line" (
        @($out -split "`r?`n" | Where-Object { $_ -match 'icacls .* /reset /T' }).Count -eq 1)
}

# ── The reporter must return one string, not a pipeline ──
# The shared reporter must return one value even when Tauri stdout mirroring runs.
$mirrorFns = @("Get-StudioAnsi", "Write-StudioStdoutMirror", "step", "substep", "Write-PathAccessDenied")
$mirrorSrc = @()
foreach ($fn in $mirrorFns) {
    $src = Get-FunctionSource -Path $setupPath -Name $fn
    Check "setup.ps1 defines $fn" ($null -ne $src)
    if ($src) { $mirrorSrc += $src }
}
if ($mirrorSrc.Count -eq $mirrorFns.Count) {
    $mirrorHarness = @"
`$ErrorActionPreference = "Stop"
function Get-PathDenialDetail { param([string]`$Path) return "" }
$($mirrorSrc -join "`n")
`$script:StudioVtOk = `$false
Add-Content -LiteralPath `$args[0] -Value "REDIRECTED=`$([Console]::IsOutputRedirected)"
foreach (`$mode in @(@{}, @{OwnershipUnverified=`$true}, @{UserSupplied=`$true})) {
    `$out = @(Write-PathAccessDenied -Path "C:\Users\t\.unsloth\llama.cpp" -Label "llama.cpp install" @mode)
    Add-Content -LiteralPath `$args[0] -Value "EMITTED=`$(`$out.Count)/`$(`$out[0].GetType().Name)"
}
"@
    $mirrorFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_mir_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $mirrorFile -Value $mirrorHarness -Encoding utf8
    $pwshExe3 = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $pwshExe3) { $pwshExe3 = (Get-Command powershell).Source }
    $mirrorOutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_mir_" + [guid]::NewGuid().ToString("N") + ".txt")
    $mirrorResFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_res_" + [guid]::NewGuid().ToString("N") + ".txt")
    $lines = @()
    try {
        # Redirect stdout to activate the mirror; return verdicts in another file.
        & $pwshExe3 -NoProfile -File $mirrorFile $mirrorResFile *> $mirrorOutFile
        if (Test-Path -LiteralPath $mirrorResFile) { $lines = @(Get-Content -LiteralPath $mirrorResFile) }
    } finally {
        Remove-Item -LiteralPath $mirrorFile -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $mirrorOutFile -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $mirrorResFile -ErrorAction SilentlyContinue
    }
    $emitted = @($lines | Where-Object { $_ -match "^EMITTED=" })
    Check "the reporter ran with its stdout redirected, so the mirror is live" (
        $lines -contains "REDIRECTED=True")
    Check "the reporter ran in all three modes" ($emitted.Count -eq 3)
    Check "each mode returns exactly one string, not a pipeline" (
        $emitted.Count -eq 3 -and @($emitted | Where-Object { $_ -ne "EMITTED=1/String" }).Count -eq 0)
}

# ── Denial classification ──
Check "UnauthorizedAccessException classifies as access denied" (
    Test-AccessDeniedError ([System.UnauthorizedAccessException]::new("denied")))
Check "a wrapped UnauthorizedAccessException classifies as access denied" (
    Test-AccessDeniedError ([System.Exception]::new("outer", [System.UnauthorizedAccessException]::new("denied"))))
Check "an unrelated exception does not classify as access denied" (
    -not (Test-AccessDeniedError ([System.IO.FileNotFoundException]::new("missing"))))

# -- Assert-StudioOwnedOrAbsent -NonFatal, run for real --
# Whisper denial remains nonfatal while other ownership failures remain fatal.
$assertSrc = Get-FunctionSource -Path $setupPath -Name Assert-StudioOwnedOrAbsent
$markSrc = Get-FunctionSource -Path $setupPath -Name Mark-StudioOwned
Check "setup.ps1 defines Assert-StudioOwnedOrAbsent" ($null -ne $assertSrc)
if ($assertSrc -and $markSrc) {
    . ([scriptblock]::Create($assertSrc))
    . ([scriptblock]::Create($markSrc))
    # Reaching either of these is the failure the -NonFatal mode exists to avoid.
    function Exit-PathAccessDenied { param($Path, $Label, [switch]$UserSupplied, [switch]$OwnershipUnverified) throw "EXIT-DENIED" }
    function Exit-SetupFailure { param($Message, $Code) throw "EXIT-SETUP" }
    function step { param($a, $b, $c) }
    function substep { param($a, $b) }
    $StudioOwnedMarker = ".unsloth-studio-owned"
    $StudioHomeIsCustom = $true

    $nfRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_nf_" + [guid]::NewGuid().ToString("N"))
    $nfDenied = Join-Path $nfRoot "whisper.cpp"
    $nfUnowned = Join-Path $nfRoot "unowned"
    $nfInner = Join-Path $nfDenied "inner"
    $nfMarker = Join-Path $nfDenied $StudioOwnedMarker
    # Populate first so the denial negative control probes an existing child.
    New-Item -ItemType Directory -Force -Path $nfInner | Out-Null
    Set-Content -LiteralPath $nfMarker -Value ""
    New-Item -ItemType Directory -Force -Path $nfUnowned | Out-Null
    Set-Content -LiteralPath (Join-Path $nfUnowned "someone-elses.txt") -Value "x"
    function Set-NfDenied([bool]$on) {
        if ($onWindows) {
            $who = "$env:USERDOMAIN\$env:USERNAME"
            if ($on) { icacls $nfDenied /deny "${who}:(OI)(CI)(RX)" *>$null }
            else { icacls $nfDenied /remove:d "$who" *>$null }
        } else {
            if ($on) { chmod 000 $nfDenied } else { chmod 755 $nfDenied }
        }
    }
    try {
        Set-NfDenied $true
        # Same environment gate as above: no real denial means no real test.
        $nfReal = $false
        try { $null = Test-Path $nfMarker } catch { $nfReal = $true }
        Check "the host can actually deny a read (negative control)" $nfReal
        if ($nfReal) {
            $threw = $false
            $out = $null
            try { $out = @(Assert-StudioOwnedOrAbsent -Path $nfDenied -Label "whisper.cpp install" -NonFatal) }
            catch { $threw = $true }
            Check "-NonFatal hands a denied tree back instead of exiting" (-not $threw)
            # One bare string, not an array: a stray emit would break the caller's -eq.
            Check "-NonFatal returns exactly one value" ($out.Count -eq 1)
            Check "-NonFatal returns Denied" ($out.Count -eq 1 -and $out[0] -eq "Denied")

            $threw = $false
            try { $null = Assert-StudioOwnedOrAbsent -Path $nfDenied -Label "whisper.cpp install" } catch { $threw = $true }
            Check "without -NonFatal a denied tree still stops setup" $threw

            # Lock the parent to cover a denied directory probe too.
            $out = $null
            $threw = $false
            try { $out = @(Assert-StudioOwnedOrAbsent -Path $nfInner -Label "whisper.cpp install" -NonFatal) }
            catch { $threw = $true }
            Check "-NonFatal hands back a tree whose parent is unreadable" (
                -not $threw -and $out.Count -eq 1 -and $out[0] -eq "Denied")

            # A fresh custom home with no marker. Mode 111 allows child stat but
            # denies listing, matching Windows here.
            $bareWho = "$env:USERDOMAIN\$env:USERNAME"
            $bareModes = if ($onWindows) { @("acl") } else { @("000", "111") }
            foreach ($mode in $bareModes) {
                $nfBare = Join-Path $nfRoot "bare_$mode"
                New-Item -ItemType Directory -Force -Path (Join-Path $nfBare "sub") | Out-Null
                if ($mode -eq "acl") { icacls $nfBare /deny "${bareWho}:(OI)(CI)(RX)" *>$null }
                else { chmod $mode $nfBare }
                try {
                    $out = $null
                    $threw = $false
                    try { $out = @(Assert-StudioOwnedOrAbsent -Path $nfBare -Label "whisper.cpp install" -NonFatal) }
                    catch { $threw = $true }
                    Check "-NonFatal hands back a denied tree that has no marker ($mode)" (
                        -not $threw -and $out.Count -eq 1 -and $out[0] -eq "Denied")
                } finally {
                    if ($mode -eq "acl") { icacls $nfBare /remove:d "$bareWho" *>$null }
                    else { chmod 755 $nfBare }
                }
            }
        }
        # -NonFatal rescues the denial only: someone else's folder must still stop.
        $threw = $false
        try { $null = Assert-StudioOwnedOrAbsent -Path $nfUnowned -Label "whisper.cpp install" -NonFatal } catch { $threw = $true }
        Check "-NonFatal does not excuse an unowned tree" $threw
    } finally {
        Set-NfDenied $false
        Remove-Item -Recurse -Force -LiteralPath $nfRoot -ErrorAction SilentlyContinue
    }
}

# ── install.ps1's preflight, run for real ──
# Its copied helpers run in a child so they do not shadow setup.ps1's, and must
# return actionable guidance for a real denied tree.
$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")
$preflightFns = @("Test-AccessDeniedError", "Get-PathState", "Get-LlamaCppInstallReadState",
                  "Get-PathDenialDetail", "Write-PathAccessDenied", "Get-CanonicalDir",
                  "Test-StudioHomeIsCustom", "Get-ManagedLlamaCppDir",
                  "Invoke-ManagedLlamaCppPreflight")
$preflightSrc = @()
foreach ($fn in $preflightFns) {
    $src = Get-FunctionSource -Path $installPath -Name $fn
    Check "install.ps1 defines $fn" ($null -ne $src)
    if ($src) { $preflightSrc += $src }
}
if ($preflightSrc.Count -eq $preflightFns.Count) {
    $preflightHarness = @"
`$ErrorActionPreference = "Stop"
function step { param([string]`$Label, [string]`$Value, [string]`$Color = "Green") Write-Host "  `$Label  `$Value" }
function substep { param([string]`$Message, [string]`$Color = "DarkGray") Write-Host "    `$Message" }
$($preflightSrc -join "`n")
`$env:USERPROFILE = `$args[0]
`$onWindows = (`$args[1] -eq "win")
`$StudioHome = Join-Path `$env:USERPROFILE ".unsloth\studio"
`$dir = Join-Path `$env:USERPROFILE ".unsloth\llama.cpp"
New-Item -ItemType Directory -Force -Path `$StudioHome | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path `$dir "build") | Out-Null
Set-Content -LiteralPath (Join-Path `$dir "UNSLOTH_PREBUILT_INFO.json") -Value '{"release_tag":"app-1"}'
Write-Host "RESOLVED_DIR: `$(Get-ManagedLlamaCppDir)"
`$readable = Invoke-ManagedLlamaCppPreflight
Write-Host "READABLE_VERDICT: `$(if (`$null -eq `$readable) { "continue" } else { "stop" })"
if (`$onWindows) { icacls `$dir /deny "`$env:USERDOMAIN\`${env:USERNAME}:(OI)(CI)(RX)" *>`$null }
else { chmod 000 `$dir }
`$oldFormTerminated = `$false
try { `$null = Test-Path (Join-Path `$dir "UNSLOTH_PREBUILT_INFO.json") } catch { `$oldFormTerminated = `$true }
Write-Host "CAN_DENY: `$oldFormTerminated"
# Both override forms must switch to user-supplied wording.
if (`$args[2] -eq "supplied") { `$WithLlamaCppDir = `$dir }
if (`$args[2] -eq "env") { `$env:UNSLOTH_LOCAL_LLAMA_CPP_DIR = `$dir }
`$denied = Invoke-ManagedLlamaCppPreflight
Write-Host "DENIED_VERDICT: `$(if (`$null -eq `$denied) { "continue" } else { "stop" })"
Write-Host "DENIED_REASON: `$denied"
if (`$onWindows) { icacls `$dir /remove:d "`$env:USERDOMAIN\`$env:USERNAME" *>`$null }
else { chmod 755 `$dir }
"@
    $preflightFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_pre_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $preflightFile -Value $preflightHarness -Encoding utf8
    $pwshExe2 = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $pwshExe2) { $pwshExe2 = (Get-Command powershell).Source }
    # Use one child per mode so command-line counts remain unambiguous.
    $preflightRuns = @{}
    $preflightHome = ""
    try {
        foreach ($mode in @("managed", "supplied", "env")) {
            $runHome = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_home_" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType Directory -Force -Path $runHome | Out-Null
            if ($mode -eq "managed") { $preflightHome = $runHome }
            try {
                $preflightRuns[$mode] = & $pwshExe2 -NoProfile -File $preflightFile $runHome $(if ($onWindows) { "win" } else { "posix" }) $mode 2>&1 | Out-String
            } finally {
                Remove-Item -Recurse -Force -LiteralPath $runHome -ErrorAction SilentlyContinue
            }
        }
    } finally {
        Remove-Item -LiteralPath $preflightFile -ErrorAction SilentlyContinue
    }
    $out = $preflightRuns["managed"]
    # The path it decides about, not a path the test computed for it.
    $expectedDir = Join-Path $preflightHome ".unsloth\llama.cpp"
    Check "the preflight resolves the managed llama.cpp dir under the profile" (
        $out -match ("RESOLVED_DIR: " + [regex]::Escape($expectedDir)))
    Check "a readable llama.cpp cache does not stop the install" ($out -match "READABLE_VERDICT: continue")
    if ($out -notmatch "CAN_DENY: True") {
        Write-Host "  SKIP  cannot deny access on this host (running as root/admin?) -- preflight denial checks skipped" -ForegroundColor Yellow
    } else {
        Check "an unreadable llama.cpp cache stops the install" ($out -match "DENIED_VERDICT: stop")
        Check "the preflight reason names the folder" (
            $out -match ("DENIED_REASON: .*" + [regex]::Escape($expectedDir)))
        Check "the preflight reason says a reinstall will not help" (
            $out -match "Reinstalling the app does not reset it")
        Check "the preflight says nothing was installed" ($out -match "Nothing was installed\.")
        Check "the preflight prints the takeown command on its own line" (
            @($out -split "`r?`n" | Where-Object { $_ -match 'takeown /F' -and $_ -notmatch '^DENIED_REASON' }).Count -eq 1)
        Check "the preflight prints the icacls command on its own line" (
            @($out -split "`r?`n" | Where-Object { $_ -match 'icacls .* /reset /T' }).Count -eq 1)
        Check "the preflight says the download has not happened yet" (
            $out -match "nothing has been downloaded or installed")
        # Guidance may print ACL repair commands but must never run them.
        Check "the preflight does not run takeown or icacls itself" (
            $out -notmatch "SUCCESS: The file \(or folder\)" -and
            $out -notmatch "processed file:")

        # Overrides may name the managed location itself; never call it disposable.
        foreach ($mode in @("supplied", "env")) {
            $supplied = $preflightRuns[$mode]
            Check "a tree the user named ($mode) still stops the install" (
                $supplied -match "DENIED_VERDICT: stop")
            Check "a tree the user named ($mode) is not called a cache we own" (
                $supplied -match "DENIED_REASON: .*point UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build" -and
                $supplied -notmatch "DENIED_REASON: .*Delete or rename")
        }
        Check "the managed cache is still called one" (
            $out -match "DENIED_REASON: .*Delete or rename that folder")
    }
}

# ── Complete install/setup/update entrypoints ──
# Run every public entrypoint with network and expensive work trapped. Windows CI
# uses real ACLs; POSIX chmod is the local equivalent.
$entrypointHarness = @'
$ErrorActionPreference = "Stop"
$repoRoot = $args[0]
$testHome = $args[1]
$mode = $args[2]
$env:USERPROFILE = $testHome
$env:HOME = $testHome
$env:UNSLOTH_SKIP_AUTOSTART = "1"
Remove-Item Env:UNSLOTH_TAURI_MODE -ErrorAction SilentlyContinue
Remove-Item Env:UNSLOTH_TAURI_UPDATE -ErrorAction SilentlyContinue
Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
Remove-Item Env:STUDIO_HOME -ErrorAction SilentlyContinue
Remove-Item Env:UNSLOTH_LOCAL_LLAMA_CPP_DIR -ErrorAction SilentlyContinue

function Stop-EntrypointExpense {
    param([string]$Name)
    Write-Host "[TEST:EXPENSIVE] $Name"
    throw "entrypoint reached expensive operation: $Name"
}
function global:Invoke-WebRequest { Stop-EntrypointExpense "Invoke-WebRequest" }
function global:Invoke-RestMethod { Stop-EntrypointExpense "Invoke-RestMethod" }
function global:Invoke-Expression { Stop-EntrypointExpense "Invoke-Expression" }
function global:Start-Process { Stop-EntrypointExpense "Start-Process" }
function global:Expand-Archive { Stop-EntrypointExpense "Expand-Archive" }
function global:winget { Stop-EntrypointExpense "winget" }
function global:python { Stop-EntrypointExpense "python" }
function global:python3 { Stop-EntrypointExpense "python3" }
function global:py { Stop-EntrypointExpense "py" }
function global:uv { Stop-EntrypointExpense "uv" }
function global:pip { Stop-EntrypointExpense "pip" }
function global:nvidia-smi { Stop-EntrypointExpense "nvidia-smi" }
function global:amd-smi { Stop-EntrypointExpense "amd-smi" }
function global:rocm-smi { Stop-EntrypointExpense "rocm-smi" }
function global:git { Stop-EntrypointExpense "git" }
function global:cmake { Stop-EntrypointExpense "cmake" }
function global:node { Stop-EntrypointExpense "node" }
function global:npm { Stop-EntrypointExpense "npm" }
function global:bun { Stop-EntrypointExpense "bun" }
function global:cmd { Stop-EntrypointExpense "cmd" }

if ($mode -eq "install") {
    & (Join-Path $repoRoot "install.ps1") --tauri
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    if ($mode -in @("update", "repair")) {
        $env:STUDIO_PACKAGE_NAME = "unsloth"
        $env:STUDIO_LOCAL_INSTALL = "0"
    }
    if ($mode -eq "repair") {
        $env:UNSLOTH_TAURI_UPDATE = "1"
        $env:SKIP_STUDIO_FRONTEND = "1"
    } else {
        Remove-Item Env:UNSLOTH_TAURI_UPDATE -ErrorAction SilentlyContinue
        Remove-Item Env:SKIP_STUDIO_FRONTEND -ErrorAction SilentlyContinue
    }
    if ($mode -eq "setup") {
        Remove-Item Env:STUDIO_PACKAGE_NAME -ErrorAction SilentlyContinue
        Remove-Item Env:STUDIO_LOCAL_INSTALL -ErrorAction SilentlyContinue
    }
    & (Join-Path $repoRoot "studio/setup.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "[TEST:ENTRYPOINT_RETURNED]"
'@
$entrypointFile = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_entry_" + [guid]::NewGuid().ToString("N") + ".ps1")
Set-Content -LiteralPath $entrypointFile -Value $entrypointHarness -Encoding utf8
$pwshEntrypoint = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshEntrypoint) { $pwshEntrypoint = (Get-Command powershell).Source }

try {
    foreach ($mode in @("install", "setup", "update", "repair")) {
        $entryHome = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_whole_" + [guid]::NewGuid().ToString("N"))
        $entryLocked = Join-Path $entryHome ".unsloth\llama.cpp"
        New-Item -ItemType Directory -Force -Path (Join-Path $entryLocked "build") | Out-Null
        Set-Content -LiteralPath (Join-Path $entryLocked "UNSLOTH_PREBUILT_INFO.json") -Value '{"release_tag":"app-1"}'
        $entryWho = "$env:USERDOMAIN\$env:USERNAME"
        if ($onWindows) { icacls $entryLocked /deny "${entryWho}:(OI)(CI)(RX)" *>$null }
        else { chmod 000 $entryLocked }
        $canDenyEntrypoint = $false
        try {
            try { $null = Test-Path (Join-Path $entryLocked "UNSLOTH_PREBUILT_INFO.json") }
            catch { $canDenyEntrypoint = $true }
            if (-not $canDenyEntrypoint) {
                Write-Host "  SKIP  cannot deny access on this host -- whole $mode entrypoint skipped" -ForegroundColor Yellow
            } else {
                $entryOutput = & $pwshEntrypoint -NoProfile -File $entrypointFile $repoRoot $entryHome $mode 2>&1 | Out-String
                $entryExit = $LASTEXITCODE
                Check "whole $mode entrypoint fails on the denied managed cache" ($entryExit -ne 0)
                Check "whole $mode entrypoint reports the denied managed path" (
                    $entryOutput -match "access is denied" -and
                    $entryOutput -match [regex]::Escape($entryLocked))
                Check "whole $mode entrypoint prints the actionable recovery reason" (
                    $entryOutput -match "This folder lives outside the app, so reinstalling Unsloth Studio, to any drive, reuses it and fails the same way")
                if ($mode -in @("install", "repair")) {
                    $tauriTag = if ($mode -eq "install") { "[TAURI:ERROR_DEFAULT]" } else { "[TAURI:ERROR]" }
                    Check "whole $mode entrypoint hands the actionable reason to Tauri" (
                        $entryOutput.Contains($tauriTag))
                }
                Check "whole $mode entrypoint never reaches a trapped expensive operation" (
                    $entryOutput -notmatch '\[TEST:EXPENSIVE\]')
                Check "whole $mode entrypoint stops before dependency, frontend, Python, uv, venv, and PyTorch markers" (
                    $entryOutput -notmatch 'Checking system dependencies' -and
                    $entryOutput -notmatch '(?im)^\s*frontend\s' -and
                    $entryOutput -notmatch 'downloading Python|Installing Python|Python found:' -and
                    $entryOutput -notmatch 'Installing uv package manager|installing uv package manager' -and
                    $entryOutput -notmatch 'Creating virtual environment|setting up Python environment' -and
                    $entryOutput -notmatch 'Installing PyTorch|installing PyTorch' -and
                    $entryOutput -notmatch '\[TAURI:STEP\] (Installing unsloth|Running studio setup)' -and
                    $entryOutput -notmatch '(?im)^\s*(Installing unsloth|Unsloth Studio Installed)' -and
                    $entryOutput -notmatch '\[TEST:ENTRYPOINT_RETURNED\]')
            }
        } finally {
            if ($onWindows) { icacls $entryLocked /remove:d "$entryWho" *>$null }
            else { chmod 755 $entryLocked }
        }

        $installMarkers = @(
            Get-ChildItem -LiteralPath $entryHome -Recurse -Force -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -in @(".unsloth-studio-owned", ".unsloth-no-torch", "unsloth_install_manifest.json") }
        )
        Check "whole $mode denial leaves no venv or install marker" (
            -not (Test-Path -LiteralPath (Join-Path $entryHome ".unsloth\studio\unsloth_studio")) -and
            $installMarkers.Count -eq 0)
        Remove-Item -Recurse -Force -LiteralPath $entryHome -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item -LiteralPath $entrypointFile -ErrorAction SilentlyContinue
}

if ($script:failures -gt 0) {
    Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
