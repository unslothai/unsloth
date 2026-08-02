# Regression test for setup.ps1 path probes on an ACL-denied install tree.
#
# Test-Path raises UnauthorizedAccessException instead of returning $false when
# an ACL denies the probe, and setup.ps1 runs under "Stop", so the bare probe of
# the llama.cpp prebuilt metadata aborted setup with a raw "Test-Path : Access
# is denied" and exit code 1. ~/.unsloth/llama.cpp outlives an app reinstall, so
# reinstalling, even to another drive, hit the same line again.
#
# The probes now go through Get-PathState / Test-PathQuiet, which never
# terminate and keep "Denied" distinct from "Absent". This runs the real
# functions against a genuinely unreadable directory (chmod on Unix, icacls deny
# on Windows).
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
                  "Get-PathDenialDetail", "Get-StudioAdoptableState",
                  "Test-StudioOwnedAdoptable")) {
    $src = Get-FunctionSource -Path $setupPath -Name $fn
    Check "setup.ps1 defines $fn" ($null -ne $src)
    if ($src) { . ([scriptblock]::Create($src)) }
}

# ── Source contract: the crash site must probe state, not bare Test-Path ──
$setupText = Get-Content -Raw -LiteralPath $setupPath
Check "prebuilt metadata probe no longer uses a bare Test-Path" (
    $setupText -notmatch '\n\s*if \(Test-Path \$existingMetaPath\)')
Check "prebuilt metadata probe goes through Get-PathState" (
    $setupText -match '\$existingMetaState = Get-PathState -Path \$existingMetaPath -PathType Leaf')
Check "a denied llama.cpp install fails with an actionable message" (
    $setupText -match '\$existingMetaState -eq "Denied"' -and
    $setupText -match 'Exit-SetupFailure "Access denied reading the existing \$Label')
# Every denial route reports instead of proceeding: an unreadable parent dir, an
# unreadable metadata file, an unreadable .git checkout, and the ownership guard.
Check "the prebuilt phase stops on a denied llama.cpp dir" (
    $setupText -match '\$llamaDirState = Get-PathState -Path \$LlamaCppDir' -and
    $setupText -match '\$llamaDirState -eq "Denied"')
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

    Set-Denied $true

    # Negative control AND environment gate: the old unguarded form must blow up
    # here, otherwise this host cannot produce a denial (root / admin bypass)
    # and the assertions below would pass vacuously.
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
    }
} finally {
    Set-Denied $false
    Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
}

# ── Test-Path parity: the regression-safety invariant ──
# Wherever the old bare probe returned a value instead of throwing, Get-PathState
# must agree. Denied may only appear where the old probe threw, so no path that
# used to work can take a different branch now.
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
# Windows only: POSIX keeps a mode-000 file stat-able, so this state cannot be
# built on Unix. Collapsing it to "No" made the ownership guard report an
# Unsloth tree as an unrelated directory instead of a permissions problem.
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
# Get-PathDenialDetail runs while a failure is being reported, so a null or empty
# path must not replace the actionable message with a binding exception.
foreach ($edge in @($null, "", "   ")) {
    $edgeOk = $true
    try { $null = Get-PathDenialDetail -Path $edge } catch { $edgeOk = $false }
    Check "Get-PathDenialDetail tolerates an empty/null path" $edgeOk
}

# ── The desktop app must receive the reason, not just "exit code 1" ──
# The real Exit-PathAccessDenied with the real Exit-SetupFailure in Tauri mode:
# install.rs prefers a [TAURI:ERROR] line over its generic exit-code message, so
# this is what the user reads.
$exitDeniedSrc = Get-FunctionSource -Path $setupPath -Name Exit-PathAccessDenied
$exitSetupSrc = Get-FunctionSource -Path $setupPath -Name Exit-SetupFailure
Check "setup.ps1 defines Exit-PathAccessDenied" ($null -ne $exitDeniedSrc)
if ($exitDeniedSrc) {
    $harness = @"
`$ErrorActionPreference = "Stop"
function step { param([string]`$Label, [string]`$Value, [string]`$Color = "Green") Write-Host "  `$Label  `$Value" }
function substep { param([string]`$Message, [string]`$Color = "DarkGray") Write-Host "    `$Message" }
function Get-PathDenialDetail { param([string]`$Path) return "" }
$exitSetupSrc
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
    # takeown and icacls must be copy-pasteable. On one line, "then" is not a
    # PowerShell separator and takeown would swallow the rest as arguments.
    $takeownLines = @($out -split "`r?`n" | Where-Object { $_ -match 'takeown /F' })
    Check "takeown is printed on its own line" ($takeownLines.Count -eq 1)
    Check "icacls is not appended to the takeown line" (
        $takeownLines.Count -eq 1 -and $takeownLines[0] -notmatch 'icacls')
    Check "icacls is printed on its own line" (
        @($out -split "`r?`n" | Where-Object { $_ -match 'icacls .* /reset /T' }).Count -eq 1)
}

# ── Denial classification ──
Check "UnauthorizedAccessException classifies as access denied" (
    Test-AccessDeniedError ([System.UnauthorizedAccessException]::new("denied")))
Check "a wrapped UnauthorizedAccessException classifies as access denied" (
    Test-AccessDeniedError ([System.Exception]::new("outer", [System.UnauthorizedAccessException]::new("denied"))))
Check "an unrelated exception does not classify as access denied" (
    -not (Test-AccessDeniedError ([System.IO.FileNotFoundException]::new("missing"))))

# -- Assert-StudioOwnedOrAbsent -NonFatal, run for real --
# whisper.cpp promises "failure is never fatal", so a denied tree there must be
# handed back to the caller instead of exiting the run and taking llama.cpp
# inference with it. Everything else about the guard has to stay fatal.
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
    # Populate before locking. Windows returns $false for a MISSING child of a
    # denied directory instead of throwing, so a probe target that does not
    # exist makes the negative control read as "this host cannot deny".
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

            # A locked directory still probes Present from its readable parent, so
            # the case above only covers the marker read. Lock the parent to reach
            # the root probe itself.
            $out = $null
            $threw = $false
            try { $out = @(Assert-StudioOwnedOrAbsent -Path $nfInner -Label "whisper.cpp install" -NonFatal) }
            catch { $threw = $true }
            Check "-NonFatal hands back a tree whose parent is unreadable" (
                -not $threw -and $out.Count -eq 1 -and $out[0] -eq "Denied")

            # The realistic fresh-custom-home case: no marker was ever written.
            # Windows reports a MISSING child of a denied directory as Absent, so
            # this cannot go through the marker probe there and has to be caught
            # by the adoptable-state read instead. Either route is fine; anything
            # other than Denied is not.
            # Two shapes. chmod 000 blocks the child probes outright; chmod 111
            # allows stat of a named child but forbids listing, which is exactly
            # what Windows does and the only shape that reaches the directory
            # listing in Get-StudioAdoptableState.
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

if ($script:failures -gt 0) {
    Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
