# Regression tests for install.ps1's venv-holder probe.
#
# Resolving a PID to its image path used to open a handle to every running process from inline
# C# compiled at runtime -- a shape AV heuristics score hard, on top of the csc.exe compile the
# type already costs. Win32_Process answers the same question for the same set of processes in
# one query, so the pair was removed. These tests pin both halves: the P/Invoke must stay gone,
# and the replacement must actually resolve a real process.
$ErrorActionPreference = "Stop"
$script:failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")
$installText = Get-Content -Raw -LiteralPath $installPath

# ── Source contract ──
Check "install.ps1 no longer opens a handle per PID from inline C#" (
    $installText -notmatch 'private static extern IntPtr OpenProcess')
Check "install.ps1 no longer imports the image-path query" (
    $installText -notmatch 'QueryFullProcessImageNameW')
Check "the inline type still resolves final paths" (
    $installText -match 'GetFinalPathNameByHandleW')
Check "the venv-holder probe reads the prefetched map" (
    $installText -match '\$imagePaths = Get-StudioProcessImagePathMap')
Check "one Win32_Process query, not one per PID" (
    ([regex]::Matches($installText, 'Get-CimInstance Win32_Process')).Count -eq 1)

# ── Behaviour ──
. ([System.IO.Path]::Combine($repoRoot, "tests", "studio_setup_ps1", "Get-FunctionSource.ps1"))
$src = Get-FunctionSource -Path $installPath -Name "Get-StudioProcessImagePathMap"
Check "install.ps1 defines Get-StudioProcessImagePathMap" ($null -ne $src)

if ($src -and $IsWindows -ne $false) {
    . ([scriptblock]::Create($src))
    $map = Get-StudioProcessImagePathMap
    Check "the map is a hashtable" ($map -is [hashtable])
    # This process is running an interpreter off disk, so it must resolve.
    $own = $PID
    Check "the map resolves this process to an image path" (
        $map.ContainsKey([int]$own) -and $map[[int]$own])
    if ($map.ContainsKey([int]$own)) {
        Check "the resolved image path exists on disk" (Test-Path -LiteralPath $map[[int]$own])
    }
    # Every value must be a path, never a bare process name: the venv match compares full paths.
    $bad = @($map.Values | Where-Object { $_ -and -not ([System.IO.Path]::IsPathRooted($_)) })
    Check "every resolved image path is rooted" ($bad.Count -eq 0)
} else {
    Write-Host "  SKIP  runtime map checks (Win32_Process is Windows-only)"
}

if ($script:failures -gt 0) {
    Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
