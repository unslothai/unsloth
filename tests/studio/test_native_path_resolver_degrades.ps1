# Behaviour of install.ps1's native path resolver when the native side is not
# available, which is the state a blocked or locked-down machine is in.
#
# The change this guards replaced an Add-Type compile with reflection emit. The
# ladder around it did not move: an exact answer when the native call works, the
# lexical answer when it does not, and $Exact false either way so the runtime
# lock fails closed. This runs the real functions, extracted from install.ps1,
# and checks that both rungs still answer.
#
# POSIX is the harness on purpose: kernel32 does not resolve here, so every
# native call fails exactly as it does on a Windows host whose antivirus took the
# helper away. The Windows lanes cover the succeeding side.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. ([System.IO.Path]::Combine($repoRoot, "tests", "studio_setup_ps1", "Get-FunctionSource.ps1"))

$failures = 0
function Check {
    param([string]$Name, [bool]$Ok)
    if ($Ok) { Write-Host "  PASS  $Name" -ForegroundColor Green }
    else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")
$text = Get-Content -LiteralPath $installPath -Raw

# ── Source contract ──
Check "install.ps1 compiles no C# at all" (-not ($text -match "Add-Type -TypeDefinition"))
Check "the native imports are emitted instead" ($text -match "DefinePInvokeMethod")
Check "no source or DLL is written for the resolver" (
    -not ($text -match "csc\.exe running"))

# ── The resolver, run for real ──
$fns = @(
    "Write-StudioLine", "Write-StudioFinalPathDegraded",
    "Initialize-StudioFinalPathNativeType", "Get-StudioNativeFinalPath"
)
$src = @()
foreach ($fn in $fns) {
    $found = Get-FunctionSource -Path $installPath -Name $fn
    Check "install.ps1 defines $fn" ($null -ne $found)
    if ($found) { $src += $found }
}

if ($src.Count -eq $fns.Count) {
    $harness = @"
`$ErrorActionPreference = "Stop"
function Write-StudioLine { param([string]`$Message, `$ForegroundColor) Write-Host `$Message }
`$script:StudioFinalPathNativeState = `$null
`$script:StudioFinalPathWarned = `$false
$($src -join "`n")

# 1. The type must define without a compiler, on this engine as on 5.1.
`$defined = Initialize-StudioFinalPathNativeType
Write-Host "EMIT_OK: `$defined"
Write-Host "TYPE_PRESENT: `$(`$null -ne ("UnslothStudioFinalPathV3" -as [type]))"

# 2. Nothing may be written to the temporary directory by defining it.
`$temp = [System.IO.Path]::GetTempPath()
`$fresh = @(Get-ChildItem -LiteralPath `$temp -File -ErrorAction SilentlyContinue |
    Where-Object { `$_.LastWriteTime -gt (Get-Date).AddMinutes(-1) -and `$_.Extension -in @(".dll", ".cs", ".cmdline", ".err", ".out") })
Write-Host "TEMP_ARTIFACTS: `$(`$fresh.Count)"

# 3. The dynamic assembly must exist only in memory.
if (`$defined) {
    `$asm = ("UnslothStudioFinalPathV3" -as [type]).Assembly
    Write-Host "IN_MEMORY_ONLY: `$([string]::IsNullOrEmpty(`$asm.Location))"
}

# 4. A native call that cannot work must answer null, not throw. kernel32 is
#    absent here, which is what a host missing the helper looks like.
`$threw = `$false
`$answer = "unset"
try { `$answer = Get-StudioNativeFinalPath -Path `$temp } catch { `$threw = `$true }
Write-Host "NATIVE_THREW: `$threw"
Write-Host "NATIVE_ANSWER_NULL: `$([string]::IsNullOrEmpty(`$answer))"
"@
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_native_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $file -Value $harness -Encoding utf8
    $exe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = (Get-Command powershell).Source }
    try {
        $out = & $exe -NoProfile -File $file 2>&1 | Out-String
    } finally {
        Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue
    }

    Check "the native type defines with no compiler" ($out -match "EMIT_OK: True")
    Check "the emitted type is really there" ($out -match "TYPE_PRESENT: True")
    Check "defining it writes nothing to the temporary directory" ($out -match "TEMP_ARTIFACTS: 0")
    Check "the assembly exists only in memory" ($out -match "IN_MEMORY_ONLY: True")
    # The whole point of the ladder: a native call that cannot work is not fatal.
    Check "a native call that cannot work does not throw" ($out -match "NATIVE_THREW: False")
    Check "a native call that cannot work answers null" ($out -match "NATIVE_ANSWER_NULL: True")
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
