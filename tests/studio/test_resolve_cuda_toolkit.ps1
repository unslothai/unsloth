#!/usr/bin/env pwsh
# Unit test for Resolve-CudaToolkit in studio/setup.ps1. No GPU required: the
# detection helpers (nvidia-smi, nvcc, Find-Nvcc, ...) are stubbed so the real
# function logic runs against spoofed Blackwell sm_120 driver/toolkit scenarios.
#
# The function is extracted via AST and run in a child pwsh per scenario, because
# the -RequireOrExit path calls `exit` (which would otherwise kill this harness).
#
# Run: pwsh -NoProfile -File tests/studio/test_resolve_cuda_toolkit.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

# --- Extract the function source (not the whole installer) ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }
$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Resolve-CudaToolkit"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one Resolve-CudaToolkit, found $($fn.Count)" }
$fnText = $fn[0].Extent.Text

# Resolve-CudaToolkit calls Write-CudaDriverToolkitMismatch, so extract it too.
$mismatchFn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Write-CudaDriverToolkitMismatch"
}, $true)
if ($mismatchFn.Count -ne 1) { throw "expected exactly one Write-CudaDriverToolkitMismatch, found $($mismatchFn.Count)" }
$mismatchText = $mismatchFn[0].Extent.Text

# --- Spoof executables for driver/toolkit compatibility scenarios ---
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("rct_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$smiMajorMismatchFake = Join-Path $work "nvidia-smi-12.9.ps1"
$smiSameMajorFake     = Join-Path $work "nvidia-smi-13.2.ps1"
$nvccIncompatibleFake = Join-Path $work "nvcc-13.3.ps1"
$nvccCompatibleFake   = Join-Path $work "nvcc-12.8.ps1"
Set-Content -LiteralPath $smiMajorMismatchFake -Value "'CUDA Version: 12.9'"
Set-Content -LiteralPath $smiSameMajorFake     -Value "'CUDA Version: 13.2'"
Set-Content -LiteralPath $nvccIncompatibleFake -Value "'Cuda compilation tools, release 13.3, V13.3.0'"
Set-Content -LiteralPath $nvccCompatibleFake   -Value "'Cuda compilation tools, release 12.8, V12.8.0'"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Build + run one scenario in a child pwsh; returns @{ Exit; Out }.
function Run-Case {
    param([string]$FindMode, [bool]$Require, [string]$DriverMode = "major-mismatch")
    $requireLit = if ($Require) { '$true' } else { '$false' }
    $smiForCase = if ($DriverMode -eq "same-major") { $smiSameMajorFake } else { $smiMajorMismatchFake }
    $child = @"
`$ErrorActionPreference = 'Continue'
[Environment]::SetEnvironmentVariable('CUDA_PATH', `$null, 'Process')
`$FindNvccMode = '$FindMode'
`$NvccIncompatibleFake = '$nvccIncompatibleFake'
`$NvccCompatibleFake = '$nvccCompatibleFake'
function substep { param(`$m, `$c) Write-Host "  `$m" }
function step    { param(`$l, `$v, `$c) Write-Host "[`$l] `$v" }
function Add-ToUserPath { param(`$Directory, `$Position) `$true }
function Refresh-Environment { }
function Get-CudaComputeCapability { '120' }
function Test-NvccArchSupport { param(`$NvccExe, `$Arch) `$true }
function Get-NvccMaxArch { param(`$NvccExe) '120' }
`$script:WingetCalled = `$false
function winget { `$script:WingetCalled = `$true; 'no matching versions' }
function Find-Nvcc {
    param([string]`$MaxVersion = '')
    switch (`$FindNvccMode) {
        'compatible'   { return `$NvccCompatibleFake }
        'same-major'   { return `$NvccIncompatibleFake }
        'incompatible' { if (`$MaxVersion) { return `$null } else { return `$NvccIncompatibleFake } }
        default        { return `$null }
    }
}
`$NvidiaSmiExe  = '$smiForCase'
`$VsInstallPath = `$null
`$HasNvidiaSmi  = `$true
`$script:CudaToolkitReady = `$false
`$script:NvccPath = `$null; `$script:CudaToolkitRoot = `$null; `$script:CudaArch = `$null

$mismatchText

$fnText

if ($requireLit) { Resolve-CudaToolkit -RequireOrExit } else { Resolve-CudaToolkit }
Write-Host ("RESULT ready={0} nvcc={1} winget={2}" -f `$script:CudaToolkitReady, `$script:NvccPath, `$script:WingetCalled)
"@
    $childFile = Join-Path $work ("case_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $childFile -Value $child
    $out = & pwsh -NoProfile -File $childFile 2>&1 | Out-String
    return @{ Exit = $LASTEXITCODE; Out = $out }
}

try {
    Write-Host "Scenario 1: prebuilt path, newer-major toolkit (no -RequireOrExit) -> defers, no exit"
    $r = Run-Case -FindMode "incompatible" -Require $false
    Check "exits 0 (not blocked)"        ($r.Exit -eq 0)
    Check "CudaToolkitReady = false"     ($r.Out -match "ready=False")
    Check "winget NOT called"            ($r.Out -match "winget=False")
    Check "explains major mismatch"      ($r.Out -match "major-version mismatch")
    Check "does not blame the toolkit"   (-not ($r.Out -match "INCOMPATIBLE"))

    Write-Host "Scenario 2: forced source build, newer-major toolkit (-RequireOrExit) -> hard exit"
    $r = Run-Case -FindMode "incompatible" -Require $true
    Check "exits non-zero"               ($r.Exit -ne 0)
    Check "explains major mismatch"      ($r.Out -match "major-version mismatch")
    Check "one-line source-build error"  ($r.Out -match "CUDA source build cannot use the installed toolkit")

    Write-Host "Scenario 3: same-major newer-minor toolkit (-RequireOrExit) -> resolves, env set"
    $r = Run-Case -FindMode "same-major" -Require $true -DriverMode "same-major"
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = true"      ($r.Out -match "ready=True")
    Check "NvccPath published"           ($r.Out -match "nvcc=.*nvcc-13\.3")
    Check "no mismatch warning"          (-not ($r.Out -match "major-version mismatch"))

    Write-Host "Scenario 4: compatible older-major toolkit (-RequireOrExit) -> resolves, env set"
    $r = Run-Case -FindMode "compatible" -Require $true
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = true"      ($r.Out -match "ready=True")
    Check "NvccPath published"           ($r.Out -match "nvcc=.*nvcc")

    Write-Host "Scenario 5: no toolkit, prebuilt path (no -RequireOrExit) -> defers, no winget"
    $r = Run-Case -FindMode "none" -Require $false
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = false"     ($r.Out -match "ready=False")
    Check "winget NOT called"            ($r.Out -match "winget=False")

    Write-Host "Scenario 6: no toolkit, forced (-RequireOrExit) -> winget attempted then exit"
    # The function exits before the RESULT line here, so assert on the winget-block
    # marker in output rather than the flag.
    $r = Run-Case -FindMode "none" -Require $true
    Check "winget attempted"             ($r.Out -match "installing via winget")
    Check "exits non-zero"               ($r.Exit -ne 0)
    Check "preserved nvcc-required error" ($r.Out -match "CUDA Toolkit \(nvcc\) is required")

    Write-Host "Scenario 7: same-major toolkit only on PATH, missed by -MaxVersion (-RequireOrExit) -> accepted, not rejected"
    # -MaxVersion misses it (not in side-by-side base) but plain Find-Nvcc finds it on PATH: must be used.
    $r = Run-Case -FindMode "incompatible" -Require $true -DriverMode "same-major"
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = true"      ($r.Out -match "ready=True")
    Check "NvccPath published"           ($r.Out -match "nvcc=.*nvcc-13\.3")
    Check "no mismatch warning"          (-not ($r.Out -match "major-version mismatch"))
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $work -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
