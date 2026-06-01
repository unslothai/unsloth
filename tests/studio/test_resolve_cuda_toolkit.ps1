#!/usr/bin/env pwsh
# Unit test for Resolve-CudaToolkit in studio/setup.ps1. No GPU required: the
# detection helpers (nvidia-smi, nvcc, Find-Nvcc, ...) are stubbed so the real
# function logic runs against a spoofed Blackwell sm_120 / driver 13.2 host.
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

# --- Spoof executables: nvidia-smi reports driver max CUDA 13.2; nvcc 13.3 ---
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("rct_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$smiFake  = Join-Path $work "nvidia-smi.ps1"
$nvccFake = Join-Path $work "nvcc.ps1"
Set-Content -LiteralPath $smiFake  -Value "'CUDA Version: 13.2'"
Set-Content -LiteralPath $nvccFake -Value "'Cuda compilation tools, release 13.3, V13.3.0'"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Build + run one scenario in a child pwsh; returns @{ Exit; Out }.
function Run-Case {
    param([string]$FindMode, [bool]$Require)
    $requireLit = if ($Require) { '$true' } else { '$false' }
    $child = @"
`$ErrorActionPreference = 'Continue'
[Environment]::SetEnvironmentVariable('CUDA_PATH', `$null, 'Process')
`$FindNvccMode = '$FindMode'
`$NvccFake = '$nvccFake'
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
        'compatible'   { return `$NvccFake }
        'incompatible' { if (`$MaxVersion) { return `$null } else { return `$NvccFake } }
        default        { return `$null }
    }
}
`$NvidiaSmiExe  = '$smiFake'
`$VsInstallPath = `$null
`$HasNvidiaSmi  = `$true
`$script:CudaToolkitReady = `$false
`$script:NvccPath = `$null; `$script:CudaToolkitRoot = `$null; `$script:CudaArch = `$null

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
    Write-Host "Scenario 1: prebuilt path, too-new toolkit (no -RequireOrExit) -> defers, no exit"
    $r = Run-Case -FindMode "incompatible" -Require $false
    Check "exits 0 (not blocked)"        ($r.Exit -eq 0)
    Check "CudaToolkitReady = false"     ($r.Out -match "ready=False")
    Check "winget NOT called"            ($r.Out -match "winget=False")
    Check "no INCOMPATIBLE error text"   (-not ($r.Out -match "INCOMPATIBLE"))

    Write-Host "Scenario 2: forced source build, too-new toolkit (-RequireOrExit) -> hard exit"
    $r = Run-Case -FindMode "incompatible" -Require $true
    Check "exits non-zero"               ($r.Exit -ne 0)
    Check "preserved INCOMPATIBLE error" ($r.Out -match "is installed but INCOMPATIBLE")

    Write-Host "Scenario 3: compatible toolkit (-RequireOrExit) -> resolves, env set"
    $r = Run-Case -FindMode "compatible" -Require $true
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = true"      ($r.Out -match "ready=True")
    Check "NvccPath published"           ($r.Out -match "nvcc=.*nvcc")

    Write-Host "Scenario 4: no toolkit, prebuilt path (no -RequireOrExit) -> defers, no winget"
    $r = Run-Case -FindMode "none" -Require $false
    Check "exits 0"                      ($r.Exit -eq 0)
    Check "CudaToolkitReady = false"     ($r.Out -match "ready=False")
    Check "winget NOT called"            ($r.Out -match "winget=False")

    Write-Host "Scenario 5: no toolkit, forced (-RequireOrExit) -> winget attempted then exit"
    # The function exits before the RESULT line here, so assert on the winget-block
    # marker in output rather than the flag.
    $r = Run-Case -FindMode "none" -Require $true
    Check "winget attempted"             ($r.Out -match "installing via winget")
    Check "exits non-zero"               ($r.Exit -ne 0)
    Check "preserved nvcc-required error" ($r.Out -match "CUDA Toolkit \(nvcc\) is required")
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $work -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
