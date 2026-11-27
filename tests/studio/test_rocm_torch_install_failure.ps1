#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Failed ROCm torch on a supported AMD GPU must stop setup unless UNSLOTH_ALLOW_CPU_TORCH_FALLBACK=1.
# ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = Join-Path (Join-Path $root "studio") "setup.ps1"
$setup = Get-Content -LiteralPath $setupPath -Raw -Encoding UTF8

function Cut([string]$From, [string]$To) {
    $start = $setup.IndexOf($From)
    $end = if ($start -ge 0) { $setup.IndexOf($To, $start) } else { -1 }
    if ($start -lt 0 -or $end -le $start) { throw "setup.ps1 no longer has the block from '$From' to '$To'" }
    return $setup.Substring($start, $end - $start)
}
# ROCm, XPU and CPU arms only; the CUDA arm is closed by hand.
$block = (Cut "`$ROCmCpuFallback = `$false" "} elseif (-not `$ROCmIndexUrl -and -not `$XpuIndexUrl) {") + "`n}`n"
$helpers = (Cut "function Redact-InstallOutput {" "# _grouped_mm bug: these leaves need the torch 2.11 floor.")

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

class SetupFailed : System.Exception { SetupFailed([string]$m) : base($m) {} }
$script:Installs = @()
$script:Lines = @()
$script:RocmExit = 0
function Fast-Install {
    $line = ($args | ForEach-Object { "$_" }) -join " "
    $script:Installs += ,$line
    $global:LASTEXITCODE = if ($line -like "*repo.amd.com*") { $script:RocmExit } else { 0 }
    if ($global:LASTEXITCODE -ne 0) { return "error: Failed to fetch: https://repo.amd.com/rocm/whl-multi-arch/torch/`n  Caused by: Connection refused (os error 10061)" }
    return ""
}
function Invoke-BoundedPythonProbe { param($PythonExe, $Code, $TimeoutSec) return @{ Ok = $true; Output = "" } }
function Assert-XpuRuntimeReady { param($PythonExe) return $true }
function Exit-SetupFailure { param([string]$Message, [int]$Code = 1, [switch]$NoTauriMarker) throw [SetupFailed]::new($Message) }
function substep { param($Text, $Color) $script:Lines += ,"$Text" }
function Write-StudioLine { param($Text, $ForegroundColor) $script:Lines += ,"$Text" }
Invoke-Expression $helpers

$venvRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm_fail_venv_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $venvRoot -Force | Out-Null
$cpuIndex = "https://download.pytorch.org/whl/cpu"
$amdIndex = "https://repo.amd.com/rocm/whl-multi-arch/"

function Invoke-Block([string]$Arch = "gfx1151", [int]$RocmExit = 0, [string]$Allow = "", [bool]$Rocm = $true, [string]$Cu = "cpu") {
    $script:Installs = @()
    $script:Lines = @()
    $script:RocmExit = $RocmExit
    if ($Allow) { $env:UNSLOTH_ALLOW_CPU_TORCH_FALLBACK = $Allow } else { Remove-Item Env:UNSLOTH_ALLOW_CPU_TORCH_FALLBACK -ErrorAction SilentlyContinue }
    Remove-Item Env:UNSLOTH_ROCM_TORCH_INSTALLED -ErrorAction SilentlyContinue
    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue
    $ROCmGfxArch = $Arch
    $ROCmIndexUrl = if ($Rocm) { $amdIndex } else { $null }
    $script:ROCmMultiArch = $Rocm
    $ROCmTorchSpec = "torch[device-$Arch]==2.11.0+rocm7.14.0"
    $ROCmVisionSpec = "torchvision[device-$Arch]==0.26.0+rocm7.14.0"
    $ROCmAudioSpec = "torchaudio==2.11.0+rocm7.14.0"
    $installedTorchTag = "cpu"
    $VenvDir = $venvRoot
    $VenvPyExe = $null
    $CuTag = $Cu
    $XpuIndexUrl = $null
    $TorchInstallIndexUrl = $cpuIndex
    $PyTorchWhlBase = "https://download.pytorch.org/whl"
    $TorchIndexPinned = $false
    $WinArm64NoAudio = $false
    $script:UnslothVerbose = $false
    $script:PinChangedForceReinstall = $false
    $script:TorchImportDefinitivelyFailed = $false
    $script:NvidiaPreR450CpuFallback = $false
    $failed = $null
    try { Invoke-Expression $block } catch [SetupFailed] { $failed = $_.Exception.Message }
    return @{ Installs = @($script:Installs); Lines = @($script:Lines); Failed = $failed; RocmIndexUrl = $ROCmIndexUrl;
              RocmInstalled = $env:UNSLOTH_ROCM_TORCH_INSTALLED }
}
function CpuInstalls($r) { return @($r.Installs | Where-Object { $_ -like "*$cpuIndex*" }) }
function HasLine($r, [string]$Pattern) { return [bool](@($r.Lines | Where-Object { $_ -match $Pattern }).Count) }

try {
    Write-Host "=== ROCm install fails, no opt-in: setup stops, loudly, with the reason ==="
    $r = Invoke-Block -RocmExit 2
    Check "setup fails instead of continuing" ([bool]$r.Failed)
    Check "the failure names the arch" ($r.Failed -match 'gfx1151')
    Check "the failure carries the uv reason" ($r.Failed -match 'Connection refused')
    Check "the failure message names the opt-in (Desktop shows only this line)" ($r.Failed -match 'UNSLOTH_ALLOW_CPU_TORCH_FALLBACK=1')
    Check "no CPU torch is installed" ((CpuInstalls $r).Count -eq 0)
    Check "an [ERROR] line names the arch" (HasLine $r '^\[ERROR\] AMD ROCm PyTorch could not be installed for gfx1151')
    Check "the index is printed" (HasLine $r 'index:\s+https://repo\.amd\.com/rocm/whl-multi-arch')
    Check "the wanted trio is printed" (HasLine $r 'wanted:.*torch\[device-gfx1151\]==2\.11\.0\+rocm7\.14\.0')
    Check "the reason is printed" (HasLine $r 'reason:.*Connection refused')
    Check "the opt-in is named" (HasLine $r 'UNSLOTH_ALLOW_CPU_TORCH_FALLBACK=1')
    Check "no GPU success line" (-not (HasLine $r 'GPU ROCm PyTorch installed'))
    Check "UNSLOTH_ROCM_TORCH_INSTALLED is not set" (-not $r.RocmInstalled)

    Write-Host "=== ROCm install fails, opt-in: CPU, bounded, labelled as not a GPU install ==="
    foreach ($v in @("1", "true", "ON")) {
        $r = Invoke-Block -RocmExit 2 -Allow $v
        $cpu = @(CpuInstalls $r)
        Check "opt-in '$v': setup continues" (-not $r.Failed)
        Check "opt-in '$v': exactly one CPU install" ($cpu.Count -eq 1)
        Check "opt-in '$v': CPU torch kept inside the supported window" ($cpu.Count -eq 1 -and $cpu[0] -like '*torch>=2.4,<2.12.0 torchvision>=0.19,<0.27.0 torchaudio>=2.4,<2.12.0*')
        Check "opt-in '$v': CPU install force-reinstalls" ($cpu.Count -eq 1 -and $cpu[0] -like '*--force-reinstall*')
        Check "opt-in '$v': says it is not a GPU install" (HasLine $r 'NOT a GPU install')
        Check "opt-in '$v': still prints the [ERROR] with the reason" ((HasLine $r '^\[ERROR\] AMD ROCm PyTorch could not be installed') -and (HasLine $r 'reason:.*Connection refused'))
        Check "opt-in '$v': ROCm expectation dropped" (-not $r.RocmIndexUrl)
    }
    $r = Invoke-Block -RocmExit 2 -Allow "0"
    Check "opt-in '0' is not an opt-in" ([bool]$r.Failed -and (CpuInstalls $r).Count -eq 0)

    Write-Host "=== ROCm install succeeds: unchanged ==="
    $r = Invoke-Block -RocmExit 0
    Check "no failure" (-not $r.Failed)
    Check "one install, from the AMD multi-arch index" ($r.Installs.Count -eq 1 -and $r.Installs[0] -like "*--index-url $amdIndex*")
    Check "the pinned gfx1151 trio" ($r.Installs[0] -like 'torch`[device-gfx1151`]==2.11.0+rocm7.14.0 torchvision`[device-gfx1151`]==0.26.0+rocm7.14.0 torchaudio==2.11.0+rocm7.14.0*')
    Check "GPU success line" (HasLine $r 'GPU ROCm PyTorch installed \(gfx1151\)')
    Check "UNSLOTH_ROCM_TORCH_INSTALLED=1" ($r.RocmInstalled -eq "1")
    Check "no CPU install" ((CpuInstalls $r).Count -eq 0)

    Write-Host "=== plain CPU route (no AMD wheels): unchanged ==="
    $r = Invoke-Block -Rocm $false -Arch ""
    $cpu = @(CpuInstalls $r)
    Check "no failure" (-not $r.Failed)
    Check "one CPU install of the unbounded trio, no force" ($cpu.Count -eq 1 -and $cpu[0] -eq "torch torchvision torchaudio --index-url $cpuIndex")
    Check "no ROCm error line" (-not (HasLine $r 'ROCm PyTorch could not be installed'))

    Write-Host "=== CUDA route: the ROCm / CPU arms stay out of it ==="
    $r = Invoke-Block -Rocm $false -Arch "" -Cu "cu128"
    Check "no install from these arms" ($r.Installs.Count -eq 0 -and -not $r.Failed)
} finally {
    Remove-Item Env:UNSLOTH_ALLOW_CPU_TORCH_FALLBACK -ErrorAction SilentlyContinue
    Remove-Item Env:UNSLOTH_ROCM_TORCH_INSTALLED -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $venvRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "all checks passed"
exit 0
