#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.ps1 pairs triton-windows with torch on the AMD ROCm path after the dependency stack: the stack resolves
# unsloth's unpinned triton-windows, and 3.9 removed the block pointers torch 2.11 / 2.12's flex_decoding builds
# (gpt-oss 4k prefill 0.1 -> 6.7 GB on gfx1151). A verified install that skips the dependency pass is sent back
# through it while the pair is stale. The helpers and the post-stack block are cut out of setup.ps1 and run against
# stubs. ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
# Run: pwsh -NoProfile -File tests/studio/test_rocm_triton_windows_pairing.ps1

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
$helpers = Cut "# AMD ROCm: triton-windows paired with torch." "# End of the AMD ROCm triton-windows pairing helpers."
$block = Cut "# AMD ROCm: pair triton-windows with torch (Get-RocmTritonPairing" "# Only bitsandbytes' XPU library"

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

# Stubs.
$script:ProbeOutput = ""
$script:ProbeOk = $true
$script:Installs = @()
$script:Offline = $false
$script:OfflineFastPath = $false
$script:RealProbeDir = $null
$script:OverrideSeen = @()
function Invoke-BoundedPythonProbe {
    param($PythonExe, $Code, $TimeoutSec)
    if ($script:RealProbeDir) {
        # The real probe code, against a site directory of hand-made dist-info folders only (-S: no site-packages).
        $py = Get-Command python -ErrorAction SilentlyContinue
        if (-not $py) { $py = Get-Command python3 }
        $savedPath = $env:PYTHONPATH
        $env:PYTHONPATH = $script:RealProbeDir
        try { $out = & $py.Source -S -c $Code 2>&1 | Out-String } finally { $env:PYTHONPATH = $savedPath }
        return @{ Ok = ($LASTEXITCODE -eq 0); Output = $out }
    }
    return @{ Ok = $script:ProbeOk; Output = $script:ProbeOutput }
}
function Fast-Install { $script:Installs += ,($args -join " "); $script:OverrideSeen += ,"$env:UV_OVERRIDE"; $global:LASTEXITCODE = 0; return "" }
function Test-UvOfflineRequested { return $script:Offline }
function substep { param($Text, $Color) }
function Write-StudioLine { param($Text, $ForegroundColor) }
function Redact-InstallOutput { param($Text) return $Text }
Invoke-Expression $helpers

function Set-Probe([string]$Torch, [string]$Triton, [bool]$Ok = $true) {
    $script:ProbeOutput = "TORCH=$Torch`nTRITONWIN=$Triton"
    $script:ProbeOk = $Ok
}

function Invoke-Block([string]$Torch, [string]$Triton, [int]$StackExit = 0,
                      $IndexUrl = "https://repo.amd.com/rocm/whl/gfx1151/", [bool]$NoTorch = $false, $Tag = $null) {
    Set-Probe $Torch $Triton
    $script:Installs = @()
    $script:OverrideSeen = @()
    $stackExit = $StackExit
    $ROCmIndexUrl = $IndexUrl
    $NoTorchMode = $NoTorch
    $installedTorchTag = $Tag
    Invoke-Expression $block
    return ,$script:Installs
}

Write-Host "=== spec per torch ==="
Check "torch 2.11 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.11.0+rocm7.14.0") -eq "triton-windows>=3.8,<3.9")
Check "torch 2.12 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.12.1+rocm7.14.0") -eq "triton-windows>=3.8,<3.9")
Check "torch 2.13 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.13.0+rocm7.14.0"))
Check "torch 2.10 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.10.0+rocm7.2"))
Check "unreadable torch is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion ""))

Write-Host "=== post-stack step ==="
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "3.9 on torch 2.11 is replaced by 3.8" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.6.0.post25"
Check "3.6 (no HIP headers) is replaced by 3.8" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton ""
Check "a missing triton-windows is installed" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29"
Check "3.8 is kept" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.13.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "torch 2.13 keeps what the stack resolved" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+cu128" -Triton "3.9.0.post30" -IndexUrl $null -Tag "cu128"
Check "not the ROCm path: nothing is touched" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.2.0" -Triton "3.9.0.post30" -IndexUrl $null -Tag "rocm"
Check "a custom pinned ROCm index (no ROCm index URL) is repaired too" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+cu128" -Triton "3.9.0.post30" -IndexUrl $null -Tag "rocm"
Check "a ROCm venv the stack moved to CUDA torch is not touched" ($r.Count -eq 0)
$env:UV_OVERRIDE = "inherited-overrides.txt"
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "an inherited UV_OVERRIDE is cleared for the pairing install" ($r.Count -eq 1 -and $script:OverrideSeen[0] -eq "")
Check "and restored afterwards" ($env:UV_OVERRIDE -eq "inherited-overrides.txt")
Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "no UV_OVERRIDE stays unset" ($null -eq $env:UV_OVERRIDE)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -StackExit 1
Check "a failed stack is not touched" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -NoTorch $true
Check "no-torch mode leaves an existing Triton alone" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -NoTorch $true
Check "no-torch mode installs no Triton" ($r.Count -eq 0)

Write-Host "=== fast-path escape ==="
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "verified install with 3.9 on ROCm torch 2.11 runs the pass" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm")
Set-Probe "2.11.0+rocm7.14.0" ""
Check "a repair that failed last run (no triton-windows) is retried" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm")
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "a pinned gfx leaf tag counts as ROCm" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "gfx1151")
Set-Probe "2.11.0+rocm7.14.0" "3.8.0.post29"
Check "a paired install keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
Set-Probe "2.13.0+rocm7.14.0" "3.9.0.post30"
Check "torch 2.13 keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
Set-Probe "2.11.0+cpu" "3.9.0.post30"
Check "a CPU torch keeps the fast path (the pass would not repair it)" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "cpu"))
Set-Probe "" "" -Ok $false
Check "an unreadable probe keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
$script:Offline = $true
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "UV_OFFLINE keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
$script:Offline = $false

Write-Host "=== ROCm torch rule (escape and repair) ==="
Check "+rocm torch is ROCm" (Test-RocmTorchInstalled -Torch "2.11.0+rocm7.2.0" -InstalledTorchTag "cpu")
Check "a ROCm venv tag with a plain version is ROCm" (Test-RocmTorchInstalled -Torch "2.11.0" -InstalledTorchTag "rocm")
Check "a ROCm venv tag with a CUDA torch is not" (-not (Test-RocmTorchInstalled -Torch "2.11.0+cu128" -InstalledTorchTag "rocm"))
Check "a CPU torch is not" (-not (Test-RocmTorchInstalled -Torch "2.11.0+cpu" -InstalledTorchTag "cpu"))
Check "nothing known is not" (-not (Test-RocmTorchInstalled -Torch "" -InstalledTorchTag $null))

Write-Host "=== metadata probe (real Python) ==="
$site = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm_triton_probe_" + [guid]::NewGuid().ToString("N"))
function New-DistInfo([string]$Folder, [string]$Name, [string]$Version) {
    $dir = Join-Path $site $Folder
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $dir "METADATA") -Value "Metadata-Version: 2.1`nName: $Name`nVersion: $Version`n" -Encoding ascii
}
try {
    New-DistInfo "torch-2.11.0+rocm7.14.0.dist-info" "torch" "2.11.0+rocm7.14.0"
    New-DistInfo "~riton_windows-3.9.0.post30.dist-info" "triton-windows" "3.9.0.post30"
    $script:RealProbeDir = $site
    $pair = Get-RocmTritonPairing
    Check "a pip backup alone is not an installed triton-windows" ($pair.Torch -eq "2.11.0+rocm7.14.0" -and $pair.Triton -eq "" -and $pair.Stale)
    New-DistInfo "triton_windows-3.8.0.post29.dist-info" "triton-windows" "3.8.0.post29"
    $pair = Get-RocmTritonPairing
    Check "the active 3.8 wins over a 3.9 pip backup" ($pair.Triton -eq "3.8.0.post29" -and -not $pair.Stale)
} finally {
    $script:RealProbeDir = $null
    Remove-Item -LiteralPath $site -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "=== wiring ==="
$stackAt = $setup.IndexOf('python "$PSScriptRoot\install_python_stack.py"')
Check "the repair runs after the dependency stack" ($stackAt -ge 0 -and $setup.IndexOf($block) -gt $stackAt)
$escape = 'if ($SkipPythonDeps -and -not $NoTorchMode -and ($installedTorchTag -match ''^(rocm|gfx)'') -and (Test-RocmTritonPairingNeedsPass -InstalledTorchTag $installedTorchTag)) {'
$escapeAt = $setup.IndexOf($escape)
$gateAt = $setup.IndexOf('if (-not $SkipPythonDeps) {')
Check "the fast path asks for the pairing (outside no-torch mode)" ($escapeAt -ge 0)
Check "the helpers are defined before the fast-path escape" ($setup.IndexOf($helpers) -lt $escapeAt)
Check "the escape runs before the dependency gate" ($escapeAt -lt $gateAt)
$chainAt = if ($escapeAt -ge 0) { $setup.LastIndexOf('# After the chain', $escapeAt) } else { -1 }
$skipAt = if ($chainAt -gt 0) { $setup.IndexOf('if ($SkipPythonDeps) {', $chainAt) } else { -1 }
$gfxAt = if ($chainAt -gt 0) { $setup.IndexOf('if ($script:ROCmGfxArch) {', $chainAt) } else { -1 }
function Get-Depth([string]$Text) { return ($Text.ToCharArray() | Where-Object { $_ -eq '{' }).Count - ($Text.ToCharArray() | Where-Object { $_ -eq '}' }).Count }
Check "the escape sits in the fast-path checks after the chain" ($skipAt -gt $chainAt -and $skipAt -lt $escapeAt -and (Get-Depth $setup.Substring($skipAt, $escapeAt - $skipAt)) -eq 1)
# Outside the gfx checks: a pinned ROCm index or a hybrid host has a ROCm torch with no detected gfx.
Check "the escape is not inside the gfx-detection block" ($gfxAt -gt $skipAt -and $gfxAt -lt $escapeAt -and (Get-Depth $setup.Substring($gfxAt, $escapeAt - $gfxAt)) -eq 0)

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
