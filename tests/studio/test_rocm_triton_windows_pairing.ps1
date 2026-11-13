#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.ps1 pairs triton-windows with torch on the AMD ROCm path after the dependency stack: the stack resolves
# unsloth's unpinned triton-windows, and 3.9 removed the block pointers torch 2.11 / 2.12's flex_decoding builds
# (gpt-oss 4k prefill 0.1 -> 6.7 GB on gfx1151). Both the repair and the fast-path check read the venv as it is
# (torch version, torch.version.hip, triton-windows, whether triton imports), never an earlier tag or index URL.
# A verified install that skips the dependency pass is sent back through it while the pair is stale. The helpers and
# the post-stack block are cut out of setup.ps1 and run against stubs. ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
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

# Stubs. The probe answers from the current venv state; Fast-Install moves that state to what the install left.
$script:State = @{}
$script:Installs = @()
$script:EnvSeen = @()
$script:Offline = $false
$script:OfflineFastPath = $false
$script:RealProbeDir = $null
$script:InstallExit = 0
$script:InstallLeaves = $null
function Invoke-BoundedPythonProbe {
    param($PythonExe, $Code, $TimeoutSec)
    if ($script:RealProbeDir) {
        # The real probe code against a site directory of hand-made packages only (-S: no site-packages).
        $py = Get-Command python -ErrorAction SilentlyContinue
        if (-not $py) { $py = Get-Command python3 }
        $savedPath = $env:PYTHONPATH
        $env:PYTHONPATH = $script:RealProbeDir
        # Continue: under Stop, Windows PowerShell 5.1 turns the failing import's redirected stderr into a
        # terminating error before the exit code is read (the probe helper in setup.ps1 reads stderr itself).
        $ErrorActionPreference = "Continue"
        try { $out = & $py.Source -S -c $Code 2>&1 | Out-String } finally { $env:PYTHONPATH = $savedPath }
        return @{ Ok = ($LASTEXITCODE -eq 0); Output = $out }
    }
    if ($Code -like "*HEALTHY=*") {
        # The runtime probe: a failing import exits non-zero. Healthy / Paired default to what a working install of
        # the recorded version gives; a state can override them (namespace package, API gone, a mismatched payload).
        if (-not $script:State.Imports) { return @{ Ok = $false; Output = "" } }
        $healthy = if ($script:State.ContainsKey('Healthy')) { $script:State.Healthy } else { $true }
        $paired = if ($script:State.ContainsKey('Paired')) { $script:State.Paired } else { $healthy -and $script:State.Triton -match '^3\.8(\.|$)' }
        return @{ Ok = $true; Output = "TRITONVER=`nHEALTHY=$([int][bool]$healthy)`nPAIRED=$([int][bool]$paired)" }
    }
    if (-not $script:State.Readable) { return @{ Ok = $false; Output = "" } }
    $dists = if ($script:State.ContainsKey('Dists')) { $script:State.Dists } elseif ($script:State.Triton) { 1 } else { 0 }
    return @{ Ok = $true; Output = "TORCH=$($script:State.Torch)`nTRITONWIN=$($script:State.Triton)`nTRITONDISTS=$dists`nHIP=$($script:State.Hip)" }
}
function Fast-Install {
    $script:Installs += ,($args -join " ")
    $script:EnvSeen += ,(@('UV_OVERRIDE', 'UV_EXCLUDE', 'UV_CONSTRAINT', 'PIP_CONSTRAINT' | ForEach-Object { [Environment]::GetEnvironmentVariable($_) }) -join "|")
    $script:HashSeen += ,(@('UV_REQUIRE_HASHES', 'PIP_REQUIRE_HASHES' | ForEach-Object { [Environment]::GetEnvironmentVariable($_) }) -join "|")
    if ($script:InstallLeaves) {
        $script:State.Triton = $script:InstallLeaves.Triton
        $script:State.Imports = $script:InstallLeaves.Imports
        foreach ($k in 'Healthy', 'Paired', 'Dists') { $script:State.Remove($k) }
        foreach ($k in 'Readable', 'Healthy', 'Paired', 'Dists') {
            if ($script:InstallLeaves.ContainsKey($k)) { $script:State[$k] = $script:InstallLeaves[$k] }
        }
    }
    $global:LASTEXITCODE = $script:InstallExit
    return ""
}
function Test-UvOfflineRequested { return $script:Offline }
# "fail": the move throws; "stays": it reports success but the original remains (cross-volume copy).
$script:MoveMode = ""
function Move-Item {
    param([string]$LiteralPath, [string]$Destination, [switch]$Force, $ErrorAction)
    if ($script:MoveMode -eq "fail") { throw "the manifest is locked" }
    if ($script:MoveMode -eq "stays") { Copy-Item -LiteralPath $LiteralPath -Destination $Destination -Force; return }
    Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination -Force -ErrorAction Stop
}
function substep { param($Text, $Color) }
function Write-StudioLine { param($Text, $ForegroundColor) }
function Redact-InstallOutput { param($Text) return $Text }
Invoke-Expression $helpers

function Set-Venv([string]$Torch, [string]$Triton, [string]$Hip = "", [bool]$Imports = $true, [bool]$Readable = $true, $Extra = $null) {
    if (-not $Hip -and $Torch -match '\+rocm(\S+)') { $Hip = $Matches[1] }
    $script:State = @{ Torch = $Torch; Triton = $Triton; Hip = $Hip; Imports = $Imports; Readable = $Readable }
    if ($Extra) { foreach ($k in $Extra.Keys) { $script:State[$k] = $Extra[$k] } }
}

$venvRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm_triton_venv_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $venvRoot -Force | Out-Null
$manifestFile = Join-Path $venvRoot "unsloth_install_manifest.json"

# Runs the post-stack block. $IndexUrl and $Tag are what earlier code left behind; the block must not read them.
function Invoke-Block([string]$Torch, [string]$Triton, [string]$Hip = "", [bool]$Imports = $true, [int]$StackExit = 0,
                      [bool]$NoTorch = $false, $IndexUrl = $null, $Tag = $null, [int]$InstallExit = 0,
                      $Leaves = @{ Triton = "3.8.0.post29"; Imports = $true }, [bool]$Manifest = $true, [string]$Move = "",
                      $Extra = $null, [bool]$Offline = $false) {
    Set-Venv $Torch $Triton $Hip $Imports -Extra $Extra
    $script:Offline = $Offline
    $script:Installs = @()
    $script:EnvSeen = @()
    $script:InstallExit = $InstallExit
    $script:InstallLeaves = $Leaves
    $script:MoveMode = $Move
    Remove-Item -LiteralPath $manifestFile -Force -ErrorAction SilentlyContinue
    if ($Manifest) { Set-Content -LiteralPath $manifestFile -Value '{"complete": true}' -Encoding ascii }
    $stackExit = $StackExit
    $ROCmIndexUrl = $IndexUrl
    $installedTorchTag = $Tag
    $NoTorchMode = $NoTorch
    $VenvDir = $venvRoot
    try { Invoke-Expression $block } finally { $script:MoveMode = ""; $script:Offline = $false }
    return @{ Installs = $script:Installs; StackExit = $stackExit; Manifest = (Test-Path -LiteralPath $manifestFile -PathType Leaf);
              Pending = @(Get-ChildItem -LiteralPath $venvRoot -Filter "*.triton-pending" -ErrorAction SilentlyContinue).Count }
}
$pin = "triton-windows>=3.8,<3.9"
function Repaired($r) { return ($r.Installs.Count -eq 1 -and $r.Installs[0] -eq $pin) }
function Untouched($r) { return ($r.Installs.Count -eq 0 -and $r.StackExit -eq 0 -and $r.Manifest) }

try {
Write-Host "=== spec per torch ==="
Check "torch 2.11 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.11.0+rocm7.14.0") -eq $pin)
Check "torch 2.12 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.12.1+rocm7.14.0") -eq $pin)
Check "torch 2.13 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.13.0+rocm7.14.0"))
Check "torch 2.10 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.10.0+rocm7.2"))
Check "unreadable torch is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion ""))

Write-Host "=== post-stack repair ==="
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "3.9 on torch 2.11 is replaced by 3.8, manifest kept" ((Repaired $r) -and $r.StackExit -eq 0 -and $r.Manifest -and $r.Pending -eq 0)
$r = Invoke-Block -Torch "2.12.0+rocm7.14.0" -Triton "3.6.0.post25"
Check "3.6 (no HIP headers) is replaced by 3.8" (Repaired $r)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -Imports $false
Check "a missing triton-windows is installed" ((Repaired $r) -and $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29"
Check "a 3.8 that imports is kept" (Untouched $r)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Imports $false
# A 3.8 already satisfies the pin, so a plain install is a no-op: only a forced reinstall restores the files.
Check "a 3.8 that no longer imports is force-reinstalled" ($r.Installs.Count -eq 1 -and $r.Installs[0] -eq "--force-reinstall --no-deps $pin" -and $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Imports $false -Leaves @{ Triton = "3.8.0.post29"; Imports = $false }
Check "a forced reinstall that still does not import is not certified" (-not $r.Manifest -and $r.Pending -eq 0)
# A post-install probe that could not read the venv verifies nothing.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Leaves @{ Triton = "3.8.0.post29"; Imports = $true; Readable = $false }
Check "an unreadable post-install probe is not certified" ((Repaired $r) -and -not $r.Manifest -and $r.Pending -eq 0)
# A manifest that cannot be set aside stays valid, so nothing is replaced.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Move "fail"
Check "a manifest that cannot be moved blocks the replacement" ($r.Installs.Count -eq 0 -and $r.Manifest -and $r.StackExit -eq 0 -and $r.Pending -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Move "stays"
Check "a move that leaves the original behind blocks it too" ($r.Installs.Count -eq 0 -and $r.Manifest -and $r.Pending -eq 0)
$r = Invoke-Block -Torch "2.13.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "torch 2.13 keeps what the stack resolved" (Untouched $r)
$r = Invoke-Block -Torch "2.11.0" -Triton "3.9.0.post30" -Hip "7.2.26015"
Check "an untagged ROCm build (torch.version.hip set) is repaired" (Repaired $r)
$r = Invoke-Block -Torch "2.11.0" -Triton "3.9.0.post30"
Check "an untagged build without hip is not touched" (Untouched $r)
# A fresh install or a CPU / CUDA -> ROCm move through an opaque custom index.
foreach ($previousTag in @($null, "cpu", "cu128")) {
    $r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -IndexUrl $null -Tag $previousTag
    Check "custom-index ROCm torch is repaired after prior tag '$previousTag'" (Repaired $r)
}
# A ROCm index URL left behind does not make a CUDA / CPU / XPU torch ROCm.
foreach ($currentTorch in @("2.11.0+cu128", "2.11.0+cpu", "2.11.0+xpu")) {
    $r = Invoke-Block -Torch $currentTorch -Triton "3.6.0.post25" -IndexUrl "https://mirror.example/rocm7.2/" -Tag "rocm"
    Check "a ROCm index URL does not touch resident '$currentTorch'" (Untouched $r)
}
# Inherited uv / pip inputs are cleared for the install and restored after.
$envNames = @('UV_OVERRIDE', 'UV_EXCLUDE', 'UV_CONSTRAINT', 'PIP_CONSTRAINT')
foreach ($n in $envNames) { Set-Item "Env:$n" "inherited-$n.txt" }
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "UV_OVERRIDE / UV_EXCLUDE / UV_CONSTRAINT / PIP_CONSTRAINT are cleared for the install" ((Repaired $r) -and $script:EnvSeen[0] -eq "|||")
Check "and restored afterwards" (@($envNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -ne "inherited-$_.txt" }).Count -eq 0)
foreach ($n in $envNames) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "unset ones stay unset" (@($envNames | Where-Object { $null -ne [Environment]::GetEnvironmentVariable($_) }).Count -eq 0)
# Require-hashes is relaxed for the unhashed pin like the dependency stack does, and put back after.
$script:HashSeen = @()
$env:UV_REQUIRE_HASHES = "1"; $env:PIP_REQUIRE_HASHES = "1"
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "UV_REQUIRE_HASHES is cleared and PIP_REQUIRE_HASHES is 0 for the install" ((Repaired $r) -and $script:HashSeen[0] -eq "|0")
Check "and both are restored afterwards" ($env:UV_REQUIRE_HASHES -eq "1" -and $env:PIP_REQUIRE_HASHES -eq "1")
Remove-Item Env:UV_REQUIRE_HASHES, Env:PIP_REQUIRE_HASHES -ErrorAction SilentlyContinue
$script:HashSeen = @()
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "an unset PIP_REQUIRE_HASHES is 0 during the install and unset after" ($script:HashSeen[0] -eq "|0" -and $null -eq [Environment]::GetEnvironmentVariable('PIP_REQUIRE_HASHES'))
# Exit 0 that leaves the old version (a uv config override) is not accepted.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Leaves @{ Triton = "3.9.0.post30"; Imports = $true }
Check "exit 0 with 3.9 still installed is not certified (manifest dropped, setup continues)" ((Repaired $r) -and -not $r.Manifest -and $r.StackExit -eq 0 -and $r.Pending -eq 0)
# A replacement that fails part way and breaks a working triton fails setup and is never certified.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -InstallExit 1 -Leaves @{ Triton = "3.8.0.post29"; Imports = $false }
Check "a failed replacement that breaks triton fails setup" ((Repaired $r) -and $r.StackExit -eq 1 -and -not $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -InstallExit 0 -Leaves @{ Triton = "3.8.0.post29"; Imports = $false }
Check "even with exit 0" ($r.StackExit -eq 1 -and -not $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -InstallExit 2 -Leaves $null
Check "a failed download that leaves a working 3.9 warns, drops the manifest, setup continues" ((Repaired $r) -and $r.StackExit -eq 0 -and -not $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -Imports $false -InstallExit 2 -Leaves $null
Check "a missing triton that cannot be installed warns only (nothing was broken)" ($r.StackExit -eq 0 -and -not $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Manifest $false
Check "no manifest to hold is fine" ((Repaired $r) -and $r.StackExit -eq 0 -and -not $r.Manifest -and $r.Pending -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -StackExit 1
Check "a failed stack is not touched" ($r.Installs.Count -eq 0 -and $r.StackExit -eq 1)
# Metadata or a namespace import is not enough; the imported runtime decides.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Extra @{ Paired = $false }
Check "3.8 metadata over a runtime that is not 3.8 is repaired" ($r.Installs.Count -eq 1 -and $r.Installs[0] -eq "--force-reinstall --no-deps $pin" -and $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Extra @{ Healthy = $false; Paired = $false }
Check "a namespace-only triton.language is not certified as paired" ($r.Installs.Count -eq 1 -and $r.Installs[0] -eq "--force-reinstall --no-deps $pin")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Extra @{ Paired = $false } -Leaves @{ Triton = "3.8.0.post29"; Imports = $true; Healthy = $true; Paired = $false }
Check "an install whose runtime still is not 3.8 is not certified" (-not $r.Manifest -and $r.Pending -eq 0)
# The health baseline and the escalation use the runtime check, not a bare import.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -InstallExit 1 -Leaves @{ Triton = "3.8.0.post29"; Imports = $true; Healthy = $false; Paired = $false }
Check "a replacement that leaves an importable triton without its API fails setup" ($r.StackExit -eq 1 -and -not $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29" -Extra @{ Healthy = $false; Paired = $false } -InstallExit 1 -Leaves @{ Triton = "3.8.0.post29"; Imports = $true; Healthy = $false; Paired = $false }
Check "a namespace-only baseline is not counted as a working triton that the repair broke" ($r.StackExit -eq 0 -and -not $r.Manifest)
# The post-stack repair honours UV_OFFLINE like the fast path.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -Offline $true
Check "UV_OFFLINE leaves the post-stack repair for the next online update" (Untouched $r)
# A record without a version and a broken payload is force-reinstalled, as a broken 3.8 is.
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -Imports $false -Extra @{ Dists = 1 }
Check "a versionless record over a broken payload is force-reinstalled" ($r.Installs.Count -eq 1 -and $r.Installs[0] -eq "--force-reinstall --no-deps $pin" -and $r.Manifest)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -NoTorch $true
Check "no-torch mode leaves an existing Triton alone" (Untouched $r)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -NoTorch $true
Check "no-torch mode installs no Triton" (Untouched $r)

Write-Host "=== fast-path check (same probe) ==="
Set-Venv "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "verified install with 3.9 on ROCm torch 2.11 runs the pass" (Test-RocmTritonPairingNeedsPass)
Set-Venv "2.11.0+rocm7.14.0" "" -Imports $false
Check "a repair that failed last run (no triton-windows) is retried" (Test-RocmTritonPairingNeedsPass)
Set-Venv "2.11.0+rocm7.14.0" "3.8.0.post29" -Imports $false
Check "a 3.8 that no longer imports is retried" (Test-RocmTritonPairingNeedsPass)
Set-Venv "2.11.0" "3.9.0.post30" -Hip "7.2.26015"
Check "an untagged ROCm build is checked" (Test-RocmTritonPairingNeedsPass)
Set-Venv "2.11.0+rocm7.14.0" "3.8.0.post29"
Check "a paired install keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass))
Set-Venv "2.13.0+rocm7.14.0" "3.9.0.post30"
Check "torch 2.13 keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass))
foreach ($t in @("2.11.0+cpu", "2.11.0+cu128", "2.11.0+xpu")) {
    Set-Venv $t "3.9.0.post30"
    Check "$t keeps the fast path (the pass would not repair it)" (-not (Test-RocmTritonPairingNeedsPass))
}
Set-Venv "" "" -Readable $false
Check "an unreadable probe keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass))
$script:Offline = $true
Set-Venv "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "UV_OFFLINE keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass))
$script:Offline = $false

Write-Host "=== probe code (real Python) ==="
$site = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm_triton_probe_" + [guid]::NewGuid().ToString("N"))
function New-DistInfo([string]$Folder, [string]$Name, [string]$Version) {
    $dir = Join-Path $site $Folder
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    $meta = "Metadata-Version: 2.1`nName: $Name`n"
    if ($Version) { $meta += "Version: $Version`n" }
    Set-Content -LiteralPath (Join-Path $dir "METADATA") -Value $meta -Encoding ascii
}
function New-TorchPackage([string]$HipLine) {
    $dir = Join-Path $site "torch"
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $dir "__init__.py") -Value "raise ImportError('the probe must not import torch')" -Encoding ascii
    Set-Content -LiteralPath (Join-Path $dir "version.py") -Value "from typing import Optional`n__version__ = '2.11.0'`ncuda: Optional[str] = None`n$HipLine`n" -Encoding ascii
}
function New-TritonPackage([string]$Version = "3.8.0", [switch]$Stub, [switch]$Namespace, [string]$Doc = "") {
    $lang = Join-Path $site "triton/language"
    Remove-Item -LiteralPath (Join-Path $site "triton") -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $lang -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $site "triton/__init__.py") -Value "__version__ = '$Version'`ndef jit(fn):`n    return fn`n" -Encoding ascii
    if ($Namespace) { return }
    # Like triton: make_block_ptr sits behind a functools.wraps decorator; 3.9 left a raise-only stub.
    $body = if ($Stub) { "    raise ValueError('Block pointers have been removed in favor of the tensor descriptor API')" } else { "    return base" }
    $docLine = if ($Doc) { "    `"$Doc`"`n" } else { "" }
    $src = "import functools`ndef builtin(fn):`n    @functools.wraps(fn)`n    def wrapper(*a, **k):`n        return fn(*a, **k)`n    return wrapper`n" +
        "def program_id(axis):`n    return 0`n@builtin`ndef make_block_ptr(base, shape=None):`n$docLine$body`n"
    Set-Content -LiteralPath (Join-Path $lang "__init__.py") -Value $src -Encoding ascii
}
try {
    New-DistInfo "torch-2.11.0+rocm7.14.0.dist-info" "torch" "2.11.0+rocm7.14.0"
    New-TorchPackage "hip: Optional[str] = None"
    New-DistInfo "~riton_windows-3.9.0.post30.dist-info" "triton-windows" "3.9.0.post30"
    $script:RealProbeDir = $site
    $pair = Get-RocmTritonPairing
    Check "a pip backup alone is not an installed triton-windows" ($pair.Torch -eq "2.11.0+rocm7.14.0" -and $pair.Triton -eq "" -and $pair.Stale)
    New-DistInfo "triton_windows-3.8.0.post29.dist-info" "triton-windows" "3.8.0.post29"
    $pair = Get-RocmTritonPairing
    Check "an active 3.8 that does not import is stale" ($pair.Triton -eq "3.8.0.post29" -and $pair.Stale)
    New-TritonPackage
    $pair = Get-RocmTritonPairing
    Check "the active 3.8 that imports wins over a 3.9 pip backup" ($pair.Triton -eq "3.8.0.post29" -and -not $pair.Stale)
    # With real Python: the runtime, not the metadata, decides.
    New-TritonPackage -Version "3.9.0" -Stub
    $pair = Get-RocmTritonPairing
    Check "3.8 metadata over a 3.9 runtime is stale" ($pair.Triton -eq "3.8.0.post29" -and $pair.Stale -and $pair.Healthy)
    New-TritonPackage -Version "3.8.0" -Stub
    Check "a runtime whose make_block_ptr is the removed stub is stale" ((Get-RocmTritonPairing).Stale)
    New-TritonPackage -Version "3.8.0" -Doc "Block pointers have been removed elsewhere; this one works"
    Check "a working make_block_ptr that only mentions the removal in its docstring is paired" (-not (Get-RocmTritonPairing).Stale)
    New-TritonPackage -Version "3.8.0" -Namespace
    $pair = Get-RocmTritonPairing
    Check "a namespace-only triton.language is neither paired nor healthy" ($pair.Stale -and -not $pair.Healthy)
    foreach ($pre in @("3.8.0.dev20261001", "3.8.0rc1")) {
        New-TritonPackage -Version $pre
        Check "a $pre runtime is not a 3.8 release" ((Get-RocmTritonPairing).Stale)
    }
    New-TritonPackage -Version "3.8.0"
    # A leftover or versionless record does not hide the 3.8 record.
    New-DistInfo "triton_windows-0.dist-info" "triton-windows" ""
    New-DistInfo "triton_windows-3.9.0.post30.dist-info" "triton-windows" "3.9.0.post30"
    $pair = Get-RocmTritonPairing
    Check "a versionless and a leftover 3.9 record do not hide the 3.8 record" ($pair.Triton -eq "3.8.0.post29" -and -not $pair.Stale -and $pair.Dists -eq 3)
    Remove-Item -LiteralPath (Join-Path $site "triton_windows-0.dist-info") -Recurse -Force
    Remove-Item -LiteralPath (Join-Path $site "triton_windows-3.9.0.post30.dist-info") -Recurse -Force
    # A dist-info without Version reads as missing and keeps the torch version.
    Remove-Item -LiteralPath (Join-Path $site "triton_windows-3.8.0.post29.dist-info") -Recurse -Force
    New-DistInfo "triton_windows-3.9.0.dist-info" "triton-windows" ""
    $pair = Get-RocmTritonPairing
    Check "triton-windows without Version reads as missing, torch still read" ($pair.Torch -eq "2.11.0+rocm7.14.0" -and $pair.Triton -eq "" -and $pair.Dists -eq 1)
    # torch.version.hip, read without importing torch.
    Remove-Item -LiteralPath (Join-Path $site "torch-2.11.0+rocm7.14.0.dist-info") -Recurse -Force
    New-DistInfo "torch-2.11.0.dist-info" "torch" "2.11.0"
    New-TorchPackage "hip: Optional[str] = '7.2.26015-abc'"
    $pair = Get-RocmTritonPairing
    Check "an untagged torch with hip in version.py is ROCm" ($pair.Torch -eq "2.11.0" -and $pair.Spec -eq $pin)
    New-TorchPackage "hip: Optional[str] = None"
    $pair = Get-RocmTritonPairing
    Check "an untagged torch without hip is not" ($null -eq $pair.Spec -and -not $pair.Stale)
} finally {
    $script:RealProbeDir = $null
    Remove-Item -LiteralPath $site -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "=== wiring ==="
$repairLine = [regex]::Match($block, 'Repair with: (python[^\r\n]*?)" -ForegroundColor').Groups[1].Value
$repairCmd = $repairLine.Replace('`"', '"').Replace('$($_rocmPair.Spec)', $pin)
$parseErrors = $null
[System.Management.Automation.PSParser]::Tokenize($repairCmd, [ref]$parseErrors) | Out-Null
Check "the printed repair command is valid PowerShell (the pin is quoted)" ($repairCmd -match [regex]::Escape("`"$pin`"") -and @($parseErrors).Count -eq 0)
$stackAt = $setup.IndexOf('python "$PSScriptRoot\install_python_stack.py"')
Check "the repair runs after the dependency stack" ($stackAt -ge 0 -and $setup.IndexOf($block) -gt $stackAt)
Check "the repair reads neither the old tag nor the ROCm index URL" ($block -notmatch '\$installedTorchTag|\$ROCmIndexUrl')
$escape = 'if ($SkipPythonDeps -and -not $NoTorchMode -and (Test-RocmTritonPairingNeedsPass)) {'
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
} finally {
    Remove-Item -LiteralPath $venvRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
