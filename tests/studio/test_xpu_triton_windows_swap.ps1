#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.ps1 replaces triton-windows with torch's own XPU triton on the Intel XPU path (both own the `triton`
# package). The swap runs with inherited uv / pip resolver inputs cleared, as the AMD ROCm pairing does, and an
# install counts only once a fresh probe shows the XPU dist recorded, no triton-windows left and triton importing;
# the triton-windows fallback is verified the same way. Both the fast-path check and the swap find dists by name,
# so a ~riton_windows-* backup an interrupted pip leaves is not an installed triton-windows, and both also ask
# whether triton imports, so a backup with the files gone is still repaired. A failed uninstall puts the manifest
# back only while triton imports. The helpers, the
# fast-path check and the post-stack swap are cut out of setup.ps1 and run against stubs and real Python.
# ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
# Run: pwsh -NoProfile -File tests/studio/test_xpu_triton_windows_swap.ps1

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
$fastPath = Cut "    # An explicit xpu pin can leave IsIntelXpu false" '    $script:SkipPythonDeps = $SkipPythonDeps'
$swap = Cut '# Both own the `triton` package.' '$ErrorActionPreference = $prevEAP'

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}
# A scenario that throws (a helper setup.ps1 no longer defines) is a failed check, not an aborted run.
function Try-Scenario([string]$Name, [scriptblock]$Body) {
    try { & $Body } catch { Check "$Name (threw: $($_.Exception.Message))" $false; return $null }
}

# Stubs. The probes answer from $script:State; the installers move it to what each call leaves.
$script:State = @{}
$script:RealProbeDir = $null
$script:Calls = @()
$script:EnvSeen = @()
$script:Lines = @()
$script:InstallExits = @()
$script:InstallLeaves = @()
$script:UninstallExit = 0
$script:UninstallBreaks = $false
$envNames = @('UV_OVERRIDE', 'UV_EXCLUDE', 'UV_CONSTRAINT', 'PIP_CONSTRAINT', 'UV_REQUIRE_HASHES', 'PIP_REQUIRE_HASHES')
function Invoke-BoundedPythonProbe {
    param($PythonExe, $Code, $TimeoutSec)
    if ($script:RealProbeDir) {
        # The real probe code against a site directory of hand-made dist-infos only (-S: no site-packages).
        $py = Get-Command python -ErrorAction SilentlyContinue
        if (-not $py) { $py = Get-Command python3 }
        $savedPath = $env:PYTHONPATH
        $env:PYTHONPATH = $script:RealProbeDir
        $ErrorActionPreference = "Continue"
        try { $out = & $py.Source -S -c $Code 2>&1 | Out-String } finally { $env:PYTHONPATH = $savedPath }
        return @{ Ok = ($LASTEXITCODE -eq 0); Output = $out }
    }
    if ($Code -like "*XPURT=*") {
        # The imported triton is the XPU payload (its version and Intel backend), unless the state says otherwise.
        if (-not $script:State.Imports) { return @{ Ok = $false; Output = "" } }
        $rt = if ($script:State.ContainsKey('XpuRt')) { [int][bool]$script:State.XpuRt } else { 1 }
        return @{ Ok = $true; Output = "XPURT=$rt" }
    }
    if ($Code -like "*HEALTHY=*") {
        if (-not $script:State.Imports) { return @{ Ok = $false; Output = "" } }
        return @{ Ok = $true; Output = "TRITONVER=`nHEALTHY=1`nPAIRED=0" }
    }
    # A probe that fails part way still printed the fields it read first.
    if ($script:State.PartialSpec) { return @{ Ok = $false; Output = "TRITONWIN=$($script:State.TritonWin)`nTRITONXPU=$($script:State.PartialSpec)`n" } }
    if (-not $script:State.Readable) { return @{ Ok = $false; Output = "" } }
    # XPUOK: the stubbed venv's torch requires 3.5.0.
    $ok = if ($script:State.XpuDist -eq "3.5.0") { 1 } else { 0 }
    return @{ Ok = $true; Output = "TRITONWIN=$($script:State.TritonWin)`nTRITONXPU=$($script:State.XpuSpec)`nXPUDIST=$($script:State.XpuDist)`nXPUOK=$ok" }
}
function Save-EnvSeen { $script:EnvSeen += ,(@($envNames | ForEach-Object { [Environment]::GetEnvironmentVariable($_) }) -join "|") }
$script:DownloadExit = 0
function Fast-Download {
    $script:Calls += ,("download " + ($args -join " "))
    Save-EnvSeen
    $dest = $args[[array]::IndexOf($args, '--dest') + 1]
    if ($script:DownloadExit -eq 0) {
        Set-Content -LiteralPath (Join-Path $dest "pytorch_triton_xpu-3.5.0-cp312-cp312-win_amd64.whl") -Value "" -Encoding ascii
    }
    $global:LASTEXITCODE = $script:DownloadExit
    return ""
}
function Fast-Uninstall {
    $script:Calls += ,("uninstall " + ($args -join " "))
    Save-EnvSeen
    if ("$args" -like "*pytorch-triton-xpu*") { $script:State.XpuDist = ""; $script:State.Imports = $false }
    elseif ($script:UninstallExit -eq 0) { $script:State.TritonWin = ""; $script:State.Imports = $false }
    # A failed uninstall that already removed some of the files.
    elseif ($script:UninstallBreaks) { $script:State.Imports = $false }
    $global:LASTEXITCODE = $script:UninstallExit
    return ""
}
function Fast-Install {
    $n = @($script:Calls | Where-Object { $_ -like "install *" }).Count
    $script:Calls += ,("install " + ($args -join " "))
    Save-EnvSeen
    if ($n -lt $script:InstallLeaves.Count -and $script:InstallLeaves[$n]) {
        foreach ($k in $script:InstallLeaves[$n].Keys) { $script:State[$k] = $script:InstallLeaves[$n][$k] }
    }
    $global:LASTEXITCODE = if ($n -lt $script:InstallExits.Count) { $script:InstallExits[$n] } else { 0 }
    return ""
}
function substep { param($Text, $Color) $script:Lines += ,"$Text" }
function Write-StudioLine { param($Text, $ForegroundColor) $script:Lines += ,"$Text" }
function Redact-InstallOutput { param($Text) return $Text }
$script:Offline = $false
function Test-UvOfflineRequested { return $script:Offline }
Invoke-Expression $helpers

$venvRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("xpu_triton_venv_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $venvRoot -Force | Out-Null
$manifestFile = Join-Path $venvRoot "unsloth_install_manifest.json"
$xpuSpec = "pytorch-triton-xpu==3.5.0"
$generic = "triton-windows<3.7"
$xpuOk = @{ XpuDist = "3.5.0"; Imports = $true; XpuRt = $true }
$backOk = @{ TritonWin = "3.6.0.post25"; Imports = $true; Readable = $true }

# Runs the post-stack swap from a venv where triton-windows shadows torch's XPU triton.
function Invoke-Swap($Exits = @(), $Leaves = @($xpuOk), [string]$TritonWin = "3.9.0.post30", [string]$Spec = $xpuSpec,
                     [int]$UninstallExit = 0, [bool]$UninstallBreaks = $false, [bool]$Imports = $true,
                     [string]$XpuDist = "", [int]$DownloadExit = 0, [bool]$Managed = $false, [bool]$Offline = $false,
                     $Extra = $null) {
    $script:State = @{ TritonWin = $TritonWin; XpuSpec = $Spec; XpuDist = $XpuDist; Imports = $Imports; Readable = $true }
    if ($Extra) { foreach ($k in $Extra.Keys) { $script:State[$k] = $Extra[$k] } }
    $script:DownloadExit = $DownloadExit
    $script:Offline = $Offline
    $InstallerManagedSetup = $Managed
    $script:Calls = @(); $script:EnvSeen = @(); $script:Lines = @()
    $script:InstallExits = $Exits; $script:InstallLeaves = $Leaves; $script:UninstallExit = $UninstallExit
    $script:UninstallBreaks = $UninstallBreaks
    Set-Content -LiteralPath $manifestFile -Value '{"complete": true}' -Encoding ascii
    $stackExit = 0
    $XpuIndexUrl = "https://download.pytorch.org/whl/xpu"
    $VenvDir = $venvRoot
    $_tritonSpec = $generic
    Invoke-Expression $swap
    $script:DownloadExit = 0
    $script:Offline = $false
    return @{ Calls = $script:Calls; StackExit = $stackExit; Manifest = (Test-Path -LiteralPath $manifestFile -PathType Leaf);
              Installs = @($script:Calls | Where-Object { $_ -like "install *" }); Lines = $script:Lines }
}
function Has-Error($r) { return @($r.Lines | Where-Object { $_ -like "*[[]ERROR]*" }).Count -gt 0 }

try {
Write-Host "=== swap: verified, not just exit 0 ==="
$r = Try-Scenario "happy path" { Invoke-Swap }
Check "a verified XPU install keeps the manifest and installs once" ($r -and $r.Installs.Count -eq 1 -and $r.Manifest -and $r.StackExit -eq 0)
Check "and runs fetch, uninstall, install in order" ($r -and $r.Calls.Count -eq 3 -and $r.Calls[0] -like "download *" -and $r.Calls[1] -like "uninstall *" -and $r.Calls[2] -like "install *")
# An inherited UV_EXCLUDE drops the wheel and the install still exits 0, with triton-windows already gone.
$r = Try-Scenario "exit 0, nothing installed" { Invoke-Swap -Leaves @($null, $backOk) }
Check "exit 0 with no XPU dist falls back to triton-windows" ($r -and $r.Installs.Count -eq 2 -and $r.Installs[1] -eq "install --force-reinstall --no-deps $generic")
Check "and the restored triton-windows keeps setup going" ($r -and $r.StackExit -eq 0 -and $r.Manifest -and -not (Has-Error $r))
$r = Try-Scenario "XPU dist recorded, no import" { Invoke-Swap -Leaves @(@{ XpuDist = "3.5.0"; Imports = $false }, $backOk) }
Check "an XPU dist that does not import falls back" ($r -and $r.Installs.Count -eq 2 -and $r.StackExit -eq 0)
$r = Try-Scenario "triton-windows still recorded" { Invoke-Swap -Leaves @(@{ XpuDist = "3.5.0"; Imports = $true; TritonWin = "3.9.0.post30" }, $backOk) }
Check "an install that leaves triton-windows recorded falls back" ($r -and $r.Installs.Count -eq 2)
$r = Try-Scenario "unreadable after install" { Invoke-Swap -Leaves @(@{ XpuDist = "3.5.0"; Imports = $true; Readable = $false }, $backOk) }
Check "a post-install probe that cannot read the venv verifies nothing" ($r -and $r.Installs.Count -eq 2)
$r = Try-Scenario "XPU install fails" { Invoke-Swap -Exits @(1, 0) -Leaves @($null, $backOk) }
Check "a failed XPU install restores triton-windows and the manifest" ($r -and $r.Installs.Count -eq 2 -and $r.StackExit -eq 0 -and $r.Manifest)

Write-Host "=== fallback: verified too ==="
# The fallback exits 0 but triton still does not import: the venv has no working triton.
$r = Try-Scenario "fallback exit 0, no import" { Invoke-Swap -Exits @(1, 0) -Leaves @($null, $null) }
Check "a fallback that exits 0 without a working triton fails setup" ($r -and $r.StackExit -ne 0 -and (Has-Error $r))
Check "and the manifest is not restored" ($r -and -not $r.Manifest)
$r = Try-Scenario "fallback fails" { Invoke-Swap -Exits @(1, 7) -Leaves @($null, $null) }
Check "a failed fallback fails setup with its exit code" ($r -and $r.StackExit -eq 7 -and (Has-Error $r) -and -not $r.Manifest)

Write-Host "=== swap: resolver inputs isolated ==="
foreach ($n in $envNames) { Set-Item "Env:$n" "inherited-$n" }
$r = Try-Scenario "env isolation" { Invoke-Swap -Leaves @($null, $backOk) }
$clean = "|||||0"
Check "fetch, uninstalls, install and fallback run with UV_OVERRIDE / UV_EXCLUDE / UV_CONSTRAINT / PIP_CONSTRAINT cleared and hashes relaxed" ($r -and $script:EnvSeen.Count -eq 5 -and @($script:EnvSeen | Where-Object { $_ -ne $clean }).Count -eq 0)
Check "and every inherited value is restored afterwards" (@($envNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -ne "inherited-$_" }).Count -eq 0)
foreach ($n in $envNames) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
$r = Try-Scenario "unset env" { Invoke-Swap }
Check "unset ones stay unset after the swap" ($r -and @($envNames | Where-Object { $null -ne [Environment]::GetEnvironmentVariable($_) }).Count -eq 0)

Write-Host "=== swap: when it does not apply ==="
$r = Try-Scenario "no triton-windows" { Invoke-Swap -TritonWin "" -XpuDist "3.5.0" }
Check "no triton-windows, the XPU triton torch requires: nothing is touched" ($r -and $r.Calls.Count -eq 0 -and $r.Manifest)
$r = Try-Scenario "not the xpu wheel" { Invoke-Swap -Spec "triton==3.5.0" }
Check "a torch that does not require an XPU triton: nothing is touched" ($r -and $r.Calls.Count -eq 0 -and $r.Manifest)
$r = Try-Scenario "uninstall fails" { Invoke-Swap -UninstallExit 1 }
Check "a failed uninstall installs nothing and keeps the manifest" ($r -and $r.Installs.Count -eq 0 -and $r.Manifest -and $r.StackExit -eq 0)

Write-Host "=== swap: a failed uninstall that broke triton ==="
$r = Try-Scenario "uninstall fails part way" { Invoke-Swap -UninstallExit 5 -UninstallBreaks $true }
Check "a failed uninstall that leaves triton unimportable fails setup with its exit code" ($r -and $r.StackExit -eq 5 -and (Has-Error $r))
Check "and the manifest is not restored" ($r -and -not $r.Manifest -and $r.Installs.Count -eq 0)
$r = Try-Scenario "uninstall fails, broken before" { Invoke-Swap -UninstallExit 5 -UninstallBreaks $true -Imports $false }
Check "a triton already broken before the swap warns, keeps setup going and withholds the manifest" ($r -and $r.StackExit -eq 0 -and -not (Has-Error $r) -and -not $r.Manifest)

Write-Host "=== swap: a triton that does not import, with no triton-windows record ==="
$r = Try-Scenario "repair, no record" { Invoke-Swap -TritonWin "" -Imports $false }
Check "a damaged triton with no triton-windows record is reinstalled from the XPU wheel" ($r -and $r.Installs.Count -eq 1 -and $r.Calls[0] -like "download *" -and $r.StackExit -eq 0 -and $r.Manifest)
Check "and nothing is uninstalled" ($r -and @($r.Calls | Where-Object { $_ -like "uninstall *" }).Count -eq 0)
$r = Try-Scenario "repair fails" { Invoke-Swap -TritonWin "" -Imports $false -Exits @(1, 1) -Leaves @($null, $null) }
Check "a repair that cannot fix an already broken triton warns, keeps setup going and withholds the manifest" ($r -and $r.Installs.Count -eq 2 -and $r.StackExit -eq 0 -and -not (Has-Error $r) -and -not $r.Manifest)


Write-Host "=== swap: the XPU triton dist itself ==="
$r = Try-Scenario "no XPU dist, generic files import" { Invoke-Swap -TritonWin "" }
Check "an importable triton with no XPU triton dist recorded is replaced by the XPU wheel" ($r -and $r.Installs.Count -eq 1 -and $r.Calls[0] -like "download *" -and $r.StackExit -eq 0 -and $r.Manifest)
$r = Try-Scenario "wrong XPU version" { Invoke-Swap -TritonWin "" -XpuDist "3.4.0" }
Check "an XPU triton torch does not accept is reinstalled, with nothing to uninstall" ($r -and $r.Installs.Count -eq 1 -and @($r.Calls | Where-Object { $_ -like "uninstall *" }).Count -eq 0 -and $r.StackExit -eq 0)
$r = Try-Scenario "installs the wrong version" { Invoke-Swap -Leaves @(@{ XpuDist = "3.4.0"; Imports = $true }, $backOk) }
Check "an install that leaves a version torch does not accept is not verified and falls back" ($r -and $r.Installs.Count -eq 2 -and $r.Installs[1] -eq "install --force-reinstall --no-deps $generic")

Write-Host "=== swap: a dead mirror for a triton already broken ==="
$r = Try-Scenario "download fails, broken" { Invoke-Swap -TritonWin "" -Imports $false -DownloadExit 1 }
Check "a failed fetch for a triton that does not import withholds the manifest" ($r -and $r.Installs.Count -eq 0 -and -not $r.Manifest -and $r.StackExit -eq 0)
$r = Try-Scenario "download fails, working" { Invoke-Swap -DownloadExit 1 }
Check "a failed fetch for a working triton keeps the manifest" ($r -and $r.Installs.Count -eq 0 -and $r.Manifest -and $r.StackExit -eq 0)

Write-Host "=== swap: installer-managed setup keeps its rollback copy ==="
$r = Try-Scenario "managed, uninstall fails, broken before" { Invoke-Swap -UninstallExit 5 -UninstallBreaks $true -Imports $false -Managed $true }
Check "installer-managed: a failed uninstall leaving triton broken fails setup even when it was broken before" ($r -and $r.StackExit -ne 0 -and (Has-Error $r))
$r = Try-Scenario "managed, both installs fail, broken before" { Invoke-Swap -TritonWin "" -Imports $false -Exits @(1, 1) -Leaves @($null, $null) -Managed $true }
Check "installer-managed: a repair and fallback that both fail an already broken triton fail setup" ($r -and $r.StackExit -ne 0 -and (Has-Error $r))
$r = Try-Scenario "managed, download fails, broken" { Invoke-Swap -TritonWin "" -Imports $false -DownloadExit 1 -Managed $true }
Check "installer-managed: a failed fetch for a broken triton fails setup" ($r -and $r.StackExit -ne 0)
$r = Try-Scenario "managed, happy path" { Invoke-Swap -Managed $true }
Check "installer-managed: a verified swap keeps setup going" ($r -and $r.StackExit -eq 0 -and $r.Manifest)
$r = Try-Scenario "managed, uninstall fails, still imports" { Invoke-Swap -UninstallExit 1 -Managed $true }
Check "installer-managed: a failed uninstall that left triton importing keeps setup going" ($r -and $r.StackExit -eq 0)

Write-Host "=== swap: generic files under the XPU record ==="
$r = Try-Scenario "generic payload, XPU record" {
    Invoke-Swap -TritonWin "" -XpuDist "3.5.0" -Leaves @($xpuOk) -Extra @{ XpuRt = $false }
}
Check "an importable generic triton under the XPU dist's record is reinstalled from the XPU wheel" ($r -and $r.Installs.Count -eq 1 -and $r.Calls[0] -like "download *" -and $r.StackExit -eq 0 -and $r.Manifest)

Write-Host "=== swap: unreadable metadata ==="
$r = Try-Scenario "managed, metadata unreadable, broken" { Invoke-Swap -TritonWin "" -Imports $false -Managed $true -Extra @{ Readable = $false } }
Check "installer-managed: a broken triton fails setup even when the metadata probe could not read torch's requirement" ($r -and $r.StackExit -ne 0 -and (Has-Error $r))
$r = Try-Scenario "metadata fails part way" { Invoke-Swap -TritonWin "" -Extra @{ PartialSpec = $xpuSpec } }
Check "a metadata probe that fails after reading torch's requirement still repairs from the XPU wheel" ($r -and $r.Calls.Count -ge 2 -and $r.Calls[0] -like "download *")

Write-Host "=== swap: the triton-windows fallback ==="
$r = Try-Scenario "fallback after an unverified XPU install" { Invoke-Swap -Leaves @(@{ XpuDist = "3.5.0"; Imports = $true; Readable = $false }, $backOk) }
Check "the XPU dist is uninstalled before the triton-windows fallback, so its record does not stay over generic files" ($r -and @($r.Calls | Where-Object { $_ -like "uninstall *pytorch-triton-xpu*" }).Count -eq 1 -and $r.Calls[-1] -eq "install --force-reinstall --no-deps $generic" -and $r.StackExit -eq 0)
$r = Try-Scenario "fallback installs nothing" { Invoke-Swap -Exits @(1, 0) -Leaves @($null, @{ Imports = $true }) }
Check "a fallback that exits 0 but leaves no triton-windows record is not verified" ($r -and $r.StackExit -ne 0 -and -not $r.Manifest)
$r = Try-Scenario "managed, fallback replaces an XPU triton" { Invoke-Swap -TritonWin "" -XpuDist "3.4.0" -Exits @(1, 0) -Leaves @($null, $backOk) -Managed $true }
Check "installer-managed: falling back to triton-windows over a triton that was not triton-windows fails setup" ($r -and $r.StackExit -ne 0 -and (Has-Error $r))
$r = Try-Scenario "fallback replaces an XPU triton" { Invoke-Swap -TritonWin "" -XpuDist "3.4.0" -Exits @(1, 0) -Leaves @($null, $backOk) }
Check "a direct update that falls back over a non-triton-windows triton warns and keeps going" ($r -and $r.StackExit -eq 0 -and -not (Has-Error $r))

Write-Host "=== swap: installer-managed setup, offline ==="
$r = Try-Scenario "managed, offline, broken" { Invoke-Swap -TritonWin "" -Imports $false -Offline $true -Managed $true }
Check "installer-managed, offline: a broken triton left for the next online update still fails setup" ($r -and $r.Calls.Count -eq 0 -and $r.StackExit -ne 0 -and (Has-Error $r))
$r = Try-Scenario "managed, offline, triton-windows importing" { Invoke-Swap -Offline $true -Managed $true }
Check "installer-managed, offline: a triton-windows that imports is left in place and setup keeps going" ($r -and $r.Calls.Count -eq 0 -and $r.StackExit -eq 0 -and $r.Manifest)

Write-Host "=== swap: offline ==="
$r = Try-Scenario "offline, broken" { Invoke-Swap -TritonWin "" -Imports $false -Offline $true }
Check "offline: a broken triton is left for the next online update, nothing fetched" ($r -and $r.Calls.Count -eq 0 -and $r.StackExit -eq 0)
$r = Try-Scenario "offline, triton-windows" { Invoke-Swap -Offline $true }
Check "offline: triton-windows is left for the next online update, nothing fetched" ($r -and $r.Calls.Count -eq 0)

Write-Host "=== probe code (real Python): pip backups are not installed dists ==="
$site = Join-Path ([System.IO.Path]::GetTempPath()) ("xpu_triton_probe_" + [guid]::NewGuid().ToString("N"))
function New-DistInfo([string]$Folder, [string]$Name, [string]$Version, [string]$Requires = "") {
    $dir = Join-Path $site $Folder
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    $meta = "Metadata-Version: 2.1`nName: $Name`n"
    if ($Version) { $meta += "Version: $Version`n" }
    if ($Requires) { $meta += "Requires-Dist: $Requires`n" }
    Set-Content -LiteralPath (Join-Path $dir "METADATA") -Value $meta -Encoding ascii
}
try {
    New-DistInfo "torch-2.9.0+xpu.dist-info" "torch" "2.9.0+xpu" "$xpuSpec; platform_system == 'Windows'"
    New-DistInfo "pytorch_triton_xpu-3.5.0.dist-info" "pytorch-triton-xpu" "3.5.0"
    New-DistInfo "bitsandbytes-0.50.0.dist-info" "bitsandbytes" "0.50.0"
    New-DistInfo "~riton_windows-3.9.0.post30.dist-info" "triton-windows" "3.9.0.post30"
    # The XPU triton's files: a package whose triton.language is real and has the core API.
    $tl = Join-Path (Join-Path $site "triton") "language"
    New-Item -ItemType Directory -Path $tl -Force | Out-Null
    Set-Content -LiteralPath (Join-Path (Join-Path $site "triton") "__init__.py") -Value "__version__ = '3.5.0'`ndef jit(f):`n    return f" -Encoding ascii
    Set-Content -LiteralPath (Join-Path $tl "__init__.py") -Value "def program_id(axis):`n    return 0" -Encoding ascii
    # The XPU payload carries the Intel backend; generic triton / triton-windows files do not.
    $intel = Join-Path (Join-Path (Join-Path $site "triton") "backends") "intel"
    New-Item -ItemType Directory -Path $intel -Force | Out-Null
    Set-Content -LiteralPath (Join-Path (Split-Path $intel) "__init__.py") -Value "" -Encoding ascii
    Set-Content -LiteralPath (Join-Path $intel "__init__.py") -Value "" -Encoding ascii
    $script:RealProbeDir = $site
    $s = Try-Scenario "runtime probe, payload present" { Get-TritonRuntimeState }
    Check "runtime probe: the fixture triton imports" ($s -and $s.Healthy)

    # The fast-path check: a verified XPU install with only a backup left is not sent through the dependency pass.
    $r = Try-Scenario "fast path, backup only" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: a ~riton_windows backup alone does not force the dependency pass" ($r -eq $true)
    $s = Try-Scenario "swap probe, backup only" { Get-XpuTritonState }
    Check "swap probe: a ~riton_windows backup alone is not a triton-windows to replace" ($s -and $s.Known -and $s.TritonWin -eq "" -and $s.XpuSpec -eq $xpuSpec -and $s.XpuDist -eq "3.5.0")
    $r = Try-Scenario "swap, backup only" {
        $script:Calls = @()
        Set-Content -LiteralPath $manifestFile -Value '{"complete": true}' -Encoding ascii
        $stackExit = 0; $XpuIndexUrl = "https://download.pytorch.org/whl/xpu"; $VenvDir = $venvRoot; $_tritonSpec = $generic
        Invoke-Expression $swap
        @{ Calls = $script:Calls.Count; StackExit = $stackExit }
    }
    Check "swap: nothing is fetched or removed for a backup alone" ($r -and $r.Calls -eq 0 -and $r.StackExit -eq 0)

    # The same metadata with the files gone: an interrupted replacement that took the payload with it.
    Rename-Item -LiteralPath (Join-Path $site "triton") -NewName "triton_gone"
    $r = Try-Scenario "fast path, backup only, payload missing" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: a backup with no triton files forces the dependency pass" ($r -eq $false)
    $r = Try-Scenario "swap, backup only, payload missing" {
        $script:Calls = @(); $script:InstallExits = @(); $script:InstallLeaves = @()
        Set-Content -LiteralPath $manifestFile -Value '{"complete": true}' -Encoding ascii
        $stackExit = 0; $XpuIndexUrl = "https://download.pytorch.org/whl/xpu"; $VenvDir = $venvRoot; $_tritonSpec = $generic
        Invoke-Expression $swap
        @{ Calls = $script:Calls }
    }
    Check "swap: a backup with no triton files is repaired from the XPU wheel, with no triton-windows to uninstall" ($r -and $r.Calls.Count -ge 2 -and $r.Calls[0] -like "download *" -and $r.Calls[1] -like "install --force-reinstall --no-deps *pytorch_triton_xpu*" -and @($r.Calls | Where-Object { $_ -like "uninstall triton-windows*" }).Count -eq 0)
    $script:Offline = $true
    $r = Try-Scenario "fast path, payload missing, offline" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    $script:Offline = $false
    Check "fast path, offline: a backup with no triton files is left for the next online update" ($r -eq $true)
    Rename-Item -LiteralPath (Join-Path $site "triton_gone") -NewName "triton"

    # Generic triton files that import, with the XPU dist that should own them missing or at another version.
    $xpuInfo = Join-Path $site "pytorch_triton_xpu-3.5.0.dist-info"
    Rename-Item -LiteralPath $xpuInfo -NewName "parked_xpu"
    $r = Try-Scenario "fast path, no XPU dist" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: an importable triton without torch's XPU triton dist forces the dependency pass" ($r -eq $false)
    $s = Try-Scenario "swap probe, no XPU dist" { Get-XpuTritonState }
    Check "swap probe: no XPU triton dist is not XpuOk" ($s -and $s.Known -and -not $s.XpuOk -and $s.XpuDist -eq "")
    New-DistInfo "pytorch_triton_xpu-3.4.0.dist-info" "pytorch-triton-xpu" "3.4.0"
    $s = Try-Scenario "swap probe, wrong version" { Get-XpuTritonState }
    Check "swap probe: an XPU triton 3.4.0 under a torch that requires 3.5.0 is not XpuOk" ($s -and $s.Known -and -not $s.XpuOk -and $s.XpuDist -eq "3.4.0")
    $r = Try-Scenario "fast path, wrong version" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: an XPU triton torch does not accept forces the dependency pass" ($r -eq $false)
    Remove-Item -LiteralPath (Join-Path $site "pytorch_triton_xpu-3.4.0.dist-info") -Recurse -Force
    New-DistInfo "pytorch_triton_xpu-0.dist-info" "pytorch-triton-xpu" ""
    $s = Try-Scenario "swap probe, versionless XPU dist" { Get-XpuTritonState }
    Check "swap probe: an XPU triton record without Version is not XpuOk" ($s -and $s.Known -and -not $s.XpuOk -and $s.XpuDist -eq "?")
    Remove-Item -LiteralPath (Join-Path $site "pytorch_triton_xpu-0.dist-info") -Recurse -Force
    Rename-Item -LiteralPath (Join-Path $site "parked_xpu") -NewName "pytorch_triton_xpu-3.5.0.dist-info"
    $s = Try-Scenario "swap probe, the XPU dist torch requires" { Get-XpuTritonState }
    Check "swap probe: the XPU triton torch requires is XpuOk" ($s -and $s.Known -and $s.XpuOk)

    # A backends/intel directory proves nothing (a generic overwrite leaves the old one behind): it does not decide.
    $intelGone = Join-Path (Join-Path (Join-Path $site "triton") "backends") "intel_gone"
    Rename-Item -LiteralPath $intel -NewName "intel_gone"
    $s = Try-Scenario "swap probe, no Intel directory" { Get-XpuTritonState }
    Check "swap probe: XpuOk does not depend on a triton/backends/intel directory" ($s -and $s.Known -and $s.XpuOk)
    Rename-Item -LiteralPath $intelGone -NewName "intel"
    $tinit = Join-Path (Join-Path $site "triton") "__init__.py"
    Set-Content -LiteralPath $tinit -Value "__version__ = '3.4.0'`ndef jit(f):`n    return f" -Encoding ascii
    $s = Try-Scenario "swap probe, runtime version" { Get-XpuTritonState }
    Check "swap probe: a triton reporting 3.4.0 under a 3.5.0 XPU record is not XpuOk" ($s -and $s.Known -and -not $s.XpuOk)
    Set-Content -LiteralPath $tinit -Value "__version__ = '3.5.0'`ndef jit(f):`n    return f" -Encoding ascii
    # Same size, same second: the cached bytecode would still say 3.4.0.
    Remove-Item -LiteralPath (Join-Path (Join-Path $site "triton") "__pycache__") -Recurse -Force -ErrorAction SilentlyContinue

    # No parser importable (-S: neither packaging nor pip): an exact pin with a local label matches its own wheel.
    $torchInfo = Join-Path $site "torch-2.9.0+xpu.dist-info"
    $xpuInfo = Join-Path $site "pytorch_triton_xpu-3.5.0.dist-info"
    New-DistInfo "torch-2.9.0+xpu.dist-info" "torch" "2.9.0+xpu" "pytorch-triton-xpu==3.5.0+git1b0418a9; platform_system == 'Windows'"
    New-DistInfo "pytorch_triton_xpu-3.5.0.dist-info" "pytorch-triton-xpu" "3.5.0+git1b0418a9"
    $s = Try-Scenario "swap probe, local label" { Get-XpuTritonState }
    Check "swap probe: ==3.5.0+git... matches an installed 3.5.0+git... without a version parser" ($s -and $s.Known -and $s.XpuOk)
    New-DistInfo "torch-2.9.0+xpu.dist-info" "torch" "2.9.0+xpu" "$xpuSpec; platform_system == 'Windows'"
    New-DistInfo "pytorch_triton_xpu-3.5.0.dist-info" "pytorch-triton-xpu" "3.5.0"

    # A packaging that is found but does not import: the next parser decides, the metadata stays readable.
    $pkg = Join-Path $site "packaging"
    New-Item -ItemType Directory -Path $pkg -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $pkg "__init__.py") -Value "" -Encoding ascii
    Set-Content -LiteralPath (Join-Path $pkg "requirements.py") -Value "raise ImportError('damaged')" -Encoding ascii
    $s = Try-Scenario "swap probe, damaged packaging" { Get-XpuTritonState }
    Check "swap probe: a damaged packaging falls back to the next parser" ($s -and $s.Known -and $s.XpuOk -and $s.XpuDist -eq "3.5.0")
    Remove-Item -LiteralPath $pkg -Recurse -Force

    New-DistInfo "triton_windows-3.9.0.post30.dist-info" "triton-windows" "3.9.0.post30"
    $r = Try-Scenario "fast path, active" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: an active triton-windows still forces the dependency pass" ($r -eq $false)
    $script:Offline = $true
    $r = Try-Scenario "fast path, active, offline" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    $script:Offline = $false
    Check "fast path, offline: an active triton-windows is left for the next online update" ($r -eq $true)
    $s = Try-Scenario "swap probe, active" { Get-XpuTritonState }
    Check "swap probe: an active triton-windows is found" ($s -and $s.TritonWin -eq "3.9.0.post30")
    Remove-Item -LiteralPath (Join-Path $site "triton_windows-3.9.0.post30.dist-info") -Recurse -Force
    New-DistInfo "triton_windows-0.dist-info" "triton-windows" ""
    $s = Try-Scenario "swap probe, versionless" { Get-XpuTritonState }
    Check "swap probe: a record without Version still counts as triton-windows" ($s -and $s.Known -and $s.TritonWin -eq "?")
    Remove-Item -LiteralPath (Join-Path $site "triton_windows-0.dist-info") -Recurse -Force

    # bitsandbytes: the active 0.50 decides, not a 0.45 backup.
    New-DistInfo "~itsandbytes-0.45.0.dist-info" "bitsandbytes" "0.45.0"
    $r = Try-Scenario "fast path, bnb backup" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: a ~itsandbytes backup does not make bitsandbytes stale" ($r -eq $true)
    Remove-Item -LiteralPath (Join-Path $site "bitsandbytes-0.50.0.dist-info") -Recurse -Force
    $r = Try-Scenario "fast path, bnb backup only" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    Check "fast path: a bitsandbytes backup alone is missing bitsandbytes" ($r -eq $false)
    $script:Offline = $true
    $r = Try-Scenario "fast path, bnb backup only, offline" {
        $SkipPythonDeps = $true; $_xpuIsReachable = $true; $script:IsIntelXpu = $true; $installedTorchTag = "xpu"
        Invoke-Expression $fastPath
        $SkipPythonDeps
    }
    $script:Offline = $false
    Check "fast path, offline: a missing bitsandbytes still runs the dependency pass, as on main" ($r -eq $false)
} finally {
    $script:RealProbeDir = $null
    Remove-Item -LiteralPath $site -Recurse -Force -ErrorAction SilentlyContinue
}
} finally {
    Remove-Item -LiteralPath $venvRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
