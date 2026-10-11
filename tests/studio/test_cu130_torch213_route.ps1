#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# install.ps1 gives new Windows x64 cu130 Python 3.13 installs torch 2.13 (the only route the
# prebuilt-wheels-cu13 release covers) while an existing 2.4-2.14 install keeps its release. The
# Windows port of tests/sh/test_cu130_torch213_route.sh. Helpers are extracted from install.ps1 by
# AST; the interpreter, host arch, mirror check and PyPI fetch are stubs.
# Run: pwsh -NoProfile -File tests/studio/test_cu130_torch213_route.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPath = Join-Path $root "install.ps1"
$setupPath = Join-Path (Join-Path $root "studio") "setup.ps1"

function Get-FunctionAsts([string]$Path, [string[]]$Names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    foreach ($name in $Names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -ne 1) { throw "expected exactly one $name in $Path, found $($fn.Count)" }
        $fn[0].Extent.Text
    }
}

foreach ($text in (Get-FunctionAsts $installPath @(
        "ConvertTo-TorchNumericRelease", "Test-TorchReleaseInWindow",
        "Get-PreviousTorchPin", "Get-TorchIndexLeafName", "Test-Cu130Torch213Platform",
        "Resolve-Cu130TorchRoute", "Get-UpgradeKeptTorchPin", "Test-RequiresDistAdmitsTorch",
        "Test-PypiUnslothAdmitsTorch", "Get-TorchVersionFromDisk", "Test-UvEnvFlag"))) {
    Invoke-Expression $text
}
foreach ($text in (Get-FunctionAsts $setupPath @("Get-TritonWindowsSpec"))) { Invoke-Expression $text }
$script:Cu130TorchCeiling = "2.15.0"

# Stubs. Functions shadow cmdlets, so Invoke-RestMethod here never reaches the network.
$script:HostArch = "x86_64"
$script:VenvAnswer = "3.13 win-amd64"
$script:VenvOk = $true
$script:Mirror = $false
$script:RequiresDist = @("torch<2.15.0,>=2.4.0", "torchvision; extra == 'colab'", "numpy")
$script:FetchFails = $false
$script:Fetches = 0
function Get-HostMachineArch { return $script:HostArch }
function Invoke-BoundedPythonProbe { param($PythonExe, $Code, $TimeoutSec)
    return [pscustomobject]@{ Ok = $script:VenvOk; Output = "noise from sitecustomize`n$($script:VenvAnswer)`n"; Error = "" } }
function Test-MirrorConfigured { param($Tool) return $script:Mirror }
function Invoke-RestMethod { param($Uri, [switch]$UseBasicParsing, $TimeoutSec, $ErrorAction)
    $script:Fetches++
    $script:LastUri = $Uri
    if ($script:FetchFails) { throw "network down" }
    return [pscustomobject]@{ info = [pscustomobject]@{ requires_dist = $script:RequiresDist } } }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}
function Reset-Stubs {
    $script:HostArch = "x86_64"; $script:VenvAnswer = "3.13 win-amd64"; $script:VenvOk = $true
    $script:Mirror = $false; $script:FetchFails = $false; $script:Fetches = 0
    $script:RequiresDist = @("torch<2.15.0,>=2.4.0", "torchvision; extra == 'colab'", "numpy")
}

$cu130 = "https://download.pytorch.org/whl/cu130"
$default = "torch>=2.4,<2.12.0"
$saved = @{}
foreach ($n in @("UNSLOTH_TORCH_UPGRADE", "UV_OFFLINE", "UV_EXCLUDE_NEWER", "UV_EXCLUDE_NEWER_PACKAGE", "UNSLOTH_PYPI_JSON_URL")) {
    $saved[$n] = [Environment]::GetEnvironmentVariable($n)
    [Environment]::SetEnvironmentVariable($n, $null)
}
try {
    Write-Host "=== platform predicate ==="
    Reset-Stubs
    Check "cu130 + x64 host + native 3.13 win-amd64 venv" (Test-Cu130Torch213Platform -IndexUrl $cu130 -PythonExe "py")
    Check "cu130 with a ?token query still counts" (Test-Cu130Torch213Platform -IndexUrl "$cu130/?token=x" -PythonExe "py")
    foreach ($leaf in @("cu128", "cu126", "cpu", "xpu", "rocm7.2")) {
        Check "index $leaf is not the route" (-not (Test-Cu130Torch213Platform -IndexUrl "https://download.pytorch.org/whl/$leaf" -PythonExe "py"))
    }
    Check "no index is not the route" (-not (Test-Cu130Torch213Platform -IndexUrl "" -PythonExe "py"))
    $script:HostArch = "arm64"
    Check "Windows on ARM host is not the route (emulated x64 included)" (-not (Test-Cu130Torch213Platform -IndexUrl $cu130 -PythonExe "py"))
    Reset-Stubs
    foreach ($a in @("3.12 win-amd64", "3.11 win-amd64", "3.14 win-amd64", "3.13 win-arm64", "3.13 win32")) {
        $script:VenvAnswer = $a
        Check "venv '$a' is not the route" (-not (Test-Cu130Torch213Platform -IndexUrl $cu130 -PythonExe "py"))
    }
    Reset-Stubs
    $script:VenvOk = $false
    Check "an interpreter that does not answer is not the route" (-not (Test-Cu130Torch213Platform -IndexUrl $cu130 -PythonExe "py"))

    Write-Host "=== PyPI gate ==="
    Reset-Stubs
    Check "newest unsloth admitting 2.13 says yes" (Test-PypiUnslothAdmitsTorch -Want "2.13.0")
    Check "default URL is public PyPI" ($script:LastUri -eq "https://pypi.org/pypi/unsloth/json")
    $env:UNSLOTH_PYPI_JSON_URL = "https://mirror.example/pypi/unsloth/json"
    [void](Test-PypiUnslothAdmitsTorch -Want "2.13.0")
    Check "UNSLOTH_PYPI_JSON_URL redirects the check" ($script:LastUri -eq "https://mirror.example/pypi/unsloth/json")
    Remove-Item Env:UNSLOTH_PYPI_JSON_URL
    $script:RequiresDist = @("torch<2.13.0,>=2.4.0")
    Check "a release capped below 2.13 says no" (-not (Test-PypiUnslothAdmitsTorch -Want "2.13.0"))
    Reset-Stubs; $script:FetchFails = $true
    Check "a failed fetch says no" (-not (Test-PypiUnslothAdmitsTorch -Want "2.13.0"))
    foreach ($v in @("1", "true", "YES", " on ")) {
        Reset-Stubs; $env:UV_OFFLINE = $v
        Check "UV_OFFLINE='$v' says no without fetching" ((-not (Test-PypiUnslothAdmitsTorch -Want "2.13.0")) -and $script:Fetches -eq 0)
    }
    Remove-Item Env:UV_OFFLINE
    Reset-Stubs; $env:UV_OFFLINE = "0"
    Check "UV_OFFLINE=0 still checks" (Test-PypiUnslothAdmitsTorch -Want "2.13.0")
    Remove-Item Env:UV_OFFLINE
    foreach ($n in @("UV_EXCLUDE_NEWER", "UV_EXCLUDE_NEWER_PACKAGE")) {
        Reset-Stubs; [Environment]::SetEnvironmentVariable($n, "2026-01-01")
        Check "$n says no without fetching" ((-not (Test-PypiUnslothAdmitsTorch -Want "2.13.0")) -and $script:Fetches -eq 0)
        [Environment]::SetEnvironmentVariable($n, $null)
    }
    Reset-Stubs; $script:Mirror = $true
    Check "a configured uv index says no without fetching" ((-not (Test-PypiUnslothAdmitsTorch -Want "2.13.0")) -and $script:Fetches -eq 0)

    Write-Host "=== Requires-Dist parsing ==="
    $rd = @(
        @{ r = @("torch<2.15.0,>=2.4.0"); ok = $true; n = "range admitting 2.13" }
        @{ r = @("torch (<2.15.0,>=2.4.0)"); ok = $true; n = "parenthesised range" }
        @{ r = @("torch<2.15.0,>=2.4.0; sys_platform != 'darwin'"); ok = $true; n = "marker after the range" }
        @{ r = @("torch<2.13.0,>=2.4.0"); ok = $false; n = "ceiling at 2.13" }
        @{ r = @("torch>=2.13.1"); ok = $false; n = "floor above 2.13.0" }
        @{ r = @("torch!=2.13.0,<2.15"); ok = $false; n = "2.13.0 excluded" }
        @{ r = @("torch==2.13.0"); ok = $true; n = "exact 2.13.0" }
        @{ r = @("torch<2.15", "torch<2.14; extra == 'x'"); ok = $true; n = "extras are ignored" }
        @{ r = @("torch<2.15", "torch>=2.4"); ok = $false; n = "two unconditional specs" }
        @{ r = @("torch"); ok = $false; n = "bare torch" }
        @{ r = @("torch~=2.13"); ok = $false; n = "unparsed operator" }
        @{ r = @(); ok = $false; n = "no torch requirement" }
        @{ r = @("torchvision<0.30"); ok = $false; n = "torchvision is not torch" }
    )
    foreach ($t in $rd) {
        Check ("requires_dist: " + $t.n) ((Test-RequiresDistAdmitsTorch -RequiresDist $t.r -Want "2.13.0") -eq $t.ok)
    }

    Write-Host "=== route decision ==="
    Reset-Stubs
    $r = Resolve-Cu130TorchRoute -IndexUrl $cu130 -PythonExe "py" -DefaultSpec $default -ExistingInstall $false
    Check "new install takes the route" ($r.NewInstall -and $r.PreserveSpec -eq "torch>=2.4,<2.15.0")
    $r = Resolve-Cu130TorchRoute -IndexUrl $cu130 -PythonExe "py" -DefaultSpec $default -ExistingInstall $true
    Check "existing install does not take the route but gets the wide window" ((-not $r.NewInstall) -and $r.PreserveSpec -eq "torch>=2.4,<2.15.0")
    Check "existing install never asks PyPI" ($script:Fetches -eq 1)
    $env:UNSLOTH_TORCH_UPGRADE = "1"
    $r = Resolve-Cu130TorchRoute -IndexUrl $cu130 -PythonExe "py" -DefaultSpec $default -ExistingInstall $true
    Check "UNSLOTH_TORCH_UPGRADE=1 opts an existing install in" ($r.NewInstall)
    Remove-Item Env:UNSLOTH_TORCH_UPGRADE
    Reset-Stubs
    $r = Resolve-Cu130TorchRoute -IndexUrl $cu130 -PythonExe "py" -DefaultSpec $default -ExistingInstall $false -PackageName "unsloth-fork"
    Check "--package other than unsloth: no route, window still wide, no fetch" ((-not $r.NewInstall) -and $r.PreserveSpec -eq "torch>=2.4,<2.15.0" -and $script:Fetches -eq 0)
    $script:FetchFails = $true
    $r = Resolve-Cu130TorchRoute -IndexUrl $cu130 -PythonExe "py" -DefaultSpec $default -ExistingInstall $false
    Check "PyPI down: no route, window still wide" ((-not $r.NewInstall) -and $r.PreserveSpec -eq "torch>=2.4,<2.15.0")
    Reset-Stubs
    $r = Resolve-Cu130TorchRoute -IndexUrl "https://download.pytorch.org/whl/cu128" -PythonExe "py" -DefaultSpec $default -ExistingInstall $false
    Check "cu128: default window, no route, no fetch" ((-not $r.NewInstall) -and $r.PreserveSpec -eq $default -and $script:Fetches -eq 0)

    Write-Host "=== preservation under the route window ==="
    $wide = "torch>=2.4,<2.15.0"
    foreach ($v in @("2.11.0+cu130", "2.12.1+cu130", "2.13.0+cu130", "2.14.0+cu130")) {
        $pin = Get-PreviousTorchPin -TorchVersion $v -Constraint $wide
        Check "existing $v is kept" ($null -ne $pin -and $pin.TorchSpec -eq "torch==$(($v -split '\+')[0])")
    }
    $pin = Get-PreviousTorchPin -TorchVersion "2.13.0+cu130" -Constraint $wide
    Check "kept 2.13 pairs torchvision 0.28 and torchaudio 2.11" ($pin.VisionSpec -eq "torchvision==0.28.*" -and $pin.AudioSpec -eq "torchaudio==2.11.*")
    $pin = Get-PreviousTorchPin -TorchVersion "2.10.0+cu130" -Constraint $wide
    Check "kept 2.10 still pairs torchaudio 2.10" ($pin.AudioSpec -eq "torchaudio==2.10.*")
    Check "2.15 is outside the window" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.15.0+cu130" -Constraint $wide))
    Check "the old window would have dropped a 2.13 install" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.13.0+cu130" -Constraint $default))

    Write-Host "=== UNSLOTH_TORCH_UPGRADE=1 with the route closed ==="
    $env:UNSLOTH_TORCH_UPGRADE = "1"
    Check "upgrade drops the ordinary pin" ($null -eq (Get-PreviousTorchPin -TorchVersion "2.13.0+cu130" -Constraint $wide))
    $pin = Get-UpgradeKeptTorchPin -TorchVersion "2.13.0+cu130" -DefaultSpec $default -PreserveSpec $wide -NewInstallRoute $false
    Check "route closed: a resident 2.13 is kept, not moved down to 2.11" ($pin.TorchSpec -eq "torch==2.13.0")
    $pin = Get-UpgradeKeptTorchPin -TorchVersion "2.14.0+cu130" -DefaultSpec $default -PreserveSpec $wide -NewInstallRoute $false
    Check "route closed: a resident 2.14 is kept" ($pin.TorchSpec -eq "torch==2.14.0")
    Check "route closed: a resident 2.11 still upgrades within the default range" (
        $null -eq (Get-UpgradeKeptTorchPin -TorchVersion "2.11.0+cu130" -DefaultSpec $default -PreserveSpec $wide -NewInstallRoute $false))
    Check "route open: nothing is kept, 2.13 installs" (
        $null -eq (Get-UpgradeKeptTorchPin -TorchVersion "2.14.0+cu130" -DefaultSpec $default -PreserveSpec $wide -NewInstallRoute $true))
    Check "off the route: nothing is kept" (
        $null -eq (Get-UpgradeKeptTorchPin -TorchVersion "2.13.0+cu128" -DefaultSpec $default -PreserveSpec $default -NewInstallRoute $false))
    Remove-Item Env:UNSLOTH_TORCH_UPGRADE
    Check "without upgrade the helper never pins" (
        $null -eq (Get-UpgradeKeptTorchPin -TorchVersion "2.13.0+cu130" -DefaultSpec $default -PreserveSpec $wide -NewInstallRoute $false))

    Write-Host "=== legacy torch version read off disk ==="
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("cu130route." + [guid]::NewGuid())
    try {
        # The helper joins a Windows-style relative path; on a POSIX host that is one file name.
        $versionPy = Join-Path $tmp "Lib\site-packages\torch\version.py"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $versionPy) | Out-Null
        Set-Content -LiteralPath $versionPy -Value @("from typing import Optional", "__version__ = '2.13.0+cu130'", "git_version = 'abc'")
        Check "version.py gives the release" ((Get-TorchVersionFromDisk -VenvRoot $tmp) -eq "2.13.0+cu130")
        Check "missing version.py gives nothing" ($null -eq (Get-TorchVersionFromDisk -VenvRoot (Join-Path $tmp "nope")))
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "=== setup.ps1 triton-windows pairing ==="
    Check "torch 2.11 keeps triton-windows <3.7" ((Get-TritonWindowsSpec -TorchVersion "2.11.0+cu130") -eq "triton-windows<3.7")
    Check "torch 2.12 keeps triton-windows <3.7" ((Get-TritonWindowsSpec -TorchVersion "2.12.1+cu130") -eq "triton-windows<3.7")
    Check "torch 2.13 pairs triton-windows 3.7" ((Get-TritonWindowsSpec -TorchVersion "2.13.0+cu130") -eq "triton-windows>=3.7,<3.8")
    Check "torch 2.14 pairs triton-windows 3.8" ((Get-TritonWindowsSpec -TorchVersion "2.14.0+cu130") -eq "triton-windows>=3.8,<3.9")
    Check "unknown torch keeps <3.7" ((Get-TritonWindowsSpec -TorchVersion "") -eq "triton-windows<3.7")
    Check "Windows on ARM is unchanged" ((Get-TritonWindowsSpec -TorchVersion "2.14.0+cu134" -WinArm64 $true) -eq "triton-windows>=3.8.0.post28")

    Write-Host "=== wiring ==="
    $install = Get-Content -LiteralPath $installPath -Raw
    $setup = Get-Content -LiteralPath $setupPath -Raw
    Check "the route is decided before preservation" ($install -match '(?s)Resolve-Cu130TorchRoute -IndexUrl \$TorchIndexUrl.*?\$_routeWindow = \$_preserveTorchSpec')
    Check "the route skips ROCm and --no-torch" ($install -match 'if \(-not \$SkipTorch -and -not \$ROCmIndexUrl\) \{\s*\$_cu130Route = Resolve-Cu130TorchRoute')
    Check "2.13 applies only when nothing was kept" ($install -match '(?s)if \(-not \$script:PrevTorchPin -and \$_cu130NewInstallRoute\) \{\s*\$_pinTorchSpec = \$script:Cu130NewInstallTorchSpec\s*\$_pinVisionSpec = \$script:Cu130NewInstallVisionSpec')
    Check "the generic CUDA install uses the hoisted specs" ($install -match '\$_torchSpecs = @\(\$_pinTorchSpec, \$_pinVisionSpec, \$_pinAudioSpec\)')
    Check "every earlier layout counts as an existing install" (
        ([regex]::Matches($install, '\$script:ExistingInstall = \$true')).Count -eq 3)
    Check "an occupied venv whose interpreter cannot answer keeps its torch from disk" ($install -match '(?s)\$script:PrevTorchVer = Get-InstalledTorchVersionRaw -PythonExe \$VenvPython\s*#[^\n]*\n\s*#[^\n]*\n\s*if \(-not \$script:PrevTorchVer\) \{ \$script:PrevTorchVer = Get-TorchVersionFromDisk -VenvRoot \$VenvDir \}')
    Check "the CWD-relative legacy venv records its torch before it is migrated" ($install -match '(?s)\$_cwdVenvPy -and \(\s*\(Test-Path -LiteralPath \$_cwdVenvPy\) -or\s*\(Test-Path -LiteralPath \(Join-Path [^\n]*"pyvenv\.cfg"\) -PathType Leaf\)\)\) \{.*?\$script:ExistingInstall = \$true.*?\$script:PrevTorchVer = Get-InstalledTorchVersionRaw -PythonExe \$_cwdVenvPy \}\s*if \(-not \$script:PrevTorchVer\) \{ \$script:PrevTorchVer = Get-TorchVersionFromDisk')
    Check "a migrated venv freezes its trio for the core install" ($install -match '(?s)\$script:TorchOverridesFile = New-UnslothTorchOverridesFile -PythonExe \$VenvPython\s*\$_migOverrides = @\(\)\s*if \(\$script:TorchOverridesFile\) \{ \$_migOverrides = @\("--overrides", \$script:TorchOverridesFile\) \}.*?-Label "install unsloth \(migrated\)" \{ & \$script:UvExe pip install --python \$VenvPython @_migOverrides ')
    Check "a migrated CUDA venv keeps its release for the flavor repair" ($install -match '(?s)\$_migTorchVer = Get-InstalledTorchVersionRaw -PythonExe \$VenvPython.*?\$script:PrevTorchPin = Get-PreviousTorchPin -TorchVersion \$_migTorchVer -Constraint \$_migWindow.*?\$env:UNSLOTH_KEPT_TORCH = \$script:PrevTorchPin\.Release\.PublicBase')
    $probeRe = [regex]::Match($setup, "_residentProbe\.Output -match '([^']+)'").Groups[1].Value
    foreach ($t in @(@("2.11.0+cu130", "2.11.0"), @("2.14.0", "2.14.0"), @("2.13.0+cu130`r", "2.13.0"), @("2.14.0rc1", $null), @("2.15.0.dev20261001+cu130", $null), @("2.14.0a0+git1234", $null))) {
        $got = if ($t[0] -match $probeRe) { $Matches[1] } else { $null }
        Check ("resident probe reads '" + $t[0].Trim() + "' as " + $(if ($t[1]) { $t[1] } else { "no release" })) ($got -eq $t[1])
    }
    # setup.ps1's own CUDA torch block, run with the installer and the probe stubbed.
    $blockStart = $setup.IndexOf('    $_cudaKeptActive = $false')
    $blockEnd = $setup.IndexOf('    if ($torchInstallExit -ne 0) {', $blockStart)
    $torchBlock = [scriptblock]::Create($setup.Substring($blockStart, $blockEnd - $blockStart))
    function Invoke-SetupTorchBlock {
        param([string]$Resident, [string]$CuTag = "cu130", [string[]]$Force = @(), [int[]]$Exits = @(0), [string]$Handoff = "", [string]$Upgrade = "")
        $script:FastInstallCalls = @()
        $script:FastInstallExits = [System.Collections.Generic.Queue[int]]::new([int[]]$Exits)
        $script:ProbeCalls = 0
        function Fast-Install { $script:FastInstallCalls += ,(@($args | Where-Object { $_ -notmatch '^--index-url$|^https://' }) -join " "); $global:LASTEXITCODE = $script:FastInstallExits.Dequeue(); "" }
        function Invoke-BoundedPythonProbe { param($PythonExe, $Code) $script:ProbeCalls++; [pscustomobject]@{ Ok = [bool]$Resident; Output = $Resident } }
        function substep { }
        $env:UNSLOTH_KEPT_TORCH = $Handoff; $env:UNSLOTH_TORCH_UPGRADE = $Upgrade
        $WinArm64Venv = $false; $WinArm64NoAudio = $false; $WinArm64IndexArgs = @(); $PinnedTorchIndexUrl = $null
        $WinArm64TorchIndexUrl = $null; $TorchInstallIndexUrl = "https://download.pytorch.org/whl/$CuTag"; $script:UnslothVerbose = $false
        $cudaTorchSpec = "torch"; $cudaVisionSpec = "torchvision"; $cudaAudioSpec = "torchaudio"; $cudaForce = $Force
        . $torchBlock
        Remove-Item Env:UNSLOTH_KEPT_TORCH, Env:UNSLOTH_TORCH_UPGRADE -ErrorAction SilentlyContinue
        return [pscustomobject]@{ Calls = $script:FastInstallCalls; Exit = $torchInstallExit; Probes = $script:ProbeCalls }
    }
    $kept13 = "torch==2.13.0 torchvision==0.28.* torchaudio==2.11.*"
    $r = Invoke-SetupTorchBlock -Resident "2.13.0+cu130"
    Check "setup keeps a healthy resident 2.13 when nothing is forced (a missing torchvision cannot pull 2.14)" ($r.Calls.Count -eq 1 -and $r.Calls[0] -eq $kept13)
    $r = Invoke-SetupTorchBlock -Resident "2.13.0+cu130" -Exits 1, 0
    Check "an unforced resident pin that fails retries exactly the unpinned run" ($r.Calls.Count -eq 2 -and $r.Calls[1] -eq "torch torchvision torchaudio" -and $r.Exit -eq 0)
    $r = Invoke-SetupTorchBlock -Resident "2.10.0+cu128" -Force "--force-reinstall"
    Check "a forced repair keeps the resident release" ($r.Calls.Count -eq 1 -and $r.Calls[0] -eq "torch==2.10.0 torchvision==0.25.* torchaudio==2.10.* --force-reinstall")
    $r = Invoke-SetupTorchBlock -Resident "2.10.0+cu128" -Force "--force-reinstall" -Exits 1, 0
    Check "a forced kept pin that fails stops instead of resolving another release" ($r.Calls.Count -eq 1 -and $r.Exit -eq 1)
    $r = Invoke-SetupTorchBlock -Resident "2.13.0+cu130" -Handoff "2.11.0" -Exits 1, 0
    Check "a failed installer handoff stops instead of resolving another release" ($r.Calls.Count -eq 1 -and $r.Calls[0] -like "torch==2.11.0 *" -and $r.Exit -eq 1 -and $r.Probes -eq 0)
    $r = Invoke-SetupTorchBlock -Resident "2.13.0+cu130" -Upgrade "1"
    Check "UNSLOTH_TORCH_UPGRADE=1 resolves unpinned and never probes" ($r.Calls[0] -eq "torch torchvision torchaudio" -and $r.Probes -eq 0)
    $r = Invoke-SetupTorchBlock -Resident "2.6.0+cu126"
    Check "a resident release the cu130 index does not publish is not pinned" ($r.Calls[0] -eq "torch torchvision torchaudio")
    $r = Invoke-SetupTorchBlock -Resident "2.7.1+cu128" -CuTag "cu128"
    Check "cu128 keeps a resident 2.7" ($r.Calls[0] -eq "torch==2.7.1 torchvision==0.22.* torchaudio==2.7.*")
    $r = Invoke-SetupTorchBlock -Resident "2.14.0rc1"
    Check "a prerelease resident torch is never pinned" ($r.Calls[0] -eq "torch torchvision torchaudio")
    $r = Invoke-SetupTorchBlock -Resident "2.13.0+cu130" -CuTag "rocm7.2"
    Check "a non-CUDA leaf is never pinned" ($r.Calls[0] -eq "torch torchvision torchaudio")
    Check "setup.ps1 never falls back to the default range after a kept pin" ($setup -notmatch 'not installable from this CUDA index -- using the supported release')
    Check "setup.ps1 re-reads torch before installing triton" ($setup -match '(?s)\$_tritonTorchProbe = Invoke-BoundedPythonProbe.*?\$_tritonSpec = Get-TritonWindowsSpec -TorchVersion.*?substep "installing Triton for Windows\.\.\."')
    Check "setup.ps1 pairs every kept torch 2.12+ with torchaudio 2.11" (
        ([regex]::Matches($setup, 'torchaudio==2\.\$\(\[Math\]::Min\(\$_keptMinor, 11\)\)\.\*')).Count -eq 3 -and
        $setup -notmatch 'torchaudio==2\.\$\(\$_keptMinor\)\.\*')
} finally {
    foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) }
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
