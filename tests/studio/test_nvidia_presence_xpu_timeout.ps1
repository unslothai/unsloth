#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# An update whose XPU runtime probe times out keeps the XPU route even when the NVIDIA presence promotion fires.
param([string]$RepoRoot = (Join-Path $PSScriptRoot "../.."))
$ErrorActionPreference='Stop'
$repo=(Resolve-Path $RepoRoot).Path
$path=Join-Path $repo 'studio/setup.ps1'
$text=[IO.File]::ReadAllText($path)
$ast=[System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$null)
$functions=@($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true))
foreach($f in $functions){. ([scriptblock]::Create($f.Extent.Text))}
$ifs=@($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.IfStatementAst]},$true))
$promo=@($ifs|Where-Object {$_.Extent.Text -match '^if \(-not \$HasNvidiaSmi\) \{\s*\$presenceScan = Invoke-BoundedVideoControllerScan'})[0]
$selector=@($ifs|Where-Object {$_.Extent.Text -match '^if \(\$PinnedTorchIndexUrl\)' -and $_.Extent.Text -match 'Get-PytorchCudaTag'})[0]
$selectorStart=$text.LastIndexOf('$PinnedTorchIndexUrl = Get-PinnedTorchIndexUrl', $selector.Extent.StartOffset)
$selectorBlock=$text.Substring($selectorStart,$selector.Extent.EndOffset-$selectorStart)
$gate=@($ifs|Where-Object {$_.Extent.Text -match '^if \(-not \$ROCmIndexUrl -and \$CuTag -eq "xpu"\)'})[0]
$ss=$text.IndexOf('$shouldRebuild = $false',$text.IndexOf('$installedTorchTag = $null'))
$se=$text.IndexOf('# Outside the rebuild branch:',$ss)
$stale=$text.Substring($ss,$se-$ss)
$cs=$text.IndexOf('$cudaForce = @()',$text.IndexOf('substep "installing PyTorch with CUDA support'))
$ce=$text.IndexOf('$cudaTorchSpec =',$cs)
$cudaForceBlock=$text.Substring($cs,$ce-$cs)
$fixture=Join-Path ([IO.Path]::GetTempPath()) ('xpu-update-venv-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path "$fixture/Scripts"|Out-Null
Set-Content "$fixture/Scripts/python.exe" ''
foreach($n in @('UNSLOTH_TORCH_INDEX_URL','UNSLOTH_TORCH_INDEX_FAMILY','UNSLOTH_ROCM_GFX_ARCH','_UNSLOTH_ROCM_GFX_ARCH_HANDOFF','HIP_PATH','HIP_PATH_57','ROCM_PATH','UNSLOTH_ENABLE_AMD_SMI')){[Environment]::SetEnvironmentVariable($n,$null)}
function python { $global:LASTEXITCODE=1 }
function substep {param($Message,$Color)}
function Write-StudioLine {param($Line,$ForegroundColor)}
function Get-Command {param($Name,$CommandType,$ErrorAction) $null}
function Get-NvidiaLibraryInventory {$null}
function Get-ProbableStudioVenvDir {$fixture}
function Test-VenvTorchIsXpu {param($VenvPath) $true}
function Test-WinArm64Venv {$false}
function Get-SetupHostInterpreterInVenv {param($VenvDir) "$VenvDir/Scripts/python.exe"}
function Get-IntelRegistryAdapterNames {@('Intel Graphics')}
function Invoke-BoundedVideoControllerScan {
 [pscustomobject]@{Ok=$true;Names=@('NVIDIA RTX 4060','Intel Graphics');Adapters=@(
  [pscustomobject]@{Name='NVIDIA RTX 4060';PNPDeviceID='PCI\VEN_10DE&DEV_2882';ConfigManagerErrorCode=0;DriverVersion='32.0.15.8000'},
  [pscustomobject]@{Name='Intel Graphics';PNPDeviceID='PCI\VEN_8086&DEV_7D55';ConfigManagerErrorCode=0;DriverVersion='32.0.101.6739'})}
}
function Invoke-BoundedPythonProbe {
 param($PythonExe,$Code,$TimeoutSec)
 if($Code -match '__version__'){return [pscustomobject]@{Ok=$true;Output='2.11.0+xpu';TimedOut=$false}}
 if($script:Mode -eq 'timeout'){return [pscustomobject]@{Ok=$false;Output='';TimedOut=$true}}
 $value=if($script:Mode -eq 'live'){'True'}else{'False'}
 [pscustomobject]@{Ok=$true;Output=$(if($Code -match 'DEV='){"DEV=$value"}else{$value});TimedOut=$false}
}
$failures=0
foreach($mode in @('live','timeout','unavailable')){
 $script:Mode=$mode
 $HasNvidiaSmi=$false;$HasROCm=$false;$AmdHasGpuWheels=$false
 $script:IsIntelXpu=$false;$script:NvidiaPresenceOnly=$false;$script:NvidiaPresenceCudaFloor=$null;$script:NvidiaPresenceDriverRelease=$null;$script:NvidiaPresenceStaleGpuWheel=$null
 $script:NvidiaSmiExe=$null;$script:NvidiaSmiRejected=$true;$script:PreservedXpuVenv=$false;$script:PreservedInstallerTorchTag=$null;$script:PinChangedForceReinstall=$false;$script:TorchImportDefinitivelyFailed=$false
 $VenvDir=$fixture;$VenvPyExe="$fixture/Scripts/python.exe";$installedTorchTag='xpu';$InstallerManagedSetup=$false;$PinnedTorchIndexUrl=$null;$NoTorchMode=$false;$script:ROCmGfxArch=$null
 if($promo){. ([scriptblock]::Create($promo.Extent.Text))}
 . ([scriptblock]::Create($stale))
 $env:UNSLOTH_STUDIO_FULL_DEPS='1';$script:SkipPythonDeps=$true
 Invoke-FastPathEscapes
 . ([scriptblock]::Create($selectorBlock))
 $ROCmIndexUrl=$null;$TorchInstallIndexUrl="https://download.pytorch.org/whl/$CuTag";$XpuIndexUrl=$null
 . ([scriptblock]::Create($gate.Extent.Text))
 $cudaForce=@()
 if(-not $XpuIndexUrl){. ([scriptblock]::Create($cudaForceBlock))}
 # A definitive False may intentionally convert the stale XPU wheel. Uncertainty must keep XPU's dependency route.
 $expected=if($mode -eq 'unavailable' -and $promo){'cu126'}else{'xpu'}
 $ok=($CuTag -eq $expected) -and -not $script:SkipPythonDeps
 if($mode -eq 'timeout'){$ok=$ok -and [bool]$XpuIndexUrl -and $cudaForce.Count -eq 0}
 $result=[pscustomobject]@{Arm=$RepoRoot;Probe=$mode;Promoted=$script:NvidiaPresenceOnly;DependencyPass=(-not $script:SkipPythonDeps);CuTag=$CuTag;XpuIndexUrl=$XpuIndexUrl;CudaForce=@($cudaForce);PreservedXpu=$script:PreservedXpuVenv;Expected=$expected;Pass=$ok}
 $result|ConvertTo-Json -Compress|Write-Output
 if(-not $ok){$failures++}
}
Remove-Item -LiteralPath $fixture -Recurse -Force -ErrorAction SilentlyContinue
if($failures){Write-Host "$failures checks failed";exit 1}
Write-Host 'all checks passed';exit 0
