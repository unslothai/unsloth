#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for the UNSLOTH_RESPECT_PM_POLICY gate in the PowerShell install helpers.
# These run BEFORE install_python_stack.py is invoked, so the Python side's opt-out cannot
# cover them: without the gate an operator who set the variable would still have their
# pip.conf and uv.toml bypassed for the pinned torch install, and would only see the notice
# afterwards. The functions are AST-extracted and run in-process with the package manager
# replaced by a recorder, so what is measured is the environment each helper actually hands
# to uv/pip. Its oracle is _install_env_for_cmd() in studio/install_python_stack.py, which
# makes the same split: additive INDEX variables are scrubbed even under the opt-out
# (the pin is itself a provenance control), policy-bearing config survives.
# Run: pwsh -NoProfile -File tests/studio/test_pm_policy_optout_installers.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

function Get-FunctionText {
    param([string] $Path, [string] $Name)
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in $Path, found $($fn.Count)" }
    return $fn[0].Extent.Text
}

$setup = Join-Path $repo "studio/setup.ps1"
$bootstrap = Join-Path $repo "install.ps1"
foreach ($n in 'Test-RespectPmPolicy', 'Test-PipNoIndexOn', 'Fast-Install', 'Fast-Download') {
    Invoke-Expression (Get-FunctionText $setup $n)
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Both installers carry their own copy of the predicate, so a fix to one that misses the
# other would leave the promise half-kept. Compared with indentation normalised.
function Normalize($text) { (($text -replace "`r", "") -split "`n" | ForEach-Object { $_.Trim() }) -join "`n" }
Write-Host "the opt-out predicate is identical in both installers"
Check "install.ps1 and studio/setup.ps1 agree on Test-RespectPmPolicy" (
    (Normalize (Get-FunctionText $bootstrap 'Test-RespectPmPolicy')) -eq
    (Normalize (Get-FunctionText $setup 'Test-RespectPmPolicy')))
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the variable name" ((Get-FunctionText $setup 'Test-RespectPmPolicy') -match 'UNSLOTH_RESPECT_PM_POLICY')
Check "extraction kept the pinned-index gate" ((Get-FunctionText $setup 'Fast-Install') -match "'--index-url'")

# Stand-ins for the real managers: record the environment, report success.
$script:UseUv = $true
$UseUv = $true
$global:Recorded = $null
function Get-Command { param($Name) [pscustomobject]@{ Source = "python" } }
function uv {
    $global:Recorded = @{}
    foreach ($k in 'UV_CONFIG_FILE', 'UV_NO_CONFIG', 'PIP_CONFIG_FILE', 'PIP_NO_INDEX',
                   'PIP_FIND_LINKS', 'UV_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'UV_FIND_LINKS') {
        $global:Recorded[$k] = [Environment]::GetEnvironmentVariable($k)
    }
    $global:LASTEXITCODE = 0
}

function Reset-Env {
    foreach ($k in 'UV_CONFIG_FILE', 'UV_NO_CONFIG', 'PIP_CONFIG_FILE', 'PIP_NO_INDEX',
                   'PIP_FIND_LINKS', 'UV_INDEX_URL', 'PIP_EXTRA_INDEX_URL',
                   'UNSLOTH_RESPECT_PM_POLICY') {
        Remove-Item "Env:$k" -ErrorAction SilentlyContinue
    }
}

# The default path must not move: #6898 (an inherited mirror pulling CPU torch over the
# CUDA build) and #8530 (a user policy failing the whole install) are both fixed by it.
Write-Host "the default path is unchanged"
Reset-Env
$env:UV_CONFIG_FILE = "C:\op\uv.toml"
$env:UV_INDEX_URL = "https://mirror.corp"
$env:PIP_CONFIG_FILE = "C:\op\pip.conf"
Fast-Install --index-url https://download.pytorch.org/whl/cu124 torch | Out-Null
Check "UV_NO_CONFIG=1 is set"                 ($Recorded['UV_NO_CONFIG'] -eq '1')
Check "the operator uv.toml is dropped"       ($null -eq $Recorded['UV_CONFIG_FILE'])
Check "PIP_CONFIG_FILE points at nul"         ($Recorded['PIP_CONFIG_FILE'] -eq 'nul')
Check "the additive mirror is dropped"        ($null -eq $Recorded['UV_INDEX_URL'])
Check "UV_CONFIG_FILE is restored afterwards" ($env:UV_CONFIG_FILE -eq "C:\op\uv.toml")

Write-Host "under the opt-out the operator's policy files reach the manager"
Reset-Env
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:UV_CONFIG_FILE = "C:\op\uv.toml"
$env:UV_INDEX_URL = "https://mirror.corp"
$env:UV_FIND_LINKS = "C:\op\wheels"
$env:PIP_CONFIG_FILE = "C:\op\pip.conf"
$env:PIP_EXTRA_INDEX_URL = "https://mirror.corp"
Fast-Install --index-url https://download.pytorch.org/whl/cu124 torch | Out-Null
Check "the uv.toml reaches uv"                    ($Recorded['UV_CONFIG_FILE'] -eq "C:\op\uv.toml")
Check "UV_NO_CONFIG is not forced on"             ($null -eq $Recorded['UV_NO_CONFIG'])
Check "pip.conf is not redirected to nul"         ($Recorded['PIP_CONFIG_FILE'] -eq "C:\op\pip.conf")
Check "the additive uv mirror is still scrubbed"  ($null -eq $Recorded['UV_INDEX_URL'])
# uv's `--no-index` has no environment spelling and uv prints no resolved configuration,
# so a uv.toml that makes the wheelhouse the only permitted source is undetectable here.
# Dropping it failed every offline uv install, so it is kept unconditionally.
Check "the uv wheelhouse is kept"                 ($Recorded['UV_FIND_LINKS'] -eq "C:\op\wheels")
Check "the additive pip mirror is still scrubbed" ($null -eq $Recorded['PIP_EXTRA_INDEX_URL'])
# The finally block clears what the function SET. Under the opt-out it set neither, and
# neither was saved, so clearing them here would destroy the operator's own values.
Check "UV_CONFIG_FILE survives the call"          ($env:UV_CONFIG_FILE -eq "C:\op\uv.toml")
Check "PIP_CONFIG_FILE survives the call"         ($env:PIP_CONFIG_FILE -eq "C:\op\pip.conf")

Write-Host "an operator's own UV_NO_CONFIG is left alone"
Reset-Env
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:UV_NO_CONFIG = "1"
Fast-Install --index-url https://x torch | Out-Null
Check "it reaches uv"              ($Recorded['UV_NO_CONFIG'] -eq '1')
Check "and is still set afterwards" ($env:UV_NO_CONFIG -eq '1')

# PIP_NO_INDEX is a POLICY variable and is kept whichever way it points: pip reads the
# environment ahead of pip.conf, so PIP_NO_INDEX=0 is how an operator lifts a config
# `no-index = true` for one run. PIP_FIND_LINKS genuinely IS additive, so it survives
# only while no-index is in force and it is the sole remaining source.
Write-Host "PIP_NO_INDEX is kept under the opt-out, in both directions"
Reset-Env
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_NO_INDEX = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "an enabled PIP_NO_INDEX is kept" ($Recorded['PIP_NO_INDEX'] -eq '1')
Check "PIP_FIND_LINKS rides along with it" ($Recorded['PIP_FIND_LINKS'] -eq "C:\op\wheels")
Reset-Env
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_NO_INDEX = "0"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "an explicit OFF is carried through, not dropped" ($Recorded['PIP_NO_INDEX'] -eq '0')
Check "but find-links is scrubbed, being additive again" ($null -eq $Recorded['PIP_FIND_LINKS'])

# The effective state can come from pip.conf, which the opt-out keeps readable. Reading
# only the environment scrubbed find-links as additive and left the fallback with no
# sources at all. `python` is stubbed so the probe is a fixture, not a real pip.
Write-Host "a config-only no-index still keeps find-links"
Reset-Env
$script:PipNoIndexSections = $null
function Invoke-BoundedPythonProbe { [pscustomobject]@{ Ok = $true; Output = "global.no-index = true"; Error = "" } }
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "find-links survives a pip.conf no-index" ($Recorded['PIP_FIND_LINKS'] -eq "C:\op\wheels")

Write-Host "and with no no-index anywhere it is still additive"
Reset-Env
$script:PipNoIndexSections = $null
function Invoke-BoundedPythonProbe { [pscustomobject]@{ Ok = $true; Output = ""; Error = "" } }
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "find-links is scrubbed when nothing sets no-index" ($null -eq $Recorded['PIP_FIND_LINKS'])
$script:PipNoIndexSections = $null
Remove-Item function:Invoke-BoundedPythonProbe -ErrorAction SilentlyContinue

Write-Host "a command with no pinned index is untouched either way"
Reset-Env
$env:UV_CONFIG_FILE = "C:\op\uv.toml"
Fast-Install --upgrade pip | Out-Null
Check "no scrub without --index-url" ($Recorded['UV_CONFIG_FILE'] -eq "C:\op\uv.toml")

# The false set must match _respect_pm_policy() in install_python_stack.py exactly: a
# variable that means one thing to the shell and another to Python is worse than no gate.
Write-Host "the false set matches the Python predicate"
foreach ($pair in @(@('', $false), @('0', $false), @('false', $false), @('FALSE', $false),
                    @('no', $false), @(' No ', $false), @('1', $true), @('yes', $true),
                    @('on', $true))) {
    Reset-Env
    $env:UNSLOTH_RESPECT_PM_POLICY = $pair[0]
    Check "UNSLOTH_RESPECT_PM_POLICY='$($pair[0])' is $($pair[1])" ((Test-RespectPmPolicy) -eq $pair[1])
}

Reset-Env

# A command-prefixed key affects only the command it names, so a `[download] no-index`
# says nothing about `pip install`. Pooling them let an install keep a find-links
# location that is purely additive for it.
Write-Host "no-index is read for the command being run"
Reset-Env
$script:PipNoIndexSections = $null
function Invoke-BoundedPythonProbe { [pscustomobject]@{ Ok = $true; Output = "download.no-index = true"; Error = "" } }
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "a download-scoped no-index does not hold find-links for install" ($null -eq $Recorded['PIP_FIND_LINKS'])

Reset-Env
$script:PipNoIndexSections = $null
function Invoke-BoundedPythonProbe { [pscustomobject]@{ Ok = $true; Output = "global.no-index = true`ninstall.no-index = false"; Error = "" } }
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "an install-scoped false beats a global true" ($null -eq $Recorded['PIP_FIND_LINKS'])

Reset-Env
$script:PipNoIndexSections = $null
function Invoke-BoundedPythonProbe { [pscustomobject]@{ Ok = $true; Output = "install.no-index = true"; Error = "" } }
$env:UNSLOTH_RESPECT_PM_POLICY = "1"
$env:PIP_FIND_LINKS = "C:\op\wheels"
Fast-Install --index-url https://x torch | Out-Null
Check "an install-scoped true holds find-links" ($Recorded['PIP_FIND_LINKS'] -eq "C:\op\wheels")
$script:PipNoIndexSections = $null
Remove-Item function:Invoke-BoundedPythonProbe -ErrorAction SilentlyContinue

Write-Host ""

# A policy notice must never be able to hang an install. The Python side caps the same
# probe at 30s; a bare `& python -m pip config list` waits forever on a wedged
# interpreter, which is only reachable under the opt-out but is a hang all the same.
Write-Host "the pip probe is bounded"
$probeText = Get-FunctionText $setup 'Test-PipNoIndexOn'
Check "it goes through the bounded runner" ($probeText -match 'Invoke-BoundedPythonProbe')
Check "with an explicit timeout"           ($probeText -match '-TimeoutSec\s+30')
# Code only: the comment above the call names the shape it replaced.
$probeCode = ($probeText -split "`n" | Where-Object { $_.TrimStart() -notmatch '^#' }) -join "`n"
Check "and never calls pip unbounded"      ($probeCode -notmatch '&\s*python\s+-m\s+pip')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "all checks passed"
