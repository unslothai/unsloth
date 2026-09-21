#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# UNSLOTH_RESPECT_PM_POLICY on the Windows entry points: install.ps1's Test-RespectPmPolicy and
# Invoke-InstallCommand, studio/setup.ps1's Test-RespectPmPolicy, Fast-Install and Fast-Download.
#
# The predicate is a hand-maintained duplicate across two files, so it is pinned as identical
# text AND run over the same value matrix the sh/Python test uses. The scrubs are the part a
# predicate-only test cannot reach: an arm that keeps the right variables but whose `finally`
# deletes the operator's own UV_NO_CONFIG afterwards is a regression the predicate never sees.
# So the real functions are AST-extracted and EXECUTED here, with the environment observed
# from inside them.
#
# Nothing below retypes a function body. Where a stand-in is unavoidable (the native `uv` and
# `python` calls Fast-Install makes), the expected scrub/keep names are first asserted to be
# present in the extracted source, so a list that drifts away from the shipped one fails
# rather than quietly testing itself.
#
# Run: pwsh -NoProfile -File tests/studio/test_pm_policy_optout_installers.ps1

$ErrorActionPreference = "Stop"

$repoRoot   = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = (Resolve-Path ([System.IO.Path]::Combine($repoRoot, "install.ps1"))).Path
$setupPs1   = (Resolve-Path ([System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1"))).Path

$failures = 0
function Check($name, $cond) {
    # A PowerShell function silently absorbs surplus positional arguments into $args, so
    # `Check "x" ( ... ) .Count -eq 0` -- an easy line-wrapping slip -- would hand this an
    # integer and report the INVERSE of the intended condition as a pass. Refuse instead.
    if ($args.Count -gt 0) { throw "Check '$name' got surplus arguments ($($args -join ' ')); the condition is not fully parenthesised" }
    if ($cond -isnot [bool]) { throw "Check '$name' got a non-boolean condition of type $($cond.GetType().Name)" }
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# ── AST extraction ────────────────────────────────────────────────────────────
# ParseFile doubles as a syntax gate on both installers.
function Get-Ast($path) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    return $ast
}

function Get-FunctionText($ast, $name, $path) {
    # Recursive: install.ps1 defines Test-RespectPmPolicy and Invoke-InstallCommand INSIDE an
    # enclosing function, so a top-level-only search would find nothing and skip silently.
    $found = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($found.Count -ne 1) { throw "expected exactly one $name in $path, found $($found.Count)" }
    return $found[0].Extent.Text
}

$installAst = Get-Ast $installPs1
$setupAst   = Get-Ast $setupPs1

$predicateInstallSrc = Get-FunctionText $installAst "Test-RespectPmPolicy" $installPs1
$predicateSetupSrc   = Get-FunctionText $setupAst   "Test-RespectPmPolicy" $setupPs1
$invokeInstallSrc    = Get-FunctionText $installAst "Invoke-InstallCommand" $installPs1
$fastInstallSrc      = Get-FunctionText $setupAst   "Fast-Install" $setupPs1
$fastDownloadSrc     = Get-FunctionText $setupAst   "Fast-Download" $setupPs1
$removeUvFlagsSrc    = Get-FunctionText $setupAst   "Remove-UvOnlyResolverFlags" $setupPs1
# Fast-Install asks these two which policy to carry into uv. Extracted by name rather than
# stubbed: a stub would answer for the shipped code instead of letting it answer.
$pipEnvFlagSrc       = Get-FunctionText $setupAst   "Test-PipEnvFlag" $setupPs1
$uvEnvFlagSrc        = Get-FunctionText $setupAst   "Test-UvEnvFlag" $setupPs1

Write-Host "== extraction =="
Check "install.ps1 and studio/setup.ps1 both parse" $true
$tooShort = @(@($predicateInstallSrc, $predicateSetupSrc, $invokeInstallSrc,
                $fastInstallSrc, $fastDownloadSrc, $removeUvFlagsSrc) |
              Where-Object { $_.Length -le 40 })
Check "all six functions extracted non-empty" ($tooShort.Count -eq 0)

# ── The predicate is a duplicate, so pin it as one ────────────────────────────
# install.ps1's copy sits inside a function and is indented; setup.ps1's is top level. Compare
# with leading whitespace and line endings normalised so only real edits show up.
function Normalize-Src($text) {
    return (($text -split "`r?`n" | ForEach-Object { $_.TrimEnd() } |
             ForEach-Object { $_ -replace '^\s+', '' }) -join "`n").Trim()
}

Write-Host ""
Write-Host "== Test-RespectPmPolicy duplication =="
Check "install.ps1 and studio/setup.ps1 carry textually identical Test-RespectPmPolicy" (
    (Normalize-Src $predicateInstallSrc) -eq (Normalize-Src $predicateSetupSrc)
)
Check 'the predicate reads UNSLOTH_RESPECT_PM_POLICY via [Environment], not $env: interpolation' (
    $predicateInstallSrc -match "GetEnvironmentVariable\('UNSLOTH_RESPECT_PM_POLICY'\)"
)
Check "the predicate trims and lowercases before matching" (
    ($predicateInstallSrc -match '\.Trim\(\)') -and ($predicateInstallSrc -match 'ToLowerInvariant\(\)')
)
Check "the predicate is an allowlist of exactly 1/true/yes/on" (
    $predicateInstallSrc -match "@\('1',\s*'true',\s*'yes',\s*'on'\)"
)

# ── The value matrix ──────────────────────────────────────────────────────────
# The same matrix tests/python/test_pm_policy_optout_shell.py runs, including the
# INTERNAL-whitespace cases: an earlier revision of the shell twin collapsed the inside of a
# value, which made "t rue" and "o n" read as ON there and OFF in Python.
# Expectations are a literal truth table, not a second implementation of the predicate --
# a reimplementation would agree with a broken function for the same reason it broke.
$unsetSentinel = [object]"<<UNSET>>"
$matrix = @(
    @{ v = $unsetSentinel; e = $false; label = "<unset>" }
    @{ v = "";        e = $false }
    @{ v = "1";       e = $true  }
    @{ v = "0";       e = $false }
    @{ v = "true";    e = $true  }
    @{ v = "TRUE";    e = $true  }
    @{ v = "True";    e = $true  }
    @{ v = "TrUe";    e = $true  }
    @{ v = "yes";     e = $true  }
    @{ v = "YES";     e = $true  }
    @{ v = "on";      e = $true  }
    @{ v = "ON";      e = $true  }
    @{ v = "On";      e = $true  }
    @{ v = "false";   e = $false }
    @{ v = "no";      e = $false }
    @{ v = "off";     e = $false }
    @{ v = "garbage"; e = $false }
    @{ v = "2";       e = $false }
    @{ v = "-1";      e = $false }
    @{ v = "enabled"; e = $false }
    @{ v = "onn";     e = $false }
    @{ v = "ye";      e = $false }
    # 't' and 'y' are NOT members here, though setup.ps1's neighbouring boolish predicates
    # (the @("1","t","true","y","yes","on") pair) do take them. A copy-paste from one of those
    # into this one would widen the opt-out silently, so the near-miss is pinned.
    @{ v = "t";       e = $false }
    @{ v = "T";       e = $false }
    @{ v = "y";       e = $false }
    @{ v = "Y";       e = $false }
    @{ v = "n";       e = $false }
    @{ v = "f";       e = $false }
    # padding -- .Trim() takes all of these off, so they stay ON/OFF by their core value
    @{ v = " 1";      e = $true;  label = "' 1'" }
    @{ v = "1 ";      e = $true;  label = "'1 '" }
    @{ v = " 1 ";     e = $true;  label = "' 1 '" }
    @{ v = "  yes  "; e = $true;  label = "'  yes  '" }
    @{ v = "`ttrue";  e = $true;  label = "'<tab>true'" }
    @{ v = "true`t";  e = $true;  label = "'true<tab>'" }
    @{ v = "`t on `t"; e = $true; label = "'<tab> on <tab>'" }
    @{ v = "`n1";     e = $true;  label = "'<lf>1'" }
    @{ v = "true`n";  e = $true;  label = "'true<lf>'" }
    @{ v = "on`n";    e = $true;  label = "'on<lf>'" }
    @{ v = "`non`n";  e = $true;  label = "'<lf>on<lf>'" }
    @{ v = "`r`n1";   e = $true;  label = "'<crlf>1'" }
    @{ v = "1`r`n";   e = $true;  label = "'1<crlf>'" }
    @{ v = "1`r";     e = $true;  label = "'1<cr>'" }
    @{ v = " garbage "; e = $false; label = "' garbage '" }
    @{ v = " 0 ";     e = $false; label = "' 0 '" }
    # INTERNAL whitespace -- never an allowlist member, on any entry point
    @{ v = "t rue";   e = $false; label = "'t rue'" }
    @{ v = "o n";     e = $false; label = "'o n'" }
    @{ v = "1 1";     e = $false; label = "'1 1'" }
    @{ v = "y es";    e = $false; label = "'y es'" }
    @{ v = "tr ue";   e = $false; label = "'tr ue'" }
    @{ v = "1  1";    e = $false; label = "'1  1'" }
    @{ v = "t`true";  e = $false; label = "'t<tab>rue'" }
    @{ v = "o`nn";    e = $false; label = "'o<lf>n'" }
    @{ v = "y e s";   e = $false; label = "'y e s'" }
)

function Set-Policy($value) {
    if ([object]::ReferenceEquals($value, $unsetSentinel)) {
        Remove-Item Env:UNSLOTH_RESPECT_PM_POLICY -ErrorAction SilentlyContinue
    } else {
        [Environment]::SetEnvironmentVariable('UNSLOTH_RESPECT_PM_POLICY', $value)
    }
}

$savedPolicy = [Environment]::GetEnvironmentVariable('UNSLOTH_RESPECT_PM_POLICY')

function Test-Matrix($label) {
    $bad = @()
    foreach ($case in $matrix) {
        Set-Policy $case.v
        $got = Test-RespectPmPolicy
        if ([bool]$got -ne [bool]$case.e) {
            $lbl = if ($case.ContainsKey('label')) { $case.label } else { "'" + $case.v + "'" }
            $bad += "$lbl -> $got (expected $($case.e))"
        }
    }
    Set-Policy $unsetSentinel
    if ($bad.Count -gt 0) { Write-Host ("    " + ($bad -join "; ")) -ForegroundColor Yellow }
    Check "$label answers all $($matrix.Count) matrix values correctly" ($bad.Count -eq 0)
}

try {

Write-Host ""
Write-Host "== predicate value matrix (both copies executed) =="
Invoke-Expression $predicateInstallSrc
Test-Matrix "install.ps1's Test-RespectPmPolicy"
Invoke-Expression $predicateSetupSrc
Test-Matrix "studio/setup.ps1's Test-RespectPmPolicy"

# setup.ps1's copy is the one left defined from here on; it is asserted identical above.

# ── Shared harness for observing a real function's environment ────────────────
$UV_NAMES = @('UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX', 'UV_EXTRA_INDEX_URL',
              'UV_TORCH_BACKEND', 'UV_FIND_LINKS', 'UV_CONFIG_FILE', 'UV_NO_CONFIG',
              'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_INDEX_URL',
              'PIP_CONFIG_FILE')

# Operator-set values, deliberately distinguishable from anything the installers write.
# UV_NO_CONFIG is '0' on purpose: it is the case where an operator has explicitly asked uv to
# keep reading their config, and the one a careless `finally` destroys.
$OPERATOR = @{
    UV_DEFAULT_INDEX    = 'https://mirror.invalid/simple'
    UV_INDEX_URL        = 'https://mirror.invalid/simple'
    UV_INDEX            = 'https://mirror.invalid/simple'
    UV_EXTRA_INDEX_URL  = 'https://extra.invalid/simple'
    UV_TORCH_BACKEND    = 'cpu'
    UV_FIND_LINKS       = '/opt/wheelhouse'
    UV_CONFIG_FILE      = '/etc/uv/uv.toml'
    UV_NO_CONFIG        = '0'
    PIP_EXTRA_INDEX_URL = 'https://extra.invalid/simple'
    PIP_FIND_LINKS      = '/opt/wheelhouse'
    PIP_NO_INDEX        = '1'
    PIP_INDEX_URL       = 'https://mirror.invalid/simple'
    PIP_CONFIG_FILE     = '/etc/pip.conf'
}

function Reset-OperatorEnv {
    foreach ($n in $UV_NAMES) { [Environment]::SetEnvironmentVariable($n, $OPERATOR[$n]) }
}
function Clear-AllEnv {
    foreach ($n in $UV_NAMES) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
}
# Observed but deliberately NOT in $UV_NAMES: that list also drives the "environment
# untouched" comparisons, and a policy variable a test sets on purpose would read there as
# an environment the helper had disturbed.
$POLICY_NAMES = @('PIP_REQUIRE_HASHES', 'UV_REQUIRE_HASHES')
function Snapshot-Env {
    $h = @{}
    foreach ($n in $UV_NAMES + $POLICY_NAMES) { $h[$n] = [Environment]::GetEnvironmentVariable($n) }
    return $h
}

# Drift guard. Every name this test expects a function to scrub or keep must actually appear
# in that function's shipped text; otherwise a renamed variable would leave the assertions
# below passing against an expectation nobody maintains.
function Check-LiteralsPresent($label, $src, $names) {
    $missing = @($names | Where-Object { $src -notmatch [regex]::Escape("'$_'") })
    if ($missing.Count -gt 0) { Write-Host ("    missing: " + ($missing -join ", ")) -ForegroundColor Yellow }
    Check "$label mentions every variable this test reasons about" ($missing.Count -eq 0)
}

# ── install.ps1: Invoke-InstallCommand ────────────────────────────────────────
Write-Host ""
Write-Host "== install.ps1 Invoke-InstallCommand =="

Check-LiteralsPresent "Invoke-InstallCommand" $invokeInstallSrc @(
    'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX', 'UV_EXTRA_INDEX_URL',
    'UV_TORCH_BACKEND', 'UV_FIND_LINKS', 'UV_CONFIG_FILE', 'UV_NO_CONFIG')
Check "Invoke-InstallCommand's scrub is gated on --default-index" (
    $invokeInstallSrc -match "match '--default-index'"
)
Check "Invoke-InstallCommand consults Test-RespectPmPolicy" (
    $invokeInstallSrc -match 'Test-RespectPmPolicy'
)

# Dependencies of the real function. These are logging/redaction only -- none of them touch
# the environment the assertions read, so stubbing them changes nothing under test.
function Write-TauriLog { param($a, $b) }
function Clear-TauriInstallError { param($a) }
function Write-UvDownloadMarker { param($a) }
function Redact-InstallOutput { param($a) return "$a" }
function Write-StudioLine { param($a, $ForegroundColor) }
$script:UnslothVerbose = $false

Invoke-Expression $invokeInstallSrc

# The observation point is the ScriptBlock Invoke-InstallCommand itself invokes, so the real
# function body -- scrub, try and finally -- runs unmodified around it.
$script:seen = $null
$pinnedCmd = {
    # --default-index https://pinned.invalid/simple
    $script:seen = Snapshot-Env
}
$unpinnedCmd = {
    # no index pin here
    $script:seen = Snapshot-Env
}

function Run-InvokeInstallCommand($policy, $cmd) {
    Reset-OperatorEnv
    Set-Policy $policy
    $script:seen = $null
    $rc = Invoke-InstallCommand -Command $cmd -Label "test"
    $after = Snapshot-Env
    Set-Policy $unsetSentinel
    return @{ rc = $rc; during = $script:seen; after = $after }
}

# Default arm: no opt-out.
$r = Run-InvokeInstallCommand $unsetSentinel $pinnedCmd
Check "default arm: exit code surfaced as 0" ($r.rc -eq 0)
Check "default arm: the command actually ran (env was observed)" ($null -ne $r.during)
Check "default arm: strips UV_CONFIG_FILE" ($null -eq $r.during.UV_CONFIG_FILE)
Check "default arm: strips UV_FIND_LINKS" ($null -eq $r.during.UV_FIND_LINKS)
Check "default arm: forces UV_NO_CONFIG=1 over the operator's '0'" ($r.during.UV_NO_CONFIG -eq '1')
Check "default arm: strips all five additive index vars" (
    ($null -eq $r.during.UV_INDEX_URL) -and ($null -eq $r.during.UV_EXTRA_INDEX_URL) -and
    ($null -eq $r.during.UV_DEFAULT_INDEX) -and ($null -eq $r.during.UV_INDEX) -and
    ($null -eq $r.during.UV_TORCH_BACKEND)
)
$notRestored = @(@('UV_CONFIG_FILE','UV_FIND_LINKS','UV_NO_CONFIG','UV_INDEX_URL','UV_EXTRA_INDEX_URL',
                   'UV_DEFAULT_INDEX','UV_INDEX','UV_TORCH_BACKEND') |
                 Where-Object { $r.after[$_] -ne $OPERATOR[$_] })
Check "default arm: restores every scrubbed value afterwards, UV_NO_CONFIG='0' included" ($notRestored.Count -eq 0)

# Opt-out arm, once per truthy spelling, so the arm is not reached by one value only.
foreach ($truthy in @('1', 'TRUE', 'yes', ' on ')) {
    $r = Run-InvokeInstallCommand $truthy $pinnedCmd
    $tag = "opt-out arm ($($truthy -replace ' ', '_'))"
    Check "${tag}: keeps UV_CONFIG_FILE" ($r.during.UV_CONFIG_FILE -eq $OPERATOR.UV_CONFIG_FILE)
    Check "${tag}: keeps UV_FIND_LINKS" ($r.during.UV_FIND_LINKS -eq $OPERATOR.UV_FIND_LINKS)
    Check "${tag}: keeps the operator's UV_NO_CONFIG='0' (does not force 1)" ($r.during.UV_NO_CONFIG -eq '0')
    Check "${tag}: still strips UV_INDEX_URL/UV_EXTRA_INDEX_URL/UV_DEFAULT_INDEX/UV_TORCH_BACKEND" (
        ($null -eq $r.during.UV_INDEX_URL) -and ($null -eq $r.during.UV_EXTRA_INDEX_URL) -and
        ($null -eq $r.during.UV_DEFAULT_INDEX) -and ($null -eq $r.during.UV_TORCH_BACKEND)
    )
    Check "${tag}: still strips UV_INDEX" ($null -eq $r.during.UV_INDEX)
    # The point of the guard in the finally: UV_NO_CONFIG was never saved on this arm, so an
    # unconditional Remove-Item there would wipe the operator's own value for the whole run.
    Check "${tag}: finally leaves the operator's UV_NO_CONFIG='0' intact" ($r.after.UV_NO_CONFIG -eq '0')
    Check "${tag}: restores the additive index vars afterwards" (
        ($r.after.UV_INDEX_URL -eq $OPERATOR.UV_INDEX_URL) -and
        ($r.after.UV_DEFAULT_INDEX -eq $OPERATOR.UV_DEFAULT_INDEX) -and
        ($r.after.UV_TORCH_BACKEND -eq $OPERATOR.UV_TORCH_BACKEND)
    )
}

# A falsy / unrecognised value must land on the default arm, not the opt-out.
foreach ($falsy in @('0', 'garbage', '', 'o n', 't rue')) {
    $r = Run-InvokeInstallCommand $falsy $pinnedCmd
    Check "falsy '$falsy' takes the DEFAULT arm (UV_CONFIG_FILE stripped, UV_NO_CONFIG=1)" (
        ($null -eq $r.during.UV_CONFIG_FILE) -and ($r.during.UV_NO_CONFIG -eq '1')
    )
}

# No pin, no scrub -- on either arm.
foreach ($policy in @($unsetSentinel, '1')) {
    $r = Run-InvokeInstallCommand $policy $unpinnedCmd
    $tag = if ([object]::ReferenceEquals($policy, $unsetSentinel)) { "unset" } else { "opt-out" }
    Check "no --default-index ($tag): environment untouched" (
        @($UV_NAMES | Where-Object { $r.during[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
}

# ── studio/setup.ps1: Fast-Install ────────────────────────────────────────────
Write-Host ""
Write-Host "== studio/setup.ps1 Fast-Install =="

$FI_SCRUB = @('UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX', 'UV_EXTRA_INDEX_URL',
              'UV_TORCH_BACKEND', 'UV_FIND_LINKS', 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS',
              'PIP_NO_INDEX', 'PIP_INDEX_URL', 'UV_CONFIG_FILE', 'UV_NO_CONFIG', 'PIP_CONFIG_FILE')
$FI_KEEP  = @('UV_CONFIG_FILE', 'UV_NO_CONFIG', 'PIP_CONFIG_FILE', 'PIP_NO_INDEX',
              'PIP_FIND_LINKS', 'UV_FIND_LINKS')

Check-LiteralsPresent "Fast-Install" $fastInstallSrc $FI_SCRUB
Check "Fast-Install's scrub is gated on --index-url" ($fastInstallSrc -match "'--index-url'")
Check "Fast-Install consults Test-RespectPmPolicy" ($fastInstallSrc -match 'Test-RespectPmPolicy')

Invoke-Expression $removeUvFlagsSrc
Invoke-Expression $pipEnvFlagSrc
Invoke-Expression $uvEnvFlagSrc
Invoke-Expression $fastInstallSrc

# Stand-ins for the two NATIVE commands Fast-Install shells out to. PowerShell resolves a
# function ahead of an application, so `& uv` and `& python` inside the real body land here --
# the body itself, including the scrub and the finally, is the shipped text.
# $script:uvExit lets a test make the uv stand-in FAIL, which is the only way to reach the
# fallback branch at all. Without it every Fast-Install check ran the success path and the
# refusal below could never have been observed.
$script:uvExit = 0
# setup.ps1's own reporter. Fast-Install calls it on the refusal path, and without a
# stand-in the extracted function threw CommandNotFoundException mid-suite, which read as a
# failing assertion rather than as a missing dependency.
function substep($text, $colour) { $script:lastSubstep = "$text" }
function uv { $global:LASTEXITCODE = $script:uvExit; $script:seen = Snapshot-Env; $script:sawUv = $true }
function python { $global:LASTEXITCODE = 0; $script:seen = Snapshot-Env; $script:sawPip = $true }

function Run-FastInstall($policy, $args_, $useUv) {
    Reset-OperatorEnv
    Set-Policy $policy
    $script:seen = $null; $script:sawUv = $false; $script:sawPip = $false
    $script:UseUv = $useUv
    $UseUv = $useUv
    Fast-Install @args_ | Out-Null
    $after = Snapshot-Env
    Set-Policy $unsetSentinel
    return @{ during = $script:seen; after = $after; uv = $script:sawUv; pip = $script:sawPip }
}

$pinnedArgs = @('torch', '--index-url', 'https://pinned.invalid/simple')

# A FAILING uv, which is the branch the opt-out changes: pip must not stand in for a resolver
# that was never told what uv refused.
$script:uvExit = 1
$r = Run-FastInstall $unsetSentinel $pinnedArgs $true
Check "Fast-Install uv failure, default: still falls back to pip" ($r.uv -and $r.pip)
$r = Run-FastInstall '1' $pinnedArgs $true
Check "Fast-Install uv failure, opt-out: uv ran" $r.uv
Check "Fast-Install uv failure, opt-out: pip is NOT reached" (-not $r.pip)
Check "Fast-Install uv failure, opt-out: reports a non-zero exit" ($LASTEXITCODE -ne 0)

# A pip-expressed hash policy has to reach the uv run standing in for pip.
$script:uvExit = 0
Reset-OperatorEnv
$env:PIP_REQUIRE_HASHES = '1'
Remove-Item Env:UV_REQUIRE_HASHES -ErrorAction SilentlyContinue
$r = Run-FastInstall '1' $pinnedArgs $true
Check "Fast-Install opt-out: carries PIP_REQUIRE_HASHES into UV_REQUIRE_HASHES" (
    $r.during.UV_REQUIRE_HASHES -eq '1'
)
Check "Fast-Install opt-out: the carried UV_REQUIRE_HASHES does not outlive the call" (
    $null -eq [Environment]::GetEnvironmentVariable('UV_REQUIRE_HASHES')
)
Reset-OperatorEnv
$env:PIP_REQUIRE_HASHES = '1'
$env:UV_REQUIRE_HASHES = '0'
$r = Run-FastInstall '1' $pinnedArgs $true
Check "Fast-Install opt-out: an explicit UV_REQUIRE_HASHES the operator set is not overwritten" (
    $r.during.UV_REQUIRE_HASHES -eq '0'
)
# Reset-OperatorEnv only restores the names it owns, so clear the two this block introduced
# or they leak into every later check as an environment that was never "untouched".
Remove-Item Env:PIP_REQUIRE_HASHES -ErrorAction SilentlyContinue
Remove-Item Env:UV_REQUIRE_HASHES -ErrorAction SilentlyContinue
Reset-OperatorEnv
$script:uvExit = 0

$r = Run-FastInstall $unsetSentinel $pinnedArgs $true
Check "Fast-Install default arm: the uv stand-in ran and the env was observed" ($r.uv -and $null -ne $r.during)
Check "Fast-Install default arm: scrubs all thirteen names" (
    @($FI_SCRUB | Where-Object { $_ -notin @('UV_NO_CONFIG','PIP_CONFIG_FILE') } |
        Where-Object { $null -ne $r.during[$_] }).Count -eq 0
)
Check "Fast-Install default arm: forces UV_NO_CONFIG=1 and PIP_CONFIG_FILE=nul" (
    ($r.during.UV_NO_CONFIG -eq '1') -and ($r.during.PIP_CONFIG_FILE -eq 'nul')
)
Check "Fast-Install default arm: restores every operator value afterwards" (
    @($UV_NAMES | Where-Object { $r.after[$_] -ne $OPERATOR[$_] }).Count -eq 0
)

foreach ($truthy in @('1', 'True', 'ON')) {
    $r = Run-FastInstall $truthy $pinnedArgs $true
    $tag = "Fast-Install opt-out ($truthy)"
    Check "${tag}: keeps the full keep-list untouched" (
        @($FI_KEEP | Where-Object { $r.during[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
    $leaked = @(@('UV_DEFAULT_INDEX','UV_INDEX_URL','UV_INDEX','UV_EXTRA_INDEX_URL','UV_TORCH_BACKEND',
                  'PIP_EXTRA_INDEX_URL','PIP_INDEX_URL') |
                Where-Object { $null -ne $r.during[$_] })
    Check "${tag}: still scrubs the additive index vars" ($leaked.Count -eq 0)
    Check "${tag}: does not force UV_NO_CONFIG or PIP_CONFIG_FILE" (
        ($r.during.UV_NO_CONFIG -eq '0') -and ($r.during.PIP_CONFIG_FILE -eq $OPERATOR.PIP_CONFIG_FILE)
    )
    Check "${tag}: keeps PIP_NO_INDEX whichever way it points" ($r.during.PIP_NO_INDEX -eq '1')
    Check "${tag}: finally leaves the operator's UV_NO_CONFIG and PIP_CONFIG_FILE intact" (
        ($r.after.UV_NO_CONFIG -eq '0') -and ($r.after.PIP_CONFIG_FILE -eq $OPERATOR.PIP_CONFIG_FILE)
    )
    Check "${tag}: restores everything else afterwards" (
        @($UV_NAMES | Where-Object { $r.after[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
}

# The pip fallback must see the same environment as the uv attempt: a scrub that only holds
# for the uv half would let a user pip.conf back in on exactly the installs uv could not do.
foreach ($policy in @($unsetSentinel, '1')) {
    $r = Run-FastInstall $policy $pinnedArgs $false
    $tag = if ([object]::ReferenceEquals($policy, $unsetSentinel)) { "default" } else { "opt-out" }
    Check "Fast-Install pip fallback ($tag): reached the pip stand-in" ($r.pip)
    $expectKept = if ($tag -eq "opt-out") { $OPERATOR.UV_CONFIG_FILE } else { $null }
    Check "Fast-Install pip fallback ($tag): same UV_CONFIG_FILE decision as the uv path" (
        $r.during.UV_CONFIG_FILE -eq $expectKept
    )
}

# No --index-url means Fast-Install is not a pinned install and scrubs nothing.
foreach ($policy in @($unsetSentinel, '1')) {
    $r = Run-FastInstall $policy @('unsloth') $true
    $tag = if ([object]::ReferenceEquals($policy, $unsetSentinel)) { "default" } else { "opt-out" }
    Check "Fast-Install unpinned ($tag): environment untouched" (
        @($UV_NAMES | Where-Object { $r.during[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
}

# ── studio/setup.ps1: Fast-Download ───────────────────────────────────────────
Write-Host ""
Write-Host "== studio/setup.ps1 Fast-Download =="

$FD_SCRUB = @('PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_INDEX_URL', 'PIP_CONFIG_FILE')
$FD_KEEP  = @('PIP_CONFIG_FILE', 'PIP_NO_INDEX', 'PIP_FIND_LINKS')

Check-LiteralsPresent "Fast-Download" $fastDownloadSrc $FD_SCRUB
Check "Fast-Download consults Test-RespectPmPolicy" ($fastDownloadSrc -match 'Test-RespectPmPolicy')
Check "Fast-Download touches no UV_ variable (pip cannot read them)" (
    $fastDownloadSrc -notmatch "'UV_"
)

Invoke-Expression $fastDownloadSrc

function Run-FastDownload($policy) {
    Reset-OperatorEnv
    Set-Policy $policy
    $script:seen = $null; $script:sawPip = $false
    Fast-Download --no-deps --dest /tmp/x 'triton-xpu' --index-url 'https://pinned.invalid/simple' | Out-Null
    $after = Snapshot-Env
    Set-Policy $unsetSentinel
    return @{ during = $script:seen; after = $after; pip = $script:sawPip }
}

$r = Run-FastDownload $unsetSentinel
Check "Fast-Download default arm: the pip stand-in ran" ($r.pip -and $null -ne $r.during)
$fdLeaked = @(@('PIP_EXTRA_INDEX_URL','PIP_FIND_LINKS','PIP_NO_INDEX','PIP_INDEX_URL') |
              Where-Object { $null -ne $r.during[$_] })
Check "Fast-Download default arm: scrubs PIP_EXTRA_INDEX_URL/PIP_FIND_LINKS/PIP_NO_INDEX/PIP_INDEX_URL" ($fdLeaked.Count -eq 0)
Check "Fast-Download default arm: forces PIP_CONFIG_FILE=nul" ($r.during.PIP_CONFIG_FILE -eq 'nul')
Check "Fast-Download default arm: restores every operator value afterwards" (
    @($UV_NAMES | Where-Object { $r.after[$_] -ne $OPERATOR[$_] }).Count -eq 0
)

foreach ($truthy in @('1', 'yes', "`ttrue")) {
    $r = Run-FastDownload $truthy
    $tag = "Fast-Download opt-out"
    Check "${tag}: keeps PIP_CONFIG_FILE/PIP_NO_INDEX/PIP_FIND_LINKS" (
        @($FD_KEEP | Where-Object { $r.during[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
    Check "${tag}: still scrubs PIP_EXTRA_INDEX_URL and PIP_INDEX_URL" (
        ($null -eq $r.during.PIP_EXTRA_INDEX_URL) -and ($null -eq $r.during.PIP_INDEX_URL)
    )
    Check "${tag}: finally leaves the operator's PIP_CONFIG_FILE intact" (
        $r.after.PIP_CONFIG_FILE -eq $OPERATOR.PIP_CONFIG_FILE
    )
    Check "${tag}: restores everything else afterwards" (
        @($UV_NAMES | Where-Object { $r.after[$_] -ne $OPERATOR[$_] }).Count -eq 0
    )
}

# ── studio/setup.ps1: the XPU triton swap gate ────────────────────────────────
# Not executable in isolation (it sits mid-script, after a live python probe), so this is a
# structural assertion only: the opt-out arm must come FIRST, because an `elseif` reached
# after the swap has already removed triton-windows cannot un-remove it.
Write-Host ""
Write-Host "== studio/setup.ps1 XPU triton swap gate =="
$setupText = [System.IO.File]::ReadAllText($setupPs1)
$gateIdx  = $setupText.IndexOf("`$_tritonXpuSpec -match '(?i)xpu' -and (Test-RespectPmPolicy)")
$swapIdx  = $setupText.IndexOf("elseif (`$_tritonWinVer -and `$_tritonXpuSpec -match '(?i)xpu')")
Check "the opt-out gate exists on the XPU triton swap" ($gateIdx -gt 0)
Check "the swap is the elseif, so the opt-out is evaluated first" (($swapIdx -gt 0) -and ($gateIdx -lt $swapIdx))
Check "the declined path warns rather than proceeding silently" (
    $setupText.Substring($gateIdx, [Math]::Min(1600, $swapIdx - $gateIdx)) -match 'UNSLOTH_RESPECT_PM_POLICY is set'
)

}
finally {
    Clear-AllEnv
    if ($null -eq $savedPolicy) { Remove-Item Env:UNSLOTH_RESPECT_PM_POLICY -ErrorAction SilentlyContinue }
    else { [Environment]::SetEnvironmentVariable('UNSLOTH_RESPECT_PM_POLICY', $savedPolicy) }
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED ($failures)" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
exit 0
