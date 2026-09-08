#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A whitespace-only UNSLOTH_STUDIO_HOME must not swallow the custom-root chain.
#
# PowerShell calls " " true, so `if ($env:UNSLOTH_STUDIO_HOME)` used to win the precedence test
# on a value made of spaces: STUDIO_HOME was never examined, _CustomStudioRoots returned nothing,
# and the real custom install kept its tree and its studio.db. install.ps1:783-788 has always
# resolved the same two variables with [string]::IsNullOrWhiteSpace + .Trim(), so this was an
# installer/uninstaller disagreement, and an UNDER-delete rather than an over-delete.
#
# The closing "re-export UNSLOTH_STUDIO_HOME and re-run" hint had the same gate, so the one run
# that failed to find the install was also the one that stayed quiet about it.
#
# The uninstaller body kills processes and writes to the registry, so it cannot be executed here;
# the resolver is lifted out of the script by AST and exercised on its own, and the hint gate is
# lifted as source text and evaluated.
#
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_ws_root_precedence.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")
$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)

# Lift the resolver and its two helpers. A silently empty extraction is what makes a suite
# like this vacuous, so a miss is fatal rather than a skip.
foreach ($name in @("_ExpandTilde", "_RootFromConf", "_CustomStudioRoots")) {
    $fn = $ast.FindAll({
            param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
        }, $true) | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if (-not $fn) {
        Write-Host "  FAIL  $name not found in uninstall.ps1" -ForegroundColor Red
        exit 1
    }
    . ([scriptblock]::Create($fn.Extent.Text))
}

# install.ps1 is the producer this file has to agree with. Assert its convention still stands,
# so a future change there cannot silently re-open the disagreement from the other side.
$installText = Get-Content -LiteralPath $installPath -Raw
Check "install.ps1 still resolves UNSLOTH_STUDIO_HOME with IsNullOrWhiteSpace + .Trim()" `
    ($installText -match '\[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_STUDIO_HOME\)' -and
     $installText -match '\$env:UNSLOTH_STUDIO_HOME\.Trim\(\)')
Check "install.ps1 still resolves STUDIO_HOME the same way" `
    ($installText -match '\[string\]::IsNullOrWhiteSpace\(\$env:STUDIO_HOME\)' -and
     $installText -match '\$env:STUDIO_HOME\.Trim\(\)')

# The closing hint gate, lifted as source text so the test fails if that line regresses on its
# own. Evaluating the real expression, not a copy of it, is what keeps this honest.
$ps1Text = Get-Content -LiteralPath $ps1Path -Raw
$hintGate = $null
if ($ps1Text -match '(?m)^\s*if\s*(\((?:[^\r\n]*UNSLOTH_STUDIO_HOME[^\r\n]*STUDIO_HOME[^\r\n]*)\))\s*\{\s*$') {
    $hintGate = $Matches[1]
}
Check "the closing custom-root hint gate was found in uninstall.ps1" ($null -ne $hintGate)

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-wsroot-" + [System.Guid]::NewGuid().ToString("N"))
$savedProfile = $env:USERPROFILE
$savedLocal = $env:LOCALAPPDATA
$savedStudioHome = $env:UNSLOTH_STUDIO_HOME
$savedAlias = $env:STUDIO_HOME
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
    $real = Join-Path $tmp "real"
    $other = Join-Path $tmp "other"
    New-Item -ItemType Directory -Path $real -Force | Out-Null
    New-Item -ItemType Directory -Path $other -Force | Out-Null
    $realNorm = [System.IO.Path]::GetFullPath($real).TrimEnd('\', '/')
    $otherNorm = [System.IO.Path]::GetFullPath($other).TrimEnd('\', '/')

    # A profile and LocalAppData of our own, so the machine's real install cannot leak in and
    # the default root this resolver excludes is one that does not exist.
    $env:USERPROFILE = Join-Path $tmp "profile"
    $env:LOCALAPPDATA = Join-Path $tmp "localappdata"

    function Roots($studioHome, $alias) {
        $env:UNSLOTH_STUDIO_HOME = $studioHome
        $env:STUDIO_HOME = $alias
        return @(_CustomStudioRoots)
    }
    function HintShown($studioHome, $alias) {
        $env:UNSLOTH_STUDIO_HOME = $studioHome
        $env:STUDIO_HOME = $alias
        return [bool](Invoke-Expression $hintGate)
    }

    # --- The defect: a blank primary must not hide a genuinely set alias. ---
    foreach ($blank in @(" ", "   ", "`t", " `t ")) {
        $got = Roots $blank $real
        Check "whitespace UNSLOTH_STUDIO_HOME ('$($blank -replace "`t", '\t')') still finds STUDIO_HOME" `
            ($got -contains $realNorm)
    }
    Check "empty UNSLOTH_STUDIO_HOME still finds STUDIO_HOME" ((Roots "" $real) -contains $realNorm)
    Check "unset UNSLOTH_STUDIO_HOME still finds STUDIO_HOME" ((Roots $null $real) -contains $realNorm)

    # --- Precedence is NOT relaxed: a real primary keeps suppressing the alias. ---
    # This is what stops uninstalling install A from taking install B with it, so it has to be
    # asserted alongside the fix, not instead of it.
    $both = Roots $real $other
    Check "a genuinely set UNSLOTH_STUDIO_HOME is used" ($both -contains $realNorm)
    Check "a genuinely set UNSLOTH_STUDIO_HOME suppresses STUDIO_HOME" (-not ($both -contains $otherNorm))
    $padded = Roots (" " + $real + " ") $other
    Check "a padded UNSLOTH_STUDIO_HOME resolves to the trimmed root" ($padded -contains $realNorm)
    Check "a padded UNSLOTH_STUDIO_HOME still suppresses STUDIO_HOME" (-not ($padded -contains $otherNorm))
    Check "a padded STUDIO_HOME resolves when it is the only one set" `
        ((Roots "" (" " + $other + " ")) -contains $otherNorm)

    # --- The fix must not collapse into ignoring the variables. ---
    foreach ($pair in @(@(" ", " "), @("", ""), @($null, $null), @(" ", ""), @("", " "))) {
        $got = Roots $pair[0] $pair[1]
        Check "both blank ('$($pair[0])' / '$($pair[1])') yields no custom root" ($got.Count -eq 0)
    }

    # --- The closing hint, on the real expression from the file. ---
    Check "hint is shown when both are whitespace" (HintShown " " "  ")
    Check "hint is shown when both are empty" (HintShown "" "")
    Check "hint is shown when whitespace primary meets empty alias" (HintShown " " "")
    Check "hint is suppressed when UNSLOTH_STUDIO_HOME is genuinely set" (-not (HintShown $real ""))
    Check "hint is suppressed when only STUDIO_HOME is genuinely set" (-not (HintShown "" $other))
    Check "hint is suppressed when a padded UNSLOTH_STUDIO_HOME is set" (-not (HintShown (" $real ") ""))
}
finally {
    $env:USERPROFILE = $savedProfile
    $env:LOCALAPPDATA = $savedLocal
    $env:UNSLOTH_STUDIO_HOME = $savedStudioHome
    $env:STUDIO_HOME = $savedAlias
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    Write-Host "FAILED ($failures)" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
