#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A reparse-point Unsloth home must have its physical path in the stop scan: the backend
# resolves it, so sd-server runs under the target and a lexical prefix match never stops it.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_reparse_stop_roots.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)

# A silently empty extraction would make this suite vacuous.
$fn = $ast.FindAll({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "_ManagedPathsUnderReparseTargets"
    }, $true) | Select-Object -First 1
if (-not $fn) {
    Write-Host "  FAIL  _ManagedPathsUnderReparseTargets not found in uninstall.ps1" -ForegroundColor Red
    exit 1
}
. ([scriptblock]::Create($fn.Extent.Text))

$ps1Text = Get-Content -LiteralPath $ps1Path -Raw
# The stop scan runs before the ownership gates, so it must only see owned roots.
Check "the stop scan is given the managed paths under the target" `
    ($ps1Text -match '_StopProcessesLockingRoots -Roots \(\$stopRoots \+ @\(_ManagedPathsUnderReparseTargets \$ownedRoots\)\)')
Check "and that list is filtered by the ownership gate" `
    ($ps1Text -match '(?s)\$ownedRoots = @\(\).*?_IsStudioRoot \$defaultStudioHome -ManagedDefaultRoot.*?foreach \(\$r in \$customRoots\) \{\s*\r?\n\s*if \(\(_IsStudioRoot \$r\) -and -not \(_IsUnsafeRoot \$r\)\)')
Check "and by the deny list, which the removal loop also refuses on" `
    ($ps1Text -match '(?s)\$ownedRoots = @\(\).*?-not \(_IsUnsafeRoot \$defaultStudioHome\)')
# A stale studio.conf can name a directory somebody else now uses.
Check "the stop scan is given the gated roots, not every known one" `
    ($ps1Text -match '(?m)^\s*\$stopRoots = @\(\$ownedRoots\) \+')
Check "and so is the process sweep" `
    ($ps1Text -match '(?m)^\s*_StopStudioProcesses -KnownRoots \$ownedRoots\s*$')
# @() is false in PowerShell, so `if ($KnownRoots)` would sweep the whole machine.
Check "an explicitly empty root list still scopes the sweep" `
    ($ps1Text -match [regex]::Escape("`$scoped = `$PSBoundParameters.ContainsKey('KnownRoots')"))
Check "and the sweep branches on that, not on the array's truthiness" `
    ($ps1Text -match '(?m)^\s*if \(\$scoped\) \{\s*$')
Check "the port-file stopper is gated as well" `
    (-not ($ps1Text -match '_StopByPortFile -PortFile [^\r\n]*-KnownRoots \$knownRoots'))
Check "and that list is built before the stop step" `
    ([regex]::Match($ps1Text, '(?m)^\s*\$ownedRoots = @\(\)').Index -lt
     [regex]::Match($ps1Text, '(?m)^\s*_StopStudioProcesses -KnownRoots').Index)
Check "a partial removal puts the ownership marker back" `
    ($ps1Text -match '(?m)^\s*_RemovePath \$Path\s*\r?\n\s*_RestoreOwnerMarker \$Path\s*$')
Check "the legacy sd.cpp stop root is gated the same way" `
    ($ps1Text -match '(?s)\$customSdCppToStop = @\(\)\s*\r?\n\s*foreach \(\$r in \$customRoots\) \{\s*\r?\n\s*if \(-not \(_IsStudioRoot \$r\)\) \{ continue \}')

# $IsWindows exists only on PowerShell 6+, so 5.1 falls through to $true.
$onWindows = if ($null -ne $IsWindows) { $IsWindows } else { $true }

# New-Item -ItemType Junction does not throw on Linux pwsh; it makes a plain dir.
$kinds = if ($onWindows) { @("Junction", "SymbolicLink") } else { @("SymbolicLink") }

$ran = 0
foreach ($kind in $kinds) {
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-reparse-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        $target = Join-Path $tmp "physical"
        New-Item -ItemType Directory -Path (Join-Path $target "stable-diffusion.cpp") -Force | Out-Null
        $link = Join-Path $tmp "studio-home"
        try { New-Item -ItemType $kind -Path $link -Target $target -ErrorAction Stop | Out-Null }
        catch {
            Write-Host "  SKIP  $kind is not creatable here: $($_.Exception.Message)"
            continue
        }
        $made = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
        if (-not $made -or [string]::IsNullOrWhiteSpace(@($made.Target)[0])) {
            Write-Host "  SKIP  $kind produced no reparse target here"
            continue
        }
        $ran++

        $phys = [System.IO.Path]::GetFullPath($target).TrimEnd('\', '/')
        $got = @(_ManagedPathsUnderReparseTargets @($link))
        Check "$kind : a linked root yields the sd.cpp tree under its physical target" `
            ($got -contains (Join-Path $phys "stable-diffusion.cpp"))
        Check "$kind : ... and the venv under it" ($got -contains (Join-Path $phys "unsloth_studio"))
        Check "$kind : the bare physical target is NOT in scope" (-not ($got -contains $phys))

        $plain = Join-Path $tmp "plain"
        New-Item -ItemType Directory -Path $plain -Force | Out-Null
        Check "$kind : a plain root adds nothing" (@(_ManagedPathsUnderReparseTargets @($plain)).Count -eq 0)

        Check "$kind : a missing root adds nothing" (@(_ManagedPathsUnderReparseTargets @((Join-Path $tmp "nope"), "", $null)).Count -eq 0)

        $link2 = Join-Path $tmp "studio-home-2"
        New-Item -ItemType $kind -Path $link2 -Target $target -ErrorAction Stop | Out-Null
        $both = @(_ManagedPathsUnderReparseTargets @($link, $link2))
        Check "$kind : two links onto one target do not duplicate its subtrees" ($both.Count -eq $got.Count)
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Check "at least one reparse kind was exercised" ($ran -gt 0)

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
