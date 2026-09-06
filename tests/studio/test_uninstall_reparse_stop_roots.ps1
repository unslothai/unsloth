#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A junction / directory-symlink Unsloth home must have its PHYSICAL path in the stop scan.
#
# The backend resolves UNSLOTH_STUDIO_HOME (Path.resolve) before deriving
# <home>\stable-diffusion.cpp, so sd-server's real image path is under the reparse TARGET.
# _CustomStudioRoots only runs System.IO.Path.GetFullPath, which is lexical and leaves the link
# path as-is, and _StopProcessesLockingRoots matches Win32_Process.ExecutablePath by prefix -- so
# without the target the server is never stopped, its tree is deleted around it and it keeps
# holding its port. _ManagedPathsUnderReparseTargets supplies the target for the stop scan only.
#
# The uninstaller body kills processes and writes to the registry, so it cannot be executed here;
# the helper is lifted out of the script by AST and exercised on its own.
#
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

# Lift the helper out. A silently empty extraction is what makes a suite like this vacuous.
$fn = $ast.FindAll({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "_ManagedPathsUnderReparseTargets"
    }, $true) | Select-Object -First 1
if (-not $fn) {
    Write-Host "  FAIL  _ManagedPathsUnderReparseTargets not found in uninstall.ps1" -ForegroundColor Red
    exit 1
}
. ([scriptblock]::Create($fn.Extent.Text))

# The stop-scan call site has to actually pass the targets, or the helper is dead code.
$ps1Text = Get-Content -LiteralPath $ps1Path -Raw
Check "the stop scan is given the managed paths under the target" `
    ($ps1Text -match '_StopProcessesLockingRoots -Roots \(\$stopRoots \+ @\(_ManagedPathsUnderReparseTargets \$knownRoots\)\)')

# Both reparse kinds, because the helper reads nothing but .Target and both supply one.
# A directory SYMLINK needs elevation or Developer Mode; a JUNCTION never does. Creating only
# symlinks made this file unrunnable unprivileged -- New-Item threw
# NewItemSymbolicLinkElevationRequired before a single subtree check ran -- so the junction row
# is the one that always executes, and CI, whose Windows runner is an administrator, still
# covers both. A junction is the shape a Windows user is likelier to have anyway: mklink /J
# needs no privilege, so it is what gets reached for when moving a studio home onto another
# volume.
# $IsWindows only exists on PowerShell 6+; on Windows PowerShell 5.1 it is $null, and 5.1
# runs nowhere else.
$onWindows = if ($null -ne $IsWindows) { $IsWindows } else { $true }

# A junction is a Windows-only reparse type. New-Item -ItemType Junction does NOT throw on
# Linux pwsh -- it quietly produces a plain directory with no .Target -- so asking for one
# there yields a row that fails every positive assertion while looking like it ran. The
# kinds are chosen by platform rather than by catching a failure that never comes.
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
            # Not a failure: the other kind carries the assertions. Reported so a run that
            # covered only one kind cannot be mistaken for a run that covered both.
            Write-Host "  SKIP  $kind is not creatable here: $($_.Exception.Message)"
            continue
        }
        # Created is not the same as usable. _ManagedPathsUnderReparseTargets reads nothing
        # but .Target, so a link without one cannot exercise anything, and treating that as a
        # failure would blame the helper for the filesystem's answer.
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
        # Never the bare target: the delete leaves it standing, so anything else there is not ours.
        Check "$kind : the bare physical target is NOT in scope" (-not ($got -contains $phys))

        # A plain directory contributes nothing, so the scan does not widen for ordinary installs.
        $plain = Join-Path $tmp "plain"
        New-Item -ItemType Directory -Path $plain -Force | Out-Null
        Check "$kind : a plain root adds nothing" (@(_ManagedPathsUnderReparseTargets @($plain)).Count -eq 0)

        # Neither does a path that is not there at all, or an empty entry.
        Check "$kind : a missing root adds nothing" (@(_ManagedPathsUnderReparseTargets @((Join-Path $tmp "nope"), "", $null)).Count -eq 0)

        # Deduplicated: two links onto one target must not stack.
        $link2 = Join-Path $tmp "studio-home-2"
        New-Item -ItemType $kind -Path $link2 -Target $target -ErrorAction Stop | Out-Null
        $both = @(_ManagedPathsUnderReparseTargets @($link, $link2))
        Check "$kind : two links onto one target do not duplicate its subtrees" ($both.Count -eq $got.Count)
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# An environment that can make neither kind would otherwise report a clean pass having
# asserted nothing about reparse points at all.
Check "at least one reparse kind was exercised" ($ran -gt 0)

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
