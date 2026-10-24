#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests Get-UvSafeRequirementsPath: uv splits -r on whitespace with no quoting, so a spaced
# path must become space-free; only copies are marked temporary.
# Run: pwsh -NoProfile -File tests/studio/test_uv_safe_requirements_path.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Get-UvSafeRequirementsPath"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one Get-UvSafeRequirementsPath in install.ps1, found $($fn.Count)" }
Invoke-Expression $fn[0].Extent.Text

# Captured, not stubbed, so the give-up warning is part of the contract.
$script:Warnings = @()
function substep { param([string]$Text, [string]$Colour) $script:Warnings += $Text }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("uvsafe-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $root | Out-Null
try {
    $plainDir = Join-Path $root "plain"
    New-Item -ItemType Directory -Force -Path $plainDir | Out-Null
    $plain = Join-Path $plainDir "no-torch-runtime.txt"
    "typer" | Set-Content -LiteralPath $plain
    $r = Get-UvSafeRequirementsPath -Path $plain
    Check "a space-free path is returned unchanged" ($r.Path -eq $plain)
    Check "a space-free path is never marked temporary" (-not $r.Temporary)
    Check "the untouched path emits no warning" ($script:Warnings.Count -eq 0)

    $spacedDir = Join-Path $root "has space"
    New-Item -ItemType Directory -Force -Path $spacedDir | Out-Null
    $spaced = Join-Path $spacedDir "no-torch-runtime.txt"
    "typer" | Set-Content -LiteralPath $spaced
    $r2 = Get-UvSafeRequirementsPath -Path $spaced
    Check "a spaced path never comes back containing a space" (-not $r2.Path.Contains(" "))
    Check "the space-free path carries the same content" ((Get-Content -Raw -LiteralPath $r2.Path).Trim() -eq "typer")

    # Only a copy may be deleted. With 8.3 names the helper returns an alias of the user's file
    # (Temporary = $false), so compare file identity, not spelling, on both branches.
    function Test-SameUnderlyingFile([string]$a, [string]$b) {
        try {
            if ((Get-Item -LiteralPath $a -Force).FullName -eq (Get-Item -LiteralPath $b -Force).FullName) { return $true }
        } catch { return $false }
        # Different strings can still be one file: write through one, read through the other.
        $probe = "typer-probe-" + [guid]::NewGuid().ToString("N")
        try {
            $original = Get-Content -Raw -LiteralPath $b
            $probe | Set-Content -LiteralPath $b
            $same = ((Get-Content -Raw -LiteralPath $a).Trim() -eq $probe)
            $original.TrimEnd("`r", "`n") | Set-Content -LiteralPath $b
            return $same
        } catch { return $false }
    }

    if ($r2.Temporary) {
        Check "a copy is not the user's own file" (-not (Test-SameUnderlyingFile $r2.Path $spaced))
        Remove-Item -LiteralPath $r2.Path -Force
        Check "deleting the copy leaves the original in place" (Test-Path -LiteralPath $spaced)
    } else {
        Check "a non-temporary path is the user's own file under another name" (
            Test-SameUnderlyingFile $r2.Path $spaced)
        Check "the user's file is still there and untouched" (
            (Test-Path -LiteralPath $spaced) -and ((Get-Content -Raw -LiteralPath $spaced).Trim() -eq "typer"))
    }

    # Every candidate unusable: the original comes back with a warning.
    $script:Warnings = @()
    # .NET reads TMPDIR for GetTempPath off Windows, so it must be overridden too.
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $env:TEMP = $spacedDir
        $env:TMP = $spacedDir
        $env:TMPDIR = $spacedDir
        $r3 = Get-UvSafeRequirementsPath -Path $spaced
        if ($r3.Path.Contains(" ")) {
            Check "the original is returned when nowhere space-free is usable" ($r3.Path -eq $spaced)
            Check "the give-up path is not marked temporary" (-not $r3.Temporary)
            Check "the give-up path warns the user" (($script:Warnings -join " ") -match "space")
            Check "the warning names the variables to change" (($script:Warnings -join " ") -match "TMP")
        } else {
            Check "a space-free path was still found, which satisfies the contract" $true
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # A failed copy returns the original as not temporary. Copy-Item fails before creating
    # anything here, so the catch's cleanup of a partial copy is not exercised.
    $script:Warnings = @()
    $blocker = Join-Path $root "blocker"
    "not a directory" | Set-Content -LiteralPath $blocker
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $env:TEMP = $blocker
        $env:TMP = $blocker
        $env:TMPDIR = $blocker
        $before = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "unsloth-reqs-*").Count
        $r4 = Get-UvSafeRequirementsPath -Path $spaced
        $after = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "unsloth-reqs-*").Count
        Check "a failed copy leaves no stray unsloth-reqs file" ($after -eq $before)
        Check "a failed copy is never reported as temporary" (-not $r4.Temporary)
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # A relative temp path makes Join-Path throw, which under "Stop" would abort the install.
    $script:Warnings = @()
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $ErrorActionPreference = "Stop"
        $env:TEMP = "relative-temp-dir"
        $env:TMP = "relative-temp-dir"
        $env:TMPDIR = "relative-temp-dir"
        $threw = $false
        try { $r5 = Get-UvSafeRequirementsPath -Path $spaced } catch { $threw = $true }
        Check "a relative temp path does not abort the helper" (-not $threw)
        if (-not $threw) {
            Check "a relative temp path still yields a usable result" ($null -ne $r5.Path)
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # Join-Path throws DriveNotFoundException for an unmapped drive; that must not abort either.
    $script:Warnings = @()
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $ErrorActionPreference = "Stop"
        $env:TEMP = "Z:\stale"
        $env:TMP = "Z:\stale"
        $env:TMPDIR = "Z:\stale"
        $threw = $false
        try { $r6 = Get-UvSafeRequirementsPath -Path $spaced } catch { $threw = $true }
        Check "an unmapped temp drive does not abort the helper" (-not $threw)
        if (-not $threw) {
            Check "an unmapped temp drive still yields a result" ($null -ne $r6.Path)
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All Get-UvSafeRequirementsPath checks passed"
exit 0
