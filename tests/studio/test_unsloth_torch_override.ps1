#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Windows twin of tests/sh/test_unsloth_torch_override.sh: install.ps1's torch-trio
# --overrides guard (New-UnslothTorchOverridesFile) on the Step-2 unsloth installs.
# The generated file also folds in the caller's UV_OVERRIDE lines, which can carry
# authenticated direct URLs, so it must never outlive the run: install.sh removes its
# twin from the EXIT/signal traps and install.ps1 must do the same from the outer
# finally. Pure text/AST assertions plus one behavioural cleanup check -- no venv needed.
# Run: pwsh -NoProfile -File tests/studio/test_unsloth_torch_override.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path
$installText = Get-Content -Raw $installPath

# --- Parse install.ps1 (also serves as a syntax gate) ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Text of every Invoke-InstallCommandRetry statement carrying $label. A with-deps
# path has two: the overrides-guarded call and the no-torch-installed fallback.
function Get-InstallBlocks([string]$label) {
    $calls = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst] -and
        $n.Extent.Text -like "*-Label `"$label`" *" -and
        $n.GetCommandName() -eq "Invoke-InstallCommandRetry"
    }, $true)
    if ($calls.Count -eq 0) { throw "no Invoke-InstallCommandRetry found for '$label'" }
    return @($calls | ForEach-Object { $_.Extent.Text })
}

Write-Host "New-UnslothTorchOverridesFile"
$fnAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "New-UnslothTorchOverridesFile"
}, $true)
Check "helper defined exactly once" ($fnAst.Count -eq 1)
$fnText = $fnAst[0].Extent.Text
Check "helper returns null under --no-torch" ($fnText -match 'if \(\$SkipTorch\) \{ return \$null \}')
Check "helper folds in caller UV_OVERRIDE files" ($fnText -match '\$env:UV_OVERRIDE')

Write-Host "with-deps unsloth installs pass --overrides"
foreach ($label in @("install unsloth (local)", "install unsloth")) {
    $blocks = Get-InstallBlocks $label
    $guarded = @($blocks | Where-Object { $_ -match '--overrides \$script:TorchOverridesFile' })
    Check "'$label' has one overrides-guarded call and one plain fallback" (
        $blocks.Count -eq 2 -and $guarded.Count -eq 1)
}

Write-Host "the --no-deps no-torch installs carry no overrides"
foreach ($label in @("install unsloth (no-torch)", "install unsloth (migrated no-torch)")) {
    $blocks = Get-InstallBlocks $label
    Check "'$label' has no --overrides" (@($blocks | Where-Object { $_ -match '--overrides' }).Count -eq 0)
}

Write-Host "the generated temp file never outlives the run"
$removals = [regex]::Matches($installText, 'Remove-Item -LiteralPath \$script:TorchOverridesFile -Force')
# One in-flow removal per with-deps install, plus the outer-finally sweep.
Check "in-flow removal after each with-deps install, plus a final sweep" ($removals.Count -eq 3)
$outer = @($ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.TryStatementAst] -and
    $n.Body.Extent.Text -match 'Install-UnslothStudio @args'
}, $true))
Check "outer try/finally around Install-UnslothStudio found" ($outer.Count -eq 1)
$finallyText = $outer[0].Finally.Extent.Text
Check "outer finally removes the overrides temp file" ($finallyText -match 'Remove-Item -LiteralPath \$script:TorchOverridesFile -Force')
Check "outer finally still clears UNSLOTH_KEPT_TORCH" ($finallyText -match 'Remove-Item Env:UNSLOTH_KEPT_TORCH')
# install.sh empties _UNSLOTH_TORCH_OVERRIDES before arming its traps so an inherited
# value can never be rm'd; under `irm | iex` the script scope is the caller's session,
# so the same reset must precede the outer try.
Check "overrides path reset to null before the outer try" (
    $installText -match '(?m)^\$script:TorchOverridesFile = \$null\r?\ntry \{\r?\n\s*Install-UnslothStudio @args')

Write-Host "outer finally actually deletes the file after a terminating error"
# Behavioural: run the real finally body with a live temp file holding a credential-
# bearing inherited override line, exactly as an interrupted install would leave it.
$leakFile = [System.IO.Path]::GetTempFileName()
Set-Content -LiteralPath $leakFile -Encoding ascii -Value @(
    "torch==2.11.0+cu128",
    "private-pkg @ https://svc:TOKEN123@pkgs.corp.example/private-1.0-py3-none-any.whl")
$script:TorchOverridesFile = $leakFile
$env:UNSLOTH_KEPT_TORCH = "2.11.0"
# Strip the `finally { ... }` wrapper and run the statements themselves; Invoke-Expression
# evaluates in this scope, so the block's $script: writes land where the installer's would.
$finallyBody = ($finallyText.Trim() -replace '(?s)^\{', '') -replace '(?s)\}$', ''
try {
    try { throw "simulated terminating error mid-install" }
    finally { Invoke-Expression $finallyBody }
} catch { }
Check "temp overrides file removed" (-not (Test-Path -LiteralPath $leakFile))
Check "kept-torch handoff still cleared" ($null -eq $env:UNSLOTH_KEPT_TORCH)
Check "tracked path reset so a rerun cannot re-remove it" ($null -eq $script:TorchOverridesFile)
Remove-Item -LiteralPath $leakFile -Force -ErrorAction SilentlyContinue

Write-Host "the inherited-override filter drops the torch trio in any casing"
# PowerShell's -notmatch is case-insensitive unless written -cnotmatch, so a caller
# override spelled `Torch<2.11` is dropped and the generated exact pin wins.
$filterPattern = $null
if ($fnText -match '\$_ -notmatch ''([^'']+)''') { $filterPattern = $Matches[1] }
Check "filter pattern extracted from the helper" ($null -ne $filterPattern)
$inherited = @(
    "# comment survives",
    "Torch<2.11",
    "TORCHVISION>=0.19",
    "TorchAudio==2.1",
    "torch<2.11.0",
    "torchvision==0.25.0",
    "torchaudio!=2.11.0",
    "torchmetrics==1.0",
    "transformers>=4.57.6",
    "anyio<4.14.0"
)
$merged = @("torch==2.11.0+cu128") + @($inherited | Where-Object { $_ -notmatch $filterPattern })
$trio = @($merged | Where-Object { $_ -match '^(torch|torchvision|torchaudio)([\s<>=!~;@[]|$)' })
Check "exactly one trio requirement survives (the generated pin)" ($trio.Count -eq 1)
Check "the survivor is the generated exact pin" ($trio[0] -eq "torch==2.11.0+cu128")
foreach ($keep in @("torchmetrics==1.0", "transformers>=4.57.6", "anyio<4.14.0", "# comment survives")) {
    Check "unrelated inherited override preserved: $keep" ($merged -contains $keep)
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures check(s)" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed."
