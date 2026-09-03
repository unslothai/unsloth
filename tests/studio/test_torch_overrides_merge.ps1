#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for install.ps1's New-UnslothTorchOverridesFile, the Windows twin of
# install.sh's _build_unsloth_torch_overrides. It folds a caller-supplied UV_OVERRIDE
# file into the frozen torch-trio pins, so it must not corrupt that caller's content:
#   - non-ASCII requirement lines survive (an ascii-encoded rewrite turns them into "?")
#   - the merged file stays beside the caller's override file, because uv resolves
#     `-r nested.txt` inside an override file relative to THAT file's own directory
# AST-extracted and run in-process -- no venv, no network, no uv needed.
# Run: pwsh -NoProfile -File tests/studio/test_torch_overrides_merge.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

# --- Parse install.ps1 (also a syntax gate) and extract the helper ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "New-UnslothTorchOverridesFile"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one New-UnslothTorchOverridesFile in install.ps1, found $($fn.Count)" }
Invoke-Expression $fn[0].Extent.Text

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# The helper reads $SkipTorch from its enclosing scope.
$SkipTorch = $false

# A stand-in interpreter: the helper only ever calls it as `& $PythonExe -c <code>`
# and reads the printed `name==version` lines.
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-ovtest-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work -Force | Out-Null
$fakePy = Join-Path $work "fakepython"
Set-Content -LiteralPath $fakePy -Value @(
    "#!/usr/bin/env bash"
    "printf 'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\ntorchaudio==2.11.0+cu130\n'"
) -Encoding ascii
if ($IsLinux -or $IsMacOS) { & chmod +x $fakePy }

$acute = [char]0x00E9   # e-acute, the cheapest non-ASCII requirement character
$savedOverride = $env:UV_OVERRIDE
$made = @()

try {
    # ---- 1. caller override in its own directory, with a relative include ----
    $callerDir = Join-Path $work "callerdir"
    New-Item -ItemType Directory -Path $callerDir -Force | Out-Null
    $nested = Join-Path $callerDir "nested.txt"
    Set-Content -LiteralPath $nested -Value "idna==3.6" -Encoding ascii
    $callerOv = Join-Path $callerDir "over.txt"
    [System.IO.File]::WriteAllText(
        $callerOv,
        "-r nested.txt`ncaf${acute}pkg==1.0`ntorch==1.0`ntorchvision==0.1`nplainpkg==2.0`n",
        (New-Object System.Text.UTF8Encoding($false)))

    $env:UV_OVERRIDE = $callerOv
    $merged = New-UnslothTorchOverridesFile -PythonExe $fakePy
    $made += $merged
    Check "returns a merged overrides file" ($null -ne $merged -and (Test-Path -LiteralPath $merged))

    $mergedText = [System.IO.File]::ReadAllText($merged)

    # The two regressions under test.
    Check "non-ASCII caller requirement survives the merge (no '?' substitution)" `
        ($mergedText -match "caf${acute}pkg==1\.0" -and $mergedText -notmatch 'caf\?pkg')
    Check "merged file sits where the caller's relative includes still resolve" `
        (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $merged) "nested.txt"))

    # Guards on the behaviour that already worked, so the fix cannot regress it.
    Check "frozen torch trio is pinned first" ($mergedText -match '^torch==2\.11\.0\+cu130')
    Check "caller's torch-trio lines are dropped" `
        ($mergedText -notmatch '(?m)^torch==1\.0$' -and $mergedText -notmatch '(?m)^torchvision==0\.1$')
    Check "caller's relative include line is carried over" ($mergedText -match '(?m)^-r nested\.txt$')
    Check "caller's ordinary requirements are carried over" ($mergedText -match '(?m)^plainpkg==2\.0$')
    Check "merged file is newline terminated" ($mergedText.EndsWith("`n"))

    $bytes = [System.IO.File]::ReadAllBytes($merged)
    Check "merged file has no UTF-8 BOM" `
        (-not ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF))

    # ---- 2. no caller override: nothing relative to preserve, trio only ----
    Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
    $bare = New-UnslothTorchOverridesFile -PythonExe $fakePy
    $made += $bare
    Check "works with no caller override file" ($null -ne $bare -and (Test-Path -LiteralPath $bare))
    if ($bare) {
        $bareText = [System.IO.File]::ReadAllText($bare)
        Check "bare merge carries the whole trio" `
            ($bareText -match 'torch==2\.11\.0' -and $bareText -match 'torchvision==0\.26\.0' -and
             $bareText -match 'torchaudio==2\.11\.0')
    }

    # ---- 3. --no-torch short circuit ----
    $SkipTorch = $true
    Check "returns null under --no-torch" ($null -eq (New-UnslothTorchOverridesFile -PythonExe $fakePy))
    $SkipTorch = $false
}
finally {
    if ($null -eq $savedOverride) { Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue }
    else { $env:UV_OVERRIDE = $savedOverride }
    foreach ($m in $made) { if ($m) { Remove-Item -LiteralPath $m -Force -ErrorAction SilentlyContinue } }
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED ($failures)" -ForegroundColor Red
    exit 1
}
Write-Host "All New-UnslothTorchOverridesFile checks passed" -ForegroundColor Green
exit 0
