#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for setup.ps1's torch-index pin-hardening helpers: Trim-IndexPathSlashes
# (path-only slash trim, token-preserving), Redact-InstallOutput (credential redaction of
# captured install logs), Get-TorchIndexLeaf (ALL trailing slashes stripped) and
# Test-PipRocmFamilyLeaf (rocm7. is a custom pin, not a family). Pure helpers, AST-extracted
# and run in-process. Run: pwsh -NoProfile -File tests/studio/test_torch_index_pin_hardening.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path
$setupText = Get-Content -Raw $setupPath

# --- Parse setup.ps1 (also a syntax gate) and extract the pure helpers ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }

foreach ($name in @("Trim-IndexPathSlashes", "Redact-InstallOutput", "Get-TorchIndexLeaf", "Test-PipRocmFamilyLeaf")) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in setup.ps1, found $($fn.Count)" }
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

Write-Host "Trim-IndexPathSlashes (path-only, token-preserving)"
Check "double path slash collapsed"      ((Trim-IndexPathSlashes "https://h/whl/cu128//") -eq "https://h/whl/cu128")
Check "single trailing slash trimmed"    ((Trim-IndexPathSlashes "https://h/whl/cu128/") -eq "https://h/whl/cu128")
Check "no slash unchanged"               ((Trim-IndexPathSlashes "https://h/whl/cu128") -eq "https://h/whl/cu128")
Check "query token slash preserved"      ((Trim-IndexPathSlashes "https://h/whl/cu128?token=ab12cd/") -eq "https://h/whl/cu128?token=ab12cd/")
Check "path slash trimmed, query kept"   ((Trim-IndexPathSlashes "https://h/whl/cu128//?token=ab12cd/") -eq "https://h/whl/cu128?token=ab12cd/")
Check "fragment slash preserved"         ((Trim-IndexPathSlashes "https://h/whl/cu128#anchor/") -eq "https://h/whl/cu128#anchor/")

Write-Host "Redact-InstallOutput (credential redaction)"
Check "userinfo redacted"                ((Redact-InstallOutput "ERROR https://alice:s3cr3t@download.pytorch.org/whl/cu128") -eq "ERROR https://<redacted>@download.pytorch.org/whl/cu128")
Check "bare-token@ redacted"             ((Redact-InstallOutput "fetch https://ghp_deadbeef@host/whl/cu128 failed") -eq "fetch https://<redacted>@host/whl/cu128 failed")
Check "single query value redacted"      ((Redact-InstallOutput "url https://host/whl/cu128?token=abcd1234 unreachable") -eq "url https://host/whl/cu128?token=<redacted> unreachable")
Check "multiple query values redacted"   ((Redact-InstallOutput "https://host/whl/cu128?token=abcd1234&channel=beta") -eq "https://host/whl/cu128?token=<redacted>&channel=<redacted>")
Check "fragment token redacted"          ((Redact-InstallOutput "ERROR https://mirror.local/whl/cu128#token=SECRET123 (403)") -eq "ERROR https://mirror.local/whl/cu128#<redacted> (403)")
Check "query and fragment both redacted" ((Redact-InstallOutput "https://host/whl/cu128?token=abc#sig=xyz done") -eq "https://host/whl/cu128?token=<redacted>#<redacted> done")
Check "bare hash comment untouched"      ((Redact-InstallOutput "# retrying with --no-cache-dir") -eq "# retrying with --no-cache-dir")
Check "plain line untouched"             ((Redact-InstallOutput "Resolved 42 packages in 1.2s") -eq "Resolved 42 packages in 1.2s")
$leak = Redact-InstallOutput "https://alice:s3cr3t@host/whl/cu128?token=SUPERSECRET#frag=ALSOSECRET"
Check "no secret substring survives"     (($leak -notmatch "s3cr3t") -and ($leak -notmatch "SUPERSECRET") -and ($leak -notmatch "ALSOSECRET"))

Write-Host "Get-TorchIndexLeaf (ALL trailing slashes stripped)"
Check "double slash cu128 -> cu128"      ((Get-TorchIndexLeaf "https://m/whl/cu128//") -eq "cu128")
Check "triple slash rocm7.2 -> rocm7.2"  ((Get-TorchIndexLeaf "https://m/whl/rocm7.2///") -eq "rocm7.2")
Check "double slash + token -> cu128"    ((Get-TorchIndexLeaf "https://m/whl/cu128//?token=x") -eq "cu128")
Check "single slash cu128 -> cu128"      ((Get-TorchIndexLeaf "https://m/whl/cu128/") -eq "cu128")

Write-Host "Test-PipRocmFamilyLeaf (rocm7. is a custom pin, not a family)"
Check "rocm7 family"                     (Test-PipRocmFamilyLeaf "rocm7")
Check "rocm7.2 family"                    (Test-PipRocmFamilyLeaf "rocm7.2")
Check "gfx1151 family"                    (Test-PipRocmFamilyLeaf "gfx1151")
Check "rocm7. trailing-dot NOT family"   (-not (Test-PipRocmFamilyLeaf "rocm7."))
Check "rocm.7 leading-dot NOT family"    (-not (Test-PipRocmFamilyLeaf "rocm.7"))
Check "rocm7.2.1 two-dot NOT family"     (-not (Test-PipRocmFamilyLeaf "rocm7.2.1"))
Check "rocm7.2-private NOT family"       (-not (Test-PipRocmFamilyLeaf "rocm7.2-private"))
Check "cu128 NOT family"                 (-not (Test-PipRocmFamilyLeaf "cu128"))

Write-Host "Fast-Install pinned-install env scrub (source assertion)"
# The pip fallback honours PIP_*; PIP_NO_INDEX=1 would make it ignore the pinned --index-url
# and PIP_INDEX_URL would replace it, so both must be scrubbed for a pinned install.
Check "PIP_NO_INDEX scrubbed"            ($setupText -match "'PIP_NO_INDEX'")
Check "PIP_INDEX_URL scrubbed"           ($setupText -match "'PIP_INDEX_URL'")

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
