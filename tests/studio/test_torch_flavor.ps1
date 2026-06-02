#!/usr/bin/env pwsh
# Unit test for install.ps1's torch-flavor helpers (ConvertTo-TorchFlavorTag,
# Get-ExpectedTorchFlavorTag) that drive the stale-CPU-PyTorch repair. Pure
# helpers, AST-extracted and run in-process -- no GPU/venv needed.
# Run: pwsh -NoProfile -File tests/studio/test_torch_flavor.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

# --- Parse install.ps1 (also serves as a syntax gate) and extract the helpers ---
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

foreach ($name in @("ConvertTo-TorchFlavorTag", "Get-ExpectedTorchFlavorTag")) {
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in install.ps1, found $($fn.Count)" }
    # Pure helpers (no exit / external calls) -- safe to define in this scope.
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

Write-Host "ConvertTo-TorchFlavorTag"
Check "2.10.0+cu130 -> cu130"        ((ConvertTo-TorchFlavorTag "2.10.0+cu130") -eq "cu130")
Check "2.8.0+cu128 -> cu128"         ((ConvertTo-TorchFlavorTag "2.8.0+cu128")  -eq "cu128")
Check "2.10.0+cpu -> cpu"            ((ConvertTo-TorchFlavorTag "2.10.0+cpu")    -eq "cpu")
Check "2.10.0 (untagged) -> cpu"     ((ConvertTo-TorchFlavorTag "2.10.0")        -eq "cpu")
Check "2.11.0+rocm7.1 -> rocm"       ((ConvertTo-TorchFlavorTag "2.11.0+rocm7.1") -eq "rocm")
Check "empty -> null"                ($null -eq (ConvertTo-TorchFlavorTag ""))

Write-Host "Get-ExpectedTorchFlavorTag"
Check "cu130 index -> cu130"         ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://download.pytorch.org/whl/cu130") -eq "cu130")
Check "trailing slash -> cu130"      ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://download.pytorch.org/whl/cu130/") -eq "cu130")
Check "cpu index -> cpu"             ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://download.pytorch.org/whl/cpu") -eq "cpu")
Check "ROCm url -> rocm"             ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://download.pytorch.org/whl/cpu" -ROCmIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all/") -eq "rocm")
Check "gfx index leaf -> rocm"        ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://repo.amd.com/rocm/whl/gfx120X-all/") -eq "rocm")
Check "rocm7.2 leaf -> rocm"          ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://download.pytorch.org/whl/rocm7.2") -eq "rocm")
Check "mirror cu130 leaf -> cu130"   ((Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://my.mirror/whl/cu130") -eq "cu130")
Check "unrecognized leaf -> null"    ($null -eq (Get-ExpectedTorchFlavorTag -TorchIndexUrl "https://my.mirror/whl/simple"))
Check "empty url -> null"            ($null -eq (Get-ExpectedTorchFlavorTag -TorchIndexUrl ""))

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
