#!/usr/bin/env pwsh
# Unit test for setup.ps1's Get-NodeDecision (the isolated-Node source picker:
# system | bundled | skip). Pure helper, AST-extracted and run in-process -- no
# Node/npm/network needed. Also serves as a setup.ps1 parse/syntax gate.
# Run: pwsh -NoProfile -File tests/studio/test_node_decision.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }

$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Get-NodeDecision"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one Get-NodeDecision in setup.ps1, found $($fn.Count)" }
Invoke-Expression $fn[0].Extent.Text

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function D($node, $npm, $skip) { Get-NodeDecision -NodeVersion $node -NpmVersion $npm -SkipInstall $skip }

Write-Host "Get-NodeDecision"
# system
Check "node22 + npm11 -> system"      ((D "v22.17.1" "11.13.0" "0") -eq "system")
Check "node20.19 + npm11 -> system"   ((D "v20.19.0" "11.0.0"  "0") -eq "system")
Check "node24 + npm11 -> system"      ((D "v24.17.0" "11.13.0" "0") -eq "system")
Check "node23 + npm11 -> system"      ((D "v23.5.0"  "11.0.0"  "0") -eq "system")
# bundled (the reported bug: fine Node, stale npm)
Check "node22 + npm10 -> bundled"     ((D "v22.17.1" "10.9.2"  "0") -eq "bundled")
Check "node18 -> bundled"             ((D "v18.20.0" "11.0.0"  "0") -eq "bundled")
Check "node22.11 -> bundled"          ((D "v22.11.0" "11.0.0"  "0") -eq "bundled")
Check "node20.18 -> bundled"          ((D "v20.18.0" "11.0.0"  "0") -eq "bundled")
Check "node21 (odd) -> bundled"       ((D "v21.7.0"  "11.0.0"  "0") -eq "bundled")
Check "missing -> bundled"            ((D "" "" "0") -eq "bundled")
# skip flag
Check "npm10 + skip -> skip"          ((D "v22.17.1" "10.9.2"  "1") -eq "skip")
Check "missing + skip -> skip"        ((D "" "" "1") -eq "skip")
Check "good + skip -> system"         ((D "v22.17.1" "11.13.0" "1") -eq "system")

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
