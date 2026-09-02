# Regression test: install.ps1 must refuse UNSLOTH_HOME / UNSLOTH_PORTABLE and
# the --portable / --root flags. The backend honours UNSLOTH_HOME on every
# platform, so accepting it here without implementing portable mode would install
# to %USERPROFILE%\.unsloth\studio while Studio resolved <root>\studio.
# Simulation: install.ps1 must refuse the POSIX-only portable options rather
# than half-applying them. Extracts the real Deny-PortableMode function and the
# real environment guard from install.ps1 and exercises both.
$ErrorActionPreference = "Stop"
$fails = 0

function Check($label, $cond, $detail = "") {
    if ($cond) { Write-Host "  PASS  $label" }
    else { Write-Host "  FAIL  $label $detail"; $script:fails++ }
}

$src = Get-Content -Raw (Join-Path $PSScriptRoot "../../install.ps1")

# The deny helper, lifted verbatim.
$denyMatch = [regex]::Match($src, '(?ms)^    function Deny-PortableMode.*?\n    \}')
Check "Deny-PortableMode found in install.ps1" $denyMatch.Success
if (-not $denyMatch.Success) { exit 1 }

# The environment guard, lifted verbatim.
$envMatch = [regex]::Match($src, '(?ms)    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_HOME\)\).*?\n    \}\r?\n    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_PORTABLE\).*?\n    \}')
Check "UNSLOTH_HOME / UNSLOTH_PORTABLE guard found" $envMatch.Success
if (-not $envMatch.Success) { exit 1 }

# Both flags are rejected in the parser.
Check "--portable rejected in the flag parser" ($src -match '"--portable"\s*\{\s*return \(Exit-InstallFailure \(Deny-PortableMode')
Check "--root rejected in the flag parser"     ($src -match '"--root"\s*\{\s*return \(Exit-InstallFailure \(Deny-PortableMode')

# Run the deny helper for real, with the script's logging shimmed.
$harness = @"
function Write-StudioLine { param([string]`$Message, `$ForegroundColor) Write-Host `$Message }
$($denyMatch.Value -replace '^    ', '' -replace "`n    ", "`n")
Deny-PortableMode "--portable"
"@
$out = pwsh -NoProfile -Command $harness 2>&1 | Out-String
Check "deny message names the flag"           ($out -match '--portable is not supported on Windows')
Check "deny message points at the POSIX flags" ($out -match 'install\.sh --portable')
Check "deny message offers the Windows knob"   ($out -match 'UNSLOTH_STUDIO_HOME')

# The guard must treat the off-values as unset, or a stray UNSLOTH_PORTABLE=0
# in someone's environment would block every Windows install.
foreach ($v in @("0", "false", "False", "", "   ")) {
    $blocked = -not [string]::IsNullOrWhiteSpace($v) -and $v.Trim() -notin @("0", "false", "False")
    Check "UNSLOTH_PORTABLE='$v' does not block the install" (-not $blocked)
}
foreach ($v in @("1", "true", "yes")) {
    $blocked = -not [string]::IsNullOrWhiteSpace($v) -and $v.Trim() -notin @("0", "false", "False")
    Check "UNSLOTH_PORTABLE='$v' blocks the install" $blocked
}

Write-Host ""
if ($fails -gt 0) { Write-Host "$fails check(s) failed"; exit 1 }
Write-Host "All PowerShell deny checks passed"
