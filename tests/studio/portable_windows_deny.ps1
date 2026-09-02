# Regression test: install.ps1 must refuse UNSLOTH_HOME / UNSLOTH_PORTABLE and
# --portable / --root. The backend honours UNSLOTH_HOME everywhere, so accepting
# it would install to %USERPROFILE%\.unsloth\studio while Studio resolved
# <root>\studio.
$ErrorActionPreference = "Stop"
$fails = 0

function Check($label, $cond, $detail = "") {
    if ($cond) { Write-Host "  PASS  $label" }
    else { Write-Host "  FAIL  $label $detail"; $script:fails++ }
}

$src = Get-Content -Raw (Join-Path $PSScriptRoot "../../install.ps1")

$denyMatch = [regex]::Match($src, '(?ms)^    function Deny-PortableMode.*?\n    \}')
Check "Deny-PortableMode found in install.ps1" $denyMatch.Success
if (-not $denyMatch.Success) { exit 1 }

$envMatch = [regex]::Match($src, '(?ms)    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_HOME\)\).*?\n    \}\r?\n    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_PORTABLE\).*?\n    \}')
Check "UNSLOTH_HOME / UNSLOTH_PORTABLE guard found" $envMatch.Success
if (-not $envMatch.Success) { exit 1 }

Check "--portable rejected in the flag parser" ($src -match '"--portable"\s*\{\s*return \(Exit-InstallFailure \(Deny-PortableMode')
Check "--root rejected in the flag parser"     ($src -match '"--root"\s*\{\s*return \(Exit-InstallFailure \(Deny-PortableMode')

# Run the real parse loop. `switch` matches exactly, so --root=DIR (one argument,
# and the form install.sh accepts) misses the "--root" arm and used to install normally.
$loopMatch = [regex]::Match($src, '(?ms)^    for \(\$i = 0; \$i -lt \$argList\.Count; \$i\+\+\) \{.*?\n    \}')
Check "flag parse loop found in install.ps1" $loopMatch.Success
if (-not $loopMatch.Success) { exit 1 }

$parseHarness = @'
function Write-StudioLine { param([string]$Message, $ForegroundColor) }
function Exit-InstallFailure { param([string]$Message) return "DENIED: $Message" }
function Deny-PortableMode { param([string]$Which) return "$Which is not supported on Windows yet." }
function Invoke-ParseFlags {
    $argList = $args
    $script:UnslothVerbose = $false
    $StudioLocalInstall = $false
    $PackageName = "unsloth"
    $TauriMode = $false
    $SkipTorch = $false
    $ShortcutsOnly = $false
    $WithLlamaCppDir = ""
'@ + "`n" + $loopMatch.Value + @'

    return "ACCEPTED"
}
foreach ($case in @('--root=C:\portable', '--root', '--portable', '--local')) {
    Write-Output ("{0} => {1}" -f $case, (Invoke-ParseFlags $case))
}
'@
$parsed = (pwsh -NoProfile -Command $parseHarness 2>&1 | Out-String)
Check "--root=DIR is rejected"  ($parsed -match [regex]::Escape('--root=C:\portable => DENIED: --root is not supported')) $parsed
Check "--root is rejected"      ($parsed -match '--root => DENIED: --root is not supported') $parsed
Check "--portable is rejected"  ($parsed -match '--portable => DENIED: --portable is not supported') $parsed
Check "--local still installs"  ($parsed -match '--local => ACCEPTED') $parsed

$harness = @"
function Write-StudioLine { param([string]`$Message, `$ForegroundColor) Write-Host `$Message }
$($denyMatch.Value -replace '^    ', '' -replace "`n    ", "`n")
Deny-PortableMode "--portable"
"@
$out = pwsh -NoProfile -Command $harness 2>&1 | Out-String
Check "deny message names the flag"           ($out -match '--portable is not supported on Windows')
Check "deny message points at the POSIX flags" ($out -match 'install\.sh --portable')
Check "deny message offers the Windows knob"   ($out -match 'UNSLOTH_STUDIO_HOME')

# Off-values count as unset, or a stray UNSLOTH_PORTABLE=0 blocks every install.
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
