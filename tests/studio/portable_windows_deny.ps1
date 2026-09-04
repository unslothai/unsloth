# Regression test: install.ps1 must refuse UNSLOTH_HOME / UNSLOTH_PORTABLE and
# --portable / --root. The backend honours UNSLOTH_HOME everywhere, so accepting
# it would install to %USERPROFILE%\.unsloth\studio while Studio resolved
# <root>\studio.
#
# UNSLOTH_PORTABLE is three-way, and the second half of this file pins all three
# outcomes plus the fact that install.sh reads the same two allowlists. --portable,
# --root and UNSLOTH_HOME stay two-way on purpose: a flag is a request by being
# present, and UNSLOTH_HOME is a path, so it has no off spelling to confuse with a
# typo -- UNSLOTH_HOME=off names a directory called off.
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

$envMatch = [regex]::Match($src, '(?ms)    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_HOME\)\).*?\n    \}\r?\n(?:    #[^\r\n]*\r?\n)*    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_PORTABLE\).*?\n    \}')
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

# ── UNSLOTH_PORTABLE: the real guard out of install.ps1, run as three-way.
# Extracted, never re-implemented: a copy of the condition passes whatever the copy
# says, not whatever the installer does.
#
# Three outcomes, because there are three classes of value and the bug was collapsing
# two of them. An on-value is a real portable request and stays a hard failure, since
# portable mode is POSIX-only. An off-value, empty or unset is no request at all and
# installs normally. Anything else is refused as a bad value the way install.sh refuses
# it -- not denied as a portable request, which is what `UNSLOTH_PORTABLE=flase` used to
# get: an install aborted over a mode the user never asked for.
$guardMatch = [regex]::Match($src, '(?ms)^    if \(-not \[string\]::IsNullOrWhiteSpace\(\$env:UNSLOTH_PORTABLE\)\) \{.*?^    \}')
Check "UNSLOTH_PORTABLE guard extracted from install.ps1" $guardMatch.Success
if (-not $guardMatch.Success) { exit 1 }
$guard = $guardMatch.Value
# Self-validate the extraction: both allowlists must be inside what we are about to run,
# or a refactor would leave this section quietly exercising a guard that no longer has them.
Check "extracted guard carries the on-list"  ($guard -match '\.Trim\(\) -in @\(')    $guard
Check "extracted guard carries the off-list" ($guard -match '\.Trim\(\) -notin @\(') $guard

$acceptCases = @('UNSET', '', '   ', '0', 'false', 'off', 'no', 'FALSE', 'Off', 'NO', ' no ', ' 0 ')
$denyCases   = @('1', 'true', 'yes', 'on', 'True', 'TRUE', 'Yes', ' ON ', ' true ')
$refuseCases = @('enabled', 'flase', '2', 'bogus', 'n', 'disabled', 'ENABLED', ' enabled ', 'true false', '-1', 'yes please')
$allCases = @($acceptCases) + @($denyCases) + @($refuseCases)
$caseLiteral = (($allCases | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ', ')

$guardHarness = @'
function Write-StudioLine { param([string]$Message, $ForegroundColor) Write-Host "MSG: $Message" }
function Deny-PortableMode { param([string]$Which) return "$Which is not supported on Windows yet." }
function Exit-InstallFailure { param([string]$Message) return "FAILED: $Message" }
function Invoke-Guard {
'@ + "`n" + $guard + @'

    return "ACCEPTED"
}
function Get-Verdict {
    param([string]$Result)
    if ($Result -ceq 'ACCEPTED') { return 'ACCEPTED' }
    if ($Result -like '*is not supported on Windows yet.') { return 'DENY-PORTABLE' }
    if ($Result -like '*is not a recognized value.') { return 'REFUSE-VALUE' }
    return "OTHER"
}
'@ + "`n" + '$cases = @(' + $caseLiteral + ')' + @'

for ($i = 0; $i -lt $cases.Count; $i++) {
    if ($cases[$i] -ceq 'UNSET') { Remove-Item Env:\UNSLOTH_PORTABLE -ErrorAction SilentlyContinue }
    else { $env:UNSLOTH_PORTABLE = $cases[$i] }
    Write-Output ("CASE|{0}|{1}" -f $i, (Get-Verdict (Invoke-Guard)))
}
# The refusal a user actually reads. Printed once, after the table.
$env:UNSLOTH_PORTABLE = 'flase'
$null = Invoke-Guard
'@
$guardOut = (pwsh -NoProfile -Command $guardHarness 2>&1 | Out-String)
$verdicts = @{}
foreach ($line in ($guardOut -split "`r?`n")) {
    $vm = [regex]::Match($line, '^CASE\|(\d+)\|(\S+)$')
    if ($vm.Success) { $verdicts[[int]$vm.Groups[1].Value] = $vm.Groups[2].Value }
}
Check "every UNSLOTH_PORTABLE case produced a verdict" ($verdicts.Count -eq $allCases.Count) $guardOut

function Expect-Verdict($index, $value, $want) {
    $got = if ($verdicts.ContainsKey($index)) { $verdicts[$index] } else { "<none>" }
    Check "UNSLOTH_PORTABLE='$value' -> $want" ($got -ceq $want) "got $got"
}
$idx = 0
foreach ($v in $acceptCases) { Expect-Verdict $idx $v 'ACCEPTED';      $idx++ }
foreach ($v in $denyCases)   { Expect-Verdict $idx $v 'DENY-PORTABLE'; $idx++ }
foreach ($v in $refuseCases) { Expect-Verdict $idx $v 'REFUSE-VALUE';  $idx++ }

# The message has to name the variable and both spellings, or a piped install cannot be
# fixed from what scrolled past, and it must not repeat the portable-mode denial.
Check "refusal quotes the offending value"   ($guardOut -match "MSG: ERROR: UNSLOTH_PORTABLE='flase' is not a recognized value\.") $guardOut
Check "refusal names the off spellings"      ($guardOut -match 'MSG:\s+Use 0, false, off, no') $guardOut
Check "refusal names the on spellings"       ($guardOut -match 'MSG:.*1, true, yes, on') $guardOut
Check "refusal offers the Windows knob"      ($guardOut -match 'MSG:.*UNSLOTH_STUDIO_HOME') $guardOut

# ── Cross-file: install.ps1 and install.sh read the same two allowlists.
# The mirror image of tests/sh/test_install_portable_env_value_contract.sh, so neither
# file can drift on its own. Both lists are compared, not just the off-list: the on-list
# is what decides that a value is a portable request at all.
$shPath = Join-Path $PSScriptRoot "../../install.sh"
$sh = Get-Content -Raw $shPath
$shCase = [regex]::Match($sh, '(?ms)^case "\$\(_trim_ws "\$\{UNSLOTH_PORTABLE:-\}".*?^esac')
Check "install.sh UNSLOTH_PORTABLE case found" $shCase.Success
if (-not $shCase.Success) { exit 1 }
Check "install.sh refuses unrecognized values" ($shCase.Value -match 'is not a recognized value') $shCase.Value

function Get-SortedList($csv, $sep) {
    return (($csv -split $sep | ForEach-Object { $_.Trim().Trim('"').Trim("'") } |
        Where-Object { $_ -ne "" } | Sort-Object) -join ",")
}
$ps1On  = Get-SortedList ([regex]::Match($guard, '\.Trim\(\) -in @\(([^)]*)\)').Groups[1].Value) ','
$ps1Off = Get-SortedList ([regex]::Match($guard, '\.Trim\(\) -notin @\(([^)]*)\)').Groups[1].Value) ','
$shOn   = Get-SortedList ([regex]::Match($shCase.Value, '(?m)^\s*([^\s()]+)\)\s*_PORTABLE_MODE=true').Groups[1].Value) '\|'
$shOff  = Get-SortedList ([regex]::Match($shCase.Value, "(?m)^\s*(''\|[^\s()]+)\)\s*;;").Groups[1].Value) '\|'

# Pin the literals too: two files that drifted to the same wrong list would still agree.
Check "install.ps1 on-list is 1/true/yes/on"    ($ps1On  -ceq "1,on,true,yes")     $ps1On
Check "install.ps1 off-list is 0/false/off/no"  ($ps1Off -ceq "0,false,no,off")    $ps1Off
Check "install.sh on-list matches install.ps1"  ($shOn   -ceq $ps1On)  "sh=$shOn ps1=$ps1On"
Check "install.sh off-list matches install.ps1" ($shOff  -ceq $ps1Off) "sh=$shOff ps1=$ps1Off"
# Overlap would make the order of the two ifs decide the answer.
Check "the two lists are disjoint" (-not (@($ps1On -split ',') | Where-Object { $_ -cin @($ps1Off -split ',') })) "$ps1On / $ps1Off"

$env:UNSLOTH_PORTABLE = $null

Write-Host ""
if ($fails -gt 0) { Write-Host "$fails check(s) failed"; exit 1 }
Write-Host "All PowerShell deny checks passed"
