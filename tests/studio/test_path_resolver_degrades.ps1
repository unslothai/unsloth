
$ErrorActionPreference = "Stop"

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "SKIP: this file describes a host where the lower rungs of the ladder cannot answer, which Windows is not."
    exit 0
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")

$errors = $null; $tokens = $null
$script:InstallAst = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw "$installPath : $($errors[0].Message)" }
function Get-InstallFunctionSource([string]$Name) {
    $found = $script:InstallAst.FindAll({
        $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $args[0].Name -eq $Name
    }, $true)
    if ($found.Count -eq 0) { return $null }
    return $found[0].Extent.Text
}

$failures = 0
function Check {
    param([string]$Name, [bool]$Ok)
    if ($Ok) { Write-Host "  PASS  $Name" -ForegroundColor Green }
    else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

$bannedConstructs = @(
    @("compiles no C# at all", "(?m)^[ \t]*Add-Type\b(?![^\r\n]*-AssemblyName)"),
    @("emits no P/Invoke stubs", "DefinePInvokeMethod"),
    @("defines no dynamic assembly", "DefineDynamicAssembly"),
    @("declares no native import", "DllImport")
)
$bannedTypeNames = @("UnslothStudioFinalPathV3", "UnslothStudioProcessImageV1", "UnslothShellIconRefresh", "UnslothStudioConsoleModeV1")
foreach ($pair in @(@("install.ps1", $installPath), @("studio/setup.ps1", $setupPath))) {
    $body = Get-Content -LiteralPath $pair[1] -Raw
    foreach ($ban in $bannedConstructs) { Check "$($pair[0]) $($ban[0])" ([regex]::Matches($body, $ban[1]).Count -eq 0) }
    foreach ($typeName in $bannedTypeNames) {
        Check "$($pair[0]) no longer mentions $typeName" ($body -notmatch [regex]::Escape($typeName))
    }
}

$banSelfTest = @'
Add-Type -TypeDefinition "public class X {}"
$null = [AppDomain]::CurrentDomain.DefineDynamicAssembly($name, "Run")
$null = $builder.DefinePInvokeMethod("CreateFileW", "kernel32.dll")
[DllImport("kernel32.dll")]
class UnslothStudioFinalPathV3 {}
'@
foreach ($ban in $bannedConstructs) {
    Check "the '$($ban[0])' rule can fire at all" ([regex]::Matches($banSelfTest, $ban[1]).Count -gt 0)
}
Check "the deleted-type rule can fire at all" ($banSelfTest -match [regex]::Escape($bannedTypeNames[0]))

$fns = @(
    "Write-StudioFinalPathDegraded", "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Get-StudioEarlyPython", "Invoke-StudioEarlyPythonScript", "Invoke-StudioEarlyPython", "Get-StudioPythonFinalPath",
    "Resolve-StudioFinalPathInfo", "Get-StudioPythonProcessImageTable", "Get-StudioProcessImagePath"
)
$src = @()
foreach ($fn in $fns) {
    $found = Get-InstallFunctionSource $fn
    Check "install.ps1 defines $fn" ($null -ne $found)
    if ($found) { $src += $found }
}

if ($src.Count -eq $fns.Count) {
    # Each step prints TAG_THREW plus TAG_* facts; the parent matches them below.
    $harness = @"
`$ErrorActionPreference = "Stop"
function Write-StudioLine { param([string]`$Message, `$ForegroundColor) Write-Host "LINE: `$Message" }
`$script:StudioFinalPathWarned = `$false
`$script:StudioSubstMap = `$null
`$script:StudioEarlyPythonProbed = `$false
`$script:StudioEarlyPython = `$null
`$script:StudioPythonProcessImageProbed = `$false
`$script:StudioPythonProcessImageTable = `$null
`$script:StudioProcessImageTable = `$null
$($src -join "`n")
function Try-Step([string]`$Tag, [scriptblock]`$Body) {
    `$r = `$null; `$threw = `$false
    try { `$r = & `$Body } catch { `$threw = `$true; Write-Host "`${Tag}_ERROR: `$(`$_.Exception.Message)" }
    Write-Host "`${Tag}_THREW: `$threw"
    return `$r
}
`$temp = [System.IO.Path]::GetTempPath().TrimEnd('/', '\')

# 1. The interpreter rung, which this host CAN reach: pathlib resolve is the same system call the
#    deleted native rung made, so Exact = `$true here is earned rather than assumed.
`$py = Try-Step PY { Get-StudioPythonFinalPath -Path `$temp }
Write-Host "PY_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$py))"
Write-Host "PY_ABSOLUTE: `$(if (`$py) { [bool](Split-Path -IsAbsolute `$py) } else { `$false })"

# 2. The full ladder with an interpreter: exact, and no degradation warning.
`$exactInfo = Try-Step EXACT { Resolve-StudioFinalPathInfo -Path `$temp }
if (`$exactInfo) {
    Write-Host "EXACT_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$exactInfo.Path))"
    Write-Host "EXACT_FLAG: `$(`$exactInfo.Exact)"
}
Write-Host "EXACT_WARNED: `$script:StudioFinalPathWarned"

# 3. A not-yet-created tail (`$StudioHome before it exists): the prefix resolves exactly, the rest is appended.
`$missing = Join-Path `$temp ("uns_missing_" + [guid]::NewGuid().ToString("N"))
`$missingInfo = Try-Step MISSING { Resolve-StudioFinalPathInfo -Path `$missing }
if (`$missingInfo) {
    Write-Host "MISSING_KEEPS_LEAF: `$(`$missingInfo.Path.EndsWith((Split-Path -Leaf `$missing)))"
    Write-Host "MISSING_EXACT: `$(`$missingInfo.Exact)"
}

# 4. The Python-less host, via the kill switch. The latch and the per-path cache are cleared too,
#    or step 1's answer is replayed and this checks nothing.
`$env:UNSLOTH_EARLY_PYTHON_PROBE = "0"
`$script:StudioEarlyPythonProbed = `$false
`$script:StudioEarlyPython = `$null
`$script:StudioPythonFinalPathCache = `$null
`$script:StudioFinalPathWarned = `$false
Write-Host "KILLSWITCH_DECLINES: `$(`$null -eq (Get-StudioEarlyPython))"
`$info = Try-Step LADDER { Resolve-StudioFinalPathInfo -Path `$temp }
if (`$info) {
    Write-Host "LADDER_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$info.Path))"
    Write-Host "LADDER_EXACT: `$(`$info.Exact)"
}
Write-Host "LADDER_WARNED: `$script:StudioFinalPathWarned"
Write-Host "LADDER_REPEAT_EXACT: `$((Resolve-StudioFinalPathInfo -Path `$temp).Exact)"

# 5. The process-image ladder: Get-Process answers our own image, so no lower rung is paid for.
`$img = Try-Step PROCIMG { Get-StudioProcessImagePath -ProcessId `$PID }
Write-Host "PROCIMG_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$img))"
Write-Host "PROCIMG_STAYED_ON_TOP_RUNG: `$((-not `$script:StudioPythonProcessImageProbed) -and (`$null -eq `$script:StudioProcessImageTable))"

# 6. The rungs below Get-Process, reached by making it answer nothing. On this engine the ctypes rung
#    declines and Win32_Process does not exist, so the answer must be null, probed once and counted.
function Get-Process { param(`$Id, `$ErrorAction) return `$null }
`$deep = Try-Step PROCIMG_DEEP { Get-StudioProcessImagePath -ProcessId `$PID }
Write-Host "PROCIMG_DEEP_NULL: `$([string]::IsNullOrWhiteSpace(`$deep))"
Write-Host "PROCIMG_PROBED_ONCE: `$script:StudioPythonProcessImageProbed"
`$script:ProbeCallCount = 0
function Get-StudioPythonProcessImageTable { `$script:ProbeCallCount++; return `$null }
`$null = Get-StudioProcessImagePath -ProcessId `$PID
Write-Host "PROCIMG_NOT_REPROBED: `$(`$script:ProbeCallCount -eq 0)"
Write-Host "PROCIMG_TABLE_STILL_NULL: `$(`$null -eq `$script:StudioPythonProcessImageTable)"
"@
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_resolver_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $file -Value $harness -Encoding utf8
    $exe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = (Get-Command powershell).Source }
    try { $out = & $exe -NoProfile -File $file 2>&1 | Out-String }
    finally { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }

    if ($out -notmatch "PY_THREW") { Write-Host "  ---- harness output ----"; Write-Host $out }

    foreach ($row in @(
        @("PY_THREW: False", "the interpreter rung does not throw"),
        @("PY_ANSWERED: True", "the interpreter rung answers"),
        @("PY_ABSOLUTE: True", "and answers with an absolute path"),
        @("EXACT_THREW: False", "the full ladder does not throw with an interpreter present"),
        @("EXACT_ANSWERED: True", "the full ladder answers"),
        @("EXACT_FLAG: True", "and reports the answer as exact"),
        @("EXACT_WARNED: False", "an exact answer prints no degradation warning"),
        @("MISSING_THREW: False", "a not-yet-created path does not throw"),
        @("MISSING_KEEPS_LEAF: True", "and keeps the segments that do not exist yet"),
        @("MISSING_EXACT: True", "and is still exact, since its existing prefix resolved"),
        @("KILLSWITCH_DECLINES: True", "the kill switch really removes the interpreter"),
        @("LADDER_THREW: False", "the ladder without an interpreter does not throw"),
        @("LADDER_ANSWERED: True", "the ladder without an interpreter still answers"),
        @("LADDER_EXACT: False", "and reports the answer as inexact"),
        @("LADDER_WARNED: True", "and says out loud that it is degraded"),
        # The warning latches; the degraded VERDICT must not.
        @("LADDER_REPEAT_EXACT: False", "a second call is still inexact")
    )) { Check $row[1] ($out -match $row[0]) }
    Check "the degradation warning is printed once" (
        ([regex]::Matches($out, "LINE: \[WARN\] Could not resolve a path exactly")).Count -eq 1)
    foreach ($row in @(
        @("PROCIMG_THREW: False", "the process image ladder does not throw"),
        @("PROCIMG_ANSWERED: True", "the process image ladder answers our own image"),
        @("PROCIMG_STAYED_ON_TOP_RUNG: True", "and pays for no rung below the one that answered"),
        @("PROCIMG_DEEP_THREW: False", "the rungs below Get-Process do not throw either"),
        @("PROCIMG_DEEP_NULL: True", "an unanswerable process reads as unknown, not as a path"),
        @("PROCIMG_PROBED_ONCE: True", "the child interpreter is asked once per run"),
        # Counted, not inferred from the latch: a ladder that probes unconditionally also sets it.
        @("PROCIMG_NOT_REPROBED: True", "and is not asked again on the next process"),
        @("PROCIMG_TABLE_STILL_NULL: True", "and an empty result is not re-probed")
    )) { Check $row[1] ($out -match $row[0]) }
}

$probeFn = Get-InstallFunctionSource "Test-StudioSameDirectoryByProbe"
Check "install.ps1 defines Test-StudioSameDirectoryByProbe" ($null -ne $probeFn)
if ($probeFn) {
    Invoke-Expression $probeFn
    $probeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-probe-" + [guid]::NewGuid().ToString("N"))
    $realDir = Join-Path $probeRoot "real"
    $otherDir = Join-Path $probeRoot "other"
    $absent = Join-Path $probeRoot "absent"
    $null = New-Item -ItemType Directory -Path $realDir -Force
    $null = New-Item -ItemType Directory -Path $otherDir -Force
    $aliasDir = Join-Path $probeRoot "alias"
    $madeAlias = $false
    try { $null = New-Item -ItemType SymbolicLink -Path $aliasDir -Target $realDir -ErrorAction Stop; $madeAlias = $true } catch { }
    $rows = @(
        @("two different directories are not one", $realDir, $otherDir, $false),
        @("a directory is itself", $realDir, $realDir, $true)
    )
    if ($madeAlias) { $rows += , @("two spellings that reach one directory are one", $aliasDir, $realDir, $true) }
    else { Write-Host "  SKIP  this host cannot create a link to test the aliased spelling" }
    # A missing side cannot be an alias of one that does exist, and must be answered without a write.
    $rows += @(
        @("a missing right side is not a match", $realDir, $absent, $false),
        @("a missing left side is not a match", $absent, $realDir, $false),
        @("an empty spelling is not a match", "", $realDir, $false)
    )
    foreach ($r in $rows) { Check $r[0] ((Test-StudioSameDirectoryByProbe -Left $r[1] -Right $r[2]) -eq $r[3]) }
    Check "the probe file is removed again" (
        @(Get-ChildItem -LiteralPath $realDir -Force -ErrorAction SilentlyContinue).Count -eq 0)
    Remove-Item -LiteralPath $probeRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$ancestorFn = Get-InstallFunctionSource "Split-StudioExistingAncestor"
Check "install.ps1 defines Split-StudioExistingAncestor" ($null -ne $ancestorFn)
if ($ancestorFn) {
    Invoke-Expression $ancestorFn
    $ancRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-anc-" + [guid]::NewGuid().ToString("N"))
    $null = New-Item -ItemType Directory -Path $ancRoot -Force
    $tail = Join-Path "not-there" "either"
    $split = Split-StudioExistingAncestor -Path (Join-Path $ancRoot $tail)
    Check "the deepest existing ancestor is found" ($split -and $split.Root -eq $ancRoot)
    Check "and the part that does not exist is kept whole" ($split -and $split.Tail -eq $tail)
    $here = Split-StudioExistingAncestor -Path $ancRoot
    Check "a path that exists is its own ancestor, with nothing below it" (
        $here -and $here.Root -eq $ancRoot -and $here.Tail -eq "")
    Check "a path with no existing ancestor at all answers nothing" (
        $null -eq (Split-StudioExistingAncestor -Path (Join-Path "/nonexistent-root-xyz" "a")))
    Check "an empty path answers nothing" ($null -eq (Split-StudioExistingAncestor -Path ""))
    Remove-Item -LiteralPath $ancRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$installText = [System.IO.File]::ReadAllText($installPath)
Check "the --tauri gate falls back to comparing existing ancestors" (
    $installText -match 'Split-StudioExistingAncestor -Path \$_tauriOverride')
Check "and requires the segments below them to match" ($installText -match '\$_tauriLeft\.Tail, \$_tauriRight\.Tail')
Check "the --tauri gate asks it before refusing an override" (
    $installText -match 'Test-StudioSameDirectoryByProbe -Left \$_tauriOverride -Right \$_legacyTauriRoot')

$selfText = Get-Content -LiteralPath $PSCommandPath -Raw
$afterGate = $selfText.Substring($selfText.IndexOf('if ($fail' + 'ures -gt 0) {'))
Check "no check is written after the failure gate" ($afterGate -notmatch "(?m)^\s*Check ")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
