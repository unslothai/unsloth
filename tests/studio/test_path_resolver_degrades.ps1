
$ErrorActionPreference = "Stop"

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "SKIP: this file describes a host where the lower rungs of the ladder cannot answer, which Windows is not."
    exit 0
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$script:InstallAst = $null
function Get-InstallFunctionSource {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Name)
    if ($null -eq $script:InstallAst) {
        $errors = $null; $tokens = $null
        $script:InstallAst = [System.Management.Automation.Language.Parser]::ParseFile(
            $Path, [ref]$tokens, [ref]$errors)
        if ($errors.Count -gt 0) { throw "$Path : $($errors[0].Message)" }
    }
    $found = $script:InstallAst.FindAll({
        $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $args[0].Name -eq $Name
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

$installPath = [System.IO.Path]::Combine($repoRoot, "install.ps1")
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")

$bannedConstructs = @(
    @("compiles no C# at all", "(?m)^[ \t]*Add-Type\b(?![^\r\n]*-AssemblyName)"),
    @("emits no P/Invoke stubs", "DefinePInvokeMethod"),
    @("defines no dynamic assembly", "DefineDynamicAssembly"),
    @("declares no native import", "DllImport")
)
$bannedTypeNames = @(
    "UnslothStudioFinalPathV3",
    "UnslothStudioProcessImageV1",
    "UnslothShellIconRefresh",
    "UnslothStudioConsoleModeV1"
)
foreach ($pair in @(@("install.ps1", $installPath), @("studio/setup.ps1", $setupPath))) {
    $body = Get-Content -LiteralPath $pair[1] -Raw
    foreach ($ban in $bannedConstructs) {
        $hits = [regex]::Matches($body, $ban[1])
        Check "$($pair[0]) $($ban[0])" ($hits.Count -eq 0)
    }
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
    "Write-StudioFinalPathDegraded",
    "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Get-StudioEarlyPython", "Invoke-StudioEarlyPythonScript",
    "Invoke-StudioEarlyPython", "Get-StudioPythonFinalPath",
    "Resolve-StudioFinalPathInfo",
    "Get-StudioPythonProcessImageTable",
    "Get-StudioProcessImagePath"
)
$src = @()
foreach ($fn in $fns) {
    $found = Get-InstallFunctionSource -Path $installPath -Name $fn
    Check "install.ps1 defines $fn" ($null -ne $found)
    if ($found) { $src += $found }
}

if ($src.Count -eq $fns.Count) {
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

`$temp = [System.IO.Path]::GetTempPath().TrimEnd('/', '\')

# 1. The interpreter rung, which this host CAN reach. It has to answer, and the answer
#    has to be exact: pathlib resolve is the same system call the deleted native rung
#    made, so Exact = `$true here is earned rather than assumed.
`$pyThrew = `$false
`$py = "unset"
try { `$py = Get-StudioPythonFinalPath -Path `$temp } catch { `$pyThrew = `$true; Write-Host "PY_ERROR: `$(`$_.Exception.Message)" }
Write-Host "PY_THREW: `$pyThrew"
Write-Host "PY_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$py))"
Write-Host "PY_ABSOLUTE: `$(if (`$py -and `$py -ne 'unset') { [bool](Split-Path -IsAbsolute `$py) } else { `$false })"

# 2. The full ladder on the same host. Resolve-StudioFinalPathInfo is what every caller
#    actually uses, and with an interpreter present it must report the answer as exact
#    and must NOT print the degradation warning: a host that resolved a path exactly
#    being told it did not is how an operator stops believing the warning.
`$exactThrew = `$false
`$exactInfo = `$null
try { `$exactInfo = Resolve-StudioFinalPathInfo -Path `$temp } catch { `$exactThrew = `$true; Write-Host "EXACT_ERROR: `$(`$_.Exception.Message)" }
Write-Host "EXACT_THREW: `$exactThrew"
if (`$exactInfo) {
    Write-Host "EXACT_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$exactInfo.Path))"
    Write-Host "EXACT_FLAG: `$(`$exactInfo.Exact)"
}
Write-Host "EXACT_WARNED: `$script:StudioFinalPathWarned"

# 3. A path whose tail does not exist yet, which is the ordinary case: `$StudioHome is
#    resolved before it is created. The existing prefix resolves exactly and the missing
#    segments are appended, so the answer must still end with them.
`$missing = Join-Path `$temp ("uns_missing_" + [guid]::NewGuid().ToString("N"))
`$missingInfo = `$null
`$missingThrew = `$false
try { `$missingInfo = Resolve-StudioFinalPathInfo -Path `$missing } catch { `$missingThrew = `$true }
Write-Host "MISSING_THREW: `$missingThrew"
if (`$missingInfo) {
    Write-Host "MISSING_KEEPS_LEAF: `$(`$missingInfo.Path.EndsWith((Split-Path -Leaf `$missing)))"
    Write-Host "MISSING_EXACT: `$(`$missingInfo.Exact)"
}

# 4. The Python-less host, which is the whole point of the change. The kill switch is
#    the supported way to produce it, and it reaches the same branch a machine with no
#    interpreter reaches: Get-StudioEarlyPython returns nothing, so the ladder drops to
#    the lexical rung. The latch and the per-path answer cache have to be cleared too, or
#    the answer from step 1 is simply replayed and this checks nothing: a real run sets the
#    switch before anything is resolved, so it never has a cached answer to replay.
`$env:UNSLOTH_EARLY_PYTHON_PROBE = "0"
`$script:StudioEarlyPythonProbed = `$false
`$script:StudioEarlyPython = `$null
`$script:StudioPythonFinalPathCache = `$null
`$script:StudioFinalPathWarned = `$false
Write-Host "KILLSWITCH_DECLINES: `$(`$null -eq (Get-StudioEarlyPython))"
`$info = `$null
`$ladderThrew = `$false
try { `$info = Resolve-StudioFinalPathInfo -Path `$temp } catch { `$ladderThrew = `$true; Write-Host "LADDER_ERROR: `$(`$_.Exception.Message)" }
Write-Host "LADDER_THREW: `$ladderThrew"
if (`$info) {
    Write-Host "LADDER_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$info.Path))"
    Write-Host "LADDER_EXACT: `$(`$info.Exact)"
}
# Said out loud, once. Exact = `$false makes Test-StudioPathEqual answer `$null and the
# caller take BOTH runtime locks, so this degradation fails closed; an operator should
# not have to infer that from behaviour.
Write-Host "LADDER_WARNED: `$script:StudioFinalPathWarned"
`$repeat = Resolve-StudioFinalPathInfo -Path `$temp
Write-Host "LADDER_REPEAT_EXACT: `$(`$repeat.Exact)"

# 5. The process-image ladder, top rung. This shell can read its own .Path, so
#    Get-Process answers and nothing below it is consulted.
`$imgThrew = `$false
`$img = "unset"
try { `$img = Get-StudioProcessImagePath -ProcessId `$PID } catch { `$imgThrew = `$true; Write-Host "IMG_ERROR: `$(`$_.Exception.Message)" }
Write-Host "PROCIMG_THREW: `$imgThrew"
Write-Host "PROCIMG_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$img))"
# The rungs below must not have been touched: the ctypes rung costs a child process and
# the WMI rung a full Win32_Process enumeration, once per run each, and neither is worth
# paying for when the top rung already answered.
Write-Host "PROCIMG_STAYED_ON_TOP_RUNG: `$((-not `$script:StudioPythonProcessImageProbed) -and (`$null -eq `$script:StudioProcessImageTable))"

# 6. The rungs BELOW Get-Process, which step 5 never reaches. That made the lower half of
#    the ladder untested here and would hide a helper missing from the extraction list
#    above until some future scenario reached it. Force Get-Process to answer nothing,
#    which is what a process this shell cannot inspect looks like from in here.
function Get-Process { param(`$Id, `$ErrorAction) return `$null }
`$deepThrew = `$false
`$deep = "unset"
try { `$deep = Get-StudioProcessImagePath -ProcessId `$PID } catch { `$deepThrew = `$true; Write-Host "DEEP_ERROR: `$(`$_.Exception.Message)" }
Write-Host "PROCIMG_DEEP_THREW: `$deepThrew"
# Null, not an exception and not a wrong path. On this engine the ctypes rung declines
# (it is Windows-only) and Win32_Process does not exist, so the ladder runs out. A caller
# reading null treats the process as unknown and keeps its lock, which is the safe end.
Write-Host "PROCIMG_DEEP_NULL: `$([string]::IsNullOrWhiteSpace(`$deep))"
# Asked once, not once per process. Get-RunningStudioVenvProcesses calls this for every
# process on the machine; without the latch that is one child interpreter each.
#
# Counted, not inferred from the latch variable. Reading the flag only proves the flag was
# set, and a ladder that probes unconditionally sets it every time too: that spelling
# passed this check while re-probing on every call.
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
    try {
        $out = & $exe -NoProfile -File $file 2>&1 | Out-String
    } finally {
        Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue
    }

    if ($out -notmatch "PY_THREW") { Write-Host "  ---- harness output ----"; Write-Host $out }

    Check "the interpreter rung does not throw" ($out -match "PY_THREW: False")
    Check "the interpreter rung answers" ($out -match "PY_ANSWERED: True")
    Check "and answers with an absolute path" ($out -match "PY_ABSOLUTE: True")
    Check "the full ladder does not throw with an interpreter present" ($out -match "EXACT_THREW: False")
    Check "the full ladder answers" ($out -match "EXACT_ANSWERED: True")
    Check "and reports the answer as exact" ($out -match "EXACT_FLAG: True")
    Check "an exact answer prints no degradation warning" ($out -match "EXACT_WARNED: False")
    Check "a not-yet-created path does not throw" ($out -match "MISSING_THREW: False")
    Check "and keeps the segments that do not exist yet" ($out -match "MISSING_KEEPS_LEAF: True")
    Check "and is still exact, since its existing prefix resolved" ($out -match "MISSING_EXACT: True")
    Check "the kill switch really removes the interpreter" ($out -match "KILLSWITCH_DECLINES: True")
    Check "the ladder without an interpreter does not throw" ($out -match "LADDER_THREW: False")
    Check "the ladder without an interpreter still answers" ($out -match "LADDER_ANSWERED: True")
    Check "and reports the answer as inexact" ($out -match "LADDER_EXACT: False")
    Check "and says out loud that it is degraded" ($out -match "LADDER_WARNED: True")
    # The warning latches; the degraded VERDICT must not, or the second caller believes
    # an inexact path is exact.
    Check "a second call is still inexact" ($out -match "LADDER_REPEAT_EXACT: False")
    Check "the degradation warning is printed once" (
        ([regex]::Matches($out, "LINE: \[WARN\] Could not resolve a path exactly")).Count -eq 1)
    Check "the process image ladder does not throw" ($out -match "PROCIMG_THREW: False")
    Check "the process image ladder answers our own image" ($out -match "PROCIMG_ANSWERED: True")
    Check "and pays for no rung below the one that answered" ($out -match "PROCIMG_STAYED_ON_TOP_RUNG: True")
    Check "the rungs below Get-Process do not throw either" ($out -match "PROCIMG_DEEP_THREW: False")
    Check "an unanswerable process reads as unknown, not as a path" ($out -match "PROCIMG_DEEP_NULL: True")
    Check "the child interpreter is asked once per run" ($out -match "PROCIMG_PROBED_ONCE: True")
    Check "and is not asked again on the next process" ($out -match "PROCIMG_NOT_REPROBED: True")
    Check "and an empty result is not re-probed" ($out -match "PROCIMG_TABLE_STILL_NULL: True")
}

$probeFn = Get-InstallFunctionSource -Path $installPath -Name "Test-StudioSameDirectoryByProbe"
Check "install.ps1 defines Test-StudioSameDirectoryByProbe" ($null -ne $probeFn)
if ($probeFn) {
    Invoke-Expression $probeFn
    $probeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-probe-" + [guid]::NewGuid().ToString("N"))
    $realDir = Join-Path $probeRoot "real"
    $otherDir = Join-Path $probeRoot "other"
    $null = New-Item -ItemType Directory -Path $realDir -Force
    $null = New-Item -ItemType Directory -Path $otherDir -Force
    $aliasDir = Join-Path $probeRoot "alias"
    $madeAlias = $false
    try {
        $null = New-Item -ItemType SymbolicLink -Path $aliasDir -Target $realDir -ErrorAction Stop
        $madeAlias = $true
    } catch { }
    Check "two different directories are not one" (
        (Test-StudioSameDirectoryByProbe -Left $realDir -Right $otherDir) -eq $false)
    Check "a directory is itself" (
        (Test-StudioSameDirectoryByProbe -Left $realDir -Right $realDir) -eq $true)
    if ($madeAlias) {
        Check "two spellings that reach one directory are one" (
            (Test-StudioSameDirectoryByProbe -Left $aliasDir -Right $realDir) -eq $true)
    } else {
        Write-Host "  SKIP  this host cannot create a link to test the aliased spelling"
    }
    # A side that does not exist cannot be an alias of one that does, and the answer must come
    # without writing anything: the caller is about to refuse the install on it.
    Check "a missing right side is not a match" (
        (Test-StudioSameDirectoryByProbe -Left $realDir -Right (Join-Path $probeRoot "absent")) -eq $false)
    Check "a missing left side is not a match" (
        (Test-StudioSameDirectoryByProbe -Left (Join-Path $probeRoot "absent") -Right $realDir) -eq $false)
    Check "an empty spelling is not a match" (
        (Test-StudioSameDirectoryByProbe -Left "" -Right $realDir) -eq $false)
    Check "the probe file is removed again" (
        @(Get-ChildItem -LiteralPath $realDir -Force -ErrorAction SilentlyContinue).Count -eq 0)
    Remove-Item -LiteralPath $probeRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$ancestorFn = Get-InstallFunctionSource -Path $installPath -Name "Split-StudioExistingAncestor"
Check "install.ps1 defines Split-StudioExistingAncestor" ($null -ne $ancestorFn)
if ($ancestorFn) {
    Invoke-Expression $ancestorFn
    $ancRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-anc-" + [guid]::NewGuid().ToString("N"))
    $null = New-Item -ItemType Directory -Path $ancRoot -Force
    $missing = Join-Path $ancRoot (Join-Path "not-there" "either")
    $split = Split-StudioExistingAncestor -Path $missing
    Check "the deepest existing ancestor is found" ($split -and $split.Root -eq $ancRoot)
    Check "and the part that does not exist is kept whole" (
        $split -and $split.Tail -eq (Join-Path "not-there" "either"))
    $here = Split-StudioExistingAncestor -Path $ancRoot
    Check "a path that exists is its own ancestor, with nothing below it" (
        $here -and $here.Root -eq $ancRoot -and $here.Tail -eq "")
    Check "a path with no existing ancestor at all answers nothing" (
        $null -eq (Split-StudioExistingAncestor -Path (Join-Path "/nonexistent-root-xyz" "a")))
    Check "an empty path answers nothing" ($null -eq (Split-StudioExistingAncestor -Path ""))
    Remove-Item -LiteralPath $ancRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$installTextForGate = [System.IO.File]::ReadAllText($installPath)
Check "the --tauri gate falls back to comparing existing ancestors" (
    $installTextForGate -match 'Split-StudioExistingAncestor -Path \$_tauriOverride')
Check "and requires the segments below them to match" (
    $installTextForGate -match '\$_tauriLeft\.Tail, \$_tauriRight\.Tail')

$installText = [System.IO.File]::ReadAllText($installPath)
Check "the --tauri gate asks it before refusing an override" (
    $installText -match 'Test-StudioSameDirectoryByProbe -Left \$_tauriOverride -Right \$_legacyTauriRoot')

$selfText = Get-Content -LiteralPath $PSCommandPath -Raw
$gateIndex = $selfText.IndexOf('if ($fail' + 'ures -gt 0) {')
$afterGate = $selfText.Substring($gateIndex)
Check "no check is written after the failure gate" ($afterGate -notmatch "(?m)^\s*Check ")

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
