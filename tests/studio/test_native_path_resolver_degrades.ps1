# Behaviour of install.ps1's native helpers when the native side is not available,
# which is the state a blocked or locked-down machine is in.
#
# The change this guards replaced every Add-Type compile with reflection emit. The
# ladders around them did not move: an exact path when the native call works, the
# lexical path when it does not with $Exact false so the runtime lock fails closed,
# and for the process scan a native image path first, then Get-Process, then
# Win32_Process. This runs the real functions, parsed out of install.ps1, and checks
# that every rung still answers.
#
# POSIX is the harness on purpose. Two different failures live here and they are not
# the same: reflection emit SUCCEEDS on this engine, so the types build exactly as
# they do on 5.1, while kernel32 and shell32 do not resolve, so every call through
# them fails exactly as it does on a Windows host whose antivirus took the helper
# away. Between them that covers the build side and the call side. The Windows lanes
# cover the case where the call itself succeeds.

$ErrorActionPreference = "Stop"

# Refuse rather than assert nonsense. Half the checks below require the native calls to
# FAIL, which on a healthy Windows host they do not, so a Windows lane running this file
# would report a pile of failures that say nothing about the code. It was scheduled on a
# windows-latest job once; exiting here means the next time it happens the message names
# the reason instead.
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "SKIP: this file describes a host where kernel32 does not resolve, which Windows is not."
    exit 0
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# The parser, not tests/studio_setup_ps1/Get-FunctionSource.ps1, which counts braces
# without tokenizing and so runs past the end of any function holding a brace inside
# a string. Resolve-StudioLinkTarget has one ('Volume{'), and the naive walk swallowed
# the rest of the file. The parser is exact and is right here in the box.
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
$text = Get-Content -LiteralPath $installPath -Raw

# ── Source contract ──
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")
foreach ($pair in @(@("install.ps1", $installPath), @("studio/setup.ps1", $setupPath))) {
    $body = Get-Content -LiteralPath $pair[1] -Raw
    # -MemberDefinition compiles as surely as -TypeDefinition does. -AssemblyName
    # only loads an assembly that already exists and reaches no compiler, so it is
    # not what this bans.
    $compiles = [regex]::Matches($body, "(?m)^[ \t]*Add-Type\b(?![^\r\n]*-AssemblyName)")
    Check "$($pair[0]) compiles no C# at all" ($compiles.Count -eq 0)
    Check "$($pair[0]) emits its native imports instead" ($body -match "DefinePInvokeMethod")
    # Both spellings of "define a dynamic assembly". Windows PowerShell 5.1 on .NET
    # Framework and pwsh on .NET Core do not agree about which one exists, and only
    # one of them can be exercised from here: losing either branch would degrade
    # every install on the other host with no error to see, since the caller caches
    # the failure and falls through to the lexical path.
    Check "$($pair[0]) tries the static DefineDynamicAssembly" (
        $body -match "\[System\.Reflection\.Emit\.AssemblyBuilder\]::DefineDynamicAssembly")
    Check "$($pair[0]) falls back to the AppDomain spelling" (
        $body -match "\[AppDomain\]::CurrentDomain\.DefineDynamicAssembly")
}

# ── The resolver, run for real ──
# The whole ladder, not only its top rung: the claim under test is that a host
# without the native side still gets an answer, and gets told it is inexact.
$fns = @(
    "Write-StudioLine", "Write-StudioFinalPathDegraded",
    "Test-StudioCanDefineNativeTypes", "Test-StudioEmitInChildProcess", "New-StudioDynamicAssembly",
    "New-StudioEmittedNativeType",
    "Initialize-StudioFinalPathNativeType", "Get-StudioNativeFinalPath",
    "Resolve-StudioLinkTarget", "Get-StudioSubstTarget", "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo",
    "Initialize-StudioProcessImageNativeType", "Get-StudioNativeProcessImagePath",
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
function Write-StudioLine { param([string]`$Message, `$ForegroundColor) Write-Host `$Message }
`$script:StudioFinalPathNativeState = `$null
`$script:StudioFinalPathWarned = `$false
`$script:StudioNativeResolveWarned = `$false
`$script:StudioSubstMap = `$null
`$script:StudioProcessImageNativeState = `$null
`$script:StudioProcessImageTable = `$null
`$script:StudioProcessImageWarned = `$false
$($src -join "`n")

# Snapshot first. A runner is a shared machine, and anything else that wrote a
# matching name into the shared temp directory in the preceding minute would
# otherwise be counted as this emission's output and fail the run for it.
`$temp = [System.IO.Path]::GetTempPath()
`$patterns = @(".dll", ".cs", ".cmdline", ".err", ".out")
`$before = @(Get-ChildItem -LiteralPath `$temp -File -ErrorAction SilentlyContinue |
    Where-Object { `$_.Extension -in `$patterns } | Select-Object -ExpandProperty FullName)

# 1. The type must define without a compiler, on this engine as on 5.1.
`$defined = Initialize-StudioFinalPathNativeType
Write-Host "EMIT_OK: `$defined"
Write-Host "TYPE_PRESENT: `$(`$null -ne ("UnslothStudioFinalPathV3" -as [type]))"

# 2. Nothing may be written to the temporary directory by defining it. Set
#    difference against the snapshot above, not a recency window: what this asserts
#    is that emission wrote nothing, and the clock cannot tell those apart.
`$after = @(Get-ChildItem -LiteralPath `$temp -File -ErrorAction SilentlyContinue |
    Where-Object { `$_.Extension -in `$patterns } | Select-Object -ExpandProperty FullName)
`$fresh = @(`$after | Where-Object { `$before -notcontains `$_ })
Write-Host "TEMP_ARTIFACTS: `$(`$fresh.Count)"

# 3. The dynamic assembly must exist only in memory.
if (`$defined) {
    `$asm = ("UnslothStudioFinalPathV3" -as [type]).Assembly
    Write-Host "IN_MEMORY_ONLY: `$([string]::IsNullOrEmpty(`$asm.Location))"
}

# 4. A native call that cannot work must answer null, not throw. kernel32 is
#    absent here, which is what a host missing the helper looks like.
`$threw = `$false
`$answer = "unset"
try { `$answer = Get-StudioNativeFinalPath -Path `$temp } catch { `$threw = `$true }
Write-Host "NATIVE_THREW: `$threw"
Write-Host "NATIVE_ANSWER_NULL: `$([string]::IsNullOrEmpty(`$answer))"

# 5. The other two emitted types must define with the same signatures the C# they
#    replace declared. `out uint` has to arrive as a by-ref, and a void return has
#    to stay void, or the call marshals wrongly on a host where it does run.
`$vt = New-StudioEmittedNativeType -TypeName "StudioVTNative" -Imports @(
    @{ Name = "GetStdHandle"; Library = "kernel32.dll"; Return = [IntPtr]; Args = @([int]) },
    @{ Name = "GetConsoleMode"; Library = "kernel32.dll"; Return = [bool]
       Args = @([IntPtr], [uint32].MakeByRefType()) },
    @{ Name = "SetConsoleMode"; Library = "kernel32.dll"; Return = [bool]
       Args = @([IntPtr], [uint32]) })
`$icon = New-StudioEmittedNativeType -TypeName "UnslothShellIconRefresh" -Imports @(
    @{ Name = "SHChangeNotify"; Library = "shell32.dll"; Return = [System.Void]
       Args = @([int], [uint32], [string], [IntPtr]) })
Write-Host "VT_EMIT_OK: `$vt"
Write-Host "ICON_EMIT_OK: `$icon"
`$gcm = ("StudioVTNative" -as [type]).GetMethod("GetConsoleMode")
Write-Host "VT_BYREF_OUT: `$(`$gcm.GetParameters()[1].ParameterType.IsByRef)"
`$shcn = ("UnslothShellIconRefresh" -as [type]).GetMethod("SHChangeNotify")
Write-Host "ICON_RETURNS_VOID: `$(`$shcn.ReturnType -eq [System.Void])"

# 6. Defining a type twice must not corrupt the first one. Callers guard with
#    -as [type], but a partial first attempt would otherwise leave a half-built
#    type behind, which is what Add-Type's "already exists" error used to prevent.
`$redefineThrew = `$false
try { `$null = New-StudioEmittedNativeType -TypeName "StudioVTNative" -Imports @(
    @{ Name = "GetStdHandle"; Library = "kernel32.dll"; Return = [IntPtr]; Args = @([int]) }) }
catch { `$redefineThrew = `$true }
Write-Host "REDEFINE_THREW: `$redefineThrew"
Write-Host "REDEFINE_TYPE_INTACT: `$(`$null -ne (("StudioVTNative" -as [type]).GetMethod("SetConsoleMode")))"

# 7. The other rung. Resolve-StudioFinalPathInfo is what every caller actually
#    uses, and the point of the change is that it still answers when the native
#    side cannot: a lexical path, and Exact false so the runtime lock fails closed.
`$info = `$null
`$ladderThrew = `$false
try { `$info = Resolve-StudioFinalPathInfo -Path `$temp } catch { `$ladderThrew = `$true; Write-Host "LADDER_ERROR: `$(`$_.Exception.Message)" }
Write-Host "LADDER_THREW: `$ladderThrew"
if (`$info) {
    Write-Host "LADDER_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$info.Path))"
    Write-Host "LADDER_EXACT: `$(`$info.Exact)"
}

# 8. The process-image ladder. Two states, and they are different states: the
#    emit SUCCEEDS here even though kernel32 does not resolve, so this takes the
#    native rung and must answer null without throwing, exactly as it did when the
#    type came from a compile that succeeded and a call that then failed. Forcing
#    the type unavailable is what exercises the Get-Process rung below it.
`$imgThrew = `$false
`$img = "unset"
try { `$img = Get-StudioProcessImagePath -ProcessId `$PID } catch { `$imgThrew = `$true }
Write-Host "PROCIMG_NATIVE_STATE: `$(Initialize-StudioProcessImageNativeType)"
Write-Host "PROCIMG_THREW: `$imgThrew"
Write-Host "PROCIMG_NATIVE_NULL: `$([string]::IsNullOrWhiteSpace(`$img))"

# The real initializer caches on type PRESENCE, so clearing the state variable
# would not un-define what step 8 just built. Override the guard instead: the rungs
# under test are the ones below it, and this is the only way to reach them once a
# type exists in the session.
function Initialize-StudioProcessImageNativeType { return `$false }
`$fallbackThrew = `$false
`$fallback = "unset"
try { `$fallback = Get-StudioProcessImagePath -ProcessId `$PID } catch { `$fallbackThrew = `$true }
Write-Host "PROCIMG_FALLBACK_THREW: `$fallbackThrew"
Write-Host "PROCIMG_FALLBACK_ANSWERED: `$(-not [string]::IsNullOrWhiteSpace(`$fallback))"
Write-Host "PROCIMG_WARNED: `$script:StudioProcessImageWarned"
"@
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_native_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $file -Value $harness -Encoding utf8
    $exe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = (Get-Command powershell).Source }
    try {
        $out = & $exe -NoProfile -File $file 2>&1 | Out-String
    } finally {
        Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue
    }

    # A parse failure in the generated harness would otherwise read as every check
    # failing at once, with nothing saying why.
    if ($out -notmatch "EMIT_OK") { Write-Host "  ---- harness output ----"; Write-Host $out }
    Check "the native type defines with no compiler" ($out -match "EMIT_OK: True")
    Check "the emitted type is really there" ($out -match "TYPE_PRESENT: True")
    Check "defining it writes nothing to the temporary directory" ($out -match "TEMP_ARTIFACTS: 0")
    Check "the assembly exists only in memory" ($out -match "IN_MEMORY_ONLY: True")
    # The whole point of the ladder: a native call that cannot work is not fatal.
    Check "a native call that cannot work does not throw" ($out -match "NATIVE_THREW: False")
    Check "a native call that cannot work answers null" ($out -match "NATIVE_ANSWER_NULL: True")
    Check "the console thunk emits too" ($out -match "VT_EMIT_OK: True")
    Check "the icon-refresh thunk emits too" ($out -match "ICON_EMIT_OK: True")
    # These two are what a hand-written emit gets wrong: `out uint` silently
    # becoming a by-value uint, and a void return becoming something else.
    Check "GetConsoleMode's out parameter is by-ref" ($out -match "VT_BYREF_OUT: True")
    Check "SHChangeNotify still returns void" ($out -match "ICON_RETURNS_VOID: True")
    Check "defining a type twice does not throw" ($out -match "REDEFINE_THREW: False")
    Check "and does not corrupt the first definition" ($out -match "REDEFINE_TYPE_INTACT: True")
    # The rung that matters to callers, not just the one that failed.
    Check "the full path ladder does not throw" ($out -match "LADDER_THREW: False")
    Check "the full path ladder still answers" ($out -match "LADDER_ANSWERED: True")
    Check "and reports the answer as inexact" ($out -match "LADDER_EXACT: False")
    # The native rung is REACHED here: emit succeeds on this engine, kernel32 does
    # not resolve, and null without an exception is what the caller skips on.
    Check "the process image type emits too" ($out -match "PROCIMG_NATIVE_STATE: True")
    Check "the process image ladder does not throw" ($out -match "PROCIMG_THREW: False")
    Check "an unusable native process query answers null" ($out -match "PROCIMG_NATIVE_NULL: True")
    # And with the type forced away, the rungs below it still answer, which is what
    # keeps a host from finding no running processes and overwriting an open venv.
    Check "the fallback rung does not throw" ($out -match "PROCIMG_FALLBACK_THREW: False")
    Check "the fallback rung still names our own image" ($out -match "PROCIMG_FALLBACK_ANSWERED: True")
    Check "and says out loud that it is degraded" ($out -match "PROCIMG_WARNED: True")
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed" -ForegroundColor Green
