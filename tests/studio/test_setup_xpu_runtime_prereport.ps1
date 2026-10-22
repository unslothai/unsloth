#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The hardware report must not say "none (chat-only / GGUF)" on a working XPU host.
# Test-VenvTorchIsXpu must stay free: CPU-only hosts must not pay for `import torch`.
# Run: pwsh -NoProfile -File tests/studio/test_setup_xpu_runtime_prereport.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setup = Join-Path $repo "studio/setup.ps1"

function Get-FunctionText {
    param([string] $Path, [string] $Name)
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in $Path, found $($fn.Count)" }
    return $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$venvDirFn = Get-FunctionText $setup "Get-ProbableStudioVenvDir"
$isXpuFn = Get-FunctionText $setup "Test-VenvTorchIsXpu"
# An empty extraction would make every case below pass vacuously.
Check "extraction kept the home override" ($venvDirFn -match 'UNSLOTH_STUDIO_HOME')
Check "extraction kept the version file"  ($isXpuFn -match 'version\.py')

# Test-Path is shadowed so Linux CI asserts the Windows path the helper built.
function Invoke-VenvDir {
    param([hashtable] $Env, [string[]] $Exists = @())
    $sb = [scriptblock]::Create(@"
param(`$Exists)
function Join-Path {
    [CmdletBinding()] param([Parameter(Position=0)] [string] `$Path, [Parameter(Position=1)] [string] `$ChildPath)
    # Shadowed: the real cmdlet validates the drive qualifier, so "C:\..." throws on Linux CI.
    # Windows semantics for the only shape used here -- join with a single backslash.
    if (-not `$ChildPath) { return `$Path }
    return (`$Path.TrimEnd('\', '/') + '\' + `$ChildPath.TrimStart('\', '/'))
}
function Test-Path {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    return (`$Exists -contains `$LiteralPath)
}
$venvDirFn
Get-ProbableStudioVenvDir
"@)
    $saved = @{}
    foreach ($k in 'UNSLOTH_STUDIO_HOME', 'STUDIO_HOME', 'USERPROFILE') {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        if ($Env.ContainsKey($k)) { Set-Item "Env:$k" $Env[$k] } else { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    }
    try { return (& $sb $Exists) }
    finally {
        foreach ($k in $saved.Keys) {
            if ($null -eq $saved[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $saved[$k] }
        }
    }
}

$DEFAULT = "C:\Users\me\.unsloth\studio\unsloth_studio"
Write-Host "the venv directory is found where the canonical resolver would put it"
Check "default location"        ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me" } @($DEFAULT)) -eq $DEFAULT)
Check "UNSLOTH_STUDIO_HOME"     ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "D:\alt" } @("D:\alt\unsloth_studio")) -eq "D:\alt\unsloth_studio")
Check "STUDIO_HOME alias"       ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; STUDIO_HOME = "D:\alt" } @("D:\alt\unsloth_studio")) -eq "D:\alt\unsloth_studio")
Check "UNSLOTH_STUDIO_HOME wins" ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "D:\a"; STUDIO_HOME = "D:\b" } @("D:\a\unsloth_studio")) -eq "D:\a\unsloth_studio")
Check "tilde expands"           ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "~/s" } @("C:\Users\me\s\unsloth_studio")) -eq "C:\Users\me\s\unsloth_studio")
Check "bare tilde expands"      ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "~" } @("C:\Users\me\unsloth_studio")) -eq "C:\Users\me\unsloth_studio")
Check "whitespace override ignored" ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "   " } @($DEFAULT)) -eq $DEFAULT)

Write-Host "and reports nothing rather than guessing"
Check "absent venv -> null"     ($null -eq (Invoke-VenvDir @{ USERPROFILE = "C:\Users\me" } @()))
Check "no USERPROFILE -> null"  ($null -eq (Invoke-VenvDir @{} @($DEFAULT)))
Check "tilde, no USERPROFILE"   ($null -eq (Invoke-VenvDir @{ UNSLOTH_STUDIO_HOME = "~/s" } @("C:\Users\me\s\unsloth_studio")))

function Invoke-IsXpu {
    param([string] $VenvPath, [string] $VersionPyBody, [switch] $Missing, [switch] $Throws)
    $sb = [scriptblock]::Create(@"
param(`$Body, `$Missing, `$Throws)
function Join-Path {
    [CmdletBinding()] param([Parameter(Position=0)] [string] `$Path, [Parameter(Position=1)] [string] `$ChildPath)
    # Shadowed: the real cmdlet validates the drive qualifier, so "C:\..." throws on Linux CI.
    # Windows semantics for the only shape used here -- join with a single backslash.
    if (-not `$ChildPath) { return `$Path }
    return (`$Path.TrimEnd('\', '/') + '\' + `$ChildPath.TrimStart('\', '/'))
}
function Test-Path {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    return (-not `$Missing)
}
function Get-Content {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    if (`$Throws) { throw "access denied" }
    return (`$Body -split "``n")
}
$isXpuFn
Test-VenvTorchIsXpu -VenvPath '$VenvPath'
"@)
    return (& $sb $VersionPyBody ([bool]$Missing) ([bool]$Throws))
}

$XPU = "__version__ = '2.9.1+xpu'`ndebug = False`nxpu: Optional[str] = '2025.2'"
$CU  = "__version__ = '2.9.1+cu128'`ndebug = False`ncuda: Optional[str] = '12.8'"
$ROC = "__version__ = '2.9.1+rocm6.4'"
$BARE = "__version__ = '2.9.1'"

Write-Host "the wheel flavour is read off disk"
Check "xpu wheel"               (Invoke-IsXpu "C:\v" $XPU)
Check "cuda wheel"              (-not (Invoke-IsXpu "C:\v" $CU))
Check "rocm wheel"              (-not (Invoke-IsXpu "C:\v" $ROC))
Check "untagged wheel"          (-not (Invoke-IsXpu "C:\v" $BARE))
# torch.version.xpu is None on some +xpu builds, so only the local label counts.
Check "xpu attr but cuda wheel" (-not (Invoke-IsXpu "C:\v" ($CU + "`nxpu: Optional[str] = None")))
Check "no torch installed"      (-not (Invoke-IsXpu "C:\v" $XPU -Missing))
Check "unreadable file"         (-not (Invoke-IsXpu "C:\v" $XPU -Throws))
Check "no venv path"            (-not (Invoke-IsXpu "" $XPU))

# The fast path must judge the wheel only: no dependency pass repairs a driver.
$isXpuSupFn = Get-FunctionText $setup "Test-VenvTorchIsXpuSupported"
Check "extraction kept the range" ($isXpuSupFn -match '11')
function Invoke-IsXpuSupported {
    param([string] $VenvPath, [string] $VersionPyBody, [switch] $Missing, [switch] $Throws)
    $sb = [scriptblock]::Create(@"
param(`$Body, `$Missing, `$Throws)
function Join-Path {
    [CmdletBinding()] param([Parameter(Position=0)] [string] `$Path, [Parameter(Position=1)] [string] `$ChildPath)
    if (-not `$ChildPath) { return `$Path }
    return (`$Path.TrimEnd('\', '/') + '\' + `$ChildPath.TrimStart('\', '/'))
}
function Test-Path {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    return (-not `$Missing)
}
function Get-Content {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    if (`$Throws) { throw "access denied" }
    return (`$Body -split "``n")
}
$isXpuSupFn
Test-VenvTorchIsXpuSupported -VenvPath '$VenvPath'
"@)
    return (& $sb $VersionPyBody ([bool]$Missing) ([bool]$Throws))
}

Write-Host "flavour AND range, both off disk"
Check "2.9.1+xpu supported"  (Invoke-IsXpuSupported "C:\v" $XPU)
Check "2.6.0+xpu is the floor" (Invoke-IsXpuSupported "C:\v" "__version__ = '2.6.0+xpu'")
Check "2.10.0+xpu supported" (Invoke-IsXpuSupported "C:\v" "__version__ = '2.10.0+xpu'")
Check "2.5.1+xpu below floor" (-not (Invoke-IsXpuSupported "C:\v" "__version__ = '2.5.1+xpu'"))
Check "2.11.0+xpu at ceiling" (-not (Invoke-IsXpuSupported "C:\v" "__version__ = '2.11.0+xpu'"))
Check "3.0.0+xpu above range" (-not (Invoke-IsXpuSupported "C:\v" "__version__ = '3.0.0+xpu'"))
Check "cuda wheel"           (-not (Invoke-IsXpuSupported "C:\v" $CU))
Check "untagged wheel"       (-not (Invoke-IsXpuSupported "C:\v" $BARE))
Check "xpu attr but cuda wheel" (-not (Invoke-IsXpuSupported "C:\v" ($CU + "`nxpu: Optional[str] = None")))
Check "dev label reads its base" (Invoke-IsXpuSupported "C:\v" "__version__ = '2.9.0.dev20260101+xpu'")
Check "junk label"           (-not (Invoke-IsXpuSupported "C:\v" "__version__ = 'unknown'"))
Check "no version line"      (-not (Invoke-IsXpuSupported "C:\v" "debug = False"))
Check "no torch installed"   (-not (Invoke-IsXpuSupported "C:\v" $XPU -Missing))
Check "unreadable file"      (-not (Invoke-IsXpuSupported "C:\v" $XPU -Throws))
Check "no venv path"         (-not (Invoke-IsXpuSupported "" $XPU))

# Normalised to LF: on a CRLF checkout \n-anchored patterns match nothing and -not checks pass vacuously.
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"
Write-Host "the scan asks before the report, and only when it has to"
Check "promotion is gated on the scan having failed" ($setupText -match '(?s)if \(-not \$script:IsIntelXpu\) \{\s*try \{\s*\$_probeVenv = Get-ProbableStudioVenvDir')
# Its own try: a junk UNSLOTH_STUDIO_HOME makes Join-Path throw and must not abort setup.
Check "promotion cannot abort setup" ($setupText -match '(?s)\$_probeVenv = Get-ProbableStudioVenvDir.{0,600}?\} catch \{\}')
Check "disk read gates the probe" ($setupText -match '(?s)if \(Test-VenvTorchIsXpu \$_probeVenv\) \{\s*\$_probePy = .*?Test-TorchXpuAvailable')
$_scanLine = ($setupText -split "`n" | Select-String -Pattern '\$_probeVenv = Get-ProbableStudioVenvDir' | Select-Object -First 1).LineNumber
$_reportLine = ($setupText -split "`n" | Select-String -Pattern 'none \(chat-only / GGUF\)' | Select-Object -First 1).LineNumber
Check "promotion precedes the hardware report" ($_scanLine -and $_reportLine -and $_scanLine -lt $_reportLine)

Write-Host "the fast-path escape judges the wheel, never the driver"
# A driver-only failure must leave the fast path intact; clearing it costs a full resolution.
$_fastPat = '(?s)(if \(\$script:IsIntelXpu -and \$SkipPythonDeps -and \$_xpuIsReachable\) \{.*?\n        \}\n)'
$_fast = if ($setupText -match $_fastPat) { $Matches[1] } else { "" }
Check "the escape was found"           ($_fast -ne "")
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $_fastPat))
Check "it reads the installed wheel"   ($_fast -match 'Test-VenvTorchIsXpuSupported -VenvPath \$VenvDir')
Check "it clears the fast path"        ($_fast -match '\$SkipPythonDeps = \$false')
Check "readiness does not decide it"   (-not ($_fast -match 'Test-TorchXpuAvailable'))
# On an Arc box with a stalled driver, `import torch` is exactly what hangs.
Check "it launches no interpreter"     (-not ($_fast -match 'Invoke-BoundedPythonProbe|& python'))
Check "the driver warning is elsewhere" ($setupText -match '(?s)function Assert-XpuRuntimeReady \{.*?Test-TorchXpuAvailable')

Write-Host "a timed-out probe must not destroy a working XPU venv"
# A stalled Arc driver hangs `import torch` while version.py still names a good +xpu wheel.
$_rescue = if ($setupText -match '(?s)(\$_verProbe = Invoke-BoundedPythonProbe.*?# Missing python\.exe means)') { $Matches[1] } else { "" }
Check "the probe chain was found"      ($_rescue -ne "")
Check "an unreadable flavour is rescued off disk" ($_rescue -match 'elseif \(Test-VenvTorchIsXpu -VenvPath \$VenvDir\)')
Check "the rescue classifies it as xpu" ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?\$installedTorchTag = "xpu"')
Check "the rescue never sets shouldRebuild" ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?\} else \{\s*\$shouldRebuild = \$true' -and
    -not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu[^}]*\$shouldRebuild = \$true'))
# Match the substep argument: the comment above the branch also says "compute driver".
Check "the rescue names the driver fix" ($_rescue -match 'substep "[^"]*compute driver')
Check "other families still rebuild"   ($_rescue -match '(?s)\} else \{\s*\$shouldRebuild = \$true')
Check "the rescue launches no interpreter" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?Invoke-BoundedPythonProbe'))

Write-Host "a direct update never deletes an XPU venv"
# On a hybrid NVIDIA+Arc box a later pinless update would otherwise wipe the working Arc venv.
$_keep = if ($setupText -match '(?s)(\$script:PinChangedForceReinstall = \$true.*?if \(\$shouldRebuild\) \{)') { $Matches[1] } else { "" }
Check "the rebuild decision was found"  ($_keep -ne "")
Check "an xpu venv is spared the wipe"  ($_keep -match '\$installedTorchTag -eq "xpu"')
Check "the escape clears shouldRebuild" ($_keep -match '(?s)\$installedTorchTag -eq "xpu"\) \{.*?\$shouldRebuild = \$false')
# Installer-managed runs reach the in-place repair below, which already covers xpu.
Check "installer-managed runs skip the xpu-only escape" ($_keep -match '-not \$InstallerManagedSetup')
Check "the escape says how to replace it" ($_keep -match 'substep "[^"]*install\.ps1')
# The CUDA arm would install triton-windows over torch's XPU triton; stay on the xpu index.
Check "the escape steers the install too" ($_keep -match '\$script:PreservedXpuVenv = \$true')
Check "the flag is declared up front"     ($setupText -match '(?m)^\$script:PreservedXpuVenv = \$false')
$_tagChain = if ($setupText -match '(?s)(\$PinnedTorchIndexUrl\) \{\s*\$CuTag = Get-TorchIndexLeaf.*?\$CuTag = "cpu")') { $Matches[1] } else { "" }
Check "the index chain was found"         ($_tagChain -ne "")
Check "a preserved venv picks the xpu leaf" ($_tagChain -match '(?s)\$script:PreservedXpuVenv\) \{.*?\$CuTag = "xpu"')
Check "it outranks the NVIDIA arm" (
    $_tagChain.IndexOf('$script:PreservedXpuVenv') -ge 0 -and
    $_tagChain.IndexOf('$script:PreservedXpuVenv') -lt $_tagChain.IndexOf('$HasNvidiaSmi'))
Check "an explicit pin still outranks it" (
    $_tagChain.IndexOf('$PinnedTorchIndexUrl') -lt $_tagChain.IndexOf('$script:PreservedXpuVenv'))
# The report must stay honest: there is an NVIDIA GPU in this machine.
Check "the report is not forced to Intel" (-not ($_keep -match '\$script:IsIntelXpu = \$true'))

Write-Host "a Triton swap that strands the venv must fail the setup"
# Both triton installs failing must fail setup, or install.ps1 commits a broken venv.
$_bothFailed = if ($setupText -match '(?s)(neither triton would reinstall.*?\n\s*\}\s*\n)') { $Matches[1] } else { "" }
Check "the both-failed branch was found" ($_bothFailed -ne "")
Check "it sets a nonzero stack exit"     ($_bothFailed -match '\$stackExit = \$tritonBackExit')
Check "the exit code is the real failure" (-not ($_bothFailed -match '\$stackExit = 0'))
Check "a nonzero stack exit fails setup" ($setupText -match '(?s)if \(\$stackExit -ne 0\) \{.*?Exit-SetupFailure')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
