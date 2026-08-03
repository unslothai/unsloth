#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The hardware report must not say "none (chat-only / GGUF)" on a working XPU environment.
#
# The WMI scan and the registry fallback can both miss a real Arc host (wedged CIM service,
# an Intel part outside the Arc|Data Center regex). setup.ps1 already re-checks the truth with
# torch.xpu.is_available(), but that runs AFTER the report, so the user was told no
# training-capable GPU exists and then watched setup keep the XPU venv. These two helpers let
# the scan reach the same answer before it prints.
#
# The point of Test-VenvTorchIsXpu is that it is FREE: a CPU-only host must not pay for an
# `import torch` on every `studio update` just to be told it has no Intel GPU. So the tests
# below check both the answer and that the cheap gate is what decides.
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
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the home override" ($venvDirFn -match 'UNSLOTH_STUDIO_HOME')
Check "extraction kept the version file"  ($isXpuFn -match 'version\.py')

# --- Get-ProbableStudioVenvDir -------------------------------------------------------------
# Runs on Linux CI too, so Test-Path is shadowed rather than backed by a real hive/venv: the
# helper builds Windows paths, and this asserts the path it BUILT, not what the OS resolves.
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
# A literal ~ would leave a cwd-relative path and probe the wrong tree.
Check "tilde expands"           ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "~/s" } @("C:\Users\me\s\unsloth_studio")) -eq "C:\Users\me\s\unsloth_studio")
Check "bare tilde expands"      ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "~" } @("C:\Users\me\unsloth_studio")) -eq "C:\Users\me\unsloth_studio")
Check "whitespace override ignored" ((Invoke-VenvDir @{ USERPROFILE = "C:\Users\me"; UNSLOTH_STUDIO_HOME = "   " } @($DEFAULT)) -eq $DEFAULT)

Write-Host "and reports nothing rather than guessing"
Check "absent venv -> null"     ($null -eq (Invoke-VenvDir @{ USERPROFILE = "C:\Users\me" } @()))
Check "no USERPROFILE -> null"  ($null -eq (Invoke-VenvDir @{} @($DEFAULT)))
Check "tilde, no USERPROFILE"   ($null -eq (Invoke-VenvDir @{ UNSLOTH_STUDIO_HOME = "~/s" } @("C:\Users\me\s\unsloth_studio")))

# --- Test-VenvTorchIsXpu -------------------------------------------------------------------
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
# torch.version.xpu is None on some +xpu builds, so the local label is the only reliable
# signal; a line mentioning xpu without the label must not count.
Check "xpu attr but cuda wheel" (-not (Invoke-IsXpu "C:\v" ($CU + "`nxpu: Optional[str] = None")))
Check "no torch installed"      (-not (Invoke-IsXpu "C:\v" $XPU -Missing))
Check "unreadable file"         (-not (Invoke-IsXpu "C:\v" $XPU -Throws))
Check "no venv path"            (-not (Invoke-IsXpu "" $XPU))

# --- wiring ---------------------------------------------------------------------------------
$setupText = Get-Content -Raw $setup
Write-Host "the scan asks before the report, and only when it has to"
# The promotion must sit inside the Intel scan (which the report reads), not after it.
Check "promotion is gated on the scan having failed" ($setupText -match '(?s)if \(-not \$script:IsIntelXpu\) \{\s*try \{\s*\$_probeVenv = Get-ProbableStudioVenvDir')
# Its own try, not the scan's: it must still run when the scan threw (the case it exists for),
# and a junk UNSLOTH_STUDIO_HOME makes Join-Path throw, which would otherwise abort setup.
Check "promotion cannot abort setup" ($setupText -match '(?s)\$_probeVenv = Get-ProbableStudioVenvDir.{0,600}?\} catch \{\}')
# The cheap disk read must gate the interpreter launch, or every CPU-only `studio update`
# pays for an `import torch`.
Check "disk read gates the probe" ($setupText -match '(?s)if \(Test-VenvTorchIsXpu \$_probeVenv\) \{\s*\$_probePy = .*?Test-TorchXpuAvailable')
$_scanLine = ($setupText -split "`n" | Select-String -Pattern '\$_probeVenv = Get-ProbableStudioVenvDir' | Select-Object -First 1).LineNumber
$_reportLine = ($setupText -split "`n" | Select-String -Pattern 'none \(chat-only / GGUF\)' | Select-Object -First 1).LineNumber
Check "promotion precedes the hardware report" ($_scanLine -and $_reportLine -and $_scanLine -lt $_reportLine)

Write-Host "a timed-out probe must not destroy a working XPU venv"
# Bounding the flavour probe turned a hang into "rebuild", and the host most likely to time out
# inside `import torch` is precisely an Arc box whose compute driver stalled -- where
# torch/version.py still names a good +xpu wheel. Without a currently exported pin the stale
# path then DELETES $VenvDir and exits. The rescue is bounded by the surrounding if-chain here
# rather than a line offset, so an edit inside it cannot move the window off the code.
$_rescue = if ($setupText -match '(?s)(\$_verProbe = Invoke-BoundedPythonProbe.*?# Missing python\.exe means)') { $Matches[1] } else { "" }
Check "the probe chain was found"      ($_rescue -ne "")
Check "an unreadable flavour is rescued off disk" ($_rescue -match 'elseif \(Test-VenvTorchIsXpu -VenvPath \$VenvDir\)')
Check "the rescue classifies it as xpu" ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?\$installedTorchTag = "xpu"')
# It must NOT set the rebuild flag, or the whole point is lost.
Check "the rescue never sets shouldRebuild" ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?\} else \{\s*\$shouldRebuild = \$true' -and
    -not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu[^}]*\$shouldRebuild = \$true'))
# A silent rescue would leave an Arc owner with a venv that trains on nothing and no clue why.
# Match the substep ARGUMENT, not the region: the comment above the branch says "compute
# driver" too, so a loose match stayed green with the message itself deleted.
Check "the rescue names the driver fix" ($_rescue -match 'substep "[^"]*compute driver')
# Non-Intel families must still rebuild on an unreadable flavour, exactly as before.
Check "other families still rebuild"   ($_rescue -match '(?s)\} else \{\s*\$shouldRebuild = \$true')
# The rescue reads the disk, so it must not launch a second interpreter after the first hung.
Check "the rescue launches no interpreter" (-not ($_rescue -match '(?s)elseif \(Test-VenvTorchIsXpu.*?Invoke-BoundedPythonProbe'))

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
