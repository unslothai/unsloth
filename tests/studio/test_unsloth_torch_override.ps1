#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Windows twin of tests/sh/test_unsloth_torch_override.sh: install.ps1's torch-trio --overrides
# guard (New-UnslothTorchOverridesFile) on the Step-2 unsloth installs. The generated file folds in
# the caller's UV_OVERRIDE lines, which can carry authenticated URLs, so it must never outlive the
# run: install.sh removes its twin from the traps, install.ps1 from the outer finally.
# Every one of those removals goes through Remove-UnslothTempFileQuietly, because Remove-Item's
# -ErrorAction cannot suppress the terminating error the FileSystem provider raises for a path it
# cannot resolve, and an unguarded removal therefore aborts the rest of the cleanup (#11290).
# Run: pwsh -NoProfile -File tests/studio/test_unsloth_torch_override.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path
$installText = Get-Content -Raw $installPath

# Parse install.ps1 (also serves as a syntax gate).
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Text of every Invoke-InstallCommandRetry statement carrying $label; a with-deps path has two.
function Get-InstallBlocks([string]$label) {
    $calls = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst] -and
        $n.Extent.Text -like "*-Label `"$label`" *" -and
        $n.GetCommandName() -eq "Invoke-InstallCommandRetry"
    }, $true)
    if ($calls.Count -eq 0) { throw "no Invoke-InstallCommandRetry found for '$label'" }
    return @($calls | ForEach-Object { $_.Extent.Text })
}

Write-Host "New-UnslothTorchOverridesFile"
$fnAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "New-UnslothTorchOverridesFile"
}, $true)
Check "helper defined exactly once" ($fnAst.Count -eq 1)
$fnText = $fnAst[0].Extent.Text
Check "helper returns null under --no-torch" ($fnText -match 'if \(\$SkipTorch\) \{ return \$null \}')
Check "helper folds in caller UV_OVERRIDE files" ($fnText -match '\$env:UV_OVERRIDE')
# Space-free is not sufficient for an 8.3 alias. A volume can hand back a name that does not
# resolve, and this alias is both uv's --overrides argument and the caller's delete target, so an
# unresolvable one fails the install and then throws on the way out (#11290).
Check "the 8.3 alias is accepted only once it resolves" (
    $fnText -match 'Test-Path -LiteralPath \$short -PathType Leaf')
# Both places the helper gives up on the file it created (a failed write, and no usable short name)
# must also drop the tracked path, or the outer sweep is handed a file that is already gone.
Check "every removal of the created file clears the tracked path" (
    ([regex]::Matches($fnText, 'Remove-UnslothTempFileQuietly -Path \$f')).Count -eq 2 -and
    ([regex]::Matches($fnText, '\$script:TorchOverridesFile = \$null')).Count -eq 2)

Write-Host "with-deps unsloth installs pass --overrides"
foreach ($label in @("install unsloth (local)", "install unsloth")) {
    $blocks = Get-InstallBlocks $label
    $guarded = @($blocks | Where-Object { $_ -match '--overrides \$script:TorchOverridesFile' })
    Check "'$label' has one overrides-guarded call and one plain fallback" (
        $blocks.Count -eq 2 -and $guarded.Count -eq 1)
}

Write-Host "the --no-deps no-torch installs carry no overrides"
foreach ($label in @("install unsloth (no-torch)", "install unsloth (migrated no-torch)")) {
    $blocks = Get-InstallBlocks $label
    Check "'$label' has no --overrides" (@($blocks | Where-Object { $_ -match '--overrides' }).Count -eq 0)
}

Write-Host "the generated temp file never outlives the run"
$removals = [regex]::Matches($installText, 'Remove-UnslothTempFileQuietly -Path \$script:TorchOverridesFile')
# One in-flow removal per with-deps install, plus the outer-finally sweep.
Check "in-flow removal after each with-deps install, plus a final sweep" ($removals.Count -eq 3)
Check "no unguarded Remove-Item on the tracked overrides path survives" (
    $installText -notmatch 'Remove-Item -LiteralPath \$script:TorchOverridesFile')
$outer = @($ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.TryStatementAst] -and
    $n.Body.Extent.Text -match 'Install-UnslothStudio @args'
}, $true))
Check "outer try/finally around Install-UnslothStudio found" ($outer.Count -eq 1)
$finallyText = $outer[0].Finally.Extent.Text
Check "outer finally removes the overrides temp file" (
    $finallyText -match 'Remove-UnslothTempFileQuietly -Path \$script:TorchOverridesFile')
Check "outer finally removes the WOA session overrides the same way" (
    $finallyText -match 'Remove-UnslothTempFileQuietly -Path \$script:WoaSessionOverrides')
Check "outer finally still clears UNSLOTH_KEPT_TORCH" ($finallyText -match 'Remove-Item Env:UNSLOTH_KEPT_TORCH')
# install.sh empties _UNSLOTH_TORCH_OVERRIDES before arming its traps so an inherited value is
# never rm'd; under `irm | iex` the same reset must precede the outer try.
Check "overrides path reset to null before the outer try" (
    $installText -match '(?m)^\$script:TorchOverridesFile = \$null\r?\ntry \{\r?\n\s*Install-UnslothStudio @args')

$guardAst = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Remove-UnslothTempFileQuietly"
}, $true)
Check "Remove-UnslothTempFileQuietly defined exactly once" ($guardAst.Count -eq 1)
# WHERE it is defined is the whole point. A function nested inside Install-UnslothStudio lives in
# that call's local scope and is gone the moment it returns, so the outer finally -- which runs
# after exactly that -- would hit CommandNotFoundException and leave the credential-bearing
# overrides file on disk. Defined at top level it is in scope both inside the installer and after
# it returns, so assert the definition is not a descendant of any function (#11290).
$guardEnclosing = $null
$_guardParent = $guardAst[0].Parent
while ($_guardParent) {
    if ($_guardParent -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
        $guardEnclosing = $_guardParent.Name
        break
    }
    $_guardParent = $_guardParent.Parent
}
Check "the guard is defined at top level, not nested inside a function" ($null -eq $guardEnclosing)
if ($guardEnclosing) { Write-Host "        enclosed by $guardEnclosing, so the outer finally cannot see it" -ForegroundColor Red }
# Define it the way install.ps1 does, from its own source, rather than assuming this scope has it.
Invoke-Expression $guardAst[0].Extent.Text

Write-Host "Remove-UnslothTempFileQuietly is a no-op for anything that is not there"
$liveFile = [System.IO.Path]::GetTempFileName()
Remove-UnslothTempFileQuietly -Path $liveFile
Check "a live temp file is removed" (-not (Test-Path -LiteralPath $liveFile))
$helperThrew = $false
$absent = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-absent-" + [guid]::NewGuid().ToString("N") + ".txt")
foreach ($candidate in @($absent, $null, "")) {
    try { Remove-UnslothTempFileQuietly -Path $candidate } catch { $helperThrew = $true }
}
Check "a missing, null or empty path never throws" (-not $helperThrew)

Write-Host "outer finally actually deletes the file after a terminating error"
# Behavioural: run the real finally body with a live temp file holding a credential line.
$leakFile = [System.IO.Path]::GetTempFileName()
Set-Content -LiteralPath $leakFile -Encoding ascii -Value @(
    "torch==2.11.0+cu128",
    "private-pkg @ https://svc:TOKEN123@pkgs.corp.example/private-1.0-py3-none-any.whl")
$script:TorchOverridesFile = $leakFile
$env:UNSLOTH_KEPT_TORCH = "2.11.0"
# Strip the `finally { ... }` wrapper and run the statements: Invoke-Expression uses this scope.
$finallyBody = ($finallyText.Trim() -replace '(?s)^\{', '') -replace '(?s)\}$', ''
try {
    try { throw "simulated terminating error mid-install" }
    finally { Invoke-Expression $finallyBody }
} catch { }
Check "temp overrides file removed" (-not (Test-Path -LiteralPath $leakFile))
Check "kept-torch handoff still cleared" ($null -eq $env:UNSLOTH_KEPT_TORCH)
Check "tracked path reset so a rerun cannot re-remove it" ($null -eq $script:TorchOverridesFile)
Remove-Item -LiteralPath $leakFile -Force -ErrorAction SilentlyContinue

Write-Host "and it deletes it in a scope built only from install.ps1's own top-level definitions"
# The replay above runs in THIS scope, which was handed the guard by the Invoke-Expression higher
# up -- a scope production does not have. A fresh runspace inherits nothing, so seeding it with
# only install.ps1's top-level function definitions is what the script itself has when the outer
# finally runs. Without the guard at top level, this leaves the file behind (#11290).
$topLevelDefs = @($ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    -not ($n.Parent.Parent -is [System.Management.Automation.Language.FunctionDefinitionAst])
}, $false) | Where-Object { $_.Name -ne "Install-UnslothStudio" } | ForEach-Object { $_.Extent.Text })
$isolatedLeak = [System.IO.Path]::GetTempFileName()
Set-Content -LiteralPath $isolatedLeak -Encoding ascii -Value "torch==2.11.0+cu128"
$ps = [powershell]::Create()
try {
    $null = $ps.AddScript(@"
`$ErrorActionPreference = 'Continue'
$($topLevelDefs -join "`n")
`$script:WoaResolverEnvSaved = `$null
`$script:WoaSessionOverrides = `$null
`$script:TorchOverridesFile = '$isolatedLeak'
$finallyBody
"@)
    $null = $ps.Invoke()
} finally { $ps.Dispose() }
Check "the file is gone when the finally runs with install.ps1's own scope" (-not (Test-Path -LiteralPath $isolatedLeak))
Remove-Item -LiteralPath $isolatedLeak -Force -ErrorAction SilentlyContinue

Write-Host "an unresolvable tracked path cannot abort the cleanup (#11290)"
# Shadow Remove-Item for the rest of the file. The reported failure is a *terminating* error the
# FileSystem provider raises for a path it cannot resolve, which -ErrorAction cannot suppress, so a
# mock that throws is the only way to reproduce that class on a host whose 8.3 names all resolve.
$script:RemoveItemCalls = @()
function Remove-Item {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $script:RemoveItemCalls += ($Rest -join ' ')
    # The same finally also clears an Env: variable; only the file-system removal is under test.
    if ($Rest -contains "Env:UNSLOTH_KEPT_TORCH") {
        Microsoft.PowerShell.Management\Remove-Item -Path "Env:UNSLOTH_KEPT_TORCH" -ErrorAction SilentlyContinue
        return
    }
    throw "simulated terminating PSArgumentException from FileSystemProvider"
}
$script:TorchOverridesFile = $absent
$script:WoaSessionOverrides = $null
$env:UNSLOTH_KEPT_TORCH = "2.11.0"
$threw = $false
try {
    # The inner catch swallows only the simulation, so the outer one reports the finally body.
    try { throw "simulated terminating error mid-install" }
    catch { }
    finally { Invoke-Expression $finallyBody }
} catch { $threw = $true }
Check "the outer finally does not throw on an unresolvable path" (-not $threw)
Check "Remove-Item is never reached for a path that is not there" (
    -not @($script:RemoveItemCalls | Where-Object { $_ -like "*unsloth-absent-*" }))
Check "the tracked path is still cleared" ($null -eq $script:TorchOverridesFile)

# The catch is what keeps the helper non-fatal, so force the throw with a path that does exist.
$presentFile = [System.IO.Path]::GetTempFileName()
$escaped = $false
try { Remove-UnslothTempFileQuietly -Path $presentFile } catch { $escaped = $true }
Check "a terminating Remove-Item error cannot escape the helper" (-not $escaped)
Check "and the existing path really was attempted" (
    @($script:RemoveItemCalls | Where-Object { $_ -like "*$([System.IO.Path]::GetFileName($presentFile))*" }).Count -ge 1)
[System.IO.File]::Delete($presentFile)

Write-Host "an unreadable alias path cannot abort the install (#11290)"
# install.ps1:3277 documents it: under the installer's "Stop", Test-Path inside an ACL-denied
# directory THROWS UnauthorizedAccessException rather than returning false. None of the 8.3
# guards is inside a try, so a bare Test-Path there turns "reject this alias" into "abort the
# whole install". Every guard added for #11290 must therefore carry -ErrorAction SilentlyContinue.
$guardedPaths = [regex]::Matches($installText, 'Test-Path -LiteralPath \$(short|dirShort)\b[^\)]*\)')
# Three is what #11290 added; the count is not the assertion, the absence of a bare one is.
Check "the 8.3 guards were found" ($guardedPaths.Count -ge 3)
$bareGuards = @($guardedPaths | Where-Object { $_.Value -notmatch '-ErrorAction SilentlyContinue' })
Check "no 8.3 guard asks the filesystem without -ErrorAction SilentlyContinue" ($bareGuards.Count -eq 0)
foreach ($b in $bareGuards) { Write-Host "        bare: $($b.Value)" -ForegroundColor Red }

# Behavioural, on this host: chmod 000 reproduces the same throw class PowerShell raises for a
# Windows deny ACE. Skipped under a uid that bypasses permission bits, which would pass vacuously.
if ((& id -u) -eq "0") {
    Write-Host "  SKIP  running as root: a denied directory is still readable, so this cannot fail"
} else {
    $deniedRoot = Join-Path ([System.IO.Path]::GetTempPath()) "denied_$([guid]::NewGuid().ToString('N'))"
    [System.IO.Directory]::CreateDirectory($deniedRoot) | Out-Null
    $deniedLeaf = Join-Path $deniedRoot "ov.tmp"
    Set-Content -LiteralPath $deniedLeaf -Value "torch==2.10.0"
    & chmod 000 $deniedRoot
    try {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Stop"   # what install.ps1 line 18 sets
        $bareThrew = $false
        try { $null = Test-Path -LiteralPath $deniedLeaf -PathType Leaf } catch { $bareThrew = $true }
        $guardedThrew = $false
        try { $null = Test-Path -LiteralPath $deniedLeaf -PathType Leaf -ErrorAction SilentlyContinue } catch { $guardedThrew = $true }
        $ErrorActionPreference = $prevEap
        # A negative control that can fail: if the bare form stopped throwing on this host, the
        # assertion above is no longer testing anything and must say so rather than pass quietly.
        Check "a bare Test-Path really does throw here (negative control)" $bareThrew
        Check "the guarded form does not throw" (-not $guardedThrew)
    } finally {
        & chmod 755 $deniedRoot
        Remove-Item -LiteralPath $deniedRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "the inherited-override filter drops the torch trio in any casing"
# PowerShell's -notmatch is case-insensitive, so a caller override `Torch<2.11` is dropped.
$filterPattern = $null
# The merge folds entry by entry, so the trio filter reads `-match ... { continue }` instead.
if ($fnText -match '\$ovEntry\.Line -match ''([^'']+)''') { $filterPattern = $Matches[1] }
Check "filter pattern extracted from the helper" ($null -ne $filterPattern)
$inherited = @(
    "# comment survives",
    "Torch<2.11",
    "TORCHVISION>=0.19",
    "TorchAudio==2.1",
    "torch<2.11.0",
    "torchvision==0.25.0",
    "torchaudio!=2.11.0",
    "torchmetrics==1.0",
    "transformers>=4.57.6",
    "anyio<4.14.0"
)
$merged = @("torch==2.11.0+cu128") + @($inherited | Where-Object { $_ -notmatch $filterPattern })
$trio = @($merged | Where-Object { $_ -match '^(torch|torchvision|torchaudio)([\s<>=!~;@[]|$)' })
Check "exactly one trio requirement survives (the generated pin)" ($trio.Count -eq 1)
Check "the survivor is the generated exact pin" ($trio[0] -eq "torch==2.11.0+cu128")
foreach ($keep in @("torchmetrics==1.0", "transformers>=4.57.6", "anyio<4.14.0", "# comment survives")) {
    Check "unrelated inherited override preserved: $keep" ($merged -contains $keep)
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures check(s)" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed."
