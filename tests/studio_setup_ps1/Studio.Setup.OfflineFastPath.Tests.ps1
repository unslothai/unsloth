# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    `unsloth studio update` treats an unreachable PyPI as a reason to update anyway,
    which is right for a blip. It is wrong when the caller SET UV_OFFLINE: uv then
    refuses to reach a network at all, so the pass that follows can only fail, and an
    install that was already complete ends the update broken-looking and non-zero.

    Test-UvOfflineRequested is what tells those two apart, so it has to read the same
    boolish spellings the POSIX half does -- a user who wrote UV_OFFLINE=true on one
    platform means the same thing on the other.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }

    $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-UvOfflineRequested'
    if (-not $src) { throw "Test-UvOfflineRequested is gone from setup.ps1." }
    . ([scriptblock]::Create($src))

    $script:SetupText = Get-Content -Raw -LiteralPath $script:SetupPs1
    $script:SavedOffline = $env:UV_OFFLINE
}

AfterAll {
    if ($null -eq $script:SavedOffline) {
        Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
    } else {
        $env:UV_OFFLINE = $script:SavedOffline
    }
}

Describe 'Test-UvOfflineRequested' {
    It 'accepts <value>' -ForEach @(
        @{ value = '1' }, @{ value = 'true' }, @{ value = 'TRUE' }, @{ value = 'True' },
        @{ value = 'yes' }, @{ value = 'on' }, @{ value = '  true  ' }
    ) {
        $env:UV_OFFLINE = $value
        Test-UvOfflineRequested | Should -BeTrue
    }

    It 'rejects <value>' -ForEach @(
        @{ value = '0' }, @{ value = 'false' }, @{ value = '' }, @{ value = 'maybe' },
        @{ value = 'offline' }
    ) {
        $env:UV_OFFLINE = $value
        Test-UvOfflineRequested | Should -BeFalse
    }

    It 'reads an unset variable as not offline' {
        Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
        Test-UvOfflineRequested | Should -BeFalse
    }
}

Describe 'the offline skip' {
    It 'is gated on an installed version, offline mode and a verified tree' {
        $script:SetupText | Should -Match ([regex]::Escape(
            'if ($InstalledVer -and (Test-UvOfflineRequested) -and (Test-StudioInstallVerified)) {'))
    }

    It 'still updates to be safe when any condition fails' {
        # The else branch is the whole default behaviour; losing it turns every
        # unreachable-PyPI update into a silent skip.
        $script:SetupText | Should -Match 'could not reach PyPI, updating to be safe'
    }

    It 'asks the same question the incomplete-install guard asks' {
        ([regex]::Matches($script:SetupText, [regex]::Escape('verify_install(deep = True)'))).Count |
            Should -Be 1
    }
}

<#
    Keeping a verified install is only half the rule. A tree can verify and still be
    below the backend version the desktop app requires, hold the anyio/tokenizers damage,
    or sit on a CPU wheel next to an Intel or AMD GPU -- and only the dependency pass
    repairs any of that. The up-to-date branch has always escaped on those; the offline
    branch used to skip past every one of them, so an offline `unsloth studio update` of a
    below-floor install reported success and repaired nothing.

    Both branches now reach them through Invoke-FastPathEscapes. This runs the real
    offline branch and the real helper, sliced out of setup.ps1 into a standalone driver
    script -- a script, not a dot-sourced scriptblock, because the helper writes
    $script:SkipPythonDeps and the scope it means has to be a real script scope.
#>
Describe 'the offline skip takes the shared escapes' {
    BeforeAll {
        $escapeSrc = Get-FunctionSource -Path $script:SetupPs1 -Name 'Invoke-FastPathEscapes'
        if (-not $escapeSrc) { throw "Invoke-FastPathEscapes is gone from setup.ps1." }
        if ($escapeSrc -notmatch 'UNSLOTH_DESKTOP_BACKEND_VERSION') {
            throw "Invoke-FastPathEscapes no longer consults UNSLOTH_DESKTOP_BACKEND_VERSION."
        }
        $offlineSrc = Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-UvOfflineRequested'

        # The offline arm itself, from its elseif to the `}` that closes the whole chain.
        # Sliced rather than restated: a branch that stopped calling the helper has to make
        # this file fail, and a copy of the branch here could not.
        $lines = $script:SetupText -split "`r?`n"
        $start = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^\s*\}\s*elseif \(-not \$LatestVer\) \{\s*$') { $start = $i; break }
        }
        if ($start -lt 0) { throw "the '-not `$LatestVer' arm is gone from setup.ps1." }
        $end = -1
        for ($i = $start + 1; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -eq '    }') { $end = $i; break }
        }
        if ($end -lt 0) { throw "could not find the end of the '-not `$LatestVer' arm." }
        $branch = $lines[($start + 1)..($end - 1)] -join "`n"
        # A slice that lost either half would let every case below pass vacuously.
        if ($branch -notmatch 'Invoke-FastPathEscapes') {
            throw ("the offline arm no longer calls Invoke-FastPathEscapes -- an offline skip " +
                   "that ducks the shared escapes reports success and repairs nothing.")
        }
        if ($branch -notmatch 'could not reach PyPI') {
            throw "the offline arm lost its updating-to-be-safe default."
        }

        $script:DriverDir = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-offline-" + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $script:DriverDir -Force | Out-Null
        $script:DriverPath = Join-Path $script:DriverDir 'driver.ps1'

        $prelude = @'
param(
    [AllowEmptyString()][string]$InstalledVer,
    [AllowEmptyString()][string]$UvOffline,
    [AllowEmptyString()][string]$DesktopVer,
    [bool]$Verified
)
$env:UV_OFFLINE = $UvOffline
$env:UNSLOTH_DESKTOP_BACKEND_VERSION = $DesktopVer
$LatestVer = ""
$SkipPythonDeps = $false
$installedTorchTag = "cu128"
$NoTorchMode = $false
$VenvDir = Join-Path $PSScriptRoot "venv"
$_PkgName = "unsloth"
$script:IsIntelXpu = $false
$script:ROCmGfxArch = $null
$script:Substeps = @()

function step { param([string]$a, [string]$b, [string]$c) }
function substep { param([string]$Message, [string]$Color = "DarkGray") $script:Substeps += $Message }
function Test-StudioInstallVerified { return $Verified }
function Get-PinnedTorchIndexUrl { return "" }
function Get-TorchIndexLeaf { param([AllowNull()][string]$Url) return "" }
function Invoke-BoundedPythonProbe {
    param([string]$PythonExe, [string]$Code, [int]$TimeoutSec = 30)
    return [pscustomobject]@{ Ok = $true; Output = ""; Error = ""; TimedOut = $false }
}
function Test-VenvTorchIsXpuSupported { param([string]$VenvPath) return $true }
function python {
    # Not the real interpreter. The escapes ask three fixed questions and read the answer
    # out of $LASTEXITCODE, so this answers them directly; a function call does not set
    # $LASTEXITCODE by itself, hence the explicit $global: writes.
    $code = [string]$args[1]
    if ($code -match 'anyio') { $global:LASTEXITCODE = 1; return }
    if ($code -match 'tokenizers') { $global:LASTEXITCODE = 1; return }
    if ($code -match 'parse_v') {
        # Mirrors the interpreter's own no-packaging fallback: anything that is not a plain
        # three-part version cannot be ordered, and unorderable forces the dependency pass.
        $installed = [string]$args[2]
        $required = [string]$args[3]
        $plain = '^\d+\.\d+\.\d+$'
        if ($installed -notmatch $plain -or $required -notmatch $plain) {
            $global:LASTEXITCODE = 1
            return
        }
        $global:LASTEXITCODE = if ([version]$installed -ge [version]$required) { 0 } else { 1 }
        return
    }
    $global:LASTEXITCODE = 0
}
'@
        $epilogue = @'

"SKIP=$SkipPythonDeps"
"SUBSTEPS=" + ($script:Substeps -join '|')
'@
        Set-Content -LiteralPath $script:DriverPath -Encoding utf8 -Value (
            $prelude + "`n" + $offlineSrc + "`n" + $escapeSrc + "`n`nif (`$true) {`n" +
            $branch + "`n}`n" + $epilogue)

        function script:Invoke-OfflineBranch {
            param([string]$InstalledVer, [string]$UvOffline, [string]$DesktopVer, [bool]$Verified = $true)
            $out = & $script:DriverPath -InstalledVer $InstalledVer -UvOffline $UvOffline `
                -DesktopVer $DesktopVer -Verified $Verified
            $text = ($out | Out-String)
            $skip = if ($text -match '(?m)^SKIP=(\S+)\s*$') { $Matches[1] } else { "<no SKIP line>" }
            $steps = if ($text -match '(?m)^SUBSTEPS=(.*)$') { $Matches[1] } else { "" }
            # The branch has exactly two outcomes and both announce themselves; neither
            # appearing means the slice never ran and every assertion below is vacuous.
            if ($steps -notmatch 'keeping the verified install' -and $steps -notmatch 'could not reach PyPI') {
                throw "the offline arm did not run (substeps: '$steps', output: '$text')"
            }
            return $skip
        }
    }

    AfterAll {
        # Guarded: when the slice checks above throw, BeforeAll never got as far as naming a
        # driver directory, and an unguarded Remove-Item here would bury that message under a
        # parameter-binding error from this block.
        if ($script:DriverDir) {
            Remove-Item -Recurse -Force -LiteralPath $script:DriverDir -ErrorAction SilentlyContinue
        }
        Remove-Item Env:UNSLOTH_DESKTOP_BACKEND_VERSION -ErrorAction SilentlyContinue
    }

    It 'keeps a verified install when nothing else is owed' {
        Invoke-OfflineBranch -InstalledVer '2026.8.15' -UvOffline '1' -DesktopVer '' |
            Should -Be 'True'
    }

    It 'keeps it when the install already meets the desktop backend floor' {
        Invoke-OfflineBranch -InstalledVer '2026.8.15' -UvOffline '1' -DesktopVer '2026.8.15' |
            Should -Be 'True'
    }

    It 'forces the dependency pass for an install BELOW the desktop backend floor' {
        # The regression this exists for: verified, offline, and still too old for the
        # desktop app that launched it. Only the dependency pass raises it.
        Invoke-OfflineBranch -InstalledVer '2026.8.4' -UvOffline '1' -DesktopVer '2026.8.15' |
            Should -Be 'False'
    }

    It 'forces the dependency pass for a requirement it cannot order' {
        Invoke-OfflineBranch -InstalledVer '2026.8.15' -UvOffline '1' -DesktopVer '2026.8.15.post1' |
            Should -Be 'False'
    }

    It 'still updates to be safe when UV_OFFLINE is not set' {
        Invoke-OfflineBranch -InstalledVer '2026.8.15' -UvOffline '' -DesktopVer '' |
            Should -Be 'False'
    }

    It 'still updates to be safe when the tree does not verify' {
        Invoke-OfflineBranch -InstalledVer '2026.8.15' -UvOffline '1' -DesktopVer '' -Verified $false |
            Should -Be 'False'
    }

    It 'still updates to be safe when nothing is installed' {
        Invoke-OfflineBranch -InstalledVer '' -UvOffline '1' -DesktopVer '' | Should -Be 'False'
    }
}
