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
