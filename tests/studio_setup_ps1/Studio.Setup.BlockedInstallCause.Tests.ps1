# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    Pester v5 unit tests for the "install blocked" (exit 3) reporting in
    studio/setup.ps1.

    install_llama_prebuilt.py exits 3 when the existing install could not be
    moved aside. On Windows that is WinError 5, which the OS raises both for a
    file another process holds open and for a tree whose ACLs are unreadable --
    is_busy_lock_error (studio/prebuilt_core.py) classifies 5 as busy for that
    reason, and _access_denied_recovery_lines prints repair commands for the ACL
    case. Both setup.ps1 sites asserted the process cause anyway, so the ACL
    commands from Write-LlamaFailureLog arrived bracketed by advice that
    contradicted them (#9928).

    Two sites, because the local-directory link path says in its own comment
    that it mirrors the prebuilt path, and that "Denied counts as surviving:
    unreadable is not gone" -- so it reaches the same message by the same route.

    Source scan rather than execution: setup.ps1 is a top-level installer and
    these branches only run after a genuinely blocked install.
#>

BeforeAll {
    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }
    $script:SetupText = Get-Content -Raw -LiteralPath $script:SetupPs1

    # These are inline installer flow, not functions, so Get-FunctionSource cannot
    # reach them; slice from the marker to the first close at the marker's depth.
    function Get-BlockSource {
        param(
            [Parameter(Mandatory)][string]$Path,
            [Parameter(Mandatory)][string]$Marker,
            [Parameter(Mandatory)][string]$ClosePattern
        )
        $lines = @(Get-Content -LiteralPath $Path)
        $start = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match [regex]::Escape($Marker)) { $start = $i; break }
        }
        if ($start -lt 0) { return $null }
        $indent = ($lines[$start] -replace '\S.*$', '').Length
        for ($j = $start + 1; $j -lt $lines.Count; $j++) {
            $depth = ($lines[$j] -replace '\S.*$', '').Length
            if ($lines[$j] -match $ClosePattern -and $depth -le $indent) {
                return ($lines[$start..($j - 1)] -join "`n")
            }
        }
        return $null
    }

    $script:LlamaBusy = Get-BlockSource -Path $script:SetupPs1 `
        -Marker '$prebuiltExit -eq 3' -ClosePattern '^\s*\}\s*(elseif|else)\b'
    $script:WhisperBusy = Get-BlockSource -Path $script:SetupPs1 `
        -Marker '$whisperExit -eq 3' -ClosePattern '^\s*\}\s*(elseif|else)\b'
    $script:LocalLinkBusy = Get-BlockSource -Path $script:SetupPs1 `
        -Marker '(Get-PathState -Path $LlamaCppDir) -ne "Absent"' -ClosePattern '^\s*\}\s*$'
}

Describe 'no llama.cpp blocked-install message names a cause setup cannot determine' {
    It 'the process headline appears nowhere in setup.ps1' {
        $script:SetupText | Should -Not -Match 'install blocked by active llama\.cpp process'
    }

    It 'the process failure message appears nowhere in setup.ps1' {
        $script:SetupText | Should -Not -Match 'blocked by an active llama\.cpp process'
    }

    It 'every blocked-install site offers ACL repair' {
        ([regex]::Matches($script:SetupText, 'repair the ACLs')).Count |
            Should -BeGreaterOrEqual 2
    }
}

Describe 'the prebuilt exit-3 branch' {
    It 'is present in setup.ps1' {
        $script:LlamaBusy | Should -Not -BeNullOrEmpty
    }

    It 'offers ACL repair alongside closing other users' {
        $script:LlamaBusy | Should -Match 'repair the ACLs'
    }

    It 'only names ACL repair when the helper emitted recovery guidance' {
        $script:LlamaBusy | Should -Match '\$prebuiltOutput -match [''"]takeown /F[''"]'
    }

    It 'keeps a close-users-only fallback without recovery guidance' {
        $script:LlamaBusy | Should -Match '(?s)else\s*\{.+Close Unsloth or other llama\.cpp users and retry'
    }

    It 'still terminates with exit code 3' {
        $script:LlamaBusy | Should -Match 'Exit-SetupFailure .+ 3'
    }

    It 'still prints the installer output carrying the takeown/icacls commands' {
        $script:LlamaBusy | Should -Match 'Write-LlamaFailureLog -Output \$prebuiltOutput'
    }

    It 'still reports that the previous install was restored' {
        $script:LlamaBusy | Should -Match 'Existing install was restored'
    }

    It 'still tells the user to close other llama.cpp users' {
        $script:LlamaBusy | Should -Match 'Close Unsloth or other llama\.cpp users'
    }
}

Describe 'the local-directory link path mirrors it, as its comment claims' {
    It 'is present in setup.ps1' {
        $script:LocalLinkBusy | Should -Not -BeNullOrEmpty
    }

    It 'offers ACL repair on the install directory' {
        $script:LocalLinkBusy | Should -Match 'repair the ACLs on \$LlamaCppDir'
    }

    It 'still terminates with exit code 3' {
        $script:LocalLinkBusy | Should -Match 'Exit-SetupFailure .+ 3'
    }

    It 'still tells the user to close other llama.cpp users' {
        $script:LocalLinkBusy | Should -Match 'Close Unsloth or other llama\.cpp users'
    }
}

Describe 'the whisper.cpp exit-3 branch remains the cause-neutral precedent' {
    It 'is present in setup.ps1' {
        $script:WhisperBusy | Should -Not -BeNullOrEmpty
    }

    It 'names no process and no scanner' {
        $script:WhisperBusy | Should -Not -Match '(?i)process|scanner'
    }
}
