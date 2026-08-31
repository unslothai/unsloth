# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    Pester v5 unit tests for the short-flag trap in setup.ps1's Fast-Install /
    Fast-Uninstall / Fast-Download wrappers, guarding the Intel XPU triton swap
    silently doing nothing (#10018):

        [WARN] could not fetch triton-xpu==3.6.0 (exit 1); triton-windows 3.8.0.post28
               left in place -- it still shadows torch XPU triton
        ERROR: Directory 'C:\...\Temp\unsloth_triton_xpu_0da4251b' is not installable.
               Neither 'setup.py' nor 'pyproject.toml' found.

    All three wrappers declare `param([Parameter(ValueFromRemainingArguments=$true)]$Args_)`.
    A [Parameter()] attribute makes a function ADVANCED, so PowerShell binds the
    common parameters BEFORE collecting remaining arguments -- and `-d`, pip's own
    spelling of `--dest`, is a unique prefix of `-Debug`. It binds there and never
    reaches $Args_, so `pip download` reads the destination directory as a positional
    requirement and dies before it ever queries the index. The reported symptom is a
    package that looks missing; the cause is a flag that was eaten.

    That is why the diagnosis in #10018 (wrong package name) does not hold:
    $_tritonXpuSpec is read from the installed torch's own `requires('torch')`
    metadata, and tests/studio/test_xpu_triton_swap.py already covers the
    pytorch-triton-xpu -> triton-xpu rename.

    Long flags are safe: `--dest` is not a parameter-name token, so it passes
    through untouched. The invariant is therefore a source-level one -- no Fast-*
    call site may pass a short flag that prefixes a common parameter -- because the
    defect lives at the call site, not inside the wrapper.

    Pure string and argument-binding behaviour: no GPU, no Intel Arc, no XPU index
    and no network, so it runs on the same stock runner as the rest of this folder.

    The real wrapper is extracted from setup.ps1 and dot-sourced (the script is a
    top-level installer and cannot be loaded wholesale). Path resolution honors
    $env:SETUP_PS1_PATH and falls back to the repo-relative path; a missing function
    FAILS loudly rather than silently passing.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }
    Write-Host "setup.ps1 under test: $script:SetupPs1"

    $script:SetupText = Get-Content -Raw -LiteralPath $script:SetupPs1

    # Derived, not hard-coded: the binder compares against whatever this host's
    # [CommonParameters] actually declares, so the guard cannot drift from the rule
    # it is enforcing. SupportsShouldProcess is not set on any wrapper, so -WhatIf
    # and -Confirm are deliberately out of scope.
    $script:CommonParameterNames = [System.Management.Automation.Internal.CommonParameters].GetProperties().Name

    # Lines that INVOKE a wrapper. The `function Fast-* {` declarations and comment
    # lines are not call sites and would otherwise be scanned for flags they cannot have.
    $script:WrapperCallLines = @(
        ($script:SetupText -split "`r?`n") |
            ForEach-Object -Begin { $n = 0 } -Process {
                $n++
                [pscustomobject]@{ Number = $n; Text = $_ }
            } |
            Where-Object {
                $_.Text -match 'Fast-(Install|Uninstall|Download)\b' -and
                $_.Text -notmatch '^\s*#' -and
                $_.Text -notmatch '^\s*function\s+Fast-'
            }
    )

    function Test-BindsAsCommonParameter {
        param([Parameter(Mandatory)][string]$Token)
        # A single-dash token binds as a common parameter when it prefixes one, whether
        # uniquely (-d -> -Debug, bound and swallowed) or ambiguously (-e -> ErrorAction
        # / ErrorVariable, a loud bind error). Both break the call; neither is acceptable.
        $bare = $Token.TrimStart('-')
        if (-not $bare) { return $false }
        return [bool](@($script:CommonParameterNames | Where-Object {
            $_.StartsWith($bare, [System.StringComparison]::OrdinalIgnoreCase)
        }).Count)
    }
}

Describe 'Fast-* wrappers are advanced functions (why short flags are unsafe)' {
    It 'declares ValueFromRemainingArguments on all three wrappers' {
        foreach ($fn in @('Fast-Install', 'Fast-Uninstall', 'Fast-Download')) {
            $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
            $src | Should -Not -BeNullOrEmpty -Because "$fn must exist to be tested"
            $src | Should -Match 'ValueFromRemainingArguments' -Because "$fn forwards its arguments verbatim"
        }
    }

    It 'swallows -d as -Debug instead of forwarding it' {
        # The mechanism itself, on a stand-in with the same param block. This is the
        # reason the call sites must use long flags; it is PowerShell behaviour and
        # holds with or without the fix.
        function script:Probe-Args { param([Parameter(ValueFromRemainingArguments=$true)]$Args_) return , @($Args_) }

        $withShort = Probe-Args --no-deps -d 'C:\tmp' 'pkg==1.0'
        $withShort | Should -Not -Contain '-d' -Because '-d binds to the common parameter -Debug'
        $withShort.Count | Should -Be 3

        $withLong = Probe-Args --no-deps --dest 'C:\tmp' 'pkg==1.0'
        $withLong | Should -Contain '--dest' -Because 'a long flag is not a parameter-name token'
        $withLong.Count | Should -Be 4
    }
}

Describe 'No Fast-* call site passes a short flag the binder would capture (#10018)' {
    It 'finds the wrapper call sites at all' {
        # Guards the guard: a rename or a refactor that stops this scan matching must
        # fail here rather than turn every assertion below into a vacuous pass.
        $script:WrapperCallLines.Count | Should -BeGreaterThan 10
    }

    It 'passes no single-dash flag that prefixes a common parameter' {
        $offenders = foreach ($line in $script:WrapperCallLines) {
            # Single-dash tokens only. `--no-deps` and `--only-binary=:all:` are long
            # flags; `-Filter`/`-LiteralPath` etc. belong to piped cmdlets on the same
            # line, so the scan starts at the wrapper name.
            $call = $line.Text.Substring($line.Text.IndexOf('Fast-'))
            foreach ($m in [regex]::Matches($call, '(?<![\w-])-(?<name>[A-Za-z]{1,2})(?=\s|$)')) {
                $token = "-$($m.Groups['name'].Value)"
                if (Test-BindsAsCommonParameter -Token $token) {
                    "setup.ps1:$($line.Number): '$token' binds as a common parameter -- use the long flag"
                }
            }
        }
        # -join so a failure names every offending line instead of just a count.
        ($offenders -join "`n") | Should -BeNullOrEmpty
    }

    It 'fetches the XPU triton wheel with --dest' {
        # The specific regression: both the verbose and the quiet branch of the swap.
        $destCalls = @($script:WrapperCallLines | Where-Object { $_.Text -match 'Fast-Download' -and $_.Text -match '--dest' })
        $destCalls.Count | Should -Be 2 -Because 'the swap fetches through both the verbose and the quiet branch'
        foreach ($line in $destCalls) {
            $line.Text | Should -Match '--dest\s+\$_tritonTmp' -Because 'the wheel must land in the staged temp dir'
        }
    }
}

Describe 'Fast-Download forwards the destination through to pip' {
    BeforeAll {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Fast-Download'
        if (-not $src) { throw "Function 'Fast-Download' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }

    It 'reaches pip with --dest and the directory adjacent' {
        # A function named `python` outranks the external command, so the real wrapper
        # runs unmodified and records what it would have executed.
        function script:python { $script:PipArgs = @($args); return '' }
        $script:PipArgs = @()

        Fast-Download --no-deps --only-binary=:all: --dest 'C:\tmp\stage' 'triton-xpu==3.6.0' --index-url 'https://example.invalid/xpu' | Out-Null

        $script:PipArgs | Should -Contain '--dest'
        $idx = [array]::IndexOf($script:PipArgs, '--dest')
        $script:PipArgs[$idx + 1] | Should -Be 'C:\tmp\stage' -Because "pip reads the directory as the value of --dest"
        # The spec must NOT be the argument pip treats as the download directory.
        $script:PipArgs[0..2] -join ' ' | Should -Be '-m pip download'
    }

    It 'loses the directory when the short flag is used (the #10018 failure)' {
        function script:python { $script:PipArgs = @($args); return '' }
        $script:PipArgs = @()

        Fast-Download --no-deps --only-binary=:all: -d 'C:\tmp\stage' 'triton-xpu==3.6.0' --index-url 'https://example.invalid/xpu' | Out-Null

        $script:PipArgs | Should -Not -Contain '-d' -Because 'the binder took it as -Debug'
        # What pip actually receives: the directory as a bare positional requirement,
        # which is the "Directory ... is not installable" error in the report.
        $script:PipArgs | Should -Contain 'C:\tmp\stage'
        [array]::IndexOf($script:PipArgs, 'C:\tmp\stage') | Should -BeGreaterThan 2
    }
}
