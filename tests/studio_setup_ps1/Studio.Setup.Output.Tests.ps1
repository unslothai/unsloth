<#
    Pester v5 unit tests for step / substep / Write-StudioStdoutMirror in
    studio/setup.ps1, guarding the desktop setup log printing every step twice
    with the first copy split across two lines:

        gpu
      none (chat-only / GGUF)
        gpu            none (chat-only / GGUF)

    Two causes. step/substep called Write-Host AND the console mirror, and the
    CLI spawns setup.ps1 as `-Command "& '...' *>&1"`, so both reached the pipe.
    And step's non-VT branch built one line from two Write-Host calls with
    -NoNewline, which a redirected consumer splits at the record boundary.

    Invariant now: exactly ONE sink. Redirected -> console handle. Interactive
    -> Write-Host.

    Pure string formatting, so it runs on any pwsh host. Functions are extracted
    and dot-sourced because setup.ps1 is a top-level installer; a missing one
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

    $script:InstallPs1 = Join-Path $PSScriptRoot '..\..\install.ps1'

    foreach ($fn in @('Get-StudioAnsi', 'Write-StudioStdoutMirror', 'step', 'substep')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }

    # Capture the console-handle sink without a real pipe, so the redirected
    # path can be asserted from an ordinary interactive test host.
    function Invoke-CapturingConsoleOut {
        param(
            [Parameter(Mandatory = $true)][scriptblock]$Body,
            [Parameter(Mandatory = $true)][bool]$Redirected
        )
        $script:StudioStdoutRedirected = $Redirected
        $previous = [Console]::Out
        $writer = New-Object System.IO.StringWriter
        try {
            [Console]::SetOut($writer)
            # 6>&1 folds Write-Host into the pipeline so a stray one on the
            # redirected path is caught, not swallowed by the test host.
            $hostRecords = & $Body 6>&1
        } finally {
            [Console]::SetOut($previous)
        }
        [pscustomobject]@{
            Console = $writer.ToString()
            HostRecordCount = @($hostRecords).Count
        }
    }

    # Split on the real line separator only: the split-label bug produced a
    # genuine newline, not a CR redraw.
    #
    # The leading comma is load-bearing. `return @($x)` unrolls a one-element
    # array to a scalar, and a scalar string answers .Count = 1 while [0] gives
    # its first CHARACTER, so a "one line, and it reads X" test would pass the
    # count then compare against a single space.
    function Get-EmittedLines {
        param([string]$Text)
        if ([string]::IsNullOrEmpty($Text)) { return , @() }
        return , @($Text -split "`r?`n" | Where-Object { $_ -ne '' })
    }

    # Strip comments before asserting a construct is absent: the scripts under
    # test describe -NoNewline in prose, which a naive match would hit.
    function Get-CodeWithoutComments {
        param([string]$Source)
        $stripped = $Source -replace '(?m)#.*$', ''
        return $stripped
    }
}

Describe 'step / substep emit exactly one copy when stdout is redirected' {
    BeforeEach {
        $script:StudioVtOk = $false
        $env:NO_COLOR = $null
    }

    It 'emits a step once, with label and value on the SAME line' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            step "gpu" "none (chat-only / GGUF)"
        }
        $lines = Get-EmittedLines $r.Console
        $lines.Count | Should -Be 1
        $lines[0] | Should -Be "  gpu            none (chat-only / GGUF)"
    }

    It 'emits a substep once' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            substep "installing OXC validator runtime..."
        }
        (Get-EmittedLines $r.Console).Count | Should -Be 1
    }

    It 'writes NOTHING through Write-Host when redirected (the second copy)' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            step "long paths" "enabled"
            substep "detail"
        }
        $r.HostRecordCount | Should -Be 0
    }

    It 'keeps one line per step across a realistic run' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            step "gpu" "none (chat-only / GGUF)"
            step "long paths" "enabled"
            step "git" "git version 2.53.0.windows.2"
            substep "setting up Python environment..."
        }
        (Get-EmittedLines $r.Console).Count | Should -Be 4
    }

    It 'truncates an over-long label to the 15-column field without wrapping' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            step "an-extremely-long-label" "value"
        }
        $lines = Get-EmittedLines $r.Console
        $lines.Count | Should -Be 1
        $lines[0] | Should -Be "  an-extremely-lovalue"
    }
}

Describe 'step / substep stay on Write-Host when attached to a console' {
    BeforeEach {
        $script:StudioVtOk = $false
        $env:NO_COLOR = $null
    }

    It 'writes nothing to the console handle when NOT redirected' {
        $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
            step "gpu" "none (chat-only / GGUF)"
        }
        $r.Console | Should -BeNullOrEmpty
    }

    It 'emits a step as a SINGLE Write-Host record (not -NoNewline label + value)' {
        $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
            step "gpu" "none (chat-only / GGUF)"
        }
        $r.HostRecordCount | Should -Be 1
    }

    It 'emits a substep as a single Write-Host record' {
        $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
            substep "detail"
        }
        $r.HostRecordCount | Should -Be 1
    }

    It 'still emits one record on the ANSI path' {
        $script:StudioVtOk = $true
        $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
            step "gpu" "ok"
            substep "detail"
        }
        $r.HostRecordCount | Should -Be 2
    }

    It 'still emits one record with NO_COLOR set' {
        $script:StudioVtOk = $true
        $env:NO_COLOR = '1'
        try {
            $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
                step "gpu" "ok"
            }
            $r.HostRecordCount | Should -Be 1
        } finally {
            $env:NO_COLOR = $null
        }
    }
}

Describe 'Write-StudioStdoutMirror honors the resolved sink' {
    It 'writes when the sink says redirected' {
        $r = Invoke-CapturingConsoleOut -Redirected $true -Body {
            Write-StudioStdoutMirror "hello"
        }
        (Get-EmittedLines $r.Console) | Should -Be @('hello')
    }

    It 'stays silent when the sink says interactive' {
        $r = Invoke-CapturingConsoleOut -Redirected $false -Body {
            Write-StudioStdoutMirror "hello"
        }
        $r.Console | Should -BeNullOrEmpty
    }
}

Describe 'Source contracts that keep the fix from regressing' {
    It 'resolves the redirected sink once, into a script-scoped variable' {
        $source = Get-Content -Raw -LiteralPath $script:SetupPs1
        $source | Should -Match '\$script:StudioStdoutRedirected\s*=\s*\[Console\]::IsOutputRedirected'
    }

    It 'no longer calls IsOutputRedirected from inside the mirror' {
        # Reading it per-call let the sink disagree with the branch
        # step/substep took; it must be resolved once, up front.
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Write-StudioStdoutMirror'
        $src | Should -Not -Match 'IsOutputRedirected'
    }

    It 'does not use -NoNewline in setup.ps1 step (record boundaries become newlines)' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'step'
        (Get-CodeWithoutComments $src) | Should -Not -Match '-NoNewline'
    }

    It 'does not use -NoNewline in install.ps1 step either (it has no mirror)' {
        $src = Get-FunctionSource -Path $script:InstallPs1 -Name 'step'
        $src | Should -Not -BeNullOrEmpty
        (Get-CodeWithoutComments $src) | Should -Not -Match '-NoNewline'
    }

    It 'sets the UTF-8 console encoding in both entry scripts' {
        foreach ($path in @($script:SetupPs1, $script:InstallPs1)) {
            $source = Get-Content -Raw -LiteralPath $path
            $source | Should -Match '\[Console\]::OutputEncoding\s*=\s*\$_UnslothUtf8NoBom'
            $source | Should -Match "PYTHONUTF8"
            $source | Should -Match "PYTHONIOENCODING"
        }
    }
}
