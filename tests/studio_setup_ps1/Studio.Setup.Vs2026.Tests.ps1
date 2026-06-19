<#
    Pester v5 unit tests for the Visual Studio 2026 completion helpers in
    studio/setup.ps1:
      - Get-VcBuildCustomizationsDir : derive the VC MSBuild BuildCustomizations
        folder (v160 / v170 / v180) from the detected VS generator.
      - Test-CmakeSupportsGenerator  : gate the "Visual Studio 18 2026" generator
        on CMake >= 4.2 (no-op for older VS generators).

    Both are pure functions (no GPU, no Visual Studio, no CUDA, no network), so the
    suite runs on a stock windows-latest runner - and on any pwsh host.

    The real functions are extracted from setup.ps1 and dot-sourced (the script is
    a top-level installer and cannot be loaded wholesale). Path resolution honors
    $env:SETUP_PS1_PATH (set by the PR-validate workflow) and falls back to the
    repo-relative path. If a target function cannot be found, the suite FAILS
    loudly rather than silently passing.
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

    foreach ($fn in @('Find-VsBuildTools', 'Get-VcBuildCustomizationsDir', 'Test-CmakeSupportsGenerator')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }
}

Describe 'Find-VsBuildTools (VS 2026 generator discovery)' {
    # Exercises the production discovery entry point (not just the downstream
    # helpers) so the v180 path + CMake guard cannot be reachable in routing while
    # discovery stays dead. Windows-only: Find-VsBuildTools builds candidate paths
    # with backslash separators, which only resolve as real directories on Windows.
    BeforeEach {
        $script:OrigPF    = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles}      = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
    }

    function New-FakeVsTree {
        param([string]$Root, [string]$VersionDir, [string]$Edition = 'BuildTools')
        $clDir = Join-Path $Root "Microsoft Visual Studio\$VersionDir\$Edition\VC\Tools\MSVC\14.50.00000\bin\Hostx64\x64"
        New-Item -ItemType Directory -Path $clDir -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $clDir 'cl.exe') -Force | Out-Null
    }

    It 'detects a filesystem-only VS 2026 BuildTools install (dir "18")' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF'
        New-FakeVsTree -Root $root -VersionDir '18'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86'   # no vswhere here -> filesystem fallback
        $r = Find-VsBuildTools
        $r.Generator | Should -Be 'Visual Studio 18 2026'
    }

    It 'detects a filesystem-only VS 2026 install under the year dir ("2026")' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF2026'
        New-FakeVsTree -Root $root -VersionDir '2026'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86b'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 18 2026'
    }

    It 'still detects VS 2022 (no regression)' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF2022'
        New-FakeVsTree -Root $root -VersionDir '2022'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86c'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Get-VcBuildCustomizationsDir (CUDA to VS MSBuild integration path)' {
    # Use Pester's real TestDrive as the install root so Join-Path resolves on
    # any OS (a fake 'C:\' base errors on non-Windows hosts). Assertions match the
    # toolset segment with either path separator, so they hold on Windows + Linux.

    It 'derives v180 for the VS 2026 generator' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 18 2026' |
            Should -Match 'VC[\\/]v180[\\/]BuildCustomizations$'
    }

    It 'derives v170 for VS 2022 (unchanged behavior)' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 17 2022' |
            Should -Match 'VC[\\/]v170[\\/]BuildCustomizations$'
    }

    It 'derives v160 for VS 2019' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 16 2019' |
            Should -Match 'VC[\\/]v160[\\/]BuildCustomizations$'
    }

    It 'falls back to v170 when the generator is empty/unparseable (backwards compatible)' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator '' |
            Should -Match 'VC[\\/]v170[\\/]BuildCustomizations$'
    }

    It 'roots the path under the supplied VS install path' {
        $p = Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 18 2026'
        $p.StartsWith("$TestDrive") | Should -BeTrue
    }
}

Describe 'Test-CmakeSupportsGenerator (CMake 4.2 guard for VS 2026)' {

    It 'rejects CMake 3.31.0 with the VS 2026 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.31.0' -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'accepts CMake 4.2.1 with the VS 2026 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.2.1' -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'accepts CMake exactly 4.2 with the VS 2026 generator (boundary)' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.2' -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'rejects CMake 4.1.0 with the VS 2026 generator (boundary)' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.1.0' -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'is a no-op (accepts any CMake) for the VS 2022 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.20.0' -Generator 'Visual Studio 17 2022' | Should -BeTrue
    }

    It 'is a no-op (accepts any CMake) for the VS 2019 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.10.0' -Generator 'Visual Studio 16 2019' | Should -BeTrue
    }
}
