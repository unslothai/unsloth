<#
    Pester tests for setup.ps1's Visual Studio 2026 helpers (Get-VcBuildCustomizationsDir,
    Test-CmakeSupportsGenerator). Pure functions, extracted and dot-sourced; honors
    $env:SETUP_PS1_PATH and fails loudly if a target function is missing.
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

    # Stubbed: without cmake the "CMake not found" line hits Write-StudioLine first, and unstubbed
    # that is a command-not-found terminating error.
    function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }

    foreach ($fn in @('Resolve-VsGeneratorFromLabel', 'Find-VsBuildTools', 'Get-VcBuildCustomizationsDir',
                      'Test-CmakeSupportsGenerator', 'Get-CmakeVersion', 'Test-CmakeListsGenerator',
                      'Test-CmakeCanDriveGenerator', 'Get-FallbackVsGenerator',
                      'Ensure-BuildToolsForLlamaSourceBuild', 'Test-VCRedistInstalled',
                      'Get-HostMachineArch')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }
}

Describe 'Resolve-VsGeneratorFromLabel (vswhere/dir label -> generator)' {
    # vswhere reports VS 2026's internal major as '18'; both forms must map.
    It 'maps the VS 2026 internal major "18" to the VS 2026 generator' {
        Resolve-VsGeneratorFromLabel '18' | Should -Be 'Visual Studio 18 2026'
    }
    It 'maps the VS 2026 year label "2026" to the VS 2026 generator' {
        Resolve-VsGeneratorFromLabel '2026' | Should -Be 'Visual Studio 18 2026'
    }
    It 'maps the VS 2022 year "2022" and major "17" to the VS 2022 generator' {
        Resolve-VsGeneratorFromLabel '2022' | Should -Be 'Visual Studio 17 2022'
        Resolve-VsGeneratorFromLabel '17'   | Should -Be 'Visual Studio 17 2022'
    }
    It 'maps 2019/2017 (year and major) to their generators' {
        Resolve-VsGeneratorFromLabel '2019' | Should -Be 'Visual Studio 16 2019'
        Resolve-VsGeneratorFromLabel '16'   | Should -Be 'Visual Studio 16 2019'
        Resolve-VsGeneratorFromLabel '2017' | Should -Be 'Visual Studio 15 2017'
        Resolve-VsGeneratorFromLabel '15'   | Should -Be 'Visual Studio 15 2017'
    }
    It 'trims whitespace (vswhere output can carry a trailing newline)' {
        Resolve-VsGeneratorFromLabel "  18 `n" | Should -Be 'Visual Studio 18 2026'
    }
    It 'returns null for unknown or empty labels' {
        Resolve-VsGeneratorFromLabel '2015' | Should -BeNullOrEmpty
        Resolve-VsGeneratorFromLabel ''     | Should -BeNullOrEmpty
        Resolve-VsGeneratorFromLabel $null  | Should -BeNullOrEmpty
    }
}

Describe 'Find-VsBuildTools (VS 2026 generator discovery)' {
    # Windows-only: Find-VsBuildTools builds backslash paths that only resolve there.
    BeforeAll {
        # Pester 5 runs the Describe body only at discovery, so define helpers in BeforeAll.
        function New-FakeVsTree {
            param([string]$Root, [string]$VersionDir, [string]$Edition = 'BuildTools')
            $clDir = Join-Path $Root "Microsoft Visual Studio\$VersionDir\$Edition\VC\Tools\MSVC\14.50.00000\bin\Hostx64\x64"
            New-Item -ItemType Directory -Path $clDir -Force | Out-Null
            New-Item -ItemType File -Path (Join-Path $clDir 'cl.exe') -Force | Out-Null
        }
    }
    BeforeEach {
        $script:OrigPF    = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles}      = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
    }

    It 'detects a filesystem-only VS 2026 BuildTools install (dir "18")' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF'
        New-FakeVsTree -Root $root -VersionDir '18'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86'
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

    It 'detects an older VS installed under the Preview edition dir' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF2022prev'
        New-FakeVsTree -Root $root -VersionDir '2022' -Edition 'Preview'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86d'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Get-VcBuildCustomizationsDir (CUDA to VS MSBuild integration path)' {

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

Describe 'Test-CmakeListsGenerator (probe cmake --help)' {
    # Mock cmake as a function: PowerShell caches its app-path table, so a PATH shim is unreliable.

    It 'returns true when cmake --help lists the generator' {
        Mock cmake { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files.`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'returns false when cmake --help does not list the generator' {
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'returns false when cmake produces no help output' {
        Mock cmake { $null }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }
}

Describe 'Test-CmakeCanDriveGenerator (probe OR version floor)' {
    It 'accepts a sub-4.2 cmake that lists the VS 2026 generator (bundled cmake)' {
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 3.31.0' }
            else { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files." }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'accepts a 4.2 cmake via the version floor when the help probe misses it' {
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 4.2.0' }
            else { 'Generators' }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'rejects a sub-4.2 cmake that does not list the VS 2026 generator' {
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 3.31.0' }
            else { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }
}

Describe 'Get-FallbackVsGenerator (older VS the cmake can drive)' {
    BeforeAll {
        function New-FakeVsTree2 {
            param([string]$Root, [string]$VersionDir, [string]$Edition = 'BuildTools')
            $clDir = Join-Path $Root "Microsoft Visual Studio\$VersionDir\$Edition\VC\Tools\MSVC\14.39.00000\bin\Hostx64\x64"
            New-Item -ItemType Directory -Path $clDir -Force | Out-Null
            New-Item -ItemType File -Path (Join-Path $clDir 'cl.exe') -Force | Out-Null
        }
    }
    BeforeEach {
        $script:OrigPF = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles} = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
    }

    It 'returns the VS 2022 generator when VS 2022 is installed and cmake lists it' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_fb'
        New-FakeVsTree2 -Root $root -VersionDir '2022'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_fb'
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        $r = Get-FallbackVsGenerator
        $r.Generator | Should -Be 'Visual Studio 17 2022'
    }

    It 'returns null when the cmake cannot drive any installed older VS' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_none'
        New-FakeVsTree2 -Root $root -VersionDir '2022'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_none'
        Mock cmake { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files." }
        $r = Get-FallbackVsGenerator
        $r | Should -BeNullOrEmpty
    }

    It 'falls back to an older VS installed under the Preview edition dir' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_prev'
        New-FakeVsTree2 -Root $root -VersionDir '2022' -Edition 'Preview'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_prev'
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        (Get-FallbackVsGenerator).Generator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Deferred build tools (prebuilt path needs no VS/CMake)' {
    # Phase-1 detection must be non-fatal; the install + exit-1 path is covered by
    # studio-windows-no-vs-smoke.yml.
    BeforeEach {
        $script:OrigPF    = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles}      = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
        $script:VsInstallPath = $null
        $script:CmakeGenerator = $null
    }

    It 'Find-VsBuildTools returns null when no VS is present (probe stays non-fatal)' {
        ${env:ProgramFiles}      = (Join-Path $TestDrive 'EmptyPF')
        ${env:ProgramFiles(x86)} = (Join-Path $TestDrive 'EmptyPFx86')
        New-Item -ItemType Directory -Force -Path ${env:ProgramFiles}, ${env:ProgramFiles(x86)} | Out-Null
        Find-VsBuildTools | Should -BeNullOrEmpty
    }

    It 'Ensure-BuildToolsForLlamaSourceBuild no-ops when VS is already detected' {
        $script:VsInstallPath  = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
        $script:CmakeGenerator = 'Visual Studio 17 2022'
        { Ensure-BuildToolsForLlamaSourceBuild } | Should -Not -Throw
        $script:VsInstallPath  | Should -Be 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
        $script:CmakeGenerator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Source-build ordering invariant: CUDA integration runs AFTER the VS generator is finalized (#6473 review)' {
    # Resolve-CudaToolkit copies .targets into the current generator's dir, so it must run after the
    # VS 2026 fallback or the build fails with "No CUDA toolset found".
    It 'the source-build Resolve-CudaToolkit call appears AFTER the Get-FallbackVsGenerator fallback' {
        $text = Get-Content -Raw -LiteralPath $script:SetupPs1
        $idxFallback = $text.IndexOf('$fallback = Get-FallbackVsGenerator')
        $idxResolve  = $text.IndexOf('Resolve-CudaToolkit -RequireOrExit')
        $idxFallback | Should -BeGreaterThan 0
        $idxResolve  | Should -BeGreaterThan 0
        $idxResolve  | Should -BeGreaterThan $idxFallback
    }
}

Describe 'Get-FallbackVsGenerator discovery is symmetric with Find-VsBuildTools (#6473 review)' {
    # The fallback must query vswhere too, or a custom-location VS is missed there.
    It 'queries vswhere as part of fallback discovery' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name Get-FallbackVsGenerator
        $src | Should -Match 'vswhere'
    }
}

Describe 'Test-VCRedistInstalled (VC++ 2015-2022 runtime needed by the prebuilt llama.cpp + PyTorch)' {
    # The prebuilts link the VC++ runtime DLLs, which the Universal CRT lacks.
    BeforeEach { $script:OrigSysRoot = $env:SystemRoot }
    AfterEach  { $env:SystemRoot = $script:OrigSysRoot }

    It 'returns true when vcruntime140_1.dll is present in System32' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $true }
        Test-VCRedistInstalled | Should -BeTrue
    }
    It 'returns true via the registry when the DLL is not found (Installed=1, >= 14.20)' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { [pscustomobject]@{ Installed = 1; Major = 14; Minor = 29 } }
        Test-VCRedistInstalled | Should -BeTrue
    }
    It 'returns false when neither the DLL nor a >= 14.20 registry entry exists' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { throw 'no key' }
        Test-VCRedistInstalled | Should -BeFalse
    }
    It 'returns false for an old 2015-only redist (Installed=1 but < 14.20)' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { [pscustomobject]@{ Installed = 1; Major = 14; Minor = 0 } }
        Test-VCRedistInstalled | Should -BeFalse
    }
}
