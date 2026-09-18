<#
    Find-InstalledUv, executed rather than read.

    tests/sh/test_setup_reuse_installed_uv.sh runs the shell finder for real, but checks the
    PowerShell one only by grepping setup.ps1. Mutation-tested: deleting the winget alias
    directory from the candidates, and making the finder skip every candidate so it can never
    reuse anything, both left that suite green at 43 of 43. Windows is where the re-download
    cost 42 of a 53 s no-op update, so it is the side that most needs running.

    Fixtures are a real executable copied into a candidate directory, and a text file for the
    one that cannot run, so both the Test-Path gate and the launch are exercised.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')
    $setup = Join-Path $PSScriptRoot '../../studio/setup.ps1'
    foreach ($fn in @('Get-UvInstallDir', 'Get-SetupUvExecutableVerdict', 'Get-InstalledUvVerdict',
                      'Test-SetupUvVersionAtLeast', 'Find-InstalledUv')) {
        $src = Get-FunctionSource -Path $setup -Name $fn
        if (-not $src) { throw "could not extract $fn from setup.ps1" }
        . ([scriptblock]::Create($src))
    }
    # The floor the finder compares against, read from setup.ps1 rather than repeated here, so
    # a change to the number cannot leave these cases testing the old one.
    $script:SetupUvMinVersionSource = (Select-String -Path $setup -Pattern '^\$SetupUvMinVersion = "([^"]+)"' |
        Select-Object -First 1).Matches[0].Groups[1].Value
    if (-not $script:SetupUvMinVersionSource) { throw "could not read `$SetupUvMinVersion from setup.ps1" }
    $SetupUvMinVersion = $script:SetupUvMinVersionSource
    # setup.ps1 reports through substep; the finder must not depend on its formatting.
    function substep { param([string]$Message, [string]$Color = 'DarkGray') }

    $script:UvCandidateVars = @(
        'UV_INSTALL_DIR', 'UV_UNMANAGED_INSTALL', 'XDG_BIN_HOME', 'XDG_DATA_HOME', 'LOCALAPPDATA'
    )

    function New-Sandbox {
        $dir = Join-Path ([System.IO.Path]::GetTempPath()) ('finduv_' + [guid]::NewGuid().ToString('N').Substring(0, 10))
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        New-Item -ItemType Directory -Force -Path (Join-Path $dir 'home') | Out-Null
        foreach ($v in $script:UvCandidateVars) { Remove-Item -Path ("env:" + $v) -ErrorAction SilentlyContinue }
        # A home with no uv in it, so the default tier is a miss unless a test populates it.
        $env:USERPROFILE = Join-Path $dir 'home'
        if (-not $IsWindows) { $env:HOME = Join-Path $dir 'home' }
        return $dir
    }

    function Copy-RunnableExe {
        # A stand-in for "a uv that runs": copied ALONE into a candidate directory, it still has
        # to launch and exit 0 for `--version`. pwsh.exe does not qualify, however obvious it
        # looks: on Windows it cannot start without its runtime files beside it, so a fixture
        # built from it made the finder correctly return nothing and the tests fail for a reason
        # that had nothing to do with setup.ps1. Windows' own curl.exe and tar.exe are ordinary
        # console binaries that link only against System32 and take --version for real.
        param([Parameter(Mandatory)][string]$Destination)
        foreach ($src in $script:RunnableExeCandidates) {
            if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { continue }
            try { Copy-Item -LiteralPath $src -Destination $Destination -Force -ErrorAction Stop } catch { continue }
            if (Test-ExeAnswers -Path $Destination) { return $src }
        }
        throw ("no stand-in executable on this host answers --version with exit 0; tried: " +
               ($script:RunnableExeCandidates -join ', '))
    }

    function Test-ExeAnswers {
        param([Parameter(Mandatory)][string]$Path)
        $out = [System.IO.Path]::GetTempFileName()
        $err = [System.IO.Path]::GetTempFileName()
        try {
            $p = Start-Process -FilePath $Path -ArgumentList "--version" -NoNewWindow -PassThru `
                -RedirectStandardOutput $out -RedirectStandardError $err -ErrorAction Stop
            if (-not $p.WaitForExit(15000)) { try { $p.Kill() } catch {}; return $false }
            $p.WaitForExit()
            return ($p.ExitCode -eq 0)
        } catch {
            return $false
        } finally {
            Remove-Item -LiteralPath $out, $err -Force -ErrorAction SilentlyContinue
        }
    }

    $sys = if ($env:SystemRoot) { Join-Path $env:SystemRoot 'System32' } else { '' }
    $script:RunnableExeCandidates = if ($IsWindows) {
        @((Join-Path $sys 'curl.exe'), (Join-Path $sys 'tar.exe'))
    } else {
        @()
    }

    function New-FakeUv {
        # -Kind ok      : an executable that really runs and exits 0
        # -Kind broken  : a file that exists and cannot be launched
        # -Kind folder  : a directory named uv.exe
        # -Kind old     : runs, and answers with a version below the floor (POSIX hosts only,
        #                 where the stand-in is a script and can say what it likes)
        param([Parameter(Mandatory)][string]$Dir, [string]$Kind = 'ok')
        New-Item -ItemType Directory -Force -Path $Dir | Out-Null
        $exe = Join-Path $Dir 'uv.exe'
        if ($Kind -eq 'folder') {
            New-Item -ItemType Directory -Force -Path $exe | Out-Null
            return $exe
        }
        if ($Kind -eq 'broken') {
            Set-Content -LiteralPath $exe -Value 'this is not an executable'
            return $exe
        }
        if ($IsWindows) {
            if ($Kind -eq 'old') { throw "the Windows stand-in cannot choose what it prints" }
            Copy-RunnableExe -Destination $exe | Out-Null
        } else {
            $version = if ($Kind -eq 'old') { '0.4.0' } else { '0.12.1' }
            Set-Content -LiteralPath $exe -Value "#!/bin/sh`nprintf 'uv $version\n'`nexit 0`n"
            & chmod +x $exe
        }
        return $exe
    }
}

Describe "the fixtures themselves" {
    It "has a stand-in that really runs, so a finder miss cannot be blamed on the fixture" {
        $dir = Join-Path ([System.IO.Path]::GetTempPath()) ('fixtureprobe_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        try {
            $exe = New-FakeUv -Dir $dir
            Test-ExeAnswers -Path $exe | Should -BeTrue
        } finally {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "has a broken stand-in that really cannot run" {
        $dir = Join-Path ([System.IO.Path]::GetTempPath()) ('fixtureprobe_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        try {
            $exe = New-FakeUv -Dir $dir -Kind broken
            Test-ExeAnswers -Path $exe | Should -BeFalse
        } finally {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Describe "Find-InstalledUv" {
    BeforeAll {
        # On Windows the stand-in is a real console binary that is NOT uv, so its --version line
        # cannot clear the finder's floor and every discovery case here would miss for a reason
        # that has nothing to do with discovery. The floor is answered for this block and tested
        # on its own in "the uv version floor" below, including the refusal path.
        if ($IsWindows) {
            function Test-SetupUvVersionAtLeast { param([string]$VersionLine, [string]$Minimum) return $true }
        }
    }
    BeforeEach { $script:Sandbox = New-Sandbox }
    AfterEach {
        foreach ($v in $script:UvCandidateVars) { Remove-Item -Path ("env:" + $v) -ErrorAction SilentlyContinue }
        if ($script:Sandbox) { Remove-Item -LiteralPath $script:Sandbox -Recurse -Force -ErrorAction SilentlyContinue }
    }

    It "is a miss when no destination holds a uv, and names what it looked at" {
        Find-InstalledUv | Should -BeNullOrEmpty
        $script:InstalledUvLooked | Should -Not -BeNullOrEmpty
        $script:InstalledUvProbeMiss | Should -BeNullOrEmpty
    }

    It "reuses a uv that runs, at the destination the installer writes to" {
        $dest = Join-Path $script:Sandbox 'custom dir'
        $env:UV_INSTALL_DIR = $dest
        New-FakeUv -Dir $dest | Out-Null
        (Get-UvInstallDir) | Should -Be $dest
        Find-InstalledUv | Should -Be $dest
    }

    It "skips a uv that cannot run and keeps looking" {
        $broken = Join-Path $script:Sandbox 'broken dir'
        $env:UV_INSTALL_DIR = $broken
        New-FakeUv -Dir $broken -Kind broken | Out-Null
        $good = Join-Path $env:USERPROFILE '.local/bin'
        New-FakeUv -Dir $good | Out-Null
        Find-InstalledUv | Should -Be ([System.IO.Path]::Combine($env:USERPROFILE, '.local', 'bin'))
        $script:InstalledUvProbeMiss | Should -Not -BeNullOrEmpty
    }

    It "does not reuse a directory that happens to be named uv.exe" {
        $dest = Join-Path $script:Sandbox 'folder dir'
        $env:UV_INSTALL_DIR = $dest
        New-FakeUv -Dir $dest -Kind folder | Out-Null
        Find-InstalledUv | Should -BeNullOrEmpty
    }

    It "finds the uv install.ps1 put in winget's alias directory" {
        # winget's alias directory reaches PATH only through the registry, so a process started
        # before that install never saw it: this is the tier that made the Windows no-op update
        # download uv again even though install.ps1 had just installed it.
        $local = Join-Path $script:Sandbox 'localappdata'
        $env:LOCALAPPDATA = $local
        $links = [System.IO.Path]::Combine($local, 'Microsoft', 'WinGet', 'Links')
        New-FakeUv -Dir $links | Out-Null
        Find-InstalledUv | Should -Be $links
    }

    It "searches the installer's destination before any other tier" {
        $dest = Join-Path $script:Sandbox 'install dest'
        $env:UV_INSTALL_DIR = $dest
        Find-InstalledUv | Out-Null
        $script:InstalledUvLooked[0] | Should -Be ([System.IO.Path]::Combine($dest, 'uv.exe'))
    }

    It "looks at one directory once, however many variables point at it" {
        $dest = Join-Path $script:Sandbox 'shared dest'
        $env:UV_INSTALL_DIR = $dest
        $env:UV_UNMANAGED_INSTALL = $dest
        $env:XDG_BIN_HOME = $dest
        Find-InstalledUv | Out-Null
        @($script:InstalledUvLooked | Where-Object { $_ -like (Join-Path $dest '*') }).Count |
            Should -Be 1
    }

    It "honours the documented priority order" {
        $first = Join-Path $script:Sandbox 'tier one'
        $second = Join-Path $script:Sandbox 'tier two'
        $env:UV_INSTALL_DIR = $first
        $env:UV_UNMANAGED_INSTALL = $second
        New-FakeUv -Dir $first | Out-Null
        New-FakeUv -Dir $second | Out-Null
        Find-InstalledUv | Should -Be $first
    }

    It "refuses a uv that runs but does not clear the floor, and says which one" {
        # The refusal itself, on every platform: the gate answers no for this case only, so
        # what is under test is the finder's handling of it, not the version parsing.
        function Test-SetupUvVersionAtLeast { param([string]$VersionLine, [string]$Minimum) return $false }
        $dest = Join-Path $script:Sandbox 'old uv'
        $env:UV_INSTALL_DIR = $dest
        $exe = New-FakeUv -Dir $dest
        Find-InstalledUv | Should -BeNullOrEmpty
        $script:InstalledUvTooOld | Should -Be $exe
        $script:InstalledUvProbeMiss | Should -BeNullOrEmpty
    }

    It "reuses a uv whose version clears the floor" -Skip:$IsWindows {
        $dest = Join-Path $script:Sandbox 'new uv'
        $env:UV_INSTALL_DIR = $dest
        New-FakeUv -Dir $dest | Out-Null
        Find-InstalledUv | Should -Be $dest
        $script:InstalledUvTooOld | Should -BeNullOrEmpty
    }

    It "does not reuse a real uv older than the floor" -Skip:$IsWindows {
        $dest = Join-Path $script:Sandbox 'ancient uv'
        $env:UV_INSTALL_DIR = $dest
        $exe = New-FakeUv -Dir $dest -Kind old
        Find-InstalledUv | Should -BeNullOrEmpty
        $script:InstalledUvTooOld | Should -Be $exe
    }
}

Describe "the uv version floor" {
    # The parsing on its own, so the discovery cases above do not have to carry it. install.ps1
    # refuses a uv below the same number outright; reusing one instead of downloading the pinned
    # release would hand the dependency pass the interpreter that refusal exists to avoid.
    It "reads <line> as clearing <floor>: <expected>" -ForEach @(
        @{ line = 'uv 0.12.1 (0123456 2026-01-01)'; floor = '0.9.3'; expected = $true }
        @{ line = 'uv 0.9.3';                       floor = '0.9.3'; expected = $true }
        @{ line = 'uv 1.0.0';                       floor = '0.9.3'; expected = $true }
        @{ line = 'uv 0.9.2';                       floor = '0.9.3'; expected = $false }
        @{ line = 'uv 0.4.0';                       floor = '0.9.3'; expected = $false }
        @{ line = 'curl 8.9.1 (x86_64-pc-linux-gnu)'; floor = '0.9.3'; expected = $false }
        @{ line = 'not a version at all';           floor = '0.9.3'; expected = $false }
        @{ line = '';                               floor = '0.9.3'; expected = $false }
    ) {
        (Test-SetupUvVersionAtLeast -VersionLine $line -Minimum $floor) | Should -Be $expected
    }

    It "compares numerically, not as text" {
        # 0.12.1 sorts below 0.9.3 as a string; that is the comparison this must not be.
        Test-SetupUvVersionAtLeast -VersionLine 'uv 0.12.1' -Minimum '0.9.3' | Should -BeTrue
    }

    It "reads the floor setup.ps1 actually ships" {
        $script:SetupUvMinVersionSource | Should -Not -BeNullOrEmpty
        Test-SetupUvVersionAtLeast -VersionLine 'uv 0.12.1' -Minimum $script:SetupUvMinVersionSource |
            Should -BeTrue
    }
}
