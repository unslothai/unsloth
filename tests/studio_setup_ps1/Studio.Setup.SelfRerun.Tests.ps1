# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    `unsloth studio update` runs setup.ps1 out of site-packages, and the dependency pass
    in the middle of it upgrades the package that ships that file. PowerShell keeps
    executing the copy it parsed, so every phase a release added after the dependency
    step (audio.cpp, for one) was skipped by the very update that installed it, until
    the user ran the update a second time.

    setup.ps1 now notices the replacement and hands off, once, to the new copy. These
    tests build an old/new pair from the REAL pieces of setup.ps1 -- the start capture,
    the helpers, the elevation block and the rerun block, sliced out by literal anchors
    -- and run it the way the CLI does: powershell.exe and pwsh, `-Command "<prelude>& '<p>'
    ... *>&1"` with no console window, plus `-File`. A slice that drifts fails loudly.
#>

BeforeDiscovery {
    $script:Hosts = @()
    if ($IsWindows -or $PSVersionTable.PSEdition -eq 'Desktop') {
        $winPs = Get-Command powershell.exe -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($winPs) { $script:Hosts += @{ HostName = 'powershell.exe'; HostPath = $winPs.Source } }
    }
    $core = if ($PSVersionTable.PSEdition -eq 'Core') { (Get-Process -Id $PID).Path }
            else { (Get-Command pwsh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source }
    if ($core) { $script:Hosts += @{ HostName = 'pwsh'; HostPath = $core } }
}

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }
    $script:SetupPs1 = (Resolve-Path -LiteralPath $script:SetupPs1).Path

    # Read as UTF-8 on every host: 5.1's Get-Content would decode the U+2500 rule in the
    # section marker as ANSI and the anchor below would never match.
    $script:SetupText = [System.IO.File]::ReadAllText($script:SetupPs1, [System.Text.Encoding]::UTF8)
    $lines = $script:SetupText -split "`r?`n"

    function Get-LineSlice {
        param([string]$What, [scriptblock]$IsStart, [scriptblock]$IsEnd)
        $start = -1
        for ($i = 0; $i -lt $lines.Count; $i++) { if (& $IsStart $i) { $start = $i; break } }
        if ($start -lt 0) { throw "drift: the start of $What is gone from setup.ps1." }
        for ($j = $start; $j -lt $lines.Count; $j++) {
            if (& $IsEnd $j) { return ($lines[$start..$j] -join "`n") }
        }
        throw "drift: could not find the end of $What in setup.ps1."
    }

    $startBlock = Get-LineSlice 'the start capture' `
        { param($i) $lines[$i] -eq '$script:SetupSelfPath = $MyInvocation.MyCommand.Path' } `
        { param($j) $lines[$j].StartsWith('try { $script:SetupStartEnv = ') }
    foreach ($needle in @('$script:SetupSelfAtStart', '$script:SetupArgs = @($args)')) {
        if (-not $startBlock.Contains($needle)) { throw "drift: the start capture no longer sets $needle." }
    }

    $utf8Block = Get-LineSlice 'the UTF-8 output block' `
        { param($i) $lines[$i].StartsWith('$_UnslothUtf8NoBom = New-Object') } `
        { param($j) $lines[$j] -eq 'try { $script:StudioStdoutRedirected = [Console]::IsOutputRedirected } catch { }' }

    # Up to the closing brace of the DIAG `if`, then the outer `if` is closed by hand: the
    # rest of that block resolves install roots and has nothing to do with the marker.
    $elevationBlock = (Get-LineSlice 'the elevation report' `
        { param($i) $lines[$i] -eq 'if ($env:SKIP_STUDIO_BASE -ne "1") {' -and
                    $lines[$i + 1] -match '^\s*\$ElevationState = "unknown"' } `
        { param($j) $j -gt 0 -and $lines[$j] -eq '    }' -and
                    ($lines[($j - 3)..($j - 1)] -join "`n").Contains('[TAURI:DIAG] elevated=') }) + "`n}"

    $rule = [string][char]0x2500 + [char]0x2500
    $marker = "# $rule Finish with the setup script this update installed $rule"
    $script:RerunBlock = Get-LineSlice 'the rerun block' `
        { param($i) $lines[$i] -eq $marker } `
        { param($j) $lines[$j] -eq '}' }
    foreach ($needle in @('Test-SetupScriptReplaced', 'New-Module', '$PSDefaultParameterValues = $Defaults',
                          'Remove-WoaMergedOverrides', '-NoTauriMarker')) {
        if (-not $script:RerunBlock.Contains($needle)) { throw "drift: the rerun block no longer contains $needle." }
    }
    if ($script:RerunBlock -notmatch 'step "setup" "([^"]+)"') { throw "drift: the rerun block lost its handoff step." }
    $script:Handoff = $Matches[1]

    $functions = foreach ($name in @('Write-StudioLine', 'Remove-WoaMergedOverrides', 'Exit-SetupFailure',
                                     'Test-SetupScriptReplaced', 'Restore-SetupStartEnvironment')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $name
        if (-not $src) { throw "drift: $name is gone from setup.ps1." }
        $src
    }

    # One template for every variant. The banner reads $leakProbe before this copy assigns
    # it, so a non-empty value there can only have come from the run that handed off.
    $head = @'
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
__START__
__UTF8__
__FUNCS__
function step { param([string]$Label, [string]$Value) Write-StudioLine ("  {0,-15}{1}" -f $Label, $Value) }
$env:PSModulePath = "MUTATED;" + $env:PSModulePath
Write-StudioLine ("  " + [char]::ConvertFromUtf32(0x1F9A5) + " __TAG__ banner " + [char]0x2500 + " args=[" + ($script:SetupArgs -join '|') + "] argc=[" + $script:SetupArgs.Count + "] guard=[" + $env:UNSLOTH_SETUP_RERUN + "] mutated=[" + @($env:PSModulePath -split ';' | Where-Object { $_ -eq 'MUTATED' }).Count + "] leaked=[" + $leakProbe + "] extra=[" + $env:SETUP_ADDED + "] full=[" + $env:UNSLOTH_STUDIO_FULL_DEPS + "] date=[" + (Get-Date -Date 2001-02-03) + "]")
$env:SETUP_ADDED = "added-by-__TAG__"
$leakProbe = "parent-__TAG__"
__ELEVATION__
Write-StudioLine "__TAG__: deps"
'@
    # uv replaces files by unlink + create; rename over an open file fails on Windows.
    $oldDeps = @'
if ($env:MODE -eq 'replace') {
    Remove-Item -LiteralPath $script:SetupSelfPath
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'new.ps1') -Destination $script:SetupSelfPath
}
if ($env:MODE -eq 'identical') {
    $same = [System.IO.File]::ReadAllBytes($script:SetupSelfPath)
    Remove-Item -LiteralPath $script:SetupSelfPath
    [System.IO.File]::WriteAllBytes($script:SetupSelfPath, $same)
}
if ($env:FAIL_DEPS -eq '1') { Exit-SetupFailure -Message "Python dependency installation failed (exit code 2)" -Code 2 }
'@
    $newDeps = @'
if ($env:MODE_NEW -eq 'third') {
    Remove-Item -LiteralPath $script:SetupSelfPath
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'third.ps1') -Destination $script:SetupSelfPath
}
'@
    $oldTail = @'
Write-StudioLine "__TAG__: llama.cpp"
Write-StudioLine "__TAG__: done"
'@
    $newTail = @'
Write-StudioLine "__TAG__: llama.cpp"
Write-StudioLine "__TAG__: audio.cpp"
if ($env:NEW_FAIL) { Exit-SetupFailure -Message "audio.cpp failed specifically" -Code ([int]$env:NEW_FAIL) }
if ($env:NEW_THROW) { throw "audio.cpp threw" }
& ([System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName) -NoProfile -NonInteractive -Command 'exit 7'
Write-StudioLine "__TAG__: done"
'@

    function New-Variant {
        param([string]$Tag, [string]$Deps, [string]$Tail, [string]$Rerun)
        $text = $head.Replace('__START__', $startBlock).Replace('__UTF8__', $utf8Block).
            Replace('__FUNCS__', ($functions -join "`n")).Replace('__ELEVATION__', $elevationBlock)
        # In setup.ps1 the block sits inside `if (-not $SkipPythonDeps) {`.
        $text = @($text, $Deps, 'if ($true) {', $Rerun, '}', $Tail, '') -join "`n"
        return $text.Replace('__TAG__', $Tag)
    }

    # Controls: the same block with the module swapped for a bare `&`, and with the module
    # kept but $PSDefaultParameterValues not handed across.
    $moduleFrom = $script:RerunBlock.IndexOf('$_setupRerunner = New-Module')
    $moduleToNeedle = '$_setupRerunCode = & $_setupRerunner { $script:Code }'
    $moduleTo = $script:RerunBlock.IndexOf($moduleToNeedle)
    if ($moduleFrom -lt 0 -or $moduleTo -lt 0) { throw "drift: the rerun block no longer runs the new copy in a module." }
    $bareRerun = $script:RerunBlock.Substring(0, $moduleFrom) +
        "try {`n        & `$script:SetupSelfPath @_setupRerunArgs`n        `$_setupRerunOk = `$?`n        `$_setupRerunCode = `$LASTEXITCODE" +
        $script:RerunBlock.Substring($moduleTo + $moduleToNeedle.Length)
    $noForwardRerun = $script:RerunBlock.Replace('$PSDefaultParameterValues = $Defaults', '')

    $script:Templates = Join-Path $TestDrive 'templates'
    New-Item -ItemType Directory -Path $script:Templates -Force | Out-Null
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    $variants = @{
        'old.ps1'       = New-Variant 'OLD' $oldDeps $oldTail $script:RerunBlock
        'new.ps1'       = New-Variant 'NEW' $newDeps $newTail $script:RerunBlock
        'third.ps1'     = New-Variant 'THIRD' '' $newTail $script:RerunBlock
        'old-bare.ps1'  = New-Variant 'OLD' $oldDeps $oldTail $bareRerun
        'old-nofwd.ps1' = New-Variant 'OLD' $oldDeps $oldTail $noForwardRerun
    }
    foreach ($name in $variants.Keys) {
        [System.IO.File]::WriteAllText((Join-Path $script:Templates $name), $variants[$name], $utf8NoBom)
    }

    # Everything a case sets, cleared first so the runner's own environment cannot steer one.
    $script:CaseVars = @('MODE', 'MODE_NEW', 'FAIL_DEPS', 'NEW_FAIL', 'NEW_THROW', 'SETUP_ADDED',
        'UNSLOTH_SETUP_RERUN', 'UNSLOTH_TAURI_UPDATE', 'UNSLOTH_TAURI_MODE', 'SKIP_STUDIO_BASE',
        'UNSLOTH_STUDIO_FULL_DEPS')

    function Invoke-Setup {
        param(
            [string]$HostPath,
            [ValidateSet('Command', 'File')][string]$Shape = 'Command',
            [string]$Start = 'old.ps1',
            [hashtable]$Env = @{},
            [string]$Suffix = ''
        )
        $dir = Join-Path $TestDrive ([guid]::NewGuid().ToString('n'))
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        foreach ($name in @('new.ps1', 'third.ps1')) { Copy-Item -LiteralPath (Join-Path $script:Templates $name) -Destination $dir }
        $setup = Join-Path $dir 'setup.ps1'
        Copy-Item -LiteralPath (Join-Path $script:Templates $Start) -Destination $setup

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $HostPath
        # The CLI's argv (studio.py): a prelude that fills $PSDefaultParameterValues, then the
        # script with every stream folded into stdout.
        $psi.Arguments = if ($Shape -eq 'Command') {
            "-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command `"`$PSDefaultParameterValues['Get-Date:Format'] = 'yyyy-MM-dd'; & '$setup' --verbose 'a b' *>&1$Suffix`""
        } else {
            "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$setup`" --verbose `"a b`""
        }
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        foreach ($k in $script:CaseVars) { $psi.EnvironmentVariables.Remove($k) }
        if ([System.IO.Path]::GetFileNameWithoutExtension($HostPath) -eq 'powershell') {
            # A pwsh runner's PSModulePath leads with PowerShell 7 modules, which 5.1 cannot load.
            $psi.EnvironmentVariables['PSModulePath'] = (@(
                (Join-Path $env:ProgramFiles 'WindowsPowerShell\Modules'),
                (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules')) -join ';')
        }
        foreach ($k in $Env.Keys) { $psi.EnvironmentVariables[$k] = [string]$Env[$k] }

        $proc = [System.Diagnostics.Process]::Start($psi)
        $errTask = $proc.StandardError.ReadToEndAsync()
        $buffer = New-Object System.IO.MemoryStream
        $proc.StandardOutput.BaseStream.CopyTo($buffer)
        if (-not $proc.WaitForExit(300000)) { $proc.Kill(); throw "setup run timed out in $dir" }
        $proc.WaitForExit()
        $bytes = $buffer.ToArray()
        # Strict, as the desktop app's from_utf8_lossy would otherwise hide a bad byte as U+FFFD.
        $utf8Ok = $true
        try { $text = (New-Object System.Text.UTF8Encoding $false, $true).GetString($bytes) }
        catch { $utf8Ok = $false; $text = [System.Text.Encoding]::UTF8.GetString($bytes) }
        [pscustomobject]@{
            Rc = $proc.ExitCode
            Text = $text
            Lines = @($text -split "`r?`n")
            Utf8Ok = $utf8Ok
            Stderr = $errTask.Result
        }
    }

    function Get-Count {
        param($Run, [string]$Pattern)
        @($Run.Lines | Where-Object { $_.Contains($Pattern) }).Count
    }

    function Get-Banner {
        param($Run, [string]$Tag)
        $hit = @($Run.Lines | Where-Object { $_.Contains(" $Tag banner ") })
        if ($hit.Count -ne 1) { throw "expected one $Tag banner, got $($hit.Count):`n$($Run.Text)`n$($Run.Stderr)" }
        $hit[0]
    }
}

Describe 'the rerun block in setup.ps1' {
    It 'runs at script top level, after a successful dependency pass, with nothing capturing its output' {
        $tokens = $null; $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseInput($script:SetupText, [ref]$tokens, [ref]$errors)
        $errors.Count | Should -Be 0
        $ifs = @($ast.FindAll({
            param($n)
            $n -is [System.Management.Automation.Language.IfStatementAst] -and
                $n.Clauses[0].Item1.Extent.Text -eq 'Test-SetupScriptReplaced'
        }, $true))
        $ifs.Count | Should -Be 1
        $p = $ifs[0].Parent
        while ($p) {
            # A function, or an assignment or pipeline around it, would capture the new copy's
            # output instead of streaming it.
            $p | Should -Not -BeOfType [System.Management.Automation.Language.FunctionDefinitionAst]
            $p | Should -Not -BeOfType [System.Management.Automation.Language.AssignmentStatementAst]
            $p | Should -Not -BeOfType [System.Management.Automation.Language.PipelineAst]
            $p = $p.Parent
        }
        $at = $ifs[0].Extent.StartOffset
        $at | Should -BeGreaterThan $script:SetupText.IndexOf('if ($stackExit -ne 0) {')
        $at | Should -BeLessThan $script:SetupText.IndexOf('step "python" "dependencies up to date"')
    }
}

Describe 'elevation prompts across the rerun' {
    BeforeAll {
        $script:LongPathsBlock = Get-LineSlice 'the Long Paths block' `
            { param($i) $lines[$i] -eq '$LongPathsEnabled = $false' } `
            { param($j) $lines[$j] -eq '}' }
        if (-not $script:LongPathsBlock.Contains('-Verb RunAs')) { throw "drift: the Long Paths block no longer elevates." }
        $script:CudaTargetsBlock = Get-LineSlice 'the CUDA .targets block' `
            { param($i) $lines[$i].StartsWith('# CUDA installed before VS Build Tools leaves .targets missing') } `
            { param($j) $lines[$j] -eq '}' }
        if (-not $script:CudaTargetsBlock.Contains('-Verb RunAs')) { throw "drift: the CUDA .targets block no longer elevates." }
        $script:GitBlock = Get-LineSlice 'the Git block' `
            { param($i) $lines[$i] -eq '$HasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)' } `
            { param($j) $lines[$j] -eq '}' }
        if (-not $script:GitBlock.Contains('winget install Git.Git')) { throw "drift: the Git block no longer installs Git." }
        $tokens = $null; $errors = $null
        $script:Ast = [System.Management.Automation.Language.Parser]::ParseInput($script:SetupText, [ref]$tokens, [ref]$errors)
        $script:HandoffAt = $script:SetupText.IndexOf('if (Test-SetupScriptReplaced) {')
        if ($script:HandoffAt -lt 0) { throw "drift: the rerun block is gone." }
        function step { param([string]$Label, [string]$Value) }
        function substep { param([string]$Message) }
        function Write-StudioLine { param([string]$Text) }
        function Get-VcBuildCustomizationsDir { param($VsInstallPath, $Generator) $script:VsCustomDir }
    }
    AfterEach { Remove-Item Env:UNSLOTH_SETUP_RERUN -ErrorAction SilentlyContinue }

    It 'Long Paths runs at top level before the handoff, so the first pass has already asked' {
        $at = $script:SetupText.IndexOf('$LongPathsEnabled = $false')
        $at | Should -BeLessThan $script:HandoffAt
        $node = $script:Ast.Find({ param($n) $n.Extent.StartOffset -eq $at -and $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true)
        $node.Parent.Parent | Should -BeOfType [System.Management.Automation.Language.ScriptBlockAst]
    }

    It 'Long Paths: asks on a first run and not on the rerun (<Guard>)' -ForEach @(
        @{ Guard = ''; Expected = 1 }, @{ Guard = '1'; Expected = 0 }
    ) {
        Mock Get-ItemProperty { [pscustomobject]@{ LongPathsEnabled = 0 } }
        Mock Start-Process { [pscustomobject]@{ ExitCode = 0 } }
        if ($Guard) { $env:UNSLOTH_SETUP_RERUN = $Guard }
        $StageRoot = $null
        . ([scriptblock]::Create($script:LongPathsBlock))
        Should -Invoke Start-Process -Times $Expected -Exactly
    }

    It 'the VC++ runtime install runs at top level before the handoff' {
        $calls = @($script:Ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Ensure-VCRedist' }, $true))
        $calls.Count | Should -Be 1
        $calls[0].Extent.StartOffset | Should -BeLessThan $script:HandoffAt
        $calls[0].Parent.Parent.Parent | Should -BeOfType [System.Management.Automation.Language.ScriptBlockAst]
    }

    It 'VC++ runtime: tries on a first run and not on the rerun (<Guard>)' -ForEach @(
        @{ Guard = ''; Expected = 1 }, @{ Guard = '1'; Expected = 0 }
    ) {
        . ([scriptblock]::Create((Get-FunctionSource -Path $script:SetupPs1 -Name 'Ensure-VCRedist')))
        function Test-VCRedistInstalled { $false }
        function Refresh-Environment { }
        function Invoke-SetupCommand { param([scriptblock]$Block) }
        Mock Invoke-SetupCommand { }
        Mock Get-Command { [pscustomobject]@{ Name = 'winget' } } -ParameterFilter { $Name -eq 'winget' }
        Mock Invoke-WebRequest { throw 'offline' }
        if ($Guard) { $env:UNSLOTH_SETUP_RERUN = $Guard }
        $StageRoot = $null
        Ensure-VCRedist
        Should -Invoke Invoke-SetupCommand -Times $Expected -Exactly
        Should -Invoke Invoke-WebRequest -Times $Expected -Exactly
    }

    It 'Git runs at top level before the handoff' {
        $script:SetupText.IndexOf('winget install Git.Git') | Should -BeLessThan $script:HandoffAt
    }

    It 'Git: installs only when a clone needs it (rerun=<Guard>, needed=<Needed>)' -ForEach @(
        @{ Guard = ''; Needed = $false; Expected = 0 }, @{ Guard = '1'; Needed = $false; Expected = 0 },
        @{ Guard = ''; Needed = $true; Expected = 1 }, @{ Guard = '1'; Needed = $true; Expected = 1 }
    ) {
        function Invoke-SetupCommand { param([scriptblock]$Block) }
        function Refresh-Environment { }
        function Exit-SetupFailure { param([string]$Message) throw $Message }
        Mock Invoke-SetupCommand { }
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'git' }
        Mock Get-Command { [pscustomobject]@{ Name = 'winget' } } -ParameterFilter { $Name -eq 'winget' }
        foreach ($k in @('STUDIO_LOCAL_INSTALL', 'UNSLOTH_LOCAL_LLAMA_CPP_DIR', 'UNSLOTH_LLAMA_PR_FORCE', 'UNSLOTH_LLAMA_TAG',
                         'UNSLOTH_LLAMA_FORCE_COMPILE', 'UNSLOTH_LLAMA_PR')) {
            Remove-Item "Env:$k" -ErrorAction SilentlyContinue
        }
        $DefaultLlamaPrForce = ''; $DefaultLlamaSource = 'https://github.com/ggml-org/llama.cpp'; $DefaultLlamaTag = 'b1'
        if ($Guard) { $env:UNSLOTH_SETUP_RERUN = $Guard }
        if ($Needed) { $env:UNSLOTH_LLAMA_FORCE_COMPILE = '1' }
        $StageRoot = $null
        # Git stays missing under the mock, so a needed install ends in Exit-SetupFailure.
        try { . ([scriptblock]::Create($script:GitBlock)) } catch { if (-not $Needed) { throw } }
        Remove-Item Env:UNSLOTH_LLAMA_FORCE_COMPILE -ErrorAction SilentlyContinue
        Should -Invoke Invoke-SetupCommand -Times $Expected -Exactly
    }

    # Resolve-CudaToolkit only runs for a llama.cpp source build, after the handoff, so a first pass
    # that hands off never reaches this prompt and the rerun must still ask.
    It 'CUDA .targets is only reached after the handoff' {
        $calls = @($script:Ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Resolve-CudaToolkit' }, $true))
        $calls.Count | Should -BeGreaterThan 0
        foreach ($call in $calls) { $call.Extent.StartOffset | Should -BeGreaterThan $script:HandoffAt }
    }

    It 'CUDA .targets: still asks on the rerun (<Guard>)' -ForEach @(
        @{ Guard = '' }, @{ Guard = '1' }
    ) {
        $id = if ($Guard) { 'rerun' } else { 'first' }
        $CudaToolkitRoot = Join-Path $TestDrive "cuda-$id"
        New-Item -ItemType Directory -Path (Join-Path $CudaToolkitRoot 'extras\visual_studio_integration\MSBuildExtensions') -Force | Out-Null
        $script:VsCustomDir = Join-Path $TestDrive "vs-$id"
        New-Item -ItemType Directory -Path $script:VsCustomDir -Force | Out-Null
        $VsInstallPath = 'C:\VS'; $CmakeGenerator = 'Visual Studio 17 2022'
        Mock Copy-Item { throw 'access denied' }
        Mock Start-Process { }
        if ($Guard) { $env:UNSLOTH_SETUP_RERUN = $Guard }
        . ([scriptblock]::Create($script:CudaTargetsBlock))
        Should -Invoke Start-Process -Times 1 -Exactly
    }
}

Describe 'finishing with the setup script the update installed (<HostName>)' -ForEach $script:Hosts {
    Context '-Command, the CLI shape' {
        BeforeAll {
            $script:Replaced = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace' }
        }

        It 'A1: skips the old phases and runs the new copy once' {
            $r = $script:Replaced
            $r.Rc | Should -Be 0 -Because $r.Text
            Get-Count $r $script:Handoff | Should -Be 1
            Get-Count $r 'OLD: llama.cpp' | Should -Be 0
            Get-Count $r 'OLD: done' | Should -Be 0
            Get-Count $r ' NEW banner ' | Should -Be 1
            Get-Count $r 'NEW: audio.cpp' | Should -Be 1
            Get-Count $r 'NEW: done' | Should -Be 1
        }

        It 'A9: keeps UTF-8 intact across the rerun with no console' {
            $r = $script:Replaced
            $r.Utf8Ok | Should -BeTrue
            $glyphs = [char]::ConvertFromUtf32(0x1F9A5) + ' NEW banner ' + [char]0x2500
            Get-Count $r $glyphs | Should -Be 1
        }

        It 'A2: does not rerun when nothing replaced the script (<Mode>)' -ForEach @(
            @{ Mode = 'identical' }, @{ Mode = 'none' }
        ) {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = $Mode }
            $r.Rc | Should -Be 0 -Because $r.Text
            Get-Count $r $script:Handoff | Should -Be 0
            Get-Count $r 'OLD: llama.cpp' | Should -Be 1
            Get-Count $r 'NEW' | Should -Be 0
        }

        It 'A3: does not rerun after a failed dependency step, and reports that failure alone' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace'; FAIL_DEPS = '1'; UNSLOTH_TAURI_UPDATE = '1' }
            $r.Rc | Should -Not -Be 0
            Get-Count $r $script:Handoff | Should -Be 0
            Get-Count $r 'NEW' | Should -Be 0
            Get-Count $r '[TAURI:ERROR]' | Should -Be 1
            Get-Count $r '[TAURI:ERROR] Python dependency installation failed' | Should -Be 1
        }

        It 'A4: hands off once even when the new copy is replaced again' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace'; MODE_NEW = 'third' }
            $r.Rc | Should -Be 0 -Because $r.Text
            Get-Count $r $script:Handoff | Should -Be 1
            Get-Count $r 'THIRD' | Should -Be 0
            Get-Count $r 'NEW: audio.cpp' | Should -Be 1
        }

        It 'A4: does not hand off when the guard is already set' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace'; UNSLOTH_SETUP_RERUN = '1' }
            $r.Rc | Should -Be 0 -Because $r.Text
            Get-Count $r $script:Handoff | Should -Be 0
            Get-Count $r 'OLD: llama.cpp' | Should -Be 1
            Get-Count $r ' NEW banner ' | Should -Be 0
        }

        It 'A5: exits as a direct run of the new copy would (<Case>)' -ForEach @(
            @{ Case = 'success after a nonzero native exit'; Extra = @{}; Rc = 0; Errors = 0 }
            @{ Case = 'Exit-SetupFailure -Code 4'; Extra = @{ NEW_FAIL = '4'; UNSLOTH_TAURI_UPDATE = '1' }; Rc = 1; Errors = 1 }
            @{ Case = 'an uncaught throw'; Extra = @{ NEW_THROW = '1' }; Rc = 1; Errors = 0 }
        ) {
            $direct = Invoke-Setup -HostPath $HostPath -Start 'new.ps1' -Env $Extra
            $envs = @{ MODE = 'replace' } + $Extra
            $r = Invoke-Setup -HostPath $HostPath -Env $envs
            $direct.Rc | Should -Be $Rc -Because $direct.Text
            $r.Rc | Should -Be $direct.Rc -Because $r.Text
            Get-Count $r $script:Handoff | Should -Be 1
            Get-Count $r '[TAURI:ERROR]' | Should -Be $Errors
            if ($Errors) { Get-Count $r '[TAURI:ERROR] audio.cpp failed specifically' | Should -Be 1 }
        }

        It 'A6: gives the new copy the caller''s environment and arguments plus the guard' {
            $banner = Get-Banner $script:Replaced 'NEW'
            $banner | Should -BeLike '*args=`[--verbose|a b`] argc=`[2`]*'
            $banner | Should -BeLike '*guard=`[1`]*'
            $banner | Should -BeLike '*extra=`[`]*'
            $banner | Should -BeLike '*mutated=`[1`]*'
        }

        It 'A6: leaves no guard behind in the session' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace' } -Suffix '; ''after-guard=['' + $env:UNSLOTH_SETUP_RERUN + '']'''
            Get-Count $r 'NEW: done' | Should -Be 1 -Because $r.Text
            Get-Count $r 'after-guard=[]' | Should -Be 1
        }

        It 'A12: does not force a second full dependency pass in the rerun' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace'; UNSLOTH_STUDIO_FULL_DEPS = '1' }
            Get-Banner $r 'OLD' | Should -BeLike '*full=`[1`]*'
            Get-Banner $r 'NEW' | Should -BeLike '*full=`[`]*'
        }

        It 'A7: does not let the old copy''s variables reach the new one' {
            Get-Banner $script:Replaced 'NEW' | Should -BeLike '*leaked=`[`]*'
        }

        It 'A7 control: a bare & leaks them, so the check above can fail' {
            $r = Invoke-Setup -HostPath $HostPath -Start 'old-bare.ps1' -Env @{ MODE = 'replace' }
            Get-Count $r 'NEW: audio.cpp' | Should -Be 1 -Because $r.Text
            Get-Banner $r 'NEW' | Should -BeLike '*leaked=`[parent-OLD`]*'
        }

        It 'A8: keeps the prelude''s $PSDefaultParameterValues' {
            Get-Banner $script:Replaced 'NEW' | Should -BeLike '*date=`[2001-02-03`]*'
        }

        It 'A8 control: a module that is not handed them loses them' {
            $r = Invoke-Setup -HostPath $HostPath -Start 'old-nofwd.ps1' -Env @{ MODE = 'replace' }
            Get-Count $r 'NEW: audio.cpp' | Should -Be 1 -Because $r.Text
            Get-Banner $r 'NEW' | Should -Not -BeLike '*date=`[2001-02-03`]*'
        }

        It 'A11: reports elevation once per update' {
            $r = Invoke-Setup -HostPath $HostPath -Env @{ MODE = 'replace'; UNSLOTH_TAURI_UPDATE = '1' }
            Get-Count $r 'NEW: done' | Should -Be 1 -Because $r.Text
            Get-Count $r '[TAURI:DIAG] elevated=' | Should -Be 1
        }
    }

    Context '-File' {
        It 'A1: skips the old phases and runs the new copy once' {
            $r = Invoke-Setup -HostPath $HostPath -Shape File -Env @{ MODE = 'replace' }
            $r.Rc | Should -Be 0 -Because $r.Text
            Get-Count $r 'OLD: llama.cpp' | Should -Be 0
            Get-Count $r 'NEW: audio.cpp' | Should -Be 1
        }

        It 'A5: exits as a direct run of the new copy would (<Case>)' -ForEach @(
            @{ Case = 'Exit-SetupFailure -Code 4'; Extra = @{ NEW_FAIL = '4'; UNSLOTH_TAURI_UPDATE = '1' }; Rc = 4; Errors = 1 }
            @{ Case = 'an uncaught throw'; Extra = @{ NEW_THROW = '1' }; Rc = 1; Errors = 0 }
        ) {
            $direct = Invoke-Setup -HostPath $HostPath -Shape File -Start 'new.ps1' -Env $Extra
            $r = Invoke-Setup -HostPath $HostPath -Shape File -Env (@{ MODE = 'replace' } + $Extra)
            $direct.Rc | Should -Be $Rc -Because $direct.Text
            $r.Rc | Should -Be $direct.Rc -Because $r.Text
            Get-Count $r '[TAURI:ERROR]' | Should -Be $Errors
        }
    }
}
