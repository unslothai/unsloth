#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests New-UnslothTorchOverridesFile: non-ASCII lines survive and relative refs are rebased.
# Run: pwsh -NoProfile -File tests/studio/test_torch_overrides_merge.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "New-UnslothTorchOverridesFile"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one New-UnslothTorchOverridesFile in install.ps1, found $($fn.Count)" }
# PowerShell does not hoist, so the helpers the merge calls have to be defined here too.
foreach ($helper in "Get-WoaRequirementEntries", "Resolve-WoaOverrideLine") {
    $h = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $helper
    }, $true)
    if ($h.Count -ne 1) { throw "expected exactly one $helper in install.ps1, found $($h.Count)" }
    Invoke-Expression $h[0].Extent.Text
}
Invoke-Expression $fn[0].Extent.Text

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# The helper reads $SkipTorch from its enclosing scope.
$SkipTorch = $false

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-ovtest-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work -Force | Out-Null
$onWindows = -not ($IsLinux -or $IsMacOS)
if ($onWindows) {
    # Windows starts a stand-in only through an extension it can execute.
    $fakePy = Join-Path $work "fakepython.cmd"
    Set-Content -LiteralPath $fakePy -Value @(
        "@echo torch==2.11.0+cu130"
        "@echo torchvision==0.26.0+cu130"
        "@echo torchaudio==2.11.0+cu130"
    ) -Encoding ascii
} else {
    $fakePy = Join-Path $work "fakepython"
    Set-Content -LiteralPath $fakePy -Value @(
        "#!/usr/bin/env bash"
        "printf 'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\ntorchaudio==2.11.0+cu130\n'"
    ) -Encoding ascii
    & chmod +x $fakePy
}

$acute = [char]0x00E9
$savedOverride = $env:UV_OVERRIDE
$made = @()

try {
    $callerDir = Join-Path $work "callerdir"
    New-Item -ItemType Directory -Path $callerDir -Force | Out-Null
    $nested = Join-Path $callerDir "nested.txt"
    Set-Content -LiteralPath $nested -Value "idna==3.6" -Encoding ascii
    $callerOv = Join-Path $callerDir "over.txt"
    [System.IO.File]::WriteAllText(
        $callerOv,
        "-r nested.txt`ncaf${acute}pkg==1.0`ntorch==1.0`ntorchvision==0.1`nplainpkg==2.0`n",
        (New-Object System.Text.UTF8Encoding($false)))

    $env:UV_OVERRIDE = $callerOv
    $merged = New-UnslothTorchOverridesFile -PythonExe $fakePy
    $made += $merged
    Check "returns a merged overrides file" ($null -ne $merged -and (Test-Path -LiteralPath $merged))

    $mergedText = [System.IO.File]::ReadAllText($merged)

    Check "non-ASCII caller requirement survives the merge (no '?' substitution)" `
        ($mergedText -match "caf${acute}pkg==1\.0" -and $mergedText -notmatch 'caf\?pkg')
    Check "the caller's relative include is rebased, not carried as a relative line" `
        ($mergedText -notmatch '(?m)^\s*-r\s')
    Check "and its contents come across instead" ($mergedText -match '(?m)^idna==3\.6$')

    Check "frozen torch trio is pinned first" ($mergedText -match '^torch==2\.11\.0\+cu130')
    Check "caller's torch-trio lines are dropped" `
        ($mergedText -notmatch '(?m)^torch==1\.0$' -and $mergedText -notmatch '(?m)^torchvision==0\.1$')
    Check "caller's ordinary requirements are carried over" ($mergedText -match '(?m)^plainpkg==2\.0$')
    Check "merged file is newline terminated" ($mergedText.EndsWith("`n"))

    $bytes = [System.IO.File]::ReadAllBytes($merged)
    Check "merged file has no UTF-8 BOM" `
        (-not ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF))

    Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
    $bare = New-UnslothTorchOverridesFile -PythonExe $fakePy
    $made += $bare
    Check "works with no caller override file" ($null -ne $bare -and (Test-Path -LiteralPath $bare))
    if ($bare) {
        $bareText = [System.IO.File]::ReadAllText($bare)
        Check "bare merge carries the whole trio" `
            ($bareText -match 'torch==2\.11\.0' -and $bareText -match 'torchvision==0\.26\.0' -and
             $bareText -match 'torchaudio==2\.11\.0')
    }

    $SkipTorch = $true
    Check "returns null under --no-torch" ($null -eq (New-UnslothTorchOverridesFile -PythonExe $fakePy))
    $SkipTorch = $false

    # UV_OVERRIDE is space-separated itself, so only %TEMP% can hand the helper a spaced path.
    if ($onWindows) {
        $spacedTemp = Join-Path $work "John Doe"
        New-Item -ItemType Directory -Path $spacedTemp -Force | Out-Null
        $dirShort = $null
        try { $dirShort = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($spacedTemp).ShortPath } catch { }
        $script:substepCalls = @()
        function substep { param($Message, $Color) $script:substepCalls += $Message }
        Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
        $savedTmp = $env:TMP
        $env:TMP = $spacedTemp   # GetTempPath reads TMP first
        try { $spaced = New-UnslothTorchOverridesFile -PythonExe $fakePy }
        finally { $env:TMP = $savedTmp }
        $made += $spaced
        if ($dirShort -and -not $dirShort.Contains(" ")) {
            Check "a spaced temp path comes back without a space" ($spaced -and -not $spaced.Contains(" "))
            Check "the short path names a file in that same temp directory" (
                $spaced -and (Test-Path -LiteralPath $spaced) -and ((Split-Path -Parent $spaced) -eq $dirShort))
        } else {
            Check "no 8.3 name: no overrides file is returned" ($null -eq $spaced)
            Check "no 8.3 name: the file it created is removed" (@(Get-ChildItem -LiteralPath $spacedTemp).Count -eq 0)
            Check "no 8.3 name: the fallback is announced" ($script:substepCalls.Count -gt 0)
        }
    } else {
        Write-Host "  SKIP  spaced-path checks need Windows 8.3 names"
    }

    # The merged copy can hold an authenticated URL, so it must be tracked and locked down.

    $env:UV_OVERRIDE = $callerOv
    $script:TorchOverridesFile = $null
    $merged = New-UnslothTorchOverridesFile -PythonExe $fakePy
    $made += $merged
    Check "the merged path is tracked for the outer cleanup" ($script:TorchOverridesFile -eq $merged)

    $src = $fn[0].Extent.Text
    $iTrack = $src.IndexOf('$script:TorchOverridesFile = $f')
    $iWrite = $src.IndexOf('[System.IO.File]::WriteAllText(')
    Check "the path is tracked BEFORE the write that can throw" (($iTrack -ge 0) -and ($iTrack -lt $iWrite))
    Check "a failed write removes the file it created" ($src -match 'catch \{\s*\r?\n\s*Remove-UnslothTempFileQuietly -Path \$f')
    # An unresolvable 8.3 alias fails the install and then throws during cleanup.
    Check "the 8.3 alias is accepted only once it resolves" (
        $src -match 'Test-Path -LiteralPath \$short -PathType Leaf')
    Check "the give-up branch clears the path it just deleted" (
        ([regex]::Matches($src, '\$script:TorchOverridesFile = \$null')).Count -eq 2)

    # PS 5.1 decodes BOM-less files as ANSI; pwsh on Linux cannot reproduce it, so check the reader.
    $scanSrc = ($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq "Get-WoaRequirementEntries"
    }, $true))[0].Extent.Text
    Check "the caller's override is read as UTF-8, not by the shell's default codepage" (
        ($scanSrc -match '\[System\.IO\.File\]::ReadAllLines\(') -and
        ($src -notmatch 'Get-Content -LiteralPath \$ovFile')
    )
    Check "the inherited ACL is replaced rather than kept" (
        ($src -match 'SetAccessRuleProtection\(\$true, \$false\)') -and ($src -match 'Set-Acl -LiteralPath \$f')
    )

    Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
    # `python` first on Windows: `python3` there can be the Microsoft Store alias stub.
    $pyNames = if ($onWindows) { @("python", "python3") } else { @("python3", "python") }
    $hostPy = (Get-Command $pyNames -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
    if (-not $hostPy) { throw "no python on PATH for the PYTHONPATH shadow case" }
    $venv = Join-Path $work "venv"
    & $hostPy -m venv --without-pip $venv
    $venvPy = if ($onWindows) { Join-Path $venv "Scripts\python.exe" } else { Join-Path $venv "bin/python" }
    $site = (& $venvPy -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
    $shadow = Join-Path $work "shadow"
    foreach ($d in @(@($site, "torch", "2.11.0+cu130"), @($site, "torchvision", "0.26.0+cu130"), @($shadow, "torch", "2.9.0a0+50eac811a6.nv25.9"))) {
        $info = Join-Path $d[0] "$($d[1])-$($d[2]).dist-info"
        New-Item -ItemType Directory -Path $info -Force | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $info "METADATA"), "Metadata-Version: 2.1`nName: $($d[1])`nVersion: $($d[2])`n")
    }
    $savedPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $shadow
    try { $shadowed = New-UnslothTorchOverridesFile -PythonExe $venvPy }
    finally { if ($null -eq $savedPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $savedPythonPath } }
    $made += $shadowed
    $pins = if ($shadowed) { [System.IO.File]::ReadAllText($shadowed) } else { "" }
    Check "PYTHONPATH shadow: the venv's torch is frozen" ($pins -match '(?m)^torch==2\.11\.0\+cu130\r?$')
    Check "PYTHONPATH shadow: the shadowing torch is not" ($pins -notmatch 'nv25\.9')
}
finally {
    if ($null -eq $savedOverride) { Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue }
    else { $env:UV_OVERRIDE = $savedOverride }
    foreach ($m in $made) { if ($m) { Remove-Item -LiteralPath $m -Force -ErrorAction SilentlyContinue } }
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED ($failures)" -ForegroundColor Red
    exit 1
}
Write-Host "All New-UnslothTorchOverridesFile checks passed" -ForegroundColor Green
exit 0
