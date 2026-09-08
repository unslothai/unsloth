#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The ownership gate must still recognise the layouts older installers left: on Windows every
# sentinel a root can show postdates the bin\ shim dir, while install.ps1 still migrates .venv.
# What is INSIDE a venv proves ownership only at the managed root, so every case names its mode.
# The uninstaller body writes to the registry, so the helpers are lifted out by AST.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_legacy_layout_gate.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)

# _IsStudioRoot calls _IsUnslothCmdShim, so both must come across or the suite is vacuous.
$allFns = $ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }, $true)
foreach ($name in @("_IsUnslothCmdShim", "_IsStudioRoot")) {
    $fn = $allFns | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if (-not $fn) {
        Write-Host "  FAIL  $name not found in uninstall.ps1" -ForegroundColor Red
        exit 1
    }
    . ([scriptblock]::Create($fn.Extent.Text))
}

$shimText = @"
@echo off
rem unsloth-studio-managed-launcher
"%~dp0..\unsloth_studio\Scripts\python.exe" -X utf8 -c "from unsloth_cli import app" %*
"@

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-legacy-gate-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
    function Make([string]$Name, [string[]]$Files) {
        $root = Join-Path $tmp $Name
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        foreach ($f in $Files) {
            $p = Join-Path $root $f
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($p)) -Force | Out-Null
            if ($f -eq "bin\unsloth.cmd") { Set-Content -LiteralPath $p -Value $shimText }
            else { Set-Content -LiteralPath $p -Value "x" }
        }
        # Every one of these stands for an install that holds chat history.
        Set-Content -LiteralPath (Join-Path $root "studio.db") -Value "chat history"
        return $root
    }

    # -ManagedDefaultRoot is passed for %USERPROFILE%\.unsloth\studio and for nothing else.
    Check "current layout (venv owner marker)" `
        (_IsStudioRoot (Make "cur-marker" @("unsloth_studio\.unsloth-studio-owned")) -ManagedDefaultRoot)
    Check "shim .exe" `
        (_IsStudioRoot (Make "cur-exe" @("bin\unsloth.exe")) -ManagedDefaultRoot)
    Check "shim .cmd alone, when a policy quarantined the .exe" `
        (_IsStudioRoot (Make "cur-cmd" @("bin\unsloth.cmd")) -ManagedDefaultRoot)

    Check "legacy .venv carrying our owner marker" `
        (_IsStudioRoot (Make "old-marker" @(".venv\.unsloth-studio-owned", ".venv\Scripts\python.exe")) -ManagedDefaultRoot)
    Check "legacy .venv carrying the unsloth console script" `
        (_IsStudioRoot (Make "old-cli" @(".venv\Scripts\python.exe", ".venv\Scripts\unsloth.exe")) -ManagedDefaultRoot)
    Check "pre-marker unsloth_studio venv carrying the unsloth console script" `
        (_IsStudioRoot (Make "pre-marker" @("unsloth_studio\Scripts\python.exe", "unsloth_studio\Scripts\unsloth.exe")) -ManagedDefaultRoot)
    # Antivirus takes the .exe out of a venv that still runs, leaving nothing but the package.
    Check "pre-marker venv whose unsloth.exe antivirus quarantined" `
        (_IsStudioRoot (Make "quarantined" @(".venv\Scripts\python.exe", ".venv\Lib\site-packages\unsloth_cli\__init__.py")) -ManagedDefaultRoot)

    # A marker is proof wherever the root sits; it is the only thing a custom root can offer.
    Check "a custom root with the venv owner marker" `
        (_IsStudioRoot (Make "custom-marker" @("unsloth_studio\.unsloth-studio-owned")))
    Check "a custom root with share\studio.conf" `
        (_IsStudioRoot (Make "custom-conf" @("share\studio.conf")))

    Check "a hand-made directory is refused" `
        (-not (_IsStudioRoot (Make "scratch" @("notes.md")) -ManagedDefaultRoot))
    Check "an ordinary project venv is refused" `
        (-not (_IsStudioRoot (Make "plain-venv" @(".venv\Scripts\python.exe", "notes.md")) -ManagedDefaultRoot))
    Check "a venv merely NAMED unsloth_studio is refused" `
        (-not (_IsStudioRoot (Make "named-venv" @("unsloth_studio\Scripts\python.exe")) -ManagedDefaultRoot))
    Check "somebody else's bin\unsloth.cmd is refused" `
        (-not (_IsStudioRoot (Make "foreign-cmd" @("bin\python.exe")) -ManagedDefaultRoot))
    # Why the flag exists: pip writes Scripts\unsloth.exe into ANY venv with the wheel.
    Check "a project venv with unsloth pip-installed is refused as a custom root" `
        (-not (_IsStudioRoot (Make "custom-project" @(".venv\Scripts\python.exe", ".venv\Scripts\unsloth.exe", "pyproject.toml"))))
    Check "that project's site-packages is refused as a custom root too" `
        (-not (_IsStudioRoot (Make "custom-pkg" @(".venv\Scripts\python.exe", ".venv\Lib\site-packages\unsloth_cli\__init__.py"))))
    $foreign = Make "foreign-shim" @()
    New-Item -ItemType Directory -Path (Join-Path $foreign "bin") -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $foreign "bin\unsloth.cmd") -Value "@echo off`r`npython -m mytool %*`r`n"
    Check "a bin\unsloth.cmd without the trampoline is refused" (-not (_IsStudioRoot $foreign -ManagedDefaultRoot))
    Check "a missing path is refused" (-not (_IsStudioRoot (Join-Path $tmp "does-not-exist") -ManagedDefaultRoot))
    Check "an empty path is refused" (-not (_IsStudioRoot "" -ManagedDefaultRoot))
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "all checks passed"
