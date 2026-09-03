#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# scripts/uninstall.ps1 must remove the same ~/.unsloth artifacts as scripts/uninstall.sh.
#
# The two uninstallers are maintained separately, so a prebuilt added to the
# POSIX side (whisper.cpp, the node/whisper install locks) silently keeps
# leaking on native Windows until someone notices. setup.ps1 installs
# whisper.cpp under %USERPROFILE%\.unsloth\whisper.cpp exactly like setup.sh,
# and a single surviving zero-byte lock keeps the empty-dir prune of
# ~/.unsloth from running, so the whole tree stays on disk.
#
# The uninstaller body kills processes and writes to the registry, so it cannot
# be executed here; this asserts PARITY of the artifact list instead, plus that
# uninstall.ps1 still parses.
#
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_prebuilt_parity.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$shPath  = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.sh")
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# uninstall.ps1 must parse (a syntax error would make every Windows uninstall a no-op).
$tokens = $null; $errors = $null
[System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors) | Out-Null
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)
if ($errors -and $errors.Count -gt 0) { $errors | ForEach-Object { Write-Host "    $_" } }

$shText  = Get-Content -LiteralPath $shPath -Raw
$ps1Text = Get-Content -LiteralPath $ps1Path -Raw

# Every `_remove_path "$HOME/.unsloth/<name>"` in the POSIX uninstaller. Only
# direct children: nested paths are removed with their parent.
$shArtifacts = [regex]::Matches($shText, '_remove_path\s+"\$HOME/\.unsloth/([^"/]+)"') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
Check "found the POSIX artifact list" ($shArtifacts.Count -ge 5)

# WSL-only helpers have no native-Windows counterpart.
#
# The Windows-on-ARM + NVIDIA route runs the whole install inside WSL, so these
# six live at /root/.unsloth/<name> in the distro and are never written to
# %USERPROFILE%\.unsloth: install.ps1 creates provision_llama_cuda.sh,
# run_llama_build.sh, llama_cuda_build.log, .skip-wsl-windows-shortcut and
# unsloth-install.sh through the WSL tunnel, and setup.sh writes .install-ok.
# uninstall.ps1 already clears them with its `rm -rf /root/.unsloth` inside the
# distro; a native-Windows removal would be dead code. Two of them (the
# provision script and its log) match the substring test only because that same
# WSL command line names them, which is an accident, not parity, so they are
# listed here too rather than left passing for the wrong reason.
$wslOnly = @(
    "librocdxg", "rocm-smoketest",
    "provision_llama_cuda.sh", "run_llama_build.sh", "llama_cuda_build.log",
    ".skip-wsl-windows-shortcut", ".install-ok", "unsloth-install.sh"
)
foreach ($artifact in $shArtifacts) {
    if ($wslOnly -contains $artifact) { continue }
    Check "uninstall.ps1 removes ~/.unsloth/$artifact" ($ps1Text -match [regex]::Escape($artifact))
}

# The stale-lock sweep must exist on both sides: a crash between the rename and
# the unlink in install_node_prebuilt.py strands a `.stale.<pid>` file, and that
# alone blocks the ~/.unsloth prune.
Check "uninstall.sh sweeps stale install locks"  ($shText  -match '\.install\.lock\.stale\.')
Check "uninstall.ps1 sweeps stale install locks" ($ps1Text -match '\.install\.lock\.stale\.')

# The prune must stay conditional on the dir being empty, so user content is kept.
Check "uninstall.ps1 prunes ~/.unsloth only when empty" `
    ($ps1Text -match 'Get-ChildItem -LiteralPath \$defaultUnslothHome')

if ($failures -gt 0) { Write-Host ""; Write-Host "FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host ""; Write-Host "All tests passed."; exit 0
