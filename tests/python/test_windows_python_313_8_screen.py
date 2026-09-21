# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows must not build the venv on a CPython that cannot import torch.

CPython 3.13.8 carries python/cpython#139783: inspect.getsourcelines() drops a
function body when a decorator is followed by a comment, which is the shape of
the @_overload_method blocks torch/nn/modules/rnn.py parses at import time, so
`import torch` raises IndentationError (#7803).

Windows reaches such an interpreter differently from install.sh: uv is handed a
resolved path rather than a version, so it never picks the patch itself, but
Find-CompatiblePython matches on the *minor* version and would happily return an
already-installed 3.13.8. Remove-SkippedPython is what turns that into "not
found", so the caller installs $PythonFallbackFullVersion instead.

The function is extracted from install.ps1 and executed under pwsh rather than
reimplemented, so the test cannot drift from the text the installer runs.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "pwsh is required to execute install.ps1 blocks"
)


def _extract(pattern: str) -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 no longer contains {pattern!r}"
    return match.group(0)


def _blocks() -> tuple:
    """The skip list and the screen, straight out of install.ps1.

    Hoisted out of the f-strings below: a backslash inside an f-string expression
    is a syntax error before 3.12, and this repo is 3.9+ (ruff targets py311).
    """
    return (
        _extract(
            r"    # Patch releases the stack cannot run.*?"
            r"if \(\$SkipTorch\) \{ \$PythonSkip = @\(\) \}"
        ),
        _extract(r"    function Remove-SkippedPython \{.*?\n    \}"),
    )


def _fake_python(tmp_path: Path, version: str) -> Path:
    """An executable that reports ``version`` for the resolver's probe."""
    if os.name == "nt":
        exe = tmp_path / "python.cmd"
        exe.write_text(f"@echo off\r\necho {version}\r\n", encoding = "utf-8")
        return exe
    exe = tmp_path / "python"
    exe.write_text(f'#!/bin/sh\necho "{version}"\n', encoding = "utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _run(tmp_path: Path, version: str | None) -> str:
    candidate = (
        "$null"
        if version is None
        else f'@{{ Version = "3.13"; Path = "{_fake_python(tmp_path, version)}" }}'
    )
    skip_block, screen_block = _blocks()
    script = f"""
$ErrorActionPreference = "Stop"
$SkipTorch = $false
# Write-Host, like the real substep: Write-Output would put the message on
# the pipeline, so the function would return @(message, $null) and every
# `if ($DetectedPython)` downstream would read it as truthy.
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
{skip_block}
{screen_block}
$result = Remove-SkippedPython ({candidate})
if ($null -eq $result) {{ Write-Output "RESULT: rejected" }}
else {{ Write-Output "RESULT: kept" }}
"""
    # The whole verdict is the single RESULT line this script prints, so a pwsh that aborts at startup would read as a
    # screen that reached the opposite conclusion.
    completed = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
    )
    return completed.stdout + completed.stderr


def test_a_skipped_patch_is_rejected(tmp_path):
    out = _run(tmp_path, "3.13.8")
    assert "RESULT: rejected" in out, out
    assert "cannot import torch" in out, "the user should be told why: " + out


def test_a_good_patch_of_the_same_minor_is_kept(tmp_path):
    # The screen is per patch: 3.13 itself is fine and must not be refused.
    out = _run(tmp_path, "3.13.13")
    assert "RESULT: kept" in out, out


def test_nothing_found_stays_nothing(tmp_path):
    out = _run(tmp_path, None)
    assert "RESULT: rejected" in out, out


def test_an_unreadable_interpreter_is_not_treated_as_bad(tmp_path):
    # A probe that cannot run is not evidence of a bad version, and refusing it would send a working machine down the
    # install path for no reason.
    missing = tmp_path / "does-not-exist"
    skip_block, screen_block = _blocks()
    script = f"""
$ErrorActionPreference = "Stop"
$SkipTorch = $false
# Write-Host, like the real substep: Write-Output would put the message on
# the pipeline, so the function would return @(message, $null) and every
# `if ($DetectedPython)` downstream would read it as truthy.
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
{skip_block}
{screen_block}
$result = Remove-SkippedPython (@{{ Version = "3.13"; Path = "{missing}" }})
if ($null -eq $result) {{ Write-Output "RESULT: rejected" }}
else {{ Write-Output "RESULT: kept" }}
"""
    # This case asserts the screen KEEPS an interpreter it could not probe, so an
    # interpreter that dies would masquerade as the screen wrongly rejecting it.
    completed = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
    )
    assert "RESULT: kept" in completed.stdout + completed.stderr, (
        completed.stdout + completed.stderr
    )


def test_the_resolver_is_screened_at_every_entry_point():
    """A bare Find-CompatiblePython in the install flow would defeat the screen."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    flow = source[source.index("# ── Install Python if no compatible version") :]
    flow = flow[: flow.index("# ── Install uv ──")]
    bare = [
        line.strip()
        for line in flow.splitlines()
        if "= Find-CompatiblePython" in line and "Remove-SkippedPython" not in line
    ]
    assert not bare, f"unscreened resolver calls in the install flow: {bare}"


# ── The screen inside the resolver ──
# The window above starts at the install step, so it never sees the recovery
# paths: Install-PythonFromPythonOrg and Install-X64Python both end in a bare
# `return (Find-CompatiblePython)`. Screening every candidate as it is
# enumerated is what makes those safe, and is also what lets the resolver carry
# on to its next minor instead of giving up on the machine.
def _every_version_match_screens_the_patch() -> list[str]:
    body = _extract(r"    function Find-CompatiblePython \{.*?\n    \}")
    lines = body.splitlines()
    unscreened = []
    for i, line in enumerate(lines):
        if 'match "Python' not in line:
            continue
        window = "\n".join(lines[i : i + 9])
        if "$PythonSkip -contains" not in window:
            unscreened.append(line.strip())
    return unscreened


def test_every_enumerated_candidate_is_screened():
    assert (
        len(
            [
                l
                for l in _extract(r"    function Find-CompatiblePython \{.*?\n    \}").splitlines()
                if 'match "Python' in l
            ]
        )
        == 3
    ), "the resolver's enumeration sites moved; re-check the screen"
    assert not _every_version_match_screens_the_patch(), (
        "Find-CompatiblePython enumerates a candidate without checking $PythonSkip: "
        f"{_every_version_match_screens_the_patch()}"
    )


# The launcher below is a /bin/sh script.
# Windows has no shebang and no PATHEXT entry for an extensionless file, so `Get-Command py` does not find it and the
# resolver reports "none" whatever the versions are -- which would make the negative case pass for the wrong reason.
# The PowerShell under test is the same text on every platform, and pwsh runs it here, so these three cases run on POSIX
# and the rest of the file still covers Windows.
_POSIX_LAUNCHER_ONLY = pytest.mark.skipif(
    os.name == "nt", reason = "the fake py launcher is a /bin/sh script"
)


def _fake_launcher(root: Path, versions: dict[str, str]) -> Path:
    """A `py` launcher over fake interpreters, one per minor in ``versions``."""
    root.mkdir(parents = True, exist_ok = True)
    branches = []
    for minor, full in versions.items():
        exe = root / f"python{minor.replace('.', '')}"
        # -S -c "import sys; print(sys.base_prefix)" for the conda screen.
        exe.write_text('#!/bin/sh\necho "/usr"\n', encoding = "utf-8")
        exe.chmod(0o755)
        branches.append(f'  {minor}) ver="{full}"; exe="{exe}" ;;')
    launcher = root / "py"
    launcher.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        + "\n".join(f"  -{b.lstrip()}" for b in branches)
        + "\n  *) exit 1 ;;\nesac\n"
        "shift\n"
        'case "$1" in\n'
        '  --version) echo "Python $ver" ;;\n'
        '  -S) echo "$exe" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def _resolve(tmp_path: Path, versions: dict[str, str]) -> str:
    """Run the real Find-CompatiblePython over ``versions`` and report the hit."""
    root = tmp_path / "bin"
    _fake_launcher(root, versions)
    skip_block, screen_block = _blocks()
    # Hoisted for the same reason as _blocks: a backslash in an f-string expression does not parse before 3.12.
    conda_block = _extract(r"    function Test-IsCondaPython \{.*?\n    \}")
    tag_block = _extract(r"    function Get-PythonPlatformTag \{.*?\n    \}")
    resolver_block = _extract(r"    function Find-CompatiblePython \{.*?\n    \}")
    script = f"""
$ErrorActionPreference = "Stop"
$SkipTorch = $false
$env:PATH = "{root}"
$PythonVersion = "3.13"
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
function Get-HostMachineArch {{ return "x86_64" }}
{skip_block}
$script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'
{conda_block}
{tag_block}
{resolver_block}
$found = Find-CompatiblePython
if ($null -eq $found) {{ Write-Output "RESULT: none" }}
else {{ Write-Output "RESULT: $($found.Version)" }}
"""
    # The caller scrapes the resolver's chosen minor out of stdout; a crashed pwsh leaves nothing to scrape and would
    # fail as if Find-CompatiblePython went silent.
    completed = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
    )
    out = completed.stdout + completed.stderr
    match = re.search(r"RESULT: (\S+)", out)
    assert match is not None, out
    return match.group(1)


@_POSIX_LAUNCHER_ONLY
def test_the_resolver_falls_through_to_the_next_minor(tmp_path):
    # The offline/locked-down case: 3.13.8 and a healthy 3.12 both installed and nothing installable. Ending the search
    # on the 3.13 would leave the caller with a Python that cannot import torch; refusing it outright would fail a
    # machine that has a perfectly good interpreter one entry down the list.
    assert _resolve(tmp_path, {"3.13": "3.13.8", "3.12": "3.12.11"}) == "3.12"


@_POSIX_LAUNCHER_ONLY
def test_a_good_preferred_minor_still_wins(tmp_path):
    assert _resolve(tmp_path, {"3.13": "3.13.13", "3.12": "3.12.11"}) == "3.13"


@_POSIX_LAUNCHER_ONLY
def test_nothing_usable_is_still_nothing(tmp_path):
    # Paired with a positive control over the same tree, because "none" is also
    # what a harness that cannot run the launcher at all reports: without the
    # control this case would pass on a machine where it proves nothing.
    assert _resolve(tmp_path / "good", {"3.13": "3.13.13"}) == "3.13"
    assert _resolve(tmp_path / "bad", {"3.13": "3.13.8"}) == "none"


@_POSIX_LAUNCHER_ONLY
def test_no_torch_mode_keeps_the_skipped_patch(tmp_path):
    """The list is about `import torch`; -NoTorch never imports it.

    A locked-down GGUF-only machine whose only Python is 3.13.8 would otherwise
    be pushed into winget/python.org recovery it may not be able to complete.
    """
    root = tmp_path / "bin"
    _fake_launcher(root, {"3.13": "3.13.8"})
    skip_block, screen_block = _blocks()
    conda_block = _extract(r"    function Test-IsCondaPython \{.*?\n    \}")
    tag_block = _extract(r"    function Get-PythonPlatformTag \{.*?\n    \}")
    resolver_block = _extract(r"    function Find-CompatiblePython \{.*?\n    \}")
    script = f"""
$ErrorActionPreference = "Stop"
$SkipTorch = $true
$env:PATH = "{root}"
$PythonVersion = "3.13"
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
function Get-HostMachineArch {{ return "x86_64" }}
{skip_block}
$script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda|miniforge|mambaforge)'
{conda_block}
{tag_block}
{resolver_block}
$found = Find-CompatiblePython
if ($null -eq $found) {{ Write-Output "RESULT: none" }}
else {{ Write-Output "RESULT: $($found.Version)" }}
"""
    # -NoTorch must leave 3.13.8 in place, and dying before the RESULT line is printed is indistinguishable here from
    # the screen having removed it.
    completed = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
    )
    out = completed.stdout + completed.stderr
    assert "RESULT: 3.13" in out, out
