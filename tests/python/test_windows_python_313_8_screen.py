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
import subprocess
from pathlib import Path

import pytest


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
    script = f"""
$ErrorActionPreference = "Stop"
# Write-Host, like the real substep: Write-Output would put the message on
# the pipeline, so the function would return @(message, $null) and every
# `if ($DetectedPython)` downstream would read it as truthy.
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
{_extract(r'    # Patch releases the stack cannot run.*?\$PythonSkip = @\([^\)]*\)')}
{_extract(r'    function Remove-SkippedPython \{.*?\n    \}')}
$result = Remove-SkippedPython ({candidate})
if ($null -eq $result) {{ Write-Output "RESULT: rejected" }}
else {{ Write-Output "RESULT: kept" }}
"""
    completed = subprocess.run(
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
    # A probe that cannot run is not evidence of a bad version, and refusing it
    # would send a working machine down the install path for no reason.
    missing = tmp_path / "does-not-exist"
    script = f"""
$ErrorActionPreference = "Stop"
# Write-Host, like the real substep: Write-Output would put the message on
# the pipeline, so the function would return @(message, $null) and every
# `if ($DetectedPython)` downstream would read it as truthy.
function substep {{ param($m, $c) Write-Host "SUBSTEP: $m" }}
{_extract(r'    # Patch releases the stack cannot run.*?\$PythonSkip = @\([^\)]*\)')}
{_extract(r'    function Remove-SkippedPython \{.*?\n    \}')}
$result = Remove-SkippedPython (@{{ Version = "3.13"; Path = "{missing}" }})
if ($null -eq $result) {{ Write-Output "RESULT: rejected" }}
else {{ Write-Output "RESULT: kept" }}
"""
    completed = subprocess.run(
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
