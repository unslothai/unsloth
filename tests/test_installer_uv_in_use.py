# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install.ps1 must not blame a verified uv for a destination it could not replace (#9804)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _locate(haystack: str, needle: str, what: str) -> int:
    index = haystack.find(needle)
    if index < 0:
        pytest.fail(f"install.ps1 no longer contains {what}: {needle!r}")
    return index


def _extract_function(name: str) -> str:
    src = _install_ps1()
    start = _locate(src, f"    function {name} {{", f"the {name} helper")
    end = _locate(src[start:], "\n    }\n", f"the end of {name}") + start
    return src[start : end + len("\n    }\n")]


def test_the_release_installer_copies_through_copy_uvset():
    body = _extract_function("Install-UvFromRelease")
    assert "Copy-UvSet -Work $srcRoot -DestDir $destDir" in body
    assert body.count("could not run on this machine") == 1
    assert "could not be replaced" in body
    assert "in use" in body


def test_copy_uvset_skips_a_destination_that_already_holds_the_verified_bytes():
    body = _extract_function("Copy-UvSet")
    skip = _locate(
        body,
        "if (Test-UvFileMatches -Source $src -Destination $dst) { continue }",
        "the identical-bytes skip",
    )
    copy = _locate(
        body, "Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop", "the copy"
    )
    assert skip < copy, "the skip must be decided before the copy is attempted"


def test_copy_uvset_retries_a_transient_lock_then_names_the_file():
    body = _extract_function("Copy-UvSet")
    assert "for ($attempt = 1; $attempt -le 3; $attempt++)" in body
    assert "$script:UvCopyBlockedAt = $dst" in body


def _run_copy_uvset(
    tmp_path: Path,
    prelude: str,
    epilogue: str = "",
) -> subprocess.CompletedProcess:
    script = (
        _extract_function("Test-UvFileMatches")
        + _extract_function("Copy-UvSet")
        + prelude
        + "\n$ok = Copy-UvSet -Work $work -DestDir $dest\n"
        + '"ok=$ok blocked=$($script:UvCopyBlockedAt)"\n'
        + epilogue
    )
    path = tmp_path / "probe.ps1"
    path.write_text(script, encoding = "utf-8")
    return run_pwsh(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(path)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        check = True,
    )


@requires_pwsh
def test_copy_uvset_puts_the_set_in_place(tmp_path):
    work, dest = tmp_path / "work", tmp_path / "dest"
    work.mkdir()
    dest.mkdir()
    for name in ("uv.exe", "uvx.exe", "uvw.exe"):
        (work / name).write_bytes(b"pinned " + name.encode())
    res = _run_copy_uvset(tmp_path, f"$work = '{work}'\n$dest = '{dest}'\n")
    assert "ok=True blocked=" in res.stdout, res.stdout + res.stderr
    assert (dest / "uv.exe").read_bytes() == b"pinned uv.exe"
    assert (dest / "uvw.exe").read_bytes() == b"pinned uvw.exe"


@requires_pwsh
def test_copy_uvset_leaves_an_identical_locked_destination_alone(tmp_path):
    work, dest = tmp_path / "work", tmp_path / "dest"
    work.mkdir()
    dest.mkdir()
    (work / "uv.exe").write_bytes(b"pinned uv.exe")
    (dest / "uv.exe").write_bytes(b"pinned uv.exe")
    prelude = (
        f"$work = '{work}'\n$dest = '{dest}'\n"
        # A running exe admits readers and refuses writers, hence share Read.
        "$h = [System.IO.File]::Open((Join-Path $dest 'uv.exe'), 'Open', 'Read', 'Read')\n"
    )
    res = _run_copy_uvset(tmp_path, prelude, "$h.Dispose()\n")
    assert "ok=True blocked=" in res.stdout, res.stdout + res.stderr


@requires_pwsh
def test_copy_uvset_names_a_locked_destination_it_could_not_replace(tmp_path):
    work, dest = tmp_path / "work", tmp_path / "dest"
    work.mkdir()
    dest.mkdir()
    (work / "uv.exe").write_bytes(b"pinned uv.exe")
    (dest / "uv.exe").write_bytes(b"older uv.exe")
    prelude = (
        f"$work = '{work}'\n$dest = '{dest}'\n"
        "$h = [System.IO.File]::Open((Join-Path $dest 'uv.exe'), 'Open', 'Read', 'None')\n"
    )
    res = _run_copy_uvset(tmp_path, prelude, "$h.Dispose()\n")
    assert "ok=False" in res.stdout, res.stdout + res.stderr
    assert str(dest / "uv.exe") in res.stdout
    assert (dest / "uv.exe").read_bytes() == b"older uv.exe", "a refused copy must not half-write"
