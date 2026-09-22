# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The browser matrix keeps TMPDIR short, whatever the checkout is called.

Branded Chrome and Edge create `$TMPDIR/com.google.Chrome.XXXXXX/SingletonSocket` at launch
and abort with "Socket path too long" when that passes the 107 usable bytes of a unix socket
path. With TMPDIR under the checkout the length tracked the repository name:
`/home/runner/work/unsloth/unsloth/...` came to 100 bytes, and a fork named
`unsloth-staging-2` to 120, which failed every chrome and msedge leg before a page loaded.
Chromium, firefox and webkit create no such socket, so the break showed up as two browsers
out of five.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "studio-composer-compatibility.yml"
# sockaddr_un.sun_path less its terminating NUL, per hosted runner: 108 bytes on Linux, 104 on macOS.
SUN_PATH_MAX = {"linux": 107, "macos": 103, "windows": 107}
# What Chrome appends to TMPDIR, at its length.
CHROME_SOCKET = "/com.google.Chrome.XXXXXX/SingletonSocket"


def _workflow_tmpdirs() -> dict[str, str]:
    text = WORKFLOW.read_text(encoding = "utf-8")
    found = dict(re.findall(r'echo "(TMPDIR|TMP|TEMP)=([^"]+)" >> "\$GITHUB_ENV"', text))
    assert set(found) == {
        "TMPDIR",
        "TMP",
        "TEMP",
    }, f"the workflow stopped exporting a temp dir: {found}"
    return found


def test_the_workflow_temp_dir_does_not_grow_with_the_checkout():
    for name, value in _workflow_tmpdirs().items():
        assert "GITHUB_WORKSPACE" not in value and "github.workspace" not in value, (
            f"{name}={value} sits under the checkout, so Chrome's socket path grows with the "
            "repository name"
        )
        assert value.startswith("$RUNNER_TEMP/"), f"{name}={value}"


def test_the_workflow_socket_path_fits_on_the_hosted_runners():
    # RUNNER_TEMP on the hosted images: Linux, macOS and Windows (Git Bash spelling).
    for host, runner_temp in (
        ("linux", "/home/runner/work/_temp"),
        ("macos", "/Users/runner/work/_temp"),
        ("windows", "D:/a/_temp"),
    ):
        for name, value in _workflow_tmpdirs().items():
            path = value.replace("$RUNNER_TEMP", runner_temp) + CHROME_SOCKET
            assert len(path) <= SUN_PATH_MAX[host], f"{name} on {host}: {path} is {len(path)} bytes"


def _runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tests" / "studio"))
    sys.modules.pop("run_composer_compatibility", None)
    import run_composer_compatibility as runner

    return runner


def _fits(path: Path, limit: int) -> bool:
    return len(os.fsencode(path)) + len(CHROME_SOCKET) <= limit


@pytest.mark.skipif(os.name == "nt", reason = "Chrome's singleton is a named pipe on Windows")
def test_the_local_runner_uses_a_short_inherited_temp_dir(monkeypatch):
    runner = _runner(monkeypatch)
    short = Path(tempfile.mkdtemp(prefix = "s", dir = "/tmp"))
    try:
        monkeypatch.setattr(runner.tempfile, "tempdir", str(short))
        first = runner.browser_tmpdir()
        second = runner.browser_tmpdir()
        assert first.parent == short and second.parent == short, "a short TMPDIR was not honoured"
        assert first != second, "two runs must not share a temp dir"
        assert _fits(first, runner.sun_path_max())
    finally:
        shutil.rmtree(short, ignore_errors = True)


@pytest.mark.skipif(os.name == "nt", reason = "Chrome's singleton is a named pipe on Windows")
def test_the_local_runner_falls_back_when_the_inherited_temp_dir_is_too_long(tmp_path, monkeypatch):
    runner = _runner(monkeypatch)
    deep = tmp_path / ("d" * 80)
    deep.mkdir()
    monkeypatch.setattr(runner.tempfile, "tempdir", str(deep))
    made = runner.browser_tmpdir()
    try:
        assert _fits(made, runner.sun_path_max()), f"{made} leaves Chrome's socket path too long"
        assert made.parent == Path("/tmp")
        assert not any(deep.iterdir()), "the rejected temp dir was left behind"
    finally:
        shutil.rmtree(made, ignore_errors = True)


@pytest.mark.skipif(os.name == "nt", reason = "Chrome's singleton is a named pipe on Windows")
def test_the_local_runner_never_uses_a_temp_dir_inside_the_checkout(monkeypatch):
    runner = _runner(monkeypatch)
    inside = Path(tempfile.mkdtemp(prefix = ".t", dir = ROOT))
    try:
        monkeypatch.setattr(runner.tempfile, "tempdir", str(inside))
        made = runner.browser_tmpdir()
        try:
            assert ROOT not in made.resolve().parents, f"{made} is under the checkout"
        finally:
            shutil.rmtree(made, ignore_errors = True)
    finally:
        shutil.rmtree(inside, ignore_errors = True)


@pytest.mark.skipif(os.name == "nt", reason = "Chrome's singleton is a named pipe on Windows")
def test_the_local_runner_applies_the_macos_limit_on_macos(monkeypatch):
    """A path of 104 to 107 bytes fits Linux's sun_path and not macOS's, so it must fall back there."""
    runner = _runner(monkeypatch)
    # mkdtemp adds "/uqv-" plus 8 characters: pick the parent so the socket path lands on 105.
    parent_len = 105 - len(CHROME_SOCKET) - len("/uqv-") - 8
    base = Path(tempfile.mkdtemp(prefix = "m", dir = "/tmp"))
    try:
        parent = Path(str(base) + "/" + "p" * (parent_len - len(str(base)) - 1))
        parent.mkdir()
        assert len(str(parent)) == parent_len
        monkeypatch.setattr(runner.tempfile, "tempdir", str(parent))
        monkeypatch.setattr(runner.sys, "platform", "linux")
        on_linux = runner.browser_tmpdir()
        assert on_linux.parent == parent, "105 bytes fits Linux and should have been kept"
        monkeypatch.setattr(runner.sys, "platform", "darwin")
        on_macos = runner.browser_tmpdir()
        try:
            assert on_macos.parent == Path(
                "/tmp"
            ), "105 bytes is over macOS's limit and must fall back"
            assert _fits(on_macos, 103)
        finally:
            shutil.rmtree(on_macos, ignore_errors = True)
    finally:
        shutil.rmtree(base, ignore_errors = True)
