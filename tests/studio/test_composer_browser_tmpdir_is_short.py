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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "studio-composer-compatibility.yml"
# sockaddr_un.sun_path is 108 bytes on Linux, including the terminating NUL.
SUN_PATH_MAX = 107
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
    for runner_temp in ("/home/runner/work/_temp", "/Users/runner/work/_temp", "D:/a/_temp"):
        for name, value in _workflow_tmpdirs().items():
            path = value.replace("$RUNNER_TEMP", runner_temp) + CHROME_SOCKET
            assert len(path) <= SUN_PATH_MAX, f"{name}: {path} is {len(path)} bytes"


def test_the_local_runner_puts_the_temp_dir_outside_the_checkout(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tests" / "studio"))
    sys.modules.pop("run_composer_compatibility", None)
    import run_composer_compatibility as runner

    monkeypatch.setattr(runner.tempfile, "tempdir", str(tmp_path))
    first = runner.browser_tmpdir()
    second = runner.browser_tmpdir()
    assert first.is_dir() and second.is_dir()
    assert first != second, "two runs must not share a temp dir"
    for made in (first, second):
        assert made.parent == tmp_path, "the temp dir did not follow the system temp dir"
        assert ROOT not in made.parents, f"{made} is under the checkout"
    # A short system temp dir is the premise; the name the runner adds must not use up the budget.
    assert len(os.fspath(first)) - len(os.fspath(tmp_path)) < 16
