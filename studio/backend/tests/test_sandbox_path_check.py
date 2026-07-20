# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the out-of-workdir path check on the terminal tool (#7242).

A lightweight, additive keyword scan that rejects shell path arguments whose
realpath escapes the session working directory. It is defence in depth, not the
real boundary (the kernel filesystem sandbox is), and is skipped when the
sandbox is disabled (Bypass Permissions).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tools import _bash_exec, _paths_outside_workdir


def test_absolute_path_outside_workdir_is_flagged(tmp_path):
    wd = str(tmp_path)
    # /etc/hostname is not a blocklisted credential path, so only the new check
    # catches it.
    assert _paths_outside_workdir("cat /etc/hostname", wd) == ["/etc/hostname"]


def test_paths_inside_workdir_are_allowed(tmp_path):
    wd = str(tmp_path)
    (tmp_path / "data.csv").write_text("x")
    assert _paths_outside_workdir("cat data.csv", wd) == []
    assert _paths_outside_workdir("cat sub/dir/data.csv", wd) == []
    assert _paths_outside_workdir(f"cat {wd}/data.csv", wd) == []


def test_relative_traversal_escape_is_flagged(tmp_path):
    wd = str(tmp_path / "session")
    os.makedirs(wd)
    outside = _paths_outside_workdir("cat ../secret.txt", wd)
    assert outside and outside[0].endswith("secret.txt")
    # It resolved above the workdir.
    assert not outside[0].startswith(os.path.realpath(wd) + os.sep)


def test_normal_commands_and_devices_are_untouched(tmp_path):
    wd = str(tmp_path)
    assert _paths_outside_workdir("echo hello", wd) == []
    assert _paths_outside_workdir("pip install requests", wd) == []
    # Redirection to /dev/null is not a filesystem escape.
    assert _paths_outside_workdir("python train.py 2>/dev/null", wd) == []
    # URLs are not local filesystem paths.
    assert _paths_outside_workdir("git clone https://github.com/a/b", wd) == []


def test_bash_exec_blocks_out_of_workdir_path():
    msg = _bash_exec("cat /etc/hostname", session_id = "pathcheck-block")
    assert "outside the sandbox working directory" in msg
    assert "/etc/hostname" in msg


def test_bash_exec_allows_normal_command():
    msg = _bash_exec("echo hello", session_id = "pathcheck-normal")
    assert "outside the sandbox working directory" not in msg
    assert "hello" in msg


def test_bypass_skips_the_out_of_workdir_block():
    # Bypass Permissions skips the blocklist and this check alike.
    msg = _bash_exec(
        "cat /etc/hostname", session_id = "pathcheck-bypass", disable_sandbox = True
    )
    assert "outside the sandbox working directory" not in msg
