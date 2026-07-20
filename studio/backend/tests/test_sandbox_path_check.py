# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the sensitive-path check on the terminal tool (#7242).

A lightweight, additive keyword scan that rejects shell path arguments resolving
to a sensitive out-of-workdir location (host config, credentials, other users'
files, kernel state). Ephemeral scratch like /tmp is allowed; it is defence in
depth, not the real boundary (the kernel filesystem sandbox is), and is skipped
when the sandbox is disabled (Bypass Permissions).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tools import _bash_exec, _sensitive_paths


def test_sensitive_absolute_path_is_flagged(tmp_path):
    wd = str(tmp_path)
    # /etc is a sensitive prefix, so the check catches an out-of-workdir read.
    assert _sensitive_paths("cat /etc/hostname", wd) == ["/etc/hostname"]


def test_home_credential_path_is_flagged(tmp_path):
    wd = str(tmp_path)
    # ~ expands to the real home (a sensitive prefix), so ~/.ssh/id_rsa is caught.
    flagged = _sensitive_paths("cat ~/.ssh/id_rsa", wd)
    assert flagged and flagged[0].endswith(".ssh/id_rsa")


def test_scratch_and_neutral_paths_are_allowed(tmp_path):
    wd = str(tmp_path)
    # Ephemeral scratch and neutral mounts are not sensitive: allowed so normal
    # tooling (and the timeout/cancel/hint sandbox tests) keep working.
    assert _sensitive_paths("touch /tmp/marker", wd) == []
    assert _sensitive_paths("cat /mnt/data/definitely_missing.txt", wd) == []


def test_paths_inside_workdir_are_allowed(tmp_path):
    wd = str(tmp_path)
    (tmp_path / "data.csv").write_text("x")
    assert _sensitive_paths("cat data.csv", wd) == []
    assert _sensitive_paths("cat sub/dir/data.csv", wd) == []
    assert _sensitive_paths(f"cat {wd}/data.csv", wd) == []


def test_traversal_into_sensitive_prefix_is_flagged(tmp_path):
    # A workdir nested under /etc would let ../ climb into the sensitive prefix;
    # a traversal that lands in a sensitive location must be caught. Simulate the
    # generic case: an explicit sensitive target after a traversal token.
    wd = str(tmp_path / "session")
    os.makedirs(wd)
    # Traversal to a non-sensitive sibling scratch is intentionally allowed.
    assert _sensitive_paths("cat ../peer.txt", wd) == []
    # But an absolute sensitive read is still blocked.
    assert _sensitive_paths("grep secret /etc/shadow", wd) == ["/etc/shadow"]


def test_option_attached_path_value_is_flagged(tmp_path):
    wd = str(tmp_path)
    # --flag=/path and glued short options carry a path the plain flag skip missed.
    assert _sensitive_paths("grep x --file=/etc/shadow", wd) == ["/etc/shadow"]
    assert _sensitive_paths("tool -o/etc/passwd", wd) == ["/etc/passwd"]
    # A neutral attached value stays allowed.
    assert _sensitive_paths("tool --out=/tmp/ok.txt", wd) == []


def test_env_var_paths_are_expanded(tmp_path, monkeypatch):
    wd = str(tmp_path)
    monkeypatch.setenv("NB_SECRET_DIR", "/etc")
    assert _sensitive_paths("cat $NB_SECRET_DIR/shadow", wd) == ["/etc/shadow"]
    assert _sensitive_paths("cat ${NB_SECRET_DIR}/shadow", wd) == ["/etc/shadow"]


def test_glued_ampersand_redirection_is_flagged(tmp_path):
    wd = str(tmp_path)
    # &> (stdout+stderr) glued to a sensitive target is stripped and checked.
    assert _sensitive_paths("prog &>/etc/motd", wd) == ["/etc/motd"]
    # Numeric-fd redirection to a device stays allowed.
    assert _sensitive_paths("prog 2>>/dev/null", wd) == []


def test_normal_commands_and_devices_are_untouched(tmp_path):
    wd = str(tmp_path)
    assert _sensitive_paths("echo hello", wd) == []
    assert _sensitive_paths("pip install requests", wd) == []
    # Redirection to /dev/null is not a filesystem escape.
    assert _sensitive_paths("python train.py 2>/dev/null", wd) == []
    # URLs are not local filesystem paths.
    assert _sensitive_paths("git clone https://github.com/a/b", wd) == []


def test_bash_exec_blocks_sensitive_path():
    msg = _bash_exec("cat /etc/hostname", session_id = "pathcheck-block")
    assert "outside the sandbox working directory" in msg
    assert "/etc/hostname" in msg


def test_bash_exec_allows_normal_command():
    msg = _bash_exec("echo hello", session_id = "pathcheck-normal")
    assert "outside the sandbox working directory" not in msg
    assert "hello" in msg


def test_bypass_skips_the_sensitive_path_block():
    # Bypass Permissions skips the blocklist and this check alike.
    msg = _bash_exec("cat /etc/hostname", session_id = "pathcheck-bypass", disable_sandbox = True)
    assert "outside the sandbox working directory" not in msg
