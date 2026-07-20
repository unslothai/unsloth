# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the string-only static filesystem screen (#7248).

The pure tests need no backend deps (the screen is stdlib-only); injecting
``ntpath`` exercises Windows path semantics on a posix CI. The executor tests
confirm the screen is wired into the real python/bash tools, blocks before any
subprocess is spawned, and honours the Bypass Permissions skip.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import sandbox_static_fs as s

WD = "/work/session"
ENV = {"HOME": "/home/u"}


def _status(
    raw,
    wd = WD,
    env = ENV,
    pathmod = posixpath,
):
    return s.classify_path(raw, wd, env, pathmod = pathmod)[0]


# --- classify_path: posix ---


def test_workdir_relative_is_inside():
    assert _status("data.csv") == "inside"
    assert _status("sub/dir/data.csv") == "inside"
    assert _status(f"{WD}/out.txt") == "inside"


def test_absolute_outside():
    assert _status("/etc/passwd") == "outside"
    assert _status("/home/u/.ssh/id_rsa") == "outside"


def test_traversal_escape_vs_inside():
    assert _status("../peer/secret") == "outside"
    assert _status("sub/../data.csv") == "inside"


def test_prefix_sibling_is_not_inside():
    assert (
        s.classify_path("/work/session_evil/x", "/work/session", ENV, pathmod = posixpath)[0]
        == "outside"
    )


def test_temp_roots_inside():
    # The OS temp tree and the child's own TMPDIR count as best-effort scratch.
    assert _status("/tmp/scratch") == "inside"
    assert _status("/var/tmp/x") == "inside"
    env = {"HOME": "/home/u", "TMPDIR": "/work/session/tmp"}
    assert s.classify_path("/work/session/tmp/scratch", WD, env, pathmod = posixpath)[0] == "inside"


def test_remap_prefix_absence_gated(monkeypatch):
    real_exists = os.path.exists
    # Prefix absent -> the shim remaps it onto the workdir, so it is inside.
    monkeypatch.setattr(
        s.os.path, "exists", lambda p: False if p in s._REMAP_PREFIXES else real_exists(p)
    )
    assert _status("/workspace/x") == "inside"
    # Prefix present on the host -> not remapped -> stays outside.
    monkeypatch.setattr(s.os.path, "exists", lambda p: True)
    assert _status("/workspace/x") == "outside"


def test_home_and_var_expansion_shell():
    assert _status("~/notes") == "outside"  # ~ -> /home/u (shell expand)
    assert _status("$HOME/notes") == "outside"
    assert _status("$MISSING/x", env = {}) == "unknown"


# --- classify_path: windows via ntpath injection ---


def _wstatus(raw, wd = "C:\\work\\sess"):
    return s.classify_path(raw, wd, {"USERPROFILE": "C:\\Users\\u"}, pathmod = ntpath)[0]


def test_windows_inside_other_drive_and_system():
    assert _wstatus("data.csv") == "inside"
    assert _wstatus("C:\\work\\sess\\out.txt") == "inside"
    assert _wstatus("C:/work/sess/sub/x") == "inside"
    assert _wstatus("C:\\Windows\\system32\\x") == "outside"
    assert _wstatus("D:\\other\\x") == "outside"


# --- scan_shell ---


def test_scan_shell_flags_outside_only():
    assert s.scan_shell("cat /etc/hostname", WD, ENV, posixpath) == ["/etc/hostname"]
    assert s.scan_shell("grep secret /etc/shadow", WD, ENV, posixpath) == ["/etc/shadow"]
    assert s.scan_shell("echo hello", WD, ENV, posixpath) == []
    assert s.scan_shell("cat data.csv", WD, ENV, posixpath) == []


def test_scan_shell_glued_redirect_bypass_closed():
    # Regression: a redirect glued to the previous word must be caught like the spaced form.
    assert s.scan_shell("cat</etc/passwd", WD, ENV, posixpath) == ["/etc/passwd"]
    assert s.scan_shell("cat < /etc/passwd", WD, ENV, posixpath) == ["/etc/passwd"]
    assert s.scan_shell("wc -c</etc/hostname", WD, ENV, posixpath) == ["/etc/hostname"]
    assert s.scan_shell("prog >/etc/motd", WD, ENV, posixpath) == ["/etc/motd"]
    assert s.scan_shell("prog 2>>/etc/passwd", WD, ENV, posixpath) == ["/etc/passwd"]
    assert s.scan_shell("python x.py 2>/dev/null", WD, ENV, posixpath) == []


def test_scan_shell_command_position_exempt():
    # An absolute executable path is the command word, not a file operand.
    assert s.scan_shell("/usr/bin/env python x.py", WD, ENV, posixpath) == []
    assert s.scan_shell("/bin/ls data.csv", WD, ENV, posixpath) == []
    # ... but an outside operand after the command is still flagged.
    assert s.scan_shell("/bin/cat /etc/hostname", WD, ENV, posixpath) == ["/etc/hostname"]


def test_scan_shell_multi_command_and_pipe():
    assert s.scan_shell("echo hi; cat /etc/passwd", WD, ENV, posixpath) == ["/etc/passwd"]
    assert s.scan_shell("cat /etc/passwd | grep x", WD, ENV, posixpath) == ["/etc/passwd"]


def test_scan_shell_var_url_dev():
    # $PATH expands to a pathsep-joined list, not a single path -> not flagged.
    assert (
        s.scan_shell("echo $PATH", WD, {"HOME": "/home/u", "PATH": "/usr/bin:/bin"}, posixpath)
        == []
    )
    assert s.scan_shell("git clone https://github.com/a/b", WD, ENV, posixpath) == []
    assert s.scan_shell("cat $HOME/.ssh/id_rsa", WD, ENV, posixpath) == ["$HOME/.ssh/id_rsa"]


def test_scan_shell_windows_backslash_path():
    # Windows: posix=False tokenization keeps the backslash path intact for ntpath.
    assert s.scan_shell("type C:\\Windows\\win.ini", "C:\\work\\sess", {}, ntpath) == [
        "C:\\Windows\\win.ini"
    ]


# --- scan_python ---


def test_scan_python_flags_literal_outside(tmp_path):
    wd = str(tmp_path)
    assert s.scan_python("open('/etc/passwd')", wd, ENV, posixpath) == ["/etc/passwd"]
    assert s.scan_python("import shutil\nshutil.copy('a', '/usr/x')", wd, ENV, posixpath) == [
        "/usr/x"
    ]
    assert s.scan_python("open('data.csv')", wd, ENV, posixpath) == []


def test_scan_python_literals_not_env_expanded(tmp_path):
    wd = str(tmp_path)
    # Python does not expand ~ or $VAR in a string literal -> these are in-workdir relatives.
    assert s.scan_python("open('$HOME/x')", wd, {"HOME": "/etc"}, posixpath) == []
    assert s.scan_python("open('~/x')", wd, ENV, posixpath) == []


def test_scan_python_runtime_paths_are_allowed(tmp_path):
    wd = str(tmp_path)
    assert s.scan_python("open(f'{base}/x')", wd, ENV, posixpath) == []
    assert s.scan_python("p = os.path.join(root, 'x')\nopen(p)", wd, ENV, posixpath) == []


def test_scan_python_missing_parent_deferred_to_shim(tmp_path):
    wd = str(tmp_path)
    assert s.scan_python("open('/nonexistent_root_xyz/deep/f', 'w')", wd, ENV, posixpath) == []


# --- policy entry point + switch ---


def test_check_static_fs_messages(tmp_path):
    wd = str(tmp_path)
    msg = s.check_static_fs("shell", "cat /etc/hostname", wd, ENV, posixpath)
    assert msg and "/etc/hostname" in msg and "outside the sandbox working directory" in msg
    assert s.check_static_fs("shell", "echo hi", wd, ENV, posixpath) is None
    py = s.check_static_fs("python", "open('/etc/passwd')", wd, ENV, posixpath)
    assert py and "/etc/passwd" in py


def test_static_screen_enabled_switch():
    assert s.static_screen_enabled({}) is True
    assert s.static_screen_enabled({"UNSLOTH_STUDIO_SANDBOX_FS_CONFINE": "0"}) is False
    assert s.static_screen_enabled({"UNSLOTH_STUDIO_SANDBOX_FS_CONFINE": "auto"}) is True


# --- executor integration ---


def test_bash_exec_blocks_outside_read():
    from core.inference.tools import _bash_exec
    msg = _bash_exec("cat /etc/hostname", session_id = "static-block")
    assert "outside the sandbox working directory" in msg


def test_bash_exec_glued_redirect_blocked():
    from core.inference.tools import _bash_exec
    msg = _bash_exec("cat</etc/hostname", session_id = "static-glued")
    assert "outside the sandbox working directory" in msg


def test_bash_exec_allows_normal_command():
    from core.inference.tools import _bash_exec

    msg = _bash_exec("echo hello", session_id = "static-ok")
    assert "outside the sandbox working directory" not in msg
    assert "hello" in msg


def test_blocked_command_never_spawns(monkeypatch):
    import core.inference.tools as t

    calls = []
    real_popen = t.subprocess.Popen
    monkeypatch.setattr(
        t.subprocess, "Popen", lambda *a, **k: (calls.append(1), real_popen(*a, **k))[1]
    )
    msg = t._bash_exec("cat /etc/hostname", session_id = "never-spawn")
    assert "outside the sandbox working directory" in msg
    assert calls == []  # blocked before any Popen


def test_python_exec_blocks_outside_write():
    from core.inference.tools import _python_exec
    msg = _python_exec("open('/usr/x_evil', 'w')", session_id = "py-block")
    assert "outside the sandbox working directory" in msg


def test_bypass_permissions_skips_static_screen():
    from core.inference.tools import _bash_exec
    msg = _bash_exec("cat /etc/hostname", session_id = "static-bypass", disable_sandbox = True)
    assert "outside the sandbox working directory" not in msg
