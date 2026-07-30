# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types as _types

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils import parent_watchdog as pw


def test_fires_when_reparented(monkeypatch):
    calls = {"n": 0}

    def _ppid():
        calls["n"] += 1
        return 4242 if calls["n"] < 3 else 1

    monkeypatch.setattr(pw.os, "getppid", _ppid)
    fired = threading.Event()
    pw._watch_unix(4242, fired.set, threading.Event(), poll_seconds = 0.01)
    assert fired.is_set()


def test_fires_immediately_when_already_orphaned(monkeypatch):
    monkeypatch.setattr(pw.os, "getppid", lambda: 1)
    fired = threading.Event()
    assert pw.start_parent_watchdog(fired.set) is None
    assert fired.is_set()


def test_stop_event_ends_the_watch(monkeypatch):
    monkeypatch.setattr(pw.os, "getppid", lambda: 4242)
    fired = threading.Event()
    stop = pw.start_parent_watchdog(fired.set, poll_seconds = 0.01)
    assert stop is not None
    stop.set()
    time.sleep(0.05)
    assert not fired.is_set()


def test_callback_exception_is_contained():
    def _boom():
        raise RuntimeError("shutdown failed")

    pw._fire(_boom)  # must not raise


def test_explicit_owner_pid_uses_a_liveness_probe(monkeypatch):
    # Not our direct parent (e.g. a wrapper in between): probe the owner pid.
    monkeypatch.setattr(pw.os, "getppid", lambda: 4242)

    def _dead(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(pw.os, "kill", _dead)
    fired = threading.Event()
    pw._watch_unix(9999, fired.set, threading.Event(), poll_seconds = 0.01)
    assert fired.is_set()


def test_liveness_probe_treats_eperm_as_alive(monkeypatch):
    monkeypatch.setattr(pw.os, "getppid", lambda: 4242)
    calls = {"n": 0}

    def _kill(pid, sig):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError  # exists, owned by someone else: alive
        raise ProcessLookupError

    monkeypatch.setattr(pw.os, "kill", _kill)
    fired = threading.Event()
    pw._watch_unix(9999, fired.set, threading.Event(), poll_seconds = 0.01)
    assert fired.is_set()
    assert calls["n"] == 3


@pytest.mark.skipif(sys.platform == "win32", reason = "unix reparenting path")
def test_process_exits_when_parent_dies(tmp_path):
    # The callback writes a marker before exiting: liveness cannot be probed
    # with kill(pid, 0), which also succeeds while the child is an unreaped
    # zombie under a non-reaping init.
    marker = tmp_path / "fired"
    watcher = tmp_path / "watcher.py"
    watcher.write_text(
        f"""
import logging, os, sys, time, types
sys.path.insert(0, {_BACKEND_DIR!r})
stub = types.ModuleType("loggers")
stub.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", stub)
from utils.parent_watchdog import start_parent_watchdog

def _exit():
    open({str(marker)!r}, "w").write("fired")
    os._exit(0)

start_parent_watchdog(_exit, poll_seconds = 0.05)
time.sleep(30)
os._exit(1)
"""
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"""
import subprocess, sys
subprocess.Popen([sys.executable, {str(watcher)!r}])
"""
    )

    # The intermediate parent spawns the watcher and exits immediately,
    # orphaning it; the watchdog must notice and exit the watcher.
    subprocess.run([sys.executable, str(parent)], check = True, timeout = 15)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if marker.exists():
            return
        time.sleep(0.1)
    pytest.fail("watchdog never fired after the parent died")
