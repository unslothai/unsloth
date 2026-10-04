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


@pytest.mark.skipif(
    sys.platform == "win32", reason = "dispatches to the Windows watcher, which opens the real pid"
)
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


def test_fires_immediately_when_the_owner_is_not_the_current_parent(monkeypatch):
    # Explicit owner pid but already reparented: the owner died during backend
    # startup (reaped or zombie alike). Must fire before the first wait.
    monkeypatch.setattr(pw.os, "getppid", lambda: 7777)
    fired = threading.Event()
    started = time.monotonic()
    pw._watch_unix(4242, fired.set, threading.Event(), poll_seconds = 30)
    assert fired.is_set()
    assert time.monotonic() - started < 1


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

# Explicit owner pid, like the production handshake: under a child-subreaper
# the parent can die and reparent us before this line runs, and a getppid
# sample here would watch the subreaper forever.
start_parent_watchdog(_exit, parent_pid = int(sys.argv[1]), poll_seconds = 0.05)
time.sleep(30)
os._exit(1)
"""
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"""
import os, subprocess, sys
subprocess.Popen([sys.executable, {str(watcher)!r}, str(os.getpid())])
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
