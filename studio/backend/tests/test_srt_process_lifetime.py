# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native process ownership across the SRT helper acknowledgement boundary."""

import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from core.inference import tools


def test_srt_launcher_is_terminated_before_helper(monkeypatch):
    calls = []
    parent = ("windows-job", SimpleNamespace(terminate = lambda: calls.append("helper")))
    monkeypatch.setattr(tools, "_capture_parent_process_group", lambda *a, **kw: parent)
    monkeypatch.setattr(tools, "_windows_pid_identity", lambda pid: "same-process")
    monkeypatch.setattr(tools, "_windows_taskkill_tree", lambda *args: calls.append("launcher"))
    if hasattr(os, "killpg"):
        monkeypatch.setattr(os, "killpg", lambda *args: calls.append("launcher"))
    captured = tools._capture_process_group(SimpleNamespace(_srt_launcher_pid = 43210))
    tools._killpg_captured(captured)
    assert calls == ["launcher", "helper"]


@pytest.mark.parametrize("value", [None, 0, 1, True, "43210"])
def test_only_trusted_process_ids_add_a_launcher_target(monkeypatch, value):
    parent = object()
    monkeypatch.setattr(tools, "_capture_parent_process_group", lambda *a, **kw: parent)
    assert tools._capture_process_group(SimpleNamespace(_srt_launcher_pid = value)) is parent


def test_real_launcher_started_before_capture_dies_with_helper(tmp_path):
    # The child already exists when Studio receives the helper acknowledgement.
    # On POSIX it has a separate session; on Windows a later helper Job does not
    # retroactively adopt it. Both are real OS ownership boundaries here.
    script = tmp_path / "helper.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        "start_new_session=os.name != 'nt')\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        start_new_session = os.name != "nt",
    )
    child = None
    captured = None
    try:
        launcher = int(proc.stdout.readline())
        child = psutil.Process(launcher)
        assert child.is_running()
        proc._srt_launcher_pid = launcher
        captured = tools._capture_process_group(proc)
        assert captured[0] == "srt-tree"
        if os.name == "nt":
            assert captured[3] is not None
        tools._killpg_captured(captured)
        proc.wait(timeout = 5)
        child.wait(timeout = 5)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if child is not None:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout = 5)
        proc.stdout.close()
        proc.stderr.close()


@pytest.mark.skipif(os.name != "nt", reason = "Windows Job Object semantics")
def test_srt_startup_job_allows_explicit_breakaway_and_kills_on_close():
    code = (
        "import subprocess, sys, time\n"
        "input()\n"
        "child = subprocess.run([sys.executable, '-c', \"print('BREAKAWAY_OK')\"], "
        "creationflags=0x01000000, capture_output=True, text=True, check=True)\n"
        "print(child.stdout.strip(), flush=True)\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    job = None
    try:
        job = tools._windows_job_capture(proc, allow_breakaway = True)
        assert job is not None
        proc.stdin.write("ready\n")
        proc.stdin.flush()
        assert proc.stdout.readline().strip() == "BREAKAWAY_OK"
        assert proc.poll() is None
        job.close()
        proc.wait(timeout = 5)
    finally:
        if job is not None:
            job.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout = 5)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
