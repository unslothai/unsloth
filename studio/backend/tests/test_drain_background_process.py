# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A backgrounded process holding stdout must not stall a call whose leader already exited."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import (
    _bash_exec,
    _capture_process_group,
    _drain_process_output,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")


def _spawn(script):
    proc = subprocess.Popen(
        ["bash", "-c", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        stdin = subprocess.DEVNULL,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        start_new_session = True,
    )
    return proc, _capture_process_group(proc)


def _group_alive(pgid):
    proc_dir = Path("/proc")
    if proc_dir.is_dir():
        for stat in proc_dir.glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            # A reparented zombie is harmless; only a running member counts.
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_group_gone(pgid):
    deadline = time.monotonic() + 3
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _group_alive(pgid), "the backgrounded process outlived the call"


def _cleanup(pgid):
    try:
        os.killpg(pgid, 9)
    except OSError:
        pass


def test_backgrounded_process_does_not_stall_finite_drain():
    proc, pgid = _spawn("sleep 30 & echo started")
    try:
        started = time.monotonic()
        output, timed_out, _ = _drain_process_output(proc, 10, None, pgid = pgid)
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"drain waited {elapsed:.1f}s for the backgrounded sleep"
        assert not timed_out
        assert proc.returncode == 0
        assert output.startswith("started\n")
        assert "nohup" in output
        _assert_group_gone(pgid)
    finally:
        _cleanup(pgid)


def test_backgrounded_process_does_not_stall_unlimited_drain():
    proc, pgid = _spawn("sleep 30 & echo started")
    result = {}

    def drain():
        result["value"] = _drain_process_output(proc, None, None, pgid = pgid)

    try:
        started = time.monotonic()
        worker = threading.Thread(target = drain, daemon = True)
        worker.start()
        worker.join(timeout = 8)
        assert not worker.is_alive(), "timeout=None drain hung on the backgrounded sleep"
        assert time.monotonic() - started < 5
        output, timed_out, _ = result["value"]
        assert not timed_out
        assert proc.returncode == 0
        assert "started" in output
        _assert_group_gone(pgid)
    finally:
        _cleanup(pgid)


@pytest.mark.parametrize("timeout", [10, None])
def test_short_lived_background_output_is_still_captured(timeout):
    proc, pgid = _spawn("( sleep 0.3; echo late ) & echo early")
    try:
        output, timed_out, _ = _drain_process_output(proc, timeout, None, pgid = pgid)
        assert not timed_out
        assert output == "early\nlate\n"
    finally:
        _cleanup(pgid)


def test_bash_exec_reports_the_command_not_a_timeout():
    started = time.monotonic()
    result = _bash_exec("sleep 30 & echo started", timeout = 10, disable_sandbox = True)
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"Terminal call stalled {elapsed:.1f}s on a backgrounded process"
    assert "started" in result
    assert "timed out" not in result
    assert not result.startswith("Exit code")
