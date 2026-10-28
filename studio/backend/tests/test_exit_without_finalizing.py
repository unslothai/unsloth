# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skip interpreter finalization, but still finish what a normal exit finishes: threads, atexit hooks, stdout."""

import subprocess
import sys
from pathlib import Path

_CHILD = """
import atexit, os, threading, time
import run

class _FinalizedOnTeardown:
    def __del__(self):
        os.write(1, b"finalized")

_sentinel = _FinalizedOnTeardown()

atexit.register(lambda: print("atexit hook"))

def _late_writer():
    # Longer than any bounded join would allow: an in-process writer must finish before the hooks.
    time.sleep(1.5)
    print("thread finished")

threading.Thread(target = _late_writer).start()
run._exit_without_finalizing()
print("unreachable")
"""


def test_threads_then_atexit_hooks_run_and_stdout_is_flushed():
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd = Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("thread finished\natexit hook\n"), result.stdout
    assert "unreachable" not in result.stdout
    assert "finalized" not in result.stdout, "interpreter finalization ran"


def test_an_unflushable_stdout_exits_120_like_cpython():
    child = """
import sys
import run

class _BrokenStream:
    def write(self, text):
        return len(text)

    def flush(self):
        raise BrokenPipeError

sys.stdout = _BrokenStream()
run._exit_without_finalizing()
"""
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd = Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 120,
    )

    assert result.returncode == 120, result.stderr
