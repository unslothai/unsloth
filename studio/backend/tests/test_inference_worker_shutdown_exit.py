# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from studio.backend.core.inference import worker  # noqa: E402
from utils.worker_stderr import WorkerStderrCapture  # noqa: E402


class _Queue:
    def __init__(self, raises: bool = False):
        self.raises = raises
        self.closed = False
        self.joined = False

    def close(self):
        self.closed = True
        if self.raises:
            raise OSError("already closed")

    def join_thread(self):
        self.joined = True


def _capture_exit(monkeypatch):
    codes: list[int] = []
    monkeypatch.setattr(worker.os, "_exit", lambda code: codes.append(code))
    return codes


def test_shutdown_exits_the_process_after_flushing(monkeypatch):
    codes = _capture_exit(monkeypatch)
    queue = _Queue()

    worker._exit_after_shutdown(queue)

    assert codes == [0]
    assert queue.closed and queue.joined


def test_shutdown_still_exits_when_the_queue_is_already_closed(monkeypatch):
    codes = _capture_exit(monkeypatch)

    worker._exit_after_shutdown(_Queue(raises = True))

    assert codes == [0]


_MIRRORED_CHILD = r"""
import sys
sys.path.insert(0, %(backend)r)
from core.inference import worker
from utils.worker_stderr import install_worker_stderr_mirror

class Q:
    def close(self): pass
    def join_thread(self): pass

assert install_worker_stderr_mirror(%(path)r) is True
for i in range(200):
    sys.stderr.write(f"bulk {i}\n")
sys.stderr.write("the last line before shutdown\n")
worker._exit_after_shutdown(Q())
"""


def test_shutdown_exit_still_relays_the_mirrored_stderr_tail(tmp_path):
    # os._exit skips the atexit drain, which dropped every line the mirror pump had not relayed yet.
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    script = _MIRRORED_CHILD % {"backend": str(_BACKEND_ROOT), "path": capture.path}
    result = subprocess.run(
        [sys.executable, "-c", script], stderr = subprocess.PIPE, text = True, timeout = 120
    )
    capture.close()

    assert result.returncode == 0, result.stderr
    assert "the last line before shutdown" in result.stderr
    assert result.stderr.count("bulk ") == 200
