# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# A worker that only returns from its command loop stays alive long enough for the parent to log
# "Inference subprocess did not exit gracefully, terminating" 10-18s after a correct answer, so the
# shutdown branch exits the process itself once the models are unloaded and the ack is sent.

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from studio.backend.core.inference import worker  # noqa: E402


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
