# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#11465: a training worker that dies without an error event must keep its exit code
and the first meaningful stderr line, not only "exited unexpectedly"."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.worker_stderr import (  # noqa: E402
    WorkerStderrCapture,
    first_crash_line,
    format_exit_code,
    install_worker_stderr_mirror,
    unexpected_exit_message,
)


def test_windows_breakpoint_status_is_hex():
    # ExitProcess(0x80000003) came back as this unsigned value on Windows.
    assert format_exit_code(2147483651) == "0x80000003"
    assert format_exit_code(-2147483645) == "0x80000003"
    assert format_exit_code(-9) == "-9"
    assert format_exit_code(1) == "1"
    assert format_exit_code(None) == "unknown"


def test_the_reason_is_kept_when_the_stack_is_what_follows():
    reason = "LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16"
    text = reason + "\n" + "\n".join(f"frame {index}" for index in range(400))
    assert first_crash_line(text) == reason
    message = unexpected_exit_message(356, 2147483651, text)
    assert message.startswith("Training process exited unexpectedly (pid=356, exitcode=0x80000003)")
    assert reason in message


def _die_after_a_long_stack(path: str) -> None:
    assert install_worker_stderr_mirror(path) is True
    sys.stderr.write("LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16\n")
    sys.stderr.write("Windows fatal exception: code 0x80000003\n")
    sys.stderr.flush()
    # Longer than the 64 KiB tail window, shorter than the 256 KiB sink cap.
    os.write(2, b"frame\n" * 12000)
    if sys.platform == "win32":
        ctypes.windll.kernel32.ExitProcess(0x80000003)
    os._exit(3)


def test_the_mirror_keeps_the_reason_that_the_tail_window_drops(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    context = mp.get_context("spawn")
    proc = context.Process(target = _die_after_a_long_stack, args = (capture.path,))
    proc.start()
    proc.join(30)
    assert not proc.is_alive()
    body = capture.text()
    assert "LLVM ERROR: Cannot select" in body
    assert "LLVM ERROR" not in capture.tail()
    message = unexpected_exit_message(proc.pid, proc.exitcode, body)
    assert "LLVM ERROR: Cannot select" in message
    if sys.platform == "win32":
        assert "0x80000003" in message
        assert proc.exitcode == 2147483651


class _DeadProc:
    def __init__(self, exitcode: int):
        self.exitcode = exitcode
        self.pid = 356

    def is_alive(self) -> bool:
        return False


class _EmptyQueue:
    def get(self, *args, **kwargs):
        raise queue.Empty

    def get_nowait(self, *args, **kwargs):
        raise queue.Empty


class _OneErrorQueue:
    def __init__(self):
        self._pending = [{"type": "error", "error": "CUDA out of memory", "stack": ""}]

    def get(self, *args, **kwargs):
        if self._pending:
            return self._pending.pop(0)
        raise queue.Empty

    def get_nowait(self, *args, **kwargs):
        if self._pending:
            return self._pending.pop(0)
        raise queue.Empty


@pytest.fixture
def training_backend(monkeypatch):
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    monkeypatch.setattr(backend, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(backend, "_finalize_run_in_db", lambda **kwargs: None)
    return backend


def test_the_pump_reports_the_exit_code_and_the_llvm_line(training_backend, tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    capture_path = Path(capture.path)
    capture_path.write_text(
        "LLVM ERROR: Cannot select: intrinsic demo\n" + "frame\n" * 12000,
        encoding = "utf-8",
    )
    training_backend._stderr_capture = capture
    training_backend._proc = _DeadProc(2147483651)
    training_backend._event_queue = _EmptyQueue()
    training_backend._progress.is_training = True

    training_backend._pump_loop()

    assert training_backend._progress.is_training is False
    assert "0x80000003" in training_backend._progress.error
    assert "LLVM ERROR: Cannot select: intrinsic demo" in training_backend._progress.error
    assert "frame" not in training_backend._progress.error


def test_a_queued_error_wins_over_the_exit_line(training_backend):
    training_backend._proc = _DeadProc(2147483651)
    training_backend._event_queue = _OneErrorQueue()
    training_backend._progress.is_training = True
    training_backend._stderr_capture = None

    training_backend._pump_loop()

    assert training_backend._progress.error == "CUDA out of memory"
