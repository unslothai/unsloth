# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#11465: a worker dying without an error event keeps its exit code and first stderr reason."""

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
    text = reason + "\n" + "\n".join(f'  File "m.py", line {index} in f' for index in range(400))
    assert first_crash_line(text) == reason
    message = unexpected_exit_message(356, 2147483651, text)
    assert message.startswith("Training process exited unexpectedly (pid=356, exitcode=0x80000003)")
    assert reason in message


def test_linux_segfault_reports_the_reason_not_a_frame():
    text = (
        "Fatal Python error: Segmentation fault\n\n"
        "Current thread 0x0000758f06f62080 (most recent call first):\n"
        '  File "/usr/lib/python3.13/ctypes/__init__.py", line 546 in string_at\n'
        "\nExtension modules: numpy._core._multiarray_umath, torch._C (total: 2)\n"
    )
    assert first_crash_line(text) == "Fatal Python error: Segmentation fault"
    assert (
        first_crash_line('Thread 0x1 (most recent call first):\n  File "x.py", line 1 in f\n') == ""
    )


def test_cpp_abort_reports_what_not_a_frame():
    text = (
        "terminate called after throwing an instance of 'c10::Error'\n"
        "  what():  CUDA error: an illegal memory access was encountered\n"
        "Exception raised from c10_cuda_check_implementation (most recent call first):\n"
        "frame #0: c10::Error::Error() + 0x57 (0x7f1 in libc10.so)\n"
    )
    assert first_crash_line(text) == "what():  CUDA error: an illegal memory access was encountered"
    assert first_crash_line("terminate called without an active exception\n") == (
        "terminate called without an active exception"
    )


def test_a_killed_worker_does_not_blame_routine_output():
    text = "Hugging Face endpoint unreachable; HF_HUB_OFFLINE=1 set for this worker.\n 40%|####  | 4/10\n"
    assert first_crash_line(text) == ""
    assert unexpected_exit_message(7, -9, text) == (
        "Training process exited unexpectedly (pid=7, exitcode=-9)"
    )
    assert first_crash_line("loading\nRuntimeError: boom\n") == "RuntimeError: boom"


def test_a_recovered_warning_is_not_the_cause():
    text = "CUDA out of memory; reducing batch size and retrying\n" + "step log\n" * 50
    assert first_crash_line(text) == ""
    windows = "LLVM ERROR: Cannot select\nWindows fatal exception: code 0x80000003\n"
    assert (
        first_crash_line(text + windows + '  File "m.py", line 1 in f\n')
        == "LLVM ERROR: Cannot select"
    )


def test_the_reason_is_redacted():
    text = "LLVM ERROR: cannot open /home/alice/secret_project/kernel.so token hf_abcdefghijklmnopqrstuvwxyz0123\n"
    message = unexpected_exit_message(1, -6, text)
    assert "LLVM ERROR" in message
    assert "/home/alice" not in message
    assert "hf_abcdefghijklmnopqrstuvwxyz0123" not in message


def test_a_marked_recovered_traceback_is_not_the_cause():
    text = (
        "\x1fretry failed\n"
        "    | Traceback (most recent call last):\n"
        "    | RuntimeError: transient failure\n"
        "step log\n"
    )
    assert first_crash_line(text) == ""


def test_native_threads_do_not_push_the_reason_out():
    threads = "".join(
        f"Thread 0x{n:04x} (most recent call first):\n  <no Python frame>\n" for n in range(40)
    )
    text = "Fatal Python error: Aborted\n\n" + threads
    assert first_crash_line(text) == "Fatal Python error: Aborted"


@pytest.mark.parametrize(
    "reason",
    [
        "Bus error",
        "Illegal instruction",
        "free(): invalid pointer",
        "*** stack smashing detected ***: terminated",
        "GGML_ASSERT(ctx) failed",
    ],
)
def test_other_native_reasons(reason):
    assert first_crash_line(reason + "\n" + '  File "m.py", line 1 in f\n') == reason


def _die_after_a_long_stack(path: str) -> None:
    assert install_worker_stderr_mirror(path) is True
    sys.stderr.write("LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16\n")
    sys.stderr.write("Windows fatal exception: code 0x80000003\n")
    sys.stderr.flush()
    # Longer than the 64 KiB tail window, shorter than the 256 KiB sink cap.
    os.write(2, b'  File "m.py", line 1 in f\n' * 2500)
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


def test_the_reason_after_a_long_routine_log_is_read(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_text(
        "step log\n" * 35000 + "LLVM ERROR: late\n" + '  File "m.py", line 1 in f\n' * 100,
        encoding = "utf-8",
    )
    assert first_crash_line(capture.text()) == "LLVM ERROR: late"


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
        "LLVM ERROR: Cannot select: intrinsic demo\n" + '  File "m.py", line 1 in f\n' * 2500,
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
    assert "m.py" not in training_backend._progress.error


def test_a_queued_error_wins_over_the_exit_line(training_backend):
    training_backend._proc = _DeadProc(2147483651)
    training_backend._event_queue = _OneErrorQueue()
    training_backend._progress.is_training = True
    training_backend._stderr_capture = None

    training_backend._pump_loop()

    assert training_backend._progress.error == "CUDA out of memory"
