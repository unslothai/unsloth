# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Teardown from atexit must not log to closed streams; the tracebacks bury the pytest summary."""

import io
import logging
import os
import subprocess
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference import llama_cpp as mod


def _stub() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = None
    backend._healthy = True
    return backend


class _Unterminable:
    """Stands in for a loaded server without a Popen; a backend torn down mid-start holds partial state."""


class _RecordingLogger:
    """Stands in for the module logger, which is structlog, so caplog never sees warnings from it."""

    def __init__(self):
        self.warnings = []
        self.other = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def __getattr__(self, name):
        def sink(
            msg = "",
            *a,
            **k,
        ):
            self.other.append((name, str(msg)))

        return sink


class _Reader:
    def __init__(self):
        self.joined = False

    def join(self, timeout = None):
        self.joined = True


def test_a_process_that_cannot_be_terminated_is_not_an_error(monkeypatch, tmp_path):
    recorder = _RecordingLogger()
    monkeypatch.setattr(mod, "logger", recorder)
    backend = _stub()
    backend._process = _Unterminable()
    log_fh = open(tmp_path / "llama.log", "w")
    reader = _Reader()
    backend._llama_log_fh = log_fh
    backend._stdout_thread = reader

    backend._kill_process()

    assert backend._process is None, "the state has to be cleared either way"
    assert backend._healthy is False
    assert recorder.warnings == [], f"warned about a non-process: {recorder.warnings}"
    assert log_fh.closed, "the log handle was left open"
    assert backend._llama_log_fh is None
    assert reader.joined, "the stdout reader was never joined"
    assert backend._stdout_thread is None


class _RaisingLogger:
    """Must raise on write like structlog's PrintLogger, since stdlib logging prints a traceback instead."""

    def __getattr__(self, name):
        def boom(*a, **k):
            raise ValueError("I/O operation on closed file")

        return boom


def test_a_logger_that_raises_does_not_escape_the_atexit_handler(monkeypatch):
    monkeypatch.setattr(mod, "logger", _RaisingLogger())
    backend = _stub()
    backend._process = _Unterminable()

    backend._cleanup()


def test_the_atexit_handler_quiets_stdlib_loggers_too(monkeypatch, capsys):
    """Stdlib loggers from other libraries fire during teardown and print their own traceback about
    a closed handler instead of raising, so the except above never sees them."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    stream.close()
    other = logging.getLogger("unsloth-atexit-test-stdlib")
    other.addHandler(handler)
    other.propagate = False

    ran = []

    # **_kw: _cleanup passes teardown=True and swallows a TypeError from a stale signature
    def kill_and_log(**_kw):
        ran.append(1)
        other.warning("something a dependency logs at exit")

    backend = _stub()
    monkeypatch.setattr(backend, "_kill_process", kill_and_log)
    try:
        backend._cleanup()
        assert ran, "the kill double never ran; this assertion proves nothing"
        assert capsys.readouterr().err == ""
    finally:
        other.removeHandler(handler)
        other.propagate = True


class _StubbornProcess:
    """A llama-server that ignores SIGTERM, which is what SIGKILL is for."""

    def __init__(self):
        self.killed = False

    def terminate(self):
        pass

    def wait(self, timeout = None):
        if not self.killed:
            raise subprocess.TimeoutExpired("llama-server", timeout)

    def kill(self):
        self.killed = True


def test_sigkill_still_happens_when_the_log_write_fails(monkeypatch):
    """SIGKILL must still be sent when the warning's log write raises, or the process is left running."""
    monkeypatch.setattr(mod, "logger", _RaisingLogger())
    backend = _stub()
    proc = _StubbornProcess()
    backend._process = proc

    try:
        backend._kill_process()
    except ValueError:
        # the write still fails; what matters is that it failed after the kill
        pass

    assert proc.killed, "SIGKILL was skipped because the warning raised first"


class _UnkillableProcess(_StubbornProcess):
    """Ignores SIGKILL too, e.g. stuck in an uninterruptible wait."""

    def wait(self, timeout = None):
        raise subprocess.TimeoutExpired("llama-server", timeout)


def test_an_unkillable_server_is_still_reported(monkeypatch):
    """The second wait raises from inside the handler, so nothing catches it; the warning must come
    first or an unkillable server goes unreported."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(mod, "logger", recorder)
    backend = _stub()
    backend._process = _UnkillableProcess()

    with pytest.raises(subprocess.TimeoutExpired):
        backend._kill_process()

    assert any(
        "SIGKILL" in w for w in recorder.warnings
    ), "an unkillable server was dropped without a word about it"


def test_the_handler_leaves_raise_exceptions_as_it_found_it(monkeypatch):
    """Only atexit is quieted; a live run must still surface a broken logging handler."""
    monkeypatch.setattr(logging, "raiseExceptions", True)
    backend = _stub()
    backend._process = _Unterminable()

    backend._cleanup()

    assert logging.raiseExceptions is True


def test_a_failing_kill_does_not_escape_the_atexit_handler(monkeypatch):
    """atexit swallows it anyway, and there is nowhere left to report it."""
    backend = _stub()

    raised = []

    def boom(**_kw):
        raised.append(1)
        raise RuntimeError("teardown went wrong")

    monkeypatch.setattr(backend, "_kill_process", boom)

    backend._cleanup()
    assert raised, "the failing kill never ran; the handler swallowed the wrong error"


class _Exited:
    pid = None

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 3

    def poll(self):
        return 3


def test_a_raising_logger_still_closes_the_attempt_log_and_names_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "logger", _RaisingLogger())
    backend = _stub()
    backend._process = _Exited()
    path = tmp_path / "llama.log"
    backend._llama_log_fh = open(path, "w", encoding = "utf-8")
    backend._llama_log_path = path

    backend._kill_process()

    assert backend._llama_log_fh is None
    assert path.read_text(encoding = "utf-8").endswith("reason=exited exit_code=3\n")


class _SlowToReapProcess(_StubbornProcess):
    """Exits after SIGKILL, but only once the driver has released its memory."""

    def wait(self, timeout = None):
        if not self.killed or (timeout or 0) < 10:
            raise subprocess.TimeoutExpired("llama-server", timeout)


def test_a_killed_server_gets_time_to_release_its_memory(monkeypatch):
    """A reload on a busy GPU failed with "timed out after 5 seconds" while the killed
    server was still releasing a large model."""
    monkeypatch.setattr(mod, "logger", _RecordingLogger())
    backend = _stub()
    proc = _SlowToReapProcess()
    backend._process = proc

    backend._kill_process()

    assert proc.killed
