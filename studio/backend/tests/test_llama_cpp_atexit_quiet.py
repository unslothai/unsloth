# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The atexit teardown must not print tracebacks after the program has ended.

_cleanup runs from atexit, by which point the streams the log handlers write to
can already be closed. Logging then prints its own traceback about the closed
stream on top of whatever it was trying to report, so one warning about a kill
that did not work became several unrelated tracebacks after the pytest summary --
which is how this was found, by them burying the summary line.
"""

import io
import logging
import os
import subprocess
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


def _stub() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = None
    backend._healthy = True
    return backend


class _Unterminable:
    """What a backend can be holding: something that is not a Popen.

    Tests stand one in to mean "a server is loaded" without spawning anything, and
    a backend torn down mid-start holds whatever __init__ got as far as.
    """


class _RecordingLogger:
    """Stands in for the module logger, which is a structlog bound logger rather
    than a stdlib one -- caplog never sees it, so an assertion against caplog would
    hold however loudly this warned."""

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
    from core.inference import llama_cpp as mod

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
    # The whole finalizer, not the three assignments an earlier version of this
    # duplicated: the log handle has to be closed and the reader joined, or a
    # teardown that takes this path leaks them.
    assert log_fh.closed, "the log handle was left open"
    assert backend._llama_log_fh is None
    assert reader.joined, "the stdout reader was never joined"
    assert backend._stdout_thread is None


class _RaisingLogger:
    """A logger whose writes fail, like the real one once stdout is closed.

    The module logger is a structlog PrintLogger writing straight to stdout, so a
    closed stream raises ValueError out of the call. Deliberately not a stdlib
    logger: that reports a broken handler by printing its own traceback rather
    than raising, so a stdlib stand-in exercises raiseExceptions and proves
    nothing about the path this module actually takes.
    """

    def __getattr__(self, name):
        def boom(*a, **k):
            raise ValueError("I/O operation on closed file")

        return boom


def test_a_logger_that_raises_does_not_escape_the_atexit_handler(monkeypatch):
    from core.inference import llama_cpp as mod

    monkeypatch.setattr(mod, "logger", _RaisingLogger())
    backend = _stub()
    backend._process = _Unterminable()

    backend._cleanup()


def test_the_atexit_handler_quiets_stdlib_loggers_too(monkeypatch, capsys):
    """Other libraries install stdlib loggers that fire during teardown, and those
    print their own traceback about a closed handler rather than raising, so the
    except above never sees them."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    stream.close()
    other = logging.getLogger("unsloth-atexit-test-stdlib")
    other.addHandler(handler)
    other.propagate = False

    from core.inference import llama_cpp as mod

    def kill_and_log():
        other.warning("something a dependency logs at exit")

    backend = _stub()
    monkeypatch.setattr(backend, "_kill_process", kill_and_log)
    try:
        backend._cleanup()
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
    """The escalation must not depend on a log write succeeding. logger here is a
    structlog PrintLogger straight to stdout, so a closed stream raises out of the
    warning, and reporting first meant the kill was skipped while the finally
    dropped the last reference to the process -- leaving the server running with
    nothing left to kill it."""
    from core.inference import llama_cpp as mod

    monkeypatch.setattr(mod, "logger", _RaisingLogger())
    backend = _stub()
    proc = _StubbornProcess()
    backend._process = proc

    try:
        backend._kill_process()
    except ValueError:
        pass  # the write still fails; what matters is that it failed after the kill

    assert proc.killed, "SIGKILL was skipped because the warning raised first"


class _UnkillableProcess(_StubbornProcess):
    """Ignores SIGKILL too, e.g. stuck in an uninterruptible wait."""

    def wait(self, timeout = None):
        raise subprocess.TimeoutExpired("llama-server", timeout)


def test_an_unkillable_server_is_still_reported(monkeypatch):
    """The second wait raises from inside the handler it was raised from, so it is
    not caught there and escapes. If the warning came after it, the one case an
    operator most needs to see would be reported by nothing at all."""
    from core.inference import llama_cpp as mod

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
    """Only atexit gets the quiet treatment; a live run must still surface a
    broken logging handler."""
    from core.inference import llama_cpp as mod

    monkeypatch.setattr(logging, "raiseExceptions", True)
    backend = _stub()
    backend._process = _Unterminable()

    backend._cleanup()

    assert logging.raiseExceptions is True


def test_a_failing_kill_does_not_escape_the_atexit_handler(monkeypatch):
    """atexit swallows it anyway, and there is nowhere left to report it."""
    backend = _stub()

    def boom():
        raise RuntimeError("teardown went wrong")

    monkeypatch.setattr(backend, "_kill_process", boom)

    backend._cleanup()
