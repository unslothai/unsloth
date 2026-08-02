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
import sys

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


def test_a_process_that_cannot_be_terminated_is_not_an_error(monkeypatch):
    from core.inference import llama_cpp as mod

    recorder = _RecordingLogger()
    monkeypatch.setattr(mod, "logger", recorder)
    backend = _stub()
    backend._process = _Unterminable()

    backend._kill_process()

    assert backend._process is None, "the state has to be cleared either way"
    assert backend._healthy is False
    assert recorder.warnings == [], f"warned about a non-process: {recorder.warnings}"


def test_the_atexit_handler_writes_nothing_to_a_closed_stream(monkeypatch, capsys):
    """The failure mode itself: a handler whose stream has gone, which is the state
    the interpreter leaves them in by the time atexit runs."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    stream.close()

    from core.inference import llama_cpp as mod

    monkeypatch.setattr(mod, "logger", logging.getLogger("unsloth-atexit-test"))
    mod.logger.addHandler(handler)
    mod.logger.propagate = False
    try:
        backend = _stub()
        backend._process = _Unterminable()

        backend._cleanup()

        # Logging reports a broken handler by printing to stderr itself, so that
        # is where the noise lands rather than in caplog.
        assert capsys.readouterr().err == ""
    finally:
        mod.logger.removeHandler(handler)
        mod.logger.propagate = True


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
