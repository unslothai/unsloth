# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the Colab "OutStream has no attribute 'watch_fd_thread'"
startup crash.

Field report (Colab): Unsloth Studio dies at server startup with
``❌ Unsloth Studio failed to start: 'OutStream' object has no attribute
'watch_fd_thread'``.

Root cause chain:
  * Colab's ipykernel ``OutStream`` is created with ``watchfd=False``, so it
    never gains a ``watch_fd_thread``; the ``OutStream.close()`` shipped in the
    affected ipykernel versions joins that thread unconditionally and raises
    ``AttributeError`` (ipython/ipykernel#867).
  * ``run._setup_server_disk_logging()`` replaces ``sys.stdout``/``sys.stderr``
    with a ``_TeeStream``. That changes the console object identity, so Colab's
    ``absl`` logging handler -- which captured the ORIGINAL OutStream and whose
    ``close()`` deliberately skips ``sys.stdout``/``sys.stderr`` -- no longer
    recognizes it as the live console.
  * ``run_server`` builds ``uvicorn.Config(...)``, whose ``configure_logging`` ->
    ``logging.config.dictConfig`` -> ``logging.shutdown`` closes every existing
    handler. The absl handler then calls ``OutStream.close()`` on the orphaned
    stream, and the AttributeError aborts startup.

These tests reproduce the mechanism with a stand-in OutStream (Colab-identical
constructs are not importable off Colab) and assert the tee/console path used at
startup survives it.
"""

from __future__ import annotations

import io
import logging
import sys
import weakref
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import run as run_mod  # noqa: E402


class _ColabOutStream(io.TextIOBase):
    """Stand-in for Colab's ipykernel OutStream built with ``watchfd=False``:
    no ``watch_fd_thread`` and an unguarded ``close()`` that joins it
    (ipython/ipykernel#867)."""

    def __init__(self, name: str, sink: io.StringIO):
        self.name = name
        self._sink = sink

    def write(self, s):
        return self._sink.write(s)

    def flush(self):
        pass

    def writable(self):
        return True

    def isatty(self):
        return False

    def close(self):
        # Never set because watchfd=False -> AttributeError, exactly as Colab.
        self.watch_fd_thread.join()

    def __del__(self):
        # io.TextIOBase.__del__ would call our buggy close() at GC (the harmless
        # "Exception ignored" tail seen in Colab); silence it so the test is clean.
        pass


class _WatchingOutStream(_ColabOutStream):
    """OutStream with fd-watching ON: ``watch_fd_thread`` exists, close() is
    well behaved and must keep working unchanged."""

    def __init__(self, name: str, sink: io.StringIO):
        super().__init__(name, sink)
        self.close_ran = False
        self.watch_fd_thread = type("_T", (), {"join": lambda self: None})()

    def close(self):
        self.watch_fd_thread.join()
        self.close_ran = True


class _AbslLikeHandler(logging.StreamHandler):
    """Mirror of ``absl.logging.PythonHandler.close()``: close the captured
    stream unless it is (still) one of the user-managed console streams."""

    def close(self):
        try:
            user_managed = (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__)
            if self.stream not in user_managed and (
                not hasattr(self.stream, "isatty") or not self.stream.isatty()
            ):
                self.stream.close()
        except ValueError:
            pass
        super().close()


class TestHardenConsoleClose:
    def test_neutralizes_watchfd_false_close(self):
        stream = _ColabOutStream("stdout", io.StringIO())
        with pytest.raises(AttributeError):
            stream.close()  # baseline: the ipykernel #867 bug is real

        stream = _ColabOutStream("stdout", io.StringIO())
        run_mod._harden_console_close(stream)
        assert stream.close() is None  # swallowed, no crash

    def test_healthy_close_still_runs_fully(self):
        stream = _WatchingOutStream("stdout", io.StringIO())
        run_mod._harden_console_close(stream)
        stream.close()
        assert stream.close_ran is True

    def test_only_attributeerror_is_swallowed(self):
        class _Boom:
            def close(self):
                raise ValueError("real teardown failure")

        stream = _Boom()
        run_mod._harden_console_close(stream)
        with pytest.raises(ValueError):
            stream.close()

    def test_unrelated_attributeerror_still_propagates(self):
        # Only #867 is neutralized; a genuine missing attribute during teardown
        # must still surface instead of looking like a clean close.
        class _Console:
            def close(self):
                return self.not_a_real_attribute

        stream = _Console()
        run_mod._harden_console_close(stream)
        with pytest.raises(AttributeError, match = "not_a_real_attribute"):
            stream.close()

    def test_swallowed_across_attributeerror_message_shapes(self):
        # Python 3.12 appends a "Did you mean" tail; the match must survive it,
        # and pre-3.10 AttributeErrors carry no ``name``, only the message.
        class _Suggesting:
            def close(self):
                raise AttributeError(
                    "'OutStream' object has no attribute 'watch_fd_thread'. "
                    "Did you mean: '_watch_pipe_fd'?"
                )

        stream = _Suggesting()
        run_mod._harden_console_close(stream)
        assert stream.close() is None

    def test_unsettable_close_is_left_alone(self):
        # A stream whose close cannot be reassigned must not raise from hardening.
        class _Frozen:
            __slots__ = ()

            def close(self):
                return "ok"

        stream = _Frozen()
        run_mod._harden_console_close(stream)  # must not raise
        assert stream.close() == "ok"


class TestTeeStreamClose:
    def test_tee_close_over_buggy_stream_never_raises(self):
        console = _ColabOutStream("stdout", io.StringIO())
        log = io.StringIO()
        tee = run_mod._TeeStream(console, log)
        tee.write("before-close")
        tee.close()  # must not raise despite the wrapped stream's broken close
        assert log.getvalue() == "before-close"

    def test_tee_close_flushes_log(self):
        class _FlushCounting(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        console, log = io.StringIO(), _FlushCounting()
        tee = run_mod._TeeStream(console, log)
        tee.write("x")
        tee.close()
        assert log.flushes >= 1


class TestColabStartupRegression:
    """End-to-end: the exact trigger -- an absl-style handler closing the
    orphaned OutStream during the ``logging.shutdown`` that uvicorn's
    ``uvicorn.Config`` -> ``dictConfig`` runs -- must not crash Studio, and the
    tee must keep logging afterwards.

    ``logging.shutdown`` is driven over a LOCAL weakref list (identical code path
    to ``logging.config._clearExistingHandlers``) so the global logging state and
    pytest's own capture are untouched.
    """

    def _make_console_and_handlers(self, monkeypatch):
        out_sink, err_sink = io.StringIO(), io.StringIO()
        out_stream = _ColabOutStream("stdout", out_sink)
        err_stream = _ColabOutStream("stderr", err_sink)
        monkeypatch.setattr(sys, "stdout", out_stream)
        monkeypatch.setattr(sys, "stderr", err_stream)
        # absl-like handlers capture the ORIGINAL OutStreams (as in Colab).
        handlers = [_AbslLikeHandler(sys.stdout), _AbslLikeHandler(sys.stderr)]
        return out_sink, err_sink, out_stream, err_stream, handlers

    def test_baseline_reproduces_crash_without_fix(self, monkeypatch):
        # Prove the test exercises the real path: swapping the console identity
        # (what the tee does) makes the absl-like close hit #867.
        _, _, out_stream, err_stream, handlers = self._make_console_and_handlers(monkeypatch)
        try:
            monkeypatch.setattr(sys, "stdout", io.StringIO())
            monkeypatch.setattr(sys, "stderr", io.StringIO())
            with pytest.raises(AttributeError, match = "watch_fd_thread"):
                logging.shutdown([weakref.ref(h) for h in handlers])
        finally:
            # Neutralize so a lingering handler can't crash global teardown.
            run_mod._harden_console_close(out_stream)
            run_mod._harden_console_close(err_stream)
            for h in handlers:
                try:
                    h.close()
                except Exception:
                    pass

    def test_startup_survives_with_harden_and_tee(self, monkeypatch):
        out_sink, _, out_stream, err_stream, handlers = self._make_console_and_handlers(monkeypatch)

        # Exactly what _setup_server_disk_logging does before serving:
        run_mod._harden_console_close(sys.stdout)
        run_mod._harden_console_close(sys.stderr)
        log_fh = io.StringIO()
        monkeypatch.setattr(sys, "stdout", run_mod._TeeStream(sys.stdout, log_fh))
        monkeypatch.setattr(sys, "stderr", run_mod._TeeStream(sys.stderr, log_fh))

        # The close-storm uvicorn triggers via dictConfig -> logging.shutdown,
        # closing the absl-like handlers over the (now orphaned) OutStreams.
        logging.shutdown([weakref.ref(h) for h in handlers])  # must NOT raise

        # The tee still tees to both console and disk afterwards.
        print("post-startup-line")
        sys.stdout.flush()
        assert "post-startup-line" in out_sink.getvalue()
        assert "post-startup-line" in log_fh.getvalue()
