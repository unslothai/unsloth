# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the server session log + native-crash capture in run.py.

Field regression: Unsloth "terminates without a warning" -- a native crash in
the GPU runtime kills the process with no Python traceback, and a desktop-
shortcut console closes before anything can be read. The server must tee its
console output to disk and aim faulthandler at the same file so even hard
crashes leave evidence.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import run as run_mod  # noqa: E402


class TestTeeStream:
    def test_writes_reach_both_and_return_original(self):
        console, log = io.StringIO(), io.StringIO()
        tee = run_mod._TeeStream(console, log)
        n = tee.write("hello")
        assert console.getvalue() == "hello" == log.getvalue()
        assert n == 5  # delegate's return value, console contract unchanged

    def test_log_failure_never_breaks_console(self):
        class Broken:
            def write(self, data):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

        console = io.StringIO()
        tee = run_mod._TeeStream(console, Broken())
        assert tee.write("still works") == len("still works")
        tee.flush()  # must not raise
        assert console.getvalue() == "still works"

    def test_attribute_proxy(self):
        console, log = io.StringIO(), io.StringIO()
        tee = run_mod._TeeStream(console, log)
        # isatty / encoding probes must see the original stream's answers.
        assert tee.isatty() == console.isatty()

    def test_missing_console_is_a_null_sink(self):
        # Production never builds this, but _TeeStream(None, ...) must not crash.
        log = io.StringIO()
        tee = run_mod._TeeStream(None, log)
        assert tee.write("hello") == len("hello")  # text-stream write contract
        tee.flush()
        tee.close()
        assert log.getvalue() == "hello"
        assert not log.closed  # the tee does not own the log handle
        run_mod._harden_console_close(None)  # must not raise


class TestNormalizeStandardStreams:
    """A Windows process with no valid std handles starts with them all None."""

    def test_missing_streams_become_usable_text_streams(self, monkeypatch):
        for name in ("stdin", "stdout", "stderr"):
            monkeypatch.setattr(sys, name, None)
            monkeypatch.setattr(sys, f"__{name}__", None)
        run_mod._normalize_standard_streams()
        try:
            for name in ("stdin", "stdout", "stderr"):
                stream = getattr(sys, name)
                assert stream is not None
                assert getattr(sys, f"__{name}__") is not None
                # uvicorn's default formatter probes isatty(); logging needs write().
                assert stream.isatty() is False
                assert stream.encoding
                assert stream.fileno() >= 0
            sys.stdout.write("discarded")
            sys.stdout.flush()
            print("also discarded")
        finally:
            for name in ("stdin", "stdout", "stderr"):
                stream = getattr(sys, name)
                if stream is not None:
                    stream.close()

    def test_existing_streams_are_left_alone(self, monkeypatch):
        console = io.StringIO()
        monkeypatch.setattr(sys, "stdout", console)
        monkeypatch.setattr(sys, "stderr", console)
        run_mod._normalize_standard_streams()
        # Identity, not truthiness: replacing a live console would break Colab
        # (ipykernel OutStream), Tauri's stdout protocol and pytest capture.
        assert sys.stdout is console
        assert sys.stderr is console

    def test_runs_before_the_logger_import(self):
        # structlog binds `from sys import stdout` at import time, so normalizing
        # after the loggers import leaves None captured forever.
        src = (Path(_BACKEND_DIR) / "run.py").read_text(encoding = "utf-8")
        call = "\n_normalize_standard_streams()"
        assert call in src, "run.py never calls _normalize_standard_streams()"
        assert src.index(call) < src.index("\nfrom loggers import get_logger")


class TestSetupServerDiskLogging:
    def test_opt_out_env(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_FILE_LOG", "1")
        assert run_mod._setup_server_disk_logging() is None

    def test_creates_log_and_enables_faulthandler(self, monkeypatch, tmp_path):
        import faulthandler

        monkeypatch.delenv("UNSLOTH_STUDIO_NO_FILE_LOG", raising = False)
        monkeypatch.delenv("PYTHONFAULTHANDLER", raising = False)
        # Both resolution paths (utils.paths.studio_root and the env
        # fallback) honor UNSLOTH_STUDIO_HOME, so this redirects the log dir.
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
        orig_out, orig_err = sys.stdout, sys.stderr
        was_enabled = faulthandler.is_enabled()
        try:
            log_path = run_mod._setup_server_disk_logging()
            assert log_path is not None
            assert Path(log_path).is_file()
            assert "logs" in str(log_path)
            # faulthandler armed at the file; children inherit the env switch.
            assert faulthandler.is_enabled()
            import os

            assert os.environ.get("PYTHONFAULTHANDLER") == "1"
            print("tee-capture-marker")
            sys.stdout.flush()
            assert "tee-capture-marker" in Path(log_path).read_text(
                encoding = "utf-8", errors = "replace"
            )
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
            if not was_enabled:
                faulthandler.disable()

    def test_run_server_wires_logging_before_main_import(self):
        src = (Path(_BACKEND_DIR) / "run.py").read_text(encoding = "utf-8")
        call_idx = src.index("_setup_server_disk_logging()", src.index("def run_server"))
        main_import_idx = src.index("from main import app", src.index("def run_server"))
        assert call_idx < main_import_idx, (
            "disk logging must be armed before importing main so import-time "
            "failures leave evidence on disk"
        )

    def test_structlog_is_configured_after_the_tee_and_before_the_first_line(self):
        """Order, not presence, is the invariant.

        ``LogConfig.setup_logging`` hands structlog a
        ``PrintLoggerFactory(file = sys.stdout)``, which snapshots the stream it is given,
        and ``cache_logger_on_first_use`` then freezes that snapshot into any logger that
        has already emitted a line. Configure before the tee and this module's ``logger``
        is pinned to the console for the rest of the process -- every later run.py line
        goes missing from the session log. Configure after it and the whole session,
        starting with the first line, renders one way into both.
        """
        src = (Path(_BACKEND_DIR) / "run.py").read_text(encoding = "utf-8")
        body = src.index("def run_server")
        tee_idx = src.index("_setup_server_disk_logging()", body)
        setup_idx = src.index("LogConfig.setup_logging(", body)
        first_log_idx = src.index("logger.info(", body)
        assert tee_idx < setup_idx < first_log_idx, (
            "run_server must install the tee, then configure structlog, then log; "
            f"got tee@{tee_idx} setup@{setup_idx} first-log@{first_log_idx}"
        )
        assert (
            "LogConfig.setup_logging(" not in src[:body]
        ), "configuring structlog at import time pins it to the pre-tee sys.stdout"

    def test_run_py_does_not_import_a_loggers_submodule_at_module_scope(self):
        """`loggers` must be a real package for `loggers.config` to resolve.

        run.py is loaded by tests that stand a bare ``types.ModuleType`` in for it
        (tests/studio/install/test_selection_logic.py). A bare module has no ``__path__``,
        so a module-scope submodule import fails during collection and takes every test in
        that file with it. Import it where it is used instead.
        """
        import ast

        tree = ast.parse((Path(_BACKEND_DIR) / "run.py").read_text(encoding = "utf-8"))
        offenders = []
        for node in tree.body:  # module scope only
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("loggers."):
                offenders.append(f"line {node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                offenders += [
                    f"line {node.lineno}: import {a.name}"
                    for a in node.names
                    if a.name.startswith("loggers.")
                ]
        assert not offenders, "; ".join(offenders)

    def test_conflicting_flags_are_rejected_before_the_tee_is_installed(self):
        """A deterministic preflight failure must not leave the process streams swapped.

        ``_setup_server_disk_logging()`` replaces ``sys.stdout``/``sys.stderr`` and opens a
        log handle. An embedder that catches this ``SystemExit`` keeps all of it, and its
        next ``run_server()`` call nests a second tee, writing every line twice.
        """
        src = (Path(_BACKEND_DIR) / "run.py").read_text(encoding = "utf-8")
        body = src.index("def run_server")
        reject_idx = src.index("--secure requires the Cloudflare tunnel", body)
        # Anchor on the assignment, not the bare name: a comment mentioning the call
        # would otherwise satisfy this.
        tee_idx = src.index("_session_log = _setup_server_disk_logging()", body)
        assert reject_idx < tee_idx, (
            "the --secure/--no-cloudflare rejection must run before the tee is installed; "
            f"got reject@{reject_idx} tee@{tee_idx}"
        )

    def test_a_rejected_flag_combination_leaves_the_streams_alone(self):
        import run as run_mod
        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            with pytest.raises(SystemExit):
                run_mod.run_server(secure = True, cloudflare = False, silent = True)
            assert sys.stdout is orig_out
            assert sys.stderr is orig_err
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
