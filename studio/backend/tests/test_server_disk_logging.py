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
        call_idx = src.index(
            "_setup_server_disk_logging()", src.index("def run_server")
        )
        main_import_idx = src.index("from main import app", src.index("def run_server"))
        assert call_idx < main_import_idx, (
            "disk logging must be armed before importing main so import-time "
            "failures leave evidence on disk"
        )
