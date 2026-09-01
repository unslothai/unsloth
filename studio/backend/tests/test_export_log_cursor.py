# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression tests for the export log ring-buffer cursor semantics.

Race: the frontend opens the SSE connection AFTER the POST that starts the
export, so lines emitted in the gap (seqs 1..k) are unreachable when the SSE
default cursor `get_current_log_seq()` returns k.

Fix: `clear_logs()` snapshots the pre-run seq into `_run_start_seq` (via
`get_run_start_seq()`), and the SSE cursor defaults to that snapshot, so the
client sees the full run regardless of connect time.

These tests exercise the orchestrator contract only (no subprocess/FastAPI).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Backend root on sys.path so core.export.orchestrator resolves without bootstrap.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Stub orchestrator.py's heavy top-level imports so it loads without the venv.
_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog is only a module-level import here; a bare stub suffices.
sys.modules.setdefault("structlog", types.ModuleType("structlog"))

# Stub utils.paths so orchestrator.py's top-level import resolves.
_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = []  # mark as package
_utils_paths_stub = types.ModuleType("utils.paths")
_utils_paths_stub.outputs_root = lambda: Path("/tmp")
sys.modules.setdefault("utils", _utils_pkg)
sys.modules.setdefault("utils.paths", _utils_paths_stub)


@pytest.fixture
def orchestrator():
    """Fresh ExportOrchestrator with only the log-buffer state exercised."""
    from core.export.orchestrator import ExportOrchestrator
    return ExportOrchestrator()


def _append(
    orch,
    line: str,
    stream: str = "stdout",
) -> None:
    """Shortcut for simulating a worker log message."""
    orch._append_log({"type": "log", "stream": stream, "line": line, "ts": 0.0})


# ---------------------------------------------------------------------------
# clear_logs() semantics
# ---------------------------------------------------------------------------


def test_run_start_seq_is_zero_before_any_logs(orchestrator) -> None:
    """run_start_seq == 0 on a new orchestrator, so the first SSE picks up
    every line from seq 1 onward."""
    assert orchestrator.get_run_start_seq() == 0


def test_clear_logs_snapshots_current_seq(orchestrator) -> None:
    """clear_logs() captures _log_seq BEFORE clearing, so the next run can
    anchor its SSE cursor at the snapshot."""
    _append(orchestrator, "old run line 1")
    _append(orchestrator, "old run line 2")
    _append(orchestrator, "old run line 3")
    assert orchestrator.get_current_log_seq() == 3

    orchestrator.clear_logs()

    assert orchestrator.get_run_start_seq() == 3
    assert orchestrator.get_current_log_seq() == 3  # seq counter preserved


# ---------------------------------------------------------------------------
# Race regression: SSE connects AFTER lines have been emitted
# ---------------------------------------------------------------------------


def test_sse_default_cursor_catches_all_current_run_lines(orchestrator) -> None:
    """POST-then-SSE race: worker emits lines right after clear_logs(), SSE
    connects later. get_run_start_seq() as the default cursor returns every
    line since clear_logs() (pre-fix it missed them)."""
    # Previous run leaves some buffered lines.
    _append(orchestrator, "previous run line A")
    _append(orchestrator, "previous run line B")

    # New run starts: orchestrator clears the buffer and snapshots seq.
    orchestrator.clear_logs()
    run_start = orchestrator.get_run_start_seq()

    # Worker emits early lines BEFORE the SSE connects.
    _append(orchestrator, "Importing Unsloth...")
    _append(orchestrator, "Loading checkpoint: /foo/bar")
    _append(orchestrator, "Starting export...")

    # SSE connects and asks for everything after the run start cursor.
    entries, new_cursor = orchestrator.get_logs_since(run_start)

    # All three early lines must be present. Pre-fix this was [].
    lines = [e["line"] for e in entries]
    assert lines == [
        "Importing Unsloth...",
        "Loading checkpoint: /foo/bar",
        "Starting export...",
    ]
    assert new_cursor == entries[-1]["seq"]


def test_sse_default_cursor_excludes_previous_run(orchestrator) -> None:
    """After clear_logs(), previous-run lines must not leak into the new
    run's SSE stream (the fix must preserve this)."""
    _append(orchestrator, "previous run line 1")
    _append(orchestrator, "previous run line 2")
    _append(orchestrator, "previous run line 3")
    assert orchestrator.get_current_log_seq() == 3

    orchestrator.clear_logs()
    run_start = orchestrator.get_run_start_seq()

    _append(orchestrator, "new run line")

    entries, _ = orchestrator.get_logs_since(run_start)
    assert [e["line"] for e in entries] == ["new run line"]


def test_clear_logs_twice_advances_run_start(orchestrator) -> None:
    """Back-to-back clear_logs() calls each re-anchor run_start at the current
    seq, giving successive runs a fresh low-water mark."""
    _append(orchestrator, "run 1 line a")
    _append(orchestrator, "run 1 line b")

    orchestrator.clear_logs()
    assert orchestrator.get_run_start_seq() == 2

    _append(orchestrator, "run 2 line a")
    _append(orchestrator, "run 2 line b")
    _append(orchestrator, "run 2 line c")

    orchestrator.clear_logs()
    assert orchestrator.get_run_start_seq() == 5
