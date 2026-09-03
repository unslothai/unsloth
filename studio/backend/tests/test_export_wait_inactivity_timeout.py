# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The export response wait must be an INACTIVITY timeout, not an absolute deadline.

An absolute deadline cannot tell a busy worker from a hung one, so a large export dies at exactly
one hour and the cleanup that follows SIGKILLs it mid-write, leaving a half-written model on disk.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", types.ModuleType("structlog"))

_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = []
_utils_paths_stub = types.ModuleType("utils.paths")
_utils_paths_stub.outputs_root = lambda: Path("/tmp")
sys.modules.setdefault("utils", _utils_pkg)
sys.modules.setdefault("utils.paths", _utils_paths_stub)


TIMEOUT = 10.0
READ_SECONDS = 4.0


@pytest.fixture
def waiting_orchestrator(monkeypatch):
    """(orchestrator, clock, script) on a fake clock; a read past the end of *script* is a quiet one."""
    from core.export import orchestrator as orchestrator_module

    clock = types.SimpleNamespace(now = 0.0)
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: clock.now)

    orch = orchestrator_module.ExportOrchestrator()
    script: list = []

    def fake_read(timeout = None):
        clock.now += READ_SECONDS
        return script.pop(0) if script else None

    monkeypatch.setattr(orch, "_read_resp", fake_read)
    monkeypatch.setattr(orch, "_ensure_subprocess_alive", lambda: True)
    return orch, clock, script


def test_a_worker_that_keeps_logging_survives_past_the_timeout(waiting_orchestrator) -> None:
    orch, clock, script = waiting_orchestrator
    script.extend(
        [
            {"type": "log", "stream": "stdout", "line": f"writing shard {n}", "ts": 0.0}
            for n in range(6)
        ]
    )
    script.append({"type": "export_merged_done", "path": "/out/model"})

    resp = orch._wait_response("export_merged_done", timeout = TIMEOUT)

    assert resp["type"] == "export_merged_done"
    assert clock.now > TIMEOUT, "the fixture must run the wait past the timeout to be meaningful"


def test_a_status_message_also_resets_the_deadline(waiting_orchestrator) -> None:
    orch, clock, script = waiting_orchestrator
    script.extend(
        [{"type": "status", "message": f"Quantizing block {n}", "ts": 0.0} for n in range(6)]
    )
    script.append({"type": "export_gguf_done", "path": "/out/model.gguf"})

    assert orch._wait_response("export_gguf_done", timeout = TIMEOUT)["type"] == "export_gguf_done"
    assert clock.now > TIMEOUT


def test_a_quiet_worker_still_times_out(waiting_orchestrator) -> None:
    orch, clock, _script = waiting_orchestrator

    with pytest.raises(RuntimeError):
        orch._wait_response("export_merged_done", timeout = TIMEOUT)

    assert clock.now < TIMEOUT * 2, "a quiet wait must end near the timeout, not run on"
