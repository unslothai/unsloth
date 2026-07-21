# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parent-side training Xet->HTTP fallback: a model-load stall respawns the
worker once with Xet disabled, preserving the DB run row. Driven via
_handle_event with a fake spawn context; no GPU, no network, no real subprocess.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the heavy module-level imports of core/training/training.py so it imports
# under CPU-only/no-network, then restore them (see the restore loop below).
_SAVED: dict = {}


def _stub(name, mod):
    _SAVED[name] = sys.modules.get(name)
    sys.modules[name] = mod


_lg = _types.ModuleType("loggers")
_lg.get_logger = lambda name: logging.getLogger(name)
_stub("loggers", _lg)
_stub("structlog", _types.ModuleType("structlog"))
_mpl = _types.ModuleType("matplotlib")
_plt = _types.ModuleType("matplotlib.pyplot")
_plt.Figure = type("Figure", (), {})  # referenced in a class-def annotation
_mpl.pyplot = _plt
_stub("matplotlib", _mpl)
_stub("matplotlib.pyplot", _plt)
_hw = _types.ModuleType("utils.hardware")
_hw.prepare_gpu_selection = lambda *a, **k: (None, None)
_stub("utils.hardware", _hw)
_npl = _types.ModuleType("utils.native_path_leases")
_npl.native_path_secret_removed_for_child_start = lambda: contextlib.nullcontext()
_npl.run_without_native_path_secret = lambda fn: fn
_stub("utils.native_path_leases", _npl)
_pth = _types.ModuleType("utils.paths")
_pth.outputs_root = lambda *a, **k: "/tmp/outputs"
_stub("utils.paths", _pth)

import core.training.training as training_mod
from core.training.training import TrainingBackend

# Restore every stubbed module so this file never pollutes the shared session: a
# leaked bare ``structlog`` (no ``get_logger``) would break every later module
# that logs at import. training_mod already bound the stubs it needs at runtime.
for _name in (
    "loggers",
    "structlog",
    "matplotlib",
    "matplotlib.pyplot",
    "utils.hardware",
    "utils.native_path_leases",
    "utils.paths",
):
    _prev = _SAVED.get(_name)
    if _prev is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _prev


@pytest.fixture(autouse = True)
def _stub_worker_module():
    """Stub ``core.training.worker`` so the respawn's lazy import of the
    torch-heavy worker is never required."""
    prev = sys.modules.get("core.training.worker")
    stub = _types.ModuleType("core.training.worker")
    stub.run_training_process = lambda **kwargs: None
    sys.modules["core.training.worker"] = stub
    yield
    if prev is None:
        sys.modules.pop("core.training.worker", None)
    else:
        sys.modules["core.training.worker"] = prev


class _FakeProc:
    def __init__(self, **kwargs):
        self._alive = True
        self.pid = 4321
        self.kwargs = kwargs

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def join(self, timeout = None):
        self._alive = False


class _FakeQueue:
    def put(self, *a, **k):
        pass

    def get(self, *a, **k):
        raise queue.Empty


class _FakeCtx:
    def __init__(self):
        self.spawned: list = []

    def Queue(self):
        return _FakeQueue()

    def Process(self, **kwargs):
        self.spawned.append(kwargs)
        return _FakeProc(**kwargs)


def _backend_mid_load():
    b = TrainingBackend()
    b._last_full_config = {
        "model_name": "org/model",
        "disable_xet": False,
        "hf_token": "tok",
    }
    b._in_model_load = True
    b._xet_fallback_used = False
    proc = _FakeProc()
    b._proc = proc
    return b, proc


def test_stall_during_load_arms_respawn_and_terminates_worker():
    b, proc = _backend_mid_load()
    b._handle_event({"type": "stall", "message": "no progress for 180s"})
    assert b._needs_xet_respawn is True
    assert b._xet_fallback_used is True
    assert proc.is_alive() is False, "stalled worker must be terminated"


def test_respawn_uses_disable_xet_and_preserves_run_row(monkeypatch):
    b, _ = _backend_mid_load()
    b._handle_event({"type": "stall", "message": "x"})

    fake_ctx = _FakeCtx()
    monkeypatch.setattr(training_mod, "_CTX", fake_ctx)
    monkeypatch.setattr(b, "_pump_loop", lambda: None)  # neutralize the new pump
    created = {"n": 0}
    finalized = {"n": 0}
    monkeypatch.setattr(
        b, "_ensure_db_run_created", lambda: created.__setitem__("n", created["n"] + 1)
    )
    monkeypatch.setattr(
        b,
        "_finalize_run_in_db",
        lambda **k: finalized.__setitem__("n", finalized["n"] + 1),
    )

    b._respawn_worker_disable_xet()

    assert len(fake_ctx.spawned) == 1, "respawn must start exactly one worker"
    cfg = fake_ctx.spawned[0]["kwargs"]["config"]
    assert cfg["disable_xet"] is True, "respawned worker must run with Xet disabled"
    assert cfg["model_name"] == "org/model"
    assert created["n"] == 0, "respawn must not recreate the DB run row"
    assert (
        finalized["n"] == 0
    ), "a successful respawn must not finalize the run as error"


def test_second_stall_surfaces_error_without_respawn():
    b, proc = _backend_mid_load()
    b._xet_fallback_used = True  # HTTP fallback already spent
    b._handle_event({"type": "stall", "message": "stalled again over http"})
    assert b._needs_xet_respawn is False
    assert b._progress.error and "stalled" in b._progress.error.lower()
    assert proc.is_alive() is False


def test_model_load_completed_disarms_recovery():
    b, _ = _backend_mid_load()
    b._handle_event({"type": "model_load_completed"})
    assert b._in_model_load is False
    # A stall after the load finished is not a transport stall to recover from.
    b._handle_event({"type": "stall", "message": "post-load"})
    assert b._needs_xet_respawn is False


def test_model_load_started_arms_recovery_window():
    b = TrainingBackend()
    assert b._in_model_load is False
    b._handle_event({"type": "model_load_started"})
    assert b._in_model_load is True


def test_child_should_disable_xet_truth_table():
    from utils.hf_xet_fallback import child_should_disable_xet

    assert child_should_disable_xet({"disable_xet": True}) is True
    assert child_should_disable_xet({"disable_xet": False}) is False
    assert child_should_disable_xet({}) is False
