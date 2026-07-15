# SPDX-License-Identifier: AGPL-3.0-only
"""_shutdown_subprocess returns whether the worker actually died, and preserves the
live handle when it survives terminate/kill.

A GPU worker wedged in an uninterruptible CUDA syscall can outlive SIGKILL. If shutdown
nulled its handle anyway, is_worker_alive() would report False and the pre-swap liveness
guard would let the destructive .venv_t5_latest rename proceed while a live worker still
holds sidecar transformers modules (breaking the rename on Windows). The methods must keep
the handle and return False so callers can refuse the swap.
"""

import pytest

from core.export.orchestrator import ExportOrchestrator
from core.inference.orchestrator import InferenceOrchestrator


class _FakeProc:
    """A subprocess handle that dies only on the requested step (or never)."""

    def __init__(self, dies_on = None):
        self._alive = True
        self._dies_on = dies_on  # None | "join" | "terminate" | "kill"

    def is_alive(self):
        return self._alive

    def join(self, timeout = None):
        if self._dies_on == "join":
            self._alive = False

    def terminate(self):
        if self._dies_on == "terminate":
            self._alive = False

    def kill(self):
        if self._dies_on == "kill":
            self._alive = False


def _bare_inference():
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._stop_dispatcher = lambda: None
    o._cancel_generation = lambda: None
    o._drain_queue = lambda: []

    class _Q:
        def put(self, *a, **k):
            pass

    o._cmd_queue = _Q()
    o._resp_queue = _Q()
    o._cancel_event = None
    o._drain_event = None
    return o


def _bare_export():
    o = ExportOrchestrator.__new__(ExportOrchestrator)
    o._drain_queue = lambda: []

    class _Q:
        def put(self, *a, **k):
            pass

    o._cmd_queue = _Q()
    o._resp_queue = _Q()
    return o


@pytest.fixture(autouse = True)
def _no_sleep(monkeypatch):
    # _shutdown_subprocess sleeps 0.5s after cancelling; keep the tests instant.
    import core.inference.orchestrator as inf_mod
    monkeypatch.setattr(inf_mod.time, "sleep", lambda *_a, **_k: None)


class TestInferenceShutdownReturn:
    def test_worker_that_dies_returns_true_and_clears_handle(self):
        o = _bare_inference()
        o._proc = _FakeProc(dies_on = "terminate")
        assert o._shutdown_subprocess(timeout = 0.01) is True
        assert o._proc is None
        assert o.is_worker_alive() is False

    def test_survivor_returns_false_and_keeps_handle(self):
        o = _bare_inference()
        o._proc = _FakeProc(dies_on = None)  # outlives terminate AND kill
        assert o._shutdown_subprocess(timeout = 0.01) is False
        assert o._proc is not None
        # is_worker_alive stays truthful, so the pre-swap guard can refuse the swap.
        assert o.is_worker_alive() is True

    def test_already_dead_returns_true(self):
        o = _bare_inference()
        o._proc = _FakeProc(dies_on = "join")
        o._proc._alive = False
        assert o._shutdown_subprocess(timeout = 0.01) is True
        assert o._proc is None


class TestExportShutdownReturn:
    def test_worker_that_dies_returns_true_and_clears_handle(self):
        o = _bare_export()
        o._proc = _FakeProc(dies_on = "terminate")
        assert o._shutdown_subprocess(timeout = 0.01) is True
        assert o._proc is None
        assert o.is_worker_alive() is False

    def test_survivor_returns_false_and_keeps_handle(self):
        o = _bare_export()
        o._proc = _FakeProc(dies_on = None)
        assert o._shutdown_subprocess(timeout = 0.01) is False
        assert o._proc is not None
        assert o.is_worker_alive() is True


class TestSpawnPathsHonorFailedShutdown:
    """A fresh-load path must not spawn a second worker over one that outlived
    terminate/kill: the survivor still holds GPU memory and its handle would be lost."""

    def test_export_load_checkpoint_aborts_when_worker_survives(self, monkeypatch):
        import threading

        import utils.transformers_version as tv

        o = ExportOrchestrator.__new__(ExportOrchestrator)
        o._lock = threading.RLock()
        o._proc = _FakeProc(dies_on = None)  # survivor
        o.clear_logs = lambda: None
        o._cancel_requested = False
        o._active_op_kind = None
        o._export_active = False
        o._ensure_subprocess_alive = lambda: True
        o._shutdown_subprocess = lambda *a, **k: False
        o._spawn_subprocess = lambda cfg: pytest.fail("must not spawn over a live survivor")
        o._record_op_finished = lambda *a, **k: None
        monkeypatch.setattr(tv, "sidecar_swap_in_progress", lambda: False)

        ok, msg = o.load_checkpoint(checkpoint_path = "ckpt")

        assert ok is False
        assert "did not exit" in msg
        # The finally cleared the op flags even though we returned early.
        assert o._export_active is False
