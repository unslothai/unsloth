# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parent-side training event-pump resilience.

The pump is the only writer of the progress state /progress, /status, /metrics
and DB history read. If it died while the worker ran, the run would continue while
the UI froze -- the "training runs but no progress shows" symptom. These tests pin
two guards: a bad event/queue error can't kill the pump, and a dead pump is
detected and restarted (even after worker exit) so terminal events still finalize.
Fakes only; no GPU, network, or subprocess.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import time
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

# Whether core.training.training was already imported before this file ran; only
# evict it below if we were the one to create the (stub-bound) module instance.
_TRAINING_PRE_IMPORTED = "core.training.training" in sys.modules

from core.training.training import TrainingBackend

# Restore every stubbed module so this file never pollutes the shared session.
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

# training imported its helpers while the stubs were active, binding them to stubs.
# If we created the cached module, evict it (and its parent) so a later test
# re-imports the real one.
if not _TRAINING_PRE_IMPORTED:
    sys.modules.pop("core.training.training", None)
    sys.modules.pop("core.training", None)


class _FakeProc:
    """A subprocess handle whose liveness the test drives directly."""

    def __init__(self, alive: bool = True):
        self._alive = alive
        self.pid = 4321

    def is_alive(self):
        return self._alive

    def join(self, timeout = None):
        self._alive = False


class _IdleQueue:
    """get()/get_nowait() always signal "no event" so the pump idles."""

    def put(self, *a, **k):
        pass

    def get(self, *a, **k):
        raise queue.Empty

    def get_nowait(self, *a, **k):
        raise queue.Empty


class _ScriptedQueue:
    """Yields queued events once, then signals empty forever."""

    def __init__(self, events):
        self._events = list(events)

    def put(self, *a, **k):
        pass

    def get(self, *a, **k):
        if self._events:
            return self._events.pop(0)
        raise queue.Empty

    def get_nowait(self, *a, **k):
        if self._events:
            return self._events.pop(0)
        raise queue.Empty


def _dead_thread() -> threading.Thread:
    t = threading.Thread(target = lambda: None)
    t.start()
    t.join()
    return t


def _silence_db(monkeypatch, b):
    """Neutralize DB finalization so a started pump exits cleanly off-box."""
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **k: None)


def _wait_until(predicate, timeout = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ----------------------------------------------------------------------------
# Guarantee 1: a single bad event/queue error cannot kill the pump.
# ----------------------------------------------------------------------------


def test_pump_survives_handler_exception_and_keeps_processing(monkeypatch):
    b = TrainingBackend()
    _silence_db(monkeypatch, b)
    handled: list = []

    def fake_handle(ev):
        if ev.get("type") == "boom":
            raise RuntimeError("handler blew up")
        handled.append(ev.get("type"))

    monkeypatch.setattr(b, "_handle_event", fake_handle)

    proc = _FakeProc(alive = True)
    b._proc = proc
    b._event_queue = _ScriptedQueue(
        [{"type": "boom"}, {"type": "progress"}, {"type": "boom"}, {"type": "progress"}]
    )

    pump = threading.Thread(target = b._pump_loop, daemon = True)
    pump.start()
    try:
        assert _wait_until(
            lambda: handled.count("progress") == 2
        ), "pump must keep processing good events after handler exceptions"
        assert pump.is_alive(), "pump thread must survive handler exceptions"
        assert b._pump_running is True
    finally:
        proc._alive = False  # let the loop reach its clean exit
        pump.join(timeout = 5)

    assert not pump.is_alive()
    assert b._pump_running is False, "clean exit must clear the running flag"


def test_read_queue_narrow_contract():
    class _Q:
        def __init__(self, exc):
            self.exc = exc

        def get(self, *a, **k):
            raise self.exc

    # Expected closed/broken-queue signals read as "no event".
    for exc in (queue.Empty(), EOFError(), OSError(), ValueError()):
        assert TrainingBackend._read_queue(_Q(exc), 0.01) is None

    # Anything unexpected propagates on purpose to _pump_loop's guarded block,
    # which logs and backs off instead of swallowing it into a hot loop.
    with pytest.raises(RuntimeError):
        TrainingBackend._read_queue(_Q(RuntimeError("boom")), 0.01)


def test_pump_survives_queue_read_exception_and_recovers(monkeypatch):
    # _read_queue raising an unexpected error must be caught by the pump's outer
    # guard (log + backoff), not kill the pump; once reads recover it processes.
    b = TrainingBackend()
    _silence_db(monkeypatch, b)
    handled: list = []
    monkeypatch.setattr(b, "_handle_event", lambda ev: handled.append(ev.get("type")))

    class _FlakyQueue:
        def __init__(self):
            self.calls = 0

        def get(self, *a, **k):
            self.calls += 1
            if self.calls <= 3:
                raise RuntimeError("transient queue read error")
            if self.calls == 4:
                return {"type": "progress", "step": 1}
            raise queue.Empty

        def get_nowait(self, *a, **k):
            raise queue.Empty

    proc = _FakeProc(alive = True)
    b._proc = proc
    b._event_queue = _FlakyQueue()

    pump = threading.Thread(target = b._pump_loop, daemon = True)
    pump.start()
    try:
        assert _wait_until(
            lambda: handled == ["progress"]
        ), "pump must recover after read errors and process the next event"
        assert pump.is_alive()
    finally:
        proc._alive = False
        pump.join(timeout = 5)


def test_pump_finalizes_when_drain_queue_raises_unexpected_error(monkeypatch):
    # Worker has exited; the final drain hits an unexpected error. The run must
    # still be finalized (not wedged "active" with a dead worker).
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    class _BadDrainQueue:
        def get(self, *a, **k):
            raise queue.Empty

        def get_nowait(self, *a, **k):
            raise RuntimeError("corrupt drain payload")

    b._proc = _FakeProc(alive = False)
    b._event_queue = _BadDrainQueue()
    b._progress.is_training = True

    b._pump_loop()  # returns once it sees the dead worker

    assert b._progress.is_training is False
    assert b._progress.error == "Training process exited unexpectedly"
    assert finalized.get("status") == "error"
    assert b._pump_running is False
    assert b.is_training_active() is False


def test_pump_finalizes_when_read_keeps_raising_on_dead_worker(monkeypatch):
    # An unexpected error escapes _read_queue to the pump's outer guard; if it
    # keeps raising after worker exit, the loop must still finalize, not spin.
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    class _BrokenReadQueue:
        def get(self, *a, **k):
            raise RuntimeError("broken queue pipe")

        def get_nowait(self, *a, **k):
            raise queue.Empty

    b._proc = _FakeProc(alive = False)
    b._event_queue = _BrokenReadQueue()
    b._progress.is_training = True

    pump = threading.Thread(target = b._pump_loop, daemon = True)
    pump.start()
    pump.join(timeout = 5)
    assert not pump.is_alive(), "pump must finalize a dead worker even when reads keep raising"
    assert b._progress.is_training is False
    assert finalized.get("status") == "error"
    assert b._pump_running is False


def test_interrupted_cancel_clears_in_memory_output_dir(monkeypatch):
    # Stop-without-save interrupted before its complete event: /status must not
    # keep serving the cleared run's output_dir.
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    b._proc = _FakeProc(alive = False)
    b._event_queue = _IdleQueue()
    b._progress.is_training = True
    b._should_stop = True
    b._cancel_requested = True
    b._output_dir = "/out/x"

    b._pump_loop()

    assert b._output_dir is None
    assert finalized.get("status") == "stopped"
    assert finalized.get("output_dir") is None
    assert finalized.get("clear_output_dir") is True


def test_worker_exit_reuses_terminal_stop_save_error(monkeypatch):
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    b._proc = _FakeProc(alive = False)
    b._event_queue = _IdleQueue()
    b._progress.is_training = True
    b._should_stop = True
    b._cancel_requested = False
    b._output_dir = "/out/x"
    b.current_job_id = "job-x"
    b._terminal_finalize_payload = {
        "status": "error",
        "error_message": "checkpoint failed",
        "output_dir": "/out/x",
        "clear_output_dir": False,
        "resume_blocked": True,
        "expected_job_id": "job-x",
    }

    b._pump_loop()

    assert b._output_dir == "/out/x"
    assert finalized.get("status") == "error"
    assert finalized.get("output_dir") == "/out/x"
    assert finalized.get("clear_output_dir") is False
    assert finalized.get("resume_blocked") is True


def test_dead_worker_crash_preserves_output_dir(monkeypatch):
    # A crash (no stop requested) after output_dir was emitted must keep the dir
    # in the error finalize: checkpoints under it may still exist.
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    b._proc = _FakeProc(alive = False)
    b._event_queue = _IdleQueue()
    b._progress.is_training = True
    b._output_dir = "/out/x"

    b._pump_loop()

    assert finalized.get("status") == "error"
    assert finalized.get("output_dir") == "/out/x"
    assert finalized.get("clear_output_dir") is False


def test_start_training_clears_stale_pump_running_flag():
    # A prior pump that died abnormally leaves _pump_running True. The next
    # start_training must clear it during reset so the start-time watchdog can't
    # treat the fresh setup as a recoverable crash and spawn a duplicate pump.
    b = TrainingBackend()
    b._pump_running = True
    b._pump_thread = None
    b._proc = None

    # No model_name -> start_training bails at kwargs["model_name"] (KeyError),
    # but only AFTER the reset block that clears the stale flag.
    with pytest.raises(KeyError):
        b.start_training("job_stale_flag_test")

    assert b._pump_running is False


# ----------------------------------------------------------------------------
# Guarantee 2: a pump that dies while the worker runs is detected + restarted.
# ----------------------------------------------------------------------------


def test_ensure_pump_alive_restarts_crashed_pump(monkeypatch):
    b = TrainingBackend()
    _silence_db(monkeypatch, b)
    b._proc = _FakeProc(alive = True)
    b._event_queue = _IdleQueue()
    b._pump_running = True  # a pump started, then died abnormally
    dead = _dead_thread()
    b._pump_thread = dead

    assert b._ensure_pump_alive() is True
    try:
        assert b._pump_thread is not dead
        assert b._pump_thread.is_alive(), "a fresh pump must be running"
    finally:
        b._proc._alive = False
        b._pump_thread.join(timeout = 5)


def test_ensure_pump_alive_noop_when_pump_alive():
    b = TrainingBackend()
    b._proc = _FakeProc(alive = True)
    b._event_queue = _IdleQueue()
    b._pump_running = True
    release = threading.Event()
    alive = threading.Thread(target = release.wait, daemon = True)
    alive.start()
    b._pump_thread = alive
    try:
        assert b._ensure_pump_alive() is False
        assert b._pump_thread is alive
    finally:
        release.set()
        alive.join(timeout = 5)


def test_ensure_pump_alive_revives_crashed_pump_after_worker_exit(monkeypatch):
    # True _pump_running + dead thread = a crash (the loop clears the flag on
    # intended exits). The queue may still hold terminal events, so the pump must
    # restart to drain and finalize, else the run is stuck "running" forever.
    b = TrainingBackend()
    _silence_db(monkeypatch, b)
    b._proc = _FakeProc(alive = False)
    b._event_queue = _IdleQueue()
    b._progress.is_training = True
    b._pump_running = True
    b._pump_thread = _dead_thread()

    assert b._ensure_pump_alive() is True
    assert _wait_until(
        lambda: b._progress.is_training is False
    ), "the restarted pump must drain + finalize the stranded run"
    b._pump_thread.join(timeout = 5)
    assert b._pump_running is False
    assert b.is_training_active() is False


def test_ensure_pump_alive_noop_during_setup():
    # _pump_running is False between state-reset and the first pump actually
    # running; the watchdog must not race in and spawn a rogue pump.
    b = TrainingBackend()
    b._proc = _FakeProc(alive = True)
    b._event_queue = _IdleQueue()
    b._pump_running = False
    b._pump_thread = None
    assert b._ensure_pump_alive() is False
    assert b._pump_thread is None


def test_is_training_active_revives_dead_pump(monkeypatch):
    b = TrainingBackend()
    _silence_db(monkeypatch, b)
    b._proc = _FakeProc(alive = True)
    b._event_queue = _IdleQueue()
    b._pump_running = True
    dead = _dead_thread()
    b._pump_thread = dead

    # The status poll the SSE stream makes every second both reports activity
    # and heals the dead pump as a side effect.
    assert b.is_training_active() is True
    try:
        assert b._pump_thread is not dead
        assert b._pump_thread.is_alive()
    finally:
        b._proc._alive = False
        b._pump_thread.join(timeout = 5)


# ----------------------------------------------------------------------------
# Guarantee 3: the DB run row exists before the pump consumes any event.
# ----------------------------------------------------------------------------


def _stub_spawn(monkeypatch):
    """Stub start_training's spawn surface (GPU pick, mp context, worker)."""
    g = TrainingBackend.start_training.__globals__

    class _SpawnProc:
        pid = 4321

        def start(self):
            pass

        def is_alive(self):
            return True

    class _Ctx:
        def Queue(self):
            return _IdleQueue()

        def Process(self, **k):
            return _SpawnProc()

    # _CTX / prepare_gpu_selection resolve from the module globals; patch the
    # function's own globals so the eviction of core.training.training (done at
    # this test module's import for isolation) can't hand us a different copy.
    monkeypatch.setitem(g, "_CTX", _Ctx())
    monkeypatch.setitem(g, "prepare_gpu_selection", lambda *a, **k: (None, None))

    hw = _types.ModuleType("utils.hardware")
    hw.prepare_gpu_selection = lambda *a, **k: (None, None)
    hw.hardware = type("HW", (), {"DEVICE": "cuda", "DeviceType": type("D", (), {"MLX": "mlx"})})()
    monkeypatch.setitem(sys.modules, "utils.hardware", hw)

    pl = _types.ModuleType("utils.process_lifetime")
    pl.adopt_pid = lambda pid: None
    monkeypatch.setitem(sys.modules, "utils.process_lifetime", pl)

    worker = _types.ModuleType("core.training.worker")
    worker.run_training_process = lambda **k: None
    monkeypatch.setitem(sys.modules, "core.training.worker", worker)


def test_db_run_created_before_pump_consumes_events(monkeypatch):
    # A fast terminal worker must not race the pump into creating the DB row: by
    # the time the pump runs, start_training has already created it. The create
    # sleep widens the window so the ordering is observed, not luck.
    b = TrainingBackend()
    _stub_spawn(monkeypatch)

    def slow_create():
        time.sleep(0.05)
        b._db_run_created = True

    seen = {}

    def fake_pump():
        seen["db_created"] = b._db_run_created
        b._pump_running = False

    monkeypatch.setattr(b, "_ensure_db_run_created", slow_create)
    monkeypatch.setattr(b, "_pump_loop", fake_pump)

    assert b.start_training("job_db_order", model_name = "m") is True
    if b._pump_thread is not None:
        b._pump_thread.join(timeout = 2.0)

    # The pump observed an already-created run; it would be False if the pump
    # were started before the eager create.
    assert seen["db_created"] is True


def test_startup_flag_reports_training_active_before_proc():
    # Between freeing VRAM and _proc going live, a concurrent STT load must see
    # training as active so it does not grab the just-freed GPU.
    b = TrainingBackend()
    b._spawn_in_progress = True
    assert b.is_training_active() is True


def test_before_spawn_runs_inside_active_window(monkeypatch):
    # The VRAM-freeing hook must run while training already counts as active, or
    # an STT load racing it would place Whisper back on the freed GPU.
    b = TrainingBackend()
    _stub_spawn(monkeypatch)
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_pump_loop", lambda: setattr(b, "_pump_running", False))

    active_during_free = {}

    def before_spawn():
        active_during_free["value"] = b.is_training_active()

    assert b.start_training("job_active_window", model_name = "m", before_spawn = before_spawn) is True
    if b._pump_thread is not None:
        b._pump_thread.join(timeout = 2.0)

    assert active_during_free["value"] is True
    # The transient flag clears, but the live proc keeps training active.
    assert b._spawn_in_progress is False
    assert b.is_training_active() is True
