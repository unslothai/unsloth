# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop-watchdog escalation for a stuck training stop.

A save-stop only signals the worker and waits for it to save and exit. On some
platforms the worker saves successfully but then wedges in post-save GPU/driver
teardown and never exits, leaving the run stuck in "Stopping..." forever. These
tests pin the bounded recovery: the watchdog escalates to force_terminate() a
short grace after "complete" (save done) or after an absolute timeout (hang during
save), and never force-kills a worker that exits cleanly on its own. Fakes only;
no GPU, network, or subprocess.
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

if not _TRAINING_PRE_IMPORTED:
    sys.modules.pop("core.training.training", None)
    sys.modules.pop("core.training", None)

# The module globals hold the escalation timeouts and are the watchdog's own
# namespace; patch them here so tests run in well under a second.
_G = TrainingBackend._stop_watchdog_loop.__globals__


class _FakeProc:
    """A subprocess handle whose liveness and kill calls the test observes."""

    def __init__(self, alive: bool = True):
        self._alive = alive
        self.pid = 4321
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout = None):
        pass


def _wait_until(predicate, timeout = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _record_force_terminate(monkeypatch, b):
    """Replace force_terminate + escalation finalize with recorders (no DB/OS)."""
    calls: list = []
    monkeypatch.setattr(b, "force_terminate", lambda target_proc = None: calls.append("force"))
    monkeypatch.setattr(
        b, "_finalize_stopped_after_escalation", lambda target_proc = None: calls.append("final")
    )
    return calls


# ----------------------------------------------------------------------------
# (a) Escalate a short grace after "complete" (save done) if still alive.
# ----------------------------------------------------------------------------


def test_watchdog_escalates_after_grace_once_complete_seen(monkeypatch):
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 0.05)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)  # ensure grace, not timeout, fires
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    proc = _FakeProc(alive = True)
    b._proc = proc
    b._complete_seen.set()  # worker reported "complete" -> save is done

    b._start_stop_watchdog(cancel = False)
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "watchdog must force_terminate a worker still alive after the post-save grace"
    b._stop_watchdog.join(timeout = 5)


# ----------------------------------------------------------------------------
# (b) The absolute cap is a last-resort backstop, not a save killer.
# ----------------------------------------------------------------------------


def test_watchdog_does_not_kill_save_still_saving_within_window(monkeypatch):
    # save=True, no "complete" yet: a large/slow save is in progress. It must not be
    # force-killed while inside the (long) absolute window.
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    proc = _FakeProc(alive = True)
    b._proc = proc
    b._start_stop_watchdog(cancel = False)

    time.sleep(0.3)
    assert calls == [], "an in-progress save must not be killed within the absolute window"
    assert b._stop_watchdog.is_alive()

    proc._alive = False
    b._stop_watchdog.join(timeout = 5)


def test_watchdog_backstop_fires_for_save_after_absolute_timeout(monkeypatch):
    # Past the long save=True cap with no completion: force-terminate as last resort.
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)  # never trips (no complete)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 0.05)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    b._proc = _FakeProc(alive = True)
    b._start_stop_watchdog(cancel = False)
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "the absolute backstop must force_terminate a save that never completes"
    b._stop_watchdog.join(timeout = 5)


def test_cancel_uses_shorter_absolute_timeout(monkeypatch):
    # A cancel has nothing to save, so it escalates on the shorter cancel cap even
    # when the long save cap has not elapsed.
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)  # save cap would not fire
    monkeypatch.setitem(_G, "_CANCEL_TIMEOUT_S", 0.05)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    b._proc = _FakeProc(alive = True)
    b._start_stop_watchdog(cancel = True)
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "a cancel must escalate on the shorter cancel timeout"
    b._stop_watchdog.join(timeout = 5)


# ----------------------------------------------------------------------------
# (c) No force-kill when the worker exits cleanly and promptly.
# ----------------------------------------------------------------------------


def test_watchdog_no_op_on_clean_quick_exit(monkeypatch):
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 5.0)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 10.0)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    proc = _FakeProc(alive = True)
    b._proc = proc
    b._complete_seen.set()  # save done; worker is about to exit on its own

    b._start_stop_watchdog(cancel = False)
    # Worker exits promptly, well before the grace period elapses.
    time.sleep(0.1)
    proc._alive = False

    b._stop_watchdog.join(timeout = 5)
    assert not b._stop_watchdog.is_alive()
    assert calls == [], "a clean quick exit must not trigger force_terminate"


def test_watchdog_no_op_when_worker_superseded(monkeypatch):
    # A stale watchdog from a prior run must never kill the worker of a new run:
    # once self._proc is replaced, it exits silently.
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 0.05)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 0.05)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    old_proc = _FakeProc(alive = True)
    b._proc = old_proc
    b._complete_seen.set()
    b._start_stop_watchdog(cancel = False)

    # A new run takes over the handle before the grace elapses.
    b._proc = _FakeProc(alive = True)

    b._stop_watchdog.join(timeout = 5)
    assert calls == [], "watchdog must not force_terminate a superseded worker"


def test_new_run_gets_its_own_watchdog(monkeypatch):
    # A stale watchdog still sleeping on an old proc must not stop a new run's stop
    # from creating its own watcher.
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)
    b = TrainingBackend()
    _record_force_terminate(monkeypatch, b)

    old_proc = _FakeProc(alive = True)
    b._proc = old_proc
    b._start_stop_watchdog(cancel = False)
    first_wd = b._stop_watchdog

    # New run: fresh worker replaces the handle; its stop must get a new watcher
    # even though the old (superseded) watchdog is still alive.
    new_proc = _FakeProc(alive = True)
    b._proc = new_proc
    b._start_stop_watchdog(cancel = False)
    second_wd = b._stop_watchdog

    try:
        assert first_wd.is_alive()
        assert second_wd is not first_wd, "a new run must get its own watchdog"
        assert b._stop_watchdog_proc is new_proc
    finally:
        old_proc._alive = False
        new_proc._alive = False
        first_wd.join(timeout = 5)
        second_wd.join(timeout = 5)


def test_force_terminate_targets_only_captured_proc():
    # Superseded: force_terminate(target) must not touch a different current worker.
    b = TrainingBackend()
    old_proc = _FakeProc(alive = True)
    new_proc = _FakeProc(alive = True)
    b._proc = new_proc
    b.force_terminate(target_proc = old_proc)
    assert new_proc.terminated is False, "must not terminate the new run's worker"
    assert old_proc.terminated is False, "must not terminate a handle that is not current"

    # Matching: the captured handle is the current worker, so it is terminated.
    p = _FakeProc(alive = True)
    b._proc = p
    b.force_terminate(target_proc = p)
    assert p.terminated is True


# ----------------------------------------------------------------------------
# Post-escalation finalize leaves the parent ready for a new run.
# ----------------------------------------------------------------------------


def test_finalize_runs_even_if_force_terminate_raises(monkeypatch):
    # A wedged child can make force_terminate() raise; finalize must still run so the
    # run does not stay stuck in "Stopping...".
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 0.05)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)
    b = TrainingBackend()

    def _boom(target_proc = None):
        raise RuntimeError("kill() failed on wedged child")

    finalized: list = []
    monkeypatch.setattr(b, "force_terminate", _boom)
    monkeypatch.setattr(
        b, "_finalize_stopped_after_escalation", lambda target_proc = None: finalized.append(True)
    )

    b._proc = _FakeProc(alive = True)
    b._complete_seen.set()
    b._start_stop_watchdog(cancel = False)

    assert _wait_until(
        lambda: finalized == [True]
    ), "finalize must run even when force_terminate raises"
    b._stop_watchdog.join(timeout = 5)


def test_finalize_after_escalation_clears_state(monkeypatch):
    # Even if the OS never reaps the wedged worker, the parent must report the run
    # stopped so the UI leaves "Stopping..." and a new run can start.
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    b._proc = _FakeProc(alive = True)  # wedged: still reports alive
    b._should_stop = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation()

    assert b._proc is None, "the wedged handle must be dropped so is_training_active clears"
    assert b._progress.is_training is False
    assert b._progress.status_message == "Training stopped."
    assert finalized.get("status") == "stopped"
    assert b.is_training_active() is False


def test_finalize_after_escalation_preserves_output_dir(monkeypatch):
    # A save-stop that already emitted "complete" has the checkpoint dir; run history
    # must record it even if the watchdog wins the finalize race against the pump.
    b = TrainingBackend()
    finalized: dict = {}
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.update(kw))

    b._proc = _FakeProc(alive = True)
    b._should_stop = True
    b._output_dir = "/tmp/outputs/run-123"

    b._finalize_stopped_after_escalation()

    assert finalized.get("status") == "stopped"
    assert finalized.get("output_dir") == "/tmp/outputs/run-123"


def test_stop_training_starts_watchdog_only_when_worker_alive(monkeypatch):
    # No worker -> nothing to escalate; the watchdog must not spawn.
    b = TrainingBackend()
    b._proc = None
    assert b.stop_training(save = True) is True
    assert b._stop_watchdog is None


# ----------------------------------------------------------------------------
# (d) A stale watchdog must never clobber a run that replaced its worker.
# ----------------------------------------------------------------------------


def test_finalize_after_escalation_no_ops_when_superseded(monkeypatch):
    # A /start can slip in while the watchdog is force-terminating the old worker
    # (is_training_active() is False once _should_stop is set and the old proc is
    # dead). The escalation finalize must then leave the NEW run untouched instead
    # of dropping its handle and marking it stopped.
    b = TrainingBackend()
    finalized: list = []
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.append(kw))

    old_proc = _FakeProc(alive = False)  # force-terminated worker we were watching
    new_proc = _FakeProc(alive = True)  # a new run already took over
    b._proc = new_proc
    b.current_job_id = "job_new"
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = old_proc)

    assert b._proc is new_proc, "must not drop the new run's handle"
    assert b._progress.is_training is True, "must not mark the new run stopped"
    assert finalized == [], "must not finalize the new run in the DB"


def test_finalize_after_escalation_runs_for_its_own_worker(monkeypatch):
    # The common case: the worker we watched is still current, so finalize proceeds.
    b = TrainingBackend()
    finalized: list = []
    monkeypatch.setattr(b, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(b, "_finalize_run_in_db", lambda **kw: finalized.append(kw))

    proc = _FakeProc(alive = False)
    b._proc = proc
    b.current_job_id = "job_a"
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = proc)

    assert b._proc is None
    assert b._progress.is_training is False
    assert finalized and finalized[0].get("status") == "stopped"
    assert finalized[0].get("expected_job_id") == "job_a"


# ----------------------------------------------------------------------------
# (e) A later cancel (save=False) tightens an in-flight save watchdog.
# ----------------------------------------------------------------------------


def test_later_cancel_tightens_watchdog_timeout(monkeypatch):
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)  # never trips (no complete)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 100.0)  # save cap would not fire
    monkeypatch.setitem(_G, "_CANCEL_TIMEOUT_S", 0.05)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    b._proc = _FakeProc(alive = True)
    b._start_stop_watchdog(cancel = False)  # started as a save-stop with the long cap
    time.sleep(0.15)
    assert calls == [], "a save-stop must not escalate on the short cancel cap yet"

    # The user now cancels the in-flight stop: the watchdog must tighten its cap.
    b._cancel_requested = True
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "a later cancel must tighten the watchdog to the shorter cancel cap"
    b._stop_watchdog.join(timeout = 5)


# ----------------------------------------------------------------------------
# (f) DB finalize/flush are safe when the watchdog and pump race (see Item 4).
# ----------------------------------------------------------------------------


def _install_fake_db(monkeypatch):
    """Stub storage.studio_db + utils.downsample so the real DB helpers run without
    SQLite. Returns the recorder dict."""
    recs = {"created": [], "finished": [], "inserted": []}
    fake_storage = _types.ModuleType("storage")
    fake_db = _types.ModuleType("storage.studio_db")
    fake_db.create_run = lambda **kw: recs["created"].append(kw)
    fake_db.finish_run = lambda **kw: recs["finished"].append(kw)
    fake_db.insert_metrics_batch = lambda job_id, batch: recs["inserted"].extend(batch)
    fake_db.update_run_progress = lambda **kw: None
    fake_storage.studio_db = fake_db
    monkeypatch.setitem(sys.modules, "storage", fake_storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_db)
    fake_ds = _types.ModuleType("utils.downsample")
    fake_ds.downsample = lambda seq, n: list(seq)[:n]
    monkeypatch.setitem(sys.modules, "utils.downsample", fake_ds)
    return recs


def test_finalize_run_in_db_single_winner_under_concurrency(monkeypatch):
    # The watchdog and pump can both finalize; only one call may reach finish_run.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_x"
    b._db_run_created = True
    b._run_finalized = False

    start = threading.Barrier(8)

    def worker():
        start.wait()
        b._finalize_run_in_db(status = "stopped")

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 5)

    assert len(recs["finished"]) == 1, f"finalize must run once, got {len(recs['finished'])}"
    assert b._run_finalized is True


def test_finalize_run_in_db_no_ops_on_job_mismatch(monkeypatch):
    # A finalize captured for an old job must not finalize the run that replaced it.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_new"
    b._db_run_created = True
    b._run_finalized = False

    b._finalize_run_in_db(status = "stopped", expected_job_id = "job_old")

    assert recs["finished"] == [], "a superseded job id must not finalize the current run"
    assert b._run_finalized is False


def test_concurrent_flush_claims_each_metric_once(monkeypatch):
    # Concurrent flushes (pump periodic flush vs watchdog finalize flush) must not
    # double-remove or drop buffered metrics.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_y"
    b._db_run_created = True
    b._metric_buffer[:] = [{"step": i} for i in range(200)]

    start = threading.Barrier(6)

    def worker():
        start.wait()
        for _ in range(50):
            b._flush_metrics_to_db()

    threads = [threading.Thread(target = worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 5)
    b._flush_metrics_to_db()  # drain any remainder

    steps = sorted(m["step"] for m in recs["inserted"])
    assert steps == list(range(200)), "each metric must be inserted exactly once"
    assert b._metric_buffer == [], "the buffer must be fully drained"
