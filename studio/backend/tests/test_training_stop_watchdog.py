# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop-watchdog escalation for a stuck training stop.

A save-stop signals the worker and waits for it to save and exit. On some platforms the
worker saves but then wedges in post-save GPU/driver teardown and never exits, leaving the
run stuck in "Stopping..." forever. These tests pin the bounded recovery: the watchdog
escalates to force_terminate() a short grace after "complete" (save done) or after an
absolute timeout (hang during save), and never force-kills a worker that exits cleanly.
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
        b,
        "_finalize_stopped_after_escalation",
        lambda target_proc = None, watched_job_id = None: calls.append("final"),
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
    # save=True, no "complete" yet: a slow save in progress must not be force-killed
    # inside the (long) absolute window.
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
    # A cancel has nothing to save, so it escalates on the shorter cancel cap even before
    # the long save cap elapses.
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
    # A stale watchdog from a prior run must never kill a new run's worker: once
    # self._proc is replaced, it exits silently.
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
    # A stale watchdog sleeping on an old proc must not stop a new run's stop from
    # creating its own watcher.
    b = TrainingBackend()
    started = []
    release = threading.Event()

    def _blocked_watchdog(
        target_proc,
        cancel,
        watched_job_id = None,
    ):
        started.append(target_proc)
        # No timeout: the finally always releases this, so a superseded watchdog stays
        # alive through the assertions regardless of load; as a daemon it can't hang exit.
        release.wait()

    monkeypatch.setattr(b, "_stop_watchdog_loop", _blocked_watchdog)

    old_proc = _FakeProc(alive = True)
    b._proc = old_proc
    b._start_stop_watchdog(cancel = False)
    first_wd = b._stop_watchdog
    assert _wait_until(lambda: started == [old_proc])

    # New run: fresh worker replaces the handle; its stop must get a new watcher
    # even though the old (superseded) watchdog is still alive.
    new_proc = _FakeProc(alive = True)
    b._proc = new_proc
    b._start_stop_watchdog(cancel = False)
    second_wd = b._stop_watchdog

    try:
        assert _wait_until(lambda: started == [old_proc, new_proc])
        assert first_wd.is_alive()
        assert second_wd is not first_wd, "a new run must get its own watchdog"
        assert b._stop_watchdog_proc is new_proc
    finally:
        release.set()
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
        b,
        "_finalize_stopped_after_escalation",
        lambda target_proc = None, watched_job_id = None: finalized.append(True),
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
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append(a))

    b._proc = _FakeProc(alive = True)  # wedged: still reports alive
    b._should_stop = True
    b.current_job_id = "job_c"
    b._db_run_created = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(watched_job_id = "job_c")

    assert b._proc is None, "the wedged handle must be dropped so is_training_active clears"
    assert b._progress.is_training is False
    assert "valid current-step checkpoint" in b._progress.status_message
    assert finstop and finstop[0][0] == "job_c", "the captured run must be finalized by id"
    assert b.is_training_active() is False


def test_finalize_after_escalation_preserves_output_dir(monkeypatch):
    # A save-stop that already emitted "complete" has the checkpoint dir; run history
    # must record it even if the watchdog wins the finalize race against the pump.
    b = TrainingBackend()
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append(a))

    b._proc = _FakeProc(alive = True)
    b._should_stop = True
    b.current_job_id = "job_c"
    b._db_run_created = True
    b._output_dir = "/tmp/outputs/run-123"

    b._finalize_stopped_after_escalation(watched_job_id = "job_c")

    # _finish_stopped_run(run_id, output_dir, batch, final_step, final_loss, duration, loss_history)
    assert finstop and finstop[0][0] == "job_c"
    assert finstop[0][1] == "/tmp/outputs/run-123"


def test_finalize_after_escalation_clears_output_dir_on_cancel(monkeypatch):
    # Stop-without-saving promises no resume: a cancel that escalates through the
    # watchdog must clear the persisted output_dir, not record a checkpoint path.
    b = TrainingBackend()
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append((a, k)))

    b._proc = _FakeProc(alive = True)
    b._should_stop = True
    b._cancel_requested = True
    b.current_job_id = "job_c"
    b._db_run_created = True
    b._output_dir = "/tmp/outputs/run-123"

    b._finalize_stopped_after_escalation(watched_job_id = "job_c")

    assert finstop and finstop[0][0][0] == "job_c"
    assert finstop[0][0][1] is None, "a cancelled run must not record a checkpoint path"
    assert finstop[0][1].get("clear_output_dir") is True
    assert b._output_dir is None, "/status must stop exposing the cancelled run's dir"


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
    # A /start can slip in while the watchdog force-terminates the old worker
    # (is_training_active() is False once _should_stop is set and the old proc is dead).
    # The escalation finalize must then leave the NEW run untouched, not drop its handle.
    b = TrainingBackend()
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append(a))

    old_proc = _FakeProc(alive = False)  # force-terminated worker we were watching
    new_proc = _FakeProc(alive = True)  # a new run already took over
    b._proc = new_proc
    b.current_job_id = "job_new"
    b._db_run_created = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = old_proc)

    assert b._proc is new_proc, "must not drop the new run's handle"
    assert b._progress.is_training is True, "must not mark the new run stopped"
    assert finstop == [], "must not finalize the new run in the DB"


def test_finalize_after_escalation_runs_for_its_own_worker(monkeypatch):
    # Common case: the watched worker is still current, so finalize proceeds and
    # finalizes the captured run by id.
    b = TrainingBackend()
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append(a))

    proc = _FakeProc(alive = False)
    b._proc = proc
    b.current_job_id = "job_a"
    b._db_run_created = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = proc, watched_job_id = "job_a")

    assert b._proc is None
    assert b._progress.is_training is False
    assert finstop and finstop[0][0] == "job_a", "must finalize the captured run by id"


def test_finalize_after_escalation_no_ops_on_job_change_during_startup(monkeypatch):
    # start_training updates current_job_id BEFORE it installs the new _proc, so a stale
    # watchdog can enter while _proc is still the old (dead) handle. The job-id guard must
    # catch this even though the proc-only guard would not.
    b = TrainingBackend()
    finstop: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: finstop.append(a))

    old_proc = _FakeProc(alive = False)  # old worker, dead; new _proc not installed yet
    b._proc = old_proc  # still the old handle (== target), so proc guard would pass
    b.current_job_id = "job_new"  # but the new run already claimed the job id
    b._db_run_created = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = old_proc, watched_job_id = "job_old")

    assert b._proc is old_proc, "must not drop the handle during a new run's startup"
    assert b._progress.is_training is True, "must not mark the starting run stopped"
    assert finstop == [], "must not finalize while a new run is starting up"


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
    recs = {"created": [], "finished": [], "inserted": [], "insert_ids": [], "progress_ids": []}
    fake_storage = _types.ModuleType("storage")
    fake_db = _types.ModuleType("storage.studio_db")
    fake_db.create_run = lambda **kw: recs["created"].append(kw)
    fake_db.finish_run = lambda **kw: recs["finished"].append(kw)
    fake_db.insert_metrics_batch = lambda job_id, batch: (
        recs["inserted"].extend(batch),
        recs["insert_ids"].append(job_id),
    )
    fake_db.update_run_progress = lambda **kw: recs["progress_ids"].append(kw.get("id"))
    fake_db.mark_run_cancel_requested = lambda _run_id: True
    fake_storage.studio_db = fake_db
    monkeypatch.setitem(sys.modules, "storage", fake_storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_db)
    fake_ds = _types.ModuleType("utils.downsample")
    fake_ds.downsample = lambda seq, n: list(seq)[:n]
    monkeypatch.setitem(sys.modules, "utils.downsample", fake_ds)
    return recs


def test_stop_without_save_creates_missing_row_before_signal(monkeypatch):
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id, b._db_config = "job_missing", {"model_name": "m"}
    b._stop_queue = queue.Queue()
    assert b.stop_training(save = False) is True
    assert [run["id"] for run in recs["created"]] == ["job_missing"]
    assert b._stop_queue.get_nowait() == {"type": "stop", "save": False}

    b._cancel_requested = b._should_stop = False
    sys.modules["storage.studio_db"].mark_run_cancel_requested = lambda _run_id: False
    assert b.stop_training(save = False) is False
    assert not b._cancel_requested and b._stop_queue.empty()

    new_queue = queue.Queue()
    b.current_job_id, b._db_run_created = "job_old", True
    b._cancel_requested = b._should_stop = False

    def _supersede(_run_id):
        b.current_job_id = "job_new"
        b._stop_queue = new_queue
        return True

    sys.modules["storage.studio_db"].mark_run_cancel_requested = _supersede
    assert b.stop_training(save = False) is False
    assert not b._cancel_requested and new_queue.empty()


def test_finalize_run_in_db_single_winner_under_concurrency(monkeypatch):
    # The watchdog and pump can both finalize; only one call may reach finish_run.
    recs = _install_fake_db(monkeypatch)
    monkeypatch.setitem(_G, "_DB_FINALIZE_RETRY_S", 0.0)
    attempts = 0

    def flaky_finish(**kw):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("database is locked")
        recs["finished"].append(kw)

    sys.modules["storage.studio_db"].finish_run = flaky_finish
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
    assert attempts == 3
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


def test_flush_pins_to_passed_run_id(monkeypatch):
    # A finalizer flushes to the run it captured, even if a new /start has already
    # changed current_job_id.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_new"  # a new run is already live
    b._db_run_created = True
    b._metric_buffer[:] = [{"step": 1}, {"step": 2}]

    b._flush_metrics_to_db(run_id = "job_old")

    assert recs["insert_ids"] == ["job_old"], "metrics must go to the captured run, not the new one"
    assert recs["progress_ids"] == ["job_old"]


def test_finalize_uses_snapshot_run_id_across_new_run(monkeypatch):
    # If a new /start changes current_job_id after the finalize claim but before the DB
    # writes, finish_run must still target the run captured under the lock.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_x"
    b._db_run_created = True
    b._run_finalized = False

    def hijack(run_id = None):
        # Simulate a new run taking over during the flush (after the finalize claim).
        b.current_job_id = "job_y"

    monkeypatch.setattr(b, "_flush_metrics_to_db", hijack)

    b._finalize_run_in_db(status = "stopped", expected_job_id = "job_x")

    assert [f["id"] for f in recs["finished"]] == [
        "job_x"
    ], "finish_run must target the captured run, not the run that replaced it"


# ----------------------------------------------------------------------------
# (g) DB row creation must not be published before the insert commits.
# ----------------------------------------------------------------------------


def test_ensure_db_run_created_publishes_only_after_insert(monkeypatch):
    # _db_run_created must stay False while create_run is in flight, so a concurrent
    # finalize can't run finish_run (an UPDATE) against a not-yet-inserted row.
    b = TrainingBackend()
    b.current_job_id = "job_z"
    b._db_config = {"model_name": "m"}
    observed: dict = {}

    fake_storage = _types.ModuleType("storage")
    fake_db = _types.ModuleType("storage.studio_db")

    def _create(**kw):
        observed["flag_during_create"] = b._db_run_created
        observed["in_progress_during_create"] = b._db_create_in_progress

    fake_db.create_run = _create
    fake_storage.studio_db = fake_db
    monkeypatch.setitem(sys.modules, "storage", fake_storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_db)

    b._run_intent_lock.acquire()
    creator = threading.Thread(target = b._ensure_db_run_created)
    creator.start()
    time.sleep(0.02)
    assert b._db_create_in_progress is False
    b._run_intent_lock.release()
    creator.join(timeout = 5)

    assert observed["flag_during_create"] is False, "flag must not be published before insert"
    assert observed["in_progress_during_create"] is True
    assert b._db_run_created is True, "flag must be published after a successful insert"
    assert b._db_create_in_progress is False


def test_ensure_db_run_created_stays_unpublished_on_failure(monkeypatch):
    # If create_run raises, neither flag stays set, so a later caller can retry.
    b = TrainingBackend()
    b.current_job_id = "job_z"
    b._db_config = {"model_name": "m"}

    fake_storage = _types.ModuleType("storage")
    fake_db = _types.ModuleType("storage.studio_db")

    def _boom_create(**kw):
        raise RuntimeError("insert failed")

    fake_db.create_run = _boom_create
    fake_storage.studio_db = fake_db
    monkeypatch.setitem(sys.modules, "storage", fake_storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_db)

    b._ensure_db_run_created()

    assert b._db_run_created is False, "a failed insert must not publish the row as created"
    assert b._db_create_in_progress is False, "the in-progress flag must be cleared on failure"


def test_ensure_db_run_created_does_not_publish_for_a_new_run(monkeypatch):
    # A killed worker lets a new /start proceed while the watchdog is still creating the old
    # run's row. The stale create must not publish the backend-wide flags against the new
    # current_job_id, or the new run would skip inserting its own row.
    b = TrainingBackend()
    b.current_job_id = "job_old"
    b._db_config = {"model_name": "m"}
    b._db_run_created = False
    b._db_create_in_progress = False

    fake_storage = _types.ModuleType("storage")
    fake_db = _types.ModuleType("storage.studio_db")

    def _create(**kw):
        b.current_job_id = "job_new"  # a new run takes over during the slow create

    fake_db.create_run = _create
    fake_storage.studio_db = fake_db
    monkeypatch.setitem(sys.modules, "storage", fake_storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_db)

    b._ensure_db_run_created()

    assert b._db_run_created is False, "must not publish the created flag against the new run"
    # The stale claim is left for start_training to reset, not satisfied for the new run.
    assert b._db_create_in_progress is True, "must not clear the claim once the run is not current"


# ----------------------------------------------------------------------------
# (h) The escalation finalizes the watched run by id (so it is never left running).
# ----------------------------------------------------------------------------


def test_escalation_finalizes_watched_run_by_id_end_to_end(monkeypatch):
    # Exercise the real _finish_stopped_run against a fake DB. The watched run is finalized
    # by its captured id with its buffered metrics, so a new run that starts in the gap
    # after the backend goes idle can never leave the stopped run recorded running.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_old"
    b._db_run_created = True
    b._should_stop = True
    b._proc = _FakeProc(alive = False)
    b._progress.is_training = True
    b._progress.step = 42
    b._metric_buffer[:] = [{"step": 41}, {"step": 42}]

    b._finalize_stopped_after_escalation(target_proc = b._proc, watched_job_id = "job_old")

    assert [f["id"] for f in recs["finished"]] == ["job_old"], "must finish the captured run by id"
    assert recs["finished"][0]["status"] == "error"
    assert recs["finished"][0]["resume_blocked"] is True
    assert recs["insert_ids"] == ["job_old"], "buffered metrics must land on the captured run"
    assert b._metric_buffer == [], "the captured batch must be drained"


def test_escalation_defers_when_row_cannot_be_created_here(monkeypatch):
    # If the row does not exist and cannot be created here (no db_config, or the pump is
    # mid-create), the escalation must not claim _run_finalized or call _finish_stopped_run,
    # so the pump's create-then-finalize records the run. Parent state still clears.
    b = TrainingBackend()
    called: list = []
    monkeypatch.setattr(b, "_finish_stopped_run", lambda *a, **k: called.append(a))

    b._proc = _FakeProc(alive = False)
    b.current_job_id = "job_q"
    b._db_run_created = False  # row not created yet
    b._db_config = None  # ... and cannot be created here
    b._run_finalized = False
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = b._proc, watched_job_id = "job_q")

    assert called == [], "must not finalize when the row can't be established here"
    assert b._run_finalized is False, "must not claim the finalize the pump still owes"
    assert b._progress.is_training is False, "parent state must still clear so the UI unsticks"
    assert b._proc is None


def test_escalation_creates_row_then_finalizes_when_start_create_failed(monkeypatch):
    # A wedged worker's pump can never finalize and would bail once _proc is dropped, so if
    # the row was never created (start-time create failed) the escalation creates it and
    # finalizes by id itself, recording the terminal state before dropping the handle.
    recs = _install_fake_db(monkeypatch)
    b = TrainingBackend()
    b.current_job_id = "job_s"
    b._db_config = {"model_name": "m"}  # so _ensure_db_run_created can create the row
    b._db_run_created = False  # start-time create failed
    b._proc = _FakeProc(alive = True)  # wedged: still reports alive
    b._should_stop = True
    b._progress.is_training = True

    b._finalize_stopped_after_escalation(target_proc = b._proc, watched_job_id = "job_s")

    assert [c["id"] for c in recs["created"]] == ["job_s"], "must create the missing row"
    assert [f["id"] for f in recs["finished"]] == ["job_s"], "must finish the created row by id"
    assert b._proc is None, "handle dropped only after the terminal state is recorded"
    assert b._db_run_created is True


def test_escalation_does_not_drop_a_new_runs_handle(monkeypatch):
    # If a run replaces the worker while the finalize DB write is in flight, the final _proc
    # drop must leave the new run's handle intact (re-guarded on target_proc).
    b = TrainingBackend()
    b.current_job_id = "job_old"
    b._db_run_created = True
    old_proc = _FakeProc(alive = False)
    new_proc = _FakeProc(alive = True)
    b._proc = old_proc

    def hijack(*a, **k):
        b._proc = new_proc  # a new run takes over during the finalize

    monkeypatch.setattr(b, "_finish_stopped_run", hijack)

    b._finalize_stopped_after_escalation(target_proc = old_proc, watched_job_id = "job_old")

    assert b._proc is new_proc, "must not drop the handle a new run installed during finalize"


def _make_finish_raise(monkeypatch, calls):
    fn = sys.modules["storage.studio_db"]

    def _boom(**kw):
        calls.append(kw)
        raise RuntimeError("database is locked")

    fn.finish_run = _boom


def test_finish_stopped_run_retries_then_unclaims_on_db_error(monkeypatch):
    # The watchdog is the sole finalizer once _proc is dropped, so a transient DB error is
    # retried a few times; on final failure the finalize is unclaimed (run still current).
    monkeypatch.setitem(_G, "_DB_FINALIZE_RETRY_S", 0.0)
    _install_fake_db(monkeypatch)
    tries: list = []
    _make_finish_raise(monkeypatch, tries)
    b = TrainingBackend()
    b.current_job_id = "job_r"
    b._run_finalized = True  # the caller (escalation) already claimed

    b._finish_stopped_run("job_r", None, [{"step": 1}], 1, None, None, [])

    assert len(tries) == 3, "a transient DB error must be retried before giving up"
    assert b._run_finalized is False, "a persistent DB error must unclaim the finalize"


def test_finish_stopped_run_error_leaves_new_run_untouched(monkeypatch):
    # If the watched run was superseded, a DB error must not unclaim the new run's finalize.
    monkeypatch.setitem(_G, "_DB_FINALIZE_RETRY_S", 0.0)
    _install_fake_db(monkeypatch)
    _make_finish_raise(monkeypatch, [])
    b = TrainingBackend()
    b.current_job_id = "job_new"  # a new run is live
    b._run_finalized = True  # the new run's flag

    b._finish_stopped_run("job_old", None, [{"step": 1}], 1, None, None, [])

    assert b._run_finalized is True, "must not unclaim the new run's finalize"
