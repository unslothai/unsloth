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
    monkeypatch.setattr(b, "force_terminate", lambda: calls.append("force"))
    monkeypatch.setattr(b, "_finalize_stopped_after_escalation", lambda: calls.append("final"))
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

    b._start_stop_watchdog()
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "watchdog must force_terminate a worker still alive after the post-save grace"
    b._stop_watchdog.join(timeout = 5)


# ----------------------------------------------------------------------------
# (b) Escalate after the absolute timeout when no "complete" ever arrives.
# ----------------------------------------------------------------------------


def test_watchdog_escalates_after_absolute_timeout(monkeypatch):
    monkeypatch.setitem(_G, "_STOP_GRACE_S", 100.0)  # never trips (no complete anyway)
    monkeypatch.setitem(_G, "_STOP_TIMEOUT_S", 0.05)
    b = TrainingBackend()
    calls = _record_force_terminate(monkeypatch, b)

    proc = _FakeProc(alive = True)
    b._proc = proc
    # No _complete_seen: simulates a hang during save, no "complete" ever sent.

    b._start_stop_watchdog()
    assert _wait_until(
        lambda: calls == ["force", "final"]
    ), "watchdog must force_terminate a worker that never exits within the timeout"
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

    b._start_stop_watchdog()
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
    b._start_stop_watchdog()

    # A new run takes over the handle before the grace elapses.
    b._proc = _FakeProc(alive = True)

    b._stop_watchdog.join(timeout = 5)
    assert calls == [], "watchdog must not force_terminate a superseded worker"


# ----------------------------------------------------------------------------
# Post-escalation finalize leaves the parent ready for a new run.
# ----------------------------------------------------------------------------


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


def test_stop_training_starts_watchdog_only_when_worker_alive(monkeypatch):
    # No worker -> nothing to escalate; the watchdog must not spawn.
    b = TrainingBackend()
    b._proc = None
    assert b.stop_training(save = True) is True
    assert b._stop_watchdog is None
