# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two surfaces that hung at 100% when a finished worker would not exit (#7897).

/api/train/status and the progress SSE keyed off liveness-based is_training_active(), so a
worker wedged in post-save teardown kept reporting "training" forever; they now consult
is_run_finished() too. A live run must be unaffected, and /api/train/stop must be
terminal-aware too: a late Stop must not latch _should_stop over the finished banner.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.training as rt
from core.training.training import TrainingBackend, TrainingProgress


async def _inline_to_thread(function, /, *args, **kwargs):
    return function(*args, **kwargs)


@pytest.fixture(autouse = True)
def _run_route_helpers_inline(monkeypatch):
    monkeypatch.setattr(rt.asyncio, "to_thread", _inline_to_thread)


class _WedgedProc:
    """A worker that reported terminal and then never exits."""

    pid = 999

    def __init__(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def join(self, timeout = None):
        pass


def _running(monkeypatch, job_id = "job_1"):
    b = TrainingBackend()
    b.current_job_id = job_id
    b._proc = _WedgedProc()
    b._progress = TrainingProgress(is_training = True, status_message = "Training in progress...")
    b._finalize_run_in_db = lambda **kw: None
    b._ensure_db_run_created = lambda: None
    b._start_stop_watchdog = lambda **kw: None  # keep the worker wedged on purpose
    monkeypatch.setattr(rt, "get_training_backend", lambda: b)
    return b


_DONE = {
    "type": "complete",
    "output_dir": "/tmp/out",
    "status_message": "Training completed! Model saved to /tmp/out",
}


class _Req:
    headers: dict = {}

    async def is_disconnected(self):
        return False


async def _sse_events(timeout = 10.0):
    """Event names the real SSE generator yields until it closes."""
    resp = await rt.stream_training_progress(_Req(), current_subject = "t")
    names: list[str] = []

    async def pump():
        async for chunk in resp.body_iterator:
            for line in str(chunk).splitlines():
                if line.startswith("event: "):
                    names.append(line[7:].strip())
            if names and names[-1] in ("complete", "error"):
                return

    try:
        await asyncio.wait_for(pump(), timeout = timeout)
    except asyncio.TimeoutError:
        pass
    return names


def test_status_reports_completed_while_worker_still_wedged(monkeypatch):
    b = _running(monkeypatch)
    b._handle_event(dict(_DONE))
    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.is_training_running is False
    assert st.phase == "completed"
    assert st.message.startswith("Training completed!")
    assert b._proc.is_alive() is True


def test_status_reports_error_while_worker_still_wedged(monkeypatch):
    b = _running(monkeypatch)
    b._handle_event({"type": "error", "error": "CUDA OOM", "stack": ""})
    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.is_training_running is False
    assert st.phase == "error"


def test_status_unchanged_mid_run(monkeypatch):
    _running(monkeypatch)
    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.is_training_running is True
    assert st.phase == "training"


def test_progress_stream_completes_on_a_wedged_worker(monkeypatch):
    b = _running(monkeypatch)
    b.step_history.extend([1, 2])
    b.loss_history.extend([1.0, 0.5])
    b.lr_history.extend([1e-4, 9e-5])
    b._handle_event(dict(_DONE))
    names = asyncio.run(_sse_events())
    assert "complete" in names, names
    assert b._proc.is_alive() is True


def test_progress_stream_stays_open_while_training(monkeypatch):
    b = _running(monkeypatch)
    b.step_history.append(1)
    b.loss_history.append(1.0)
    b.lr_history.append(1e-4)

    async def _run():
        resp = await rt.stream_training_progress(_Req(), current_subject = "t")
        names: list[str] = []

        async def pump():
            async for chunk in resp.body_iterator:
                for line in str(chunk).splitlines():
                    if line.startswith("event: "):
                        names.append(line[7:].strip())

        task = asyncio.create_task(pump())
        await asyncio.sleep(1.5)
        assert "complete" not in names, names
        b._handle_event(dict(_DONE))  # run ends; worker still lingers
        for _ in range(100):
            if "complete" in names:
                break
            await asyncio.sleep(0.05)
        task.cancel()
        return names

    assert "complete" in asyncio.run(_run())


def test_late_stop_does_not_unfinish_a_completed_run(monkeypatch):
    """Stop clicked in the poll window after the run already finished.

    The button greys out only once /api/train/status reports is_training_running=False (3s
    poll), so a click can still land on a run that has saved. /stop is terminal-aware, so it
    reports idle instead of latching _should_stop and overwriting the finished banner with a
    "Stopping..." message no later path clears.
    """
    b = _running(monkeypatch)
    b._handle_event(dict(_DONE))

    resp = asyncio.run(
        rt.stop_training(
            rt.TrainingStopRequest(save = True, expected_job_id = "job_1"),
            current_subject = "t",
        )
    )
    assert resp.status == "idle"
    assert b._should_stop is False

    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.phase == "completed"
    assert st.message.startswith("Training completed!")

    # ... and it survives the watchdog reaping the wedged worker.
    b._finalize_stopped_after_escalation(target_proc = b._proc, watched_job_id = "job_1")
    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.phase == "completed"
    assert st.message.startswith("Training completed!")


def test_stop_and_save_losing_the_race_to_the_pump_keeps_the_run_completed(monkeypatch):
    """The same late Stop, except the run finishes *after* the route's terminal check.

    The pump publishes terminal state under the backend lock, so the re-test has to sit
    inside stop_training() next to the mutation it guards. Without it, /status derives
    "stopped" for a run the DB finalized as completed.
    """
    b = _running(monkeypatch)
    b._db_run_created = True
    real_stop, fired = b.stop_training, []

    def stop_after_complete(save = True, expected_job_id = None):
        if not fired:
            fired.append(True)
            b._handle_event(dict(_DONE))  # the pump wins the gap
        return real_stop(save = save, expected_job_id = expected_job_id)

    b.stop_training = stop_after_complete

    resp = asyncio.run(
        rt.stop_training(
            rt.TrainingStopRequest(save = True, expected_job_id = "job_1"),
            current_subject = "t",
        )
    )
    assert resp.status == "idle"
    assert b._should_stop is False, "a run that finished in the gap must not latch a stop"
    assert (b._terminal_finalize_payload or {}).get("status") == "completed"

    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.phase == "completed"
    assert st.message.startswith("Training completed!")


def test_stop_mid_run_still_works(monkeypatch):
    b = _running(monkeypatch)
    resp = asyncio.run(
        rt.stop_training(
            rt.TrainingStopRequest(save = True, expected_job_id = "job_1"),
            current_subject = "t",
        )
    )
    assert resp.status == "stopped"
    assert b._should_stop is True


def test_cancel_mid_run_still_works(monkeypatch):
    # save=False takes the other branch, which the guard skips: it has already mutated state.
    b = _running(monkeypatch)
    b._db_run_created = True
    monkeypatch.setattr(
        "storage.studio_db.mark_run_cancel_requested", lambda *a, **k: True, raising = False
    )
    resp = asyncio.run(
        rt.stop_training(
            rt.TrainingStopRequest(save = False, expected_job_id = "job_1"),
            current_subject = "t",
        )
    )
    assert resp.status == "stopped"
    assert b._cancel_requested is True


def test_surfaces_tolerate_a_backend_without_is_run_finished(monkeypatch):
    # A stand-in backend lacking the new method must fall back to liveness, not raise.
    class _Minimal:
        current_job_id = "job_min"
        step_history: list = []
        loss_history: list = []
        lr_history: list = []
        eval_loss_history: list = []
        eval_step_history: list = []
        eval_enabled = False
        trainer = None
        _should_stop = False

        def is_training_active(self):
            return False

    monkeypatch.setattr(rt, "get_training_backend", lambda: _Minimal())
    st = asyncio.run(rt.get_training_status(current_subject = "t"))
    assert st.is_training_running is False
    assert rt._run_finished(_Minimal()) is False
