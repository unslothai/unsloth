# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two surfaces that hung at 100% when a finished worker would not exit (#7897).

/api/train/status and the /api/train/progress SSE both keyed off is_training_active(),
which is liveness-based, so a worker wedged in post-save teardown kept the run reported as
"training" forever. They now consult is_run_finished() as well. A live run must be
unaffected, and /api/train/reset must reap a finished-but-lingering worker instead of
refusing the user's return to configuration.
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


def test_reset_reaps_a_finished_but_lingering_worker(monkeypatch):
    b = _running(monkeypatch)
    b._handle_event(dict(_DONE))
    asyncio.run(rt.reset_training(current_subject = "t"))
    assert b._proc.is_alive() is False, "reset must reap instead of returning 409"
    assert b.is_training_active() is False


def test_reset_still_refused_mid_run(monkeypatch):
    from fastapi import HTTPException

    b = _running(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(rt.reset_training(current_subject = "t"))
    assert exc.value.status_code == 409
    assert b._proc.is_alive() is True, "a live run must never be reaped by reset"
