# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "training_progress_job_scope_route",
    _BACKEND_ROOT / "routes" / "training.py",
)
rt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rt)
TrainingBackend = sys.modules["core.training.training"].TrainingBackend


class _Progress:
    def __init__(self, step = 2):
        self.step = step
        self.total_steps = 10
        self.loss = 1.0
        self.learning_rate = 0.0001
        self.epoch = 0.2
        self.grad_norm = None
        self.num_tokens = None
        self.eval_loss = None
        self.elapsed_seconds = None
        self.eta_seconds = None


class _Backend:
    def __init__(
        self,
        active,
        on_poll = None,
    ):
        self.current_job_id = "job-old"
        self._spawn_in_progress = False
        self.step_history = [2]
        self.loss_history = [1.0]
        self.lr_history = [0.0001]
        self.grad_norm_step_history = []
        self.grad_norm_history = []
        self.eval_enabled = False
        self.trainer = types.SimpleNamespace(training_progress = _Progress())
        self._active = list(active)
        self._on_poll = on_poll
        self._polls = 0

    def is_training_active(self):
        self._polls += 1
        if self._on_poll is not None:
            self._on_poll(self, self._polls)
        index = self._polls - 1
        return self._active[index] if index < len(self._active) else False


class _Request:
    def __init__(self, last_event_id = None):
        self.headers = {"last-event-id": str(last_event_id)} if last_event_id is not None else {}

    async def is_disconnected(self):
        return False


def _collect(response):
    async def drain():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)

    return asyncio.run(asyncio.wait_for(drain(), 5))


def _events(raw):
    parsed = []
    for block in raw.split("\n\n"):
        lines = block.strip().splitlines()
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if data is None:
            continue
        event = next(
            (line[7:] for line in lines if line.startswith("event: ")),
            "progress",
        )
        parsed.append((event, json.loads(data)))
    return parsed


def _stream(backend, request, expected_job_id):
    original_backend = rt.get_training_backend
    original_to_thread = rt.asyncio.to_thread

    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    rt.get_training_backend = lambda: backend
    rt.asyncio.to_thread = inline
    try:
        response = asyncio.run(
            rt.stream_training_progress(
                request,
                expected_job_id = expected_job_id,
                current_subject = "tester",
            )
        )
        return _collect(response)
    finally:
        rt.get_training_backend = original_backend
        rt.asyncio.to_thread = original_to_thread


def test_reconnect_cursor_cannot_cross_job_identity():
    backend = _Backend([True])
    backend.current_job_id = "job-new"

    raw = _stream(backend, _Request(last_event_id = 2), "job-old")

    assert _events(raw) == []


def test_active_stream_stops_when_a_new_job_takes_ownership():
    def switch_job(backend, poll):
        if poll == 2:
            backend.current_job_id = "job-new"
            backend.step_history[:] = [9]
            backend.loss_history[:] = [0.5]
            backend.lr_history[:] = [0.00005]
            backend.trainer.training_progress = _Progress(step = 9)

    backend = _Backend([True, True], switch_job)

    events = _events(_stream(backend, _Request(), "job-old"))

    assert all(payload["job_id"] == "job-old" for _, payload in events)
    assert all(payload["step"] != 9 for _, payload in events)
    assert all(event != "complete" for event, _ in events)


def test_job_replacement_during_final_probe_emits_no_completion(monkeypatch):
    async def no_sleep(_seconds):
        return None

    def switch_job(backend, poll):
        if poll == 3:
            backend.current_job_id = "job-new"

    monkeypatch.setattr(rt.asyncio, "sleep", no_sleep)
    backend = _Backend([True, True, False], switch_job)

    events = _events(_stream(backend, _Request(), "job-old"))

    assert all(event != "complete" for event, _ in events)


def test_job_replacement_during_replay_suppresses_candidate_frame():
    backend = _Backend([True])
    backend.step_history = [1, 2]
    backend.lr_history = [0.0002, 0.0001]

    class _SwitchingLosses(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            if index == 1:
                backend.current_job_id = "job-new"
            return value

    backend.loss_history = _SwitchingLosses([1.5, 1.0])

    events = _events(_stream(backend, _Request(last_event_id = 1), "job-old"))

    assert events == []


def test_same_job_completion_keeps_its_identity():
    backend = _Backend([True, False])

    events = _events(_stream(backend, _Request(), "job-old"))
    complete = [payload for event, payload in events if event == "complete"]

    assert len(complete) == 1
    assert complete[0]["job_id"] == "job-old"
    assert complete[0]["step"] == 2


def test_stalled_progress_error_does_not_emit_completion(monkeypatch):
    async def no_sleep(_seconds):
        return None

    backend = _Backend([True, True, True])
    monkeypatch.setattr(rt, "_PROGRESS_STALL_TIMEOUT_POLLS", 0)
    monkeypatch.setattr(rt.asyncio, "sleep", no_sleep)

    events = _events(_stream(backend, _Request(), "job-old"))

    assert any(event == "error" for event, _ in events)
    assert all(event != "complete" for event, _ in events)


def test_internal_progress_error_does_not_emit_completion():
    class _FailingTrainer:
        def __init__(self):
            self.reads = 0

        @property
        def training_progress(self):
            self.reads += 1
            if self.reads == 2:
                raise RuntimeError("progress read failed")
            return _Progress()

    backend = _Backend([True, True])
    backend.trainer = _FailingTrainer()

    events = _events(_stream(backend, _Request(), "job-old"))

    assert any(event == "error" for event, _ in events)
    assert all(event != "complete" for event, _ in events)


class _StatusBackend:
    def __init__(self):
        self.current_job_id = "job-old"
        self.current_start_request_id = None
        self._spawn_in_progress = False
        self._new_job_spawn_id = None
        self.eval_enabled = True
        self.step_history = [7]
        self.loss_history = [1.5]
        self.lr_history = [0.0002]
        self.grad_norm_history = [0.8]
        self.grad_norm_step_history = [7]
        self.eval_loss_history = [1.4]
        self.eval_step_history = [7]
        self._output_dir = "/old/output"
        self._should_stop = False
        self._start_request = types.SimpleNamespace(
            start_request_id = "start-new",
            job_id = "job-new",
            state = "pending",
            message = "Preparing new run",
            error = None,
        )
        self.trainer = types.SimpleNamespace(
            get_training_progress = lambda: types.SimpleNamespace(
                status_message = "Old training",
                error = None,
                warnings = ["old warning"],
                is_completed = False,
                epoch = 0.7,
                step = 7,
                total_steps = 10,
                loss = 1.5,
                learning_rate = 0.0002,
            )
        )

    def status_start_request(self):
        return self._start_request

    def get_start_request(self, _request_id):
        return self._start_request

    def is_training_active(self):
        return True


def test_pending_job_status_excludes_the_previous_owner_state(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend._spawn_in_progress = True
    backend._new_job_spawn_id = "job-new"
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-new"
    assert status.start_request_state == "pending"
    assert status.details is None
    assert status.metric_history is None
    assert status.eval_enabled is False
    assert status.warnings == []


def test_competing_pending_job_does_not_displace_the_active_owner(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-old"
    assert status.start_request_id is None
    assert status.start_request_state is None
    assert status.details["step"] == 7
    assert status.metric_history["steps"] == [7]
    assert status.eval_enabled is True
    assert status.warnings == ["old warning"]


def test_competing_rejected_job_does_not_displace_the_active_owner(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend._start_request.state = "rejected"
    backend._start_request.message = "Training already active"
    backend._start_request.error = "Training already active"
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-old"
    assert status.phase == "training"
    assert status.details["step"] == 7


def test_idle_owner_exposes_a_pending_start_without_owner_state(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend.is_training_active = lambda: False
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-new"
    assert status.start_request_state == "pending"
    assert status.phase == "configuring"
    assert status.details is None
    assert status.metric_history is None


def test_handoff_without_a_start_request_exposes_only_the_new_identity(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend._start_request = None
    backend._spawn_in_progress = True
    backend._new_job_spawn_id = "job-new"
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-new"
    assert status.start_request_id is None
    assert status.start_request_state is None
    assert status.phase == "configuring"
    assert status.details is None
    assert status.metric_history is None


def test_status_retries_when_ownership_changes_during_the_active_probe(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend._start_request = None
    polls = 0

    def switch_owner():
        nonlocal polls
        polls += 1
        if polls == 1:
            backend.current_job_id = "job-new"
            backend.step_history[:] = [1]
            backend.loss_history[:] = [0.9]
            backend.lr_history[:] = [0.0001]
            backend.trainer.get_training_progress = lambda: types.SimpleNamespace(
                status_message = "New training",
                error = None,
                warnings = [],
                is_completed = False,
                epoch = 0.1,
                step = 1,
                total_steps = 20,
                loss = 0.9,
                learning_rate = 0.0001,
            )
        return True

    backend.is_training_active = switch_owner
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert polls == 2
    assert status.job_id == "job-new"
    assert status.details["step"] == 1
    assert status.metric_history["steps"] == [1]


def test_status_retries_when_a_handoff_starts_during_the_build(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    polls = 0

    def get_progress():
        nonlocal polls
        polls += 1
        backend._spawn_in_progress = True
        backend._new_job_spawn_id = "job-new"
        return types.SimpleNamespace(
            status_message = "Old training",
            error = None,
            warnings = [],
            is_completed = False,
            epoch = 0.7,
            step = 7,
            total_steps = 10,
            loss = 1.5,
            learning_rate = 0.0002,
        )

    backend.trainer.get_training_progress = get_progress
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert polls == 1
    assert status.job_id == "job-new"
    assert status.start_request_state == "pending"
    assert status.details is None


def test_new_job_spawn_reservation_cleans_up_after_an_exception():
    backend = TrainingBackend()

    with pytest.raises(RuntimeError):
        with backend._new_job_spawn_reservation("job-new") as reserved:
            assert reserved is True
            assert backend._spawn_in_progress is True
            assert backend._new_job_spawn_id == "job-new"
            raise RuntimeError("spawn failed")

    assert backend._spawn_in_progress is False
    assert backend._new_job_spawn_id is None


def test_completed_start_cleanup_does_not_clear_a_following_xet_reservation():
    backend = TrainingBackend()

    with backend._new_job_spawn_reservation("job-new") as reserved:
        assert reserved is True
        with backend._lock:
            backend._spawn_in_progress = False
            backend._new_job_spawn_id = None
        with backend._lock:
            backend._spawn_in_progress = True

    assert backend._spawn_in_progress is True
    assert backend._new_job_spawn_id is None


def test_metrics_reject_a_job_that_does_not_own_the_backend(monkeypatch):
    backend = _StatusBackend()
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            rt.get_training_metrics(
                expected_job_id = "job-new",
                current_subject = "tester",
            )
        )

    assert exc_info.value.status_code == 409


def test_metrics_response_declares_its_owner(monkeypatch):
    backend = _StatusBackend()
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    metrics = asyncio.run(
        rt.get_training_metrics(
            expected_job_id = "job-old",
            current_subject = "tester",
        )
    )

    assert metrics.job_id == "job-old"
    assert metrics.step_history == [7]


def test_installing_job_exposes_no_previous_metrics(monkeypatch):
    backend = _StatusBackend()
    backend.current_job_id = "job-new"
    backend._start_request = None
    backend._spawn_in_progress = True
    backend._new_job_spawn_id = "job-new"
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            rt.get_training_metrics(
                expected_job_id = "job-new",
                current_subject = "tester",
            )
        )

    assert exc_info.value.status_code == 409


def test_installing_job_exposes_no_previous_status_details(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend.current_job_id = "job-new"
    backend._start_request = None
    backend._spawn_in_progress = True
    backend._new_job_spawn_id = "job-new"
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-new"
    assert status.details is None
    assert status.metric_history is None
    assert status.eval_enabled is False


def test_installing_job_cannot_open_a_progress_stream():
    backend = _Backend([True])
    backend._spawn_in_progress = True
    backend._new_job_spawn_id = "job-new"

    events = _events(_stream(backend, _Request(), "job-old"))

    assert events == []


def test_xet_respawn_preserves_the_owner_status(monkeypatch):
    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    backend = _StatusBackend()
    backend._spawn_in_progress = True
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)
    monkeypatch.setattr(rt.asyncio, "to_thread", inline)

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))

    assert status.job_id == "job-old"
    assert status.details["step"] == 7
    assert status.metric_history["steps"] == [7]
    assert status.eval_enabled is True


def test_xet_respawn_preserves_owner_metrics(monkeypatch):
    backend = _StatusBackend()
    backend._start_request = None
    backend._spawn_in_progress = True
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    metrics = asyncio.run(
        rt.get_training_metrics(
            expected_job_id = "job-old",
            current_subject = "tester",
        )
    )

    assert metrics.job_id == "job-old"
    assert metrics.step_history == [7]


def test_xet_respawn_keeps_the_owner_progress_stream_open():
    backend = _Backend([True, False])
    backend._spawn_in_progress = True

    events = _events(_stream(backend, _Request(), "job-old"))

    assert any(event == "progress" for event, _ in events)
    assert any(event == "complete" for event, _ in events)
