# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from core.training.training import (
    _MAX_START_CANCEL_TOMBSTONES,
    TrainingBackend,
    TrainingStartCancellationCapacityError,
)
from models.training import TrainingStartRequest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_training_route(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        _BACKEND_ROOT / "routes" / "training.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


def test_start_request_reservation_is_idempotent_and_serialized():
    backend = TrainingBackend()

    outcome, pending = backend.reserve_start_request("request-1", "job-1")
    assert outcome == "reserved"
    assert pending.state == "pending"

    outcome, duplicate = backend.reserve_start_request("request-1", "job-other")
    assert outcome == "existing"
    assert duplicate.job_id == "job-1"

    outcome, conflict = backend.reserve_start_request("request-2", "job-2")
    assert outcome == "conflict"
    assert conflict.start_request_id == "request-2"
    assert conflict.state == "rejected"

    rejected = backend.resolve_start_request(
        "request-1",
        state = "rejected",
        message = "Model unavailable",
        error = "Model unavailable",
        error_code = "hf_model_metadata_unavailable",
    )
    assert rejected is not None
    assert rejected.state == "rejected"
    assert rejected.error_code == "hf_model_metadata_unavailable"
    assert backend.status_start_request() == rejected
    assert backend.acknowledge_start_request("request-1") is True
    assert backend.status_start_request() is None
    assert backend.get_start_request("request-1") == rejected

    outcome, next_request = backend.reserve_start_request("request-3", "job-3")
    assert outcome == "reserved"
    assert next_request.job_id == "job-3"


def test_accepted_start_request_remains_queryable():
    backend = TrainingBackend()
    backend.reserve_start_request("request-1", "job-1")
    backend.resolve_start_request(
        "request-1",
        state = "accepted",
        message = "Training queued",
    )

    record = backend.get_start_request("request-1")
    assert record is not None
    assert record.state == "accepted"
    assert record.job_id == "job-1"


def test_cancel_before_registration_creates_a_start_tombstone():
    backend = TrainingBackend()

    outcome, cancelled = backend.cancel_start_request("request-before-start")

    assert outcome == "cancelled"
    assert cancelled.state == "rejected"
    assert cancelled.job_id == ""
    assert cancelled.error_code == "training_start_cancelled"

    reservation, duplicate = backend.reserve_start_request(
        "request-before-start",
        "job-must-not-start",
    )
    assert reservation == "existing"
    assert duplicate == cancelled


def test_cancel_tombstone_survives_later_cancellation_records():
    backend = TrainingBackend()
    backend.cancel_start_request("request-before-start")

    for index in range(64):
        backend.cancel_start_request(f"later-cancellation-{index}")

    reservation, record = backend.reserve_start_request(
        "request-before-start",
        "job-must-not-start",
    )

    assert reservation == "existing"
    assert record.error_code == "training_start_cancelled"


def test_cancel_tombstone_capacity_fails_without_eviction(monkeypatch):
    import core.training.training as training_module

    monkeypatch.setattr(training_module, "_MAX_START_CANCEL_TOMBSTONES", 2)
    backend = TrainingBackend()
    backend.cancel_start_request("request-1")
    backend.cancel_start_request("request-2")

    with pytest.raises(TrainingStartCancellationCapacityError):
        backend.cancel_start_request("request-3")

    assert len(backend._start_cancel_tombstones) == 2
    assert backend.reserve_start_request("request-1", "job-1")[0] == "existing"
    assert backend.reserve_start_request("request-2", "job-2")[0] == "existing"


def test_expired_cancel_tombstone_releases_request_id():
    backend = TrainingBackend()
    _, record = backend.cancel_start_request("request-expired")
    backend._start_cancel_tombstones["request-expired"] = (0.0, record)

    reservation, pending = backend.reserve_start_request("request-expired", "job-new")

    assert reservation == "reserved"
    assert pending.job_id == "job-new"


def test_cancel_tombstone_ttl_refreshes_on_duplicate_cancel(monkeypatch):
    import core.training.training as training_module

    now = [0.0]
    monkeypatch.setattr(training_module.time, "monotonic", lambda: now[0])
    backend = TrainingBackend()
    backend.cancel_start_request("request-delayed")

    now[0] = 299.0
    backend.cancel_start_request("request-delayed")
    now[0] = 301.0

    assert backend.reserve_start_request("request-delayed", "job-must-not-start")[0] == "existing"


def test_cancel_tombstone_ttl_refreshes_when_start_arrives(monkeypatch):
    import core.training.training as training_module

    now = [0.0]
    monkeypatch.setattr(training_module.time, "monotonic", lambda: now[0])
    backend = TrainingBackend()
    backend.cancel_start_request("request-delayed")

    now[0] = 299.0
    assert backend.reserve_start_request("request-delayed", "job-first-retry")[0] == "existing"
    now[0] = 301.0

    assert backend.reserve_start_request("request-delayed", "job-second-retry")[0] == "existing"


def test_registered_cancel_tombstone_survives_start_request_churn():
    backend = TrainingBackend()
    backend.reserve_start_request("request-cancelled", "job-cancelled")
    backend.cancel_start_request("request-cancelled")

    for index in range(64):
        request_id = f"later-request-{index}"
        backend.reserve_start_request(request_id, f"later-job-{index}")
        backend.resolve_start_request(
            request_id,
            state = "rejected",
            message = "Rejected",
        )

    reservation, record = backend.reserve_start_request(
        "request-cancelled",
        "job-must-not-start",
    )

    assert reservation == "existing"
    assert record.error_code == "training_start_cancelled"


def test_cancel_pending_start_prevents_worker_spawn():
    backend = TrainingBackend()
    backend.reserve_start_request("request-pending", "job-pending")

    outcome, cancelled = backend.cancel_start_request("request-pending")

    assert outcome == "cancelled"
    assert cancelled.state == "rejected"
    assert cancelled.error_code == "training_start_cancelled"
    assert backend.status_start_request() is None
    assert (
        backend.start_training(
            "job-pending",
            start_request_id = "request-pending",
            model_name = "unsloth/test",
        )
        is False
    )


def test_cancel_pending_start_does_not_wait_for_lifecycle_work():
    from core.training.lifecycle import training_lifecycle_guard

    backend = TrainingBackend()
    backend.reserve_start_request("request-pending", "job-pending")
    lifecycle_entered = threading.Event()
    release_lifecycle = threading.Event()
    cancel_finished = threading.Event()
    outcome = {}

    def hold_lifecycle():
        with training_lifecycle_guard():
            lifecycle_entered.set()
            assert release_lifecycle.wait(timeout = 5)

    holder = threading.Thread(target = hold_lifecycle, daemon = True)
    holder.start()
    assert lifecycle_entered.wait(timeout = 5)

    def cancel_pending():
        outcome["result"] = backend.cancel_start_request("request-pending")
        cancel_finished.set()

    cancel = threading.Thread(target = cancel_pending, daemon = True)
    cancel.start()
    try:
        assert cancel_finished.wait(timeout = 1)
    finally:
        release_lifecycle.set()
    cancel.join(timeout = 5)
    holder.join(timeout = 5)

    assert outcome["result"][0] == "cancelled"
    assert outcome["result"][1].error_code == "training_start_cancelled"


def test_cancel_during_validation_blocks_the_final_worker_spawn(monkeypatch):
    backend = TrainingBackend()
    backend.reserve_start_request("request-validating", "job-validating")
    validation_finished = threading.Event()
    release_validation = threading.Event()
    worker_started = threading.Event()
    result = {}

    class PendingProcess:
        pid = 4321

        def start(self):
            worker_started.set()

    def before_spawn():
        validation_finished.set()
        assert release_validation.wait(timeout = 5)

    monkeypatch.setattr(
        "core.training.training.prepare_gpu_selection",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr("core.training.training._CTX.Queue", lambda: object())
    monkeypatch.setattr(
        "core.training.training._CTX.Process",
        lambda **_kwargs: PendingProcess(),
    )

    start = threading.Thread(
        target = lambda: result.update(
            started = backend.start_training(
                "job-validating",
                start_request_id = "request-validating",
                before_spawn = before_spawn,
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
            )
        ),
        daemon = True,
    )
    start.start()
    assert validation_finished.wait(timeout = 5)

    outcome, cancelled = backend.cancel_start_request("request-validating")
    release_validation.set()
    start.join(timeout = 5)

    assert outcome == "cancelled"
    assert cancelled.error_code == "training_start_cancelled"
    assert result["started"] is False
    assert worker_started.is_set() is False


def test_cancel_racing_proc_start_uses_the_committed_job_scope():
    backend = TrainingBackend()
    backend.reserve_start_request("request-spawning", "job-spawning")
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    cancel_finished = threading.Event()
    calls = []
    result = {}

    class CommittedProcess:
        pid = 4321

        def __init__(self):
            self.alive = False

        def start(self):
            spawn_entered.set()
            assert release_spawn.wait(timeout = 5)
            self.alive = True

        def is_alive(self):
            return self.alive

    class PendingPump:
        def start(self):
            return None

        def is_alive(self):
            return False

    process = CommittedProcess()

    def run_start():
        result["start"] = backend.start_training(
            "job-spawning",
            start_request_id = "request-spawning",
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
        )

    def run_cancel():
        result["cancel"] = backend.cancel_start_request("request-spawning")
        cancel_finished.set()

    start = threading.Thread(target = run_start, daemon = True)
    cancel = threading.Thread(target = run_cancel, daemon = True)

    with (
        patch(
            "core.training.training.prepare_gpu_selection",
            lambda *_args, **_kwargs: (None, None),
        ),
        patch("core.training.training._CTX.Queue", lambda: object()),
        patch("core.training.training._CTX.Process", lambda **_kwargs: process),
        patch("core.training.training.threading.Thread", lambda **_kwargs: PendingPump()),
        patch("utils.process_lifetime.adopt_pid", lambda _pid: None),
        patch.object(backend, "_ensure_db_run_created", lambda: None),
        patch.object(
            backend,
            "_stop_training_with_lifecycle_reserved",
            lambda **kwargs: calls.append(("stop", kwargs)) or True,
        ),
        patch.object(
            backend,
            "reset_training_state",
            lambda expected_job_id = None: calls.append(("reset", expected_job_id)) or "reset",
        ),
    ):
        start.start()
        assert spawn_entered.wait(timeout = 5)
        cancel.start()
        assert cancel_finished.wait(timeout = 0.1) is False
        release_spawn.set()
        start.join(timeout = 5)
        cancel.join(timeout = 5)

    assert result["start"] is True
    assert result["cancel"][0] == "cancelled"
    assert calls == [
        ("stop", {"save": False, "expected_job_id": "job-spawning"}),
        ("reset", "job-spawning"),
    ]


def test_cancel_accepted_start_stops_and_resets_only_its_job(monkeypatch):
    from core.training.lifecycle import training_lifecycle_guard

    backend = TrainingBackend()
    backend.reserve_start_request("request-current", "job-current")
    backend.resolve_start_request(
        "request-current",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-current"
    backend.current_job_id = "job-current"
    backend._progress.is_training = True
    calls = []

    monkeypatch.setattr(
        backend,
        "_stop_training_with_lifecycle_reserved",
        lambda **kwargs: calls.append(("stop", kwargs)) or True,
    )

    def reset_training_state(expected_job_id = None):
        calls.append(("reset", expected_job_id))
        lifecycle_available = threading.Event()

        def acquire_lifecycle():
            with training_lifecycle_guard():
                lifecycle_available.set()

        probe = threading.Thread(target = acquire_lifecycle, daemon = True)
        probe.start()
        assert lifecycle_available.wait(timeout = 1)
        probe.join(timeout = 5)
        return "reset"

    monkeypatch.setattr(backend, "reset_training_state", reset_training_state)

    outcome, cancelled = backend.cancel_start_request("request-current")

    assert outcome == "cancelled"
    assert cancelled.state == "rejected"
    assert cancelled.error_code == "training_start_cancelled"
    assert calls == [
        ("stop", {"save": False, "expected_job_id": "job-current"}),
        ("reset", "job-current"),
    ]
    assert backend.current_start_request_id is None


@pytest.mark.parametrize("failure_stage", ["stop", "reset"])
def test_cancel_accepted_start_releases_tombstone_capacity_after_failure(
    monkeypatch, failure_stage
):
    import core.training.training as training_module

    monkeypatch.setattr(training_module, "_MAX_START_CANCEL_TOMBSTONES", 1)
    backend = TrainingBackend()
    backend.reserve_start_request("request-current", "job-current")
    backend.resolve_start_request(
        "request-current",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-current"
    backend.current_job_id = "job-current"
    backend._progress.is_training = True

    def stop_training(**_kwargs):
        if failure_stage == "stop":
            raise RuntimeError("stop failed")
        return True

    def reset_training_state(**_kwargs):
        if failure_stage == "reset":
            raise RuntimeError("reset failed")
        return "reset"

    monkeypatch.setattr(backend, "_stop_training_with_lifecycle_reserved", stop_training)
    monkeypatch.setattr(backend, "reset_training_state", reset_training_state)

    with pytest.raises(RuntimeError, match = f"{failure_stage} failed"):
        backend.cancel_start_request("request-current")

    assert backend._start_cancel_tombstone_reservations == {}
    assert backend.cancel_start_request("request-after-failure")[0] == "cancelled"


def test_concurrent_duplicate_cancel_returns_the_cancelled_tombstone(monkeypatch):
    backend = TrainingBackend()
    backend.reserve_start_request("request-current", "job-current")
    backend.resolve_start_request(
        "request-current",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-current"
    backend.current_job_id = "job-current"
    backend._progress.is_training = True
    stop_entered = threading.Event()
    release_stop = threading.Event()
    results = []

    def stop_training(**_kwargs):
        stop_entered.set()
        assert release_stop.wait(timeout = 5)
        return True

    monkeypatch.setattr(backend, "_stop_training_with_lifecycle_reserved", stop_training)
    monkeypatch.setattr(backend, "reset_training_state", lambda **_kwargs: "reset")

    first = threading.Thread(
        target = lambda: results.append(backend.cancel_start_request("request-current")),
        daemon = True,
    )
    second = threading.Thread(
        target = lambda: results.append(backend.cancel_start_request("request-current")),
        daemon = True,
    )
    first.start()
    assert stop_entered.wait(timeout = 5)
    second.start()
    release_stop.set()
    first.join(timeout = 5)
    second.join(timeout = 5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert [outcome for outcome, _record in results] == ["cancelled", "cancelled"]
    assert all(record.error_code == "training_start_cancelled" for _outcome, record in results)
    assert backend._start_cancel_tombstone_reservations == {}


def test_duplicate_cancel_holds_capacity_when_the_first_cancel_fails(monkeypatch):
    import core.training.training as training_module

    monkeypatch.setattr(training_module, "_MAX_START_CANCEL_TOMBSTONES", 1)
    backend = TrainingBackend()
    backend.reserve_start_request("request-current", "job-current")
    backend.resolve_start_request(
        "request-current",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-current"
    backend.current_job_id = "job-current"
    backend._progress.is_training = True
    reset_lock = threading.Lock()
    reset_calls = 0
    first_reset_entered = threading.Event()
    second_reset_entered = threading.Event()
    first_cancel_finished = threading.Event()
    allow_second_reset = threading.Event()
    outcomes = {}

    def reset_training_state(**_kwargs):
        nonlocal reset_calls
        with reset_lock:
            reset_calls += 1
            call_number = reset_calls
        if call_number == 1:
            first_reset_entered.set()
            assert second_reset_entered.wait(timeout = 5)
            raise RuntimeError("first reset failed")
        second_reset_entered.set()
        assert allow_second_reset.wait(timeout = 5)
        return "reset"

    def cancel_first():
        try:
            backend.cancel_start_request("request-current")
        except RuntimeError as error:
            outcomes["first_error"] = str(error)
        finally:
            first_cancel_finished.set()

    def cancel_second():
        outcomes["second"] = backend.cancel_start_request("request-current")

    monkeypatch.setattr(
        backend,
        "_stop_training_with_lifecycle_reserved",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(backend, "reset_training_state", reset_training_state)
    first = threading.Thread(target = cancel_first, daemon = True)
    second = threading.Thread(target = cancel_second, daemon = True)
    first.start()
    assert first_reset_entered.wait(timeout = 5)
    second.start()
    assert second_reset_entered.wait(timeout = 5)
    assert first_cancel_finished.wait(timeout = 5)

    with pytest.raises(TrainingStartCancellationCapacityError):
        backend.cancel_start_request("request-filler")

    allow_second_reset.set()
    first.join(timeout = 5)
    second.join(timeout = 5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert outcomes["first_error"] == "first reset failed"
    assert outcomes["second"][0] == "cancelled"
    assert len(backend._start_cancel_tombstones) == 1
    assert backend._start_cancel_tombstone_reservations == {}


def test_cancel_rejected_start_still_stops_its_owned_worker(monkeypatch):
    backend = TrainingBackend()
    backend.reserve_start_request("request-owned", "job-owned")
    backend.resolve_start_request(
        "request-owned",
        state = "rejected",
        message = "Start finalization failed",
        error = "Start finalization failed",
    )
    backend.current_start_request_id = "request-owned"
    backend.current_job_id = "job-owned"
    backend._progress.is_training = True
    calls = []

    monkeypatch.setattr(
        backend,
        "_stop_training_with_lifecycle_reserved",
        lambda **kwargs: calls.append(("stop", kwargs)) or True,
    )
    monkeypatch.setattr(
        backend,
        "reset_training_state",
        lambda expected_job_id = None: calls.append(("reset", expected_job_id)) or "reset",
    )

    outcome, record = backend.cancel_start_request("request-owned")

    assert outcome == "cancelled"
    assert record.error_code == "training_start_cancelled"
    assert calls == [
        ("stop", {"save": False, "expected_job_id": "job-owned"}),
        ("reset", "job-owned"),
    ]


def test_adopt_failure_terminates_the_spawned_worker(monkeypatch):
    backend = TrainingBackend()
    backend.reserve_start_request("request-adopt", "job-adopt")
    backend.current_job_id = "previous-job"
    backend.current_start_request_id = "previous-request"

    class UnadoptedProcess:
        pid = 4321

        def __init__(self):
            self.alive = False
            self.terminated = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout = None):
            return None

        def kill(self):
            self.alive = False

    process = UnadoptedProcess()

    def fail_adoption(_pid):
        raise RuntimeError("adoption failed")

    monkeypatch.setattr(
        "core.training.training.prepare_gpu_selection",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr("core.training.training._CTX.Queue", lambda: object())
    monkeypatch.setattr(
        "core.training.training._CTX.Process",
        lambda **_kwargs: process,
    )
    monkeypatch.setattr(
        "utils.process_lifetime.adopt_pid",
        fail_adoption,
    )

    started = backend.start_training(
        "job-adopt",
        start_request_id = "request-adopt",
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
    )

    record = backend.get_start_request("request-adopt")
    assert started is False
    assert process.terminated is True
    assert backend.current_job_id == "previous-job"
    assert backend.current_start_request_id == "previous-request"
    assert record is not None
    assert record.state == "rejected"
    assert record.error == "Failed to adopt training subprocess"


def test_cancel_accepted_start_never_targets_a_newer_job(monkeypatch):
    backend = TrainingBackend()
    backend.reserve_start_request("request-old", "job-old")
    backend.resolve_start_request(
        "request-old",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-new"
    backend.current_job_id = "job-new"
    calls = []
    monkeypatch.setattr(
        backend,
        "_stop_training_with_lifecycle_reserved",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    outcome, record = backend.cancel_start_request("request-old")

    assert outcome == "superseded"
    assert record.state == "accepted"
    assert calls == []


def test_start_training_reserves_early_and_rejects_an_overlapping_start(monkeypatch):
    backend = TrainingBackend()
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = []
    outcome = {}

    def blocking_start(job_id, **kwargs):
        calls.append((job_id, kwargs.get("start_request_id")))
        assert backend.is_training_active() is True
        assert backend._spawn_in_progress is True
        assert backend._new_job_spawn_id == job_id
        first_entered.set()
        assert release_first.wait(timeout = 5)
        return False

    monkeypatch.setattr(backend, "_start_training_with_lifecycle_reserved", blocking_start)

    first = threading.Thread(
        target = lambda: outcome.update(
            first = backend.start_training(
                "job-1",
                start_request_id = "request-1",
                model_name = "unsloth/test",
            )
        ),
        daemon = True,
    )
    first.start()
    assert first_entered.wait(timeout = 5)

    assert (
        backend.start_training(
            "job-2",
            start_request_id = "request-2",
            model_name = "unsloth/test",
        )
        is False
    )
    assert calls == [("job-1", "request-1")]

    release_first.set()
    first.join(timeout = 5)
    assert not first.is_alive()
    assert outcome["first"] is False
    assert backend._spawn_in_progress is False
    assert backend._new_job_spawn_id is None


def test_start_training_cleans_early_reservation_after_validation_error(monkeypatch):
    backend = TrainingBackend()

    def fail_start(*_args, **_kwargs):
        assert backend.is_training_active() is True
        raise RuntimeError("validation failed")

    monkeypatch.setattr(backend, "_start_training_with_lifecycle_reserved", fail_start)

    with pytest.raises(RuntimeError, match = "validation failed"):
        backend.start_training("job-1", model_name = "unsloth/test")

    assert backend._spawn_in_progress is False
    assert backend._new_job_spawn_id is None


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        ("pending", "pending"),
        ("accepted", "queued"),
        ("rejected", "error"),
    ],
)
def test_duplicate_start_response_preserves_reservation_state(state, expected_status):
    route = _load_training_route(f"training_route_duplicate_{state}_test")
    response = route._start_request_response(
        SimpleNamespace(
            job_id = "job-1",
            state = state,
            message = "status message",
            error = "rejected" if state == "rejected" else None,
            error_code = None,
        )
    )

    assert response.status == expected_status


def test_cancel_start_route_returns_the_scoped_rejection():
    route = _load_training_route("training_route_cancel_start_request_test")
    calls = []
    record = SimpleNamespace(
        start_request_id = "request-cancel",
        job_id = "job-cancel",
        state = "rejected",
        message = "Training start was cancelled",
        error = "Training start was cancelled",
        error_code = "training_start_cancelled",
    )
    backend = SimpleNamespace(
        cancel_start_request = lambda start_request_id: (
            calls.append(start_request_id) or ("cancelled", record)
        )
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        response = asyncio.run(
            route.cancel_training_start_request(
                "request-cancel",
                current_subject = "test-user",
            )
        )

    assert calls == ["request-cancel"]
    assert response.start_request_id == "request-cancel"
    assert response.job_id == "job-cancel"
    assert response.state == "rejected"
    assert response.error_code == "training_start_cancelled"


def test_cancel_start_route_reports_tombstone_capacity():
    route = _load_training_route("training_route_cancel_capacity_test")

    def reject_cancel(_start_request_id):
        raise TrainingStartCancellationCapacityError(
            "Too many training start cancellations are pending"
        )

    backend = SimpleNamespace(cancel_start_request = reject_cancel)
    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        pytest.raises(route.HTTPException) as exc_info,
    ):
        asyncio.run(
            route.cancel_training_start_request(
                "request-cancel",
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 429


def test_cancelled_route_during_spawn_keeps_the_worker_result_authoritative():
    route = _load_training_route("training_route_cancelled_start_test")
    backend = TrainingBackend()
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    spawn_finished = threading.Event()

    def start_training(**kwargs):
        spawn_entered.set()
        if not release_spawn.wait(timeout = 5):
            raise TimeoutError("test did not release the training spawn")
        backend.current_job_id = kwargs["job_id"]
        backend.current_start_request_id = kwargs["start_request_id"]
        backend._progress.is_training = True
        backend._progress.status_message = "Initializing training..."
        spawn_finished.set()
        return True

    backend.start_training = start_training
    backend.is_training_active = lambda: backend.current_job_id is not None
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        start_request_id = "cancelled-start-request",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        max_steps = 10,
    )

    async def controlled_to_thread(function, *args, **kwargs):
        if function.__name__ != "_run_backend_start":
            return function(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = loop.create_future()

        def run():
            try:
                value = function(*args, **kwargs)
            except BaseException as exc:
                loop.call_soon_threadsafe(result.set_exception, exc)
            else:
                loop.call_soon_threadsafe(result.set_result, value)

        threading.Thread(target = run, daemon = True).start()
        return await result

    async def wait_for_event(event: threading.Event) -> bool:
        for _ in range(500):
            if event.is_set():
                return True
            await asyncio.sleep(0.01)
        return False

    async def run_cancellation_race():
        handler = asyncio.create_task(
            route.start_training(request, current_subject = "test-user"),
        )
        assert await wait_for_event(spawn_entered)
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler

        pending = backend.get_start_request("cancelled-start-request")
        assert pending is not None
        assert pending.state == "pending"

        release_spawn.set()
        assert await wait_for_event(spawn_finished)
        for _ in range(100):
            record = backend.get_start_request("cancelled-start-request")
            if record is not None and record.state != "pending":
                break
            await asyncio.sleep(0.01)
        current_task = asyncio.current_task()
        background_tasks = [task for task in asyncio.all_tasks() if task is not current_task]
        if background_tasks:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions = True),
                timeout = 5,
            )
        return record

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "_hub_unreachable", return_value = False),
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(route, "_preflight_hf_dataset_request", new = lambda request: None),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", new = controlled_to_thread),
    ):
        try:
            record = asyncio.run(run_cancellation_race())
        finally:
            release_spawn.set()

    assert record is not None
    assert record.state == "accepted"
    assert backend.current_job_id == record.job_id
    assert backend.is_training_active() is True


def test_owner_can_cancel_its_active_start_when_tombstone_capacity_is_full():
    """Exhausted unknown-request tombstone capacity must not make the currently active
    start uncancellable: the UI's Stop button routes to /start-requests/{id}/cancel while
    a start request owns the run, so a 429 there would leave the worker running."""
    backend = TrainingBackend()
    backend.reserve_start_request("request-owned", "job-owned")
    backend.resolve_start_request(
        "request-owned",
        state = "accepted",
        message = "Training queued",
    )
    backend.current_start_request_id = "request-owned"
    backend.current_job_id = "job-owned"
    backend._progress.is_training = True

    calls = []
    backend._stop_training_with_lifecycle_reserved = (
        lambda **kwargs: calls.append(("stop", kwargs)) or True
    )
    backend.reset_training_state = (
        lambda expected_job_id = None: calls.append(("reset", expected_job_id)) or "reset"
    )

    for index in range(_MAX_START_CANCEL_TOMBSTONES):
        backend.cancel_start_request(f"unknown-{index}")
    assert len(backend._start_cancel_tombstones) == _MAX_START_CANCEL_TOMBSTONES

    # Unregistered ids still hit the hard cap ...
    with pytest.raises(TrainingStartCancellationCapacityError):
        backend.cancel_start_request("unknown-overflow")

    # ... but the owner of the active start reclaims a slot and really stops the run.
    outcome, record = backend.cancel_start_request("request-owned")

    assert outcome == "cancelled"
    assert record.error_code == "training_start_cancelled"
    assert calls == [
        ("stop", {"save": False, "expected_job_id": "job-owned"}),
        ("reset", "job-owned"),
    ]
    assert backend.current_start_request_id is None
    # The owner overshoots the cap by its own slot rather than evicting a live cancellation.
    assert len(backend._start_cancel_tombstones) <= _MAX_START_CANCEL_TOMBSTONES + 1
    assert backend._start_cancel_tombstone_reservations == {}


def test_owner_cancel_at_capacity_keeps_other_live_cancellations():
    """Reclaiming a slot by evicting the soonest-expiring tombstone would forget a live
    cancellation, so a delayed /start for that id could spawn the job we just cancelled.
    The owner overshoots the cap instead; there is only ever one active start."""
    backend = TrainingBackend()
    backend.cancel_start_request("victim-race")  # cancel-before-start race
    backend.reserve_start_request("owner", "job-owner")
    backend.resolve_start_request("owner", state = "accepted", message = "Training queued")
    backend.current_start_request_id = "owner"
    backend.current_job_id = "job-owner"
    backend._progress.is_training = True
    backend._stop_training_with_lifecycle_reserved = lambda **kwargs: True
    backend.reset_training_state = lambda expected_job_id = None: "reset"

    while len(backend._start_cancel_tombstones) < _MAX_START_CANCEL_TOMBSTONES:
        backend.cancel_start_request(f"unknown-{len(backend._start_cancel_tombstones)}")

    outcome, _ = backend.cancel_start_request("owner")

    assert outcome == "cancelled"
    assert "victim-race" in backend._start_cancel_tombstones
    # The delayed start for the cancelled id must not spawn.
    assert backend.reserve_start_request("victim-race", "job-victim")[0] == "existing"
    # Unregistered ids keep hitting the hard cap.
    with pytest.raises(TrainingStartCancellationCapacityError):
        backend.cancel_start_request("stranger")


def test_pending_cancels_cannot_grow_the_tombstone_table_past_the_cap():
    """Only the owner of the active run gets the over-cap slot. A pending request is not it,
    so start-then-cancel cannot be repeated to grow the table without bound."""
    backend = TrainingBackend()
    for index in range(_MAX_START_CANCEL_TOMBSTONES):
        backend.cancel_start_request(f"unknown-{index}")

    refused = 0
    for index in range(50):
        backend.reserve_start_request(f"pending-{index}", f"job-{index}")
        try:
            backend.cancel_start_request(f"pending-{index}")
        except TrainingStartCancellationCapacityError:
            refused += 1

    assert refused >= 1
    assert len(backend._start_cancel_tombstones) <= _MAX_START_CANCEL_TOMBSTONES
