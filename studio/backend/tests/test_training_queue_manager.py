# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from fastapi import HTTPException

from core.training import queue as queue_module
from core.training.queue import TrainingQueueManager
from models import TrainingStartRequest
from storage import studio_db


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


class FakeBackend:
    def __init__(self):
        self.active = False
        self.current_job_id = None
        self.on_job_finished = None

    def is_training_active(self):
        return self.active


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def manager(backend):
    m = TrainingQueueManager(backend = backend)
    m.settle_delay = 0  # no 3s sleeps in tests
    return m


def _request(**overrides) -> TrainingStartRequest:
    base = dict(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
        hf_token = "hf_secret",
    )
    base.update(overrides)
    return TrainingStartRequest(**base)


def _enqueue(manager, monkeypatch, **overrides) -> dict:
    # Bypass validate_training_request in enqueue (network/dataset-free tests);
    # launch-time validation is patched per-test.
    monkeypatch.setattr(queue_module, "validate_training_request", lambda req: None)
    return manager.enqueue(_request(**overrides), subject = "tester")


def _patch_launch(
    monkeypatch,
    result = True,
    error = None,
    record = None,
):
    def _launch(
        job_id,
        request,
        resume_output_dir,
        subject,
        backend = None,
        **kwargs,
    ):
        if record is not None:
            record.append({"job_id": job_id, "request": request, "subject": subject})
        if error is not None:
            raise error
        if result and backend is not None:
            backend.active = True
            backend.current_job_id = job_id
        return result

    monkeypatch.setattr(queue_module, "launch_training", _launch)


# -- scheduling ---------------------------------------------------------------


def test_launches_next_item_when_idle(manager, backend, monkeypatch):
    record = []
    _patch_launch(monkeypatch, record = record)
    item = _enqueue(manager, monkeypatch)

    manager._tick()

    assert len(record) == 1
    assert record[0]["subject"] == "tester"
    stored = studio_db.get_queue_item(item["id"])
    assert stored["status"] == "running"
    assert stored["job_id"] == record[0]["job_id"]
    assert backend.current_job_id == stored["job_id"]


def test_noop_while_backend_active(manager, backend, monkeypatch):
    record = []
    _patch_launch(monkeypatch, record = record)
    item = _enqueue(manager, monkeypatch)
    backend.active = True
    backend.current_job_id = "job_manual"

    manager._tick()

    assert record == []
    assert studio_db.get_queue_item(item["id"])["status"] == "pending"


def test_noop_while_paused(manager, backend, monkeypatch):
    record = []
    _patch_launch(monkeypatch, record = record)
    item = _enqueue(manager, monkeypatch)
    manager.pause("user")

    manager._tick()

    assert record == []
    assert studio_db.get_queue_item(item["id"])["status"] == "pending"

    manager.resume()
    manager._tick()
    assert len(record) == 1


def test_pause_during_settle_delay_prevents_launch(manager, backend, monkeypatch):
    record = []
    _patch_launch(monkeypatch, record = record)
    item = _enqueue(manager, monkeypatch)
    manager.settle_delay = 3.0

    def _pause_while_settling(timeout):
        manager.pause("user")
        return False  # not stopping, just done waiting

    monkeypatch.setattr(manager._stop_runner, "wait", _pause_while_settling)

    manager._tick()

    assert record == []
    assert studio_db.get_queue_item(item["id"])["status"] == "pending"


def test_launch_validation_failure_skips_and_advances(manager, backend, monkeypatch):
    item1 = _enqueue(manager, monkeypatch)
    item2 = _enqueue(manager, monkeypatch)

    calls = []

    def _validate(request):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPException(status_code = 400, detail = "Local dataset not found: gone.jsonl")
        return None

    monkeypatch.setattr(queue_module, "validate_training_request", _validate)
    record = []
    _patch_launch(monkeypatch, record = record)

    manager._tick()  # item1 -> skipped, wake set
    stored1 = studio_db.get_queue_item(item1["id"])
    assert stored1["status"] == "skipped"
    assert "not found" in stored1["error_message"]
    # Terminal items hold no secrets.
    assert "hf_secret" not in stored1["request_json"]
    assert manager._wake.is_set()

    backend.active = False
    manager._wake.clear()
    manager._tick()  # item2 launches
    assert studio_db.get_queue_item(item2["id"])["status"] == "running"
    assert len(record) == 1


def test_launch_false_reverts_to_pending_when_backend_busy(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)

    def _launch(
        job_id,
        request,
        resume_output_dir,
        subject,
        backend = None,
        **kwargs,
    ):
        # A manual /start won the race while we prepared.
        backend.active = True
        backend.current_job_id = "job_manual"
        return False

    monkeypatch.setattr(queue_module, "launch_training", _launch)
    monkeypatch.setattr(queue_module, "validate_training_request", lambda req: None)

    manager._tick()

    assert studio_db.get_queue_item(item["id"])["status"] == "pending"


def test_launch_false_skips_when_backend_inactive(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)
    _patch_launch(monkeypatch, result = False)

    manager._tick()

    stored = studio_db.get_queue_item(item["id"])
    assert stored["status"] == "skipped"
    assert "failed to start" in stored["error_message"].lower()


def test_launch_exception_skips(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)
    _patch_launch(monkeypatch, error = ValueError("Invalid gpu_ids [99]"))

    manager._tick()

    stored = studio_db.get_queue_item(item["id"])
    assert stored["status"] == "skipped"
    assert "gpu_ids" in stored["error_message"]


def test_removed_item_not_launched(manager, backend, monkeypatch):
    record = []
    _patch_launch(monkeypatch, record = record)
    item = _enqueue(manager, monkeypatch)
    studio_db.delete_queue_item_if_pending(item["id"])

    manager._tick()

    assert record == []


# -- reconcile ------------------------------------------------------------------


def _seed_run(
    job_id: str,
    status: str,
    error_message = None,
):
    studio_db.create_run(
        id = job_id,
        model_name = "unsloth/test",
        dataset_name = "org/dataset",
        config_json = "{}",
        started_at = "2026-07-05T00:00:00+00:00",
        total_steps = 10,
    )
    if status != "running":
        studio_db.finish_run(
            id = job_id,
            status = status,
            ended_at = "2026-07-05T01:00:00+00:00",
            final_step = 10,
            final_loss = 1.0,
            duration_seconds = 3600.0,
            error_message = error_message,
        )


@pytest.mark.parametrize(
    "run_status, run_error",
    [("completed", None), ("error", "CUDA out of memory"), ("stopped", None)],
)
def test_reconcile_running_item_copies_run_result(
    manager, backend, monkeypatch, run_status, run_error
):
    item = _enqueue(manager, monkeypatch)
    _seed_run("job_1", run_status, run_error)
    studio_db.update_queue_item_status(
        item["id"], "running", job_id = "job_1", started_at = "2026-07-05T00:00:00+00:00"
    )
    backend.active = False
    backend.current_job_id = "job_1"

    manager._reconcile_running_items(backend)

    stored = studio_db.get_queue_item(item["id"])
    assert stored["status"] == "done"
    assert stored["result_status"] == run_status
    assert stored["error_message"] == run_error
    assert "hf_secret" not in stored["request_json"]


def test_reconcile_leaves_active_run_alone(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)
    studio_db.update_queue_item_status(item["id"], "running", job_id = "job_1")
    backend.active = True
    backend.current_job_id = "job_1"

    manager._reconcile_running_items(backend)

    assert studio_db.get_queue_item(item["id"])["status"] == "running"


def test_reconcile_waits_for_unfinalized_run_row(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)
    _seed_run("job_1", "running")  # backend inactive but pump hasn't finalized
    studio_db.update_queue_item_status(item["id"], "running", job_id = "job_1")
    backend.active = False

    manager._reconcile_running_items(backend)

    assert studio_db.get_queue_item(item["id"])["status"] == "running"


def test_full_cycle_error_run_advances_queue(manager, backend, monkeypatch):
    # Skip-on-failure: item1's run errors, item2 still launches.
    record = []
    _patch_launch(monkeypatch, record = record)
    item1 = _enqueue(manager, monkeypatch)
    item2 = _enqueue(manager, monkeypatch)

    manager._tick()  # launches item1
    job_1 = studio_db.get_queue_item(item1["id"])["job_id"]

    # The run dies overnight: backend finalizes it as error and goes idle.
    _seed_run(job_1, "error", "NaN loss detected")
    backend.active = False
    backend.current_job_id = job_1

    manager._tick()  # reconciles item1 -> done(error), launches item2

    stored1 = studio_db.get_queue_item(item1["id"])
    assert stored1["status"] == "done"
    assert stored1["result_status"] == "error"
    assert studio_db.get_queue_item(item2["id"])["status"] == "running"
    assert len(record) == 2


# -- wake / notify ----------------------------------------------------------------


def test_start_runner_registers_wake_callback(manager, backend):
    manager.start_runner()
    try:
        assert backend.on_job_finished is not None
        manager._wake.clear()
        backend.on_job_finished()
        assert manager._wake.is_set()
    finally:
        manager.stop_runner()


def test_enqueue_sets_wake(manager, backend, monkeypatch):
    manager._wake.clear()
    _enqueue(manager, monkeypatch)
    assert manager._wake.is_set()


# -- enqueue guards -----------------------------------------------------------------


def test_enqueue_enforces_cap(manager, backend, monkeypatch):
    monkeypatch.setattr(queue_module, "validate_training_request", lambda req: None)
    manager.max_pending = 2
    manager.enqueue(_request(), subject = "t")
    manager.enqueue(_request(), subject = "t")
    with pytest.raises(HTTPException) as exc_info:
        manager.enqueue(_request(), subject = "t")
    assert exc_info.value.status_code == 409
    assert "full" in exc_info.value.detail.lower()


def test_enqueue_persists_pre_validation_request(manager, backend, monkeypatch):
    # validate_training_request rewrites resume_from_checkpoint (run dir ->
    # concrete checkpoint path); the stored payload must keep the original so
    # launch-time validation can resolve the resumable run again.
    def _validate(request):
        request.resume_from_checkpoint = "/outputs/run_x/checkpoint-500"
        return "/outputs/run_x"

    monkeypatch.setattr(queue_module, "validate_training_request", _validate)
    item = manager.enqueue(_request(resume_from_checkpoint = "/outputs/run_x"), subject = "t")
    stored = studio_db.get_queue_item(item["id"])
    data = json.loads(stored["request_json"])
    assert data["resume_from_checkpoint"] == "/outputs/run_x"


def test_enqueue_validation_failure_propagates(manager, backend, monkeypatch):
    def _validate(request):
        raise HTTPException(status_code = 400, detail = "Local dataset not found")

    monkeypatch.setattr(queue_module, "validate_training_request", _validate)
    with pytest.raises(HTTPException) as exc_info:
        manager.enqueue(_request(), subject = "t")
    assert exc_info.value.status_code == 400
    assert studio_db.count_pending_queue_items() == 0


# -- restore on startup ---------------------------------------------------------------


def test_restore_marks_orphans_and_pauses(manager, backend, monkeypatch):
    item1 = _enqueue(manager, monkeypatch)
    item2 = _enqueue(manager, monkeypatch)
    item3 = _enqueue(manager, monkeypatch)
    studio_db.update_queue_item_status(item1["id"], "running", job_id = "job_1")
    studio_db.update_queue_item_status(item2["id"], "starting")

    manager.restore_on_startup()
    try:
        stored1 = studio_db.get_queue_item(item1["id"])
        stored2 = studio_db.get_queue_item(item2["id"])
        assert stored1["status"] == "skipped"
        assert stored2["status"] == "skipped"
        assert "restarted" in stored1["error_message"]
        # Orphaning is a terminal transition: no secrets left behind.
        assert "hf_secret" not in stored1["request_json"]
        assert "hf_secret" not in stored2["request_json"]
        assert studio_db.get_queue_item(item3["id"])["status"] == "pending"
        assert studio_db.get_queue_paused() == (True, "restart")
    finally:
        manager.stop_runner()


def test_restore_clears_stale_pause_without_pending(manager, backend):
    studio_db.set_queue_paused(True, "restart")

    manager.restore_on_startup()
    try:
        assert studio_db.get_queue_paused() == (False, None)
    finally:
        manager.stop_runner()


# -- state ------------------------------------------------------------------------


def test_state_shape(manager, backend, monkeypatch):
    item = _enqueue(manager, monkeypatch)
    backend.active = True
    backend.current_job_id = "job_live"

    state = manager.state()

    assert state["paused"] is False
    assert state["pending_count"] == 1
    assert state["max_pending"] == manager.max_pending
    assert state["active_job_id"] == "job_live"
    assert [i["id"] for i in state["items"]] == [item["id"]]
