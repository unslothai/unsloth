# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from fastapi import HTTPException

from core.training import queue as queue_module
from core.training.queue import TrainingQueueManager
from models import TrainingStartRequest
from routes import training_queue as routes_module
from storage import studio_db


class FakeBackend:
    def __init__(self):
        self.active = False
        self.current_job_id = None
        self.on_job_finished = None

    def is_training_active(self):
        return self.active


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    backend = FakeBackend()
    manager = TrainingQueueManager(backend = backend)
    manager.settle_delay = 0
    # enqueue()/resume() revive the runner; keep these route tests synchronous
    # (runner lifecycle is covered in test_training_queue_manager.py).
    manager.start_runner = lambda: None
    monkeypatch.setattr(queue_module, "_queue_manager", manager)
    # Route-level validation: dataset/streaming checks are covered by
    # test_training_launch_extraction.py; here they pass by default.
    monkeypatch.setattr(queue_module, "validate_training_request", lambda req: None)
    yield {"backend": backend, "manager": manager}


def _request(**overrides) -> TrainingStartRequest:
    base = dict(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
        hf_token = "hf_secret_token",
        wandb_token = "wandb_secret",
    )
    base.update(overrides)
    return TrainingStartRequest(**base)


def _enqueue(**overrides):
    return asyncio.run(routes_module.enqueue_item(_request(**overrides), current_subject = "tester"))


def test_enqueue_returns_item():
    item = _enqueue()
    assert item.status == "pending"
    assert item.model_name == "unsloth/test"
    assert item.dataset_summary == "org/dataset"
    assert item.job_id is None


def test_item_exposes_project_name_but_no_secrets():
    # project_name is display-safe and needed to label queue-started runs;
    # it is the ONLY field extracted from the stored request payload.
    item = _enqueue(project_name = "my-experiment")
    assert item.project_name == "my-experiment"
    state = asyncio.run(routes_module.get_queue_state(current_subject = "tester"))
    assert state.items[0].project_name == "my-experiment"
    assert "hf_secret_token" not in state.model_dump_json()

    blank = _enqueue(project_name = "  ")
    assert blank.project_name is None


def test_enqueue_cap_409(_isolated):
    _isolated["manager"].max_pending = 2
    _enqueue()
    _enqueue()
    with pytest.raises(HTTPException) as exc_info:
        _enqueue()
    assert exc_info.value.status_code == 409
    assert "full" in exc_info.value.detail.lower()


def test_enqueue_via_api_key_blocked_during_inference(monkeypatch):
    from core.inference import llama_keepwarm

    monkeypatch.setattr(llama_keepwarm, "other_inference_request_count", lambda **kwargs: 1)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes_module.enqueue_item(_request(), current_subject = "api", via_api_key = True))
    assert exc_info.value.status_code == 409
    assert "inference" in exc_info.value.detail.lower()
    assert studio_db.count_pending_queue_items() == 0


def test_enqueue_via_api_key_allowed_when_inference_idle(monkeypatch):
    from core.inference import llama_keepwarm

    monkeypatch.setattr(llama_keepwarm, "other_inference_request_count", lambda **kwargs: 0)
    item = asyncio.run(
        routes_module.enqueue_item(_request(), current_subject = "api", via_api_key = True)
    )
    assert item.status == "pending"
    # Origin is persisted so the runner can defer the launch if an inference
    # request is in flight by the time this item reaches the head.
    assert studio_db.get_queue_item(item.id)["via_api_key"] == 1

    ui_item = _enqueue()
    assert studio_db.get_queue_item(ui_item.id)["via_api_key"] == 0


def test_enqueue_validation_400(monkeypatch):
    def _reject(request):
        raise HTTPException(status_code = 400, detail = "Local dataset not found: gone.jsonl")

    monkeypatch.setattr(queue_module, "validate_training_request", _reject)
    with pytest.raises(HTTPException) as exc_info:
        _enqueue()
    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail


def test_queue_state_lists_items_in_order():
    a = _enqueue()
    b = _enqueue(model_name = "unsloth/other")
    state = asyncio.run(routes_module.get_queue_state(current_subject = "tester"))
    assert state.pending_count == 2
    assert [i.id for i in state.items] == [a.id, b.id]
    assert state.paused is False
    assert state.active_job_id is None


def test_state_reports_active_job(_isolated):
    _isolated["backend"].active = True
    _isolated["backend"].current_job_id = "job_live"
    state = asyncio.run(routes_module.get_queue_state(current_subject = "tester"))
    assert state.active_job_id == "job_live"


def test_direct_start_rejected_while_queue_has_active_item(monkeypatch, _isolated):
    from routes import training as training_routes

    _enqueue()
    monkeypatch.setattr(training_routes, "get_training_backend", lambda: _isolated["backend"])

    response = asyncio.run(
        training_routes.start_training(_request(), current_subject = "tester", via_api_key = False)
    )

    assert response.status == "error"
    assert response.error == "Training queue active"


def test_no_secrets_in_queue_responses():
    item = _enqueue()
    state = asyncio.run(routes_module.get_queue_state(current_subject = "tester"))

    for payload in (item.model_dump_json(), state.model_dump_json()):
        assert "hf_secret_token" not in payload
        assert "wandb_secret" not in payload
        assert "request_json" not in payload

    # The secret IS still in the DB for the unattended launch (by design).
    stored = studio_db.get_queue_item(item.id)
    assert "hf_secret_token" in stored["request_json"]


def test_delete_pending_ok_and_404_unknown():
    item = _enqueue()
    result = asyncio.run(routes_module.remove_item(item.id, current_subject = "tester"))
    assert result == {"status": "ok"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes_module.remove_item(item.id, current_subject = "tester"))
    assert exc_info.value.status_code == 404


def test_delete_non_pending_409():
    item = _enqueue()
    studio_db.update_queue_item_status(item.id, "running", job_id = "job_1")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes_module.remove_item(item.id, current_subject = "tester"))
    assert exc_info.value.status_code == 409


def test_move_swaps_and_edge_409():
    from models import TrainingQueueMoveRequest

    a = _enqueue()
    b = _enqueue()
    state = asyncio.run(
        routes_module.move_item(
            b.id, TrainingQueueMoveRequest(direction = "up"), current_subject = "tester"
        )
    )
    assert [i.id for i in state.items if i.status == "pending"] == [b.id, a.id]

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes_module.move_item(
                b.id, TrainingQueueMoveRequest(direction = "up"), current_subject = "tester"
            )
        )
    assert exc_info.value.status_code == 409


def test_pause_resume_round_trip():
    state = asyncio.run(routes_module.pause_queue(current_subject = "tester"))
    assert state.paused is True
    assert state.paused_reason == "user"
    state = asyncio.run(routes_module.resume_queue(current_subject = "tester"))
    assert state.paused is False
    assert state.paused_reason is None
