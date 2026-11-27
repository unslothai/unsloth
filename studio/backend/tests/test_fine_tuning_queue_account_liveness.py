# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A queued /v1/fine_tuning job outlives its request: it must not start for a disabled or
deleted account."""

import asyncio

import pytest

from auth import storage
from models.training import TrainingJobResponse, TrainingStartRequest
from routes import openai_fine_tuning as ft, training as training_routes
from studio.backend.tests.test_account_lifecycle import auth_env, headers, matrix  # noqa: F401
from utils.account_context import OWNER, current_account, run_as


def _request(model = "unsloth/Qwen3-0.6B"):
    return TrainingStartRequest.model_validate(
        {"model_name": model, "format_type": "auto", "training_type": "LoRA/QLoRA"}
    )


@pytest.fixture
def queue(matrix, monkeypatch):
    for name, value in (("_QUEUE", []), ("_ALIASES", {}), ("_STARTED", {}), ("_RUN_ALIAS", {})):
        monkeypatch.setattr(ft, name, value)
    monkeypatch.setattr(ft, "_ensure_drainer", lambda: None)
    monkeypatch.setattr(ft, "_save_alias", lambda *a: None)
    monkeypatch.setattr(ft, "_remember_start", lambda *a: None)
    monkeypatch.setattr(ft, "_training_busy", lambda: False)
    started = []

    async def start_training(request, current_subject, via_api_key):
        started.append(current_account().account_id)
        return TrainingJobResponse(job_id = f"job_{len(started)}", status = "queued", message = "queued")

    monkeypatch.setattr(training_routes, "start_training", start_training)
    client, _, _ = matrix
    return client, started


def test_disabled_account_queued_job_never_starts(queue):
    client, started = queue
    bob = storage.get_account("bob")
    entry = run_as(bob, lambda: ft._enqueue(_request(), "bob", True))
    response = client.patch(
        f"/api/accounts/{bob.account_id}", headers = headers(), json = {"is_active": False}
    )
    assert response.status_code == 200
    asyncio.run(ft._drain_once())
    assert started == []
    assert entry.status == "cancelled" and entry not in ft._QUEUE


def test_deleted_account_queued_job_never_starts(queue):
    client, started = queue
    bob = storage.get_account("bob")
    entry = run_as(bob, lambda: ft._enqueue(_request(), "bob", True))
    assert client.delete(f"/api/accounts/{bob.account_id}", headers = headers()).status_code == 204
    asyncio.run(ft._drain_once())
    assert started == []
    assert entry.status == "cancelled"


def test_disabled_member_is_dropped_from_a_batch(queue, monkeypatch):
    client, started = queue
    bob = storage.get_account("bob")
    head = ft._enqueue(_request(), "unsloth", False)
    member = run_as(bob, lambda: ft._enqueue(_request(), "bob", True))
    monkeypatch.setattr(ft, "_batch_for", lambda entry: [entry, member])
    client.patch(f"/api/accounts/{bob.account_id}", headers = headers(), json = {"is_active": False})
    asyncio.run(ft._drain_once())
    assert started == [OWNER.account_id]
    assert head.status == "started" and member.status == "cancelled"
