# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image generation belongs to the account that started it, not to the model's loader."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import inference
from state import active_generations
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[allow_ambient_hf_token] = lambda: False
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(inference.router, prefix = "/v1")
    return TestClient(app)


@pytest.fixture
def shared_resident(monkeypatch):
    from core.inference import diffusion_engine_router

    running = threading.Event()
    cancelled = threading.Event()

    def generate(**kwargs):
        running.set()
        cancelled.wait(20)
        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    backend = SimpleNamespace(
        is_loaded = True,
        status = lambda: {
            "loaded": True,
            "repo_id": "org/public-model",
            "family": "z-image",
            "base_repo": None,
        },
        generate = generate,
        generate_progress = lambda: {"active": True, "step": 3, "total": 10},
        cancel_generate = lambda **kwargs: (cancelled.set(), True)[1],
    )
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", BOB.account_id)
    return SimpleNamespace(running = running, cancelled = cancelled)


def _start_alice_generation(shared_resident):
    result = {}

    def run():
        with client_for(ALICE) as client:
            result["response"] = client.post(
                "/api/inference/images/generate", json = {"prompt": "a sloth"}
            )

    thread = threading.Thread(target = run)
    thread.start()
    assert shared_resident.running.wait(20)
    return thread, result


def test_generation_progress_and_cancel_follow_the_account_that_started_it(shared_resident):
    thread, result = _start_alice_generation(shared_resident)
    try:
        with client_for(ALICE) as client:
            progress = client.get("/api/inference/images/generate-progress").json()
            assert progress["active"] is True and progress["step"] == 3
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": True
            }
    finally:
        shared_resident.cancelled.set()
        thread.join(20)
    assert result["response"].status_code == 409


def test_the_model_loader_cannot_see_or_cancel_another_accounts_generation(shared_resident):
    thread, result = _start_alice_generation(shared_resident)
    try:
        with client_for(BOB) as client:
            assert client.get("/api/inference/images/generate-progress").json() == {
                "loaded": True,
                "yours": False,
            }
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": False
            }
        assert not shared_resident.cancelled.is_set()
    finally:
        shared_resident.cancelled.set()
        thread.join(20)
    assert result["response"].status_code == 409


def test_residency_still_governs_progress_and_cancel_with_no_generation_in_flight(monkeypatch):
    from core.inference import diffusion_engine_router

    backend = SimpleNamespace(
        status = lambda: {"loaded": True, "repo_id": "org/public-model"},
        generate_progress = lambda: {"active": False},
        cancel_generate = lambda **kwargs: False,
    )
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", BOB.account_id)
    with client_for(ALICE) as client:
        assert client.get("/api/inference/images/generate-progress").json() == {
            "loaded": True,
            "yours": False,
        }
        assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": False}
    with client_for(BOB) as client:
        assert client.get("/api/inference/images/generate-progress").json()["active"] is False


def _start_alice_openai_generation(shared_resident):
    result = {}

    def run():
        with client_for(ALICE) as client:
            result["response"] = client.post(
                "/v1/images/generations",
                json = {
                    "prompt": "a sloth",
                    "size": "256x256",
                    "response_format": "b64_json",
                },
            )

    thread = threading.Thread(target = run)
    thread.start()
    assert shared_resident.running.wait(20)
    return thread, result


def test_openai_image_generations_belong_to_the_account_that_started_them(shared_resident):
    thread, result = _start_alice_openai_generation(shared_resident)
    try:
        with client_for(ALICE) as client:
            progress = client.get("/api/inference/images/generate-progress").json()
            assert progress["active"] is True and progress["step"] == 3
        with client_for(BOB) as client:
            assert client.get("/api/inference/images/generate-progress").json() == {
                "loaded": True,
                "yours": False,
            }
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": False
            }
        assert not shared_resident.cancelled.is_set()
    finally:
        shared_resident.cancelled.set()
        thread.join(20)
    assert result["response"].status_code >= 400
