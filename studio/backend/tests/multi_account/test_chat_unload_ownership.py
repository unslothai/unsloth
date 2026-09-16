# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat teardown must drop the CHAT claim, or every other account sees a phantom resident."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter, llama_keepwarm
from hub.services.models import account_access as access
from routes import inference
from utils.account_context import bind_account, reset_account

RESIDENT = "alice/private-gguf"


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.router, prefix = "/api/inference")
    return TestClient(app)


class FakeLlama:
    """Only the surface /unload and /status read; teardown flips residency like the real one."""

    def __init__(self):
        self.is_active = True
        self.is_loaded = True
        self.model_identifier = RESIDENT
        self.hf_variant = None
        self.chat_template_override = None
        self.unloaded = False

    def unload_model(self):
        self.is_active = False
        self.is_loaded = False
        self.model_identifier = None
        self.unloaded = True
        return True


@pytest.fixture
def chat_resident(monkeypatch, accounts):
    llama = FakeLlama()
    standard = SimpleNamespace(
        active_model_name = None,
        models = {},
        get_loading_model = lambda: None,
        loading_models = (),
    )
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts["alice"].account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: standard)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(inference, "_probe_llama_cpp_status", lambda _: (False, {}))
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    monkeypatch.setattr(inference, "_pending_load_attempts", {})
    return llama


def bob_status(accounts):
    with client_for(accounts["bob"]) as client:
        return client.get("/api/inference/status")


def test_manual_unload_releases_chat_ownership(chat_resident, accounts):
    with client_for(accounts["alice"]) as client:
        unload = client.post("/api/inference/unload", json = {"model_path": RESIDENT})
    assert unload.status_code == 200, unload.text
    assert chat_resident.unloaded, "the fake backend was never torn down"

    response = bob_status(accounts)
    assert response.status_code == 200, response.text
    body = response.json()
    # The hidden-resident answer is exactly {"loaded": True, "yours": False}; the real
    # status has no "yours" key and reports "loaded" as the list of resident models.
    assert "yours" not in body, f"phantom resident after unload: {body}"
    assert body.get("loaded") == [], f"phantom resident after unload: {body}"
    assert body.get("active_model") is None, body
    assert gpu_arbiter.current_owner() is None
    assert gpu_arbiter.owner_account() is None


def test_idle_auto_unload_releases_chat_ownership(chat_resident, accounts, monkeypatch):
    monkeypatch.setattr(llama_keepwarm, "_is_idle", lambda ttl: True)
    monkeypatch.setattr(llama_keepwarm, "_note_activity", lambda: None)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_auto_unload_idle_seconds", lambda: 1.0
    )
    monkeypatch.setattr("utils.openai_auto_switch_settings.get_auto_unload_api_only", lambda: False)
    monkeypatch.setattr("utils.openai_auto_switch_settings.get_auto_unload_keep_kv", lambda: False)

    async def one_tick():
        task = asyncio.ensure_future(llama_keepwarm.idle_unload_loop(poll_seconds = 0.01))
        for _ in range(500):
            await asyncio.sleep(0.01)
            if chat_resident.unloaded:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_tick())
    assert chat_resident.unloaded, "idle loop never tore the backend down"

    response = bob_status(accounts)
    assert response.status_code == 200, response.text
    body = response.json()
    # The hidden-resident answer is exactly {"loaded": True, "yours": False}; the real
    # status has no "yours" key and reports "loaded" as the list of resident models.
    assert "yours" not in body, f"phantom resident after idle unload: {body}"
    assert body.get("loaded") == [], f"phantom resident after idle unload: {body}"
    assert body.get("active_model") is None, body
    assert gpu_arbiter.current_owner() is None
    assert gpu_arbiter.owner_account() is None
