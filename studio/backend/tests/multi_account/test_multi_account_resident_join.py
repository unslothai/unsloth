# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-account resident metadata, joining, and replacement guards."""

import asyncio
import json
import secrets
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from auth import storage
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import inference
from state import active_generations
from studio.backend.tests.llama_backend_double import FakeLlamaCppBackend
from utils.account_context import bind_account, reset_account, run_as


RESIDENT_PATH = r"C:\Users\alice\secrets\GLM-5.3-Flash.gguf"
RESIDENT_LABEL = "GLM-5.3-Flash"
RESIDENT_VARIANT = "Q4_K_XL"


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(inference.router, prefix = "/api/inference")
    return TestClient(app)


class FakeLlama(FakeLlamaCppBackend):
    is_diffusion = False
    is_audio = False
    has_audio_input = False
    has_video_input = False
    requires_trust_remote_code = False
    supports_reasoning = False
    reasoning_style = "enable_thinking"
    reasoning_effort_levels: list = []
    reasoning_always_on = False
    supports_preserve_thinking = False
    tensor_parallel = True
    effective_parallel_slots = 2
    requested_parallel_slots = 2
    disable_vision = False
    vision_disabled_by_user = False
    gpu_memory_mode = "auto"
    gpu_layers = 0
    n_cpu_moe = 0
    n_moe_layers = 0

    def __init__(self):
        self.is_active = True
        self.is_loaded = True
        self.model_identifier = RESIDENT_PATH
        self._native_grant_backed = True
        self._native_display_label = RESIDENT_LABEL
        self.hf_variant = RESIDENT_VARIANT
        self.chat_template_override = None
        self.holds_no_vram = False
        self._audio_probed = True
        self._openai_gguf_companion_roots = ()
        self._openai_gguf_companion_state = ()
        self.unload_count = 0

    def adopt_load_intent_if_matched(self, intent):
        return True

    def unload_model(self):
        self.unload_count += 1
        self.is_active = False
        self.is_loaded = False
        self.model_identifier = None
        return True


@pytest.fixture
def resident(monkeypatch, accounts, isolated_auth):
    isolated_auth.create_initial_user("charlie", "account-password", secrets.token_urlsafe(32))
    accounts["charlie"] = isolated_auth.get_account("charlie")

    llama = FakeLlama()
    standard = SimpleNamespace(
        active_model_name = None,
        models = {},
        get_loading_model = lambda: None,
        loading_models = (),
    )
    for name in ("_resident_accounts", "_prior_resident_accounts", "_resident_sharers"):
        monkeypatch.setattr(access, name, {})
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts["alice"].account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: standard)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(inference, "_probe_llama_cpp_status", lambda _: (False, {}))
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    monkeypatch.setattr(inference, "_pending_load_attempts", {})
    active_generations.reset_for_tests()
    run_as(accounts["alice"], access.publish_resident, "chat", RESIDENT_PATH)
    return llama


def test_foreign_status_exposes_only_sanitized_resident_metadata(resident, accounts):
    with client_for(accounts["bob"]) as client:
        response = client.get("/api/inference/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["loaded"] == []
    assert body["yours"] is False
    assert body["resident"] == {
        "resident_available": True,
        "model": RESIDENT_LABEL,
        "variant": RESIDENT_VARIANT,
        "parallel_slots": 2,
        "tensor_parallel": True,
    }
    encoded = json.dumps(body)
    assert RESIDENT_PATH not in encoded
    assert "hf_token" not in encoded
    assert "account_id" not in encoded


def test_join_endpoint_attaches_without_gpu_reload(resident, accounts):
    with client_for(accounts["bob"]) as client:
        response = client.post("/api/inference/join-resident")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "joined"
    assert response.json()["resident"]["parallel_slots"] == 2
    assert resident.unload_count == 0
    assert access._resident_sharers["chat"] == {
        accounts["alice"].account_id,
        accounts["bob"].account_id,
    }


def test_join_after_unload_reports_resident_unavailable(monkeypatch, resident, accounts):
    from core.inference import llama_keepwarm

    class Gate:
        async def __aenter__(self):
            resident.unload_model()
            access.clear_resident("chat")
            gpu_arbiter.release(gpu_arbiter.CHAT)
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", lambda: Gate())
    with client_for(accounts["bob"]) as client:
        response = client.post("/api/inference/join-resident")

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "No resident model is loaded"


def test_different_model_while_foreign_account_generates_gets_descriptive_error(resident, accounts):
    with run_as(accounts["alice"], active_generations.ActiveGeneration, threading.Event()):
        with pytest.raises(HTTPException) as raised:
            run_as(accounts["charlie"], inference._raise_if_foreign_resident_load)

    assert raised.value.status_code == 409
    assert raised.value.detail["error"] == "resident_conflict"
    assert "Join the active model" in raised.value.detail["message"]
    assert "gpu_busy" not in raised.value.detail
    assert resident.unload_count == 0
