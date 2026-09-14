# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accounts whose matching load reused the resident chat model share it: both see and use it,
one leaving does not evict the other, and a replacement or teardown starts the set afresh."""

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from auth import policy
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from models.inference import LoadRequest
from routes import inference
from state import active_generations
from studio.backend.tests.llama_backend_double import FakeLlamaCppBackend
from utils.account_context import bind_account, reset_account, run_as

RESIDENT = "unsloth/shared-gguf"
VARIANT = "Q4_K_M"


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


class FakeLlama(FakeLlamaCppBackend):
    """The surface /load's reuse path, /unload and /status read."""

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
    tensor_parallel = False
    disable_vision = False
    vision_disabled_by_user = False
    gpu_memory_mode = "auto"
    gpu_layers = 0
    n_cpu_moe = 0
    n_moe_layers = 0

    def __init__(self):
        self.is_active = True
        self.is_loaded = True
        self.model_identifier = RESIDENT
        self.hf_variant = VARIANT
        self.chat_template_override = None
        self.holds_no_vram = False
        self._audio_probed = True
        self._openai_gguf_companion_roots = ()
        self._openai_gguf_companion_state = ()
        self.unloaded = False

    def __getattr__(self, name):
        # Status reads many optional runtime fields; an unset one reads as the real backend's None.
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def adopt_load_intent_if_matched(self, intent):
        return True

    def unload_model(self):
        self.is_active = False
        self.is_loaded = False
        self.model_identifier = None
        self.unloaded = True
        return True


@pytest.fixture
def shared(monkeypatch, accounts):
    """Bob loaded RESIDENT; the arbiter and the resident record say so."""
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
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts["bob"].account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: standard)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(inference, "_probe_llama_cpp_status", lambda _: (False, {}))
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    monkeypatch.setattr(inference, "_pending_load_attempts", {})
    active_generations.reset_for_tests()
    run_as(accounts["bob"], access.publish_resident, "chat", RESIDENT)
    return llama


def reuse_load(monkeypatch, account):
    """Run the real /load implementation down its GGUF reuse path as ``account``."""
    responses = []
    monkeypatch.setattr(
        inference,
        "_resolve_model_identifier_for_request",
        lambda *a, **k: (RESIDENT, RESIDENT, False),
    )
    monkeypatch.setattr(inference, "resolve_effective_chat_template_override", lambda *a, **k: None)
    monkeypatch.setattr(inference, "_active_gguf_intent", lambda *a, **k: object())
    monkeypatch.setattr(access, "require_model_access", lambda *a, **k: None)

    def response(_backend, status, *a, **k):
        responses.append(status)
        return SimpleNamespace(status = status, model = RESIDENT)

    monkeypatch.setattr(inference, "_gguf_load_response", response)
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    request = LoadRequest(model_path = RESIDENT, gguf_variant = VARIANT)
    result = run_as(
        account,
        lambda: asyncio.run(
            inference._load_model_impl(request, fastapi_request, current_subject = account.username)
        ),
    )
    assert responses == ["already_loaded"], responses
    return result


def status_for(account):
    with client_for(account) as client:
        response = client.get("/api/inference/status")
    assert response.status_code == 200, response.text
    return response.json()


def sees_resident(account):
    body = status_for(account)
    return body.get("model_identifier") == RESIDENT and "yours" not in body


def test_a_matching_load_shares_the_resident_without_a_reload(monkeypatch, shared, accounts):
    assert sees_resident(accounts["bob"])
    assert not sees_resident(accounts["alice"])
    reuse_load(monkeypatch, accounts["alice"])
    assert not shared.unloaded
    print(f"sharers: {access._resident_sharers}")
    assert sees_resident(accounts["alice"]) and sees_resident(accounts["bob"])
    assert run_as(accounts["alice"], inference._openai_model_objects) != []
    assert sees_resident(accounts["unsloth"])
    # A repeat load by the same account is one membership.
    reuse_load(monkeypatch, accounts["alice"])
    assert access._resident_sharers["chat"] == {
        accounts["alice"].account_id,
        accounts["bob"].account_id,
    }


def test_a_sharer_leaving_keeps_the_model_for_the_others(monkeypatch, shared, accounts):
    reuse_load(monkeypatch, accounts["alice"])
    # Bob is mid-generation: alice's share release must not be a gpu_busy refusal.
    with run_as(accounts["bob"], active_generations.ActiveGeneration, threading.Event()):
        with client_for(accounts["alice"]) as client:
            left = client.post("/api/inference/unload", json = {"model_path": RESIDENT})
    assert left.status_code == 200, left.text
    assert not shared.unloaded
    assert sees_resident(accounts["bob"])
    assert not sees_resident(accounts["alice"])
    assert status_for(accounts["alice"]) == {"loaded": [], "loading": [], "yours": False}
    # The last sharer's unload is a real teardown and clears the set.
    with client_for(accounts["bob"]) as client:
        gone = client.post("/api/inference/unload", json = {"model_path": RESIDENT})
    assert gone.status_code == 200, gone.text
    assert shared.unloaded
    assert "chat" not in access._resident_sharers


def test_a_replacement_starts_the_sharers_afresh(monkeypatch, shared, accounts):
    reuse_load(monkeypatch, accounts["alice"])
    run_as(accounts["bob"], access.publish_resident, "chat", "bob/other-model")
    assert access._resident_sharers["chat"] == {accounts["bob"].account_id}
    assert run_as(accounts["alice"], access.resident_hidden, "chat", "bob/other-model")


def test_retirement_and_eviction_drop_shares(monkeypatch, shared, accounts):
    reuse_load(monkeypatch, accounts["alice"])
    access.retire_resident_shares(accounts["alice"].account_id)
    assert not sees_resident(accounts["alice"]) and sees_resident(accounts["bob"])
    access.clear_resident("chat")
    assert "chat" not in access._resident_sharers


def test_single_account_installs_record_nothing(monkeypatch):
    monkeypatch.setattr(access, "_resident_sharers", {})
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: False)
    access.publish_resident("chat", RESIDENT)
    access.join_resident("chat")
    assert access._resident_sharers == {}
    assert access.release_shared_resident("chat") is False
