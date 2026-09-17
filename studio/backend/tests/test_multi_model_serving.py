# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Models loaded alongside the primary one are routed to by request model name."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.orchestrator as orchestrator
import routes.inference as inf
from auth.authentication import get_current_subject
from models.inference import LoadRequest, UnloadRequest


class FakeLlama:
    def __init__(self, identifier = None, variant = None):
        self.model_identifier = identifier
        self.hf_variant = variant
        self.is_loaded = identifier is not None
        self.context_length = 4096

    @property
    def is_active(self):
        return self.is_loaded

    def unload_model(self):
        self.is_loaded = False

    def _cleanup(self):
        pass


class FakeOrchestrator:
    def __init__(self, active = None):
        self.active_model_name = active
        self.models = {active: {}} if active else {}

    def _cleanup(self):
        self.active_model_name = None


@pytest.fixture
def backends(monkeypatch):
    primary = FakeLlama("org/A-GGUF", "Q4_K_M")
    extra = inf._ExtraSlot(FakeLlama("org/B-GGUF", "Q8_0"), FakeOrchestrator())
    monkeypatch.setattr(inf, "_llama_cpp_backend", primary)
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(inf, "_extra_slots", [extra])
    return primary, extra


def _routed(requested):
    async def run():
        slot = await inf._route_to_extra_slot(requested)
        return slot, inf.get_llama_cpp_backend()

    return asyncio.run(run())


def test_request_model_name_picks_the_backend(backends):
    primary, extra = backends
    assert _routed("org/B-GGUF") == (extra, extra.llama)
    assert _routed("org/B-GGUF:Q8_0") == (extra, extra.llama)
    assert _routed("org/B-GGUF:Q4_K_M") == (None, primary)
    assert _routed("org/A-GGUF") == (None, primary)
    assert _routed(None) == (None, primary)


def test_a_safetensors_slot_is_served_by_its_own_orchestrator(backends):
    safetensors = inf._ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"))
    inf._extra_slots.append(safetensors)

    async def run():
        slot = await inf._route_to_extra_slot("org/C")
        return slot, inf.get_llama_cpp_backend().is_loaded, inf.get_inference_backend()

    assert asyncio.run(run()) == (safetensors, False, safetensors.orchestrator)


def test_loaded_models_lists_every_backend(backends):
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert {entry["id"] for entry in data} == {"org/A-GGUF", "org/B-GGUF"}
    inf._extra_slots.append(inf._ExtraSlot(FakeLlama(), FakeOrchestrator("org/C")))
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert [entry["id"] for entry in data] == ["org/A-GGUF", "org/B-GGUF", "org/C"]


def _selected(monkeypatch, request):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)

    async def run():
        slot = await inf._select_load_slot(request)
        assert inf._routed_llama_backend.get() is (slot.llama if slot else None)
        return slot

    return asyncio.run(run())


def test_alongside_load_gets_a_new_slot(backends, monkeypatch):
    _, extra = backends
    fresh = _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True))
    assert fresh not in (None, extra)


def test_plain_load_replaces_the_primary(backends, monkeypatch):
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF")) is None


def test_alongside_load_uses_an_empty_primary(backends, monkeypatch):
    primary, _ = backends
    primary.is_loaded = False
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True)) is None


def test_loading_a_model_an_extra_slot_serves_reuses_it(backends, monkeypatch):
    _, extra = backends
    request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    assert _selected(monkeypatch, request) is extra


def test_unload_drops_only_the_named_extra_slot(backends):
    primary, extra = backends
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not extra.llama.is_loaded
    assert inf._extra_slots == []
