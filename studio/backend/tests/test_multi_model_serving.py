# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF models loaded alongside the primary one are routed to by request model name."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


@pytest.fixture
def backends(monkeypatch):
    primary = FakeLlama("org/A-GGUF", "Q4_K_M")
    extra = FakeLlama("org/B-GGUF", "Q8_0")
    monkeypatch.setattr(inf, "_llama_cpp_backend", primary)
    monkeypatch.setattr(inf, "_extra_llama_backends", [extra])
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(
        inf, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None, models = {})
    )
    return primary, extra


def _routed(requested):
    async def run():
        routed = await inf._route_to_extra_backend(requested)
        return routed, inf.get_llama_cpp_backend()

    return asyncio.run(run())


def test_request_model_name_picks_the_backend(backends):
    primary, extra = backends
    assert _routed("org/B-GGUF") == (True, extra)
    assert _routed("org/B-GGUF:Q8_0") == (True, extra)
    assert _routed("org/B-GGUF:Q4_K_M") == (False, primary)
    assert _routed("org/A-GGUF") == (False, primary)
    assert _routed(None) == (False, primary)


def test_a_dead_extra_backend_is_forgotten(backends):
    primary, extra = backends
    extra.is_loaded = False
    assert _routed("org/B-GGUF") == (False, primary)
    assert inf._extra_llama_backends == []


def test_loaded_models_lists_every_backend(backends):
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert {entry["id"] for entry in data} == {"org/A-GGUF", "org/B-GGUF"}


def _selected(monkeypatch, request):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)

    async def run():
        await inf._select_load_backend(request)
        return inf._routed_llama_backend.get()

    return asyncio.run(run())


def test_alongside_load_gets_a_new_backend(backends, monkeypatch):
    primary, extra = backends
    fresh = _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True))
    assert fresh not in (None, primary, extra)


def test_plain_load_replaces_the_primary(backends, monkeypatch):
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF")) is None


def test_alongside_load_uses_an_empty_primary(backends, monkeypatch):
    primary, _ = backends
    primary.is_loaded = False
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True)) is None


def test_loading_a_model_an_extra_backend_serves_reuses_it(backends, monkeypatch):
    _, extra = backends
    request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    assert _selected(monkeypatch, request) is extra


def test_unload_drops_only_the_named_extra_backend(backends):
    primary, extra = backends
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not extra.is_loaded
    assert inf._extra_llama_backends == []
