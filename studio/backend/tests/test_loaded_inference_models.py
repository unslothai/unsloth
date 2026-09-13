# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resident model discovery must not depend on the local disk/media catalog."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.inference as inf
from auth.authentication import get_current_subject


@pytest.fixture
def resident_backends(monkeypatch):
    llama = SimpleNamespace(
        is_loaded = True,
        model_identifier = "/models/Local-Q4.gguf",
        context_length = 4096,
        max_context_length = 8192,
        native_context_length = 32768,
    )
    backend = SimpleNamespace(active_model_name = None, models = {})
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inf, "get_inference_backend", lambda: backend)

    def no_scan(*args, **kwargs):
        pytest.fail("Resident discovery must not scan the disk or media catalog")

    monkeypatch.setattr(inf, "_cached_local_catalog", no_scan)
    monkeypatch.setattr(inf, "_servable_catalog_rows", no_scan)
    monkeypatch.setattr(inf, "_media_model_objects", no_scan)
    monkeypatch.setattr(inf, "_stt_model_objects", no_scan)
    return llama, backend


def _app():
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return app


def test_loaded_models_keep_public_ids_and_context_without_scan(resident_backends):
    llama, backend = resident_backends
    backend.active_model_name = "org/Other"
    backend.models = {
        "org/Other": {
            "context_length": 2048,
            "native_context_length": 16384,
            "max_context_length": 4096,
            "context_length_enforced": True,
        }
    }
    with TestClient(_app()) as client:
        response = client.get("/api/inference/loaded-models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    models = {entry["id"]: entry for entry in body["data"]}
    assert set(models) == {"Local-Q4", "org/Other"}
    assert all(entry["loaded"] is True for entry in models.values())
    assert models["Local-Q4"]["context_length"] == 4096
    assert models["Local-Q4"]["max_context_length"] == 8192
    assert models["Local-Q4"]["native_context_length"] == 32768
    assert models["org/Other"]["context_length"] == 2048
    assert models["org/Other"]["context_length_enforced"] is True
    assert "/models/" not in response.text


def test_loaded_models_reread_residency_without_scanning(resident_backends):
    llama, backend = resident_backends
    with TestClient(_app()) as client:
        assert len(client.get("/api/inference/loaded-models").json()["data"]) == 1
        llama.is_loaded = False
        assert client.get("/api/inference/loaded-models").json() == {"object": "list", "data": []}
        backend.active_model_name = "org/New"
        backend.models = {"org/New": {"context_length": 1024}}
        models = client.get("/api/inference/loaded-models").json()["data"]
        assert [entry["id"] for entry in models] == ["org/New"]


def test_loaded_models_require_auth_before_reading_backends(monkeypatch):
    def reject():
        raise HTTPException(status_code = 401, detail = "Unauthorized")

    def no_backend():
        pytest.fail("Authentication must run before backend discovery")

    app = _app()
    app.dependency_overrides[get_current_subject] = reject
    monkeypatch.setattr(inf, "_openai_model_objects", no_backend)
    with TestClient(app) as client:
        assert client.get("/api/inference/loaded-models").status_code == 401


@pytest.mark.parametrize("gguf_loaded", [False, True])
def test_discovery_does_not_construct_an_unused_orchestrator(
    resident_backends, monkeypatch, gguf_loaded
):
    from core.inference import orchestrator

    llama, _ = resident_backends
    llama.is_loaded = gguf_loaded
    # Exercise the real singleton getter/peek, not the fixture's replacement.
    monkeypatch.setattr(inf, "get_inference_backend", orchestrator.get_inference_backend)
    monkeypatch.setattr(orchestrator, "_inference_backend", None)

    def no_initialization():
        raise AssertionError("Resident discovery must not initialize an unused inference backend")

    monkeypatch.setattr(orchestrator, "InferenceOrchestrator", no_initialization)
    with TestClient(_app()) as client:
        response = client.get("/api/inference/loaded-models")
    assert response.status_code == 200
    assert [entry["id"] for entry in response.json()["data"]] == (
        ["Local-Q4"] if gguf_loaded else []
    )
    assert orchestrator.peek_inference_backend() is None
