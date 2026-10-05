# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API contract tests. Stubbed inference here is not model-backed validation."""

import base64
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from core.systemone import catalog
from routes import systemone
from utils import systemone_settings


def png_url():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(systemone_settings, "get_enabled", lambda: True)
    monkeypatch.setattr(systemone_settings, "get_model", lambda: "clef-flash")
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: "auto")
    calls = []

    def decide(checkpoint, state, questions, images):
        calls.append((checkpoint, state, questions, images))
        return {
            "model": checkpoint.name,
            "answers": {"q": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "_backend": "pytorch",
        }

    monkeypatch.setattr(systemone.decision_runtime, "decide", decide)
    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app), calls


def request(**extra):
    return {
        "state": "Inspect",
        "model": "jev-preview",
        "questions": {"q": {"type": "noul"}},
        **extra,
    }


def test_default_legacy_model_alias_uses_configured_clef_and_images(api):
    client, calls = api
    response = client.post("/v1/systemone", json = request(images = [png_url()]))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "clef-flash"
    assert response.headers["x-typesafe-request-id"]
    assert response.headers["x-unsloth-decision-backend"] == "pytorch"
    assert "_backend" not in response.json()
    assert calls[0][0] == catalog.CHECKPOINTS["clef-flash"]
    assert len(calls[0][3]) == 1


def test_legacy_text_body_keeps_wire_format(api):
    client, calls = api
    response = client.post("/v1/systemone", json = request())
    assert response.status_code == 200
    assert calls[0][3] == []
    assert set(response.json()) == {"model", "answers", "usage"}


def test_image_url_state_parts_follow_same_contract(api):
    client, calls = api
    state = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect"},
                {"type": "image_url", "image_url": {"url": png_url()}},
            ],
        }
    ]
    response = client.post("/v1/systemone", json = request(state = state))
    assert response.status_code == 200
    assert calls[0][1] == [{"role": "user", "content": [{"type": "text", "text": "Inspect"}]}]
    assert len(calls[0][3]) == 1


@pytest.mark.parametrize("field", ["audio", "videos", "media_kwargs", "temperature", "arbitrary"])
def test_unknown_or_unsupported_extensions_fail_before_runtime(api, field):
    client, calls = api
    response = client.post("/v1/systemone", json = request(**{field: []}))
    assert response.status_code == 400 and field in response.text
    assert not calls


@pytest.mark.parametrize(
    "images",
    [
        ["https://example.org/img.png"],
        ["data:image/png;base64,AAAA"],
        [png_url()] * 5,
        [1],
        "image",
    ],
)
def test_malformed_or_unbounded_media_fails_before_runtime(api, images):
    client, calls = api
    response = client.post("/v1/systemone", json = request(images = images))
    assert response.status_code == 422, response.text
    assert not calls


def test_images_are_rejected_for_text_only_saved_connections(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(systemone_settings, "get_model", lambda: "connection:provider:clef")
    response = client.post("/v1/systemone", json = request(images = [png_url()]))
    assert response.status_code == 400 and "text-only" in response.text
    assert not calls


def test_laya_rejects_media_without_changing_its_default(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(systemone_settings, "get_model", lambda: "laya-multilingual")
    response = client.post("/v1/systemone", json = request(images = [png_url()]))
    assert response.status_code == 400 and "text-only" in response.text
    assert not calls


def test_catalog_pins_both_heads_and_vision_and_preserves_aliases(monkeypatch):
    for name in ("clef", "clef-flash"):
        checkpoint = catalog.CHECKPOINTS[name]
        assert len(checkpoint.revision) == 40
        assert "joint_head.safetensors" in checkpoint.files
        assert "processor_config.json" in checkpoint.files
        assert not any(f.endswith(".py") for f in checkpoint.files)
        assert catalog.resolve(checkpoint.source) == checkpoint
        monkeypatch.setattr(systemone_settings, "get_model", lambda: name)
        for alias in catalog.DEFAULT_ALIASES:
            assert catalog.resolve(alias) == checkpoint
    assert systemone_settings.DEFAULT_MODEL == "laya-multilingual"


def test_models_discovery_reports_selected_alias_and_native_capabilities(api):
    models = {m["id"]: m for m in systemone.decision_model_objects()}
    assert models["default"]["architecture"]["input_modalities"] == ["text", "image"]
    assert models["clef"]["architecture"]["input_modalities"] == ["text", "image"]
    assert models["laya-multilingual"]["architecture"]["input_modalities"] == ["text"]


@pytest.mark.parametrize("available,images,preference,expected", [
    (False, False, "auto", "pytorch"),
    (True, False, "auto", "llama.cpp"),
    (True, True, "auto", "pytorch"),
    (True, False, "pytorch", "pytorch"),
    (False, False, "llama.cpp", None),
])
def test_selection_uses_capabilities_only(monkeypatch, available, images, preference, expected):
    from core.systemone import native_worker, runtime

    monkeypatch.setattr(native_worker, "native_availability", lambda: {
        "available": available, "reason": None if available else "old binary",
    })
    checkpoint = catalog.CHECKPOINTS["clef-flash"]
    if expected is None:
        with pytest.raises(runtime.Unavailable, match="old binary"):
            runtime.select_checkpoint(checkpoint, images=images, preference=preference)
    else:
        selected, reason = runtime.select_checkpoint(checkpoint, images=images, preference=preference)
        assert runtime._clef().is_native(selected) == (expected == "llama.cpp")
        assert bool(reason) == (preference == "auto" and (not available or images))


def test_native_only_discovery_and_image_refusal(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: "llama.cpp")
    models = {m["id"]: m for m in systemone.decision_model_objects()}
    assert models["clef"]["architecture"]["input_modalities"] == ["text"]
    response = client.post("/v1/systemone", json=request(images=[png_url()]))
    assert response.status_code == 400 and not calls


def test_native_inspection_error_is_not_a_fallback(monkeypatch):
    from core.systemone import native_worker, runtime

    def fail():
        raise RuntimeError("unexpected native error")

    monkeypatch.setattr(native_worker, "native_availability", fail)
    with pytest.raises(RuntimeError, match="unexpected native error"):
        runtime.select_checkpoint(catalog.CHECKPOINTS["clef-flash"], preference="auto")
