# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from core.systemone import catalog, clef_worker, native_worker, runtime
from routes import systemone
from utils import systemone_settings


def png_url():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def api(monkeypatch):
    for name, value in (
        ("get_enabled", True),
        ("get_model", "clef-flash"),
        ("get_backend", "auto"),
    ):
        monkeypatch.setattr(systemone_settings, name, lambda value = value: value)
    calls = []

    def decide(checkpoint, state, questions, images):
        calls.append((checkpoint, state, questions, images))
        return {
            "model": checkpoint.name,
            "answers": {"q": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "_backend": "pytorch",
        }

    monkeypatch.setattr(runtime, "decide", decide)
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


@pytest.mark.parametrize("images", [[], [png_url()]])
def test_default_alias_keeps_wire_format_and_media(api, images):
    client, calls = api
    response = client.post("/v1/systemone", json = request(**({"images": images} if images else {})))
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"model", "answers", "usage"}
    assert response.json()["model"] == "clef-flash"
    assert response.headers["x-typesafe-request-id"]
    assert response.headers["x-unsloth-decision-backend"] == "pytorch"
    assert calls[0][0] == catalog.CHECKPOINTS["clef-flash"]
    assert len(calls[0][3]) == len(images)


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


@pytest.mark.parametrize(
    "extra,status,error",
    [
        *[
            ({field: []}, 400, field)
            for field in ("audio", "videos", "media_kwargs", "temperature", "arbitrary")
        ],
        *[
            ({"images": images}, 422, None)
            for images in (
                ["https://example.org/img.png"],
                ["data:image/png;base64,AAAA"],
                [png_url()] * 5,
                [1],
                "image",
            )
        ],
    ],
)
def test_invalid_extensions_fail_before_runtime(api, extra, status, error):
    client, calls = api
    response = client.post("/v1/systemone", json = request(**extra))
    assert response.status_code == status, response.text
    if error:
        assert error in response.text
    assert not calls


@pytest.mark.parametrize("model", ["connection:provider:clef", "laya-multilingual"])
def test_images_are_rejected_for_text_only_models(api, monkeypatch, model):
    client, calls = api
    monkeypatch.setattr(systemone_settings, "get_model", lambda: model)
    response = client.post("/v1/systemone", json = request(images = [png_url()]))
    assert response.status_code == 400 and "text-only" in response.text
    assert not calls


def test_catalog_pins_both_heads_and_vision_and_preserves_aliases(monkeypatch):
    source = clef_worker._VENDORED_CLEF / "joint_schema_model.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == clef_worker._SOURCE_SHA256
    for name in ("clef", "clef-flash"):
        checkpoint = catalog.CHECKPOINTS[name]
        assert checkpoint.layout == "clef" and len(checkpoint.revision) == 40
        assert {"joint_head.safetensors", "processor_config.json"} <= set(checkpoint.files)
        assert not any(f.endswith(".py") for f in checkpoint.files)
        assert catalog.resolve(checkpoint.source) == checkpoint
        monkeypatch.setattr(systemone_settings, "get_model", lambda: name)
        assert all(catalog.resolve(alias) == checkpoint for alias in catalog.DEFAULT_ALIASES)
    assert systemone_settings.DEFAULT_MODEL == "laya-multilingual"


def test_models_discovery_reports_selected_alias_and_native_capabilities(api):
    models = {m["id"]: m for m in systemone.decision_model_objects()}
    assert models["default"]["architecture"]["input_modalities"] == ["text", "image"]
    assert models["clef"]["architecture"]["input_modalities"] == ["text", "image"]
    assert models["laya-multilingual"]["architecture"]["input_modalities"] == ["text"]


@pytest.mark.parametrize(
    "available,images,preference,expected",
    [
        (False, False, "auto", "pytorch"),
        (True, False, "auto", "llama.cpp"),
        (True, True, "auto", "pytorch"),
        (True, False, "pytorch", "pytorch"),
        (False, False, "llama.cpp", None),
    ],
)
def test_selection_uses_capabilities_only(monkeypatch, available, images, preference, expected):
    monkeypatch.setattr(
        native_worker,
        "native_availability",
        lambda: {"available": available, "reason": None if available else "old binary"},
    )
    checkpoint = catalog.CHECKPOINTS["clef-flash"]
    if expected is None:
        with pytest.raises(runtime.Unavailable, match = "old binary"):
            runtime.select_checkpoint(checkpoint, images = images, preference = preference)
    else:
        selected, reason = runtime.select_checkpoint(
            checkpoint, images = images, preference = preference
        )
        assert runtime._clef().is_native(selected) == (expected == "llama.cpp")
        assert bool(reason) == (preference == "auto" and (not available or images))


def test_native_only_discovery_and_image_refusal(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: "llama.cpp")
    models = {m["id"]: m for m in systemone.decision_model_objects()}
    assert models["clef"]["architecture"]["input_modalities"] == ["text"]
    response = client.post("/v1/systemone", json = request(images = [png_url()]))
    assert response.status_code == 400 and not calls


def test_native_inspection_error_is_not_a_fallback(monkeypatch):
    def fail():
        raise RuntimeError("unexpected native error")

    monkeypatch.setattr(native_worker, "native_availability", fail)
    with pytest.raises(RuntimeError, match = "unexpected native error"):
        runtime.select_checkpoint(catalog.CHECKPOINTS["clef-flash"], preference = "auto")
