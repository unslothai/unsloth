# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/images/generations.

The diffusion backend and image gallery are replaced with light fakes, so these
exercise the route wiring, OpenAI param mapping, validation, error envelopes, and
response shape without torch, diffusers, weights, or a GPU. The pure helpers
(`_parse_openai_image_size`, `default_generation_params`) are unit-tested directly.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.diffusion as diffusion_module
import core.inference.image_gallery as gallery_module
from auth.authentication import get_current_subject
from core.inference.diffusion_families import default_generation_params
from routes.inference import router, _parse_openai_image_size
from utils.api_errors import install_api_error_handlers


# ── pure helpers ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "repo_id, expected",
    [
        ("unsloth/Z-Image-Turbo-GGUF", (9, 0.0)),   # turbo entry, before the z-image fallback
        ("unsloth/Z-Image-GGUF", (20, 4.0)),
        ("unsloth/FLUX.1-schnell-GGUF", (4, 0.0)),   # schnell entry, before the flux.1 entry
        ("black-forest-labs/FLUX.1-dev", (28, 3.5)),
        ("unsloth/FLUX.2-klein-4B-GGUF", (4, 0.0)),
        ("unsloth/Qwen-Image-2512-GGUF", (20, 4.0)),
        ("some/unknown-model", (9, 0.0)),            # fallback
        ("", (9, 0.0)),
    ],
)
def test_default_generation_params(repo_id, expected):
    assert default_generation_params(repo_id) == expected


def test_default_generation_params_specificity_ordering():
    # The "-turbo" / "-schnell" entries must win over their broader siblings; a
    # reorder that broke this would silently mis-default.
    assert default_generation_params("x/Z-Image-Turbo") != default_generation_params("x/Z-Image")
    assert default_generation_params("x/FLUX.1-schnell") != default_generation_params("x/FLUX.1-dev")


@pytest.mark.parametrize(
    "size, expected",
    [
        ("auto", (1024, 1024)),
        ("", (1024, 1024)),
        ("AUTO", (1024, 1024)),
        ("512x512", (512, 512)),
        ("512x256", (512, 256)),
        ("1792x1024", (1792, 1024)),   # dall-e-3 named size: must pass the bounds
        ("1024x1792", (1024, 1792)),
        (" 256 x 256 ", (256, 256)),
    ],
)
def test_parse_image_size_ok(size, expected):
    assert _parse_openai_image_size(size) == expected


@pytest.mark.parametrize("size", ["abc", "100x100", "4096x4096", "300x300", "512", "x512", "0x0"])
def test_parse_image_size_rejects(size):
    with pytest.raises(ValueError):
        _parse_openai_image_size(size)


# ── route round-trip ────────────────────────────────────────────────────


class _FakeBackend:
    def __init__(self, loaded = True, repo_id = "unsloth/Z-Image-Turbo-GGUF") -> None:
        self._loaded = loaded
        self._repo_id = repo_id
        self.calls = []

    def status(self):
        return {
            "loaded": self._loaded,
            "repo_id": self._repo_id if self._loaded else None,
            "family": "z-image" if self._loaded else None,
            "base_repo": None, "device": "cpu", "dtype": "float32", "cpu_offload": False,
        }

    def generate(self, *, prompt, width, height, steps, guidance, batch_size = 1):
        if not self._loaded:
            raise RuntimeError("No diffusion model is loaded.")
        self.calls.append(dict(prompt = prompt, width = width, height = height,
                               steps = steps, guidance = guidance, batch_size = batch_size))
        return {
            "images": [object() for _ in range(batch_size)],
            "seed": 4242,
            "repo_id": self._repo_id,
        }


def _make_client(backend):
    store = {}

    def _save(image, meta):
        image_id = f"img{len(store)}"
        record = {**meta, "id": image_id, "url": f"/api/inference/images/gallery/{image_id}/file"}
        store[image_id] = record
        return record

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), store, _save


@pytest.fixture
def client(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)
    cli.backend = backend  # type: ignore[attr-defined]
    return cli


def _post(client, body):
    return client.post("/v1/images/generations", json = body)


def test_url_response_shape(client):
    resp = _post(client, {"prompt": "a sloth", "size": "256x256"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"created", "data"}
    assert isinstance(body["created"], int) and body["created"] > 0
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert "url" in item and "b64_json" not in item      # exclude_none drops the unused key
    assert item["url"].endswith("/file")
    # Z-Image-Turbo defaults (9 steps, 0 guidance) flow into the backend call.
    assert client.backend.calls[0] == dict(
        prompt = "a sloth", width = 256, height = 256, steps = 9, guidance = 0.0, batch_size = 1
    )


def test_b64_response_shape(client):
    resp = _post(client, {"prompt": "a sloth", "size": "256x256", "response_format": "b64_json"})
    assert resp.status_code == 200
    item = resp.json()["data"][0]
    assert "b64_json" in item and "url" not in item
    assert item["b64_json"] == "QUJD"


def test_n_maps_to_batch(client):
    resp = _post(client, {"prompt": "p", "size": "256x256", "n": 3})
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 3
    assert client.backend.calls[0]["batch_size"] == 3


def test_null_fields_coalesce_to_defaults(client):
    # OpenAI marks n/size/response_format nullable-with-default: null -> default.
    resp = _post(client, {"prompt": "p", "n": None, "size": None, "response_format": None})
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1
    assert "url" in resp.json()["data"][0]
    assert client.backend.calls[0]["width"] == 1024  # size null -> auto -> 1024


@pytest.mark.parametrize(
    "body, param",
    [
        ({"size": "256x256"}, "prompt"),               # missing prompt
        ({"prompt": "", "size": "256x256"}, "prompt"),  # empty prompt
        ({"prompt": "p", "size": "300x300"}, "size"),   # not multiple of 16
        ({"prompt": "p", "size": "abc"}, "size"),       # unparseable
        ({"prompt": "p", "stream": True}, "stream"),    # streaming unsupported
    ],
)
def test_validation_400_with_param(client, body, param):
    resp = _post(client, body)
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == param
    for k in ("message", "code"):
        assert k in err


@pytest.mark.parametrize("n", [0, 11, -1])
def test_n_out_of_range_400(client, n):
    resp = _post(client, {"prompt": "p", "size": "256x256", "n": n})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_no_model_loaded_503(monkeypatch):
    backend = _FakeBackend(loaded = False)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p"})
    assert resp.status_code == 503
    # 503 still wears the OpenAI envelope (api_error) on the /v1 surface.
    assert resp.json()["error"]["type"] == "api_error"


def test_auth_required():
    backend = _FakeBackend()
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    # No dependency override: the real auth dependency runs and rejects.
    resp = TestClient(app).post("/v1/images/generations", json = {"prompt": "p"})
    assert resp.status_code in (401, 403)
