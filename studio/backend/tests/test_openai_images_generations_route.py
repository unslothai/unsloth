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
import core.inference.diffusion_engine_router as engine_router
import core.inference.image_gallery as gallery_module
from auth.authentication import get_current_subject
from core.inference.diffusion_families import default_generation_params
from routes.inference import router, _parse_openai_image_size
from utils.api_errors import install_api_error_handlers


# ── pure helpers ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "repo_id, expected",
    [
        ("unsloth/Z-Image-Turbo-GGUF", (9, 0.0)),  # turbo entry, before the z-image fallback
        ("unsloth/Z-Image-GGUF", (20, 4.0)),
        ("unsloth/FLUX.1-schnell-GGUF", (4, 0.0)),  # schnell entry, before the flux.1 entry
        ("black-forest-labs/FLUX.1-dev", (28, 3.5)),
        ("unsloth/FLUX.2-klein-4B-GGUF", (4, 0.0)),
        ("unsloth/Qwen-Image-2512-GGUF", (20, 4.0)),
        ("some/unknown-model", (9, 0.0)),  # fallback
        ("", (9, 0.0)),
    ],
)
def test_default_generation_params(repo_id, expected):
    assert default_generation_params(repo_id) == expected


def test_default_generation_params_specificity_ordering():
    # The "-turbo" / "-schnell" entries must win over their broader siblings; a
    # reorder that broke this would silently mis-default.
    assert default_generation_params("x/Z-Image-Turbo") != default_generation_params("x/Z-Image")
    assert default_generation_params("x/FLUX.1-schnell") != default_generation_params(
        "x/FLUX.1-dev"
    )


def test_default_generation_params_falls_back_to_base_repo():
    # A local-path load: repo_id is a filesystem path that names no model, so the
    # resolved base repo is what identifies it (and distinguishes dev from schnell).
    assert default_generation_params("/models/my-ckpt", "black-forest-labs/FLUX.1-dev") == (28, 3.5)
    assert default_generation_params("/models/my-ckpt", "black-forest-labs/FLUX.1-schnell") == (
        4,
        0.0,
    )
    assert default_generation_params("/models/my-ckpt", "Qwen/Qwen-Image") == (20, 4.0)
    # repo_id wins when it already names the model; base repo is only a fallback.
    assert default_generation_params("unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image") == (9, 0.0)
    # Nothing identifiable -> fallback; None identifiers are skipped.
    assert default_generation_params(None, None) == (9, 0.0)
    assert default_generation_params("/models/x", None) == (9, 0.0)


@pytest.mark.parametrize(
    "size, expected",
    [
        ("auto", (1024, 1024)),
        ("", (1024, 1024)),
        ("AUTO", (1024, 1024)),
        ("512x512", (512, 512)),
        ("512x256", (512, 256)),
        ("1792x1024", (1792, 1024)),  # dall-e-3 named size: must pass the bounds
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
    def __init__(
        self,
        loaded = True,
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        base_repo = None,
        generate_error = None,
        unload_on_generate = False,
        native_seeds = False,
    ) -> None:
        self._loaded = loaded
        self._repo_id = repo_id
        self._base_repo = base_repo
        # Model the native sd.cpp engine, which returns a distinct seed per image.
        self._native_seeds = native_seeds
        # When set, generate() raises this; unload_on_generate flips is_loaded off
        # first, to model the eviction/unload race vs an in-pipeline failure (OOM).
        self._generate_error = generate_error
        self._unload_on_generate = unload_on_generate
        self.calls = []

    @property
    def is_loaded(self):
        return self._loaded

    def status(self):
        return {
            "loaded": self._loaded,
            "repo_id": self._repo_id if self._loaded else None,
            "family": "z-image" if self._loaded else None,
            "base_repo": self._base_repo if self._loaded else None,
            "device": "cpu",
            "dtype": "float32",
            "cpu_offload": False,
        }

    def generate(
        self,
        *,
        prompt,
        width,
        height,
        steps,
        guidance,
        batch_size = 1,
    ):
        if not self._loaded:
            raise RuntimeError("No diffusion model is loaded.")
        if self._generate_error is not None:
            if self._unload_on_generate:
                self._loaded = False
            raise self._generate_error
        self.calls.append(
            dict(
                prompt = prompt,
                width = width,
                height = height,
                steps = steps,
                guidance = guidance,
                batch_size = batch_size,
            )
        )
        out = {
            "images": [object() for _ in range(batch_size)],
            "seed": 4242,
            "repo_id": self._repo_id,
        }
        if self._native_seeds:
            out["seeds"] = [4242 + i for i in range(batch_size)]
        return out


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
    assert "url" in item and "b64_json" not in item  # exclude_none drops the unused key
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


def test_local_load_uses_base_repo_for_defaults(monkeypatch):
    # repo_id is a local path that names no model; base_repo identifies FLUX.1-dev,
    # so the route must pick 28 steps / 3.5 guidance, not the 9/0 fallback.
    backend = _FakeBackend(repo_id = "/models/my-flux", base_repo = "black-forest-labs/FLUX.1-dev")
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    assert backend.calls[0]["steps"] == 28 and backend.calls[0]["guidance"] == 3.5


def test_pipeline_runtime_error_is_sanitized_500(monkeypatch):
    # A RuntimeError raised inside the pipeline while the model stays loaded (e.g.
    # CUDA OOM, a RuntimeError subclass) must be a sanitized 500, not a 503 that
    # echoes the raw exception text.
    oom = RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (GPU 0; 47.5 GiB total)")
    backend = _FakeBackend(generate_error = oom)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "Image generation failed."
    assert "CUDA" not in resp.text  # raw exception text must not leak


def test_unload_race_returns_503(monkeypatch):
    # The model is evicted/unloaded between the readiness check and the call: the
    # RuntimeError with is_loaded now False is the one case that maps to 503.
    backend = _FakeBackend(
        generate_error = RuntimeError("No diffusion model is loaded."),
        unload_on_generate = True,
    )
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "api_error"
    # The 503 carries the fixed sanitized message, not the raw exception text.
    assert err["message"] == "No image model loaded. Load an image model first."


def test_non_runtime_pipeline_error_is_500(monkeypatch):
    # A non-RuntimeError from the pipeline (the model stays loaded) must not route
    # to the 503 branch (which is gated on isinstance RuntimeError) -> sanitized 500.
    backend = _FakeBackend(generate_error = ValueError("bad tensor shape"))
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 500
    assert "shape" not in resp.text


def test_n_maps_to_batch(client):
    resp = _post(client, {"prompt": "p", "size": "256x256", "n": 3})
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 3
    assert client.backend.calls[0]["batch_size"] == 3


def test_batch_persists_batch_size(monkeypatch):
    # n>1 must persist batch_size in each gallery record so the Studio restore path
    # can replay a batch_index>0 sibling (which shares the batch's single seed).
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256", "n": 3})
    assert resp.status_code == 200
    records = sorted(store.values(), key = lambda r: r["batch_index"])
    assert [r["batch_index"] for r in records] == [0, 1, 2]
    assert all(r["batch_size"] == 3 for r in records)


def test_uses_active_engine_not_diffusers_singleton(monkeypatch):
    # On a no-GPU host the loaded model lives behind the native sd_cpp engine, not the
    # diffusers singleton. The route must query get_active_diffusion_engine (like
    # /images/generate) or it 503s a model that is loaded and usable.
    active = _FakeBackend(loaded = True)  # the active (e.g. sd_cpp) engine, loaded
    idle_diffusers = _FakeBackend(loaded = False)  # diffusers singleton, empty
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: active)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: idle_diffusers)
    cli, store, _save = _make_client(active)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    assert len(active.calls) == 1  # the active engine did the work, not the idle singleton


def test_native_batch_persists_per_image_seed(monkeypatch):
    # The native sd.cpp engine returns a distinct seed per image (base+index) in
    # "seeds"; each gallery record must store its own seed (like /images/generate),
    # not the shared base, or a restored batch_index>0 image shows the wrong seed.
    backend = _FakeBackend(native_seeds = True)
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256", "n": 3})
    assert resp.status_code == 200
    records = sorted(store.values(), key = lambda r: r["batch_index"])
    assert [r["seed"] for r in records] == [4242, 4243, 4244]


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
        ({"size": "256x256"}, "prompt"),  # missing prompt
        ({"prompt": "", "size": "256x256"}, "prompt"),  # empty prompt
        ({"prompt": "p", "size": "300x300"}, "size"),  # not multiple of 16
        ({"prompt": "p", "size": "abc"}, "size"),  # unparseable
        ({"prompt": "p", "stream": True}, "stream"),  # streaming unsupported
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
