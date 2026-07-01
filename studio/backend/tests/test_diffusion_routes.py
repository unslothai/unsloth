# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the diffusion image routes.

The diffusion backend is replaced with a lightweight fake, so these exercise the
route wiring, validation (422), error mapping, and response shapes without torch,
diffusers, weights, or a GPU.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.diffusion as diffusion_module
import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.image_gallery as gallery_module
from auth.authentication import get_current_subject
from routes.inference import studio_router


class _FakeBackend:
    def __init__(self) -> None:
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def validate_load_request(
        self,
        model_path,
        *,
        gguf_filename = None,
        family_override = None,
    ):
        # Mirror the real backend's cheap validation so the route's
        # validate-before-evict ordering is exercised.
        from core.inference.diffusion_families import detect_family

        if not gguf_filename:
            raise ValueError("gguf_filename is required.")
        fam = detect_family(model_path, family_override)
        if fam is None:
            raise ValueError(f"Could not infer a diffusion family for '{model_path}'.")
        return fam

    def begin_load(self, model_path, **kwargs):
        # The real backend loads on a thread; the fake completes instantly.
        self.loaded = True
        self.last_load_kwargs = dict(kwargs)
        return {
            "loaded": True,
            "repo_id": model_path,
            "family": "z-image",
            "base_repo": kwargs.get("base_repo") or "base/repo",
            "device": "cpu",
            "dtype": "float32",
            "cpu_offload": False,
            "offload_policy": "none",
            "vae_tiling": False,
            "memory_mode": kwargs.get("memory_mode") or "auto",
        }

    def load_progress(self):
        return {
            "phase": "ready" if self.loaded else None,
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "fraction": 1.0 if self.loaded else 0.0,
            "error": None,
        }

    def generate(
        self,
        *,
        seed = None,
        batch_size = 1,
        **kwargs,
    ):
        if not self.loaded:
            raise RuntimeError("No diffusion model is loaded.")
        # The real backend returns the PIL images; the route persists them. The
        # fake returns sentinels since image_gallery is stubbed in the fixture.
        return {
            "images": [object() for _ in range(batch_size)],
            "seed": seed if seed is not None else 4242,
            "repo_id": "x/z-image",
        }

    def unload(self):
        self.loaded = False
        return _unloaded_status()

    def status(self):
        return {**_unloaded_status(), "loaded": self.loaded}


def _unloaded_status():
    return {
        "loaded": False,
        "repo_id": None,
        "family": None,
        "base_repo": None,
        "device": None,
        "dtype": None,
        "cpu_offload": False,
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    # Isolate from the real GPU arbiter: reset ownership and stub the evictors so
    # the load route's acquire_for() never touches live backend singletons.
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    # In-memory gallery backed by tmp files, so routes exercise persistence wiring
    # without PIL/real disk under studio_root.
    store: dict[str, dict] = {}

    def _save(image, meta):
        image_id = f"img{len(store)}"
        (tmp_path / f"{image_id}.png").write_bytes(b"PNG")
        record = {**meta, "id": image_id, "url": f"/api/inference/images/gallery/{image_id}/file"}
        store[image_id] = record
        return record

    def _clear():
        n = len(store)
        store.clear()
        return n

    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)

    def _list_images(limit = None, offset = 0):
        ordered = sorted(store.values(), key = lambda r: r.get("created_at", 0.0), reverse = True)
        return ordered[offset:] if limit is None else ordered[offset : offset + limit]

    monkeypatch.setattr(gallery_module, "list_images", _list_images)
    monkeypatch.setattr(
        gallery_module,
        "image_path",
        lambda i: (tmp_path / f"{i}.png") if i in store else None,
    )
    monkeypatch.setattr(gallery_module, "delete", lambda i: store.pop(i, None) is not None)
    monkeypatch.setattr(gallery_module, "clear", _clear)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_load_generate_status_unload_roundtrip(client):
    loaded = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
            "base_repo": "base/repo",
        },
    )
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["loaded"] is True and body["family"] == "z-image"

    assert client.get("/api/inference/images/status").json()["loaded"] is True

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200
    # One persisted record carrying the full recipe back.
    images = gen.json()["images"]
    assert len(images) == 1
    img = images[0]
    assert img["seed"] == 7 and img["prompt"] == "a sloth" and img["id"]

    # The image is now listable, fetchable, and deletable.
    listed = client.get("/api/inference/images/gallery").json()["images"]
    assert [i["id"] for i in listed] == [img["id"]]
    assert client.get(img["url"]).status_code == 200
    assert client.delete(img["url"].removesuffix("/file")).status_code == 200
    assert client.get("/api/inference/images/gallery").json()["images"] == []

    unloaded = client.post("/api/inference/images/unload")
    assert unloaded.status_code == 200 and unloaded.json()["loaded"] is False
    assert client.get("/api/inference/images/status").json()["loaded"] is False


def test_generate_batch_size_persists_each_image(client):
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "batch_size": 3, "seed": 5},
    )
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert len(images) == 3
    assert all(i["seed"] == 5 for i in images)  # the batch shares one seed
    assert len({i["id"] for i in images}) == 3  # but each is a distinct record
    assert len(client.get("/api/inference/images/gallery").json()["images"]) == 3


def test_gallery_pagination(client):
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    client.post("/api/inference/images/generate", json = {"prompt": "p", "batch_size": 5, "seed": 1})
    page1 = client.get("/api/inference/images/gallery?limit=2&offset=0").json()
    assert len(page1["images"]) == 2 and page1["has_more"] is True
    last = client.get("/api/inference/images/gallery?limit=2&offset=4").json()
    assert len(last["images"]) == 1 and last["has_more"] is False


def test_generate_rejects_non_multiple_of_16(client):
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    # Odd, and a multiple of 8 that isn't a multiple of 16: both rejected, since
    # Z-Image requires dimensions divisible by 16.
    for bad in (1001, 1000):
        resp = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": bad})
        assert resp.status_code == 422, bad
    # A multiple of 16 is accepted.
    ok = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": 1024})
    assert ok.status_code == 200


def test_load_requires_gguf_filename(client):
    # gguf_filename is now mandatory — a load without it is a 422.
    resp = client.post("/api/inference/images/load", json = {"model_path": "x/z-image"})
    assert resp.status_code == 422


def test_generate_without_load_returns_409(client):
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409


def test_generate_pipeline_error_returns_sanitized_500(client, monkeypatch):
    # A loaded model that fails mid-pipeline (CUDA OOM, a RuntimeError) is a server
    # failure: 500 with a generic message, not a 409 echoing the raw exception.
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _oom(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (24.00 GiB total)")

    monkeypatch.setattr(backend, "generate", _oom)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Image generation failed."
    assert "CUDA" not in resp.json()["detail"]


def test_load_unknown_family_returns_400(client, monkeypatch):
    def _raise(*a, **k):
        raise ValueError("'x/y' isn't a supported image-generation model. Supported: Z-Image.")

    backend = _FakeBackend()
    # Validation runs in the pre-flight (before the GPU is taken), so that is
    # where an unsupported model is rejected now.
    backend.validate_load_request = _raise
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/y", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 400
    assert "isn't a supported image-generation model" in resp.json()["detail"]


def test_load_validation_failure_does_not_evict_chat(client, monkeypatch):
    # A rejected image-model pick must not tear down the user's loaded chat model:
    # validation runs before acquire_for, so chat keeps the GPU on a 400.
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))

    backend = _FakeBackend()

    def _raise(*a, **k):
        raise ValueError("'x/y' isn't a supported image-generation model.")

    backend.validate_load_request = _raise
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/y", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 400
    assert evicted == []  # chat backend was never evicted
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_load_refused_during_training_does_not_evict_chat(client, monkeypatch):
    # An image load while training is active is refused (409) before the GPU is
    # taken, so the training run and the loaded chat model are both untouched.
    import core.training as core_training

    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))

    class _Training:
        def is_training_active(self):
            return True

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Training())

    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()
    assert evicted == []  # chat backend was never evicted
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_load_progress_route(client):
    # Before load: idle.
    idle = client.get("/api/inference/images/load-progress")
    assert idle.status_code == 200 and idle.json()["phase"] is None
    # After load: the fake reports ready.
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    ready = client.get("/api/inference/images/load-progress")
    assert ready.json()["phase"] == "ready"


def test_routes_require_auth():
    # No dependency override: the auth dependency must reject the request.
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    unauth = TestClient(app)
    assert unauth.get("/api/inference/images/status").status_code in (401, 403)


def test_invalid_family_returns_400_without_evicting_chat(client):
    # An undetectable family fails validation BEFORE the GPU handoff, so the
    # arbiter is never acquired and a loaded chat model would not be evicted.
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/y", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 400
    assert "family" in resp.json()["detail"]
    assert gpu_arbiter._owner is None


def test_validate_filenotfound_maps_to_400_without_eviction(client, monkeypatch):
    def _raise_fnf(*a, **k):
        raise FileNotFoundError("'q.gguf' not found under /models/x.")

    backend = _FakeBackend()
    backend.validate_load_request = _raise_fnf
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "/models/x", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 400
    assert gpu_arbiter._owner is None


def test_memory_mode_threads_through_to_backend(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "memory_mode": "low_vram"},
    )
    assert resp.status_code == 200
    assert resp.json()["memory_mode"] == "low_vram"
    assert backend.last_load_kwargs.get("memory_mode") == "low_vram"


def test_transformer_quant_threads_through_to_backend(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "transformer_quant": "auto"},
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_quant") == "auto"


def test_transformer_quant_fast_accum_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "transformer_quant": "fp8",
            "transformer_quant_fast_accum": False,
        },
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_quant_fast_accum") is False


def test_invalid_transformer_quant_returns_422_without_eviction(client):
    # An unsupported transformer_quant is rejected by the request schema (Literal), so
    # the GPU is never acquired and no chat model is evicted.
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "transformer_quant": "int2"},
    )
    assert resp.status_code == 422
    assert gpu_arbiter._owner is None


def test_invalid_memory_mode_returns_422_without_eviction(client):
    # An unsupported memory_mode is rejected by the request schema (Literal), so the
    # GPU is never acquired and no chat model is evicted.
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "memory_mode": "ultra"},
    )
    assert resp.status_code == 422
    assert gpu_arbiter._owner is None


def test_in_progress_returns_409_after_validation_passes(client, monkeypatch):
    def _busy(*a, **k):
        raise RuntimeError("A diffusion load is already in progress.")

    backend = _FakeBackend()
    backend.begin_load = _busy
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    # Validation passed first, so the GPU WAS acquired before begin_load reported busy.
    assert gpu_arbiter._owner == gpu_arbiter.DIFFUSION
