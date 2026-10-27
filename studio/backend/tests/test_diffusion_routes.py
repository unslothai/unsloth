# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion image routes: wiring, 422 validation and error mapping, against a fake backend."""

from __future__ import annotations

import types


def _cuda_target():
    """Set the real bf16 dtype and supports_default_torch_compile, or the stub looks uncompilable."""
    import torch
    return types.SimpleNamespace(
        device = "cuda",
        dtype = torch.bfloat16,
        _cc = (10, 0),
        supports_default_torch_compile = True,
    )


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.diffusion as diffusion_module
import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.image_gallery as gallery_module
from auth.authentication import authenticated_via_api_key, get_current_subject
import routes.inference as inference_routes
from routes.inference import studio_router


class _FakeBackend:
    def __init__(self) -> None:
        self.loaded = False
        # in-flight load repo ids: the unload route keeps DIFFUSION ownership during a concurrent load
        self.loading: tuple = ()
        self.active_generate_cancel = None

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def loading_repo_ids(self) -> tuple:
        return tuple(self.loading)

    def validate_load_request(
        self,
        model_path,
        *,
        gguf_filename = None,
        family_override = None,
        model_kind = None,
        base_repo = None,
    ):
        # mirror the real backend's cheap validation so validate-before-evict ordering is exercised
        from core.inference.diffusion import resolve_model_kind
        from core.inference.diffusion_families import detect_family

        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("a single-file checkpoint name is required.")
        if kind != "gguf" and not model_path.lower().startswith("unsloth/"):
            raise ValueError(
                f"Non-GGUF diffusion loads are restricted to unsloth/* repos; got '{model_path}'."
            )
        if base_repo and base_repo.strip() and not base_repo.lower().startswith("unsloth/"):
            raise ValueError(
                f"base_repo is restricted to unsloth/* repos (or a local path); got '{base_repo}'."
            )
        fam = detect_family(model_path, family_override)
        if fam is None:
            raise ValueError(f"Could not infer a diffusion family for '{model_path}'.")
        return fam

    def preflight_base_access(self, model_path, fam, **kwargs):
        return None

    def assert_precision_available(self, fam, **kwargs):
        # the fake clears every request; tests that care re-patch this to raise
        self.last_precision_kwargs = dict(kwargs)
        return None

    def download_plan(self, model_path, **kwargs):
        # records its args so a test can prove the precision refusal ran before the plan
        self.last_plan_kwargs = dict(kwargs)
        return {"entries": [], "total_bytes": 0, "incompatible_reason": None}

    def begin_load(self, model_path, **kwargs):
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
        prompts = None,
        seeds = None,
        **kwargs,
    ):
        if not self.loaded:
            raise RuntimeError("No diffusion model is loaded.")
        if prompts is not None or seeds is not None:
            # the list sets image count and seeds; batch_size is only a per-forward cap
            base = seeds[0] if seeds else (seed if seed is not None else 4242)
            count = len(prompts) if prompts is not None else len(seeds)
            per_image = seeds if seeds is not None else [base + i for i in range(count)]
            return {
                "images": [object() for _ in range(count)],
                "seed": base,
                "seeds": list(per_image),
                "repo_id": "x/z-image",
            }
        return {
            "images": [object() for _ in range(batch_size)],
            "seed": seed if seed is not None else 4242,
            "repo_id": "x/z-image",
            "workflow": (
                "inpaint"
                if kwargs.get("mask_image")
                else ("img2img" if kwargs.get("init_image") else "txt2img")
            ),
        }

    def generate_progress(self):
        return {"active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None}

    def cancel_generate(self):
        cancel = self.active_generate_cancel
        if cancel is None:
            return False
        cancel.set()
        return True

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
    # neutralise the engine router so routes drive this fake on any host and never install natively
    import core.inference.diffusion_engine_router as engine_router

    # delegate so per-test re-patches still flow through the routes
    monkeypatch.setattr(
        engine_router,
        "select_and_activate_engine",
        lambda fam, **kw: diffusion_module.get_diffusion_backend(),
    )
    monkeypatch.setattr(
        engine_router,
        "get_active_diffusion_engine",
        lambda: diffusion_module.get_diffusion_backend(),
    )
    # left real, predict_engine reaches host binaries and the live sd.cpp backend on GPU-less runners
    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: "diffusers")
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
    # stub the arbiter so acquire_for() never touches live singletons
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

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

    def _list_images(
        limit = None,
        offset = 0,
        *,
        valid = None,
        archived = False,
    ):
        # model the real shelf split and pinned-first order, or a wrong page would still pass
        ordered = [r for r in store.values() if bool(r.get("archived")) == archived]
        ordered.sort(key = lambda r: (bool(r.get("pinned")), r.get("created_at", 0.0)), reverse = True)
        if valid is not None:
            ordered = [r for r in ordered if valid(r)]
        return ordered[offset:] if limit is None else ordered[offset : offset + limit]

    monkeypatch.setattr(gallery_module, "list_images", _list_images)
    monkeypatch.setattr(
        gallery_module,
        "image_path",
        lambda i: (tmp_path / f"{i}.png") if i in store else None,
    )
    monkeypatch.setattr(
        gallery_module,
        "owned_image_path",
        lambda i: (tmp_path / f"{i}.png") if i in store else None,
    )
    monkeypatch.setattr(gallery_module, "delete", lambda i: store.pop(i, None) is not None)
    monkeypatch.setattr(gallery_module, "clear", _clear)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    # browser session: status routes redact host paths for an API-key caller
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _post_load(client, **body):
    """POST the image-model load route with ``body`` as the request payload."""
    return client.post("/api/inference/images/load", json = body)


def _post_generate(client, **body):
    """POST the image-generation route with ``body`` as the request payload."""
    return client.post("/api/inference/images/generate", json = body)


def _post_download_plan(client, **body):
    """POST the image download-plan route with ``body`` as the request payload."""
    return client.post("/api/inference/images/download-plan", json = body)


def test_load_generate_status_unload_roundtrip(client):
    loaded = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_S.gguf",
        base_repo = "unsloth/Z-Image-base",
    )
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["loaded"] is True and body["family"] == "z-image"

    assert client.get("/api/inference/images/status").json()["loaded"] is True

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200
    images = gen.json()["images"]
    assert len(images) == 1
    img = images[0]
    assert img["seed"] == 7 and img["prompt"] == "a sloth" and img["id"]

    listed = client.get("/api/inference/images/gallery").json()["images"]
    assert [i["id"] for i in listed] == [img["id"]]
    assert client.get(img["url"]).status_code == 200
    assert client.delete(img["url"].removesuffix("/file")).status_code == 200
    assert client.get("/api/inference/images/gallery").json()["images"] == []

    unloaded = client.post("/api/inference/images/unload")
    assert unloaded.status_code == 200 and unloaded.json()["loaded"] is False
    assert client.get("/api/inference/images/status").json()["loaded"] is False


def test_gallery_serve_refuses_unowned_id(client):
    # a guessed stem must 404 via the ownership guard, not stream foreign bytes
    assert client.get("/api/inference/images/gallery/family-photo/file").status_code == 404
    assert (
        client.get("/api/inference/images/gallery/family-photo/file?thumb=256").status_code == 404
    )


def test_gallery_serve_thumb_is_a_webp_and_falls_back_to_the_png(client, monkeypatch):
    _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_S.gguf",
        base_repo = "unsloth/Z-Image-base",
    )
    url = _post_generate(client, prompt = "a sloth", seed = 7).json()["images"][0]["url"]
    sizes = []

    def _thumbnail(path, size):
        sizes.append(size)
        return b"WEBP"

    monkeypatch.setattr(gallery_module, "thumbnail", _thumbnail)
    thumb = client.get(f"{url}?thumb=5000")
    assert thumb.headers["content-type"] == "image/webp" and thumb.content == b"WEBP"
    assert sizes == [1024]
    assert client.get(f"{url}?thumb=0").headers["content-type"] == "image/webp"
    assert sizes == [1024, 32]
    original = client.get(url)
    assert original.headers["content-type"] == "image/png" and original.content == b"PNG"

    def _undecodable(path, size):
        raise OSError("truncated PNG")

    monkeypatch.setattr(gallery_module, "thumbnail", _undecodable)
    fallback = client.get(f"{url}?thumb=256")
    assert fallback.headers["content-type"] == "image/png" and fallback.content == b"PNG"


def test_generate_holds_progress_active_during_persist(client, monkeypatch):
    # generate-progress must stay active while the gallery record is still being written
    _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_S.gguf",
        base_repo = "unsloth/Z-Image-base",
    )

    assert client.get("/api/inference/images/generate-progress").json()["active"] is False

    seen = {}
    real_save = gallery_module.save

    def _probe_save(image, meta):
        seen["during"] = inference_routes._diffusion_persist_active
        return real_save(image, meta)

    monkeypatch.setattr(gallery_module, "save", _probe_save)

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200
    assert seen["during"] >= 1
    assert inference_routes._diffusion_persist_active == 0
    assert client.get("/api/inference/images/generate-progress").json()["active"] is False


def test_generate_progress_route_logs_backend_snapshot(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        inference_routes,
        "log_media_generation_progress",
        lambda media, progress: seen.append((media, dict(progress))),
    )

    response = client.get("/api/inference/images/generate-progress")

    assert response.status_code == 200
    assert seen == [
        (
            "image",
            {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
            },
        )
    ]


def test_generate_resets_the_progress_stream_before_the_run(client, monkeypatch):
    # milestones are keyed on the previous poll, so image routes must rearm before generate()
    client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
            "base_repo": "unsloth/Z-Image-base",
        },
    )
    order = []
    monkeypatch.setattr(
        inference_routes,
        "reset_media_generation_progress",
        lambda media: order.append(("reset", media)),
    )
    real_save = gallery_module.save

    def _probe_save(image, meta):
        order.append(("generated", "image"))
        return real_save(image, meta)

    monkeypatch.setattr(gallery_module, "save", _probe_save)

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})

    assert gen.status_code == 200
    assert order == [("reset", "image"), ("generated", "image")]


def test_load_rejects_untrusted_base_repo(client):
    # an untrusted remote base_repo must be refused so clients cannot make the server fetch it
    r = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_S.gguf",
        base_repo = "evil/companions",
    )
    assert r.status_code == 400
    assert "base_repo" in r.json()["detail"]
    assert client.get("/api/inference/images/status").json()["loaded"] is False


def test_unload_keeps_ownership_when_a_model_is_still_resident(client, monkeypatch):
    # releasing DIFFUSION over a concurrent load lets a later chat load skip eviction and OOM
    backend = diffusion_module.get_diffusion_backend()
    gpu_arbiter._owner = gpu_arbiter.DIFFUSION

    backend.loaded = True
    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": True})
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() == gpu_arbiter.DIFFUSION

    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": False})
    backend.loaded = False
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() is None


def test_idle_unload_frees_the_pipeline_and_the_user_can_reload(client, monkeypatch):
    # the idle unload must stash nothing, so a later reload takes the ordinary path
    import asyncio
    import time

    import core.inference.media_keepwarm as media_keepwarm
    import utils.openai_auto_switch_settings as auto_switch_settings

    monkeypatch.setattr(auto_switch_settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    tracker = media_keepwarm._TRACKERS[gpu_arbiter.DIFFUSION]
    monkeypatch.setattr(tracker, "seen", None)
    monkeypatch.setattr(tracker, "_inflight", 0)
    monkeypatch.setattr(tracker, "_pending", 0)
    load = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}

    assert client.post("/api/inference/images/load", json = load).json()["loaded"] is True
    asyncio.run(media_keepwarm.idle_unload_step())
    assert client.get("/api/inference/images/status").json()["loaded"] is True

    tracker._last_active = time.monotonic() - 3600
    asyncio.run(media_keepwarm.idle_unload_step())
    assert client.get("/api/inference/images/status").json()["loaded"] is False
    assert gpu_arbiter.current_owner() is None

    assert client.post("/api/inference/images/load", json = load).json()["loaded"] is True
    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200 and gen.json()["images"][0]["seed"] == 7


def test_unload_keeps_ownership_when_a_load_is_in_flight(client, monkeypatch):
    # a concurrent load is not is_loaded yet, so keep ownership on the in-flight state alone
    backend = diffusion_module.get_diffusion_backend()
    gpu_arbiter._owner = gpu_arbiter.DIFFUSION

    backend.loaded = False
    backend.loading = ("unsloth/z-image-turbo",)
    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": False})
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() == gpu_arbiter.DIFFUSION

    backend.loading = ()


def test_generate_batch_size_persists_each_image(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    resp = _post_generate(client, prompt = "p", batch_size = 3, seed = 5)
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert len(images) == 3
    assert all(i["seed"] == 5 for i in images)
    assert len({i["id"] for i in images}) == 3
    assert len(client.get("/api/inference/images/gallery").json()["images"]) == 3


def test_generate_seed_list_records_replay_from_each_own_seed(client):
    # a seeds list gives each image its own seed, so restore must prefer batch_seed
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    resp = _post_generate(client, prompt = "p", seeds = [5, 99])
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert [i["seed"] for i in images] == [5, 99]
    assert [i["batch_seed"] for i in images] == [5, 99]
    assert [i["batch_size"] for i in images] == [1, 1]


def test_generate_prompt_list_records_each_prompt_and_seed(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    resp = _post_generate(client, prompt = "unused", prompts = ["a cat", "a dog"], seed = 10)
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert [i["prompt"] for i in images] == ["a cat", "a dog"]
    assert [i["seed"] for i in images] == [10, 11]
    assert [i["batch_seed"] for i in images] == [10, 11]
    assert [i["batch_size"] for i in images] == [1, 1]


def test_generate_legacy_batch_still_records_the_base_seed_and_size(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    resp = _post_generate(client, prompt = "p", batch_size = 3, seed = 5)
    images = resp.json()["images"]
    assert all(i["batch_seed"] == 5 for i in images)
    assert all(i["batch_size"] == 3 for i in images)
    assert [i["batch_index"] for i in images] == [0, 1, 2]


def test_generate_request_rejects_zero_denoise_strength():
    # strength 0 leaves zero denoising steps (FLUX/Qwen/Z-Image raise, SDXL crashes): 422
    import pydantic

    from models.inference import DiffusionGenerateRequest

    with pytest.raises(pydantic.ValidationError):
        DiffusionGenerateRequest(prompt = "x", strength = 0.0)
    assert DiffusionGenerateRequest(prompt = "x", strength = 0.1).strength == 0.1
    assert DiffusionGenerateRequest(prompt = "x", strength = 1.0).strength == 1.0
    assert DiffusionGenerateRequest(prompt = "x").strength is None


def test_gallery_pagination(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    client.post("/api/inference/images/generate", json = {"prompt": "p", "batch_size": 5, "seed": 1})
    page1 = client.get("/api/inference/images/gallery?limit=2&offset=0").json()
    assert len(page1["images"]) == 2 and page1["has_more"] is True
    last = client.get("/api/inference/images/gallery?limit=2&offset=4").json()
    assert len(last["images"]) == 1 and last["has_more"] is False


def test_generate_rejects_non_multiple_of_16(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    # Z-Image requires dimensions divisible by 16
    for bad in (1001, 1000):
        resp = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": bad})
        assert resp.status_code == 422, bad
    ok = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": 1024})
    assert ok.status_code == 200


def test_generate_rejects_batch_seed_past_json_safe_range(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    # per-image seeds past 2**53 - 1 are not JSON-safe
    over = _post_generate(client, prompt = "p", seed = 2**53 - 1, batch_size = 2)
    assert over.status_code == 422
    ok = _post_generate(client, prompt = "p", seed = 2**53 - 2, batch_size = 2)
    assert ok.status_code == 200


def test_non_gguf_load_restricted_to_unsloth(client):
    # no gguf_filename means a full-pipeline load, gated to unsloth/*
    resp = client.post("/api/inference/images/load", json = {"model_path": "x/z-image"})
    assert resp.status_code == 400
    assert "unsloth" in resp.json()["detail"].lower()


def test_a_too_old_diffusers_is_a_400_on_both_load_and_download_plan(client, monkeypatch):
    # an unbuildable family must be a 400 with its message, not a 409 or a bare 500
    import sys
    import types

    from core.inference.diffusion_families import assert_pipeline_class_available

    backend = diffusion_module.get_diffusion_backend()

    def _refuse(model_path, **kwargs):
        assert_pipeline_class_available("Flux2KleinPipeline", "flux.2-klein")

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(__version__ = "0.36.0"))
    monkeypatch.setattr(backend, "validate_load_request", _refuse)
    body = {
        "model_path": "unsloth/FLUX.2-klein-4B-GGUF",
        "gguf_filename": "flux2-klein-4b-Q4_0.gguf",
        "model_kind": "gguf",
    }

    load = client.post("/api/inference/images/load", json = body)
    assert load.status_code == 400
    assert "Flux2KleinPipeline" in load.json()["detail"]

    plan = client.post("/api/inference/images/download-plan", json = body)
    assert plan.status_code == 400
    assert "Flux2KleinPipeline" in plan.json()["detail"]


def test_pipeline_load_allowed_for_unsloth_repo(client):
    resp = _post_load(client, model_path = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit")
    assert resp.status_code == 200
    backend = diffusion_module.get_diffusion_backend()
    assert backend.last_load_kwargs["model_kind"] == "pipeline"
    assert backend.last_load_kwargs.get("gguf_filename") is None


def test_generate_without_load_returns_409(client):
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409


def test_generate_pipeline_error_returns_sanitized_500(client, monkeypatch):
    # mid-pipeline failures are a 500 with fixed text: engine text can leak local paths and argv
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _oom(**kwargs):
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 20.00 GiB at /home/u/models/x.safetensors"
        )

    monkeypatch.setattr(backend, "generate", _oom)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail.startswith("Image generation failed.")
    assert "ran out of memory" in detail
    for leak in ("CUDA", "20.00 GiB", "/home/u", "safetensors"):
        assert leak not in detail


def test_generate_native_process_death_names_the_engine_not_its_output(client, monkeypatch):
    # Metal hosts: the native renderer aborts inside its text encoder
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _abort(**kwargs):
        raise RuntimeError(
            "sd-server connection lost during img_gen poll (process exited, code -6)\n"
            "Last output:\n0 sd-server ggml_abort + 156 at /Users/me/.cache/sd-cli"
        )

    monkeypatch.setattr(backend, "generate", _abort)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "native image renderer stopped" in detail
    for leak in ("ggml_abort", "/Users/me", "img_gen", "code -6"):
        assert leak not in detail


def test_generate_nan_image_names_the_overflow(client, monkeypatch):
    from core.inference.diffusion_postprocess import NAN_IMAGE_MESSAGE

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _nan(**kwargs):
        raise RuntimeError(NAN_IMAGE_MESSAGE)

    monkeypatch.setattr(backend, "generate", _nan)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    assert "overflowed at this resolution" in resp.json()["detail"]


def test_generate_execution_error_with_cancelled_substring_is_sanitized_500(client, monkeypatch):
    # a raw tail merely containing "cancelled" must not misroute to 409
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _fail(**kwargs):
        raise RuntimeError("sd-cli exited 1. Last output:\nop cancelled at /home/u/models/x.gguf")

    monkeypatch.setattr(backend, "generate", _fail)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail.startswith("Image generation failed.")
    assert "cancelled" not in detail and "models" not in detail and "/home/u" not in detail


def test_generate_memory_refusal_is_a_tagged_400_and_allow_oversized_reaches_the_backend(
    client, monkeypatch
):
    from core.inference.diffusion_memory import (
        IMAGE_REFUSAL_HEADER,
        IMAGE_REFUSAL_MEMORY_ESTIMATE,
        ImageActivationShortfallError,
    )

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    seen = []

    def _guarded(**kwargs):
        seen.append(kwargs.get("allow_oversized"))
        if not kwargs.get("allow_oversized"):
            raise ImageActivationShortfallError("Generating at 2048x2048 needs about 34.00 GB")
        return {"images": [], "seed": 1, "seeds": []}

    monkeypatch.setattr(backend, "generate", _guarded)
    refused = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert refused.status_code == 400
    assert refused.headers.get(IMAGE_REFUSAL_HEADER) == IMAGE_REFUSAL_MEMORY_ESTIMATE
    assert "2048x2048" in refused.json()["detail"]
    retried = client.post(
        "/api/inference/images/generate", json = {"prompt": "p", "allow_oversized": True}
    )
    assert retried.status_code == 200
    assert seen == [False, True]

    def _plain(**kwargs):
        raise ValueError("width and height are required for this workflow.")

    monkeypatch.setattr(backend, "generate", _plain)
    other = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert other.status_code == 400
    assert IMAGE_REFUSAL_HEADER not in other.headers


def test_generate_unified_memory_load_refusal_is_structured_not_500(client, monkeypatch):
    from core.inference.diffusion_memory import (
        IMAGE_REFUSAL_HEADER,
        UNIFIED_OVERSIZE_REFUSAL_CODE,
        UnifiedMemoryShortfallError,
    )

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _refused(**kwargs):
        raise UnifiedMemoryShortfallError("Could not load the diffusers engine: needs about 30 GB")

    monkeypatch.setattr(backend, "generate", _refused)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 400
    assert resp.headers.get(IMAGE_REFUSAL_HEADER) == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert "30 GB" in resp.json()["detail"]


def test_generate_returns_the_load_refusal_of_a_refused_mlx_fallback(client, monkeypatch):
    from fastapi import HTTPException
    from core.inference.gpu_arbiter import MULTI_MODEL_REFUSAL_HEADERS
    from core.inference.mlx_diffusion_routing import FallbackLoadRefusedError

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _refused(**kwargs):
        raise FallbackLoadRefusedError(
            HTTPException(
                status_code = 409,
                detail = "Unload the chat model first.",
                headers = dict(MULTI_MODEL_REFUSAL_HEADERS),
            )
        )

    monkeypatch.setattr(backend, "generate", _refused)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Unload the chat model first."
    for key, value in MULTI_MODEL_REFUSAL_HEADERS.items():
        assert resp.headers.get(key) == value


def test_generate_user_cancellation_returns_409(client, monkeypatch):
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _cancel(**kwargs):
        raise RuntimeError("Diffusion generation was cancelled.")

    monkeypatch.setattr(backend, "generate", _cancel)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Diffusion generation was cancelled."


def test_load_unknown_family_returns_400(client, monkeypatch):
    def _raise(*a, **k):
        raise ValueError("'x/y' isn't a supported image-generation model. Supported: Z-Image.")

    backend = _FakeBackend()
    backend.validate_load_request = _raise
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(client, model_path = "x/y", gguf_filename = "q.gguf")
    assert resp.status_code == 400
    assert "isn't a supported image-generation model" in resp.json()["detail"]


def test_load_validation_failure_does_not_evict_chat(client, monkeypatch):
    # validation runs before acquire_for so a rejected pick never evicts the chat model
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))

    backend = _FakeBackend()

    def _raise(*a, **k):
        raise ValueError("'x/y' isn't a supported image-generation model.")

    backend.validate_load_request = _raise
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(client, model_path = "x/y", gguf_filename = "q.gguf")
    assert resp.status_code == 400
    assert evicted == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_gated_base_load_returns_400_without_evicting_chat(client, monkeypatch):
    # the page falls back to /images/load when the plan fails, so refuse before acquire_for here too
    import types as _types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "cuda")
    )
    backend = diffusion_module.get_diffusion_backend()
    detail = (
        "'black-forest-labs/FLUX.1-dev' is gated on Hugging Face and this model cannot be "
        "downloaded without it."
    )

    def _refuse(model_path, fam, **kwargs):
        raise ValueError(detail)

    monkeypatch.setattr(backend, "preflight_base_access", _refuse, raising = False)

    resp = _post_load(client, model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf")

    assert resp.status_code == 400
    assert resp.json()["detail"] == detail
    assert evicted == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    assert backend.is_loaded is False


def test_cpu_load_skips_the_gated_preflight(client, monkeypatch):
    # no eviction on CPU, so the route skips preflight Hub calls; the loader still catches gated bases
    import types as _types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "cpu")
    )
    backend = diffusion_module.get_diffusion_backend()
    calls: list = []
    monkeypatch.setattr(
        backend,
        "preflight_base_access",
        lambda *a, **k: calls.append(a),
        raising = False,
    )

    resp = _post_load(client, model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf")

    assert resp.status_code == 200
    assert calls == []


def test_a_cpu_mispredicted_engine_is_still_preflighted(monkeypatch):
    """The CPU path must re-ask the activated engine; a mispredict otherwise skips the preflight."""
    from types import SimpleNamespace

    import core.inference.diffusion_device as devmod
    import core.inference.diffusion_engine_router as engine_router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP", "0")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_ENGINE", raising = False)
    monkeypatch.setattr(engine_router, "_active_engine_name", ENGINE_DIFFUSERS)
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
    # the prediction says native, so the preflight is owed on the wrong engine
    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **_: ENGINE_SD_CPP)

    native = _FakeBackend()
    monkeypatch.setattr(sd_backend, "get_sd_cpp_backend", lambda: native)

    detail = (
        "'black-forest-labs/FLUX.1-dev' is gated on Hugging Face and this model cannot be "
        "downloaded without it."
    )

    def _refuse(model_path, fam, **kwargs):
        raise ValueError(detail)

    diffusers = _FakeBackend()
    diffusers.preflight_base_access = _refuse
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: diffusers)
    monkeypatch.setattr(
        engine_router,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = "cpu", device = "cpu"),
    )
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: SimpleNamespace(device = "cpu")
    )

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    local = TestClient(app)

    resp = local.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == detail
    assert diffusers.loaded is False


@pytest.mark.parametrize(
    "device, env",
    [("cuda", {}), ("cpu", {"UNSLOTH_DIFFUSION_SD_CPP": "0"})],
)
def test_gated_pick_on_an_engine_switch_keeps_the_previous_model(monkeypatch, device, env):
    """Refusal must run before the engine switch, since switching unloads the resident image model."""
    from types import SimpleNamespace

    import core.inference.diffusion_device as devmod
    import core.inference.diffusion_engine_router as engine_router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    for e in (
        "UNSLOTH_DIFFUSION_ENGINE",
        "UNSLOTH_DIFFUSION_SD_CPP",
        "UNSLOTH_DIFFUSION_SD_CPP_MPS",
        "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    ):
        monkeypatch.delenv(e, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    resident = _FakeBackend()
    resident.loaded = True
    monkeypatch.setattr(sd_backend, "get_sd_cpp_backend", lambda: resident)
    monkeypatch.setattr(engine_router, "_active_engine_name", ENGINE_SD_CPP)
    monkeypatch.setattr(engine_router, "_fallback_reason", None)

    detail = (
        "'black-forest-labs/FLUX.1-dev' is gated on Hugging Face and this model cannot be "
        "downloaded without it."
    )

    def _refuse(model_path, fam, **kwargs):
        raise ValueError(detail)

    diffusers = _FakeBackend()
    diffusers.preflight_base_access = _refuse
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: diffusers)
    monkeypatch.setattr(
        engine_router,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = device, device = device),
    )
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: SimpleNamespace(device = device)
    )
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted: list = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    local = TestClient(app)

    resp = local.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == detail
    assert resident.loaded is True
    assert engine_router.active_engine_name() == ENGINE_SD_CPP
    assert diffusers.loaded is False
    assert evicted == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_load_refused_during_training_does_not_evict_chat(client, monkeypatch):
    import core.training as core_training

    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))

    class _Training:
        def is_training_active(self):
            return True

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Training())

    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()
    assert evicted == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_load_progress_route(client, monkeypatch):
    resets = []
    monkeypatch.setattr(inference_routes, "reset_media_load_progress", resets.append)
    idle = client.get("/api/inference/images/load-progress")
    assert idle.status_code == 200 and idle.json()["phase"] is None
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    ready = client.get("/api/inference/images/load-progress")
    assert ready.json()["phase"] == "ready"
    assert resets == ["image"]


def test_load_progress_route_logs_backend_snapshot(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        inference_routes,
        "log_media_load_progress",
        lambda media, phase, fraction: seen.append((media, phase, fraction)),
    )
    backend = diffusion_module.get_diffusion_backend()
    backend.load_progress = lambda: {
        "phase": "downloading",
        "bytes_downloaded": 40,
        "bytes_total": 100,
        "fraction": 0.4,
        "error": None,
    }

    response = client.get("/api/inference/images/load-progress")

    assert response.status_code == 200
    assert seen == [("image", "downloading", 0.4)]


def test_routes_require_auth():
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    unauth = TestClient(app)
    assert unauth.get("/api/inference/images/status").status_code in (401, 403)


def test_invalid_family_returns_400_without_evicting_chat(client):
    resp = _post_load(client, model_path = "x/y", gguf_filename = "q.gguf")
    assert resp.status_code == 400
    assert "family" in resp.json()["detail"]
    assert gpu_arbiter._owner is None


def test_validate_filenotfound_maps_to_400_without_eviction(client, monkeypatch):
    def _raise_fnf(*a, **k):
        raise FileNotFoundError("'q.gguf' not found under /models/x.")

    backend = _FakeBackend()
    backend.validate_load_request = _raise_fnf
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(client, model_path = "/models/x", gguf_filename = "q.gguf")
    assert resp.status_code == 400
    assert gpu_arbiter._owner is None


def test_memory_mode_threads_through_to_backend(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        memory_mode = "low_vram",
    )
    assert resp.status_code == 200
    assert resp.json()["memory_mode"] == "low_vram"
    assert backend.last_load_kwargs.get("memory_mode") == "low_vram"


def test_transformer_quant_threads_through_to_backend(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_quant = "auto",
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_quant") == "auto"


def test_transformer_quant_fast_accum_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_quant = "fp8",
        transformer_quant_fast_accum = False,
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_quant_fast_accum") is False


def test_transformer_prequant_path_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_quant = "fp8",
        transformer_prequant_path = "/data/zimage_fp8.pt",
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_prequant_path") == "/data/zimage_fp8.pt"


def test_attention_backend_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        attention_backend = "cudnn",
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("attention_backend") == "cudnn"


def test_invalid_attention_backend_returns_422(client):
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        attention_backend = "bogus",
    )
    assert resp.status_code == 422


def test_prequant_path_doc_describes_allowlist_not_toggle():
    # UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH is a directory allowlist, not a =1 toggle
    from models.inference import DiffusionLoadRequest

    desc = DiffusionLoadRequest.model_fields["transformer_prequant_path"].description
    assert "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" in desc
    assert "=1" not in desc
    assert "allowlist" in desc.lower() or "director" in desc.lower()


def test_transformer_cache_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_cache = "fbcache",
        transformer_cache_threshold = 0.1,
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_cache") == "fbcache"
    assert backend.last_load_kwargs.get("transformer_cache_threshold") == 0.1


def test_invalid_transformer_cache_returns_422(client):
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_cache = "deepcache",
    )
    assert resp.status_code == 422


def test_out_of_range_cache_threshold_returns_422(client):
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_cache_threshold = 1.5,
    )
    assert resp.status_code == 422


def test_load_routes_to_sd_cpp_on_cpu(monkeypatch, tmp_path):
    """End-to-end through the REAL router: a CPU host with an available binary routes
    the load to the native sd.cpp engine and the response reports engine=sd_cpp."""
    from types import SimpleNamespace

    import core.inference.diffusion_engine_router as engine_router
    import core.inference.sd_cpp_backend as sd_backend

    for e in (
        "UNSLOTH_DIFFUSION_ENGINE",
        "UNSLOTH_DIFFUSION_SD_CPP",
        "UNSLOTH_DIFFUSION_SD_CPP_MPS",
        "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    ):
        monkeypatch.delenv(e, raising = False)

    validator = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: validator)
    monkeypatch.setattr(
        engine_router,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = "cpu", device = "cpu"),
    )
    # unstubbed, the router runs the real installer (108 MB into ~/.unsloth); None keeps sd-cli path
    monkeypatch.setattr(engine_router, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(engine_router, "ensure_sd_cpp_binary", lambda **_: "/x/sd-cli")
    # the router probes runnability before committing to native
    monkeypatch.setattr(
        engine_router, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0")
    )
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
    sd_fake = _FakeBackend()
    monkeypatch.setattr(sd_backend, "get_sd_cpp_backend", lambda: sd_fake)

    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    client = TestClient(app)

    resp = _post_load(client, model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "sd_cpp"
    assert body["fallback_reason"] is None
    assert sd_fake.loaded is True


def test_invalid_transformer_quant_returns_422_without_eviction(client):
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        transformer_quant = "int2",
    )
    assert resp.status_code == 422
    assert gpu_arbiter._owner is None


def test_invalid_memory_mode_returns_422_without_eviction(client):
    resp = _post_load(
        client,
        model_path = "x/z-image",
        gguf_filename = "q.gguf",
        memory_mode = "ultra",
    )
    assert resp.status_code == 422
    assert gpu_arbiter._owner is None


def test_in_progress_returns_409_after_validation_passes(client, monkeypatch):
    def _busy(*a, **k):
        raise RuntimeError("A diffusion load is already in progress.")

    backend = _FakeBackend()
    backend.begin_load = _busy
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    import types as _types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod,
        "resolve_diffusion_device_target",
        lambda: _types.SimpleNamespace(device = "cuda"),
    )
    resp = _post_load(client, model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf")
    assert resp.status_code == 409
    assert gpu_arbiter._owner == gpu_arbiter.DIFFUSION


def _force_engine(monkeypatch, backend, *, engine_name, device):
    """Pin engine selection + device so the load route's arbiter gating is deterministic."""
    import types as _types

    import core.inference.diffusion_device as devmod
    import core.inference.diffusion_engine_router as router

    monkeypatch.setattr(router, "select_and_activate_engine", lambda fam, **kw: backend)
    monkeypatch.setattr(router, "active_engine_name", lambda: engine_name)
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = device)
    )
    acquired: list = []

    def _fake_acquire(
        role,
        register = None,
        admit = None,
    ):
        acquired.append(role)
        if admit is not None:
            admit()
        return register() if register is not None else None

    monkeypatch.setattr(gpu_arbiter, "acquire_for", _fake_acquire)
    return acquired


def test_cpu_native_load_skips_gpu_arbiter(client, monkeypatch):
    # a CPU sd.cpp load never touches the GPU, so the chat model must not be evicted
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    import core.inference.diffusion_engine_router as router

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "auto")
    import core.inference.sd_cpp_backend as sd_backend

    monkeypatch.setattr(sd_backend, "managed_sd_cpp_accelerator", lambda: None)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == []


def test_cpu_torch_vulkan_native_load_takes_arbiter(client, monkeypatch):
    # #13666: a CPU torch routed to the Vulkan sd.cpp build renders on the AMD / Arc card.
    import core.inference.diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "vulkan")
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_a_retained_vulkan_build_takes_arbiter_even_when_cpu_is_requested(client, monkeypatch):
    # A failed or disabled CPU install keeps the managed Vulkan build, which renders on the card.
    import core.inference.diffusion_engine_router as router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "auto")
    monkeypatch.setattr(sd_backend, "managed_sd_cpp_accelerator", lambda: "vulkan")
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_cpu_torch_load_after_a_failed_vulkan_build_skips_arbiter(client, monkeypatch):
    # The ladder runs the CPU build once Vulkan is recorded unrunnable, so no GPU is taken.
    import core.inference.diffusion_engine_router as router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "vulkan")
    monkeypatch.setattr(sd_backend, "preferred_accelerator", lambda accelerator, card = None: "cpu")
    monkeypatch.setattr(sd_backend, "managed_sd_cpp_accelerator", lambda: "cpu")
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == []


def test_cpu_torch_diffusers_fallback_after_a_vulkan_prediction_skips_arbiter(client, monkeypatch):
    # Predicted the Vulkan sd.cpp build, but selection fell back to diffusers on the CPU.
    import core.inference.diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_DIFFUSERS, device = "cpu")
    monkeypatch.setattr(router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "vulkan")
    monkeypatch.setattr(backend, "assert_precision_available", lambda *a, **k: None)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == []


def test_cpu_torch_vulkan_route_on_a_retained_cpu_bundle_skips_arbiter(client, monkeypatch):
    # The Vulkan install failed after selection and the usable CPU bundle was kept.
    import core.inference.diffusion_engine_router as router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "vulkan")
    monkeypatch.setattr(sd_backend, "sd_cpp_binaries_all_cpu", lambda: True)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == []


def test_gpu_torch_load_on_the_cpu_rung_skips_arbiter(client, monkeypatch):
    # ROCm and Vulkan both recorded unrunnable: the managed CPU bundle runs, so no GPU is taken.
    import core.inference.diffusion_engine_router as router
    import core.inference.sd_cpp_backend as sd_backend
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cuda")
    monkeypatch.setattr(router, "image_install_accelerator", lambda backend: "rocm")
    monkeypatch.setattr(sd_backend, "preferred_accelerator", lambda accelerator, card = None: "cpu")
    monkeypatch.setattr(sd_backend, "sd_cpp_binaries_all_cpu", lambda: True)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == []
    # A GPU build, or an unowned binary, may still be what runs: keep the claim.
    monkeypatch.setattr(sd_backend, "sd_cpp_binaries_all_cpu", lambda: False)
    acquired.clear()
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_gpu_native_load_takes_arbiter(client, monkeypatch):
    # force-native sd.cpp on a GPU box does use the GPU, so the arbiter is acquired
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cuda")
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_load_forwards_the_gpu_selection(client, monkeypatch):
    # both engines must honour the UI card pick, not pin every module to ordinal 0
    backend = diffusion_module.get_diffusion_backend()
    _force_engine(monkeypatch, backend, engine_name = "diffusers", device = "cuda")
    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", lambda ids: max(ids))
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf", gpu_ids = [1])
    assert resp.status_code == 200
    assert backend.last_load_kwargs["gpu_ids"] == [1]


def test_load_refuses_a_gpu_index_this_host_does_not_have(client, monkeypatch):
    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = "diffusers", device = "cuda")
    import core.inference.diffusion_device as devmod

    def _refuse(_ids):
        raise ValueError("Requested GPU [7] but this host has 2 CUDA device(s).")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _refuse)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf", gpu_ids = [7])
    assert resp.status_code == 400
    assert "2 CUDA device" in resp.json()["detail"]
    assert acquired == []


def test_load_ignores_a_gpu_selection_off_cuda(client, monkeypatch):
    # physical ids are documented as dropped on XPU / MPS / CPU, so do not validate them there
    backend = diffusion_module.get_diffusion_backend()
    _force_engine(monkeypatch, backend, engine_name = "diffusers", device = "mps")
    import core.inference.diffusion_device as devmod

    def _never(_ids):
        raise AssertionError("the CUDA resolver must not run off a CUDA target")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _never)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf", gpu_ids = [1])
    assert resp.status_code == 200


def test_native_load_accepts_the_resolved_ordinal(client, monkeypatch):
    # the native backend must accept gpu_ordinal or every native GGUF load 500s
    import inspect

    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    assert "gpu_ordinal" in inspect.signature(SdCppDiffusionBackend.begin_load).parameters

    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cuda")
    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", lambda ids: max(ids))
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf", gpu_ids = [1])
    assert resp.status_code == 200


def test_images_info_lists_every_family(client):
    from core.inference.diffusion_auto_policy import _FAMILY_BF16_GB

    resp = client.get("/api/inference/images/info")
    assert resp.status_code == 200
    families = resp.json()["families"]
    assert {f["family"] for f in families} == set(_FAMILY_BF16_GB)
    sample = families[0]
    est = sample["estimated_resident_gb"]
    assert est["int8"] < est["bf16"]
    assert est["nvfp4"] < est["int8"]


def test_status_passes_through_resolved(client, monkeypatch):
    # the frontend renders the Auto: X badges from `resolved`
    backend = diffusion_module.get_diffusion_backend()
    resolved = {
        "speed_mode": {"value": "eager", "source": "auto", "reason": "per-kind default"},
        "transformer_quant": {
            "value": "off",
            "requested": "fp8",
            "source": "explicit",
            "status": "fell_back",
            "reason": "the dense bf16 transformer does not fit resident",
        },
        "cpu_offload": {"value": False, "source": "auto", "reason": "from the memory plan"},
        "transformer_cache": {"value": None, "source": "auto", "reason": "few-step model"},
    }
    monkeypatch.setattr(
        backend, "status", lambda: {**_unloaded_status(), "loaded": True, "resolved": resolved}
    )
    body = client.get("/api/inference/images/status").json()
    assert body["resolved"]["speed_mode"]["source"] == "auto"
    assert body["resolved"]["cpu_offload"]["value"] is False
    assert body["resolved"]["transformer_quant"] == {
        **resolved["transformer_quant"],
        "artifact": None,
        "replaced": None,
        "steps": None,
    }
    # older backend entries (no requested/status) still parse, defaulting to applied
    assert body["resolved"]["speed_mode"]["requested"] is None
    assert body["resolved"]["speed_mode"]["status"] == "applied"


def test_status_carries_the_prequant_artifact_through_the_route(client, monkeypatch):
    # without the file the UI cannot tell a hosted checkpoint from a runtime quantise
    backend = diffusion_module.get_diffusion_backend()
    resolved = {
        "transformer_quant": {
            "value": "fp8",
            "source": "auto",
            "reason": "seeded",
            "artifact": "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-FP8.pt",
        }
    }
    monkeypatch.setattr(
        backend, "status", lambda: {**_unloaded_status(), "loaded": True, "resolved": resolved}
    )
    body = client.get("/api/inference/images/status").json()
    assert (
        body["resolved"]["transformer_quant"]["artifact"]
        == "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-FP8.pt"
    )


def test_status_resolved_defaults_to_null(client):
    body = client.get("/api/inference/images/status").json()
    assert body["resolved"] is None


def test_load_refuses_an_unusable_explicit_precision_with_409(client, monkeypatch):
    # an explicit precision this host cannot honor is a 409; the frontend shows detail verbatim
    from core.inference.diffusion_auto_policy import precision_refusal_message

    backend = diffusion_module.get_diffusion_backend()
    refusal = precision_refusal_message(
        "transformer_quant",
        "fp8",
        "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)",
        off_label = "Off to run the checkpoint as-is",
    )

    def _refuse(model_path, **kwargs):
        raise RuntimeError(refusal)

    monkeypatch.setattr(backend, "begin_load", _refuse)
    resp = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "transformer_quant='fp8' could not be used" in detail
    assert "Auto" in detail and "Off" in detail
    assert client.get("/api/inference/images/status").json()["loaded"] is False


def test_precision_refusal_precedes_eviction_and_engine_selection(client, monkeypatch):
    # refuse before the GPU handoff: acquire_for evicts chat and an engine switch unloads the resident
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.diffusion_auto_policy import precision_refusal_message

    backend = diffusion_module.get_diffusion_backend()
    evicted = []
    selected = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append("chat"))
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(
        engine_router,
        "select_and_activate_engine",
        lambda fam, **kw: (selected.append(fam), backend)[1],
    )
    refusal = precision_refusal_message(
        "transformer_quant",
        "fp8",
        "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)",
        off_label = "Off to run the checkpoint as-is",
    )

    def _refuse(fam, **kwargs):
        raise RuntimeError(refusal)

    monkeypatch.setattr(backend, "assert_precision_available", _refuse)
    # the download and the eviction both hang off begin_load
    monkeypatch.setattr(
        backend,
        "begin_load",
        lambda *a, **k: pytest.fail("begin_load ran after an impossible precision"),
    )
    resp = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409
    assert "transformer_quant='fp8' could not be used" in resp.json()["detail"]
    assert evicted == []
    assert selected == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_the_native_engine_refuses_an_explicit_precision_it_cannot_honour(client, monkeypatch):
    """sd.cpp ignores transformer_quant and text_encoder_quant, so explicit fp8 must be refused."""
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    resp = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409, resp.text
    assert "native engine" in resp.json()["detail"]


def test_a_failed_engine_prediction_still_gates_the_precision_after_selection(client, monkeypatch):
    """Precision gate re-asked of the activated engine, since a failed prediction skipped both arms."""
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()

    def _boom(fam, **kw):
        raise RuntimeError("sd-cli probe failed")

    monkeypatch.setattr(engine_router, "predict_engine", _boom)
    monkeypatch.setattr(engine_router, "active_engine_name", lambda: ENGINE_SD_CPP)
    monkeypatch.setattr(
        engine_router,
        "select_and_activate_engine",
        lambda fam, **kw: backend,
    )
    monkeypatch.setattr(
        backend,
        "begin_load",
        lambda *a, **k: pytest.fail("begin_load ran after an unhonourable precision"),
    )
    resp = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409, resp.text
    assert "native engine" in resp.json()["detail"]


@pytest.mark.parametrize("quant", [None, "auto", "none"])
def test_the_native_engine_still_loads_when_nothing_was_promised(client, monkeypatch, quant):
    """The refusal is about an explicit request only. Omitted, auto and none all delegate the
    choice, so a CPU-only host's ordinary GGUF load must keep working exactly as it does today."""
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    payload = {
        "model_path": "unsloth/Z-Image-Turbo-GGUF",
        "gguf_filename": "z-image-turbo-Q4_K_M.gguf",
    }
    if quant is not None:
        payload["transformer_quant"] = quant
    resp = client.post("/api/inference/images/load", json = payload)
    assert resp.status_code == 200, resp.text


def test_the_plan_refuses_an_impossible_precision_before_anything_is_staged(client, monkeypatch):
    """Plan refuses impossible precision up front; /images/load would refuse only after the download."""
    from core.inference.diffusion_auto_policy import precision_refusal_message

    backend = diffusion_module.get_diffusion_backend()
    refusal = precision_refusal_message(
        "transformer_quant",
        "fp8",
        "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)",
        off_label = "Off to run the checkpoint as-is",
    )

    def _refuse(fam, **kwargs):
        raise RuntimeError(refusal)

    monkeypatch.setattr(backend, "assert_precision_available", _refuse)
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: pytest.fail("the plan was built for a precision that cannot be honoured"),
    )
    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409, resp.text
    assert "transformer_quant='fp8' could not be used" in resp.json()["detail"]


def test_the_plan_does_not_probe_the_gpu_while_training_holds_it(client, monkeypatch):
    """The plan must not run the GPU smoke probe while training holds the GPU; staging needs no GPU."""
    import routes.inference as inference_routes

    backend = diffusion_module.get_diffusion_backend()
    monkeypatch.setattr(inference_routes, "_training_is_active", lambda: True)
    monkeypatch.setattr(
        backend,
        "assert_precision_available",
        lambda *a, **k: pytest.fail("the precision probe must not touch the GPU during training"),
    )

    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 200, resp.text


def test_the_native_plan_refuses_the_same_request_the_native_load_would(client, monkeypatch):
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409, resp.text
    assert "native engine" in resp.json()["detail"]


def test_the_native_plan_refuses_an_explicit_precision_while_training_runs(client, monkeypatch):
    # Value-only, so it is not gated on training like the diffusers probe: the plan must not stage a native load
    # /images/load refuses once training ends.
    import core.inference.diffusion_engine_router as engine_router
    import routes.inference as inference_routes
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(inference_routes, "_training_is_active", lambda: True)
    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 409, resp.text
    assert "native engine" in resp.json()["detail"]


def test_the_plan_still_answers_when_nothing_was_promised(client):
    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
    )
    assert resp.status_code == 200, resp.text


def test_the_native_refusal_is_waived_by_the_fallback_escape_hatch(client, monkeypatch):
    """Same escape hatch as every other precision refusal, or this one becomes the only
    unbypassable member of the family."""
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_SD_CPP)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
    resp = _post_load(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert resp.status_code == 200, resp.text


def test_download_plan_forwards_the_load_time_controls(client, monkeypatch):
    # the plan must use the load's exact configuration: memory policy, prequant path and adapters too
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    # pin the engine: a GGUF pick on a GPU-less runner routes to sd.cpp's different planner
    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen["model_path"] = model_path
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
        hf_token = "hf_secret",
        speed_mode = "off",
        transformer_quant = "int8",
        memory_mode = "low_vram",
        cpu_offload = True,
        transformer_quant_fast_accum = False,
        loras = [{"id": "unsloth/some-lora", "weight": 0.8}],
    )

    assert resp.status_code == 200
    assert seen["hf_token"] == "hf_secret"
    assert seen["speed_mode"] == "off"
    assert seen["transformer_quant"] == "int8"
    assert seen["memory_mode"] == "low_vram"
    assert seen["cpu_offload"] is True
    # a forced accumulate the hosted checkpoint cannot bake must decline the seed
    assert seen["transformer_quant_fast_accum"] is False
    assert len(seen["loras"] or []) == 1


def test_download_plan_suppresses_only_the_verdict_while_training_runs(client, monkeypatch):
    # the plan must keep counting files during training; only the device-reading refusal is skipped
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS
    from routes import inference as routes_inference

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)
    body = {
        "model_path": "unsloth/FLUX.1-dev-GGUF",
        "gguf_filename": "flux1-dev-Q4_K_M.gguf",
        "model_kind": "gguf",
        "transformer_quant": "fp8",
    }

    for training, expected in ((True, False), (False, True)):
        monkeypatch.setattr(routes_inference, "_training_is_active", lambda: training)
        assert client.post("/api/inference/images/download-plan", json = body).status_code == 200
        assert seen["memory_verdict"] is expected, training
        assert "allow_device_probe" not in seen, training


@pytest.mark.parametrize("plan_failed", [False, True])
def test_download_plan_response_keeps_the_planners_checkpoint_marker(
    client, monkeypatch, plan_failed
):
    # go through the route: fields the response model does not declare are dropped silently
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()

    def _plan(model_path, **kwargs):
        return {
            "entries": [
                {
                    "repo_id": "unsloth/FLUX.1-dev",
                    "files": ["model_index.json"],
                    "bytes": 10,
                    "gguf_filename": None,
                    "checkpoint": True,
                },
                {
                    "repo_id": "some/text-encoder",
                    "files": ["model.safetensors"],
                    "bytes": 20,
                    "gguf_filename": None,
                    "checkpoint": False,
                },
            ],
            "plan_failed": plan_failed,
            "total_bytes": 30,
            "required_bytes": 30,
            "checkpoint_bytes": 10,
        }

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert resp.status_code == 200
    assert resp.json()["plan_failed"] is plan_failed
    assert [e["checkpoint"] for e in resp.json()["entries"]] == [True, False]


def test_download_plan_defaults_the_checkpoint_marker_for_an_older_planner(client, monkeypatch):
    # old plans without the marker must serialize, defaulting to not-the-checkpoint
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda model_path, **kwargs: {
            "entries": [{"repo_id": "a/b", "files": ["f"], "bytes": 1}],
            "total_bytes": 1,
        },
        raising = False,
    )

    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert resp.status_code == 200
    assert resp.json()["plan_failed"] is False
    assert resp.json()["entries"][0]["checkpoint"] is False


def test_download_plan_surfaces_a_gated_base_as_a_400(client, monkeypatch):
    # the repo id and licence URL in the ValueError are the user's fix
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()
    detail = (
        "'black-forest-labs/FLUX.1-dev' is gated on Hugging Face and this model cannot be "
        "downloaded without it. Accept its licence at "
        "https://huggingface.co/black-forest-labs/FLUX.1-dev, then add a Hugging Face token "
        "that has access in Unsloth settings and try again."
    )

    def _plan(model_path, **kwargs):
        raise ValueError(detail)

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == detail


def test_download_plan_uses_the_engine_the_load_will_pick(client, monkeypatch):
    # GPU-less GGUF picks route to sd.cpp, which reads single-file assets, not sharded components
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend as sd_cpp
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_SD_CPP)
    native_plan = {
        "entries": [
            {
                "repo_id": "unsloth/Z-Image-Turbo-ComfyUI",
                "files": ["ae.safetensors"],
                "bytes": 7,
                "gguf_filename": None,
            }
        ],
        "total_bytes": 7,
    }
    seen: dict = {}

    class _Native:
        def download_plan(self, model_path, **kwargs):
            seen["model_path"] = model_path
            seen.update(kwargs)
            return native_plan

    monkeypatch.setattr(sd_cpp, "get_sd_cpp_backend", lambda: _Native())
    diffusers_backend = diffusion_module.get_diffusion_backend()
    monkeypatch.setattr(
        diffusers_backend,
        "download_plan",
        lambda *a, **k: pytest.fail("planned with diffusers for a native-routed load"),
        raising = False,
    )

    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
        hf_token = "hf_secret",
    )

    assert resp.status_code == 200
    assert resp.json()["total_bytes"] == 7
    assert seen["model_path"] == "unsloth/Z-Image-Turbo-GGUF"
    assert seen["gguf_filename"] == "z-image-turbo-Q4_K_M.gguf"
    assert seen["hf_token"] == "hf_secret"


def test_download_plan_stays_on_diffusers_when_the_load_will(client, monkeypatch):
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend as sd_cpp
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    monkeypatch.setattr(
        sd_cpp,
        "get_sd_cpp_backend",
        lambda: pytest.fail("planned natively for a diffusers load"),
    )
    backend = diffusion_module.get_diffusion_backend()
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: {"entries": [], "total_bytes": 11},
        raising = False,
    )

    resp = _post_download_plan(
        client,
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )
    assert resp.status_code == 200 and resp.json()["total_bytes"] == 11


def test_load_refused_when_only_the_diffusion_probe_can_be_read(client, monkeypatch):
    # the two training probes are independent: a raising LLM probe must not skip the diffusion one
    import core.training as core_training
    import routes.inference as inference_routes

    class _Broken:
        def is_training_active(self):
            raise RuntimeError("training backend unavailable")

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Broken())
    monkeypatch.setattr(inference_routes, "_diffusion_training_active", lambda: True)

    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()

    monkeypatch.setattr(inference_routes, "_diffusion_training_active", lambda: False)
    resp = _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    assert resp.status_code == 200


def test_recipe_records_the_conditioned_workflow_settings(client, monkeypatch):
    # conditioned recipes record what ran so the client can name the inputs to re-add
    import base64
    import io

    from PIL import Image

    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 30, 90)).save(buf, format = "PNG")
    px = base64.b64encode(buf.getvalue()).decode()
    gen = _post_generate(
        client,
        prompt = "a sloth",
        seed = 7,
        init_image = px,
        mask_image = px,
        strength = 0.42,
    )
    assert gen.status_code == 200
    img = gen.json()["images"][0]
    assert img["workflow"] == "inpaint"
    assert img["strength"] == 0.42
    plain = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert plain.status_code == 200
    plain_img = plain.json()["images"][0]
    assert plain_img["workflow"] == "txt2img"
    assert plain_img["strength"] is None and plain_img["upscale"] is None


def test_recipe_records_the_load_time_build(client, monkeypatch):
    # a repo id alone cannot rebuild the pipeline: GGUF repos hold many quants, torchao bakes adapters
    backend = diffusion_module.get_diffusion_backend()

    def _generate(**kwargs):
        return {
            "images": [object()],
            "seed": 777,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "model_kind": "gguf",
            "gguf_filename": "z-image-turbo-Q8_0.gguf",
            "transformer_quant": "int8",
            "text_encoder_quant": "fp8",
            "memory_mode": "balanced",
            "offload_policy": "group",
            "baked_loras": ["bakedlora"],
            "active_loras": [],
            "workflow": "txt2img",
        }

    monkeypatch.setattr(backend, "generate", _generate, raising = False)

    _post_load(client, model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf")
    resp = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 777})
    assert resp.status_code == 200
    img = resp.json()["images"][0]
    assert img["model"] == "unsloth/Z-Image-Turbo-GGUF"
    assert img["model_kind"] == "gguf"
    assert img["gguf_filename"] == "z-image-turbo-Q8_0.gguf"
    assert img["transformer_quant"] == "int8"
    assert img["text_encoder_quant"] == "fp8"
    assert img["memory_mode"] == "balanced"
    assert img["offload_policy"] == "group"
    assert img["baked_loras"] == ["bakedlora"]
    assert img["loras"] == []
    # the recipe must survive a reload from the PNG's own text chunk
    listed = client.get("/api/inference/images/gallery").json()["images"][0]
    assert listed["text_encoder_quant"] == "fp8" and listed["memory_mode"] == "balanced"


def test_recipe_build_fields_absent_on_an_engine_that_omits_them(client):
    # native path and older records lack build keys; degrade to nulls, never 500
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    resp = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert resp.status_code == 200
    img = resp.json()["images"][0]
    assert img["model_kind"] is None
    assert img["gguf_filename"] is None
    assert img["transformer_quant"] is None
    # not in image_gallery._REQUIRED_META, so absent keys read back as null
    assert img["text_encoder_quant"] is None
    assert img["memory_mode"] is None and img["offload_policy"] is None
    assert len(client.get("/api/inference/images/gallery").json()["images"]) == 1
    assert img["baked_loras"] == []


def test_gallery_image_accepts_a_record_written_before_the_build_fields():
    # list_gallery_images drops invalid records, so a non-optional field would empty a gallery
    from models.inference import GalleryImage

    old = {
        "id": "img0",
        "url": "/api/inference/images/gallery/img0/file",
        "prompt": "a sloth",
        "width": 512,
        "height": 512,
        "steps": 9,
        "guidance": 3.5,
        "seed": 777,
        "created_at": 1.0,
    }
    record = GalleryImage(**old)
    assert record.model_kind is None
    assert record.gguf_filename is None
    assert record.transformer_quant is None
    assert record.baked_loras == []


def test_cancel_generation_route_reports_false_when_idle(client):
    # nothing in flight: 200/False lets the page reset its button
    resp = client.post("/api/inference/images/generate/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is False


def test_cancel_generation_route_stops_an_in_flight_generation(client):
    # the route must reach the same event the denoise loop watches
    import threading

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    started = threading.Event()

    def _wait_for_cancel(**kwargs):
        cancel = threading.Event()
        backend.active_generate_cancel = cancel
        started.set()
        try:
            assert cancel.wait(5), "cancel event was never set"
            raise RuntimeError("Diffusion generation was cancelled.")
        finally:
            backend.active_generate_cancel = None

    backend.generate = _wait_for_cancel
    result: dict = {}

    def _run():
        result["resp"] = client.post("/api/inference/images/generate", json = {"prompt": "p"})

    worker = threading.Thread(target = _run, daemon = True)
    worker.start()
    assert started.wait(5)

    cancelled = client.post("/api/inference/images/generate/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["cancelled"] is True

    worker.join(10)
    assert result["resp"].status_code == 409
    assert result["resp"].json()["detail"] == "Diffusion generation was cancelled."
    assert client.get("/api/inference/images/gallery").json()["images"] == []


def test_cancel_generation_route_requires_auth():
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    unauth = TestClient(app)
    assert unauth.post("/api/inference/images/generate/cancel").status_code in (401, 403)


@pytest.mark.parametrize(
    "memory",
    [
        {"memory_mode": "balanced"},
        {"memory_mode": "low_vram"},
        {"cpu_offload": True},
    ],
)
def test_an_offloading_memory_request_refuses_an_explicit_precision(monkeypatch, memory):
    """balanced and low_vram name their offload policy outright, and the legacy cpu_offload flag
    forces whole-module offload, so the refusal is owed before anything is staged or evicted."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            None, model_kind = "gguf", transformer_quant = "fp8", **memory
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)
    assert "offload" in str(excinfo.value)


def _pipeline_precision_gate(monkeypatch, **memory):
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    backend.assert_precision_available(
        types.SimpleNamespace(name = "z-image"),
        model_kind = "pipeline",
        transformer_quant = "fp8",
        **memory,
    )


@pytest.mark.parametrize("memory", [{"memory_mode": "low_vram"}, {"cpu_offload": True}])
def test_a_whole_module_offload_request_admits_a_pipeline_precision(monkeypatch, memory):
    """low_vram and cpu_offload pick whole-module offload, which a pipeline quantises under."""
    _pipeline_precision_gate(monkeypatch, **memory)


def test_balanced_memory_refuses_a_pipeline_precision(monkeypatch):
    """balanced is group offload, which the loader never quantises under."""
    with pytest.raises(RuntimeError) as excinfo:
        _pipeline_precision_gate(monkeypatch, memory_mode = "balanced")
    assert "group-offload hooks" in str(excinfo.value)


def test_a_measured_memory_mode_is_not_refused_by_the_precision_gate(monkeypatch):
    """fast and auto decide their policy from the MEASURED footprint, so they are not knowable
    network-free and this gate has no business refusing them."""
    from core.inference.diffusion import _memory_request_forces_offload

    assert _memory_request_forces_offload("fast", False) is False
    assert _memory_request_forces_offload("auto", False) is False
    assert _memory_request_forces_offload(None, False) is False
    # the legacy flag applies only when no mode was named, matching resolve_offload_policy
    assert _memory_request_forces_offload("fast", True) is False
    assert _memory_request_forces_offload(None, True) is True


def test_fast_and_auto_memory_do_not_refuse_a_precision(client, monkeypatch):
    """The other side of the fence: fast and auto decide their policy from the MEASURED
    footprint, so they are not knowable network-free and must not be refused here."""
    import core.inference.diffusion_engine_router as engine_router
    from core.inference.diffusion_engine_router import ENGINE_DIFFUSERS

    backend = diffusion_module.get_diffusion_backend()
    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: ENGINE_DIFFUSERS)
    seen: list = []
    monkeypatch.setattr(backend, "assert_precision_available", lambda *a, **k: seen.append(k))
    for mode in ("fast", "auto"):
        resp = _post_load(
            client,
            model_path = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_M.gguf",
            transformer_quant = "fp8",
            memory_mode = mode,
        )
        assert resp.status_code != 409, resp.text
    assert seen and all("memory_mode" in kwargs for kwargs in seen)


@pytest.mark.parametrize("mode", ["int8", "fp8_dynamic", "nvfp4"])
def test_an_offloading_memory_request_refuses_a_torchao_text_encoder(monkeypatch, mode):
    """Offload hooks use Module.to(), which torchao subclasses do not survive, so refuse up front."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "te_quant_supported", lambda target, m: True)
    monkeypatch.setattr(diffusion_module, "torchao_quantize_importable", lambda: True)
    _scheduled_family_without_a_hosted_int8_file(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            # A family WITH an int8 schedule, so the int8 case is not downgraded to fp8 first.
            types.SimpleNamespace(name = "scheduled-family"),
            model_kind = "gguf",
            text_encoder_quant = mode,
            memory_mode = "low_vram",
        )
    assert "text_encoder_quant" in str(excinfo.value)
    assert "offload" in str(excinfo.value)


def test_layerwise_fp8_survives_an_offloading_memory_request(monkeypatch):
    """fp8 is a dtype cast, not a torchao tensor subclass, so offload does not rule it out and
    refusing it would reject a load the runtime runs."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "te_quant_supported", lambda target, m: True)
    monkeypatch.setattr(diffusion_module, "torchao_quantize_importable", lambda: True)
    backend.assert_precision_available(
        types.SimpleNamespace(name = "qwen-image"),
        model_kind = "gguf",
        text_encoder_quant = "fp8",
        memory_mode = "low_vram",
    )


def _scheduled_family_without_a_hosted_int8_file(monkeypatch):
    """A family whose int8 is the torchao keep-bf16 cast. Every image family with a schedule (qwen-image, its edit and
    layered releases, FLUX.2-dev) now hosts an int8 ConvRot file, which loads as plain tensors and needs no torchao."""
    import core.inference.diffusion_precision as precision
    monkeypatch.setitem(precision._TE_INT8_SKIP, "scheduled-family", (6, 6))


@pytest.mark.parametrize("mode", ["int8", "fp8_dynamic", "nvfp4"])
def test_a_broken_torchao_refuses_a_torchao_text_encoder_before_the_download(monkeypatch, mode):
    """torchao is imported only after the download, so a broken install must be refused pre-load."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "te_quant_supported", lambda target, m: True)
    monkeypatch.setattr(diffusion_module, "torchao_quantize_importable", lambda: False)
    _scheduled_family_without_a_hosted_int8_file(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "scheduled-family"),
            model_kind = "gguf",
            text_encoder_quant = mode,
        )
    assert "torchao is not importable" in str(excinfo.value)


def test_layerwise_fp8_does_not_need_torchao(monkeypatch):
    """fp8 is a plain dtype cast, so a host without torchao still runs it and refusing it would
    reject a load that works."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "te_quant_supported", lambda target, m: True)
    monkeypatch.setattr(diffusion_module, "torchao_quantize_importable", lambda: False)
    backend.assert_precision_available(
        types.SimpleNamespace(name = "qwen-image"), model_kind = "gguf", text_encoder_quant = "fp8"
    )


def test_download_plan_sizes_its_file_set_for_the_selected_card(client, monkeypatch):
    # the plan sizes files against a card's capability and free VRAM; one ranking per request
    import types as _types

    from core.inference import diffusion_device as devmod
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "cuda")
    )
    ranked: list = []

    def _resolve(ids):
        ranked.append(list(ids))
        return 1

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _resolve)
    backend = diffusion_module.get_diffusion_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)
    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
        gpu_ids = [0, 1],
    )
    assert resp.status_code == 200
    assert seen["gpu_ordinal"] == 1
    assert backend.last_precision_kwargs["gpu_ordinal"] == 1
    # one ranking: the smoke probe allocates on the card, so a second could pick another
    assert ranked == [[0, 1]]


def test_download_plan_refuses_a_gpu_index_this_host_does_not_have(client, monkeypatch):
    # a bad pick is a 400, not a 500: this call sits inside the plan route's own try/except
    import types as _types

    from core.inference import diffusion_device as devmod
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "cuda")
    )

    def _refuse(_ids):
        raise ValueError("Requested GPU [7] but none of them are visible to this process")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _refuse)
    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
        gpu_ids = [7],
    )
    assert resp.status_code == 400
    assert "visible to this process" in resp.json()["detail"]


def test_download_plan_ignores_a_gpu_selection_off_cuda(client, monkeypatch):
    # physical ids have no applicator on XPU / MPS / CPU, so they are dropped, not refused
    import types as _types

    from core.inference import diffusion_device as devmod
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "mps")
    )

    def _never(_ids):
        raise AssertionError("the CUDA resolver must not run off a CUDA target")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _never)
    backend = diffusion_module.get_diffusion_backend()
    seen: dict = {}
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda model_path, **kwargs: (seen.update(kwargs), {"entries": [], "total_bytes": 0})[1],
        raising = False,
    )
    resp = _post_download_plan(
        client,
        model_path = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        model_kind = "gguf",
        gpu_ids = [1],
    )
    assert resp.status_code == 200
    assert seen["gpu_ordinal"] is None


def test_download_plan_still_refuses_a_bad_gpu_while_training_holds_the_cards(client, monkeypatch):
    # the training guard only skips the ranking (CUDA context), not the whole resolution
    import types as _types

    from core.inference import diffusion_device as devmod
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS
    from routes import inference as routes_inference

    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: _types.SimpleNamespace(device = "cuda")
    )
    monkeypatch.setattr(routes_inference, "_training_is_active", lambda: True)
    seen: dict = {}

    def _resolve(ids, *, allow_ranking = True):
        seen["ids"], seen["allow_ranking"] = list(ids), allow_ranking
        if ids == [7]:
            raise ValueError("Requested GPU [7] but none of them are visible to this process")
        return ids[0]

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _resolve)
    backend = diffusion_module.get_diffusion_backend()
    planned: dict = {}
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda model_path, **kwargs: (planned.update(kwargs), {"entries": [], "total_bytes": 0})[1],
        raising = False,
    )
    body = {
        "model_path": "unsloth/FLUX.1-dev-GGUF",
        "gguf_filename": "flux1-dev-Q4_K_M.gguf",
        "model_kind": "gguf",
    }
    resp = client.post("/api/inference/images/download-plan", json = {**body, "gpu_ids": [1]})
    assert resp.status_code == 200
    assert seen == {"ids": [1], "allow_ranking": False}
    assert planned["gpu_ordinal"] == 1
    assert getattr(backend, "last_precision_kwargs", None) is None

    resp = client.post("/api/inference/images/download-plan", json = {**body, "gpu_ids": [7]})
    assert resp.status_code == 400
    assert "visible to this process" in resp.json()["detail"]


def test_a_pipeline_pick_may_pin_a_precision(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    backend.assert_precision_available(
        types.SimpleNamespace(name = "z-image"),
        model_kind = "pipeline",
        transformer_quant = "fp8",
    )


def test_a_single_file_pick_still_cannot_pin_a_precision(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "ltx-2.3"),
            model_kind = "single_file",
            transformer_quant = "fp8",
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)
    assert "single_file" in str(excinfo.value)
    assert "GGUF and pipeline" in str(excinfo.value)


def test_a_pipeline_pick_is_still_refused_on_a_device_that_cannot_quantise(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: False)
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "z-image"),
            model_kind = "pipeline",
            transformer_quant = "fp8",
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)


def test_a_unet_family_is_refused_before_the_eviction_and_the_download(monkeypatch):
    """SDXL's family metadata already proves there is no transformer, so say so pre-eviction."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module,
        "select_transformer_quant_scheme",
        lambda *a, **k: pytest.fail("the UNet refusal must not need a scheme probe"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "sdxl", denoiser_attr = "unet"),
            model_kind = "pipeline",
            transformer_quant = "fp8",
        )
    message = str(excinfo.value)
    assert "transformer_quant='fp8' could not be used" in message
    assert "unet" in message and "not a transformer" in message


def test_a_transformer_family_still_reaches_the_scheme_check(monkeypatch):
    """The new branch must not swallow the families that do have a transformer."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_device_target",
        lambda self, fam: _cuda_target(),
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    seen: list = []
    monkeypatch.setattr(
        diffusion_module,
        "select_transformer_quant_scheme",
        lambda *a, **k: (seen.append(a) or "fp8"),
    )
    backend.assert_precision_available(
        types.SimpleNamespace(name = "z-image"),
        model_kind = "pipeline",
        transformer_quant = "fp8",
    )
    assert seen


def test_an_eager_pipeline_precision_is_refused_before_the_eviction(monkeypatch):
    """Eager plus an explicit scheme is decidable from the request, so it must not cost a download."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module,
        "select_transformer_quant_scheme",
        lambda *a, **k: pytest.fail("the eager refusal must not need a scheme probe"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "z-image"),
            model_kind = "pipeline",
            transformer_quant = "fp8",
            speed_mode = "eager",
        )
    assert "eager" in str(excinfo.value)


def test_a_gguf_pick_still_accepts_an_eager_precision(monkeypatch):
    """The compile guard is the pipeline path's; GGUF substitutes weights and is left as on main."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    backend.assert_precision_available(
        types.SimpleNamespace(name = "z-image"),
        model_kind = "gguf",
        transformer_quant = "fp8",
        speed_mode = "eager",
    )


def test_a_process_that_cannot_compile_refuses_a_pipeline_precision_up_front(monkeypatch):
    """A Windows install without Triton is knowable here too, so it refuses before the download."""
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(diffusion_module, "compile_eligible", lambda target, **kw: False)
    with pytest.raises(RuntimeError) as excinfo:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "z-image"),
            model_kind = "pipeline",
            transformer_quant = "fp8",
        )
    assert "compile" in str(excinfo.value)


def test_the_plan_route_refuses_an_unrecognised_model_before_planning(client):
    """The route validates exactly as /images/load does, on purpose: an unloadable pick has to fail
    HERE rather than after a multi-GB download. So an unrecognised repo never reaches the planner,
    and Download only surfaces the reason instead of fetching a model nothing can open."""
    resp = _post_download_plan(
        client,
        model_path = "someone/mixed-gguf-collection",
        gguf_filename = "totally-unknown-thing-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert resp.status_code == 400, resp.text
    assert "Could not infer a diffusion family" in resp.json()["detail"]


def test_edit_without_a_size_lets_the_backend_match_image_1(client, monkeypatch):
    """An edit that names no size must not be pinned to the schema's 1024 square: the route hands
    the backend None so it sizes from Image 1 on the family grid. A named size passes through."""
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    backend = diffusion_module.get_diffusion_backend()
    seen = []
    original = backend.generate

    def _record(**kwargs):
        seen.append(kwargs)
        out = original(**kwargs)
        out.update(workflow = "edit", reference_resolution = 512, localized_edit = "mask")
        return out

    monkeypatch.setattr(backend, "generate", _record)
    layer = {"mode": "mask", "image": "QUJD"}
    resp = _post_generate(
        client,
        prompt = "p",
        workflow = "edit",
        init_image = "QUJD",
        reference_images = ["QUJD", "QUJD"],
        reference_resolution = 512,
        localized_edit = layer,
    )
    assert resp.status_code == 200
    assert seen[-1]["width"] is None and seen[-1]["height"] is None
    assert seen[-1]["workflow"] == "edit" and seen[-1]["reference_resolution"] == 512
    assert seen[-1]["localized_edit"].mode == "mask"
    record = resp.json()["images"][0]
    assert record["reference_resolution"] == 512
    assert record["localized_edit"] == "mask"
    assert record["reference_image_count"] == 2
    _post_generate(client, prompt = "p", workflow = "edit", init_image = "QUJD", width = 1024)
    assert seen[-1]["width"] == 1024 and seen[-1]["height"] == 1024


def test_generate_schema_bounds_for_unified_editing(client):
    _post_load(client, model_path = "x/z-image", gguf_filename = "q.gguf")
    nine = _post_generate(client, prompt = "p", init_image = "QUJD", reference_images = ["QUJD"] * 9)
    assert nine.status_code == 200
    ten = _post_generate(client, prompt = "p", init_image = "QUJD", reference_images = ["QUJD"] * 10)
    assert ten.status_code == 422
    assert _post_generate(client, prompt = "p", workflow = "inpaint").status_code == 422
    assert _post_generate(client, prompt = "p", width = 2752, height = 1536).status_code == 200
    assert _post_generate(client, prompt = "p", width = 2768).status_code == 422
    bad_mode = _post_generate(client, prompt = "p", localized_edit = {"mode": "lasso", "image": "QUJD"})
    assert bad_mode.status_code == 422


def test_the_precision_gate_judges_an_explicit_scheme_on_the_pipeline_base(monkeypatch):
    """A per-base NVFP4 gate record (Qwen-Image-2512) lifts the family deny only for that base, so the
    route gate must ask about the base the load keys on, or it 409s a load the loader accepts."""
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module, "_pipeline_quant_uncompilable_reason", lambda *a, **k: None
    )
    seen = []

    def _select(target, requested, **kwargs):
        seen.append(kwargs.get("base_repo"))
        return requested

    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", _select)
    backend.assert_precision_available(
        types.SimpleNamespace(name = "qwen-image"),
        model_kind = "pipeline",
        transformer_quant = "nvfp4",
        repo_id = "Qwen/Qwen-Image-2512",
    )
    assert seen == ["Qwen/Qwen-Image-2512"]


def test_a_gpu_refusal_on_a_gated_base_is_not_blamed_on_the_family_deny(monkeypatch):
    """Qwen-Image-2512's gate record lifts the qwen-image NVFP4 deny for that base only. When the GPU then declines
    the scheme, the refusal must say so, not claim no accuracy record covers the base."""
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module, "_pipeline_quant_uncompilable_reason", lambda *a, **k: None
    )
    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", lambda *a, **k: None)
    import core.inference.diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "torchao_unavailable_reason", lambda: None)
    base = "Qwen/Qwen-Image-2512"
    assert not tq.family_denies_scheme("qwen-image", "nvfp4", base)
    with pytest.raises(RuntimeError) as err:
        backend.assert_precision_available(
            types.SimpleNamespace(name = "qwen-image"),
            model_kind = "pipeline",
            transformer_quant = "nvfp4",
            repo_id = base,
        )
    message = str(err.value)
    assert "on this GPU" in message
    assert "accuracy-gate record" not in message


def test_a_gguf_pick_whose_base_comes_from_its_card_is_judged_on_the_gated_base(monkeypatch):
    """A GGUF load that names no base_repo resolves it from the repo card at load time. The network-free gate must not
    refuse explicit NVFP4 under the generic family deny that the Qwen-Image-2512 gate record lifts."""
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    backend = DiffusionBackend.__new__(DiffusionBackend)
    monkeypatch.setattr(
        DiffusionBackend, "_resolve_device_target", lambda self, fam: _cuda_target()
    )
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module, "_pipeline_quant_uncompilable_reason", lambda *a, **k: None
    )
    import core.inference.diffusion_transformer_quant as tq

    seen = []

    def _select(target, requested, **kwargs):
        seen.append(kwargs.get("base_repo"))
        return (
            None
            if tq.family_denies_scheme(kwargs.get("family"), requested, kwargs.get("base_repo"))
            else requested
        )

    monkeypatch.setattr(diffusion_module, "select_transformer_quant_scheme", _select)
    backend.assert_precision_available(
        types.SimpleNamespace(name = "qwen-image"),
        model_kind = "gguf",
        transformer_quant = "nvfp4",
        repo_id = "unsloth/Qwen-Image-2512-GGUF",
    )
    assert seen == ["Qwen/Qwen-Image-2512"]


def test_a_managed_account_cannot_send_allow_oversized(client, monkeypatch):
    from hub.services.models import account_access

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    seen = []
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "require_media_adapters", lambda *a, **k: None)
    monkeypatch.setattr(account_access, "require_media_generation_access", lambda *a, **k: None)
    monkeypatch.setattr(
        backend,
        "generate",
        lambda **kw: seen.append(kw.get("allow_oversized"))
        or {"images": [], "seed": 1, "seeds": []},
    )
    resp = client.post(
        "/api/inference/images/generate", json = {"prompt": "p", "allow_oversized": True}
    )
    assert resp.status_code == 200
    assert seen == [False]


def _external_image_connection(monkeypatch, handler):
    import httpx

    import core.inference.external_provider as external_provider

    monkeypatch.setattr(
        external_provider,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        inference_routes.providers_db,
        "get_provider",
        lambda pid: (
            {
                "provider_type": "custom",
                "display_name": "Gufo",
                "base_url": "http://127.0.0.1:8000/v1",
                "is_enabled": True,
            }
            if pid == "conn-1"
            else None
        ),
    )
    monkeypatch.setattr(
        inference_routes, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test"
    )


def test_external_connection_generates_without_a_local_model_and_saves_to_gallery(
    client, monkeypatch
):
    import base64
    import io
    import json

    import httpx
    from PIL import Image

    sent = []

    def handler(req):
        sent.append((str(req.url), req.headers.get("authorization"), json.loads(req.content)))
        buf = io.BytesIO()
        Image.new("RGB", (512, 256), (200, 30, 30)).save(buf, format = "PNG")
        png = base64.b64encode(buf.getvalue()).decode()
        return httpx.Response(200, json = {"created": 0, "data": [{"b64_json": png}] * 2})

    _external_image_connection(monkeypatch, handler)
    resp = _post_generate(
        client,
        prompt = "a sloth",
        width = 512,
        height = 256,
        batch_size = 2,
        provider_id = "conn-1",
        model = "qwen-image",
    )
    assert resp.status_code == 200, resp.text
    images = resp.json()["images"]
    assert [(i["prompt"], i["model"], i["width"], i["height"]) for i in images] == [
        ("a sloth", "qwen-image", 512, 256)
    ] * 2
    assert len(client.get("/api/inference/images/gallery").json()["images"]) == 2
    assert sent == [
        (
            "http://127.0.0.1:8000/v1/images/generations",
            "Bearer sk-test",
            {
                "model": "qwen-image",
                "prompt": "a sloth",
                "n": 2,
                "size": "512x256",
                "response_format": "b64_json",
            },
        )
    ]


def test_external_connection_error_is_a_502_with_the_upstream_reason(client, monkeypatch):
    import httpx

    _external_image_connection(
        monkeypatch, lambda req: httpx.Response(404, text = "model 'nope' not found")
    )
    resp = _post_generate(client, prompt = "a sloth", provider_id = "conn-1", model = "nope")
    assert resp.status_code == 502
    assert "model 'nope' not found" in resp.text
    assert client.get("/api/inference/images/gallery").json()["images"] == []


def test_external_connection_does_not_hold_local_chat_swaps(client, monkeypatch):
    import base64
    import io

    import httpx
    from PIL import Image

    import core.inference.llama_keepwarm as kw

    client.app.add_middleware(kw.LlamaKeepWarmMiddleware)
    monkeypatch.setattr(kw, "_inflight", 0)
    seen = []

    def handler(req):
        seen.append(
            kw.other_inference_request_count(current_request_counted = False, include_pending = False)
        )
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format = "PNG")
        return httpx.Response(
            200, json = {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        )

    _external_image_connection(monkeypatch, handler)
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "provider_id": "conn-1", "model": "qwen-image"},
        headers = {"Authorization": "Bearer session"},
    )
    assert resp.status_code == 200, resp.text
    assert seen == [0]
    assert kw._inflight == 0


def test_external_connection_refuses_a_key_read_after_the_connection_moved(client, monkeypatch):
    import httpx

    sent = []
    _external_image_connection(monkeypatch, lambda req: sent.append(req) or httpx.Response(500))
    rows = iter(["http://127.0.0.1:8000/v1"] * 2 + ["http://10.0.0.9:9000/v1"])
    monkeypatch.setattr(
        inference_routes.providers_db,
        "get_provider",
        lambda pid: {
            "provider_type": "custom",
            "display_name": "Gufo",
            "base_url": next(rows),
            "is_enabled": True,
        },
    )
    resp = _post_generate(client, prompt = "a sloth", provider_id = "conn-1", model = "qwen-image")
    assert resp.status_code == 409
    assert "Provider changed" in resp.json()["detail"]
    assert sent == []


def test_stop_cancels_a_connection_run_and_progress_shows_it(client, monkeypatch):
    import asyncio

    import httpx

    started = asyncio.Event()
    calls = []

    async def handler(req):
        calls.append(req)
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(500)

    _external_image_connection(monkeypatch, handler)

    async def run():
        transport = httpx.ASGITransport(app = client.app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://t") as ac:
            gen = asyncio.create_task(
                ac.post(
                    "/api/inference/images/generate",
                    json = {"prompt": "a sloth", "provider_id": "conn-1", "model": "qwen-image"},
                )
            )
            await asyncio.wait_for(started.wait(), 5)
            progress = (await ac.get("/api/inference/images/generate-progress")).json()
            local = (await ac.post("/api/inference/images/generate/cancel")).json()
            stop = (await ac.post("/api/inference/images/generate/cancel?target=connection")).json()
            resp = await asyncio.wait_for(gen, 5)
            after = (await ac.get("/api/inference/images/generate-progress")).json()
            return progress, local, stop, resp, after

    progress, local, stop, resp, after = asyncio.run(run())
    assert progress["active"] is True
    # A page that resumed this run reads its Stop target from progress.
    assert progress["target"] == "connection"
    assert local == {"cancelled": False}
    assert stop == {"cancelled": True}
    assert resp.status_code == 409
    assert after["active"] is False
    assert after["target"] is None
    assert len(calls) == 1
    assert client.get("/api/inference/images/gallery").json()["images"] == []


def test_stop_reaches_a_connection_run_after_the_account_scope_moves(client, monkeypatch):
    import asyncio

    import httpx
    import hub.services.models.account_access as account_access

    started = asyncio.Event()

    async def handler(req):
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(500)

    _external_image_connection(monkeypatch, handler)

    async def run():
        transport = httpx.ASGITransport(app = client.app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://t") as ac:
            gen = asyncio.create_task(
                ac.post(
                    "/api/inference/images/generate",
                    json = {"prompt": "a sloth", "provider_id": "conn-1", "model": "qwen-image"},
                )
            )
            await asyncio.wait_for(started.wait(), 5)
            # The first managed account appears mid-run: the owner's scope goes from None to its id.
            monkeypatch.setattr(account_access, "account_scope", lambda: "owner-scope")
            progress = (await ac.get("/api/inference/images/generate-progress")).json()
            stop = (await ac.post("/api/inference/images/generate/cancel?target=connection")).json()
            resp = await asyncio.wait_for(gen, 5)
            return progress, stop, resp

    progress, stop, resp = asyncio.run(run())
    assert progress["active"] is True
    assert stop == {"cancelled": True}
    assert resp.status_code == 409


def test_external_connection_refuses_options_it_cannot_forward(client, monkeypatch):
    import httpx

    sent = []
    _external_image_connection(monkeypatch, lambda req: sent.append(req) or httpx.Response(500))
    for extra in ({"negative_prompt": "blurry"}, {"sampler": "euler"}, {"strength": 0.5}):
        resp = _post_generate(
            client, prompt = "a sloth", provider_id = "conn-1", model = "qwen-image", **extra
        )
        assert resp.status_code == 400, (extra, resp.text)
    assert sent == []


def test_connection_images_are_marked_in_the_gallery(client, monkeypatch):
    import base64
    import io

    import httpx
    from PIL import Image

    def handler(req):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format = "PNG")
        return httpx.Response(
            200, json = {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        )

    _external_image_connection(monkeypatch, handler)
    resp = _post_generate(client, prompt = "a sloth", provider_id = "conn-1", model = "qwen-image")
    assert resp.status_code == 200, resp.text
    assert resp.json()["images"][0]["model_kind"] == "connection"


def test_a_connection_run_reports_active_even_when_local_progress_is_hidden(client, monkeypatch):
    import hub.services.models.account_access as account_access

    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "generation_is_foreign", lambda kind: True)
    assert client.get("/api/inference/images/generate-progress").json()["active"] is False
    from utils.account_context import current_account_id

    monkeypatch.setitem(inference_routes._external_image_runs, current_account_id(), {object()})
    progress = client.get("/api/inference/images/generate-progress").json()
    assert progress["active"] is True
    assert progress["target"] == "connection"


def test_a_connection_call_leaves_the_media_tracker(client, monkeypatch):
    import base64
    import io

    import httpx
    from PIL import Image

    import core.inference.llama_keepwarm as kw
    import core.inference.media_keepwarm as mk

    client.app.add_middleware(kw.LlamaKeepWarmMiddleware)
    seen = []

    def handler(req):
        seen.append(mk._TRACKERS[mk.DIFFUSION].outstanding())
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format = "PNG")
        return httpx.Response(
            200, json = {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        )

    _external_image_connection(monkeypatch, handler)
    before = mk._TRACKERS[mk.DIFFUSION].outstanding()
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "provider_id": "conn-1", "model": "qwen-image"},
        headers = {"Authorization": "Bearer session"},
    )
    assert resp.status_code == 200, resp.text
    assert seen == [before]
    assert mk._TRACKERS[mk.DIFFUSION].outstanding() == before


def test_stop_after_the_connection_answered_is_not_acknowledged(client, monkeypatch):
    import asyncio

    from utils.account_context import current_account_id

    async def run():
        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        task = asyncio.ensure_future(done)
        await task
        inference_routes._external_image_runs[current_account_id()] = {task}
        try:
            return inference_routes._stop_external_image_runs()
        finally:
            inference_routes._external_image_runs.clear()

    assert asyncio.run(run()) is False


def test_connection_png_parameters_leave_out_unsent_settings():
    from core.inference.image_gallery import _params_text

    text = _params_text(
        {
            "prompt": "a sloth",
            "steps": 9,
            "guidance": 0.0,
            "seed": 123,
            "width": 512,
            "height": 256,
            "model": "qwen-image",
            "model_kind": "connection",
        }
    )
    assert text == "a sloth\nSize: 512x256, Model: qwen-image"
    local = _params_text({"prompt": "a sloth", "steps": 9, "guidance": 0.0, "seed": 123})
    assert "Steps: 9" in local and "Seed: 123" in local


def test_a_retired_flux3_full_copy_reaches_the_backend_as_its_base(client):
    # unsloth/flux-3-image-klein is a retired full copy: the plan and the load both see the base it copies, which the
    # real backend fetches through unsloth/flux-3-image-klein-FP8.
    from core.inference.diffusion_families import detect_family

    backend = diffusion_module.get_diffusion_backend()
    seen: list[tuple] = []

    def _validate(model_path, **kwargs):
        seen.append(("validate", model_path, kwargs.get("gguf_filename"), kwargs.get("model_kind")))
        return detect_family(model_path)

    plan, load = backend.download_plan, backend.begin_load
    backend.validate_load_request = _validate
    backend.download_plan = lambda model_path, **kw: (
        seen.append(("plan", model_path)),
        plan(model_path, **kw),
    )[1]
    backend.begin_load = lambda model_path, **kw: (
        seen.append(("load", model_path)),
        load(model_path, **kw),
    )[1]
    body = {
        "model_path": "unsloth/flux-3-image-klein",
        "gguf_filename": "model.safetensors",
        "model_kind": "single_file",
    }
    assert _post_download_plan(client, **body).status_code == 200
    assert _post_load(client, **body).status_code == 200
    bases = {entry[1] for entry in seen}
    assert bases == {"black-forest-labs/flux-3-image-klein"}, seen
    assert ("plan", "black-forest-labs/flux-3-image-klein") in seen
    assert ("load", "black-forest-labs/flux-3-image-klein") in seen
    validated = [entry[2:] for entry in seen if entry[0] == "validate"]
    assert validated and set(validated) == {(None, "pipeline")}, seen


@pytest.fixture
def token_client(client, monkeypatch):
    """The client, with begin_load publishing a state and moving the load epoch as the real engines do."""
    from core.inference import media_load_ids

    media_load_ids._reset_for_tests()
    backend = diffusion_module.get_diffusion_backend()
    begin, unload = backend.begin_load, backend.unload
    backend._load_token, backend._state, backend.unloads = 0, None, 0

    def begin_load(model_path, **kwargs):
        backend._load_token += 1
        backend._state = object()
        return begin(model_path, **kwargs)

    def counted_unload(**kwargs):
        only_if = kwargs.get("only_if")
        if only_if is not None and not only_if():
            return None
        backend.unloads += 1
        backend._load_token += 1
        backend._state = None
        return unload()

    monkeypatch.setattr(backend, "begin_load", begin_load)
    monkeypatch.setattr(backend, "unload", counted_unload)
    yield client, backend
    media_load_ids._reset_for_tests()


def _load_with_token(client, token):
    body = {
        "model_path": "unsloth/Z-Image-Turbo-GGUF",
        "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
    }
    return _post_load(client, **body, **({"load_token": token} if token else {}))


def test_an_image_unload_naming_the_load_token_unloads_that_load(token_client):
    client, backend = token_client
    assert _load_with_token(client, "t1").status_code == 200
    resp = client.post("/api/inference/images/unload", json = {"expected_load_id": "t1"})
    assert resp.status_code == 200
    assert (resp.json()["outcome"], resp.json()["loaded"]) == ("unloaded", False)
    assert backend.unloads == 1


@pytest.mark.parametrize("newer", [None, "t2"])
def test_an_image_unload_naming_a_stale_token_leaves_the_newer_load_alone(token_client, newer):
    client, backend = token_client
    assert _load_with_token(client, "t1").status_code == 200
    assert _load_with_token(client, newer).status_code == 200
    resp = client.post("/api/inference/images/unload", json = {"expected_load_id": "t1"})
    assert resp.json() == {"unloaded": False, "outcome": "superseded"}
    assert backend.unloads == 0 and backend.loaded


def test_an_image_unload_naming_a_load_not_started_yet_refuses_that_load(token_client):
    client, backend = token_client
    resp = client.post("/api/inference/images/unload", json = {"expected_load_id": "t1"})
    assert resp.json() == {"unloaded": False, "outcome": "cancelled"}
    refused = _load_with_token(client, "t1")
    assert refused.status_code == 409 and "cancelled" in refused.json()["detail"]
    assert backend.unloads == 0 and not backend.loaded


def test_image_status_names_the_load_an_unload_can_name(token_client):
    client, backend = token_client
    assert _load_with_token(client, "t1").status_code == 200
    assert client.get("/api/inference/images/status").json()["load_id"] == "t1"


def test_image_status_and_load_id_name_the_same_load(token_client):
    import threading

    from core.inference import media_load_ids
    from hub.services.models import account_access

    client, backend = token_client
    assert _load_with_token(client, "t1").status_code == 200
    read = backend.status
    admitted = threading.Event()

    def newer_load():
        with account_access.media_load_lock:
            prior = media_load_ids.admitting("diffusion", backend, "t2")
            backend._load_token += 1
            backend.loading = ("unsloth/Z-Image-Turbo-GGUF",)
            media_load_ids.admitted("diffusion", backend, "t2", prior)
            admitted.set()

    reads = []

    def racing_status():
        # The first read is slow and a load is admitted meanwhile, which it does not hold up.
        reads.append(True)
        if len(reads) == 1:
            worker = threading.Thread(target = newer_load)
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
        return read()

    backend.status = racing_status
    try:
        answer = client.get("/api/inference/images/status").json()
    finally:
        backend.status = read
    assert admitted.wait(5)
    # The read that straddled the admission was read again, so the id is that of the state it shows.
    assert answer["load_id"] == "t2" and len(reads) == 2
    backend.loading = ()


def test_the_image_unload_answers_are_declared_in_the_schema(token_client):
    import json

    client, backend = token_client
    schema = client.app.openapi()
    answer = schema["paths"]["/api/inference/images/unload"]["post"]["responses"]["200"]
    refs = json.dumps(answer)
    for model in ("DiffusionLoadUnloadedResponse", "MediaUnloadOutcome", "DiffusionStatusResponse"):
        assert f"#/components/schemas/{model}" in refs
    assert _load_with_token(client, "t1").status_code == 200
    plain = client.post("/api/inference/images/unload").json()
    assert set(plain) == set(inference_routes.DiffusionStatusResponse.model_fields)
    assert _load_with_token(client, "t2").status_code == 200
    named = client.post("/api/inference/images/unload", json = {"expected_load_id": "t2"}).json()
    assert set(named) == set(plain) | {"unloaded", "outcome"}
    stale = client.post("/api/inference/images/unload", json = {"expected_load_id": "t2"}).json()
    assert stale == {"unloaded": False, "outcome": "superseded"}


def test_zero_weight_loras_neither_refuse_mlx_nor_route_away_from_it():
    # Weight 0 disables an adapter and every loader drops it, so an all-zero list is no LoRA load.
    from fastapi import HTTPException
    from models.inference import DiffusionLoadRequest, LoraSpec

    off = DiffusionLoadRequest(model_path = "x/y", loras = [LoraSpec(id = "a", weight = 0.0)])
    inference_routes._refuse_mlx_loras(off)
    assert inference_routes._has_active_loras(off.loras) is False
    on = DiffusionLoadRequest(
        model_path = "x/y", loras = [LoraSpec(id = "a", weight = 0.0), LoraSpec(id = "b", weight = 0.5)]
    )
    assert inference_routes._has_active_loras(on.loras) is True
    with pytest.raises(HTTPException):
        inference_routes._refuse_mlx_loras(on)
