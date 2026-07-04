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
        model_kind = None,
    ):
        # Mirror the real backend's cheap validation so the route's
        # validate-before-evict ordering is exercised.
        from core.inference.diffusion import resolve_model_kind
        from core.inference.diffusion_families import detect_family

        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("a single-file checkpoint name is required.")
        # Non-GGUF loads are gated to unsloth/* (or a local path), like the real backend.
        if kind != "gguf" and not model_path.lower().startswith("unsloth/"):
            raise ValueError(
                f"Non-GGUF diffusion loads are restricted to unsloth/* repos; got '{model_path}'."
            )
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
    # Neutralise the engine router so the routes deterministically drive this fake
    # (diffusers) backend regardless of the host's real device, and never attempt a
    # native sd.cpp install/download. The router's selection logic is covered in
    # test_diffusion_engine_router.py; one route-level sd_cpp test lives below.
    import core.inference.diffusion_engine_router as engine_router

    # Delegate to whatever get_diffusion_backend currently returns, so per-test
    # re-patches of the backend still flow through the routes.
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
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
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


def test_non_gguf_load_restricted_to_unsloth(client):
    # gguf_filename is optional now; with none, the load is a full-pipeline kind, which
    # is gated to unsloth/* repos. A non-unsloth repo (no filename) is rejected -> 400.
    resp = client.post("/api/inference/images/load", json = {"model_path": "x/z-image"})
    assert resp.status_code == 400
    assert "unsloth" in resp.json()["detail"].lower()


def test_pipeline_load_allowed_for_unsloth_repo(client):
    # An unsloth/* repo with no filename loads as a full diffusers pipeline (kind auto
    # = pipeline); the route forwards model_kind="pipeline" to begin_load.
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"}
    )
    assert resp.status_code == 200
    backend = diffusion_module.get_diffusion_backend()
    assert backend.last_load_kwargs["model_kind"] == "pipeline"
    assert backend.last_load_kwargs.get("gguf_filename") is None


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


def test_generate_execution_error_with_cancelled_substring_is_sanitized_500(client, monkeypatch):
    # A native sd-cli execution failure whose raw tail merely CONTAINS "cancelled"
    # must stay a sanitized 500, not misroute to 409 and echo that output (path/arg
    # leak). Regression: the handler matched "cancelled" as a substring.
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _fail(**kwargs):
        raise RuntimeError("sd-cli exited 1. Last output:\nop cancelled at /home/u/models/x.gguf")

    monkeypatch.setattr(backend, "generate", _fail)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Image generation failed."
    assert "cancelled" not in resp.json()["detail"] and "models" not in resp.json()["detail"]


def test_generate_user_cancellation_returns_409(client, monkeypatch):
    # The exact cancellation sentinel both engines raise is client-state (409).
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


def test_transformer_prequant_path_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "transformer_quant": "fp8",
            "transformer_prequant_path": "/data/zimage_fp8.pt",
        },
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_prequant_path") == "/data/zimage_fp8.pt"


def test_attention_backend_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "attention_backend": "cudnn",
        },
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("attention_backend") == "cudnn"


def test_invalid_attention_backend_returns_422(client):
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "attention_backend": "bogus"},
    )
    assert resp.status_code == 422


def test_prequant_path_doc_describes_allowlist_not_toggle():
    # The field help must match the code: UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH is a
    # directory allowlist, not a =1 toggle (diffusion_prequant._allowed_prequant_roots
    # drops bare on/off tokens), so operators following the doc don't get every
    # request silently refused.
    from models.inference import DiffusionLoadRequest

    desc = DiffusionLoadRequest.model_fields["transformer_prequant_path"].description
    assert "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" in desc
    assert "=1" not in desc
    assert "allowlist" in desc.lower() or "director" in desc.lower()


def test_transformer_cache_threads_through(client, monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "transformer_cache": "fbcache",
            "transformer_cache_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    assert backend.last_load_kwargs.get("transformer_cache") == "fbcache"
    assert backend.last_load_kwargs.get("transformer_cache_threshold") == 0.1


def test_invalid_transformer_cache_returns_422(client):
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "transformer_cache": "deepcache",
        },
    )
    assert resp.status_code == 422


def test_out_of_range_cache_threshold_returns_422(client):
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "x/z-image",
            "gguf_filename": "q.gguf",
            "transformer_cache_threshold": 1.5,
        },
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

    validator = _FakeBackend()  # supplies validate_load_request (and is the diffusers fallback)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: validator)
    # Force the router's decision inputs: CPU device + an available binary.
    monkeypatch.setattr(
        engine_router,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = "cpu", device = "cpu"),
    )
    monkeypatch.setattr(engine_router, "ensure_sd_cpp_binary", lambda **_: "/x/sd-cli")
    # The router now probes runnability before committing to native; treat the stub
    # binary as executable.
    monkeypatch.setattr(
        engine_router, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0")
    )
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
    # The native backend the router will activate.
    sd_fake = _FakeBackend()
    monkeypatch.setattr(sd_backend, "get_sd_cpp_backend", lambda: sd_fake)

    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "z.gguf"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "sd_cpp"
    assert body["fallback_reason"] is None
    assert sd_fake.loaded is True  # the native engine actually received the load


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
    # Pin the resolved device to cuda: the route only takes the arbiter for non-CPU
    # loads, so on a CPU-only host the ownership assert below would never hold.
    import types as _types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod,
        "resolve_diffusion_device_target",
        lambda: _types.SimpleNamespace(device = "cuda"),
    )
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    # Validation passed first, so the GPU WAS acquired before begin_load reported busy.
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
    monkeypatch.setattr(gpu_arbiter, "acquire_for", lambda role: acquired.append(role))
    return acquired


def test_cpu_native_load_skips_gpu_arbiter(client, monkeypatch):
    # A native sd.cpp load on a pure-CPU host never touches the GPU, so the route must NOT
    # evict the resident chat model -- the arbiter handoff is skipped.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 200
    assert acquired == []  # no arbiter handoff for a CPU native load


def test_gpu_native_load_takes_arbiter(client, monkeypatch):
    # A force-native sd.cpp load on a GPU box DOES use the GPU, so the arbiter is acquired
    # (same as the always-GPU diffusers path).
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cuda")
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_images_info_lists_every_family(client):
    # The pure info endpoint is hardware-independent (no load required): it returns one
    # entry per auto-policy family, each with the quant estimates the UI shows.
    from core.inference.diffusion_auto_policy import _FAMILY_BF16_GB

    resp = client.get("/api/inference/images/info")
    assert resp.status_code == 200
    families = resp.json()["families"]
    assert {f["family"] for f in families} == set(_FAMILY_BF16_GB)
    sample = families[0]
    est = sample["estimated_resident_gb"]
    # Quantised estimates undercut bf16, and nvfp4 undercuts int8 (matches the pure helper).
    assert est["int8"] < est["bf16"]
    assert est["nvfp4"] < est["int8"]


def test_status_passes_through_resolved(client, monkeypatch):
    # The additive `resolved` provenance record round-trips through the status route so the
    # frontend can render the "Auto: X" badges.
    backend = diffusion_module.get_diffusion_backend()
    resolved = {
        "speed_mode": {"value": "eager", "source": "auto", "reason": "per-kind default"},
        "transformer_quant": {"value": "int8", "source": "explicit", "reason": "requested"},
        "cpu_offload": {"value": False, "source": "auto", "reason": "from the memory plan"},
        "transformer_cache": {"value": None, "source": "auto", "reason": "few-step model"},
    }
    monkeypatch.setattr(
        backend, "status", lambda: {**_unloaded_status(), "loaded": True, "resolved": resolved}
    )
    body = client.get("/api/inference/images/status").json()
    assert body["resolved"] == resolved
    assert body["resolved"]["speed_mode"]["source"] == "auto"
    # The cpu_offload value stays a real boolean (not coerced to a string).
    assert body["resolved"]["cpu_offload"]["value"] is False


def test_status_resolved_defaults_to_null(client):
    # A backend status without a `resolved` key leaves the additive field null (older
    # backends and the unloaded state).
    body = client.get("/api/inference/images/status").json()
    assert body["resolved"] is None
