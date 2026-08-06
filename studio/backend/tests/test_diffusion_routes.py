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
        # Repo ids of in-flight (uncommitted) loads. The unload route reads this to keep DIFFUSION ownership during a concurrent load.
        self.loading: tuple = ()

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
        # Mirror the real backend cheap validation so the route validate-before-evict ordering is exercised.
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
        # A client-supplied base_repo clears the same trust bar as the real backend, so the route rejects an untrusted companion base.
        if base_repo and base_repo.strip() and not base_repo.lower().startswith("unsloth/"):
            raise ValueError(
                f"base_repo is restricted to unsloth/* repos (or a local path); got '{base_repo}'."
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
        prompts = None,
        seeds = None,
        **kwargs,
    ):
        if not self.loaded:
            raise RuntimeError("No diffusion model is loaded.")
        if prompts is not None or seeds is not None:
            # List-driven batch: the LIST sets the image count and each image's own seed (batch_size is only a per-forward cap).
            base = seeds[0] if seeds else (seed if seed is not None else 4242)
            count = len(prompts) if prompts is not None else len(seeds)
            per_image = seeds if seeds is not None else [base + i for i in range(count)]
            return {
                "images": [object() for _ in range(count)],
                "seed": base,
                "seeds": list(per_image),
                "repo_id": "x/z-image",
            }
        # The real backend returns PIL images and the route persists them; the fake returns sentinels since image_gallery is stubbed.
        return {
            "images": [object() for _ in range(batch_size)],
            "seed": seed if seed is not None else 4242,
            "repo_id": "x/z-image",
            # The real backend reports the workflow it resolved; the recipe records it.
            "workflow": (
                "inpaint"
                if kwargs.get("mask_image")
                else ("img2img" if kwargs.get("init_image") else "txt2img")
            ),
        }

    def generate_progress(self):
        # Idle by default; the persist-window override lives in the route, not here.
        return {"active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None}

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
    # Neutralise the engine router so the routes drive this fake diffusers backend regardless of host, and never attempt a native install.
    import core.inference.diffusion_engine_router as engine_router

    # Delegate to whatever get_diffusion_backend returns, so per-test re-patches still flow through the routes.
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
    # Isolate from the real GPU arbiter: reset ownership and stub the evictors so acquire_for() never touches live singletons.
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    # In-memory gallery backed by tmp files, so routes exercise persistence wiring without PIL/real disk under studio_root.
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
    ):
        ordered = sorted(store.values(), key = lambda r: r.get("created_at", 0.0), reverse = True)
        if valid is not None:
            ordered = [r for r in ordered if valid(r)]
        return ordered[offset:] if limit is None else ordered[offset : offset + limit]

    monkeypatch.setattr(gallery_module, "list_images", _list_images)
    monkeypatch.setattr(
        gallery_module,
        "image_path",
        lambda i: (tmp_path / f"{i}.png") if i in store else None,
    )
    # The serve route resolves through owned_image_path; the fake store holds only owned records, so an unknown stem is refused.
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
    return TestClient(app)


def test_load_generate_status_unload_roundtrip(client):
    loaded = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
            "base_repo": "unsloth/Z-Image-base",
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


def test_gallery_serve_refuses_unowned_id(client):
    # The serve route resolves through the ownership guard, so a guessed stem is a 404, not a stream of foreign bytes.
    assert client.get("/api/inference/images/gallery/family-photo/file").status_code == 404


def test_generate_holds_progress_active_during_persist(client, monkeypatch):
    # generate-progress must stay active while a finished generation is still writing its gallery record. Probe the persist counter from inside save.
    import core.inference.image_gallery as gallery_module
    import routes.inference as inf

    client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
            "base_repo": "unsloth/Z-Image-base",
        },
    )

    # Idle before any generation.
    assert client.get("/api/inference/images/generate-progress").json()["active"] is False

    seen = {}
    real_save = gallery_module.save

    def _probe_save(image, meta):
        seen["during"] = inf._diffusion_persist_active
        return real_save(image, meta)

    monkeypatch.setattr(gallery_module, "save", _probe_save)

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200
    # Active while the record was being persisted, and back to idle once the route returned.
    assert seen["during"] >= 1
    assert inf._diffusion_persist_active == 0
    assert client.get("/api/inference/images/generate-progress").json()["active"] is False


def test_load_rejects_untrusted_base_repo(client):
    # A trusted GGUF paired with an untrusted remote base_repo is rejected at the route, so a client cannot make the server fetch an arbitrary companion repo.
    r = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
            "base_repo": "evil/companions",
        },
    )
    assert r.status_code == 400
    assert "base_repo" in r.json()["detail"]
    assert client.get("/api/inference/images/status").json()["loaded"] is False


def test_unload_keeps_ownership_when_a_model_is_still_resident(client, monkeypatch):
    # The unload route must drop DIFFUSION ownership only when nothing is resident: releasing over a concurrent load would let a later chat load skip eviction and OOM.
    backend = diffusion_module.get_diffusion_backend()
    gpu_arbiter._owner = gpu_arbiter.DIFFUSION

    # Simulate a concurrent load having re-loaded: unload leaves the engine resident.
    backend.loaded = True
    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": True})
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() == gpu_arbiter.DIFFUSION  # ownership retained

    # The normal case (nothing resident after unload) still releases ownership.
    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": False})
    backend.loaded = False
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() is None


def test_unload_keeps_ownership_when_a_load_is_in_flight(client, monkeypatch):
    # A concurrent /images/load re-acquires DIFFUSION but is not is_loaded yet, so ownership must be kept on the in-flight state alone.
    backend = diffusion_module.get_diffusion_backend()
    gpu_arbiter._owner = gpu_arbiter.DIFFUSION

    backend.loaded = False
    backend.loading = ("unsloth/z-image-turbo",)
    monkeypatch.setattr(backend, "unload", lambda: {**_unloaded_status(), "loaded": False})
    r = client.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert gpu_arbiter.current_owner() == gpu_arbiter.DIFFUSION  # ownership retained for the load

    backend.loading = ()


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


def test_generate_seed_list_records_replay_from_each_own_seed(client):
    # A seeds LIST sets each image's own seed, so the recipe must NOT claim the base seed + request batch_size: restore prefers batch_seed.
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "seeds": [5, 99]},
    )
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert [i["seed"] for i in images] == [5, 99]
    assert [i["batch_seed"] for i in images] == [5, 99]  # replays THIS image, not the base
    assert [i["batch_size"] for i in images] == [1, 1]  # as a single image, not a batch


def test_generate_prompt_list_records_each_prompt_and_seed(client):
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "unused", "prompts": ["a cat", "a dog"], "seed": 10},
    )
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert [i["prompt"] for i in images] == ["a cat", "a dog"]
    assert [i["seed"] for i in images] == [10, 11]
    assert [i["batch_seed"] for i in images] == [10, 11]
    assert [i["batch_size"] for i in images] == [1, 1]


def test_generate_legacy_batch_still_records_the_base_seed_and_size(client):
    # The batch_size path is unchanged: those images DO share one base seed, so restore replays the whole batch.
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "batch_size": 3, "seed": 5},
    )
    images = resp.json()["images"]
    assert all(i["batch_seed"] == 5 for i in images)
    assert all(i["batch_size"] == 3 for i in images)
    assert [i["batch_index"] for i in images] == [0, 1, 2]


def test_generate_request_rejects_zero_denoise_strength():
    # strength 0 does NOT keep the source: it leaves zero denoising steps (FLUX/Qwen/Z-Image raise, SDXL crashes), so reject it as a 422.
    import pydantic

    from models.inference import DiffusionGenerateRequest

    with pytest.raises(pydantic.ValidationError):
        DiffusionGenerateRequest(prompt = "x", strength = 0.0)
    assert DiffusionGenerateRequest(prompt = "x", strength = 0.1).strength == 0.1
    assert DiffusionGenerateRequest(prompt = "x", strength = 1.0).strength == 1.0
    assert DiffusionGenerateRequest(prompt = "x").strength is None  # unset stays the pipe default


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
    # Odd, and a multiple of 8 that is not a multiple of 16: both rejected, since Z-Image requires dimensions divisible by 16.
    for bad in (1001, 1000):
        resp = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": bad})
        assert resp.status_code == 422, bad
    # A multiple of 16 is accepted.
    ok = client.post("/api/inference/images/generate", json = {"prompt": "p", "width": 1024})
    assert ok.status_code == 200


def test_generate_rejects_batch_seed_past_json_safe_range(client):
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    # A seed at the cap with a batch derives per-image seeds past the JSON-safe range, so the request is rejected.
    over = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "seed": 2**53 - 1, "batch_size": 2},
    )
    assert over.status_code == 422
    # The top-of-batch seed lands exactly on the cap: still JSON-safe, so accepted.
    ok = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "seed": 2**53 - 2, "batch_size": 2},
    )
    assert ok.status_code == 200


def test_non_gguf_load_restricted_to_unsloth(client):
    # gguf_filename is optional; with none the load is a full-pipeline kind gated to unsloth/*, so a non-unsloth repo is a 400.
    resp = client.post("/api/inference/images/load", json = {"model_path": "x/z-image"})
    assert resp.status_code == 400
    assert "unsloth" in resp.json()["detail"].lower()


def test_a_too_old_diffusers_is_a_400_on_both_load_and_download_plan(client, monkeypatch):
    # An unbuildable family is an unloadable pick, so it is a 400 with the message intact on both routes. As a RuntimeError it
    # reached /images/load's 409 ("already in progress") and escaped /images/download-plan as a bare 500 with the message lost.
    import sys
    import types

    from core.inference.diffusion_families import assert_pipeline_class_available

    backend = diffusion_module.get_diffusion_backend()

    def _refuse(model_path, **kwargs):
        # The real gate, run against a diffusers that predates the class.
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
    # An unsloth/* repo with no filename loads as a full diffusers pipeline, so the route forwards model_kind="pipeline".
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
    # A loaded model that fails mid-pipeline (CUDA OOM, a RuntimeError) is a server failure: 500 with FIXED text, not a 409.
    # The class of failure is named so the page can suggest something; the engine's own text can carry local paths and argv.
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
    # What a Metal host hits: the native renderer aborts inside its text encoder. The page now says which component died, with the backtrace left in the log.
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


def test_generate_execution_error_with_cancelled_substring_is_sanitized_500(client, monkeypatch):
    # A native execution failure whose raw tail merely CONTAINS "cancelled" must stay a sanitized 500, not misroute to 409.
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
    # Validation runs in the pre-flight (before the GPU is taken), so that is where an unsupported model is rejected now.
    backend.validate_load_request = _raise
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/y", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 400
    assert "isn't a supported image-generation model" in resp.json()["detail"]


def test_load_validation_failure_does_not_evict_chat(client, monkeypatch):
    # A rejected image-model pick must not tear down the loaded chat model: validation runs before acquire_for.
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
    # An image load while training is active is refused (409) before the GPU is taken.
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
    # An undetectable family fails validation BEFORE the GPU handoff, so the arbiter is never acquired.
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
    # The field help must match the code: UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH is a directory allowlist, not a =1 toggle.
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
    # Stubbed because select_and_activate_engine probes THIS first with allow_install on. Unstubbed it ran the real installer,
    # downloading 108 MB into the developer's own ~/.unsloth root. Returning None also keeps this test on the sd-cli path.
    monkeypatch.setattr(engine_router, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(engine_router, "ensure_sd_cpp_binary", lambda **_: "/x/sd-cli")
    # The router probes runnability before committing to native; treat the stub binary as executable.
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
    # An unsupported transformer_quant is rejected by the request schema, so the GPU is never acquired.
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf", "transformer_quant": "int2"},
    )
    assert resp.status_code == 422
    assert gpu_arbiter._owner is None


def test_invalid_memory_mode_returns_422_without_eviction(client):
    # An unsupported memory_mode is rejected by the request schema, so the GPU is never acquired.
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
    # Pin the resolved device to cuda: the route only takes the arbiter for non-CPU loads.
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

    def _fake_acquire(role, register = None):
        # Mirror the real arbiter: record the handoff and run the (registered) load under it.
        acquired.append(role)
        return register() if register is not None else None

    monkeypatch.setattr(gpu_arbiter, "acquire_for", _fake_acquire)
    return acquired


def test_cpu_native_load_skips_gpu_arbiter(client, monkeypatch):
    # A native sd.cpp load on a pure-CPU host never touches the GPU, so the route must NOT evict the resident chat model.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cpu")
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 200
    assert acquired == []  # no arbiter handoff for a CPU native load


def test_gpu_native_load_takes_arbiter(client, monkeypatch):
    # A force-native sd.cpp load on a GPU box DOES use the GPU, so the arbiter is acquired, like the always-GPU diffusers path.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    backend = diffusion_module.get_diffusion_backend()
    acquired = _force_engine(monkeypatch, backend, engine_name = ENGINE_SD_CPP, device = "cuda")
    resp = client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    assert resp.status_code == 200
    assert acquired == [gpu_arbiter.DIFFUSION]


def test_images_info_lists_every_family(client):
    # The pure info endpoint is hardware-independent: one entry per auto-policy family with the quant estimates the UI shows.
    from core.inference.diffusion_auto_policy import _FAMILY_BF16_GB

    resp = client.get("/api/inference/images/info")
    assert resp.status_code == 200
    families = resp.json()["families"]
    assert {f["family"] for f in families} == set(_FAMILY_BF16_GB)
    sample = families[0]
    est = sample["estimated_resident_gb"]
    # Quantised estimates undercut bf16, and nvfp4 undercuts int8 (matching the pure helper).
    assert est["int8"] < est["bf16"]
    assert est["nvfp4"] < est["int8"]


def test_status_passes_through_resolved(client, monkeypatch):
    # The additive `resolved` provenance record round-trips through the status route so the frontend can render the "Auto: X" badges.
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
    # A backend status without a `resolved` key leaves the additive field null (older backends and the unloaded state).
    body = client.get("/api/inference/images/status").json()
    assert body["resolved"] is None


def test_download_plan_forwards_the_load_time_controls(client, monkeypatch):
    # The plan drives the staged download, so it must be computed from the SAME configuration the load will run with: the
    # prefetch decision reads the memory policy, prequant path and adapter selection as well as speed/quant.
    from core.inference import diffusion_engine_router as router
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    # This test is about WHICH kwargs reach the planner, not which planner is picked, so pin the engine: the pick above is a
    # GGUF one, and on a GPU-less runner that routes to native sd.cpp, whose planner is a different object than the stub below.
    # Left to the host, the assertions passed on a GPU box and died with a bare KeyError on CI. Engine SELECTION is tested next.
    monkeypatch.setattr(router, "predict_engine", lambda fam, **_: ENGINE_DIFFUSERS)
    backend = diffusion_module.get_diffusion_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen["model_path"] = model_path
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = client.post(
        "/api/inference/images/download-plan",
        json = {
            "model_path": "unsloth/FLUX.1-dev-GGUF",
            "gguf_filename": "flux1-dev-Q4_K_M.gguf",
            "model_kind": "gguf",
            "hf_token": "hf_secret",
            "speed_mode": "off",
            "transformer_quant": "int8",
            "memory_mode": "low_vram",
            "cpu_offload": True,
            "loras": [{"id": "unsloth/some-lora", "weight": 0.8}],
        },
    )

    assert resp.status_code == 200
    assert seen["hf_token"] == "hf_secret"
    assert seen["speed_mode"] == "off"
    assert seen["transformer_quant"] == "int8"
    assert seen["memory_mode"] == "low_vram"
    assert seen["cpu_offload"] is True
    assert len(seen["loras"] or []) == 1


def test_download_plan_uses_the_engine_the_load_will_pick(client, monkeypatch):
    # On a host with no usable GPU a GGUF pick routes to native sd.cpp, which reads single-file assets and never opens the base
    # repo's sharded components. Planning with diffusers there staged GB the load discards and pulled the rest inline.
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

    resp = client.post(
        "/api/inference/images/download-plan",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_M.gguf",
            "model_kind": "gguf",
            "hf_token": "hf_secret",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["total_bytes"] == 7
    # The native planner gets the same identity + token the load would use.
    assert seen["model_path"] == "unsloth/Z-Image-Turbo-GGUF"
    assert seen["gguf_filename"] == "z-image-turbo-Q4_K_M.gguf"
    assert seen["hf_token"] == "hf_secret"


def test_download_plan_stays_on_diffusers_when_the_load_will(client, monkeypatch):
    # The mirror of the above: a GPU host (or any non-GGUF kind) loads through diffusers, so the plan keeps the diffusers set.
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

    resp = client.post(
        "/api/inference/images/download-plan",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-turbo-Q4_K_M.gguf",
            "model_kind": "gguf",
        },
    )
    assert resp.status_code == 200 and resp.json()["total_bytes"] == 11


def test_load_refused_when_only_the_diffusion_probe_can_be_read(client, monkeypatch):
    # The two training probes are independent: an LLM backend that raises used to short-circuit the guard, letting an image load sail past a KNOWN-active diffusion trainer.
    import core.training as core_training
    import routes.inference as inference_routes

    class _Broken:
        def is_training_active(self):
            raise RuntimeError("training backend unavailable")

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Broken())
    monkeypatch.setattr(inference_routes, "_diffusion_training_active", lambda: True)

    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()

    # With neither trainer active the unreadable LLM probe still must not block the load.
    monkeypatch.setattr(inference_routes, "_diffusion_training_active", lambda: False)
    resp = client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 200


def test_recipe_records_the_conditioned_workflow_settings(client, monkeypatch):
    # A conditioned generation recipe used to carry only the txt2img fields, so the gallery presented an inpaint result as a
    # complete Create recipe. The images are still not persisted, but what ran IS, so the client can name the inputs to re-add.
    import base64
    import io

    from PIL import Image

    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 30, 90)).save(buf, format = "PNG")
    px = base64.b64encode(buf.getvalue()).decode()
    gen = client.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "a sloth",
            "seed": 7,
            "init_image": px,
            "mask_image": px,
            "strength": 0.42,
        },
    )
    assert gen.status_code == 200
    img = gen.json()["images"][0]
    assert img["workflow"] == "inpaint"
    assert img["strength"] == 0.42
    # A plain txt2img still records its own workflow and leaves the conditioning fields empty.
    plain = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert plain.status_code == 200
    plain_img = plain.json()["images"][0]
    assert plain_img["workflow"] == "txt2img"
    assert plain_img["strength"] is None and plain_img["upscale"] is None


def test_recipe_records_the_load_time_build(client, monkeypatch):
    # A recipe naming only the repo id cannot rebuild the pipeline that made the image: a GGUF repo holds many quants, and a
    # torchao load bakes its adapters in before quantize + compile, which is not the adapter-less build even when disabled.
    backend = diffusion_module.get_diffusion_backend()

    def _generate(**kwargs):
        return {
            "images": [object()],
            "seed": 777,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "model_kind": "gguf",
            "gguf_filename": "z-image-turbo-Q8_0.gguf",
            "transformer_quant": "int8",
            # Baked at LOAD time; the generate request below carries no adapters, so the applied set is empty.
            "baked_loras": ["bakedlora"],
            "active_loras": [],
            "workflow": "txt2img",
        }

    monkeypatch.setattr(backend, "generate", _generate, raising = False)

    client.post(
        "/api/inference/images/load",
        json = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"},
    )
    resp = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 777})
    assert resp.status_code == 200
    img = resp.json()["images"][0]
    assert img["model"] == "unsloth/Z-Image-Turbo-GGUF"
    assert img["model_kind"] == "gguf"
    assert img["gguf_filename"] == "z-image-turbo-Q8_0.gguf"
    assert img["transformer_quant"] == "int8"
    # The bake is recorded even though nothing was applied to THIS generation.
    assert img["baked_loras"] == ["bakedlora"]
    assert img["loras"] == []


def test_recipe_build_fields_absent_on_an_engine_that_omits_them(client):
    # The native path and older records report no build keys; the record must degrade to nulls rather than 500 the persist.
    client.post(
        "/api/inference/images/load", json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"}
    )
    resp = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert resp.status_code == 200
    img = resp.json()["images"][0]
    assert img["model_kind"] is None
    assert img["gguf_filename"] is None
    assert img["transformer_quant"] is None
    assert img["baked_loras"] == []


def test_gallery_image_accepts_a_record_written_before_the_build_fields():
    # Existing PNGs carry none of the build keys, and list_gallery_images DROPS records that fail validation, so a non-optional addition would empty a gallery.
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
