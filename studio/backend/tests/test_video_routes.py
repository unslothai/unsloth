# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the text-to-video routes.

The video backend is replaced with a lightweight fake, so these exercise the
route wiring, validation, error mapping, and response shapes without torch,
diffusers, weights, or a GPU. The gallery persists to a real tmp directory
(via a patched gallery_dir), so the file/list/delete/clear paths run the actual
video_gallery code.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.video as video_module
import core.inference.video_gallery as gallery_module
from auth.authentication import get_current_subject
from core.inference.video_families import (
    VIDEO_CANCELLED_MSG,
    VIDEO_GENERATION_BUSY_MSG,
    VIDEO_NOT_LOADED_MSG,
)
from routes.video import router as video_router


def _defaults():
    return {
        "steps": 40,
        "guidance": 4.0,
        "num_frames": 121,
        "fps": 24,
        "frame_step": 8,
        "resolution_multiple": 32,
        "resolution_presets": [[768, 512], [1216, 704]],
    }


def _unloaded_status():
    return {
        "loaded": False,
        "repo_id": None,
        "family": None,
        "base_repo": None,
        "device": None,
        "dtype": None,
        "model_kind": None,
        "offload_policy": None,
        "vae_tiling": False,
        "memory_mode": None,
        "speed_mode": None,
        "speed_optims": [],
        "attention_backend": None,
        "transformer_cache": None,
        "transformer_quant": None,
        "text_encoder_quant": None,
        "has_audio": False,
        "defaults": None,
        "resolved": None,
    }


class _FakeBackend(video_module.VideoBackend):
    """Overrides the heavy load/generate/status surface but INHERITS the real
    begin_generate / _run_generate / generate_progress / cancel_generate job
    machinery, so the asynchronous generate contract (immediate accept, busy
    guard, terminal completed/failed state, cancel) is exercised for real."""

    def __init__(self) -> None:
        super().__init__()
        self.last_load_kwargs: dict = {}
        # Repo ids of in-flight (not yet committed) loads; empty tuple = none. The unload
        # route reads this to keep VIDEO ownership while a concurrent load is still loading.
        self.loading: tuple = ()

    # The real backend keys "loaded" off its committed pipeline state (_state); map
    # the fake's flag onto it so the inherited begin_generate sees the same thing.
    @property
    def loaded(self) -> bool:
        return self._state is not None

    @loaded.setter
    def loaded(self, value: bool) -> None:
        self._state = object() if value else None

    def loading_repo_ids(self) -> tuple:
        return tuple(self.loading)

    def validate_load_request(
        self,
        model_path,
        *,
        gguf_filename = None,
        base_repo = None,
        family_override = None,
        model_kind = None,
        transformer_quant = None,
        text_encoder_quant = None,
        vae_quant = None,
        transformer_cache_quality = None,
        cfg_parallel = None,
    ):
        # Mirror the real backend's cheap validation so the route's
        # validate-before-evict ordering is exercised.
        from pathlib import Path

        kind = (model_kind or ("gguf" if gguf_filename else "pipeline")).lower()
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        # Non-GGUF loads are gated to unsloth/* repos, the official bases, and existing local paths.
        trusted = model_path.lower().startswith(("unsloth/", "lightricks/")) or (
            Path(model_path).expanduser().exists()
        )
        if kind != "gguf" and not trusted:
            raise ValueError(
                f"Non-GGUF video loads are limited to unsloth/* repos, the official family "
                f"base repos, and local paths; '{model_path}' is neither."
            )
        if "ltx" not in model_path.lower() and family_override is None:
            raise ValueError(
                f"'{model_path}' is not a supported text-to-video model. Supported families: ltx-2."
            )
        return object()

    def begin_load(self, model_path, **kwargs):
        # The real backend loads on a thread; the fake completes instantly.
        self.loaded = True
        self.last_load_kwargs = dict(kwargs)
        return {
            **_unloaded_status(),
            "loaded": True,
            "repo_id": model_path,
            "family": "ltx-2",
            "base_repo": kwargs.get("base_repo") or "Lightricks/LTX-2",
            "device": "cpu",
            "dtype": "float32",
            "model_kind": kwargs.get("model_kind")
            or ("gguf" if kwargs.get("gguf_filename") else "pipeline"),
            "memory_mode": kwargs.get("memory_mode") or "auto",
            "has_audio": True,
            "defaults": _defaults(),
        }

    def load_progress(self):
        return {
            "phase": "ready" if self.loaded else None,
            "downloaded_bytes": 0,
            "expected_bytes": None,
            "error": None,
        }

    def generate(
        self,
        *,
        prompt,
        seed = None,
        cancel_event = None,
        **kwargs,
    ):
        if not self.loaded:
            raise RuntimeError(VIDEO_NOT_LOADED_MSG)
        # Recorded so route tests can assert request fields (init_image) reach the backend.
        self.last_generate_kwargs = dict(kwargs)
        return {
            "mp4_bytes": b"MP4-FAKE-BYTES",
            "seed": seed if seed is not None else 4242,
            "repo_id": "unsloth/LTX-2.3-GGUF",
            "width": kwargs.get("width") or 768,
            "height": kwargs.get("height") or 512,
            "num_frames": kwargs.get("num_frames") or 121,
            "fps": kwargs.get("fps") or 24,
            "duration_s": 5.0,
            "has_audio": True,
            "steps": kwargs.get("steps") or 40,
            "guidance": 4.0 if kwargs.get("guidance") is None else kwargs.get("guidance"),
        }

    def unload(self):
        self.loaded = False
        return _unloaded_status()

    def status(self):
        if not self.loaded:
            return _unloaded_status()
        return {
            **_unloaded_status(),
            "loaded": True,
            "repo_id": "unsloth/LTX-2.3-GGUF",
            "family": "ltx-2",
            "has_audio": True,
            "defaults": _defaults(),
        }


@pytest.fixture
def client(monkeypatch, tmp_path):
    backend = _FakeBackend()
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    # Isolate from the real GPU arbiter: reset ownership and stub the evictors so
    # the load route's acquire_for() never touches live backend singletons.
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.VIDEO, lambda: None)

    # Pin the resolved device to cpu so the load route deterministically follows the non-GPU branch
    # on any host; GPU-arbiter gating is asserted in its own tests by forcing the device to cuda.
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )

    # Persist to a real tmp gallery so save/list/file/delete/clear run the actual
    # video_gallery code (MP4 + JSON sidecar pair) without touching studio_root.
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path)

    app = FastAPI()
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _wait_terminal(client, timeout = 5.0) -> dict:
    """Poll generate-progress until the background job records a terminal phase.
    Generation is asynchronous now (the POST returns as soon as the job starts),
    so its outcome is only observable here."""
    deadline = time.monotonic() + timeout
    progress: dict = {}
    while time.monotonic() < deadline:
        progress = client.get("/api/inference/video/generate-progress").json()
        if progress.get("phase") in ("completed", "failed"):
            return progress
        time.sleep(0.01)
    raise AssertionError(f"generation never reached a terminal state: {progress}")


def _generate_and_wait(client, payload) -> dict:
    """Start a generation, assert the immediate accepted response, and return the
    saved gallery record the completed progress state carries."""
    resp = client.post("/api/inference/video/generate", json = payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started" and body["video"] is None
    progress = _wait_terminal(client)
    assert progress["phase"] == "completed", progress
    assert progress["active"] is False and progress["error"] is None
    return progress["video"]


def test_load_happy_path_and_arbiter_acquired(client, monkeypatch):
    # Force the device to cuda so the load takes the GPU arbiter, and record the acquire.
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    acquired: list = []

    def _fake_acquire(role, register = None):
        # Mirror the real arbiter: record the handoff and run the (registered) load under it.
        acquired.append(role)
        return register() if register is not None else None

    monkeypatch.setattr(gpu_arbiter, "acquire_for", _fake_acquire)

    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["loaded"] is True and body["family"] == "ltx-2"
    assert body["has_audio"] is True
    assert body["defaults"]["num_frames"] == 121
    assert acquired == [gpu_arbiter.VIDEO]  # the GPU was handed to VIDEO


def test_load_value_error_returns_400(client):
    # A non-ltx repo is not a supported family: the cheap validation rejects it -> 400.
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "x/some-image-model", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 400
    assert "supported text-to-video model" in resp.json()["detail"]
    # Validation runs before the arbiter handoff, so ownership is untouched.
    assert gpu_arbiter._owner is None


def test_load_threads_options_through_to_backend(client):
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "q.gguf",
            "memory_mode": "low_vram",
            "attention_backend": "cudnn",
            "transformer_cache": "fbcache",
            "transformer_cache_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    kwargs = video_module.get_video_backend().last_load_kwargs
    assert kwargs.get("memory_mode") == "low_vram"
    assert kwargs.get("attention_backend") == "cudnn"
    assert kwargs.get("transformer_cache") == "fbcache"
    assert kwargs.get("transformer_cache_threshold") == 0.1


def test_load_threads_transformer_quant_and_guidance_2(client):
    # The new load-time transformer_quant field reaches the backend, and the new
    # per-generation guidance_2 field reaches generate() (dual-DiT MoE second guidance).
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "q.gguf",
            "transformer_quant": "fp8",
        },
    )
    assert resp.status_code == 200
    kwargs = video_module.get_video_backend().last_load_kwargs
    assert kwargs.get("transformer_quant") == "fp8"

    video = _generate_and_wait(client, {"prompt": "a sloth", "guidance": 5.0, "guidance_2": 3.0})
    assert video["guidance"] == 5.0 and video["guidance_2"] == 3.0


def test_load_rejects_bad_transformer_quant_422(client):
    # transformer_quant is a Literal, so an unknown scheme is a 422 at request validation.
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "q.gguf",
            "transformer_quant": "bogus",
        },
    )
    assert resp.status_code == 422


def test_load_threads_text_encoder_quant(client):
    # The load-time text_encoder_quant field reaches the backend (the video path now
    # quantises the dense companion encoder, not just the DiT).
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "q.gguf",
            "text_encoder_quant": "fp8",
        },
    )
    assert resp.status_code == 200
    kwargs = video_module.get_video_backend().last_load_kwargs
    assert kwargs.get("text_encoder_quant") == "fp8"


def test_load_rejects_bad_text_encoder_quant_422(client):
    # text_encoder_quant is a Literal, so an unknown scheme is a 422 at request validation.
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "q.gguf",
            "text_encoder_quant": "bogus",
        },
    )
    assert resp.status_code == 422


def test_load_progress_route(client):
    idle = client.get("/api/inference/video/load-progress")
    assert idle.status_code == 200 and idle.json()["phase"] is None
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    ready = client.get("/api/inference/video/load-progress")
    assert ready.json()["phase"] == "ready"


def test_load_local_single_file_dir_routes_through_single_file(client, tmp_path):
    # A local video-family dir with one .safetensors and no model_index.json arrives as a pipeline
    # with no filename; the route reinterprets it as a single_file load of the sole checkpoint.
    d = tmp_path / "ltx-2.3-local"
    d.mkdir()
    (d / "ltx-dit.safetensors").write_bytes(b"0")
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": str(d), "model_kind": "pipeline"},
    )
    assert resp.status_code == 200
    kwargs = video_module.get_video_backend().last_load_kwargs
    assert kwargs["model_kind"] == "single_file"
    assert kwargs["gguf_filename"] == "ltx-dit.safetensors"


def test_load_local_pipeline_dir_stays_pipeline(client, tmp_path):
    # A real diffusers directory (has model_index.json) is left as a pipeline load.
    d = tmp_path / "ltx-2.3-pipeline"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"0")
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": str(d), "model_kind": "pipeline"},
    )
    assert resp.status_code == 200
    kwargs = video_module.get_video_backend().last_load_kwargs
    assert kwargs["model_kind"] == "pipeline"
    assert not kwargs.get("gguf_filename")


def test_generate_happy_path_persists_and_reports_record(client):
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    # The POST returns at once ("started"); the saved record arrives through the
    # generate-progress terminal state (asserted inside the helper).
    video = _generate_and_wait(client, {"prompt": "a sloth surfing", "seed": 7})
    assert video["seed"] == 7 and video["prompt"] == "a sloth surfing" and video["id"]
    assert video["has_audio"] is True
    assert video["model"] == "unsloth/LTX-2.3-GGUF"
    assert video["url"].endswith(f"/gallery/{video['id']}/file")
    assert video["created_at"]  # ISO timestamp string

    # The clip is now listable and fetchable as MP4 bytes.
    listed = client.get("/api/inference/video/gallery").json()["videos"]
    assert [v["id"] for v in listed] == [video["id"]]
    fetched = client.get(video["url"])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "video/mp4"
    assert "immutable" in fetched.headers["cache-control"]
    assert fetched.content == b"MP4-FAKE-BYTES"


def test_generate_threads_init_image_to_backend(client):
    # The image-to-video source image (init_image) must ride the request through
    # begin_generate into the backend generate call untouched.
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    data_url = "data:image/png;base64,aGk="
    _generate_and_wait(client, {"prompt": "a sloth", "seed": 3, "init_image": data_url})
    backend = video_module.get_video_backend()
    assert backend.last_generate_kwargs.get("init_image") == data_url


def test_generate_without_load_returns_409(client):
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == VIDEO_NOT_LOADED_MSG


def test_generate_cancelled_reports_failed_with_sentinel(client, monkeypatch):
    # A cancel mid-run surfaces as the job's terminal failed state carrying the exact
    # sentinel (the frontend suppresses the toast on it), not as an HTTP error.
    backend = video_module.get_video_backend()
    backend.loaded = True

    def _cancel(**kwargs):
        raise RuntimeError(VIDEO_CANCELLED_MSG)

    monkeypatch.setattr(backend, "generate", _cancel)
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 200
    progress = _wait_terminal(client)
    assert progress["phase"] == "failed"
    assert progress["error"] == VIDEO_CANCELLED_MSG
    assert progress["active"] is False


def test_generate_pipeline_error_reports_sanitized_failure(client, monkeypatch):
    # A loaded model that fails mid-pipeline (CUDA OOM) is a server failure: the job's
    # terminal state carries a generic message, never the raw exception.
    backend = video_module.get_video_backend()
    backend.loaded = True

    def _oom(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 40.00 GiB")

    monkeypatch.setattr(backend, "generate", _oom)
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 200
    progress = _wait_terminal(client)
    assert progress["phase"] == "failed"
    assert progress["error"] == "Video generation failed."
    assert "CUDA" not in progress["error"]


def test_generate_value_error_reports_reason(client, monkeypatch):
    # Bad client input is feedback: the terminal failed state carries the reason.
    backend = video_module.get_video_backend()
    backend.loaded = True

    def _bad(**kwargs):
        raise ValueError("negative_prompt is not supported by this family.")

    monkeypatch.setattr(backend, "generate", _bad)
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 200
    progress = _wait_terminal(client)
    assert progress["phase"] == "failed"
    assert "not supported" in progress["error"]


def test_generate_concurrent_second_returns_409(client, monkeypatch):
    # While a job is running, a second generate is refused synchronously with the busy
    # sentinel; the first job still completes and persists once released.
    backend = video_module.get_video_backend()
    backend.loaded = True
    release = threading.Event()
    real_generate = _FakeBackend.generate

    def _slow(**kwargs):
        assert release.wait(5)
        return real_generate(backend, **kwargs)

    monkeypatch.setattr(backend, "generate", _slow)
    first = client.post("/api/inference/video/generate", json = {"prompt": "a", "seed": 1})
    assert first.status_code == 200 and first.json()["status"] == "started"

    second = client.post("/api/inference/video/generate", json = {"prompt": "b"})
    assert second.status_code == 409
    assert second.json()["detail"] == VIDEO_GENERATION_BUSY_MSG

    running = client.get("/api/inference/video/generate-progress").json()
    assert running["active"] is True

    release.set()
    progress = _wait_terminal(client)
    assert progress["phase"] == "completed" and progress["video"]["seed"] == 1
    # With the job finished, a new generate is accepted again.
    assert _generate_and_wait(client, {"prompt": "c", "seed": 2})["seed"] == 2


def test_generate_progress_route(client):
    resp = client.get("/api/inference/video/generate-progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["phase"] is None and body["video"] is None and body["error"] is None


def test_cancel_generation_route(client):
    resp = client.post("/api/inference/video/generate/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is False


def test_cancel_running_job(client, monkeypatch):
    # Cancel still works against the background job: begin_generate registers the
    # cancel event before the worker starts, so the cancel route reports True at
    # once and the job lands in the failed(cancelled) terminal state.
    backend = video_module.get_video_backend()
    backend.loaded = True

    def _wait_for_cancel(*, cancel_event = None, **kwargs):
        assert cancel_event is not None and cancel_event.wait(5)
        raise RuntimeError(VIDEO_CANCELLED_MSG)

    monkeypatch.setattr(backend, "generate", _wait_for_cancel)
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 200

    cancelled = client.post("/api/inference/video/generate/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["cancelled"] is True

    progress = _wait_terminal(client)
    assert progress["phase"] == "failed"
    assert progress["error"] == VIDEO_CANCELLED_MSG
    # Nothing was persisted for the cancelled run.
    assert client.get("/api/inference/video/gallery").json()["videos"] == []


def test_file_endpoint_404_for_bad_id(client):
    resp = client.get("/api/inference/video/gallery/does-not-exist/file")
    assert resp.status_code == 404


def test_delete_and_clear(client):
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    first = _generate_and_wait(client, {"prompt": "a"})
    second = _generate_and_wait(client, {"prompt": "b"})
    assert len(client.get("/api/inference/video/gallery").json()["videos"]) == 2

    # Delete one, then confirm it 404s and the other remains.
    assert client.delete(f"/api/inference/video/gallery/{first['id']}").status_code == 200
    assert client.delete(f"/api/inference/video/gallery/{first['id']}").status_code == 404
    remaining = client.get("/api/inference/video/gallery").json()["videos"]
    assert [v["id"] for v in remaining] == [second["id"]]

    # Clear wipes the rest.
    cleared = client.delete("/api/inference/video/gallery")
    assert cleared.status_code == 200 and cleared.json()["removed"] == 1
    assert client.get("/api/inference/video/gallery").json()["videos"] == []


def test_gallery_pagination(client):
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    for i in range(5):
        _generate_and_wait(client, {"prompt": f"clip {i}", "seed": i})
    page1 = client.get("/api/inference/video/gallery?limit=2&offset=0").json()
    assert len(page1["videos"]) == 2 and page1["has_more"] is True
    last = client.get("/api/inference/video/gallery?limit=2&offset=4").json()
    assert len(last["videos"]) == 1 and last["has_more"] is False


def test_status_passthrough(client, monkeypatch):
    backend = video_module.get_video_backend()
    resolved = {
        "speed_mode": {"value": "eager", "source": "auto", "reason": "GGUF default"},
        "transformer_cache": {"value": None, "source": "auto", "reason": "few-step model"},
    }
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {
            **_unloaded_status(),
            "loaded": True,
            "family": "ltx-2",
            "has_audio": True,
            "defaults": _defaults(),
            "resolved": resolved,
        },
    )
    body = client.get("/api/inference/video/status").json()
    assert body["loaded"] is True and body["family"] == "ltx-2"
    assert body["resolved"] == resolved
    assert body["defaults"]["frame_step"] == 8


def test_status_resolved_defaults_to_null(client):
    body = client.get("/api/inference/video/status").json()
    assert body["resolved"] is None and body["defaults"] is None


def test_unload_releases_arbiter(client, monkeypatch):
    # Pin VIDEO as the current owner; unload must drop that claim.
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)
    resp = client.post("/api/inference/video/unload")
    assert resp.status_code == 200 and resp.json()["loaded"] is False
    assert gpu_arbiter.current_owner() is None


def test_unload_keeps_ownership_when_a_load_is_in_flight(client, monkeypatch):
    # A concurrent /video/load re-acquires VIDEO and starts a background load, so the backend
    # is NOT loaded yet (the pipeline commits later) but a load IS in flight. The unload route
    # must keep ownership on the in-flight state alone, or a later chat/image load would see no
    # owner, skip eviction, and OOM against the newly resident pipeline. The committed-loaded
    # state stays False the whole load window, so the loaded-only check is insufficient here.
    backend = video_module.get_video_backend()
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

    backend.loaded = False
    backend.loading = ("unsloth/ltx-video-2b",)
    resp = client.post("/api/inference/video/unload")
    assert resp.status_code == 200 and resp.json()["loaded"] is False
    assert gpu_arbiter.current_owner() == gpu_arbiter.VIDEO  # ownership retained for the load

    backend.loading = ()


def test_load_refused_during_training(client, monkeypatch):
    # A video load while training is active is refused (409) before the GPU is taken.
    import core.training as core_training

    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted: list = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append(True))

    class _Training:
        def is_training_active(self):
            return True

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Training())

    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()
    assert evicted == []  # chat backend was never evicted
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_generate_missing_prompt_returns_422(client):
    resp = client.post("/api/inference/video/generate", json = {})
    assert resp.status_code == 422


def test_routes_require_auth():
    # No dependency override: the auth dependency must reject the request.
    app = FastAPI()
    app.include_router(video_router, prefix = "/api/inference")
    unauth = TestClient(app)
    assert unauth.get("/api/inference/video/status").status_code in (401, 403)


def test_export_endpoint_validation(client, monkeypatch):
    # Unknown format is a 400 before any work happens.
    resp = client.get("/api/inference/video/gallery/x/export?format=avi")
    assert resp.status_code == 400
    # Unknown id is a 404.
    resp = client.get("/api/inference/video/gallery/does-not-exist/export?format=gif")
    assert resp.status_code == 404

    # A missing codec surfaces as 501 with the transcoder's message.
    def _boom(video_id, fmt):
        raise RuntimeError("WebM export needs the 'av' package (PyAV).")

    monkeypatch.setattr(gallery_module, "transcode", _boom)
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    video = _generate_and_wait(client, {"prompt": "a"})
    resp = client.get(f"/api/inference/video/gallery/{video['id']}/export?format=webm")
    assert resp.status_code == 501
    assert "PyAV" in resp.json()["detail"]
