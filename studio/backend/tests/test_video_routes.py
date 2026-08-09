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
        # Repo ids of in-flight loads; empty = none. The unload route reads this to keep VIDEO ownership during a concurrent load.
        self.loading: tuple = ()

    # The real backend keys "loaded" off its committed pipeline state, so map the fake flag onto it for the inherited begin_generate.
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
    ):
        # Mirror the real backend cheap validation so the route validate-before-evict ordering is exercised.
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
    # Isolate from the real GPU arbiter: reset ownership and stub the evictors so acquire_for() never touches live singletons.
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.VIDEO, lambda: None)

    # Pin the device to cpu so the load route deterministically takes the non-GPU branch; the arbiter gating has its own tests.
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )

    # Persist to a real tmp gallery so save/list/file/delete/clear run the actual video_gallery code without touching studio_root.
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
    # The load-time transformer_quant field reaches the backend, and the per-generation guidance_2 reaches generate() (dual-DiT MoE).
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
    # The load-time text_encoder_quant field reaches the backend (the video path also quantises the dense companion encoder).
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
    # A local video-family dir with one .safetensors and no model_index.json arrives as a pipeline with no filename; the route reinterprets it as a single_file load.
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
    # The POST returns at once ("started"); the saved record arrives through the generate-progress terminal state.
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


def test_generate_without_load_returns_409(client):
    resp = client.post("/api/inference/video/generate", json = {"prompt": "p"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == VIDEO_NOT_LOADED_MSG


def test_generate_cancelled_reports_failed_with_sentinel(client, monkeypatch):
    # A cancel mid-run surfaces as the job's terminal failed state carrying the exact sentinel, not as an HTTP error.
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
    # A loaded model failing mid-pipeline (CUDA OOM) is a server failure: the terminal state carries a generic message, never the raw exception.
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
    # While a job runs, a second generate is refused synchronously with the busy sentinel; the first still completes and persists.
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
    # begin_generate registers the cancel event before the worker starts, so the cancel route reports True at once and the job lands in failed(cancelled).
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


def test_serve_and_export_refuse_orphan_mp4(client, tmp_path):
    # An orphan MP4 is hidden by the listing, and serve / export resolve through the ownership guard, so a guessed stem can neither stream nor transcode it out.
    (tmp_path / "recording.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert client.get("/api/inference/video/gallery/recording/file").status_code == 404
    assert client.get("/api/inference/video/gallery/recording/export?format=gif").status_code == 404


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


def test_deleting_a_clip_clears_the_terminal_generation_record(client):
    """The completed record outlives its job so a page mounting late still sees the clip. Once the
    clip is deleted that record points at a file that is gone, and the Video page merged it back on
    every reload as a card whose fetch 404s."""
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    clip = _generate_and_wait(client, {"prompt": "a"})
    progress = client.get("/api/inference/video/generate-progress").json()
    assert progress["phase"] == "completed" and progress["video"]["id"] == clip["id"]

    assert client.delete(f"/api/inference/video/gallery/{clip['id']}").status_code == 200

    after = client.get("/api/inference/video/generate-progress").json()
    assert after.get("video") is None
    assert after.get("phase") != "completed"
    assert after["active"] is False


def test_clearing_the_gallery_clears_the_terminal_generation_record(client):
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    _generate_and_wait(client, {"prompt": "a"})
    assert client.delete("/api/inference/video/gallery").status_code == 200

    after = client.get("/api/inference/video/generate-progress").json()
    assert after.get("video") is None
    assert after["active"] is False


def test_deleting_a_different_clip_leaves_the_terminal_record_alone(client):
    """Only the terminal clip's own deletion clears it: deleting an older clip must not drop the
    record the page needs to show the run that just finished."""
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    older = _generate_and_wait(client, {"prompt": "a"})
    newest = _generate_and_wait(client, {"prompt": "b"})

    assert client.delete(f"/api/inference/video/gallery/{older['id']}").status_code == 200

    after = client.get("/api/inference/video/generate-progress").json()
    assert after["phase"] == "completed" and after["video"]["id"] == newest["id"]


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
    # A concurrent /video/load re-acquires VIDEO and starts a background load, so the backend is not loaded yet but a load IS in
    # flight: ownership must be kept on that alone, or a later chat/image load skips eviction and OOMs against the new pipeline.
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

    monkeypatch.setattr(gallery_module, "transcode_to_file", _boom)
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    video = _generate_and_wait(client, {"prompt": "a"})
    resp = client.get(f"/api/inference/video/gallery/{video['id']}/export?format=webm")
    assert resp.status_code == 501
    assert "PyAV" in resp.json()["detail"]


def test_delete_guard_protects_the_loaded_video_companion_base(monkeypatch):
    # For a GGUF / single-file video load the companion base supplies the VAE and text encoders, so it is as much the live model as the checkpoint; the guard only compared repo_id.
    from hub.services.models import deletion

    class _Backend:
        def status(self):
            return {
                "loaded": True,
                "repo_id": "unsloth/LTX-2.3-GGUF",
                "base_repo": "unsloth/LTX-2.3",
            }

        def loading_repo_ids(self):
            return ()

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())
    assert deletion._video_blocks_delete("unsloth/LTX-2.3-GGUF") is not None
    assert deletion._video_blocks_delete("unsloth/LTX-2.3") is not None
    assert deletion._video_blocks_delete("unsloth/something-else") is None


def test_delete_guard_protects_the_native_video_companion_repos(monkeypatch):
    # The native H3 runtime re-reads its Qwen encoder and both VAEs from companion repos that are
    # neither repo_id nor the BF16 base_repo the status publishes, so the guard needs loaded_repo_ids().
    from hub.services.models import deletion

    class _Backend:
        def status(self):
            return {
                "loaded": True,
                "repo_id": "unsloth/MiniMax-H3-GGUF",
                "base_repo": "MiniMaxAI/MiniMax-H3",
            }

        def loaded_repo_ids(self):
            return ("unsloth/MiniMax-H3-GGUF", "Comfy-Org/MiniMax-H3")

        def loading_repo_ids(self):
            return ()

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())
    assert deletion._video_blocks_delete("Comfy-Org/MiniMax-H3") is not None
    assert deletion._video_blocks_delete("unsloth/something-else") is None


def test_delete_guard_protects_the_native_video_companion_repos_mid_load(monkeypatch):
    # The in-flight twin of the check above: during an H3 load status()["loaded"] is still False,
    # yet the download is pulling from the same companion repos, so deleting one would yank blobs
    # out from under it. loading_repo_ids() reported only repo_id and base_repo.
    from hub.services.models import deletion

    class _Backend:
        def status(self):
            return {"loaded": False, "repo_id": None, "base_repo": None}

        def loaded_repo_ids(self):
            return ()

        def loading_repo_ids(self):
            return (
                "leejet/MiniMax-H3-GGUF",
                "MiniMaxAI/MiniMax-H3",
                "unsloth/MiniMax-H3-GGUF",
                "Comfy-Org/MiniMax-H3",
            )

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())
    assert deletion._video_blocks_delete("Comfy-Org/MiniMax-H3") is not None
    # Loading from the leejet mirror still pulls the Qwen encoder from the unsloth GGUF companion.
    assert deletion._video_blocks_delete("unsloth/MiniMax-H3-GGUF") is not None
    assert deletion._video_blocks_delete("unsloth/something-else") is None


def test_video_download_plan_forwards_the_encoder_policy(client, monkeypatch):
    # The plan drives the staged download, so it must use the encoder policy the load will run with: an fp8 request takes a hosted pre-cast encoder, and staging the dense one pulls ~49 GB of Gemma3.
    backend = video_module.get_video_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen["model_path"] = model_path
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "hf_token": "hf_secret",
            "text_encoder_quant": "fp8",
        },
    )

    assert resp.status_code == 200
    assert seen["text_encoder_quant"] == "fp8"
    assert seen["hf_token"] == "hf_secret"


def test_video_load_guard_still_checks_diffusion_when_the_llm_probe_raises(client, monkeypatch):
    # Same independence rule as the image guard: a raising LLM probe used to return early, so a video load ran into an active diffusion trainer on the same GPU.
    import core.training as core_training
    import routes.video as video_routes

    class _Broken:
        def is_training_active(self):
            raise RuntimeError("training backend unavailable")

    class _Diffusion:
        def is_active(self):
            return True

    monkeypatch.setattr(core_training, "get_training_backend", lambda: _Broken())
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service",
        lambda: _Diffusion(),
        raising = False,
    )

    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 409
    assert "training" in resp.json()["detail"].lower()
    assert video_routes is not None


def test_signed_video_link_streams_without_a_bearer(client):
    # A clip is tens to hundreds of MB, so the gallery cannot fetch it into a blob like a PNG: that buffers the whole MP4, kills seeking and pins the bytes. The signed link makes the range-capable /file route usable as a plain <video src>.
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    video = _generate_and_wait(client, {"prompt": "a"})
    vid = video["id"]

    minted = client.get(f"/api/inference/video/gallery/{vid}/signed-url")
    assert minted.status_code == 200, minted.text
    url = minted.json()["url"]
    assert url.startswith(f"/api/inference/video/gallery/{vid}/file-signed?token=")

    # Served with no Authorization header at all, and byte-identical to the bearer route.
    signed = client.get(url, headers = {})
    assert signed.status_code == 200
    assert signed.headers["content-type"] == "video/mp4"
    assert signed.content == client.get(f"/api/inference/video/gallery/{vid}/file").content

    # Range requests work, which is the point: the player seeks instead of downloading everything.
    ranged = client.get(url, headers = {"Range": "bytes=0-3"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 4


def test_signed_video_link_rejects_tampering_and_other_ids(client):
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    first = _generate_and_wait(client, {"prompt": "a"})["id"]
    second = _generate_and_wait(client, {"prompt": "b"})["id"]
    token = (
        client.get(f"/api/inference/video/gallery/{first}/signed-url")
        .json()["url"]
        .split("token=", 1)[1]
    )

    # The token names exactly one clip.
    assert (
        client.get(f"/api/inference/video/gallery/{second}/file-signed?token={token}").status_code
        == 401
    )
    # A flipped signature, a malformed token, and an expired one are all refused.
    assert (
        client.get(
            f"/api/inference/video/gallery/{first}/file-signed?token={token[:-1]}x"
        ).status_code
        == 401
    )
    assert (
        client.get(f"/api/inference/video/gallery/{first}/file-signed?token=nonsense").status_code
        == 401
    )
    from routes import video as video_routes

    expired = video_routes._sign_video_id(first)
    payload, _sig = expired.rsplit(".", 1)
    stale_id, _exp = payload.rsplit(".", 1)
    import hashlib
    import hmac

    stale_payload = f"{stale_id}.1"
    stale_sig = hmac.new(
        video_routes._VIDEO_LINK_SECRET, stale_payload.encode(), hashlib.sha256
    ).hexdigest()
    assert (
        client.get(
            f"/api/inference/video/gallery/{first}/file-signed?token={stale_payload}.{stale_sig}"
        ).status_code
        == 401
    )


def test_signed_url_mint_is_bearer_gated_and_404s_for_an_unknown_clip(client):
    assert client.get("/api/inference/video/gallery/does-not-exist/signed-url").status_code == 404


def test_video_download_plan_forwards_the_denoiser_policy(client, monkeypatch):
    # Same rule as the encoder policy above, for the denoiser: with a hosted pre-quantized
    # checkpoint the plan must drop the dense transformer shards, and without this forwarding it
    # stages ~66 GB the load then never opens.
    backend = video_module.get_video_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen["model_path"] = model_path
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "transformer_quant": "int8",
        },
    )

    assert resp.status_code == 200
    assert seen["transformer_quant"] == "int8"


def test_video_download_plan_refuses_an_unsupported_combination_before_staging(client, monkeypatch):
    # The whole point of moving the refusal into validation: this pick used to return a 200 plan,
    # stage ~98.7 GB, and only then fail inside the loader. Runs the REAL validation rather than
    # the fake backend's mirror of it, since the rule under test lives in the real one.
    backend = video_module.get_video_backend()
    monkeypatch.setattr(
        backend,
        "validate_load_request",
        video_module.VideoBackend.validate_load_request.__get__(backend),
        raising = False,
    )

    def _plan(model_path, **kwargs):  # pragma: no cover - reaching this IS the regression
        raise AssertionError("download_plan must not be reached for a refused pick")

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "MiniMaxAI/MiniMax-H3",
            "gguf_filename": "minimax_h3_fl2va_pruned_int8_rowwise.safetensors",
            "model_kind": "single_file",
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # A refusal that does not name the alternative just moves the dead end earlier.
    assert "MiniMaxAI/MiniMax-H3" in detail
    assert "unsloth/MiniMax-H3-GGUF" in detail


def test_video_download_plan_refuses_an_unavailable_transformer_quant(client, monkeypatch):
    # And the quant-keyed refusal has to fire on THIS route too, which needs the route to forward
    # transformer_quant into validation: it is the route that stages the download.
    backend = video_module.get_video_backend()
    monkeypatch.setattr(
        backend,
        "validate_load_request",
        video_module.VideoBackend.validate_load_request.__get__(backend),
        raising = False,
    )
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be reached")),
        raising = False,
    )

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "MiniMaxAI/MiniMax-H3",
            "model_kind": "pipeline",
            "transformer_quant": "nvfp4",
        },
    )

    assert resp.status_code == 400
    assert "nvfp4" in resp.json()["detail"]
