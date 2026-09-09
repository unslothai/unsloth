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
from fastapi import FastAPI, HTTPException
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
import routes.video as video_routes
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
        # Minimal committed state required by inherited generation validation.
        import types

        from core.inference.video_families import detect_video_family
        self._state = (
            types.SimpleNamespace(
                family = detect_video_family("Lightricks/LTX-2"),
                h3_task = None,
                engine = "diffusers",
            )
            if value
            else None
        )

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
        h3_task = None,
    ):
        # Mirror the real backend cheap validation so the route validate-before-evict ordering is exercised.
        from pathlib import Path

        kind = (model_kind or ("gguf" if gguf_filename else "pipeline")).lower()
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        # Non-GGUF loads are gated to unsloth/* repos, the official bases, and existing local paths.
        # minimaxai/: the real gate trusts MiniMaxAI/MiniMax-H3 as an official family base repo.
        trusted = model_path.lower().startswith(("unsloth/", "lightricks/", "minimaxai/")) or (
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
            # Include every field persisted to the gallery sidecar.
            "conditioning": "t2va",
            "flow_shift": None,
            "audio_flow_shift": None,
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


@pytest.fixture(autouse = True)
def _healthy_diffusers(healthy_diffusers):
    """These tests are about the route, not about the runner's diffusers.

    The module docstring promises they run without diffusers, and most do, but the
    MiniMax-H3 download plan reaches `import diffusers` in video.py's modular-workflow
    branch. Backend CI installs no diffusers (it lives in requirements/diffusers-pin.txt,
    which only install_python_stack.py applies), so without the proxy that one test dies
    on ModuleNotFoundError. Same fixture the diffusion test modules already use.
    """


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


def test_load_forwards_the_gpu_selection(client, monkeypatch):
    # /video/load carried no gpu_ids at all, so sd.cpp and diffusers both pinned ordinal 0.
    import types

    import core.inference.diffusion_device as devmod
    import core.inference.video as video_module

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", lambda ids: max(ids))
    monkeypatch.setattr(gpu_arbiter, "acquire_for", lambda role, register = None: register())
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf",
            "gpu_ids": [1],
        },
    )
    assert resp.status_code == 200
    assert video_module.get_video_backend().last_load_kwargs["gpu_ids"] == [1]


def test_load_refuses_a_gpu_index_this_host_does_not_have(client, monkeypatch):
    # Refused BEFORE the arbiter evicts chat, so a bad pick costs a resident model nothing.
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )

    def _refuse(_ids):
        raise ValueError("Requested GPU [7] but this host has 2 CUDA device(s).")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _refuse)
    resp = client.post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf",
            "gpu_ids": [7],
        },
    )
    assert resp.status_code == 400
    assert "2 CUDA device" in resp.json()["detail"]
    assert gpu_arbiter._owner is None


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


def test_load_threads_transformer_quant_and_guidance_2(client, monkeypatch):
    # The load-time transformer_quant field reaches the backend, and the per-generation guidance_2 reaches generate() (dual-DiT MoE).
    # The dense DiT quant is pipeline-only, so a GGUF pick with an explicit scheme is refused by the
    # route's precision gate before it gets here; stub the gate out, since this is about threading.
    monkeypatch.setattr(video_module, "assert_video_precision_available", lambda fam, **kw: None)
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


def test_load_threads_text_encoder_quant(client, monkeypatch):
    # The load-time text_encoder_quant field reaches the backend (the video path also quantises the dense companion encoder).
    #
    # Under the precision escape hatch, because whether an fp8 encoder is available is a
    # property of the HOST: on a CPU-only runner the gate refuses this pick with a 409 and the
    # forwarding this test is about never happens. The hatch is the product's own bypass.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
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


def test_load_progress_route(client, monkeypatch):
    resets = []
    monkeypatch.setattr(video_routes, "reset_media_load_progress", resets.append)
    idle = client.get("/api/inference/video/load-progress")
    assert idle.status_code == 200 and idle.json()["phase"] is None
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    ready = client.get("/api/inference/video/load-progress")
    assert ready.json()["phase"] == "ready"
    assert resets == ["video"]


def test_load_progress_route_logs_backend_snapshot(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        video_routes,
        "log_media_load_progress",
        lambda media, phase, fraction: seen.append((media, phase, fraction)),
    )
    backend = video_module.get_video_backend()
    backend.load_progress = lambda: {
        "phase": "downloading",
        "downloaded_bytes": 30,
        "expected_bytes": 100,
        "error": None,
    }

    response = client.get("/api/inference/video/load-progress")

    assert response.status_code == 200
    assert seen == [("video", "downloading", 0.3)]


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


def test_generate_happy_path_persists_and_reports_record(client, monkeypatch):
    resets = []
    monkeypatch.setattr(video_routes, "reset_media_generation_progress", resets.append)
    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    # The POST returns at once ("started"); the saved record arrives through the generate-progress terminal state.
    video = _generate_and_wait(client, {"prompt": "a sloth surfing", "seed": 7})
    assert resets == ["video"]
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


def test_generate_accepts_a_half_specified_size_without_a_keyframe(client):
    """Half a canvas is only ambiguous next to a keyframe, so the route must still take it.

    validate_video_request_shape has always resolved a missing axis against the family's default
    preset (768 alone means 768x512 on LTX-2) and that behaviour is pinned at the family level, so
    a request-model XOR that fires with no keyframe present makes the two layers disagree and
    breaks the half-spec case for every video family through the API.
    """
    backend = video_module.get_video_backend()
    backend.loaded = True

    video = _generate_and_wait(client, {"prompt": "a cat", "width": 768})
    assert (video["width"], video["height"]) == (768, 512)

    # With a keyframe the ambiguity is real and the refusal stands.
    resp = client.post(
        "/api/inference/video/generate",
        json = {
            "prompt": "a cat",
            "width": 768,
            "first_frame": "data:image/png;base64,AAAA",
        },
    )
    assert resp.status_code == 422
    assert "width and height must be sent together" in str(resp.json())


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


def test_generate_progress_route(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        video_routes,
        "log_media_generation_progress",
        lambda media, progress: seen.append((media, dict(progress))),
    )
    resp = client.get("/api/inference/video/generate-progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["phase"] is None and body["video"] is None and body["error"] is None
    assert seen == [("video", {"active": False, "total_steps": 0, "fraction": 0.0})]


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
        # A declined explicit precision has to survive the boundary here too (the Linux half of
        # P1-2: the label kept reading BF16 while telemetry confirmed NVFP4).
        "transformer_quant": {
            "value": "off",
            "requested": "nvfp4",
            "source": "explicit",
            "status": "unsupported",
            "reason": "this GPU has no NVFP4 cores",
        },
        "text_encoder_quant": {
            "value": "fp8",
            "requested": "int8",
            "source": "explicit",
            "status": "fell_back",
            "reason": "int8 has no measured keep-bf16 schedule for this family",
        },
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
    assert body["resolved"]["transformer_quant"] == resolved["transformer_quant"]
    assert body["resolved"]["text_encoder_quant"] == resolved["text_encoder_quant"]
    # Entries from an older backend (no requested/status) still parse, defaulted to "applied".
    assert body["resolved"]["speed_mode"]["requested"] is None
    assert body["resolved"]["speed_mode"]["status"] == "applied"
    assert body["defaults"]["frame_step"] == 8


def test_status_resolved_defaults_to_null(client):
    body = client.get("/api/inference/video/status").json()
    assert body["resolved"] is None and body["defaults"] is None


def _load_fake_video(client):
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    assert resp.status_code == 200


def test_saved_clip_records_the_engaged_build(client, monkeypatch):
    # A saved MP4 has to name the BUILD it came off, like a saved PNG already does: without it a
    # clip rendered at bf16 is indistinguishable from one rendered at the precision that was asked
    # for. Every value here is the ENGAGED state, never the load request.
    backend = video_module.get_video_backend()
    real_generate = backend.generate

    def _generate(**kwargs):
        return {
            **real_generate(**kwargs),
            "model_kind": "gguf",
            "gguf_filename": "ltx-2.3-Q4_K_M.gguf",
            "transformer_quant": None,
            "text_encoder_quant": "fp8",
            "memory_mode": "low_vram",
            "offload_policy": "sequential",
        }

    monkeypatch.setattr(backend, "generate", _generate)
    _load_fake_video(client)
    video = _generate_and_wait(client, {"prompt": "a sloth", "seed": 3})
    assert video["model_kind"] == "gguf"
    assert video["gguf_filename"] == "ltx-2.3-Q4_K_M.gguf"
    assert video["transformer_quant"] is None
    assert video["text_encoder_quant"] == "fp8"
    assert video["memory_mode"] == "low_vram"
    assert video["offload_policy"] == "sequential"
    # ...and it round-trips through the on-disk sidecar, not just the in-memory record.
    listed = client.get("/api/inference/video/gallery").json()["videos"]
    assert listed[0]["text_encoder_quant"] == "fp8"
    assert listed[0]["gguf_filename"] == "ltx-2.3-Q4_K_M.gguf"


def test_sidecars_without_the_build_fields_still_list(client):
    # The new keys are NOT in video_gallery._REQUIRED_META, so a clip written before they existed
    # lists with them null instead of being skipped as a foreign sidecar.
    _load_fake_video(client)
    video = _generate_and_wait(client, {"prompt": "a sloth"})
    assert video["model_kind"] is None and video["transformer_quant"] is None
    listed = client.get("/api/inference/video/gallery").json()["videos"]
    assert len(listed) == 1 and listed[0]["text_encoder_quant"] is None


def test_load_refuses_an_unusable_explicit_precision_with_409(client, monkeypatch):
    # begin_load refuses an EXPLICIT precision this host cannot honor; the route answers 409 with
    # the reason instead of accepting the load and denoising at some other precision.
    from core.inference.diffusion_auto_policy import precision_refusal_message

    backend = video_module.get_video_backend()
    refusal = precision_refusal_message(
        "transformer_quant",
        "nvfp4",
        "'nvfp4' is not usable for family 'ltx-2' on this GPU",
        off_label = "Off to run the DiT at bf16",
    )

    def _refuse(model_path, **kwargs):
        raise RuntimeError(refusal)

    monkeypatch.setattr(backend, "begin_load", _refuse)
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "Lightricks/LTX-2.3", "transformer_quant": "nvfp4"},
    )
    assert resp.status_code == 409
    assert "transformer_quant='nvfp4' could not be used" in resp.json()["detail"]
    assert client.get("/api/inference/video/status").json()["loaded"] is False


def test_precision_refusal_precedes_eviction(client, monkeypatch):
    # The refusal belongs to validate_load_request, which the route calls BEFORE the GPU handoff:
    # acquire_for evicts chat under the arbiter lock before it runs begin_load, so a refusal made
    # there arrives after the eviction the 409 exists to prevent.
    from core.inference.diffusion_auto_policy import precision_refusal_message

    backend = video_module.get_video_backend()
    evicted = []
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: evicted.append("chat"))
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    refusal = precision_refusal_message(
        "transformer_quant",
        "nvfp4",
        "'nvfp4' is not usable for family 'ltx-2' on this GPU",
        off_label = "Off to run the DiT at bf16",
    )

    def _refuse(fam, **kwargs):
        raise RuntimeError(refusal)

    monkeypatch.setattr(video_module, "assert_video_precision_available", _refuse)
    monkeypatch.setattr(
        backend,
        "begin_load",
        lambda *a, **k: pytest.fail("begin_load ran after an impossible precision"),
    )
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "Lightricks/LTX-2.3", "transformer_quant": "nvfp4"},
    )
    assert resp.status_code == 409
    assert "transformer_quant='nvfp4' could not be used" in resp.json()["detail"]
    assert evicted == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


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
    # Same host-independence as the load twin: the plan now makes the precision check too, so
    # without the escape hatch a GPU-less runner answers 409 and forwards nothing.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
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


def test_the_video_plan_refuses_an_impossible_precision_before_anything_is_staged(
    client, monkeypatch
):
    # The video weights are the tens-of-GB case: /video/load refuses a precision this host cannot
    # honour, but the UI plans and stages first, so the refusal used to arrive after the download.
    import core.inference.video as video_core

    backend = video_module.get_video_backend()

    def _refuse(fam, **kwargs):
        raise RuntimeError(
            "text_encoder_quant='fp8' could not be used: this device cannot cast the encoder."
        )

    monkeypatch.setattr(video_core, "assert_video_precision_available", _refuse)
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: pytest.fail("the plan was built for a precision that cannot be honoured"),
        raising = False,
    )

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "text_encoder_quant": "fp8",
        },
    )

    assert resp.status_code == 409, resp.text
    assert "could not be used" in resp.json()["detail"]


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


def test_keyless_caller_cannot_mint_a_signed_video_link(client):
    from routes import video as video_routes

    client.post(
        "/api/inference/video/load",
        json = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "q.gguf"},
    )
    video_id = _generate_and_wait(client, {"prompt": "a"})["id"]
    client.app.dependency_overrides[video_routes.request_admitted_without_credential] = lambda: True

    response = client.get(f"/api/inference/video/gallery/{video_id}/signed-url")

    assert response.status_code == 403
    assert "API key" in response.json()["detail"]


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
    # What is under test is the ROUTE forwarding the denoiser policy into download_plan, not the
    # precision gate that now runs ahead of it. That gate has its own tests and it refuses an
    # explicit scheme on a gguf pick before download_plan is ever reached, so stub it out here and
    # keep this test on the forwarding.
    monkeypatch.setattr(
        video_module, "assert_video_precision_available", lambda fam, **kw: None, raising = False
    )

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


def test_video_download_plan_forwards_the_h3_partition(client, monkeypatch):
    # h3_task decides WHICH of the two 66.28 GB MiniMax-H3 denoiser folders is staged. It was
    # swallowed by **load_kwargs, so a ref2va plan staged the fl2va partition and the one the load
    # actually opens came down inline, outside the download panel's preflight.
    backend = video_module.get_video_backend()
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen["model_path"] = model_path
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)
    monkeypatch.setattr(
        video_module, "assert_video_precision_available", lambda fam, **kw: None, raising = False
    )

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/MiniMax-H3",
            "model_kind": "pipeline",
            "family_override": "minimax-h3",
            "h3_task": "ref2va",
        },
    )

    assert resp.status_code == 200, resp.json()
    assert seen["h3_task"] == "ref2va"


def test_video_download_plan_judges_a_quantized_reference_pick_per_partition(client, monkeypatch):
    # The plan route asks the same (scheme, PARTITION) question the load does, so a pick it
    # answers 200 is one the load will honour. Both halves matter and they are one test because
    # either alone passes for the wrong reason: refusing everything, or accepting everything.
    #
    # ref2va now has its own hosted int8 and fp8 denoisers, so the reference partition is a real
    # pick rather than a keyframe checkpoint wearing the wrong name. A scheme with no checkpoint
    # at all is still refused BEFORE staging, which is the failure this route check was added for
    # -- a 200 plan carrying 20 GB for a request the load then answered with a 400.
    #
    # The host-level precision gate is a DIFFERENT question from the one under test, and on a
    # box with no CUDA and no torchao it answers 409 before the partition check is ever reached.
    # Stubbing it keeps the availability refusal (a 400, raised by validate_load_request below)
    # under test everywhere, including the Backend CI matrix that installs no torchao. Same stub
    # the neighbouring route tests use; test_video_h3_te_quant.py covers the gate itself.
    monkeypatch.setattr(
        video_module, "assert_video_precision_available", lambda fam, **kw: None, raising = False
    )
    backend = video_module.get_video_backend()
    monkeypatch.setattr(
        backend,
        "validate_load_request",
        video_module.VideoBackend.validate_load_request.__get__(backend),
        raising = False,
    )
    seen: dict = {}

    def _plan(model_path, **kwargs):
        seen.update(kwargs)
        return {"entries": [], "total_bytes": 0}

    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)

    def _ask(scheme):
        return client.post(
            "/api/inference/video/download-plan",
            json = {
                "model_path": "MiniMaxAI/MiniMax-H3",
                "family_override": "minimax-h3",
                "model_kind": "pipeline",
                "transformer_quant": scheme,
                "h3_task": "ref2va",
            },
        )

    served = _ask("int8")
    assert served.status_code == 200, served.json()
    # Planned for the partition that was asked for, not the keyframe one it used to fall back to.
    assert seen["h3_task"] == "ref2va"
    assert seen["transformer_quant"] == "int8"

    seen.clear()
    refused = _ask("nvfp4")
    assert refused.status_code == 400
    detail = refused.json()["detail"]
    # Named per task: "unavailable" here is a claim about ref2va, not about the whole family.
    assert "ref2va" in detail
    # A refusal that does not name the alternative just moves the dead end earlier.
    assert "int8" in detail and "fp8" in detail
    assert not seen, "download_plan must not be reached for a refused pick"


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


def test_video_download_plan_refuses_a_quantized_reference_task(client, monkeypatch):
    # One of the quant-keyed refusals is task-keyed: a pre-quantized H3 denoiser belongs to ONE
    # partition, so a scheme whose only artifact is the keyframe one must not be seeded into the
    # reference workflow. Validation only sees the task when the route forwards h3_task, and this
    # is the route that stages the download -- so without it the plan pulls the 66 GB dense
    # transformer_ref/ AND the wrong-partition quant before /video/load rejects the same request.
    # nvfp4 stands in for that pair here: int8 and fp8 both ship a reference artifact now, so
    # neither is refused any more (test_a_quantized_reference_load_resolves_the_reference_denoiser
    # in test_video_backend.py pins that), and the per-scheme table in test_video_prequant.py
    # covers a family where the pair itself is missing.
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
            "h3_task": "ref2va",
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "nvfp4" in detail
    # Naming the way out, not just the dead end.
    assert "int8" in detail and "fp8" in detail


def test_video_download_plan_hands_the_h3_task_to_validation(client, monkeypatch):
    # The other side of the same gate. The refusal above would still pass if a refactor dropped
    # the forwarding and something else happened to reject that pick, so assert the forwarding
    # itself -- and that fl2va, which is exactly what the hosted checkpoints are, still plans.
    #
    # Validation is stubbed rather than real here because its later transformer_class probe
    # imports the family's diffusers module, which is an environment question and not this
    # test's; the negative case above exercises the real validator, since the ref2va refusal
    # fires before that probe.
    backend = video_module.get_video_backend()
    fam = video_module._detect_load_family("MiniMaxAI/MiniMax-H3", None, None)
    assert fam is not None
    seen: dict = {}

    def _validate(model_path, **kwargs):
        seen["validate"] = kwargs
        return fam

    def _plan(model_path, **kwargs):
        seen["plan"] = kwargs
        return {"files": [], "total_bytes": 0, "cached_bytes": 0}

    monkeypatch.setattr(backend, "validate_load_request", _validate, raising = False)
    monkeypatch.setattr(backend, "download_plan", _plan, raising = False)
    monkeypatch.setattr(
        video_module, "assert_video_precision_available", lambda *a, **k: None, raising = False
    )

    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "MiniMaxAI/MiniMax-H3",
            "model_kind": "pipeline",
            "transformer_quant": "fp8",
            "h3_task": "fl2va",
        },
    )

    assert resp.status_code == 200, resp.text
    assert seen["validate"]["h3_task"] == "fl2va"
    assert seen["validate"]["transformer_quant"] == "fp8"
    assert seen["plan"]["h3_task"] == "fl2va"


def test_the_training_guard_runs_before_the_precision_probe(client, monkeypatch):
    # The precision gate's support check quantises a real Linear on the GPU and synchronises, so
    # running it first initialises a CUDA context and allocates next to the training subprocess
    # for a load that is about to be refused anyway -- and an OOM there under that contention is
    # not a verdict on the scheme. The image route already guards first; this one now does too.
    import routes.video as video_routes

    def _refuse_training() -> None:
        raise HTTPException(status_code = 409, detail = "Training is running.")

    monkeypatch.setattr(video_routes, "_guard_video_load_against_training", _refuse_training)
    monkeypatch.setattr(
        video_module,
        "assert_video_precision_available",
        lambda fam, **kw: pytest.fail("the precision probe ran while training was active"),
    )
    resp = client.post(
        "/api/inference/video/load",
        json = {"model_path": "Lightricks/LTX-2.3", "transformer_quant": "nvfp4"},
    )
    assert resp.status_code == 409 and resp.json()["detail"] == "Training is running."


@pytest.mark.parametrize("memory", ["balanced", "low_vram"])
def test_the_video_precision_gate_sees_the_memory_request(monkeypatch, memory):
    """balanced and low_vram settle the offload policy before anything is measured, and an
    offloaded DiT skips the torchao build -- so load_pipeline's strict refusal arrived after
    acquire_for and the teardown had already evicted the resident model."""
    import types

    monkeypatch.setattr(video_module, "precision_fallback_allowed", lambda: False)
    monkeypatch.setattr(
        video_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cuda", dtype = "bfloat16"),
    )
    monkeypatch.setattr(video_module, "dense_transformer_supported", lambda target: True)
    with pytest.raises(RuntimeError) as excinfo:
        video_module.assert_video_precision_available(
            types.SimpleNamespace(name = "wan2.2"),
            model_kind = "pipeline",
            transformer_quant = "fp8",
            memory_mode = memory,
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)
    assert "offload" in str(excinfo.value)


def test_the_video_gate_leaves_a_measured_memory_mode_alone(monkeypatch):
    """fast and auto are decided from the measured footprint, so this gate cannot judge them."""
    import types

    monkeypatch.setattr(video_module, "precision_fallback_allowed", lambda: False)
    monkeypatch.setattr(
        video_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cuda", dtype = "bfloat16"),
    )
    monkeypatch.setattr(video_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(video_module, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    video_module.assert_video_precision_available(
        types.SimpleNamespace(name = "wan2.2"),
        model_kind = "pipeline",
        transformer_quant = "fp8",
        memory_mode = "fast",
    )


def test_video_download_plan_sizes_its_file_set_for_the_selected_card(client, monkeypatch):
    # The H3 planner sets its denoiser partition and memory policy from device capacity, so a
    # plan sized against the default card stages the wrong weights.
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    ranked: list = []
    monkeypatch.setattr(
        devmod,
        "resolve_selected_cuda_ordinal",
        lambda ids: (ranked.append(list(ids)), 1)[1],
    )
    backend = video_module.get_video_backend()
    seen: dict = {}
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda model_path, **kwargs: (seen.update(kwargs), {"entries": [], "total_bytes": 0})[1],
        raising = False,
    )
    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "gpu_ids": [0, 1],
        },
    )
    assert resp.status_code == 200
    assert seen["gpu_ordinal"] == 1
    # One ranking for the request, shared by the precision preflight and the plan.
    assert ranked == [[0, 1]]


def test_video_download_plan_refuses_a_gpu_index_this_host_does_not_have(client, monkeypatch):
    import types

    import core.inference.diffusion_device as devmod

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )

    def _refuse(_ids):
        raise ValueError("Requested GPU [7] but none of them are visible to this process")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _refuse)
    backend = video_module.get_video_backend()
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: pytest.fail("a refused GPU pick must not reach the planner"),
        raising = False,
    )
    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "gpu_ids": [7],
        },
    )
    assert resp.status_code == 400
    assert "visible to this process" in resp.json()["detail"]


def test_video_download_plan_still_refuses_a_bad_gpu_while_training_holds_the_cards(
    client, monkeypatch
):
    # Same rule as the image twin: the training guard bars the ranking, not the validation,
    # which reads the mask and nvidia-smi and opens no CUDA context.
    import types

    import core.inference.diffusion_device as devmod
    from routes import video as routes_video

    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    monkeypatch.setattr(routes_video, "_training_is_active", lambda: True)
    seen: dict = {}

    def _resolve(ids, *, allow_ranking = True):
        seen["ids"], seen["allow_ranking"] = list(ids), allow_ranking
        raise ValueError("Requested GPU [7] but none of them are visible to this process")

    monkeypatch.setattr(devmod, "resolve_selected_cuda_ordinal", _resolve)
    backend = video_module.get_video_backend()
    monkeypatch.setattr(
        backend,
        "download_plan",
        lambda *a, **k: pytest.fail("a refused GPU pick must not reach the planner"),
        raising = False,
    )
    resp = client.post(
        "/api/inference/video/download-plan",
        json = {
            "model_path": "unsloth/LTX-2.3-GGUF",
            "gguf_filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
            "model_kind": "gguf",
            "gpu_ids": [7],
        },
    )
    assert resp.status_code == 400
    assert seen == {"ids": [7], "allow_ranking": False}
