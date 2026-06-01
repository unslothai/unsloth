# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level tests for ``/api/inference/images/*``.

Mounts the actual ``inference_router`` on a fresh FastAPI app with the
auth dependency replaced by a stub so we exercise the same FastAPI
handlers Studio ships in production. The diffusion backend is replaced
with an in-memory stub so we don't need diffusers / GPUs to run these.

To stay runnable in a minimal CPU-only env, ``routes/inference.py``
is loaded directly via ``importlib`` so we do NOT trigger
``routes/__init__.py`` -- that file eagerly imports training /
datasets / data_recipe / export and would drag in heavy deps
(matplotlib, etc.) that the diffusion tests do not need.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _import_inference_module():
    """Load ``routes/inference.py`` without executing ``routes/__init__``.

    The package init imports training / datasets / data_recipe / export
    routers, which pull in matplotlib / pandas / training stack. The
    diffusion tests only need the inference module so we side-step the
    package import via importlib.spec_from_file_location.
    """
    # If a previous test already imported routes the normal way, reuse
    # the cached module instead of re-loading.
    cached = sys.modules.get("routes.inference")
    if cached is not None:
        return cached
    target = _BACKEND_ROOT / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "routes.inference",
        target,
        # We do NOT set submodule_search_locations for routes itself
        # because that would re-trigger routes/__init__.py. The module
        # uses relative imports sparingly; absolute imports resolve via
        # sys.path[0] = backend root.
    )
    assert spec and spec.loader, "could not build spec for routes/inference.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules["routes.inference"] = module
    # Round 15 P3 #9: drop the half-initialised module from
    # sys.modules if exec_module() raises, otherwise later tests pick
    # up the poisoned entry and report a misleading AttributeError
    # instead of the original ImportError.
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("routes.inference", None)
        raise
    return module


class _FakeBackend:
    def __init__(self) -> None:
        self._loaded = False
        self._repo: str | None = None
        self.calls: list[dict] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> dict:
        return {
            "is_loaded": self._loaded,
            "is_loading": False,
            "repo_id": self._repo,
            "family": "flux.2-klein" if self._loaded else None,
            "pipeline_class": "Flux2KleinPipeline" if self._loaded else None,
            "media_kind": "image" if self._loaded else None,
            "base_repo": "black-forest-labs/FLUX.2-klein" if self._loaded else None,
            "gguf_filename": None,
            "text_encoder_gguf_repo": None,
            "text_encoder_gguf_filename": None,
            "prompt_enhancer_gguf_repo": None,
            "prompt_enhancer_gguf_filename": None,
            "lora": None,
            "gguf_quantized_cpu_resident": False,
            "gguf_pin_cpu_resident": False,
            "offload_policy": None,
            "active_repo_id": self._repo,
            "active_base_repo": (
                "black-forest-labs/FLUX.2-klein" if self._loaded else None
            ),
            # Round 14: guard-facing GGUF filename is now the full
            # caller-supplied value, but this fake never sets one so
            # both active and pending stay None.
            "active_gguf_filename": None,
            "active_text_encoder_gguf_repo": None,
            "active_text_encoder_gguf_filename": None,
            "active_prompt_enhancer_gguf_repo": None,
            "active_prompt_enhancer_gguf_filename": None,
            "active_lora_repo": None,
            "active_lora_weight_name": None,
            "pending_repo_id": None,
            "pending_base_repo": None,
            "pending_gguf_filename": None,
            "pending_text_encoder_gguf_repo": None,
            "pending_text_encoder_gguf_filename": None,
            "pending_prompt_enhancer_gguf_repo": None,
            "pending_prompt_enhancer_gguf_filename": None,
            "pending_lora_repo": None,
            "pending_lora_weight_name": None,
            "device": "cpu",
            "dtype": "torch.bfloat16",
            "loaded_at": 0,
            "last_error": None,
            "supported_families": [],
        }

    def load_model(self, repo_id, **kw):
        self.calls.append({"op": "load", "repo_id": repo_id, **kw})
        self._loaded = True
        self._repo = repo_id
        return self.status()

    def unload_model(self) -> dict:
        self._loaded = False
        self._repo = None
        return {"is_loaded": False}

    def generation_defaults(self) -> dict:
        return {
            "num_inference_steps": 24,
            "guidance_scale": 3.5,
            "width": 1024,
            "height": 1024,
        }

    def generate_image(self, **kw):
        self.calls.append({"op": "generate", **kw})
        return Image.new("RGB", (kw["width"], kw["height"]), color = (123, 45, 67))

    def generate_image_with_metadata(self, **kw):
        image = self.generate_image(**kw)
        meta = {
            "model": self._repo,
            "family": "flux.2-klein" if self._loaded else None,
        }
        return image, meta

    def generate_images_with_metadata(self, **kw):
        image, meta = self.generate_image_with_metadata(**kw)
        meta = {**meta, "output_count": 1}
        return [image], meta

    def generate_video_with_metadata(self, **kw):
        self.calls.append({"op": "generate_video", **kw})
        return ["frame0"], {
            "model": self._repo,
            "family": "ltx2-3-distilled" if self._loaded else None,
            "width": kw["width"],
            "height": kw["height"],
            "num_frames": kw["num_frames"] or 121,
            "frame_rate": kw["frame_rate"] or 24.0,
            "num_inference_steps": kw["num_inference_steps"],
            "guidance_scale": kw["guidance_scale"],
            "guidance_scale_2": kw.get("guidance_scale_2"),
        }


@pytest.fixture
def app_with_stub(monkeypatch):
    """Build a FastAPI app that mounts the real inference router with
    auth disabled and the diffusion backend swapped for a stub."""
    inf = _import_inference_module()
    import core.inference.diffusion as d

    stub = _FakeBackend()
    # Override the singleton accessor the route uses.
    monkeypatch.setattr(d, "get_diffusion_backend", lambda: stub)
    monkeypatch.setattr(inf, "_get_diffusion_backend", lambda: stub)

    app = FastAPI()
    # Diffusion image routes live on studio_router so they are NOT
    # exposed under /v1 (which would let OpenAI-compat clients
    # trigger Studio-only side effects).
    app.include_router(inf.router, prefix = "/api/inference")
    app.include_router(inf.studio_router, prefix = "/api/inference")
    # Bypass auth by overriding the dependency.
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    return app, stub


def test_status_when_unloaded(app_with_stub):
    app, _ = app_with_stub
    c = TestClient(app)
    r = c.get("/api/inference/images/status")
    assert r.status_code == 200
    body = r.json()
    assert body["is_loaded"] is False
    assert body["repo_id"] is None


def test_generate_without_load_returns_400(app_with_stub):
    app, _ = app_with_stub
    c = TestClient(app)
    r = c.post(
        "/api/inference/images/generate",
        json = {"prompt": "a red sphere"},
    )
    assert r.status_code == 400
    assert "No diffusion model" in r.json()["detail"]


def test_load_then_generate_round_trip(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)

    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "unsloth/FLUX.2-klein-4B-GGUF",
            "gguf_filename": "flux-2-klein-4b-Q4_K_S.gguf",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["is_loaded"] is True

    r = c.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "a tiny synth-pop album cover",
            "width": 256,
            "height": 256,
            "num_inference_steps": 4,
            "seed": 7,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_b64"]
    assert body["image_mime"] == "image/png"
    assert body["width"] == 256
    assert body["height"] == 256
    assert body["seed"] == 7
    assert body["duration_ms"] >= 0

    # Round-trip the base64 -> PIL to confirm it is a real PNG of the
    # right size and not, say, an empty string.
    import base64
    import io

    raw = base64.b64decode(body["image_b64"])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.format == "PNG"
    assert decoded.size == (256, 256)

    # Backend stub should have recorded both calls.
    ops = [c["op"] for c in stub.calls]
    assert ops == ["load", "generate"]


def test_video_generate_round_trip(app_with_stub, monkeypatch):
    app, stub = app_with_stub
    c = TestClient(app)
    stub._loaded = True
    stub._repo = "diffusers/LTX-2.3-Distilled-Diffusers"

    import base64
    import core.inference.diffusion as d

    monkeypatch.setattr(
        d,
        "encode_mp4_base64",
        lambda frames, *, fps: base64.b64encode(b"fake-mp4").decode("ascii"),
    )

    r = c.post(
        "/api/inference/videos/generate",
        json = {
            "prompt": "a slow camera push",
            "width": 768,
            "height": 512,
            "num_frames": 121,
            "frame_rate": 24,
            "num_inference_steps": 8,
            "guidance_scale": 1,
            "seed": 3407,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video_b64"] == base64.b64encode(b"fake-mp4").decode("ascii")
    assert body["video_mime"] == "video/mp4"
    assert body["width"] == 768
    assert body["height"] == 512
    assert body["num_frames"] == 121
    assert body["frame_rate"] == 24.0
    assert body["num_inference_steps"] == 8
    assert body["guidance_scale"] == 1.0
    assert body["seed"] == 3407
    assert body["seed_str"] == "3407"
    assert stub.calls[-1]["op"] == "generate_video"


def test_generate_decodes_base64_image_input(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)
    stub._loaded = True
    stub._repo = "unsloth/Qwen-Image-Edit-2511-GGUF"

    import base64
    import io

    source = Image.new("RGBA", (12, 16), color = (255, 0, 0, 128))
    buf = io.BytesIO()
    source.save(buf, format = "PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    r = c.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "turn the object blue",
            "image_b64": encoded,
            "width": 256,
            "height": 256,
            "num_inference_steps": 4,
        },
    )

    assert r.status_code == 200, r.text
    generate_call = stub.calls[-1]
    assert generate_call["op"] == "generate"
    assert len(generate_call["input_images"]) == 1
    assert generate_call["input_images"][0].mode == "RGBA"
    assert generate_call["input_images"][0].size == (12, 16)
    assert r.json()["output_count"] == 1


def test_generate_rejects_duplicate_image_input_fields(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)
    stub._loaded = True

    r = c.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "x",
            "image_b64": "AAAA",
            "images_b64": ["AAAA"],
        },
    )

    assert r.status_code == 422


def test_load_forwards_text_encoder_gguf_fields(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)

    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "unsloth/FLUX.2-dev-GGUF",
            "gguf_filename": "flux2-dev-Q4_K_M.gguf",
            "base_repo": "black-forest-labs/FLUX.2-dev",
            "text_encoder_gguf_repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
            "text_encoder_gguf_filename": "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
            "gguf_quantized_cpu_resident": True,
            "gguf_pin_cpu_resident": True,
        },
    )
    assert r.status_code == 200, r.text

    assert stub.calls[-1] == {
        "op": "load",
        "repo_id": "unsloth/FLUX.2-dev-GGUF",
        "gguf_filename": "flux2-dev-Q4_K_M.gguf",
        "base_repo": "black-forest-labs/FLUX.2-dev",
        "text_encoder_gguf_repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "text_encoder_gguf_filename": "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
        "text_encoder_gguf_component": None,
        "prompt_enhancer_gguf_repo": None,
        "prompt_enhancer_gguf_filename": None,
        "lora_repo": None,
        "lora_weight_name": None,
        "lora_adapter_name": None,
        "lora_scale": None,
        "lora_fuse": False,
        "family_override": None,
        "hf_token": None,
        "enable_model_cpu_offload": True,
        "offload_policy": None,
        "gguf_quantized_cpu_resident": True,
        "gguf_pin_cpu_resident": True,
        "ignore_public_load_pending_workload": "diffusion",
    }


def test_load_forwards_lora_fields(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)

    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "owner/FLUX.1-finetune-diffusers",
            "family": "flux.1",
            "lora_repo": "owner/my-flux-lora",
            "lora_weight_name": "pytorch_lora_weights.safetensors",
            "lora_adapter_name": "studio-style",
            "lora_scale": 0.75,
            "lora_fuse": True,
        },
    )
    assert r.status_code == 200, r.text

    assert stub.calls[-1]["lora_repo"] == "owner/my-flux-lora"
    assert stub.calls[-1]["lora_weight_name"] == "pytorch_lora_weights.safetensors"
    assert stub.calls[-1]["lora_adapter_name"] == "studio-style"
    assert stub.calls[-1]["lora_scale"] == 0.75
    assert stub.calls[-1]["lora_fuse"] is True


def test_load_forwards_offload_policy(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)

    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "unsloth/Qwen-Image-Edit-GGUF",
            "gguf_filename": "qwen-image-edit-Q4_K_M.gguf",
            "offload_policy": "balanced",
        },
    )
    assert r.status_code == 200, r.text

    assert stub.calls[-1]["offload_policy"] == "balanced"
    assert stub.calls[-1]["enable_model_cpu_offload"] is True
    assert stub.calls[-1]["gguf_quantized_cpu_resident"] is None
    assert stub.calls[-1]["gguf_pin_cpu_resident"] is None


def test_generate_rejects_off_grid_size(app_with_stub):
    app, stub = app_with_stub
    c = TestClient(app)
    c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "unsloth/FLUX.2-klein-4B-GGUF",
            "gguf_filename": "x.gguf",
        },
    )
    r = c.post(
        "/api/inference/images/generate",
        json = {"prompt": "x", "width": 513, "height": 512},
    )
    # Pydantic v2 wraps validator errors in 422 by default.
    assert r.status_code in (400, 422), r.text


def test_unload_clears_state(app_with_stub):
    app, _ = app_with_stub
    c = TestClient(app)
    c.post(
        "/api/inference/images/load",
        json = {"repo_id": "unsloth/FLUX.2-klein-4B-GGUF", "gguf_filename": "x.gguf"},
    )
    r = c.post("/api/inference/images/unload")
    assert r.status_code == 200
    assert r.json()["is_loaded"] is False
    r = c.get("/api/inference/images/status")
    assert r.json()["is_loaded"] is False


def test_load_rejects_embedded_hf_token(app_with_stub):
    """Round 15 P1 #5: URL-embedded ``hf_xxxxx`` tokens in repo_id /
    base_repo must be rejected with 422 so they never reach
    ``self._repo_id`` and get echoed back by ``status()``."""
    app, _ = app_with_stub
    c = TestClient(app)
    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "https://hf_abcdefghij0123456789@huggingface.co/owner/repo",
        },
    )
    assert r.status_code == 422, r.text
    body = r.json()
    text = repr(body).lower()
    assert "hf_token" in text or "embed" in text
    # base_repo is also rejected.
    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "owner/repo",
            "gguf_filename": "x.gguf",
            "base_repo": "https://hf_abcdefghij0123456789@huggingface.co/base/repo",
        },
    )
    assert r.status_code == 422, r.text
    # The text-encoder GGUF fields are also echoed through status/log
    # paths, so reject embedded credentials there too.
    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "owner/repo",
            "gguf_filename": "x.gguf",
            "text_encoder_gguf_repo": "https://hf_abcdefghij0123456789@huggingface.co/base/repo",
        },
    )
    assert r.status_code == 422, r.text
    # LoRA identifiers are also reflected through status/log paths.
    r = c.post(
        "/api/inference/images/load",
        json = {
            "repo_id": "owner/repo",
            "lora_repo": "https://hf_abcdefghij0123456789@huggingface.co/owner/lora",
        },
    )
    assert r.status_code == 422, r.text


def test_load_rejects_control_chars_in_repo_id(app_with_stub):
    """Newline-laden repo ids must be rejected by Pydantic BEFORE the
    log line that echoes them. Catches log-injection from authenticated
    callers (issues a 422 instead of forging a fake log line)."""
    app, _ = app_with_stub
    c = TestClient(app)
    r = c.post(
        "/api/inference/images/load",
        json = {"repo_id": "owner/model\nFAKE_LOG_LINE"},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    text = repr(body).lower()
    assert "control" in text or "repo_id" in text


def test_generate_rejects_oversize_seed(app_with_stub):
    """Huge seeds raise inside torch.Generator.manual_seed; Pydantic
    must clamp first with a 422 instead of a 500 traceback."""
    app, _ = app_with_stub
    c = TestClient(app)
    c.post(
        "/api/inference/images/load",
        json = {"repo_id": "unsloth/FLUX.2-klein-4B-GGUF", "gguf_filename": "x.gguf"},
    )
    r = c.post(
        "/api/inference/images/generate",
        json = {"prompt": "x", "seed": 2**100},
    )
    assert r.status_code == 422, r.text


def test_generate_accepts_uint64_max_seed(app_with_stub):
    """Boundary value: 2**64 - 1 (uint64 max) is the largest seed
    torch.Generator on CPU accepts; reject would frustrate users
    who paste large seeds from other tooling."""
    app, _ = app_with_stub
    c = TestClient(app)
    c.post(
        "/api/inference/images/load",
        json = {"repo_id": "unsloth/FLUX.2-klein-4B-GGUF", "gguf_filename": "x.gguf"},
    )
    r = c.post(
        "/api/inference/images/generate",
        json = {"prompt": "x", "seed": (2**64) - 1},
    )
    # The fake backend returns 200 on success; we only care that the
    # request did NOT 422 on seed bounds.
    assert r.status_code != 422, r.text
