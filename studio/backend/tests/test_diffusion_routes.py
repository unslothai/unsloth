# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Route-level tests for ``/api/inference/images/*``.

Mounts the actual ``inference_router`` on a fresh FastAPI app with the
auth dependency replaced by a stub so we exercise the same FastAPI
handlers Studio ships in production. The diffusion backend is replaced
with an in-memory stub so we don't need diffusers / GPUs to run these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


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
            "base_repo": "black-forest-labs/FLUX.2-klein" if self._loaded else None,
            "gguf_filename": None,
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

    def generate_image(self, **kw):
        self.calls.append({"op": "generate", **kw})
        return Image.new("RGB", (kw["width"], kw["height"]), color = (123, 45, 67))


@pytest.fixture
def app_with_stub(monkeypatch):
    """Build a FastAPI app that mounts the real inference router with
    auth disabled and the diffusion backend swapped for a stub."""
    from routes import inference as inf
    import core.inference.diffusion as d

    stub = _FakeBackend()
    # Override the singleton accessor the route uses.
    monkeypatch.setattr(d, "get_diffusion_backend", lambda: stub)
    monkeypatch.setattr(inf, "_get_diffusion_backend", lambda: stub)

    app = FastAPI()
    app.include_router(inf.router, prefix = "/api/inference")
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
        json = {"prompt": "x", "seed": 2 ** 100},
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
        json = {"prompt": "x", "seed": (2 ** 64) - 1},
    )
    # The fake backend returns 200 on success; we only care that the
    # request did NOT 422 on seed bounds.
    assert r.status_code != 422, r.text
