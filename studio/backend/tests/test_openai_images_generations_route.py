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
from core.inference.api_monitor import api_monitor
from core.inference.diffusion_families import (
    DiffusionModelReplacedError,
    default_generation_params,
    load_identity,
)
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
        ("unsloth/FLUX.2-klein-4B-GGUF", (4, 1.0)),
        ("unsloth/Qwen-Image-2512-GGUF", (20, 4.0)),
        ("some/unknown-model", (9, 0.0)),  # fallback
        ("", (9, 0.0)),
    ],
)
def test_default_generation_params(repo_id, expected):
    assert default_generation_params(repo_id) == expected


def test_default_generation_params_specificity_ordering():
    # The "-turbo" / "-schnell" entries must win over their broader siblings; a reorder would silently mis-default.
    assert default_generation_params("x/Z-Image-Turbo") != default_generation_params("x/Z-Image")
    assert default_generation_params("x/FLUX.1-schnell") != default_generation_params(
        "x/FLUX.1-dev"
    )


def test_default_generation_params_falls_back_to_base_repo():
    # A local-path load: repo_id names no model, so the resolved base repo identifies it (and separates dev from schnell).
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
        # Supported workflows: a list, or a repo_id -> list map so a replacement changes them.
        workflows = None,
        # (repo_id, base_repo) pairs generate() reports as loaded, one per call: models a
        # replacement landing between the route's status() read and its lock (#9448).
        replaced_by = None,
    ) -> None:
        self._loaded = loaded
        self._repo_id = repo_id
        self._base_repo = base_repo
        self._workflows = workflows
        self._replaced_by = list(replaced_by or [])
        # Model the native sd.cpp engine, which returns a distinct seed per image.
        self._native_seeds = native_seeds
        # When set, generate() raises this; unload_on_generate flips is_loaded off first, to model the eviction race vs an in-pipeline OOM.
        self._generate_error = generate_error
        self._unload_on_generate = unload_on_generate
        self.calls = []

    @property
    def is_loaded(self):
        return self._loaded

    def status(self):
        out = {
            "loaded": self._loaded,
            "repo_id": self._repo_id if self._loaded else None,
            "family": "z-image" if self._loaded else None,
            "base_repo": self._base_repo if self._loaded else None,
            "device": "cpu",
            "dtype": "float32",
            "cpu_offload": False,
        }
        if isinstance(self._workflows, dict):
            out["workflows"] = self._workflows.get(self._repo_id, [])
        elif self._workflows is not None:
            out["workflows"] = self._workflows
        return out

    def generate(
        self,
        *,
        prompt,
        width,
        height,
        steps,
        guidance,
        batch_size = 1,
        expected_load = None,
    ):
        if not self._loaded:
            raise RuntimeError("No diffusion model is loaded.")
        # Both engines refuse in-lock when the caller's snapshot named a different model.
        if self._replaced_by:
            self._repo_id, self._base_repo = self._replaced_by.pop(0)
        loaded = load_identity(self._repo_id, self._base_repo, "z-image")
        if expected_load is not None and expected_load != loaded:
            raise DiffusionModelReplacedError(expected_load, loaded)
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
                expected_load = expected_load,
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
    # Signed link, not the bearer-gated /file route: an OpenAI client downloads this URL with a plain GET and no auth header.
    assert "/images/gallery/img0/file-signed?token=" in item["url"]
    # Z-Image-Turbo defaults (9 steps, 0 guidance) flow into the backend call.
    assert client.backend.calls[0] == dict(
        prompt = "a sloth",
        width = 256,
        height = 256,
        steps = 9,
        guidance = 0.0,
        batch_size = 1,
        expected_load = load_identity("unsloth/Z-Image-Turbo-GGUF", None, "z-image"),
    )


def test_b64_response_shape(client):
    resp = _post(client, {"prompt": "a sloth", "size": "256x256", "response_format": "b64_json"})
    assert resp.status_code == 200
    item = resp.json()["data"][0]
    assert "b64_json" in item and "url" not in item
    assert item["b64_json"] == "QUJD"


def test_local_load_uses_base_repo_for_defaults(monkeypatch):
    # repo_id is a local path naming no model; base_repo identifies FLUX.1-dev, so the route picks 28 steps / 3.5 guidance, not the 9/0 fallback.
    backend = _FakeBackend(repo_id = "/models/my-flux", base_repo = "black-forest-labs/FLUX.1-dev")
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    assert backend.calls[0]["steps"] == 28 and backend.calls[0]["guidance"] == 3.5


def test_pipeline_runtime_error_is_sanitized_500(monkeypatch):
    # A RuntimeError raised inside the pipeline while the model stays loaded (e.g. CUDA OOM) is a sanitized 500, not a 503 echoing the raw text.
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
    # The model is evicted between the readiness check and the call: a RuntimeError with is_loaded now False is the one case that maps to 503.
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
    # A non-RuntimeError from the pipeline must not take the 503 branch (gated on isinstance RuntimeError), so it is a sanitized 500.
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
    # n>1 must persist batch_size in each record so the restore path can replay a batch_index>0 sibling (which shares the batch seed).
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
    # On a no-GPU host the loaded model lives behind the native sd_cpp engine, so the route must query get_active_diffusion_engine or it 503s a usable model.
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
    # The native sd.cpp engine returns a distinct seed per image (base+index), so each record must store its own or a restored batch_index>0 image shows the wrong one.
    backend = _FakeBackend(native_seeds = True)
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256", "n": 3})
    assert resp.status_code == 200
    records = sorted(store.values(), key = lambda r: r["batch_index"])
    assert [r["seed"] for r in records] == [4242, 4243, 4244]


def test_null_fields_coalesce_to_defaults(client):
    # OpenAI marks n/size/response_format nullable-with-default: null means the default.
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


def _signed_link_app(monkeypatch, backend, png: "object"):
    """An app carrying BOTH surfaces: /v1 for the compat POST and /api/inference for the gallery,
    which is where the returned link points. get_current_subject is overridden for the POST only in
    the sense that the signed route never depends on it."""
    from routes.inference import studio_router

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    monkeypatch.setattr(gallery_module, "owned_image_path", lambda i: png if i == "img0" else None)
    return TestClient(app)


def test_url_response_link_is_fetchable_without_the_bearer(monkeypatch, tmp_path):
    # The point of response_format=url: an image client hands data[].url to a plain downloader with no Authorization header.
    # Fetch the returned link with the auth header stripped and assert the PNG comes back.
    png = tmp_path / "img0.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    backend = _FakeBackend()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    store = {}

    def _save(image, meta):
        image_id = f"img{len(store)}"
        record = {**meta, "id": image_id, "url": f"/api/inference/images/gallery/{image_id}/file"}
        store[image_id] = record
        return record

    monkeypatch.setattr(gallery_module, "save", _save)
    cli = _signed_link_app(monkeypatch, backend, png)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    url = resp.json()["data"][0]["url"]

    fetched = cli.get(url.replace("http://testserver", ""), headers = {})
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.content == b"\x89PNG\r\n\x1a\nfake"


def test_signed_image_link_rejects_tampering_and_expiry(monkeypatch, tmp_path):
    # The token names one image and carries its own expiry, so a swapped id, a forged signature and a stale link all 401.
    import routes.inference as inference_routes

    png = tmp_path / "img0.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    backend = _FakeBackend()
    cli = _signed_link_app(monkeypatch, backend, png)
    token = inference_routes._sign_image_id("img0")

    base = "/api/inference/images/gallery"
    assert cli.get(f"{base}/img0/file-signed?token={token}").status_code == 200
    # Same signature, different image.
    assert cli.get(f"{base}/img1/file-signed?token={token}").status_code == 401
    assert cli.get(f"{base}/img0/file-signed?token=img0.9999999999.dead").status_code == 401
    assert cli.get(f"{base}/img0/file-signed?token=nonsense").status_code == 401
    monkeypatch.setattr(inference_routes, "_IMAGE_LINK_TTL", -1)
    expired = inference_routes._sign_image_id("img0")
    assert cli.get(f"{base}/img0/file-signed?token={expired}").status_code == 401


def test_activation_shortfall_is_an_actionable_400(monkeypatch):
    """The one exception here whose text is written FOR the caller. Sanitising it into a bare 500
    left an OpenAI client with a server error for a request only they can fix, while the Studio
    route showed them the resolution, the budget and the remedies."""
    from core.inference.diffusion_memory import ImageActivationShortfallError

    reason = (
        "Generating at 2048x2048 needs about 15.55 GB of working memory, but only about "
        "13.50 GB is usable on this device. Generate at a smaller resolution."
    )
    backend = _FakeBackend(generate_error = ImageActivationShortfallError(reason))
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["message"] == reason
    assert err["param"] == "size"
    # Still typed: an ordinary ValueError keeps its sanitized 500 (test above), so no other raw
    # exception text rides out on the back of this.


def test_generation_opens_a_monitor_row(client):
    api_monitor.clear()
    assert _post(client, {"prompt": "a sloth in space"}).status_code == 200
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/images/generations"
    assert rows[0]["method"] == "POST"
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_preview"] == "a sloth in space"
    # Relabelled to what served, not the informational body.model.
    assert rows[0]["model"] == "unsloth/Z-Image-Turbo-GGUF"


def test_no_loaded_model_records_an_error_row(monkeypatch):
    backend = _FakeBackend(loaded = False)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    api_monitor.clear()
    assert _post(cli, {"prompt": "a sloth"}).status_code == 503
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    # The route's 503 detail, not _friendly_error's generic fallback.
    assert "image model" in rows[0]["error"]


def test_local_directory_load_is_not_leaked_into_the_monitor(monkeypatch):
    # A local pick puts the host directory in repo_id and the row goes out over the tunnel,
    # so the label gets the same path-free treatment as active_model.
    backend = _FakeBackend(repo_id = "/home/ana/models/my-flux")
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    api_monitor.clear()
    assert (
        cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"}).status_code
        == 200
    )
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["model"] == "my-flux"


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "p", "size": "300x300"},
        {"prompt": "p", "size": "abc"},
        {"prompt": "p", "stream": True},
    ],
)
def test_a_refused_request_records_nothing(client, body):
    """size and stream were validated inside the monitor, so a request refused before any
    work still produced a red error row. /audio/speech and /audio/transcriptions reject
    their bad parameters before opening a row, and this route now matches them."""
    api_monitor.clear()
    assert _post(client, body).status_code == 400
    assert api_monitor.snapshot(include_details = False) == []


def test_a_failed_generation_does_not_leak_a_local_path_label(monkeypatch):
    """The relabel only runs after a successful generation, so a failure left whatever the
    client sent in body.model on a row that goes out over the tunnel."""
    backend = _FakeBackend(loaded = False)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    for sent, expected in (
        ("/home/ana/models/my-flux", "my-flux"),
        ("C:\\models\\my-flux", "my-flux"),
        ("unsloth/Z-Image-Turbo-GGUF", "unsloth/Z-Image-Turbo-GGUF"),
    ):
        api_monitor.clear()
        assert _post(cli, {"prompt": "a sloth", "model": sent}).status_code == 503
        row = api_monitor.snapshot(include_details = False)[0]
        assert row["status"] == "error"
        assert row["model"] == expected


# ── model replaced mid-request (#9448) ──────────────────────────────────


def _replacement_client(monkeypatch, backend):
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    return cli, store


def test_generation_pins_the_status_read_it_derived_its_params_from(monkeypatch):
    # Without the pin the backend cannot tell a stale snapshot from a fresh one.
    backend = _FakeBackend()
    cli, _ = _replacement_client(monkeypatch, backend)
    assert cli.post("/v1/images/generations", json = {"prompt": "p"}).status_code == 200
    assert backend.calls[0]["expected_load"] == load_identity(
        "unsloth/Z-Image-Turbo-GGUF", None, "z-image"
    )


def test_replacement_retries_once_with_the_new_models_params(monkeypatch):
    # Z-Image-Turbo (9 steps, guidance 0) is replaced by Z-Image (20 steps, guidance 4). The first
    # attempt is refused in-lock; the retry must re-derive from fresh state, not reuse the turbo's.
    backend = _FakeBackend(replaced_by = [("unsloth/Z-Image-GGUF", None)])
    cli, _ = _replacement_client(monkeypatch, backend)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    assert len(backend.calls) == 1  # the refused attempt never generated
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (20, 4.0)


def test_second_replacement_is_a_503_not_a_sanitized_500(monkeypatch):
    # Bounded at one retry, and the client is told to retry rather than handed a server error.
    backend = _FakeBackend(replaced_by = [("a/one", None), ("b/two", None)])
    cli, _ = _replacement_client(monkeypatch, backend)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 503
    assert backend.calls == []
    err = resp.json()["error"]
    assert err["type"] == "api_error" and err["param"] == "model"
    assert "replaced" in err["message"]


def test_replacement_into_an_edit_only_model_is_a_400(monkeypatch):
    # The retry re-decides eligibility too, not just the parameters.
    edit_only = "unsloth/Qwen-Image-Edit-GGUF"
    backend = _FakeBackend(
        replaced_by = [(edit_only, None)],
        workflows = {"unsloth/Z-Image-Turbo-GGUF": ["txt2img"], edit_only: ["img2img"]},
    )
    cli, _ = _replacement_client(monkeypatch, backend)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 400
    assert "edit-only" in resp.json()["error"]["message"]
    assert backend.calls == []


def test_edit_only_model_is_an_actionable_400(monkeypatch):
    # No test covered this gate, so a broken error-envelope call there turned it into a 500.
    backend = _FakeBackend(workflows = ["img2img", "inpaint"])
    cli, _ = _replacement_client(monkeypatch, backend)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "model"
    assert "edit-only" in err["message"]
    assert backend.calls == []


def test_txt2img_capable_model_passes_the_gate(monkeypatch):
    # A model advertising txt2img among its workflows must not be caught by the edit-only refusal.
    backend = _FakeBackend(workflows = ["txt2img", "img2img"])
    cli, _ = _replacement_client(monkeypatch, backend)
    assert cli.post("/v1/images/generations", json = {"prompt": "p"}).status_code == 200


def test_backend_reporting_no_repo_id_still_generates(monkeypatch):
    # An engine reporting no repo_id still pins its base and family, and still generates.
    backend = _FakeBackend(repo_id = None)
    cli, _ = _replacement_client(monkeypatch, backend)
    assert cli.post("/v1/images/generations", json = {"prompt": "p"}).status_code == 200
    assert backend.calls[0]["expected_load"] == load_identity(None, None, "z-image")


def test_same_repo_reloaded_under_a_different_base_is_a_replacement(monkeypatch):
    # base_repo is settable per load, and decides steps/guidance when repo_id names nothing
    # known. Pinning repo_id alone let FLUX.1-dev's 28 steps / 3.5 reach a schnell pipeline.
    local = "/models/my-ckpt"
    dev, schnell = "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"
    assert default_generation_params(local, dev) != default_generation_params(local, schnell)
    backend = _FakeBackend(repo_id = local, base_repo = dev, replaced_by = [(local, schnell)])
    cli, _ = _replacement_client(monkeypatch, backend)
    resp = cli.post("/v1/images/generations", json = {"prompt": "p", "size": "256x256"})
    assert resp.status_code == 200
    assert len(backend.calls) == 1  # the refused attempt never generated
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (4, 0.0)


def test_the_pin_covers_every_field_the_request_is_derived_from():
    # Whatever the route derives parameters or eligibility from has to be in the pin, or a
    # replacement that differs only there is accepted as the snapshot.
    turbo, base = "unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image"
    pin = load_identity(turbo, base, "z-image")
    assert pin != load_identity("unsloth/Z-Image-GGUF", base, "z-image")  # steps/guidance
    assert pin != load_identity(turbo, "black-forest-labs/FLUX.1-dev", "z-image")  # steps/guidance
    assert pin != load_identity(turbo, base, "qwen-image-edit")  # edit-only verdict
    # None and "" describe the same absent field, so they must not read as a replacement.
    assert load_identity(turbo, None, "z-image") == load_identity(turbo, "", "z-image")
