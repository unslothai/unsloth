# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /v1/images/generations output_format / output_compression, on the real gallery (tmp dir); the
diffusion backend is faked and returns real PIL images, so no GPU."""

from __future__ import annotations

import base64
import io
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import core.inference.diffusion as diffusion_module
import core.inference.image_gallery as gallery_module
import routes.inference as inference_routes
from auth.authentication import get_current_subject
from core.inference.diffusion_families import load_identity
from routes.inference import router, studio_router
from utils.api_errors import install_api_error_handlers


def _image(
    width = 256,
    height = 256,
    seed = 0,
):
    # A gradient plus noise-ish pattern, so lossy formats actually differ by quality.
    img = Image.new("RGB", (width, height))
    img.putdata(
        [
            ((x * 7 + seed) % 256, (y * 5 + x * y) % 256, (x ^ y ^ seed) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return img


class _Backend:
    def __init__(self):
        self.calls = 0

    @property
    def is_loaded(self):
        return True

    def status(self):
        return {
            "loaded": True,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "family": "z-image",
            "base_repo": None,
            "device": "cpu",
            "dtype": "float32",
            "cpu_offload": False,
        }

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
        seed = None,
        negative_prompt = None,
    ):
        assert expected_load in (None, load_identity("unsloth/Z-Image-Turbo-GGUF", None, "z-image"))
        self.calls += 1
        return {
            "images": [_image(width, height, seed = i) for i in range(batch_size)],
            "seed": 4242,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "negative_prompt": None,
        }


@pytest.fixture
def gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(monkeypatch, gallery):
    backend = _Backend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    # Fixed clock and ids, so two runs of one request can be compared byte for byte.
    ids = iter(f"{i:032x}" for i in range(1000))
    monkeypatch.setattr(gallery_module.uuid, "uuid4", lambda: type("U", (), {"hex": next(ids)})())
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    with TestClient(app) as cli:
        cli.backend = backend  # type: ignore[attr-defined]
        yield cli


def _post(client, **body):
    return client.post(
        "/v1/images/generations", json = {"prompt": "a sloth", "size": "256x256", **body}
    )


def _gallery_pngs(
    gallery,
    count,
    timeout = 10.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = sorted(gallery.glob("*.png"))
        if len(files) >= count and inference_routes._diffusion_persist_active == 0:
            return files
        time.sleep(0.02)
    raise AssertionError(f"gallery never reached {count} PNGs: {sorted(gallery.iterdir())}")


def test_default_b64_is_the_gallery_png_without_reading_it_back(client, gallery, monkeypatch):
    def _no_read_back(_image_id):
        raise AssertionError("the response must base64 the bytes already in memory")

    monkeypatch.setattr(gallery_module, "image_b64", _no_read_back)
    resp = _post(client, response_format = "b64_json")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"created", "data"}  # no output_format echo on a default request
    (png,) = _gallery_pngs(gallery, 1)
    assert base64.b64decode(body["data"][0]["b64_json"]) == png.read_bytes()


def test_explicit_png_matches_the_default_bytes(client, gallery):
    default = _post(client, response_format = "b64_json").json()
    explicit = _post(
        client, response_format = "b64_json", output_format = "png", output_compression = 10
    ).json()
    assert explicit["output_format"] == "png"
    assert set(explicit) == {"created", "data", "output_format"}
    # Same pixels and recipe except the clock: compare decoded pixels and that both are the gallery files.
    a, b = (base64.b64decode(r["data"][0]["b64_json"]) for r in (default, explicit))
    files = _gallery_pngs(gallery, 2)
    assert {a, b} == {f.read_bytes() for f in files}
    assert Image.open(io.BytesIO(b)).format == "PNG"
    assert Image.open(io.BytesIO(a)).tobytes() == Image.open(io.BytesIO(b)).tobytes()


def test_null_output_format_is_png(client):
    resp = _post(client, response_format = "b64_json", output_format = None)
    assert resp.status_code == 200, resp.text
    assert (
        Image.open(io.BytesIO(base64.b64decode(resp.json()["data"][0]["b64_json"]))).format == "PNG"
    )


@pytest.mark.parametrize("fmt, pil_format", [("jpeg", "JPEG"), ("webp", "WEBP")])
def test_lossy_formats_round_trip_and_the_gallery_keeps_a_png(client, gallery, fmt, pil_format):
    resp = _post(client, response_format = "b64_json", output_format = fmt, output_compression = 80, n = 2)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output_format"] == fmt
    assert len(body["data"]) == 2
    for item in body["data"]:
        assert set(item) == {"b64_json"}
        decoded = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
        assert decoded.format == pil_format and decoded.size == (256, 256)
    files = _gallery_pngs(gallery, 2)
    for path in files:
        with Image.open(path) as stored:
            assert stored.format == "PNG"
            # Lossless and carrying the recipe, exactly like a png request's file.
            assert stored.text["unsloth"]
    pixels = {Image.open(p).convert("RGB").tobytes() for p in files}
    assert pixels == {_image(256, 256, seed = i).tobytes() for i in range(2)}


def test_lossy_format_without_response_format_answers_inline(client, gallery):
    resp = _post(client, output_format = "jpeg")
    assert resp.status_code == 200, resp.text
    item = resp.json()["data"][0]
    assert "url" not in item
    assert base64.b64decode(item["b64_json"])[:3] == b"\xff\xd8\xff"
    _gallery_pngs(gallery, 1)


def test_keyless_caller_may_ask_for_jpeg_without_response_format(client, gallery, monkeypatch):
    import auth.authentication as authentication

    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _r: True)
    resp = _post(client, output_format = "webp")
    assert resp.status_code == 200, resp.text
    _gallery_pngs(gallery, 1)


def test_compression_drives_the_lossy_size(client, gallery):
    sizes = {}
    for quality in (5, 95):
        resp = _post(
            client, response_format = "b64_json", output_format = "jpeg", output_compression = quality
        )
        sizes[quality] = len(base64.b64decode(resp.json()["data"][0]["b64_json"]))
    assert sizes[5] < sizes[95]
    _gallery_pngs(gallery, 2)


@pytest.mark.parametrize(
    "extra, param",
    [
        ({"output_compression": -1}, "output_compression"),
        ({"output_compression": 101}, "output_compression"),
        ({"output_compression": "high"}, "output_compression"),
        ({"output_format": "gif"}, "output_format"),
        ({"output_format": "bmp"}, "output_format"),
        ({"output_format": "jpeg", "response_format": "url"}, "output_format"),
    ],
)
def test_bad_output_fields_are_a_400_before_any_run(client, gallery, extra, param):
    resp = _post(client, **extra)
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == param
    assert client.backend.calls == 0
    assert list(gallery.glob("*.png")) == []


@pytest.mark.parametrize("quality", [0, 100])
def test_compression_bounds_are_inclusive(client, gallery, quality):
    resp = _post(
        client, response_format = "b64_json", output_format = "jpeg", output_compression = quality
    )
    assert resp.status_code == 200, resp.text
    _gallery_pngs(gallery, 1)


def test_url_response_still_links_the_written_png(client, gallery):
    resp = _post(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"created", "data"}
    url = body["data"][0]["url"]
    (png,) = sorted(gallery.glob("*.png"))  # written before the response, not after
    fetched = client.get(url.replace("http://testserver", ""))
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.content == png.read_bytes()


def test_explicit_png_with_url_is_allowed(client, gallery):
    resp = _post(client, output_format = "png", response_format = "url")
    assert resp.status_code == 200, resp.text
    assert "url" in resp.json()["data"][0]


def test_a_failed_background_gallery_write_keeps_the_response_and_releases_liveness(
    client, gallery, monkeypatch
):
    def _broken(_items):
        raise OSError("disk full")

    monkeypatch.setattr(gallery_module, "save_many", _broken)
    resp = _post(client, response_format = "b64_json", output_format = "jpeg")
    assert resp.status_code == 200, resp.text
    deadline = time.monotonic() + 10
    while inference_routes._diffusion_persist_active and time.monotonic() < deadline:
        time.sleep(0.02)
    assert inference_routes._diffusion_persist_active == 0
    assert list(gallery.glob("*.png")) == []


def test_the_liveness_hold_is_taken_before_a_lossy_response_returns(client, gallery):
    # The background write takes the persist marker before the handler returns, so between the
    # response and the gallery record liveness never reads idle: either the hold or the file is there.
    for _ in range(5):
        resp = _post(client, response_format = "b64_json", output_format = "jpeg")
        assert resp.status_code == 200, resp.text
        held = inference_routes._diffusion_persist_active
        written = len(list(gallery.glob("*.png")))
        assert held > 0 or written >= 1
        _gallery_pngs(gallery, written + (1 if held else 0))


def test_null_response_format_with_a_lossy_format_answers_inline(client, gallery):
    """An SDK that sends response_format: null gets the omitted-field default, not a 400."""
    resp = _post(client, response_format = None, output_format = "jpeg")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data and data[0].get("b64_json") and not data[0].get("url")


def test_the_liveness_hold_covers_lossy_encoding(client, gallery, monkeypatch):
    """Liveness must not read idle while a finished image is still being encoded as jpeg/webp."""
    seen = []
    real = gallery_module.encode_lossy

    def spy(image, fmt, quality):
        seen.append(inference_routes._diffusion_persist_active)
        return real(image, fmt, quality)

    monkeypatch.setattr(gallery_module, "encode_lossy", spy)
    resp = _post(client, response_format = "b64_json", output_format = "jpeg")
    assert resp.status_code == 200, resp.text
    assert seen and all(n > 0 for n in seen)
