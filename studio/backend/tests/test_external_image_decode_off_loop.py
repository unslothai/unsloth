# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External image connections decode the provider's base64 images on a worker thread, in order."""

from __future__ import annotations

import asyncio
import base64
import io

import httpx
import pytest
from PIL import Image, ImageFile

import routes.inference as inference_routes
from .test_diffusion_routes import (  # noqa: F401 -- client is a fixture
    _external_image_connection,
    _post_generate,
    client,
)


def _png(size, color) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _serve(monkeypatch, items):
    _external_image_connection(
        monkeypatch,
        lambda req: httpx.Response(200, json = {"data": [{"b64_json": item} for item in items]}),
    )


def _generate(client, n):
    return _post_generate(
        client, prompt = "a sloth", batch_size = n, provider_id = "conn-1", model = "qwen-image"
    )


def test_decode_runs_off_the_event_loop_and_keeps_provider_order(client, monkeypatch):
    sizes = [(64, 32), (16, 48), (40, 40)]
    _serve(monkeypatch, [_png(s, (i * 60, 10, 10)) for i, s in enumerate(sizes)])
    on_loop = []
    load = ImageFile.ImageFile.load

    def recording(self):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return load(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", recording)
    resp = _generate(client, len(sizes))
    assert resp.status_code == 200, resp.text
    assert [(i["width"], i["height"]) for i in resp.json()["images"]] == sizes
    assert [i["batch_index"] for i in resp.json()["images"]] == [0, 1, 2]
    assert on_loop[: len(sizes)] == [False] * len(sizes)


def test_decode_helper_loads_every_image_in_order():
    images = inference_routes._decode_external_images(
        [_png((3, 5), (1, 2, 3)), _png((7, 2), (4, 5, 6))]
    )
    assert [im.size for im in images] == [(3, 5), (7, 2)]
    assert images[1].getpixel((0, 0)) == (4, 5, 6)


@pytest.mark.parametrize(
    "bad",
    [
        "!!not base64**",
        base64.b64encode(b"not an image at all").decode(),
        base64.b64encode(base64.b64decode(_png((64, 64), (9, 9, 9)))[:60]).decode(),
        None,
    ],
)
def test_an_unusable_image_is_still_a_502_and_saves_nothing(client, monkeypatch, bad):
    _serve(monkeypatch, [_png((8, 8), (1, 1, 1)), bad])
    resp = _generate(client, 2)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "The image endpoint did not return a usable image."
    assert client.get("/api/inference/images/gallery").json()["images"] == []
