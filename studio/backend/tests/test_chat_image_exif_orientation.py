# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import os
import sys
from io import BytesIO
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest
from PIL import Image, JpegImagePlugin

from routes.inference import _decode_and_resize_image, _normalize_openai_image_parts_for_llama

RED, GREEN, BLUE, YELLOW = (220, 20, 20), (20, 200, 20), (20, 20, 220), (220, 200, 20)

_AS_DISPLAYED = {
    1: ((300, 100), ["red", "green", "blue", "yellow"]),
    2: ((300, 100), ["green", "red", "yellow", "blue"]),
    3: ((300, 100), ["yellow", "blue", "green", "red"]),
    4: ((300, 100), ["blue", "yellow", "red", "green"]),
    5: ((100, 300), ["red", "blue", "green", "yellow"]),
    6: ((100, 300), ["blue", "red", "yellow", "green"]),
    7: ((100, 300), ["yellow", "green", "blue", "red"]),
    8: ((100, 300), ["green", "yellow", "red", "blue"]),
}


def _photo(
    orientation,
    fmt = "JPEG",
    subsampling = 0,
) -> bytes:
    img = Image.new("RGB", (300, 100))
    img.paste(RED, (0, 0, 150, 50))
    img.paste(GREEN, (150, 0, 300, 50))
    img.paste(BLUE, (0, 50, 150, 100))
    img.paste(YELLOW, (150, 50, 300, 100))
    exif = img.getexif()
    if orientation is not None:
        exif[0x0112] = orientation
    buf = BytesIO()
    extra = {"quality": 95, "subsampling": subsampling} if fmt == "JPEG" else {}
    img.save(buf, format = fmt, exif = exif, **extra)
    return buf.getvalue()


def _quadrants(img) -> list:
    img = img.convert("RGB")
    w, h = img.size
    names = {RED: "red", GREEN: "green", BLUE: "blue", YELLOW: "yellow"}

    def at(fx, fy):
        px = img.getpixel((int(w * fx), int(h * fy)))
        return names[min(names, key = lambda c: sum((a - b) ** 2 for a, b in zip(c, px)))]

    return [at(0.25, 0.25), at(0.75, 0.25), at(0.25, 0.75), at(0.75, 0.75)]


def _sent_to_llama(raw: bytes, mime = "image/jpeg") -> str:
    url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": url}}
    _normalize_openai_image_parts_for_llama([{"role": "user", "content": [part]}])
    return part["image_url"]["url"]


@pytest.mark.parametrize("orientation", sorted(_AS_DISPLAYED))
def test_llama_server_gets_a_tagged_jpeg_the_way_the_chat_shows_it(orientation):
    head, b64 = _sent_to_llama(_photo(orientation)).split(",", 1)
    assert head == "data:image/jpeg;base64"
    sent = Image.open(BytesIO(base64.b64decode(b64)))
    assert sent.format == "JPEG"
    size, quadrants = _AS_DISPLAYED[orientation]
    assert sent.size == size
    assert _quadrants(sent) == quadrants
    assert sent.getexif().get(0x0112) in (None, 1)


@pytest.mark.parametrize("orientation", [None, 1])
def test_an_upright_jpeg_still_reaches_llama_server_byte_for_byte(orientation):
    raw = _photo(orientation)
    assert _sent_to_llama(raw) == "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def test_llama_server_gets_a_tagged_png_upright():
    head, b64 = _sent_to_llama(_photo(6, "PNG"), "image/png").split(",", 1)
    assert head == "data:image/png;base64"
    sent = Image.open(BytesIO(base64.b64decode(b64)))
    assert sent.size == (100, 300)
    assert _quadrants(sent) == _AS_DISPLAYED[6][1]


def test_a_tagged_png_is_decoded_once(monkeypatch):
    opened = []
    real_open = Image.open

    def counting_open(*args, **kwargs):
        opened.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_open)
    _sent_to_llama(_photo(6, "PNG"), "image/png")
    assert len(opened) == 1


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_a_tagged_jpeg_keeps_its_quality_and_colour_detail(subsampling):
    raw = _photo(6, subsampling = subsampling)
    head, b64 = _sent_to_llama(raw).split(",", 1)
    assert head == "data:image/jpeg;base64"
    sent = Image.open(BytesIO(base64.b64decode(b64)))
    assert sent.quantization == Image.open(BytesIO(raw)).quantization
    assert JpegImagePlugin.get_sampling(sent) == subsampling


def test_a_tagged_jpeg_whose_tables_start_at_one_is_still_accepted():
    img = Image.new("L", (300, 100))
    exif = img.getexif()
    exif[0x0112] = 6
    buf = BytesIO()
    img.save(buf, format = "JPEG", exif = exif, quality = 95)
    raw = bytearray(buf.getvalue())
    raw[raw.index(b"\xff\xdb") + 4] = 1  # DQT defines table 1
    raw[raw.index(b"\xff\xc0") + 12] = 1  # SOF0's one component reads it
    head, b64 = _sent_to_llama(bytes(raw)).split(",", 1)
    assert head == "data:image/jpeg;base64"
    sent = Image.open(BytesIO(base64.b64decode(b64)))
    assert sent.size == (100, 300)
    assert list(sent.quantization.values()) == list(Image.open(BytesIO(raw)).quantization.values())


def test_a_webp_tag_is_skipped_the_way_chromium_skips_it():
    head, b64 = _sent_to_llama(_photo(6, "WEBP"), "image/webp").split(",", 1)
    assert head == "data:image/png;base64"
    sent = Image.open(BytesIO(base64.b64decode(b64)))
    assert sent.size == (300, 100)
    assert _quadrants(sent) == _AS_DISPLAYED[1][1]


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_the_transformers_and_mlx_path_decodes_a_tagged_jpeg_upright(orientation):
    backend = SimpleNamespace(resize_image = lambda img: img)
    encoded = base64.b64encode(_photo(orientation)).decode("ascii")
    img = _decode_and_resize_image(backend, encoded)
    size, quadrants = _AS_DISPLAYED[orientation]
    assert img.size == size
    assert _quadrants(img) == quadrants
    assert img.getexif().get(0x0112) in (None, 1)
