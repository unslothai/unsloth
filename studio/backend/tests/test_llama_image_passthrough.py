# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Images reach llama-server as the client sent them when its stb_image can decode them."""

import base64
import os
import sys
from io import BytesIO

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest
from fastapi import HTTPException
from PIL import Image

from models.inference import ChatCompletionRequest
from routes.inference import (
    _llama_image_data_url,
    _normalize_openai_image_parts_for_llama,
    _openai_messages_for_passthrough,
    _stb_reads_jpeg,
    _stb_reads_png,
)


def _photo(mode = "RGB", size = (64, 48)):
    img = Image.effect_noise(size, 60).convert("L")
    if mode == "RGB":
        img = Image.merge("RGB", (img, img.rotate(90, expand = False), img.transpose(0)))
    return img.convert(mode) if img.mode != mode else img


def _encode(img, fmt, **kw) -> bytes:
    buf = BytesIO()
    img.save(buf, format = fmt, **kw)
    return buf.getvalue()


def _split(url: str) -> tuple[str, bytes]:
    head, b64 = url.split(",", 1)
    return head, base64.b64decode(b64)


def _patched_frame(
    raw: bytes,
    marker: int = None,
    precision: int = None,
) -> bytes:
    """The same JPEG with its SOF marker or sample precision rewritten."""
    i = raw.index(b"\xff\xc0")
    out = bytearray(raw)
    if marker is not None:
        out[i + 1] = marker
    if precision is not None:
        out[i + 4] = precision
    return bytes(out)


@pytest.mark.parametrize(
    "img, kw",
    [
        (_photo(), {}),
        (_photo(), {"progressive": True}),
        (_photo(), {"subsampling": 0}),
        (_photo("L"), {}),
    ],
    ids = ["baseline", "progressive", "444", "grey"],
)
def test_stb_readable_jpeg_is_forwarded_byte_for_byte(img, kw):
    raw = _encode(img, "JPEG", quality = 90, **kw)
    head, out = _split(_llama_image_data_url(raw))
    assert head == "data:image/jpeg;base64"
    assert out == raw


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "I;16"])
def test_png_is_forwarded_byte_for_byte(mode):
    raw = _encode(_photo("L").convert(mode), "PNG")
    head, out = _split(_llama_image_data_url(raw))
    assert head == "data:image/png;base64"
    assert out == raw


def test_jpeg_is_not_inflated():
    # PNG conversion can inflate photos past llama-server's request limit.
    raw = _encode(_photo(size = (640, 480)), "JPEG", quality = 90)
    assert len(_split(_llama_image_data_url(raw))[1]) == len(raw)


@pytest.mark.parametrize(
    "raw",
    [
        _encode(_photo(), "WEBP"),
        _encode(_photo(), "GIF"),
        _encode(_photo(), "BMP"),
        _encode(_photo(), "TIFF"),
        _encode(_photo("CMYK"), "JPEG"),
    ],
    ids = ["webp", "gif", "bmp", "tiff", "cmyk-jpeg"],
)
def test_other_formats_are_still_reencoded_to_png(raw):
    head, out = _split(_llama_image_data_url(raw))
    assert head == "data:image/png;base64"
    assert out.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(BytesIO(out)).mode == "RGB"


def test_jpeg_frames_stb_rejects_are_not_passed_through():
    raw = _encode(_photo(), "JPEG")
    assert _stb_reads_jpeg(raw)
    assert _stb_reads_jpeg(_patched_frame(raw, marker = 0xC1))  # Extended sequential.
    assert not _stb_reads_jpeg(_patched_frame(raw, marker = 0xC3))  # Lossless.
    assert not _stb_reads_jpeg(_patched_frame(raw, marker = 0xC9))  # Arithmetic.
    assert not _stb_reads_jpeg(_patched_frame(raw, precision = 12))
    assert not _stb_reads_jpeg(_encode(_photo("CMYK"), "JPEG"))
    assert not _stb_reads_jpeg(raw[:20])  # Ends before any frame header.


# The 64x64 PNG from the GGUF vision smoke test: its deflate stream is cut short and it has no
# IEND. Pillow decodes it; stb_image rejects it.
_UNTERMINATED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYklEQVR4nO3PMQ0AIADAMEAI/k"
    "UhBhEcDcmqYJtn7/GzpQNeNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA"
    "1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaBdCJ0BmMJ25zMAAAAASUVORK5CYII="
)


def test_png_chunks_stb_rejects_are_reencoded():
    raw = _encode(_photo(), "PNG")
    assert _stb_reads_png(raw)
    assert not _stb_reads_png(raw[: raw.rindex(b"IEND") - 4])  # No IEND.
    assert not _stb_reads_png(_UNTERMINATED_PNG)
    head, out = _split(_llama_image_data_url(_UNTERMINATED_PNG))
    assert head == "data:image/png;base64"
    assert _stb_reads_png(out)
    assert Image.open(BytesIO(out)).size == (64, 64)


@pytest.mark.parametrize(
    "raw",
    [
        b"not an image",
        b"\x89PNG\r\n\x1a\n" + b"\0" * 32,
        b"\xff\xd8\xff\xe0garbage",
        # Reject truncated data even when stb_image would accept it.
        _encode(_photo(size = (640, 480)), "JPEG", quality = 90)[:4000],
        _encode(_photo(size = (640, 480)), "PNG")[:4000],
    ],
    ids = ["text", "png-signature-only", "jpeg-soi-only", "truncated-jpeg", "truncated-png"],
)
def test_undecodable_bytes_still_fail_as_400(raw):
    url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    with pytest.raises(HTTPException) as exc:
        _normalize_openai_image_parts_for_llama(msgs)
    assert exc.value.status_code == 400


def test_mislabelled_jpeg_gets_its_real_mime_and_keeps_detail():
    raw = _encode(_photo(), "JPEG")
    b64 = base64.b64encode(raw).decode("ascii")
    part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
    }
    _normalize_openai_image_parts_for_llama([{"role": "user", "content": [part]}])
    assert part["image_url"] == {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}


def test_legacy_image_base64_jpeg_is_forwarded_unchanged():
    raw = _encode(_photo(), "JPEG")
    b64 = base64.b64encode(raw).decode("ascii")
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "describe"}],
        image_base64 = b64,
    )
    messages = _openai_messages_for_passthrough(payload, vision = True)
    parts = [p for p in messages[-1]["content"] if p.get("type") == "image_url"]
    assert [p["image_url"]["url"] for p in parts] == [f"data:image/jpeg;base64,{b64}"]
