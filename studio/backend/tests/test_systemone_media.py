# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io

import pytest
from PIL import Image

from core.systemone import media


def image_url(
    fmt = "PNG",
    *,
    size = (16, 16),
    color = "red",
):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = fmt)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[fmt]
    return "data:image/" + mime + ";base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_real_image_containers_are_validated(fmt):
    state, raw = media.prepare("Inspect", [image_url(fmt)], accepts_images = True)
    assert state == "Inspect" and len(raw) == 1
    with Image.open(io.BytesIO(raw[0])) as image:
        assert image.size == (16, 16) and image.format == fmt


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/img.png",
        "data:image/gif;base64,AAAA",
        "data:image/png,AAAA",
        "data:image/png;base64,%%%",
        "data:image/png;base64,AAAA",
        "data:image/png;base64,",
        None,
        3,
    ],
)
def test_remote_unsupported_and_malformed_images_are_refused(url):
    with pytest.raises(media.InvalidMedia):
        media.prepare("Inspect", [url], accepts_images = True)


def test_mime_mismatch():
    with pytest.raises(media.InvalidMedia, match = "MIME"):
        media.prepare(
            "Inspect", [image_url().replace("image/png", "image/jpeg")], accepts_images = True
        )


def test_image_count_counts_top_level_and_state():
    state = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url()}}]}
    ]
    with pytest.raises(media.InvalidMedia, match = "At most 4"):
        media.prepare(state, [image_url()] * 4, accepts_images = True)


def test_state_image_parts_are_extracted_and_text_is_preserved():
    part = {"type": "text", "text": "What color?"}
    state = [
        {
            "role": "user",
            "content": [part, {"type": "image_url", "image_url": {"url": image_url()}}],
        }
    ]
    cleaned, raw = media.prepare(state, None, accepts_images = True)
    assert cleaned == [{"role": "user", "content": [part]}] and len(raw) == 1
    assert len(state[0]["content"]) == 2


@pytest.mark.parametrize(
    "kind", ["input_audio", "audio_url", "video_url", "input_video", "input_image", ["text"]]
)
def test_unsupported_message_parts_are_never_rendered_as_plain_json(kind):
    with pytest.raises(media.InvalidMedia):
        media.prepare(
            [{"role": "user", "content": [{"type": kind, "data": "..."}]}],
            None,
            accepts_images = True,
        )


def test_plain_structured_state_is_not_a_media_upload():
    state = {"type": "audio", "invoice": {"amount": 12}}
    assert media.prepare(state, None, accepts_images = False) == (state, [])


def test_text_only_backend_refuses_media_before_decoding():
    with pytest.raises(media.InvalidMedia, match = "text-only") as exc:
        media.prepare("state", ["malformed"], accepts_images = False)
    assert exc.value.unsupported


def test_per_image_decoded_byte_limit(monkeypatch):
    raw = base64.b64decode(image_url().partition(",")[2])
    monkeypatch.setattr(media, "MAX_IMAGE_BYTES", len(raw) - 1)
    with pytest.raises(media.InvalidMedia, match = "4 MiB"):
        media.prepare("state", [image_url()], accepts_images = True)


def test_aggregate_decoded_byte_limit(monkeypatch):
    raw = base64.b64decode(image_url().partition(",")[2])
    monkeypatch.setattr(media, "MAX_TOTAL_BYTES", len(raw) * 2 - 1)
    with pytest.raises(media.InvalidMedia, match = "8 MiB"):
        media.prepare("state", [image_url(), image_url()], accepts_images = True)


def test_pixel_limit_checked_before_decode(monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 255)
    with pytest.raises(media.InvalidMedia, match = "16 million"):
        media.prepare("state", [image_url()], accepts_images = True)


def test_image_url_part_unknown_fields_are_refused():
    with pytest.raises(media.InvalidMedia, match = "only supports"):
        media.prepare(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(), "detail": "high"}}
                    ],
                }
            ],
            None,
            accepts_images = True,
        )
