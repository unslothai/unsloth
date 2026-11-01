# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/v1/videos decodes each conditioning image once: the readability check's decode feeds the render."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

import routes.video as video_routes
from core.inference import diffusion
from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family
from core.inference.video_minimax_h3 import (
    H3_REF_IMAGE_SOURCE_MAX_PIXELS,
    H3_REF_IMAGE_SOURCE_MAX_SIDE,
    DecodedReferenceImages,
)

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_openai_videos_route import (  # noqa: E402,F401 -- fixtures are found by name
    _create,
    _healthy_diffusers,
    _reference_image_bytes,
    _wait_terminal,
    backend,
    client,
)


def _data_url(
    size,
    *,
    fmt = "PNG",
    mode = "RGB",
    orientation = None,
    color = (40, 90, 160),
) -> str:
    image = Image.new(mode, size, color if mode == "RGB" else color + (128,))
    image.putpixel((0, 0), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 255))
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    buf = io.BytesIO()
    image.save(buf, format = fmt, **kwargs)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _route_decodes(*sources) -> DecodedReferenceImages:
    entries = []
    for source in sources:
        image, size = diffusion.decode_b64_image_sized(
            source,
            mode = "RGB",
            max_side = H3_REF_IMAGE_SOURCE_MAX_SIDE,
            max_pixels = H3_REF_IMAGE_SOURCE_MAX_PIXELS,
        )
        entries.append((source, size, image))
    return DecodedReferenceImages(tuple(entries))


def _count_opens(monkeypatch) -> list:
    opened = []
    real_open = Image.open

    def _open(*args, **kwargs):
        opened.append(1)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", _open)
    return opened


def _pixels(image):
    return None if image is None else (image.mode, image.size, image.tobytes())


@pytest.mark.parametrize("orientation", [None, 3, 6, 8])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_reused_keyframes_match_a_fresh_decode_without_decoding(monkeypatch, orientation, mode):
    h3 = detect_video_family("", override = "minimax-h3")
    first = _data_url((320, 200), mode = mode, orientation = orientation)
    last = _data_url((200, 320), mode = mode, orientation = orientation, color = (5, 6, 7))
    fresh = VideoBackend._resolve_keyframes(h3, None, first, last, None, None)
    decoded = _route_decodes(first, last)
    opened = _count_opens(monkeypatch)
    reused = VideoBackend._resolve_keyframes(
        h3, None, first, last, None, None, decoded_images = decoded.pool()
    )
    assert opened == []
    assert [_pixels(x) for x in reused[:2]] == [_pixels(x) for x in fresh[:2]]
    assert reused[2:] == fresh[2:]


def test_reused_references_match_a_fresh_decode_and_a_repeat_gets_its_own_image(monkeypatch):
    h3 = detect_video_family("", override = "minimax-h3")
    a = _data_url((640, 360), fmt = "JPEG")
    b = _data_url((300, 500), mode = "RGBA", orientation = 6)
    images = [a, b, a]
    fresh = VideoBackend._resolve_references(
        h3, "ref2va", "diffusers", images, None, None, "match", 960, 544
    )
    decoded = _route_decodes(*images)
    opened = _count_opens(monkeypatch)
    reused = VideoBackend._resolve_references(
        h3,
        "ref2va",
        "diffusers",
        images,
        None,
        None,
        "match",
        960,
        544,
        decoded_images = decoded.pool(),
    )
    assert opened == []
    assert [_pixels(x) for x in reused.images] == [_pixels(x) for x in fresh.images]
    # Each slot consumes its own decode, as each was decoded on its own before.
    assert decoded.entries[0][2] is not decoded.entries[2][2]


def test_the_same_image_as_first_and_last_frame_stays_two_images():
    h3 = detect_video_family("", override = "minimax-h3")
    frame = _data_url((512, 512))
    decoded = _route_decodes(frame, frame)
    pool = decoded.pool()
    VideoBackend._resolve_keyframes(h3, None, frame, frame, 512, 512, decoded_images = pool)
    assert pool == []
    assert decoded.entries[0][2] is not decoded.entries[1][2]


@pytest.mark.parametrize("orientation", [None, 6])
def test_a_reused_keyframe_over_the_keyframe_limit_is_refused_with_the_same_message(
    monkeypatch, orientation
):
    """The route's limit is 8192 px; a keyframe's is 4096, judged on the stored (pre-rotation) size."""
    h3 = detect_video_family("", override = "minimax-h3")
    wide = _data_url((4100, 64), orientation = orientation)
    with pytest.raises(ValueError) as fresh:
        VideoBackend._resolve_keyframes(h3, None, wide, None, None, None)
    decoded = _route_decodes(wide)
    opened = _count_opens(monkeypatch)
    with pytest.raises(ValueError) as reused:
        VideoBackend._resolve_keyframes(
            h3, None, wide, None, None, None, decoded_images = decoded.pool()
        )
    assert opened == []
    assert (
        str(reused.value)
        == str(fresh.value)
        == ("Image is too large (4100x64); maximum is 4096px per side.")
    )


def test_an_input_the_route_did_not_decode_is_decoded_as_before(monkeypatch):
    h3 = detect_video_family("", override = "minimax-h3")
    known = _data_url((256, 256))
    other = _data_url((256, 256), color = (1, 2, 3))
    opened = _count_opens(monkeypatch)
    first, *_ = VideoBackend._resolve_keyframes(
        h3, None, other, None, 256, 256, decoded_images = _route_decodes(known).pool()
    )
    assert len(opened) == 2  # the route-side decode of ``known`` plus a fresh decode of ``other``
    assert first.getpixel((10, 10)) == (1, 2, 3)


def test_a_bad_image_after_a_good_one_keeps_the_route_error():
    with pytest.raises(ValueError, match = "Could not decode image"):
        diffusion.decode_b64_image_sized("aGVsbG8=")
    with pytest.raises(ValueError, match = "maximum is 16px per side"):
        diffusion.decode_b64_image_sized(_data_url((32, 8)), max_side = 16)
    with pytest.raises(ValueError, match = "maximum is 100 source pixels"):
        diffusion.decode_b64_image_sized(_data_url((20, 8)), max_pixels = 100)


def test_the_route_hands_its_decodes_to_the_render(client, backend, monkeypatch):
    calls = []
    real = backend.begin_generate

    def _capture(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(backend, "begin_generate", _capture)
    png = _reference_image_bytes("PNG")
    resp = _create(client, {"prompt": "a fox"}, {"input_reference": ("f.png", png, "image/png")})
    assert resp.status_code == 200, resp.json()
    _wait_terminal(client, resp.json()["id"])
    (kwargs,) = calls
    ((source, size, image),) = kwargs["decoded_reference_images"].entries
    assert source is kwargs["input_reference"]
    assert _pixels(image) == _pixels(diffusion.decode_b64_image(source, mode = "RGB"))
    assert size == Image.open(io.BytesIO(png)).size


def test_a_queued_job_keeps_no_decoded_pixels(client, backend):
    from test_openai_videos_queue import _render_one

    png = _reference_image_bytes("PNG")
    _render_one(client, backend)
    try:
        queued = _create(
            client, {"prompt": "next"}, {"input_reference": ("f.png", png, "image/png")}
        )
        assert queued.json()["status"] == "queued"
        (entry,) = [e for e in video_routes._VIDEO_QUEUE if e.id == queued.json()["id"]]
        assert entry.inputs.decoded_images is None
        assert entry.reference is not None
    finally:
        backend.gate.set()
