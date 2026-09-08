# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for EXIF orientation on the image-conditioned decode path.

The dropzone sends the picked file's raw bytes (``readAsDataURL``), so a phone photo arrives
with its EXIF Orientation tag intact and its pixel data un-rotated. ``decode_b64_image`` is the
single decode path for Transform / Edit / Inpaint / ControlNet and the video keyframes, so it
must apply the orientation the browser preview already shows -- as the diffusion trainers do.
"""

from __future__ import annotations

import base64
import io

import pytest

from core.inference.diffusion import decode_b64_image

PIL = pytest.importorskip("PIL.Image")


# One colour per stored quadrant, so a mirror is distinguishable from a rotation: a two-band
# image passes the same assertions under the diagonal flip (orientation 5) as under the
# rotation (orientation 6), which would let a mirrored fix through.
RED, GREEN, BLUE, YELLOW = (220, 20, 20), (20, 200, 20), (20, 20, 220), (220, 200, 20)


def _phone_photo_data_url(
    width: int = 64,
    height: int = 32,
    orientation: int = 6,
) -> str:
    """A JPEG whose stored pixels are landscape, tagged for display rotated."""
    img = PIL.new("RGB", (width, height))
    img.paste(RED, (0, 0, width // 2, height // 2))
    img.paste(GREEN, (width // 2, 0, width, height // 2))
    img.paste(BLUE, (0, height // 2, width // 2, height))
    img.paste(YELLOW, (width // 2, height // 2, width, height))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format = "JPEG", quality = 95, subsampling = 0, exif = exif)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _quadrants(img) -> list:
    w, h = img.size
    names = {RED: "red", GREEN: "green", BLUE: "blue", YELLOW: "yellow"}

    def at(fx, fy):
        px = img.getpixel((int(w * fx), int(h * fy)))
        return names[min(names, key = lambda c: sum((a - b) ** 2 for a, b in zip(c, px)))]

    return [at(0.25, 0.25), at(0.75, 0.25), at(0.25, 0.75), at(0.75, 0.75)]


def _mask_data_url(width: int, height: int) -> str:
    """The inpaint mask as the canvas emits it: PNG, sized from the preview's natural size."""
    buf = io.BytesIO()
    PIL.new("L", (width, height), 255).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_rotated_phone_photo_decodes_the_way_the_browser_shows_it():
    stored = PIL.open(io.BytesIO(base64.b64decode(_phone_photo_data_url().partition(",")[2])))
    assert stored.size == (64, 32)  # the bytes on the wire really are un-rotated

    img = decode_b64_image(_phone_photo_data_url())
    assert img.size == (32, 64)
    # Orientation 6 rotates 90 degrees clockwise for display: the stored bottom-left corner
    # lands top-left. A diagonal mirror would put green there instead of blue.
    assert _quadrants(img) == ["blue", "red", "yellow", "green"]


def test_inpaint_mask_from_the_preview_matches_the_decoded_source():
    # The mask canvas is sized from naturalWidth/naturalHeight, which the browser reports
    # oriented, so an un-rotated source decode leaves the mask transposed against it.
    source = decode_b64_image(_phone_photo_data_url())
    mask = decode_b64_image(_mask_data_url(32, 64), mode = "L")
    assert mask.size == source.size


def test_an_image_without_an_orientation_tag_is_untouched():
    buf = io.BytesIO()
    PIL.new("RGB", (64, 32), (10, 200, 10)).save(buf, format = "PNG")
    img = decode_b64_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert img.size == (64, 32)
    assert img.getpixel((0, 0)) == (10, 200, 10)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_orientation_decides_the_size_guard_the_same_way(orientation):
    # The guard reads the header size BEFORE the transpose swaps the axes, so it must be
    # symmetric under the swap or a portrait photo would be refused at a size its landscape
    # twin is accepted at.
    data = _phone_photo_data_url(300, 100, orientation)
    with pytest.raises(ValueError, match = "too large"):
        decode_b64_image(data, max_side = 200)
    with pytest.raises(ValueError, match = "too large"):
        decode_b64_image(data, max_pixels = 20_000)
    assert decode_b64_image(data, max_side = 4096).size in {(300, 100), (100, 300)}
