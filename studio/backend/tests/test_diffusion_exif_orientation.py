# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for EXIF orientation on the image-conditioned decode path.

The dropzone posts the picked file's raw bytes, so a phone photo arrives tagged and un-rotated,
and the inpaint mask is sized from the preview, which is oriented. The two must agree.
"""

from __future__ import annotations

import base64
import io

import pytest

from core.inference.diffusion import decode_b64_image

PIL = pytest.importorskip("PIL.Image")


# One colour per quadrant: a two-band image satisfies orientation 5 and 6 alike, so a mirrored
# fix would pass.
RED, GREEN, BLUE, YELLOW = (220, 20, 20), (20, 200, 20), (20, 20, 220), (220, 200, 20)


def _quadrant_image(width: int, height: int):
    img = PIL.new("RGB", (width, height))
    img.paste(RED, (0, 0, width // 2, height // 2))
    img.paste(GREEN, (width // 2, 0, width, height // 2))
    img.paste(BLUE, (0, height // 2, width // 2, height))
    img.paste(YELLOW, (width // 2, height // 2, width, height))
    return img


def _xmp_packet(orientation: int) -> bytes:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF'
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""'
        ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="%d"/></rdf:RDF>'
        "</x:xmpmeta>" % orientation
    ).encode()


def _phone_photo_data_url(
    width: int = 64,
    height: int = 32,
    orientation: int | None = 6,
    *,
    xmp_orientation: int | None = None,
) -> str:
    """A landscape JPEG tagged for rotated display; ``orientation = None`` writes no EXIF tag."""
    img = _quadrant_image(width, height)
    extra = {}
    if orientation is not None:
        exif = img.getexif()
        exif[0x0112] = orientation
        extra["exif"] = exif
    if xmp_orientation is not None:
        extra["xmp"] = _xmp_packet(xmp_orientation)
    buf = io.BytesIO()
    img.save(buf, format = "JPEG", quality = 95, subsampling = 0, **extra)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _raw_profile_png_data_url(
    orientation: int,
    width: int = 300,
    height: int = 100,
) -> str:
    """A PNG carrying orientation only through ImageMagick's tEXt profile: header, byte count,
    then the EXIF block as wrapped hex."""
    from PIL import PngImagePlugin

    img = _quadrant_image(width, height)
    exif = img.getexif()
    exif[0x0112] = orientation
    blob = exif.tobytes()
    wrapped = "\n".join(blob.hex()[i : i + 72] for i in range(0, len(blob.hex()), 72))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Raw profile type exif", "\nexif\n%8d\n%s\n" % (len(blob), wrapped))
    buf = io.BytesIO()
    img.save(buf, format = "PNG", pnginfo = meta)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def test_inpaint_mask_from_the_preview_matches_the_decoded_source():
    # naturalWidth/naturalHeight are oriented, and a mismatched mask is stretched, not refused.
    stored_w, stored_h = 64, 32
    source = decode_b64_image(_phone_photo_data_url(stored_w, stored_h))
    mask = decode_b64_image(_mask_data_url(stored_h, stored_w), mode = "L")
    assert mask.size == source.size


def test_an_image_without_an_orientation_tag_is_untouched():
    # Quadrants, not a flat fill: a uniform image survives a 180 turn or either mirror intact.
    buf = io.BytesIO()
    _quadrant_image(64, 32).save(buf, format = "PNG")
    img = decode_b64_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert img.size == (64, 32)
    assert _quadrants(img) == ["red", "green", "blue", "yellow"]


def test_an_xmp_only_orientation_is_ignored_the_way_the_browser_ignores_it():
    # getexif() falls back to XMP's tiff:Orientation when EXIF carries none; browsers do not.
    img = decode_b64_image(_phone_photo_data_url(300, 100, None, xmp_orientation = 6))
    assert img.size == (300, 100)
    assert _quadrants(img) == ["red", "green", "blue", "yellow"]


def test_a_real_exif_orientation_outranks_a_disagreeing_xmp_packet():
    # They must DISAGREE, or a decoder letting XMP win would pass; 6 and 8 turn opposite ways.
    img = decode_b64_image(_phone_photo_data_url(300, 100, 6, xmp_orientation = 8))
    assert img.size == (100, 300)
    assert _quadrants(img) == _AS_DISPLAYED[6][1]


def test_an_imagemagick_text_profile_is_ignored_the_way_the_browser_ignores_it():
    img = decode_b64_image(_raw_profile_png_data_url(6))
    assert img.size == (300, 100)
    assert _quadrants(img) == ["red", "green", "blue", "yellow"]


def test_an_xmp_orientation_cached_during_open_is_still_ignored():
    # Resolution tags without Orientation make the opener read the DPI, filling its EXIF cache
    # from the XMP fallback: stripping metadata after open() is too late.
    img = _quadrant_image(300, 100)
    exif = img.getexif()
    exif[0x011A], exif[0x011B] = 72.0, 72.0  # resolution, no Orientation
    buf = io.BytesIO()
    img.save(buf, format = "JPEG", quality = 95, subsampling = 0, exif = exif, xmp = _xmp_packet(6))
    decoded = decode_b64_image(
        "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    )
    assert decoded.size == (300, 100)
    assert _quadrants(decoded) == ["red", "green", "blue", "yellow"]


def test_a_malformed_metadata_block_does_not_refuse_a_decodable_image():
    # The pixels decode and the pre-change decoder accepted this; reading the tag must not 400 it.
    buf = io.BytesIO()
    _quadrant_image(64, 32).save(buf, format = "PNG", exif = b"bad")
    img = decode_b64_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert img.size == (64, 32)
    assert _quadrants(img) == ["red", "green", "blue", "yellow"]


def test_a_real_exif_orientation_still_applies_to_a_png():
    buf = io.BytesIO()
    img = _quadrant_image(300, 100)
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(buf, format = "PNG", exif = exif)
    decoded = decode_b64_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert decoded.size == (100, 300)
    assert _quadrants(decoded) == _AS_DISPLAYED[6][1]


# For a 300x100 photo stored red/green over blue/yellow, read off the tag definitions (2/4 mirror,
# 3 turns 180, 5/7 flip diagonally, 6/8 turn a quarter) and never off Pillow, which would agree
# with a wrong implementation.
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


@pytest.mark.parametrize("orientation", sorted(_AS_DISPLAYED))
def test_every_orientation_decodes_the_way_the_viewer_shows_it(orientation):
    data = _phone_photo_data_url(300, 100, orientation)
    stored = PIL.open(io.BytesIO(base64.b64decode(data.partition(",")[2])))
    assert stored.size == (300, 100)  # the bytes on the wire really are un-rotated

    size, quadrants = _AS_DISPLAYED[orientation]
    img = decode_b64_image(data)
    assert img.size == size
    assert _quadrants(img) == quadrants


def test_no_orientation_survives_the_decode_to_be_applied_twice():
    # Carries one in BOTH an EXIF block and an XMP packet: clearing only the acted-on source fails.
    img = decode_b64_image(_phone_photo_data_url(300, 100, 6, xmp_orientation = 8))
    assert img.size == (100, 300)

    survivors = PIL.Exif()
    if img.info.get("exif"):
        survivors.load(img.info["exif"])
    assert survivors.get(0x0112) is None
    assert "xmp" not in img.info


def test_a_webp_orientation_is_skipped_the_way_chromium_skips_it():
    # Conforming (its EXIF chunk holds the TIFF stream directly), so this pins Chromium's policy.
    buf = io.BytesIO()
    img = _quadrant_image(300, 100)
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(buf, format = "WEBP", exif = exif, lossless = True)
    decoded = decode_b64_image(
        "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    )
    assert decoded.size == (300, 100)
    assert _quadrants(decoded) == ["red", "green", "blue", "yellow"]


def test_an_oversized_image_is_refused_before_its_pixels_are_read(monkeypatch):
    # Both bounds are symmetric, so only refusing to load at all separates the two orders.
    data = _phone_photo_data_url(300, 100, 6)  # built BEFORE the patch: save() loads too
    loads: list[int] = []

    def _no_load(self, *a, **kw):
        loads.append(1)
        raise AssertionError("pixels were read before the size guard refused the image")

    monkeypatch.setattr(PIL.Image, "load", _no_load, raising = True)
    with pytest.raises(ValueError, match = "too large"):
        decode_b64_image(data, max_side = 200)
    assert loads == []
