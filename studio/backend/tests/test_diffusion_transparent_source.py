# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io

import pytest

from core.inference.diffusion import decode_b64_image
from core.inference.diffusion_controlnet import preprocess_control

PIL = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")


def _data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _black_ink_on_transparent():
    img = PIL.new("RGBA", (200, 80), (0, 0, 0, 0))
    img.paste((0, 0, 0, 255), (60, 20, 140, 60))
    return img


def test_a_transparent_source_reaches_the_model_on_white():
    img = decode_b64_image(_data_url(_black_ink_on_transparent()))
    assert img.mode == "RGB"
    assert img.getpixel((5, 5)) == (255, 255, 255)
    assert img.getpixel((100, 40)) == (0, 0, 0)


def test_transparent_line_art_still_gives_a_canny_edge_map():
    sketch = PIL.new("RGBA", (128, 128), (0, 0, 0, 0))
    for x in range(20, 108):
        for y in (30, 31, 96, 97):
            sketch.putpixel((x, y), (0, 0, 0, 255))
    edges = preprocess_control(decode_b64_image(_data_url(sketch)), "canny")
    assert np.asarray(edges).max() == 255


def test_a_paletted_transparent_png_reaches_the_model_on_white():
    img = decode_b64_image(_data_url(_black_ink_on_transparent().quantize(colors = 4)))
    assert img.getpixel((5, 5)) == (255, 255, 255)
    assert img.getpixel((100, 40)) == (0, 0, 0)


def test_rgba_and_mask_decodes_keep_their_alpha_and_values():
    data = _data_url(_black_ink_on_transparent())
    assert decode_b64_image(data, mode = "RGBA").getpixel((5, 5)) == (0, 0, 0, 0)
    assert decode_b64_image(data, mode = "L").getpixel((5, 5)) == 0


def test_an_opaque_rgba_canvas_export_decodes_like_rgb():
    img = PIL.new("RGBA", (64, 48), (10, 200, 30, 255))
    img.paste((200, 0, 120, 255), (16, 12, 48, 36))
    assert decode_b64_image(_data_url(img)).tobytes() == img.convert("RGB").tobytes()
