# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The avatar-less tile as a PNG, so an exported card looks like it does in the app."""

from __future__ import annotations

import colorsys
import io

_HUES = (12, 28, 160, 190, 215, 250, 280, 330)


def _name_hue(name: str) -> int:
    # Matches nameHue in features/characters/images.ts.
    value = 0
    for char in name:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
    return _HUES[value % len(_HUES)]


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h / 360, light, s)
    return round(r * 255), round(g * 255), round(b * 255)


def placeholder_png(name: str) -> bytes:
    """A 400x600 gradient in the name's hue."""
    from PIL import Image

    size = (400, 600)
    hue = _name_hue(name)
    top = Image.new("RGB", size, _hsl_to_rgb(hue, 0.42, 0.62))
    bottom = Image.new("RGB", size, _hsl_to_rgb((hue + 28) % 360, 0.46, 0.40))
    placeholder = Image.composite(bottom, top, Image.linear_gradient("L").resize(size))
    buffer = io.BytesIO()
    placeholder.save(buffer, format = "PNG")
    return buffer.getvalue()
