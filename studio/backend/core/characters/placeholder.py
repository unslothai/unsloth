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


# Matches TITLES in features/characters/prompt/speakers.ts.
_TITLES = frozenset(
    "captain capt inspector detective professor prof doctor dr sir lady lord dame mr mrs ms miss mx"
    " king queen prince princess commander general lieutenant lt sergeant sgt officer agent father"
    " sister brother master madam".split()
)


def _initials(name: str) -> str:
    # Matches initials in features/characters/components/portrait-monogram.ts.
    words = name.split()
    start = next((i for i, word in enumerate(words) if word.lower().rstrip(".") not in _TITLES), 0)
    words = words[start:]
    if not words:
        return "?"
    return (words[0][0] + (words[-1][0] if len(words) > 1 else "")).upper()


def placeholder_png(name: str) -> bytes:
    """A 400x600 gradient in the name's hue with its initials, as the app shows it."""
    from PIL import Image, ImageDraw, ImageFont

    size = (400, 600)
    hue = _name_hue(name)
    top = Image.new("RGB", size, _hsl_to_rgb(hue, 0.42, 0.62))
    bottom = Image.new("RGB", size, _hsl_to_rgb((hue + 28) % 360, 0.46, 0.40))
    placeholder = Image.composite(bottom, top, Image.linear_gradient("L").resize(size))
    initials = _initials(name)
    # Pillow's built-in font covers Latin only; other scripts keep the plain gradient, not boxes.
    if initials.isascii() and initials.isalnum():
        try:
            font = ImageFont.load_default(size = 150)
        except (OSError, TypeError):
            font = None
        if font is not None:
            ImageDraw.Draw(placeholder).text(
                (size[0] / 2, size[1] / 2),
                initials,
                font = font,
                fill = "white",
                anchor = "mm",
                stroke_width = 5,
                stroke_fill = "white",
            )
    buffer = io.BytesIO()
    placeholder.save(buffer, format = "PNG")
    return buffer.getvalue()
