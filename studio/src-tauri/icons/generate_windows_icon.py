# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Rebuild the Windows ICO from the original 1024px branding (Pillow required).

Run: python studio/src-tauri/icons/generate_windows_icon.py
The original icon.png is never rewritten; other platforms retain their artwork.
Each ICO entry contains the corresponding native-size PNG, not an upscaled frame.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
SIZES = (16, 24, 32, 48, 64, 256)
# Original top-edge inset is ~107/1024. An inset of 122 is a modest increase.
CORNER_FRACTION = 122 / 1024


def rounded_mask(size: int) -> Image.Image:
    # Draw at 8x to keep a smooth contour, even for a 16px taskbar icon.
    scale = 8
    canvas = Image.new("L", (size * scale, size * scale), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (0, 0, size * scale - 1, size * scale - 1),
        radius = round(size * CORNER_FRACTION * scale),
        fill = 255,
    )
    mask = canvas.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 32:
        # Lanczos ringing leaves the 16px corner partially opaque otherwise.
        for point in ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)):
            mask.putpixel(point, 0)
    return mask


def frame(source: Image.Image, size: int) -> Image.Image:
    image = source.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 32:
        # Sharpen only the mascot: unsharp-mask processing of the green field
        # produces a dark fringe at the square's outer boundary. This is a
        # native-size detail pass, never a reduction of a smaller frame.
        sharp = image.convert("RGB").filter(
            ImageFilter.UnsharpMask(radius = 0.65, percent = 125, threshold = 2)
        )
        green = source.getpixel((source.width // 2, source.height // 16))[:3]
        pixels = image.load()
        detail = sharp.load()
        for y in range(size):
            for x in range(size):
                r, g, b, a = pixels[x, y]
                if max(abs(r - green[0]), abs(g - green[1]), abs(b - green[2])) > 55:
                    sr, sg, sb = detail[x, y]
                    pixels[x, y] = sr, sg, sb, a
    # Retain source alpha in the interior. Intersection prevents changing the
    # mascot or adding previously transparent pixels outside the rounded tile.
    alpha = ImageChops.multiply(image.getchannel("A"), rounded_mask(size))
    image.putalpha(alpha)
    return image


def build(root: Path = ROOT) -> None:
    source = Image.open(ROOT / "icon.png").convert("RGBA")
    if source.size != (1024, 1024):
        raise ValueError("Windows icon source must be 1024x1024")
    (root / "windows-icon.png").parent.mkdir(parents = True, exist_ok = True)
    frame(source, 1024).save(root / "windows-icon.png", optimize = False)
    payloads = []
    for size in SIZES:
        output = io.BytesIO()
        frame(source, size).save(output, format = "PNG", optimize = False)
        data = output.getvalue()
        (root / f"windows-{size}.png").write_bytes(data)
        payloads.append(data)
    # ICO directory + PNG-compressed images. Windows supports PNG-compressed ICO
    # resources; sizes <= 255 use their exact size, 256 is encoded as zero.
    offset = 6 + 16 * len(SIZES)
    entries = []
    for size, data in zip(SIZES, payloads):
        entries.append(
            struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        )
        offset += len(data)
    (root / "icon.ico").write_bytes(
        struct.pack("<HHH", 0, 1, len(SIZES)) + b"".join(entries) + b"".join(payloads)
    )


if __name__ == "__main__":
    build()
