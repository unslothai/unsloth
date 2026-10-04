#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render the macOS DMG install-window background.

Writes studio/src-tauri/dmg/background.tiff, the image Finder draws behind the
app icon and the Applications alias when the disk image opens.

The layout is tied to the Finder coordinates in studio/src-tauri/tauri.macos.conf.json.
Finder maps icon positions onto the background from the same origin, so the base
page has to match dmg.windowSize exactly or the artwork slides out from under the
icons. tests/studio/test_tauri_branding_contract.py holds that contract.

Output is a two-page TIFF: a base page at window size and a second page at twice
that, which is how Finder picks up a sharp image on a retina display.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "studio/src-tauri/dmg/background.tiff"

# window geometry in points, mirroring tauri.macos.conf.json
WIN_W, WIN_H = 660, 400
APP_X, APP_Y = 180, 170
APPS_X, APPS_Y = 480, 170

# the base page renders at 1pt per pixel, the hidpi page at SCALE
SCALE = 2
W, H = WIN_W * SCALE, WIN_H * SCALE

# Finder's title bar and bottom path bar eat the rest of the window, so only about this much of the image is ever on
# screen. measured, not derived.
VISIBLE_H = 340

TOP_COLOR = "#FFFFFF"
BOTTOM_COLOR = "#F5F8F7"

# One brand green fading to a lighter green further out.
GLOW_CORE = "#17B88B"
GLOW_EDGE = "#7BE8A6"
GLOW_EDGE_MIX = 0.45
GLOW_MIX_RADIUS = 110.0

# strength is the peak tint, sigma sets how far the halo reaches.
GLOW_STRENGTH = 1.48
GLOW_SIGMA = 50.0

# the icon label sits just below the icon, so the halo is eased off down there.
# the taper waits until START, which is inside the icon's own lower half, so the
# disc still reads round and the transition is hidden behind the artwork rather
# than eating the visible bottom edge. left, right and top keep the full falloff.
GLOW_BOTTOM_FLOOR = 0.25
GLOW_BOTTOM_START = 50.0
GLOW_BOTTOM_SPAN = 30.0

# chevron between the two icons, sized to match the macOS installers this mirrors:
# a light 16x27pt mark in neutral grey, not a heavy arrow.
CHEVRON_HALF_W, CHEVRON_HALF_H = 8.0, 13.5
CHEVRON_STROKE = 6.0
CHEVRON_COLOR = (87, 87, 87, 255)

# ImageDraw has no anti-aliasing, so the chevron is drawn oversized and scaled back down.
CHEVRON_SUPERSAMPLE = 4


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype = np.float32)


def smoothstep(edge: np.ndarray) -> np.ndarray:
    clamped = np.clip(edge, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def base_canvas() -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, H, dtype = np.float32)[:, None, None]
    return hex_rgb(TOP_COLOR) * (1.0 - ramp) + hex_rgb(BOTTOM_COLOR) * ramp


def render_glow(canvas: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = APP_X * SCALE, APP_Y * SCALE
    sigma = GLOW_SIGMA * SCALE

    radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    weight = GLOW_STRENGTH * np.exp(-(radius**2) / (2.0 * sigma**2))

    # smoothstep, so the taper eases in and out with no seam at either end
    taper = smoothstep((ys - cy - GLOW_BOTTOM_START * SCALE) / (GLOW_BOTTOM_SPAN * SCALE))
    weight = weight * (1.0 - (1.0 - GLOW_BOTTOM_FLOOR) * taper)

    # a peak strength above 1 saturates the core rather than extrapolating past
    # the glow colour. that core sits under the app icon either way.
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    mix = (GLOW_EDGE_MIX * smoothstep(radius / (GLOW_MIX_RADIUS * SCALE)))[..., None]
    color = hex_rgb(GLOW_CORE) * (1.0 - mix) + hex_rgb(GLOW_EDGE) * mix

    return canvas * (1.0 - weight) + color * weight


def draw_chevron(image: Image.Image) -> None:
    """Rounded chevron pointing from the app icon toward Applications."""
    ss = CHEVRON_SUPERSAMPLE
    layer = Image.new("RGBA", (image.width * ss, image.height * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cx = ((APP_X + APPS_X) / 2) * SCALE * ss
    cy = APP_Y * SCALE * ss
    half_w, half_h = CHEVRON_HALF_W * SCALE * ss, CHEVRON_HALF_H * SCALE * ss
    stroke = CHEVRON_STROKE * SCALE * ss

    points = [(cx - half_w, cy - half_h), (cx + half_w, cy), (cx - half_w, cy + half_h)]
    draw.line(points, fill = CHEVRON_COLOR, width = int(stroke), joint = "curve")
    for x, y in points:
        draw.ellipse(
            [x - stroke / 2, y - stroke / 2, x + stroke / 2, y + stroke / 2],
            fill = CHEVRON_COLOR,
        )

    image.alpha_composite(layer.resize(image.size, Image.LANCZOS))


def build() -> Image.Image:
    canvas = render_glow(base_canvas())
    pixels = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGBA")
    draw_chevron(image)
    return image


def report() -> None:
    """Print how the halo sits relative to the icon and its label."""
    base = base_canvas()
    # the halo on its own, with the base gradient taken back out so the vertical ramp does not read as part of it
    halo = np.clip(base - render_glow(base), 0.0, None).max(axis = 2)
    cx, cy = APP_X * SCALE, APP_Y * SCALE

    # 60pt is the disc a viewer reads as round, 90pt is out in the label's row
    for distance in (60, 90):
        for name, sample in (
            ("left", halo[cy, cx - distance * SCALE]),
            ("right", halo[cy, cx + distance * SCALE]),
            ("up", halo[cy - distance * SCALE, cx]),
            ("down", halo[cy + distance * SCALE, cx]),
        ):
            print(f"  {name:<5} at {distance}pt   {sample * 100:5.1f}%")

    visible = np.nonzero(halo[cy, :cx] > 0.03)[0]
    print(f"  reach              {(cx - visible.min()) / SCALE:5.0f}pt")

    label = halo[238 * SCALE : 260 * SCALE, 140 * SCALE : 220 * SCALE]
    print(f"  icon label         {label.mean() * 100:5.1f}% mean, {label.max() * 100:5.1f}% peak")


def write_tiff(image: Image.Image, destination: Path) -> None:
    """Combine a base and a hidpi page into the multi-page TIFF Finder expects."""
    with tempfile.TemporaryDirectory() as workspace:
        base_page = Path(workspace) / "base.png"
        hidpi_page = Path(workspace) / "hidpi.png"

        image.resize((WIN_W, WIN_H), Image.LANCZOS).convert("RGB").save(base_page)
        image.convert("RGB").save(hidpi_page)

        destination.parent.mkdir(parents = True, exist_ok = True)
        subprocess.run(
            [
                "tiffutil",
                "-cathidpicheck",
                str(base_page),
                str(hidpi_page),
                "-out",
                str(destination),
            ],
            check = True,
            stdout = subprocess.DEVNULL,
        )


def main() -> None:
    write_tiff(build(), OUTPUT)
    report()
    print(f"wrote {OUTPUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
