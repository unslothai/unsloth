# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
from PIL import Image, ImageDraw

from core.training.diffusion_dit_trainer import _load_pixel_tensor, _load_pixel_tensor_planned
from core.training.diffusion_lora_trainer import _load_image_tensor, _load_image_tensor_planned

SIZE = 64
YELLOW = (255, 220, 0)


def _draw_sticker(draw, outline, fill):
    draw.ellipse((8, 8, SIZE - 9, SIZE - 9), fill = outline)
    draw.ellipse((14, 14, SIZE - 15, SIZE - 15), fill = fill)


def _write_sticker(path, mode):
    # Exporters commonly store fully transparent pixels as (0, 0, 0, 0).
    if mode == "P":
        img = Image.new("P", (SIZE, SIZE), 0)
        img.putpalette([0, 0, 0, *YELLOW, 0, 0, 0])
        _draw_sticker(ImageDraw.Draw(img), 2, 1)
        img.save(path, transparency = 0)
        return
    if mode == "LA":
        img = Image.new("LA", (SIZE, SIZE), (0, 0))
        _draw_sticker(ImageDraw.Draw(img), (0, 255), (200, 255))
    else:
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        _draw_sticker(ImageDraw.Draw(img), (0, 0, 0, 255), (*YELLOW, 255))
    img.save(path, lossless = True)


def _loaders():
    rng = random.Random(0)
    return {
        "sdxl": lambda p: _load_image_tensor(str(p), SIZE, True, False, rng)[0],
        "sdxl_cached": lambda p: _load_image_tensor_planned(str(p), SIZE, True, 0.0, 0.0, False)[0],
        "dit": lambda p: _load_pixel_tensor(str(p), SIZE, True, False, rng),
        "dit_cached": lambda p: _load_pixel_tensor_planned(str(p), SIZE, True, 0.0, 0.0, False),
    }


@pytest.mark.parametrize("loader", ["sdxl", "sdxl_cached", "dit", "dit_cached"])
@pytest.mark.parametrize(
    "name, mode",
    [("rgba.png", "RGBA"), ("rgba.webp", "RGBA"), ("la.png", "LA"), ("palette.png", "P")],
)
def test_transparent_training_image_trains_on_white_with_its_outline(tmp_path, loader, name, mode):
    path = tmp_path / name
    _write_sticker(path, mode)

    tensor = _loaders()[loader](path)
    pixel = lambda x, y: [round(v) for v in ((tensor[:, y, x] + 1.0) * 127.5).tolist()]

    assert pixel(2, 2) == [255, 255, 255]
    assert pixel(SIZE // 2, 10) == [0, 0, 0]
    assert pixel(SIZE // 2, SIZE // 2) != [255, 255, 255]


@pytest.mark.parametrize("loader", ["sdxl", "dit"])
def test_opaque_training_image_is_unchanged(tmp_path, loader):
    path = tmp_path / "opaque.png"
    Image.new("RGB", (SIZE, SIZE), (10, 20, 30)).save(path)

    tensor = _loaders()[loader](path)

    assert [round(v) for v in ((tensor[:, 5, 5] + 1.0) * 127.5).tolist()] == [10, 20, 30]
