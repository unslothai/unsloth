# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Decode shipped ICO frames rather than trusting preview resizes or file names."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

from PIL import Image, ImageChops

ICONS = Path(__file__).resolve().parents[1] / "icons"
SIZES = (16, 24, 32, 48, 64, 256)


def test_native_windows_icon_frames(tmp_path):
    ico_bytes = (ICONS / "icon.ico").read_bytes()
    assert struct.unpack_from("<HHH", ico_bytes) == (0, 1, len(SIZES))
    with Image.open(ICONS / "icon.ico") as ico:
        assert ico.ico.sizes() == {(s, s) for s in SIZES}
        for index, size in enumerate(SIZES):
            frame = ico.ico.getimage((size, size)).convert("RGBA")
            alpha = frame.getchannel("A")
            if size <= 32:
                corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
                assert all(alpha.getpixel(point) == 0 for point in corners), (
                    f"{size}px Windows icon must have genuinely transparent rounded corners"
                )
                assert sum(pixel == 0 for pixel in alpha.get_flattened_data()) >= 4
                # The black hair and white face/outline are identifiable in
                # every small frame, rather than collapsing to green blur.
                center = frame.crop((size // 4, size // 4, 3 * size // 4, 3 * size // 4))
                assert sum(max(r, g, b) < 70 and a > 220 for r, g, b, a in center.get_flattened_data()) >= 3
                assert sum(min(r, g, b) > 205 and a > 220 for r, g, b, a in center.get_flattened_data()) >= 3
            w, h, _, _, planes, bits, length, offset = struct.unpack_from("<BBBBHHII", ico_bytes, 6 + 16 * index)
            assert (w or 256, h or 256, planes, bits) == (size, size, 1, 32)
            # Direct native-resolution frame, not a scaled-up screenshot or a
            # Pillow-generated resampling of another frame on ICO export.
            assert ico_bytes[offset:offset + length] == (ICONS / f"windows-{size}.png").read_bytes()

    spec = importlib.util.spec_from_file_location("generate_windows_icon", ICONS / "generate_windows_icon.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(tmp_path)
    for name in ("icon.ico", "windows-icon.png", *(f"windows-{s}.png" for s in SIZES)):
        assert (tmp_path / name).read_bytes() == (ICONS / name).read_bytes(), f"stale generated asset: {name}"

    with Image.open(ICONS / "windows-icon.png") as highres, Image.open(ICONS / "icon.png") as original:
        # The mascot itself is untouched; only tile corners and small-frame
        # rasterization change. The other platforms retain original icon.png.
        region = (220, 180, 805, 850)
        assert ImageChops.difference(highres.crop(region), original.crop(region)).getbbox() is None


def test_windows_app_and_installer_use_the_same_ico():
    config = json.loads((ICONS.parent / "tauri.conf.json").read_text())
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert config["bundle"]["windows"]["nsis"]["installerIcon"] == "./icons/icon.ico"
