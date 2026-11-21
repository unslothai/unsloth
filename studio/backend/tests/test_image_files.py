# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""utils/image_files: type sniffing and the size-capped request body."""

import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from utils import device_sync_settings
from utils.image_files import read_limited_body, sniff_image_type


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format = "PNG")
    return buffer.getvalue()


def test_image_helpers():
    assert sniff_image_type(_png()) == "image/png"
    assert sniff_image_type(b"GIF89a....") == "image/gif"
    assert sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_type(b"nope") is None
    with pytest.raises(ValueError):
        device_sync_settings.save_background_image(b"GIF89a" + b"\x00" * 20)

    class _Request:
        headers = {}

        async def stream(self):
            yield b"x" * 6
            yield b"x" * 6

    with pytest.raises(HTTPException) as refused:
        asyncio.run(read_limited_body(_Request(), 10, "too big"))
    assert refused.value.status_code == 413
    assert asyncio.run(read_limited_body(_Request(), 12, "too big")) == b"x" * 12
