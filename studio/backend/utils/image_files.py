# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small helpers for user-uploaded image files: type sniffing, atomic writes, size-capped bodies."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import Request

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_image_type(data: bytes) -> Optional[str]:
    """The type the bytes say, never the client's claim; callers narrow it to what they accept."""
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def atomic_write_bytes(
    path: Path,
    data: bytes,
    prefix: str = ".upload-",
) -> None:
    fd, tmp = tempfile.mkstemp(dir = path.parent, prefix = prefix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise


async def read_limited_body(request: "Request", limit: int, detail: str) -> bytes:
    """The raw body, refused with 413 as soon as it passes *limit* (declared or streamed)."""
    from fastapi import HTTPException

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code = 413, detail = detail)
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code = 413, detail = detail)
    return bytes(data)
