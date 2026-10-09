# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Media an agent hands to a tool: by Studio id, as inline data, or as a file path. A path makes the server read a file on the Studio computer, so it is honoured only when the agent itself runs there (a direct loopback /mcp request); anyone else gets told to send the bytes or an id, and the file is never touched."""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, model_validator

from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward

PATH_REMOTE = (
    "File paths work only when the agent runs on the Studio computer. "
    "Send data_url/data_base64 or a Studio id instead."
)
# The image routes take at most 32 MiB of base64 per image.
MAX_IMAGE_BYTES = 32 * 1024 * 1024 * 3 // 4
IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp")
_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,(.*)$", re.DOTALL)


class ImageInput(BaseModel):
    """Exactly one of ``path`` (a file on the Studio computer), ``data_url`` (data:image/png|jpeg|webp;base64,...) or ``gallery_id`` (an image in the Studio Images gallery)."""

    model_config = ConfigDict(extra = "forbid")

    path: Optional[str] = None
    data_url: Optional[str] = None
    gallery_id: Optional[str] = None

    @model_validator(mode = "after")
    def _exactly_one(self) -> "ImageInput":
        if sum(value is not None for value in (self.path, self.data_url, self.gallery_id)) != 1:
            raise ValueError("Give exactly one of path, data_url or gallery_id.")
        return self


def sniff_image(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def read_local(caller: Caller, path: str, max_bytes: int) -> bytes:
    """A file named by the agent, read only for a caller on this computer and only when within ``max_bytes``."""
    if not caller.direct_local:
        raise ToolError(PATH_REMOTE)
    file = Path(path).expanduser()
    try:
        size = file.stat().st_size
    except OSError:
        raise ToolError(f"Cannot read {path}") from None
    if not file.is_file():
        raise ToolError(f"{path} is not a file")
    if size > max_bytes:
        raise ToolError(f"{path} is larger than {max_bytes // (1024 * 1024)} MiB")
    try:
        return file.read_bytes()
    except OSError:
        raise ToolError(f"Cannot read {path}") from None


def decode_base64(data: str, max_bytes: int, what: str) -> bytes:
    # Checked on the encoded length, before anything is decoded.
    if len(data) > 4 * ((max_bytes + 2) // 3):
        raise ToolError(f"{what} is larger than {max_bytes // (1024 * 1024)} MiB")
    try:
        return base64.b64decode(data, validate = True)
    except (binascii.Error, ValueError):
        raise ToolError(f"{what} is not valid base64") from None


async def resolve_image(
    caller: Caller,
    image: ImageInput,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    mimes: tuple[str, ...] = IMAGE_MIMES,
) -> tuple[bytes, str]:
    """The image's bytes and type."""
    if image.path is not None:
        data = read_local(caller, image.path, max_bytes)
    elif image.data_url is not None:
        match = _DATA_URL.match(image.data_url)
        if match is None:
            raise ToolError("data_url must be data:image/png, image/jpeg or image/webp;base64,...")
        data = decode_base64(match.group(2), max_bytes, "data_url")
    else:
        response = await forward(
            caller,
            "GET",
            f"/api/inference/images/gallery/{quote(image.gallery_id, safe = '')}/file",
        )
        raise_for_route(response)
        data = response.content
        if len(data) > max_bytes:
            raise ToolError(f"Gallery image {image.gallery_id} is too large to send")
    mime = sniff_image(data)
    if mime not in mimes:
        allowed = ", ".join(m.removeprefix("image/").upper() for m in mimes)
        raise ToolError(f"Images must be {allowed}")
    return data, mime


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
