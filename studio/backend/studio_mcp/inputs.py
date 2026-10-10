# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Media an agent hands to a tool: by Unsloth Studio id, as inline data, or as a file path. A path makes the server read a file on the Unsloth Studio computer, so it is honoured only when the agent itself runs there (a direct loopback /mcp request); anyone else gets told to send the bytes or an id, and the file is never touched."""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Any, Optional

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, model_validator

from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.media import image_gallery_path

PATH_REMOTE = (
    "File paths work only when the agent runs on the Unsloth Studio computer. "
    "Send data_url/data_base64 or an Unsloth Studio id instead."
)
# The image routes take at most 32 MiB of base64 per image.
MAX_IMAGE_BYTES = 32 * 1024 * 1024 * 3 // 4
IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp")
_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,(.*)$", re.DOTALL)


class ImageInput(BaseModel):
    """Exactly one of ``path`` (a file on the Unsloth Studio computer), ``data_url`` (data:image/png|jpeg|webp;base64,...) or ``gallery_id`` (an image in the Unsloth Studio Images gallery). Inline data rides in the MCP request, which is limited to 4 MiB."""

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
        response = await forward(caller, "GET", image_gallery_path(image.gallery_id))
        raise_for_route(response)
        data = response.content
        if len(data) > max_bytes:
            raise ToolError(f"Gallery image {image.gallery_id} is too large to send")
    mime = sniff_image(data)
    if mime not in mimes:
        allowed = ", ".join(m.removeprefix("image/").upper() for m in mimes)
        raise ToolError(f"Images must be {allowed}")
    return data, mime


async def resolve_images(
    caller: Caller, images: list[ImageInput], *, total_cap: int, too_large: str, **limits: Any
) -> list[str]:
    """Each image as a data URL, refused with ``too_large`` as soon as their bytes together pass ``total_cap``."""
    urls, total = [], 0
    for image in images:
        data, mime = await resolve_image(caller, image, **limits)
        total += len(data)
        if total > total_cap:
            raise ToolError(too_large)
        urls.append(data_url(data, mime))
    return urls


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# The /audio/inputs route's own limit.
MAX_AUDIO_BYTES = 200 * 1024 * 1024
_AUDIO_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class AudioInput(BaseModel):
    """Exactly one of ``path`` (a file on the Unsloth Studio computer), ``data_base64`` with a ``filename``, or an Unsloth Studio id: ``input_id`` (an uploaded clip), ``clip_id`` (an Audio history clip) or ``voice_id`` (a saved voice). Inline data rides in the MCP request, which is limited to 4 MiB."""

    model_config = ConfigDict(extra = "forbid")

    path: Optional[str] = None
    data_base64: Optional[str] = None
    filename: Optional[str] = None
    input_id: Optional[str] = None
    clip_id: Optional[str] = None
    voice_id: Optional[str] = None

    @model_validator(mode = "after")
    def _exactly_one(self) -> "AudioInput":
        sources = (self.path, self.data_base64, self.input_id, self.clip_id, self.voice_id)
        if sum(value is not None for value in sources) != 1:
            raise ValueError(
                "Give exactly one of path, data_base64, input_id, clip_id or voice_id."
            )
        if (self.filename is not None) != (self.data_base64 is not None):
            raise ValueError("filename goes with data_base64, and data_base64 needs one.")
        for value in (self.input_id, self.clip_id, self.voice_id):
            if value is not None and not _AUDIO_ID.match(value):
                raise ValueError("Unsloth Studio audio ids are letters, digits, - and _.")
        return self

    @property
    def is_upload(self) -> bool:
        return self.path is not None or self.data_base64 is not None


def audio_bytes(caller: Caller, audio: AudioInput) -> tuple[bytes, str]:
    """The bytes and a display name of an audio input given as a path or inline data."""
    if audio.path is not None:
        return read_local(caller, audio.path, MAX_AUDIO_BYTES), Path(audio.path).name
    return decode_base64(audio.data_base64, MAX_AUDIO_BYTES, "data_base64"), audio.filename


async def upload_audio(caller: Caller, data: bytes, name: str) -> str:
    """Store audio with Unsloth Studio and return its input id. Raw bytes, so the upload route sees a Content-Length; a re-upload of the same audio answers 200 with the existing id."""
    if len(data) > MAX_AUDIO_BYTES:
        raise ToolError("Audio is larger than 200 MiB")
    response = await forward(
        caller, "POST", "/v1/audio/inputs", params = {"name": name[:255] or "audio"}, content = data
    )
    record = raise_for_route(response)
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        raise ToolError("Unsloth Studio did not store the audio")
    return record["id"]


async def audio_ref(caller: Caller, audio: AudioInput) -> dict[str, str]:
    """The ``{input_id|clip_id|voice_id}`` reference an Unsloth Studio audio route takes, uploading first when needed."""
    if audio.is_upload:
        data, name = audio_bytes(caller, audio)
        return {"input_id": await upload_audio(caller, data, name)}
    for key in ("input_id", "clip_id", "voice_id"):
        value = getattr(audio, key)
        if value is not None:
            return {key: value}
    raise ToolError("No audio was given")
