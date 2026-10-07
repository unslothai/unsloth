# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded Decisions image extension. No URL fetches, video decoding, or audio fallback."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_BODY_BYTES = 13 * 1024 * 1024
_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
_UNSUPPORTED = {"audio", "input_audio", "audio_url", "video", "input_video", "video_url"}


class InvalidMedia(ValueError):
    def __init__(
        self,
        message: str,
        *,
        unsupported: bool = False,
    ):
        super().__init__(message)
        self.unsupported = unsupported


def _image(data_url: Any) -> bytes:
    if not isinstance(data_url, str):
        raise InvalidMedia("Images must be PNG, JPEG or WebP base64 data URLs.")
    header, separator, encoded = data_url.partition(",")
    mime = header.removeprefix("data:").removesuffix(";base64")
    if not separator or mime not in _FORMATS or header != f"data:{mime};base64":
        raise InvalidMedia(
            "Images must be PNG, JPEG or WebP base64 data URLs; remote URLs are not supported."
        )
    if not encoded or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise InvalidMedia("Each image must contain at most 4 MiB of decoded data.")
    try:
        raw = base64.b64decode(encoded, validate = True)
    except (binascii.Error, ValueError):
        raise InvalidMedia("Image base64 is malformed.") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise InvalidMedia("Each image must contain at most 4 MiB of decoded data.")
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != _FORMATS[mime]:
                raise InvalidMedia("Image content does not match its declared MIME type.")
            if (
                image.width <= 0
                or image.height <= 0
                or image.width * image.height > MAX_IMAGE_PIXELS
            ):
                raise InvalidMedia("Each image must contain at most 16 million pixels.")
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidMedia("Animated images and video are not supported.", unsupported = True)
            image.verify()
        # verify checks containers, not every pixel stream (notably JPEG/WebP).
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
    except InvalidMedia:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise InvalidMedia("Image data is malformed or exceeds the pixel limit.") from None
    return raw


def prepare(
    state: Any, images: list[str] | None, *, accepts_images: bool
) -> tuple[Any, list[bytes]]:
    """Extract message image parts; ordinary structured JSON is not a media upload."""
    urls: list[Any] = list(images or [])
    cleaned = state
    if isinstance(state, list):
        cleaned = []
        for message in state:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or not isinstance(message.get("content"), list)
            ):
                cleaned.append(message)
                continue
            content = []
            for part in message["content"]:
                kind = part.get("type") if isinstance(part, dict) else None
                if not isinstance(kind, (str, type(None))):
                    raise InvalidMedia("Message content type must be text or image_url.")
                if kind in _UNSUPPORTED:
                    raise InvalidMedia(
                        "The Decision API does not serve video or audio.", unsupported = True
                    )
                if kind not in (None, "text", "image_url"):
                    raise InvalidMedia(
                        f"Unsupported message content type: {kind}", unsupported = True
                    )
                if kind == "image_url":
                    value = part.get("image_url")
                    if isinstance(value, dict):
                        if set(value) - {"url"}:
                            raise InvalidMedia(
                                "An image_url part only supports its data URL (url)."
                            )
                        value = value.get("url")
                    urls.append(value)
                else:
                    content.append(part)
            cleaned.append({**message, "content": content})
    if urls and not accepts_images:
        raise InvalidMedia(
            "This Decision API backend is text-only; select a local Clef model for images.",
            unsupported = True,
        )
    if len(urls) > MAX_IMAGES:
        raise InvalidMedia(
            "At most 4 images are supported per request, including image_url parts in state."
        )
    decoded: list[bytes] = []
    total = 0
    for url in urls:
        raw = _image(url)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise InvalidMedia("Images may contain at most 8 MiB of decoded data in total.")
        decoded.append(raw)
    return cleaned, decoded
