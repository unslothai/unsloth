# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Content-addressed flow assets under the account's ``flows/assets``: embedded package files and engine outputs.
``flowasset:<sha>`` and ``sha256:<sha>`` name the same store."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from utils.paths.storage_roots import account_path

_HEX = re.compile(r"^[0-9a-f]{64}$")

MIME_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "application/json": "json",
}
EXT_MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif",
            "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg", "flac": "audio/flac",
            "mp4": "video/mp4", "webm": "video/webm", "json": "application/json"}  # fmt: skip


def assets_dir() -> Path:
    path = account_path("flows/assets")
    path.mkdir(parents = True, exist_ok = True)
    return path


def sniff_mime(data: bytes) -> Optional[str]:
    """The media type from magic bytes; None for anything Flows do not store."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[:4] == b"fLaC":
        return "audio/flac"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video/webm"
    return None


def kind_of(mime: str) -> str:
    return mime.split("/", 1)[0] if mime.split("/", 1)[0] in ("image", "audio", "video") else "file"


def parse_ref(ref: str) -> Optional[str]:
    """The sha256 hex of a ``sha256:``/``flowasset:`` ref, else None."""
    for prefix in ("sha256:", "flowasset:"):
        if ref.startswith(prefix):
            digest = ref[len(prefix) :]
            return digest if _HEX.match(digest) else None
    return None


def path_for(digest: str) -> Optional[Path]:
    if not _HEX.match(digest):
        return None
    base = assets_dir()
    for ext in set(MIME_EXT.values()):
        candidate = base / f"{digest}.{ext}"
        if candidate.is_file():
            return candidate
    return None


def store(data: bytes, mime: Optional[str] = None) -> tuple[str, str]:
    """Write ``data`` once; returns (sha256 hex, mime). The mime is sniffed, never trusted from the caller."""
    sniffed = sniff_mime(data)
    if sniffed is None:
        if mime == "application/json":
            sniffed = mime
        else:
            raise ValueError(
                "Unsupported asset type: only PNG, JPEG, WebP, GIF, WAV, MP3, OGG, FLAC, MP4 and WebM"
            )
    digest = hashlib.sha256(data).hexdigest()
    target = assets_dir() / f"{digest}.{MIME_EXT[sniffed]}"
    if not target.exists():
        fd, tmp = tempfile.mkstemp(dir = target.parent, prefix = ".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return digest, sniffed


def read(digest: str) -> Optional[tuple[bytes, str]]:
    path = path_for(digest)
    if path is None:
        return None
    return path.read_bytes(), EXT_MIME[path.suffix.lstrip(".")]


def url_for(digest: str) -> str:
    return f"/api/flows/assets/sha256:{digest}"


def media_value(digest: str, mime: str, **extra) -> dict:
    value = {"kind": kind_of(mime), "ref": f"sha256:{digest}", "url": url_for(digest), "mime": mime}
    if value["kind"] == "image" and "width" not in extra:
        try:
            from io import BytesIO

            from PIL import Image

            got = read(digest)
            if got is not None:
                with Image.open(BytesIO(got[0])) as im:
                    value["width"], value["height"] = im.size
        except Exception:
            pass
    value.update({k: v for k, v in extra.items() if v is not None})
    return value
