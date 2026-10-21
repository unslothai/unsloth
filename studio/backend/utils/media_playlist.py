# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse uploads that FFmpeg would probe as a playlist (HLS, concat), which open the files they name."""

from __future__ import annotations

import io
import os
from typing import Any

_HEAD_BYTES = 4096
_PLAYLIST_MARKS = (b"#EXTM3U", b"ffconcat")


class PlaylistRefused(ValueError):
    """The media is a playlist that would read other files on the server."""


def _head(source: Any) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source[:_HEAD_BYTES])
    if isinstance(source, (str, os.PathLike)):
        try:
            with open(source, "rb") as handle:
                return handle.read(_HEAD_BYTES)
        except OSError:
            return b""
    if isinstance(source, io.IOBase) and source.seekable():
        at = source.tell()
        try:
            return source.read(_HEAD_BYTES) or b""
        finally:
            source.seek(at)
    return b""


def refuse_playlist(source: Any) -> None:
    """Raise ``PlaylistRefused`` when ``source`` starts like an HLS or concat playlist."""
    head = _head(source).lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(_PLAYLIST_MARKS):
        raise PlaylistRefused("Playlists are not accepted as media uploads.")
