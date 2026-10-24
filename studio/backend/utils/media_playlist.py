# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse uploads that FFmpeg would probe as a playlist (HLS, concat, DASH), which open the files they name."""

from __future__ import annotations

import io
import os
from typing import Any

_HEAD_BYTES = 1 << 20  # FFmpeg grows its probe buffer past a few KB
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
    """Raise ``PlaylistRefused`` when ``source`` looks like an HLS, concat or DASH playlist."""
    raw = _head(source)
    head = raw.lstrip(b"\xef\xbb\xbf \t\r\n")
    # Mirrors FFmpeg's dash_probe: case-insensitive "<MPD" plus a DASH profile, up to the first NUL.
    text = raw.split(b"\x00", 1)[0].lower()
    dash = b"<mpd" in text and (b"dash:profile" in text or b"3gpp:pss:profile:dash1" in text)
    if head.startswith(_PLAYLIST_MARKS) or dash:
        raise PlaylistRefused("Playlists are not accepted as media uploads.")
