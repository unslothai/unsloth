# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Unsloth Studio library: ``list_media`` pages through the galleries and ``voices`` manages saved voices."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Annotated, Any, Literal, Optional

from fastmcp.exceptions import ToolError
from pydantic import Field

from studio_mcp.caller import Caller, current_caller
from studio_mcp.inputs import AudioInput, audio_ref
from studio_mcp.media import (
    audio_gallery_path,
    image_gallery_path,
    public_url,
    signed_url,
    video_gallery_path,
    voice_path,
)
from studio_mcp.outputs import (
    AudioRow,
    ImageRow,
    MediaPage,
    TranscriptRow,
    VideoRow,
    VoiceRow,
    VoicesResult,
)
from studio_mcp.tools import DESTRUCTIVE, READ_ONLY, as_dict, number, opt_text, present, route_json

MediaKind = Literal["image", "audio", "video", "transcript", "voice"]
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_OFFSET = re.compile(r"^\d{1,9}$")
_AUDIO_CURSOR = re.compile(r"^(-?[0-9.e+-]*)\|(-?[0-9.e+-]+)\|([A-Za-z0-9_-]{1,128})$")
# The transcripts route's own cursor: the last row's created_at and id.
_TRANSCRIPT_CURSOR = re.compile(r"^[0-9A-Za-z:.+_-]{1,40}\|[A-Za-z0-9_-]{1,59}$")


def _rows(payload: Any, key: str) -> list[dict]:
    rows = as_dict(payload).get(key)
    return (
        [r for r in rows if isinstance(r, dict) and opt_text(r.get("id"))]
        if isinstance(rows, list)
        else []
    )


def _finite(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _cursor(kind: str, cursor: Optional[str]) -> Optional[str]:
    """The kind-specific part of a cursor list_media returned, refused before any call when it belongs to another kind or is malformed."""
    if cursor is None:
        return None
    prefix, _, rest = cursor.partition(":")
    if prefix != kind:
        raise ToolError(
            f"This cursor is not from list_media(kind='{kind}'); pass the next_cursor that call returned."
        )
    if kind in ("image", "video", "voice"):
        malformed = not _OFFSET.fullmatch(rest)
    elif kind == "audio":
        audio = _AUDIO_CURSOR.fullmatch(rest)
        malformed = not audio or not all(_finite(n) for n in (audio[2], audio[1] or "0"))
    else:
        malformed = not _TRANSCRIPT_CURSOR.fullmatch(rest)
    if malformed:
        raise ToolError(
            "The cursor is malformed; pass next_cursor exactly as list_media returned it."
        )
    return rest


async def _with_links(caller: Caller, kind: str, rows: list[dict]) -> list[Optional[str]]:
    return list(await asyncio.gather(*(signed_url(caller, kind, row["id"]) for row in rows)))


async def _images(caller: Caller, limit: int, offset: int) -> tuple[list[Any], Optional[str]]:
    params = {"limit": limit, "offset": offset}
    payload = await route_json("GET", "/api/inference/images/gallery", caller = caller, params = params)
    rows = _rows(payload, "images")
    links = await _with_links(caller, "image", rows)
    items = [
        ImageRow.from_route(
            row, url = public_url(caller, image_gallery_path(row["id"])), download_url = link
        )
        for row, link in zip(rows, links)
    ]
    more = as_dict(payload).get("has_more") is True
    return items, f"image:{offset + len(rows)}" if more and rows else None


async def _videos(caller: Caller, limit: int, offset: int) -> tuple[list[Any], Optional[str]]:
    params = {"limit": limit, "offset": offset}
    payload = await route_json("GET", "/api/inference/video/gallery", caller = caller, params = params)
    rows = _rows(payload, "videos")
    links = await _with_links(caller, "video", rows)
    items = [
        VideoRow.from_route(
            row, url = public_url(caller, video_gallery_path(row["id"])), download_url = link
        )
        for row, link in zip(rows, links)
    ]
    more = as_dict(payload).get("has_more") is True
    return items, f"video:{offset + len(rows)}" if more and rows else None


async def _audio(
    caller: Caller, limit: int, cursor: Optional[str]
) -> tuple[list[Any], Optional[str]]:
    params: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        pin, mtime, clip_id = cursor.split("|")
        params.update(present(before_pin = pin or None, before_mtime = mtime, before_id = clip_id))
    payload = as_dict(
        await route_json("GET", "/api/inference/audio/gallery", caller = caller, params = params)
    )
    rows = _rows(payload, "audio")
    links = await _with_links(caller, "audio", rows)
    items = [
        AudioRow.from_route(
            row, url = public_url(caller, audio_gallery_path(row["id"])), download_url = link
        )
        for row, link in zip(rows, links)
    ]
    mtime, last = number(payload.get("next_before_mtime")), opt_text(payload.get("next_before_id"))
    pin = number(payload.get("next_before_pin"))
    more = (
        payload.get("has_more") is True
        and mtime is not None
        and last is not None
        and _ID.match(last)
    )
    return items, f"audio:{'' if pin is None else repr(pin)}|{mtime!r}|{last}" if more else None


async def _transcripts(
    caller: Caller, limit: int, cursor: Optional[str]
) -> tuple[list[Any], Optional[str]]:
    params = present(limit = limit, before = cursor)
    payload = as_dict(
        await route_json("GET", "/api/inference/audio/transcripts", caller = caller, params = params)
    )
    items = [TranscriptRow.from_route(row) for row in _rows(payload, "transcripts")]
    after = opt_text(payload.get("next_cursor"))
    return items, f"transcript:{after}" if after and _TRANSCRIPT_CURSOR.match(after) else None


def _voice(caller: Caller, row: Any) -> Optional[VoiceRow]:
    row = as_dict(row)
    if not opt_text(row.get("id")) or not opt_text(row.get("name")):
        return None
    return VoiceRow.from_route(row, url = public_url(caller, voice_path(row["id"], "/file")))


async def _voice_list(caller: Caller) -> list[VoiceRow]:
    payload = await route_json("GET", voice_path(), caller = caller)
    rows = as_dict(payload).get("voices")
    voices = [_voice(caller, row) for row in rows] if isinstance(rows, list) else []
    return [voice for voice in voices if voice is not None]


async def list_media(
    kind: Annotated[MediaKind, Field(description = "Which Unsloth Studio library to list.")],
    limit: Annotated[int, Field(ge = 1, le = 50, description = "Items per page.")] = 20,
    cursor: Annotated[
        Optional[str], Field(description = "next_cursor from the previous page of the same kind.")
    ] = None,
) -> MediaPage:
    """List earlier work saved in Unsloth Studio, newest first: images, audio clips, videos, transcripts or saved voices. Each row has its id, which other tools take as gallery_id, clip_id, video_id or voice_id, and for images, audio and videos a download_url that works without the API key for about an hour. Pass next_cursor back as cursor for the next page; image and video pages count rows, so deleting items between calls can shift them by a row."""
    caller = current_caller()
    rest = _cursor(kind, cursor)
    if kind == "image":
        items, after = await _images(caller, limit, int(rest or 0))
        return MediaPage(kind = kind, images = items, next_cursor = after)
    if kind == "video":
        items, after = await _videos(caller, limit, int(rest or 0))
        return MediaPage(kind = kind, videos = items, next_cursor = after)
    if kind == "audio":
        items, after = await _audio(caller, limit, rest)
        return MediaPage(kind = kind, audio = items, next_cursor = after)
    if kind == "transcript":
        items, after = await _transcripts(caller, limit, rest)
        return MediaPage(kind = kind, transcripts = items, next_cursor = after)
    voices, offset = await _voice_list(caller), int(rest or 0)
    after = f"voice:{offset + limit}" if offset + limit < len(voices) else None
    return MediaPage(kind = kind, voices = voices[offset : offset + limit], next_cursor = after)


VOICE_NOT_FOUND = {404: "list_media(kind='voice') lists the saved voices."}


async def voices(
    action: Annotated[
        Literal["list", "save", "rename", "delete"],
        Field(
            description = "list saved voices, save a new one from source, rename (or edit) one, or delete one."
        ),
    ],
    voice_id: Annotated[Optional[str], Field(description = "The voice to rename or delete.")] = None,
    source: Annotated[
        Optional[AudioInput],
        Field(
            description = "For save: the reference audio, a few seconds of clean speech; not a voice_id."
        ),
    ] = None,
    name: Annotated[
        Optional[str], Field(min_length = 1, max_length = 80, description = "The voice's name.")
    ] = None,
    transcript: Annotated[
        Optional[str], Field(max_length = 4000, description = "What the reference audio says.")
    ] = None,
    language: Annotated[
        Optional[str], Field(max_length = 64, description = "The reference's language.")
    ] = None,
) -> VoicesResult:
    """Manage the voices saved in Unsloth Studio, which generate_audio takes as reference={"voice_id": ...}. save stores source under name (with its transcript and language when known); rename changes a voice's name, transcript or language; delete removes it for good."""
    caller = current_caller()
    if action == "list":
        return VoicesResult(action = action, voices = await _voice_list(caller))
    if action == "save":
        if source is None or name is None:
            raise ToolError("save needs source and name.")
        if source.voice_id is not None:
            raise ToolError(
                "A voice is saved from audio or an Audio history clip, not from another saved voice."
            )
        body = {"source": await audio_ref(caller, source), "name": name}
        body.update(present(transcript = transcript, language = language))
        saved = await route_json("POST", voice_path(), caller = caller, json_body = body)
        return VoicesResult(action = action, voice = _voice(caller, saved))
    if voice_id is None or not _ID.match(voice_id):
        raise ToolError(
            f"{action} needs voice_id, an Unsloth Studio voice id from list_media(kind='voice')."
        )
    if action == "rename":
        changes = present(name = name, transcript = transcript, language = language)
        if not changes:
            raise ToolError("rename needs name, transcript or language.")
        edited = await route_json(
            "PATCH", voice_path(voice_id), caller = caller, json_body = changes, hints = VOICE_NOT_FOUND
        )
        return VoicesResult(action = action, voice = _voice(caller, edited))
    await route_json("DELETE", voice_path(voice_id), caller = caller, hints = VOICE_NOT_FOUND)
    return VoicesResult(action = action, removed = True)


TOOLS = (
    (list_media, READ_ONLY, MediaPage.model_json_schema()),
    (voices, DESTRUCTIVE, VoicesResult.model_json_schema()),
)
