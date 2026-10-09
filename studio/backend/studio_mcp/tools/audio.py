# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio tools: ``generate_audio`` runs any Audio page workflow."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.inputs import AudioInput, audio_ref
from studio_mcp.media import INLINE_CAP, audio_content, media_result, public_url, resource_link
from studio_mcp.outputs import AudioClip, AudioResult
from studio_mcp.tools import WRITES, integer, number
from studio_mcp.tools import text as route_text

LOAD_AUDIO_HINT = "Load a text-to-speech or music model with load_model(kind='llm') first."
NOT_LOADED = "No model loaded"
# 16-bit samples; channels are not reported, so mono is assumed and a stereo clip may still be fetched and then linked.
_WAV_BYTES_PER_SAMPLE = 2


def _gallery_path(clip_id: str) -> str:
    return f"/v1/audio/gallery/{quote(clip_id, safe = '')}/file"


async def _clip_contents(caller: Caller, clip: AudioClip) -> list[Any]:
    """A clip inline when its WAV fits the cap, else a link; one that cannot fit is never fetched."""
    estimate = (clip.duration_s or 0) * (clip.sample_rate or 0) * _WAV_BYTES_PER_SAMPLE
    if clip.duration_s and clip.sample_rate and estimate <= INLINE_CAP:
        response = await forward(caller, "GET", _gallery_path(clip.id))
        if response.status_code == 200 and len(response.content) <= INLINE_CAP:
            mime = response.headers.get("content-type", "audio/wav").split(";")[0]
            return [audio_content(response.content, mime)]
    return [resource_link(clip.url, f"{clip.id}.wav", "audio/wav")]


async def generate_audio(
    workflow: Literal["clone", "speak", "edit", "convert", "music", "separate"],
    text: Optional[str] = None,
    language: Optional[str] = None,
    instructions: Optional[str] = None,
    reference: Optional[AudioInput] = None,
    source: Optional[AudioInput] = None,
    emotion: Optional[AudioInput] = None,
    target: Optional[AudioInput] = None,
    reference_text: Optional[str] = None,
    source_text: Optional[str] = None,
    mode: Optional[Literal["song", "sfx", "edit"]] = None,
    lyrics: Optional[str] = None,
    instrumental: bool = False,
    duration_s: Optional[float] = None,
    variations: int = 1,
    edit: Optional[dict[str, Any]] = None,
    convert: Optional[dict[str, Any]] = None,
    speed: Optional[float] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    options: Optional[dict[str, Any]] = None,
) -> ToolResult:
    """Run an Audio page workflow with the audio model loaded in Studio; clips are saved to Audio history and returned inline when small, else as links. clone: speak ``text`` in the voice of ``reference`` (a clip, or a saved voice as voice_id; ``reference_text`` is its transcript). speak: ``text`` in the loaded model's voice, or a saved voice via ``reference``. edit: change the words or delivery of ``source`` per ``edit``. convert: make ``source`` sound like ``target``. music: ``mode`` song or sfx from ``text`` (the style) and ``lyrics``; ``variations`` up to 4 share a group_id. separate: split ``source`` into stems. Audio inputs are a Studio id, inline base64 with a filename, or a path on the Studio computer."""
    caller = current_caller()
    inputs: dict[str, Any] = {}
    for key, audio in (
        ("reference", reference),
        ("source", source),
        ("emotion", emotion),
        ("target", target),
    ):
        if audio is not None:
            inputs[key] = await audio_ref(caller, audio)
    if reference_text is not None:
        inputs["reference_text"] = reference_text
    if source_text is not None:
        inputs["source_text"] = source_text
    body: dict[str, Any] = {"workflow": workflow, "inputs": inputs}
    for key, value in (
        ("text", text),
        ("language", language),
        ("instructions", instructions),
        ("mode", mode),
        ("lyrics", lyrics),
        ("duration_s", duration_s),
        ("edit", edit),
        ("convert", convert),
        ("speed", speed),
        ("seed", seed),
        ("max_tokens", max_tokens),
        ("options", options),
    ):
        if value is not None:
            body[key] = value
    if instrumental:
        body["instrumental"] = True
    if variations != 1:
        body["variations"] = variations
    response = await forward(caller, "POST", "/v1/audio/run", json_body = body)
    hints = {400: LOAD_AUDIO_HINT} if NOT_LOADED in response.text else None
    payload = raise_for_route(response, hints = hints)
    if not isinstance(payload, dict):
        raise ToolError("Studio returned no audio")
    clips, contents = [], []
    for row in payload.get("clips") or []:
        if not isinstance(row, dict) or not route_text(row.get("id")):
            continue
        clip = AudioClip(
            id = row["id"],
            role = route_text(row.get("role")) or "output",
            url = public_url(caller, _gallery_path(row["id"])),
            duration_s = number(row.get("duration_s")),
            sample_rate = integer(row.get("sample_rate")),
        )
        clips.append(clip)
        contents.extend(await _clip_contents(caller, clip))
    saved = True
    fallback = payload.get("audio")
    if isinstance(fallback, dict) and isinstance(fallback.get("data"), str):
        # History could not save the clip, so the route sent it inline instead.
        saved = False
        try:
            data = base64.b64decode(fallback["data"], validate = True)
        except (binascii.Error, ValueError):
            raise ToolError("Studio returned unreadable audio") from None
        contents.append(audio_content(data, f"audio/{route_text(fallback.get('format')) or 'wav'}"))
    if not clips and saved:
        raise ToolError("Studio returned no audio")
    result = AudioResult(
        model = route_text(payload.get("model")),
        group_id = route_text(payload.get("group_id")),
        clips = clips,
        saved = saved,
    )
    return media_result(contents, result)


def register_audio(mcp: FastMCP) -> None:
    mcp.tool(generate_audio, annotations = WRITES, output_schema = AudioResult.model_json_schema())
