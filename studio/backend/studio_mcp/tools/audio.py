# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio tools: ``generate_audio`` runs any Audio page workflow and ``transcribe`` turns speech into text."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from studio_mcp import loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward, ndjson_last
from studio_mcp.inputs import AudioInput, audio_bytes, audio_ref, upload_audio
from studio_mcp.media import (
    INLINE_CAP,
    audio_content,
    audio_gallery_path,
    media_result,
    public_url,
    resource_link,
)
from studio_mcp.outputs import AudioClip, AudioResult, TranscriptResult, TranscriptSegment
from studio_mcp.tools import WRITES, opt_text, present, route_json, try_json

LOAD_AUDIO_HINT = "Load a text-to-speech or music model with load_model(kind='tts') first."
NOT_LOADED = "No model loaded"
# 16-bit samples; channels are not reported, so mono is assumed and a stereo clip may still be fetched and then linked.
_WAV_BYTES_PER_SAMPLE = 2


async def _clip_contents(caller: Caller, clip: AudioClip) -> list[Any]:
    """A clip inline when its WAV fits the cap, else a link; one that cannot fit is never fetched."""
    estimate = (clip.duration_s or 0) * (clip.sample_rate or 0) * _WAV_BYTES_PER_SAMPLE
    if clip.duration_s and clip.sample_rate and estimate <= INLINE_CAP:
        response = await forward(caller, "GET", audio_gallery_path(clip.id))
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
    """Run an Audio page workflow with the audio model loaded in Unsloth Studio; clips are saved to Audio history and returned inline when small, else as links. clone: speak ``text`` in the voice of ``reference`` (a clip, or a saved voice as voice_id; ``reference_text`` is its transcript). speak: ``text`` in the loaded model's default voice, or a saved voice via ``reference``. edit: change the words or delivery of ``source`` per ``edit``. convert: make ``source`` sound like ``target``. music: ``mode`` song or sfx from ``text`` (the style) and ``lyrics``; ``variations`` up to 4 share a group_id. separate: split ``source`` into stems. Audio inputs are an Unsloth Studio id, inline base64 with a filename, or a path on the Unsloth Studio computer."""
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
    inputs.update(present(reference_text = reference_text, source_text = source_text))
    body: dict[str, Any] = {"workflow": workflow, "inputs": inputs}
    body.update(
        present(
            text = text,
            language = language,
            instructions = instructions,
            mode = mode,
            lyrics = lyrics,
            duration_s = duration_s,
            edit = edit,
            convert = convert,
            speed = speed,
            seed = seed,
            max_tokens = max_tokens,
            options = options,
        )
    )
    if instrumental:
        body["instrumental"] = True
    if variations != 1:
        body["variations"] = variations
    payload = await route_json(
        "POST",
        "/v1/audio/run",
        caller = caller,
        json_body = body,
        hint_if = (NOT_LOADED, {400: LOAD_AUDIO_HINT}),
    )
    if not isinstance(payload, dict):
        raise ToolError("Unsloth Studio returned no audio")
    clips, contents = [], []
    for row in payload.get("clips") or []:
        if not isinstance(row, dict) or not opt_text(row.get("id")):
            continue
        clip = AudioClip.from_route(row, url = public_url(caller, audio_gallery_path(row["id"])))
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
            raise ToolError("Unsloth Studio returned unreadable audio") from None
        if len(data) > INLINE_CAP:
            raise ToolError(
                f"Unsloth Studio could not save the clip to Audio history, and at "
                f"{len(data) / (1024 * 1024):.1f} MiB it is too large to return inline."
            )
        contents.append(audio_content(data, f"audio/{opt_text(fallback.get('format')) or 'wav'}"))
    if not clips and saved:
        raise ToolError("Unsloth Studio returned no audio")
    return media_result(contents, AudioResult.from_route(payload, clips = clips, saved = saved))


# OpenAI's own upload limit for /audio/transcriptions; past it the tool uploads first.
MULTIPART_LIMIT = 25 * 1024 * 1024
NOT_DOWNLOADED = "is not downloaded"
TIMESTAMPS_HINT = "Pass language with timestamps, or use an audio.cpp speech-to-text model."


def _segments(payload: dict) -> Optional[list[TranscriptSegment]]:
    rows = payload.get("segments")
    if not isinstance(rows, list):
        return None
    return [TranscriptSegment.from_route(row) for row in rows if isinstance(row, dict)]


def _default_stt(status: dict) -> Optional[str]:
    default = status.get("transformers") if isinstance(status.get("transformers"), dict) else {}
    return opt_text(default.get("default_model")) or opt_text(status.get("default_model"))


async def _download_then_retry(
    caller: Caller, ctx: Optional[Context], model: Optional[str]
) -> None:
    """Fetch the STT model a transcription refused as missing, the way load_model does."""
    status = await loading.stt_status(caller, model)
    if model is None:
        model = _default_stt(status)
    if model is None:
        raise ToolError("Unsloth Studio did not name a default speech-to-text model to download")
    engine = loading.stt_engine(status, model)
    await loading.download_stt(caller, ctx, model = model, engine = engine, hf_token = None)


async def _multipart(
    caller: Caller,
    data: bytes,
    name: str,
    *,
    language: Optional[str],
    translate: bool,
    timestamps: bool,
    model: Optional[str],
):
    fields = {"response_format": "verbose_json" if timestamps else "json"}
    if model:
        fields["model"] = model
    if language and not translate:
        fields["language"] = language
    path = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
    return await forward(
        caller, "POST", path, files = {"file": (name, data, "application/octet-stream")}, data = fields
    )


async def _stt_model(caller: Caller, model: Optional[str]) -> tuple[str, str]:
    """The model and engine to transcribe with: the one asked for, else the loaded one, else Unsloth Studio's default."""
    status = await loading.stt_status(caller, model)
    if model is None:
        for engine in loading.STT_ENGINES:
            state = status.get(engine)
            if isinstance(state, dict) and opt_text(state.get("loaded_model")):
                return state["loaded_model"], engine
        model = _default_stt(status)
    if model is None:
        raise ToolError("Unsloth Studio did not name a speech-to-text model; pass model.")
    return model, loading.stt_engine(status, model)


async def _transcribe_source(
    caller: Caller,
    ctx: Optional[Context],
    source: dict[str, str],
    *,
    language: Optional[str],
    timestamps: bool,
    model: Optional[str],
) -> TranscriptResult:
    """Audio Unsloth Studio already holds, transcribed by id; this route saves the transcript to history and streams NDJSON."""
    model, engine = await _stt_model(caller, model)
    body: dict[str, Any] = {
        "source": source,
        "model": model,
        "engine": engine,
        "timestamps": timestamps,
    }
    if language:
        body["language"] = language

    async def send():
        response = await forward(
            caller, "POST", "/api/inference/audio/transcribe/source", json_body = body
        )
        if response.status_code >= 400:
            return response, None
        try:
            return response, ndjson_last(response.content)
        except ValueError:
            raise ToolError("Unsloth Studio returned no transcript") from None

    response, payload = await send()
    # A missing model is refused as a 409 before the stream starts, or as an error line within it.
    if payload is None:
        missing = response.status_code == 409 and NOT_DOWNLOADED in response.text
    else:
        # Only an error line: a transcript can say anything.
        missing = (
            isinstance(payload, dict)
            and payload.get("type") == "error"
            and NOT_DOWNLOADED in str(payload.get("message", ""))
        )
    if missing:
        await _download_then_retry(caller, ctx, model)
        response, payload = await send()
    if payload is None:
        raise_for_route(response)
    raise_for_route(response, payload = payload)
    if not isinstance(payload, dict) or payload.get("type") != "complete":
        raise ToolError("The transcription did not finish")
    return TranscriptResult(
        text = payload.get("text") if isinstance(payload.get("text"), str) else "",
        language = opt_text(payload.get("language")),
        model = model,
        segments = _segments(payload) if timestamps else None,
        saved_to_history = True,
    )


async def transcribe(
    audio: AudioInput,
    language: Optional[str] = None,
    translate: bool = False,
    timestamps: bool = False,
    model: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> TranscriptResult:
    """Transcribe speech with Unsloth Studio's speech-to-text, or translate it to English with ``translate``. ``audio`` is inline base64 with a filename, a path on the Unsloth Studio computer, or an Unsloth Studio id. Audio over 25 MB or given by id is stored with Unsloth Studio first and its transcript is saved to Audio history (``saved_to_history``); translation needs a smaller file. ``timestamps`` adds segments; most engines then need ``language``. A missing model is downloaded first. Without ``model`` the loaded speech-to-text model or Unsloth Studio's default is used."""
    caller = current_caller()
    data, name = audio_bytes(caller, audio) if audio.is_upload else (None, None)
    if data is None or len(data) > MULTIPART_LIMIT:
        if translate:
            raise ToolError("Translation needs a file under 25 MB sent as data or a path.")
        source = (
            {"input_id": await upload_audio(caller, data, name)}
            if data is not None
            else await audio_ref(caller, audio)
        )
        return await _transcribe_source(
            caller, ctx, source, language = language, timestamps = timestamps, model = model
        )

    async def send():
        return await _multipart(
            caller,
            data,
            name,
            language = language,
            translate = translate,
            timestamps = timestamps,
            model = model,
        )

    response = await send()
    if response.status_code == 409 and NOT_DOWNLOADED in response.text:
        await _download_then_retry(caller, ctx, model)
        response = await send()
    hints = {501: TIMESTAMPS_HINT} if timestamps else None
    payload = raise_for_route(response, hints = hints)
    if not isinstance(payload, dict):
        raise ToolError("Unsloth Studio returned no transcript")
    return TranscriptResult(
        text = payload.get("text") if isinstance(payload.get("text"), str) else "",
        # This route answers with the text alone, so report what was asked for and what ran.
        # A translation is always English; its verbose form spells it "english".
        language = "en" if translate else (opt_text(payload.get("language")) or language),
        model = model or await _resident_stt(caller),
        segments = _segments(payload) if timestamps else None,
        saved_to_history = False,
    )


async def _resident_stt(caller: Caller) -> Optional[str]:
    payload = await try_json(caller, "/api/inference/audio/stt/status")
    for state in payload.values() if isinstance(payload, dict) else ():
        if isinstance(state, dict) and opt_text(state.get("loaded_model")):
            return state["loaded_model"]
    return None


TOOLS = ((generate_audio, WRITES, AudioResult.model_json_schema()), (transcribe, WRITES))
