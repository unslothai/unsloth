# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio tools: ``generate_audio`` runs any Audio page workflow, ``transcribe`` turns speech into text and ``analyze_audio`` measures a clip."""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_audio, raise_for_route
from studio_mcp.forward import forward, ndjson_last
from studio_mcp.inputs import AudioInput, audio_bytes, audio_ref, upload_audio
from studio_mcp.media import (
    INLINE_CAP,
    ReturnMedia,
    audio_content,
    audio_gallery_path,
    media_result,
    public_url,
    resource_link,
    signed_url,
)
from studio_mcp.outputs import (
    AnalysisResult,
    AudioClip,
    AudioResult,
    TranscriptResult,
    TranscriptSegment,
)
from studio_mcp.progress import with_heartbeat
from studio_mcp.tools import WRITES, opt_text, present, route_json, strings, try_json

LOAD_AUDIO_HINT = "Load a text-to-speech or music model with load_model(kind='tts') first."
NOT_LOADED = "No model loaded"
# 16-bit samples; channels are not reported, so mono is assumed and a stereo clip may still be fetched and then linked.
_WAV_BYTES_PER_SAMPLE = 2


async def _clip_contents(
    caller: Caller,
    clip: AudioClip,
    return_media: ReturnMedia = "link",
) -> list[Any]:
    """A link, or with "inline" the clip itself when its WAV fits the cap; one that cannot fit is never fetched. Audio has no preview, so "preview" is a link."""
    estimate = (clip.duration_s or 0) * (clip.sample_rate or 0) * _WAV_BYTES_PER_SAMPLE
    fits = clip.duration_s and clip.sample_rate and estimate <= INLINE_CAP
    if return_media == "inline" and fits:
        response = await forward(caller, "GET", audio_gallery_path(clip.id))
        if response.status_code == 200 and len(response.content) <= INLINE_CAP:
            mime = response.headers.get("content-type", "audio/wav").split(";")[0]
            return [audio_content(response.content, mime)]
    return [resource_link(clip.download_url or clip.url, f"{clip.id}.wav", "audio/wav")]


async def _checked_options(
    caller: Caller,
    workflow: str,
    options: dict[str, Any],
    model: Optional[str] = None,
) -> dict[str, Any]:
    """``options`` with each enum value spelled as the loaded model lists it; the route drops a value it does not list, so that is refused here instead."""
    status = await try_json(caller, "/api/inference/status")
    if not isinstance(status, dict):
        return options
    if model and model.split(":", 1)[0].lower() not in {
        str(status.get(key) or "").lower() for key in ("model_identifier", "active_model")
    }:
        # Another model will run it, so the loaded model's choices say nothing about these.
        return options
    by_workflow = status.get("audio_options_by_workflow")
    schema = (by_workflow.get(workflow) if isinstance(by_workflow, dict) else None) or status.get(
        "audio_options"
    )
    checked = dict(options)
    for option in schema if isinstance(schema, list) else []:
        if not isinstance(option, dict) or option.get("type") != "enum":
            continue
        name, values = option.get("name"), strings(option.get("values"))
        if not isinstance(name, str) or checked.get(name) is None or not values:
            continue
        value = str(checked[name])
        match = (
            value
            if value in values
            else next((v for v in values if v.casefold() == value.casefold()), None)
        )
        if match is None:
            raise ToolError(f"options.{name} '{value}' is not one of: {', '.join(values)}.")
        checked[name] = match
    return checked


@dataclass
class _Run:
    cancelled: bool = False
    in_flight: bool = False


# generate_audio runs by (account, cancel_id), and ids cancelled lately; the route keeps an early
# cancel as long, so a reused id would be stopped on arrival.
_RUNS: dict[tuple[str, str], _Run] = {}
_CANCELLED: dict[tuple[str, str], float] = {}
CANCEL_MEMORY_S = 30.0


def _forget_old_cancels() -> None:
    cutoff = time.monotonic() - CANCEL_MEMORY_S
    for key in [key for key, at in _CANCELLED.items() if at < cutoff]:
        del _CANCELLED[key]


def stop_run(account: str, cancel_id: str) -> Optional[bool]:
    """Mark run ``cancel_id`` cancelled: None when no generate_audio run here has that id, True when it stops between takes, False when a take is running and the route must stop it."""
    _forget_old_cancels()
    _CANCELLED[(account, cancel_id)] = time.monotonic()
    run = _RUNS.get((account, cancel_id))
    if run is None:
        return None
    run.cancelled = True
    return not run.in_flight


async def _post_run(caller: Caller, body: dict[str, Any]) -> Any:
    response = await forward(caller, "POST", "/v1/audio/run", json_body = body)
    named = response.status_code == 400 and NOT_LOADED in response.text
    return raise_for_audio(response, hints = {400: LOAD_AUDIO_HINT} if named else None)


async def _takes_how(caller: Caller, body: dict[str, Any]) -> Optional[str]:
    """How the loaded music model makes variations of this mode: "batch", "sequential" or None."""
    status = await try_json(caller, "/api/inference/status")
    if not isinstance(status, dict):
        return None
    if body.get("model") and body["model"] != status.get("model_identifier"):
        return None
    music = status.get("audio_music")
    modes = music.get("modes") if isinstance(music, dict) else None
    for mode in modes if isinstance(modes, list) else []:
        if isinstance(mode, dict) and mode.get("id") == (body.get("mode") or "song"):
            how = mode.get("variations")
            return opt_text(how.get("how")) if isinstance(how, dict) else None
    return None


def _merge_take(merged: dict[str, Any], payload: dict[str, Any]) -> None:
    merged["clips"].extend(row for row in payload.get("clips") or [] if isinstance(row, dict))
    merged["model"] = payload.get("model") or merged.get("model")
    for key in ("ignored_options", "warnings"):
        for item in payload.get(key) or []:
            if item not in merged[key]:
                merged[key].append(item)
    if isinstance(payload.get("applied"), dict):
        merged["applied"].update(payload["applied"])


async def _run_takes(
    caller: Caller, body: dict[str, Any], *, variations: int, cancel_id: str
) -> Any:
    """Run ``body`` under ``cancel_id``. Music whose model makes variations one at a time goes take by take under one group_id, as the Audio page sends it, so a cancel keeps the finished takes."""
    key = (caller.account_id, cancel_id)
    _forget_old_cancels()
    if key in _RUNS or key in _CANCELLED:
        raise ToolError(
            f"cancel_id {cancel_id!r} is running or was just cancelled; pass a new cancel_id."
        )
    if not 1 <= variations <= 4:
        # The route refuses more, but take by take only after the first four ran.
        raise ToolError("variations must be 1 to 4.")
    run = _RUNS[key] = _Run()
    try:
        takes = variations if body["workflow"] == "music" else 1
        sequential = takes > 1 and await _takes_how(caller, body) == "sequential"
        if run.cancelled:
            raise ToolError("The audio run was cancelled before it started.")
        if not sequential:
            run.in_flight = True
            payload = await _post_run(caller, {**body, "cancel_id": cancel_id})
            if isinstance(payload, dict):
                payload["cancel_id"] = cancel_id
            return payload
        group_id = uuid.uuid4().hex
        seed = body.get("seed")
        single = {name: value for name, value in body.items() if name != "variations"}
        merged: dict[str, Any] = {
            "clips": [],
            "group_id": group_id,
            "ignored_options": [],
            "warnings": [],
            "applied": {},
            "cancel_id": cancel_id,
        }
        finished = 0
        for take in range(1, takes + 1):
            if run.cancelled:
                break
            sent = {**single, "take": take, "group_id": group_id, "cancel_id": cancel_id}
            if seed is not None:
                sent["seed"] = seed
            run.in_flight = True
            try:
                payload = await _post_run(caller, sent)
            except ToolError as exc:
                if not merged["clips"]:
                    raise
                if not run.cancelled:
                    merged["warnings"].append(f"Take {take} of {takes} failed: {exc}")
                break
            finally:
                run.in_flight = False
            if not isinstance(payload, dict):
                break
            if take == 1:
                # Later takes reuse take 1's seed, as the Audio page does.
                first = next(iter(payload.get("clips") or []), None)
                drawn = first.get("seed") if isinstance(first, dict) else None
                seed = drawn if isinstance(drawn, int) and not isinstance(drawn, bool) else seed
            _merge_take(merged, payload)
            finished = take
            if isinstance(payload.get("audio"), dict):
                # History could not save this take, so it came back inline; stop here.
                merged["audio"] = payload["audio"]
                break
        if run.cancelled and finished < takes:
            merged["warnings"].append(f"Stopped after take {finished} of {takes}: cancelled.")
        return merged
    finally:
        _RUNS.pop(key, None)
        if run.cancelled:
            _CANCELLED[key] = time.monotonic()


@with_heartbeat("Generating audio")
async def generate_audio(
    workflow: Literal["clone", "speak", "edit", "convert", "music", "separate"],
    model: Optional[str] = None,
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
    cancel_id: Annotated[
        Optional[str],
        Field(
            min_length = 8,
            max_length = 64,
            pattern = r"^[A-Za-z0-9_-]+$",
            description = "Your id for this run, so cancel(kind='audio', id=...) can stop it from "
            "another call; made when omitted and returned as cancel_id.",
        ),
    ] = None,
    edit: Optional[dict[str, Any]] = None,
    convert: Optional[dict[str, Any]] = None,
    speed: Optional[float] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_media: Annotated[
        ReturnMedia,
        Field(
            description = 'How each clip comes back besides its id and links: "link" (default) links only; '
            '"inline" the WAV itself when it fits 1.5 MiB; "preview" is the same as "link" for audio.'
        ),
    ] = "link",
    preview_px: Annotated[
        Optional[int], Field(ge = 32, le = 1024, description = "Unused for audio, which has no preview.")
    ] = None,
    options: Optional[dict[str, Any]] = None,
    ctx: Optional[Context] = None,
) -> ToolResult:
    """Run an Audio page workflow with the audio model loaded in Unsloth Studio, or with ``model`` (an id from list_models(workflow=...)), which Unsloth Studio loads first when "Switch model by request" is on in Settings > API; a model that cannot run the workflow is refused before anything is unloaded, and the refusal names the pages it runs on. Clips are saved to Audio history and returned as links, or inline with return_media="inline" when the WAV fits 1.5 MiB. clone: speak ``text`` in the voice of ``reference`` (a clip, or a saved voice as voice_id; ``reference_text`` is its transcript). speak: ``text`` in the loaded model's default voice, or a saved voice via ``reference``. edit: change the words or delivery of ``source`` per ``edit``. convert: make ``source`` sound like ``target``. music: ``mode`` song or sfx from ``text`` (the style) and ``lyrics``; ``variations`` up to 4 share a group_id. separate: split ``source`` into stems. ``options`` are the loaded model's ``audio_options`` that studio_status lists, such as a built-in ``voice``. Audio inputs are an Unsloth Studio id, inline base64 with a filename, or a path on the Unsloth Studio computer. Failures start with a [code] tag, such as [runtime_stopped] (retry once, then reload) or [out_of_memory] (CPU or a smaller variant)."""
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
    if options:
        options = await _checked_options(caller, workflow, options, model)
    body: dict[str, Any] = {"workflow": workflow, "inputs": inputs}
    body.update(
        present(
            model = model,
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
    payload = await _run_takes(
        caller, body, variations = variations, cancel_id = cancel_id or f"mcp-{uuid.uuid4().hex}"
    )
    if not isinstance(payload, dict):
        raise ToolError("Unsloth Studio returned no audio")
    clips, contents = [], []
    for row in payload.get("clips") or []:
        if not isinstance(row, dict) or not opt_text(row.get("id")):
            continue
        # workflow, voice_id, source_clip_id and seed are copied when the route sends them.
        clip = AudioClip.from_route(
            row,
            url = public_url(caller, audio_gallery_path(row["id"])),
            download_url = await signed_url(caller, "audio", row["id"]),
        )
        clips.append(clip)
        contents.extend(await _clip_contents(caller, clip, return_media))
    saved = True
    fallback = payload.get("audio")
    if isinstance(fallback, dict) and isinstance(fallback.get("data"), str):
        # History could not save the clip, so the route sent it inline instead.
        saved = False
        try:
            data = base64.b64decode(fallback["data"], validate = True)
        except (binascii.Error, ValueError):
            raise ToolError("Unsloth Studio returned unreadable audio") from None
        too_large = (
            f"Unsloth Studio could not save the clip to Audio history, and at "
            f"{len(data) / (1024 * 1024):.1f} MiB it is too large to return inline."
        )
        if len(data) <= INLINE_CAP:
            contents.append(
                audio_content(data, f"audio/{opt_text(fallback.get('format')) or 'wav'}")
            )
        elif clips:
            # Earlier takes were saved: report them, with the lost take as a warning.
            payload["warnings"] = [*(payload.get("warnings") or []), too_large]
        else:
            raise ToolError(too_large)
    if not clips and saved:
        raise ToolError("Unsloth Studio returned no audio")
    return media_result(contents, AudioResult.from_route(payload, clips = clips, saved = saved))


# OpenAI's own upload limit for /audio/transcriptions; past it the tool uploads first.
MULTIPART_LIMIT = 25 * 1024 * 1024
NOT_DOWNLOADED = "is not downloaded"
TIMESTAMPS_HINT = (
    "Timestamps need a speech-to-text model that times segments, such as Qwen3-ASR (audio.cpp) "
    "or Parakeet-TDT; pass it as model."
)


def _segments(payload: dict) -> list[TranscriptSegment]:
    # The routes leave segments out when there are none, which is not the same as not asking.
    rows = payload.get("segments")
    if not isinstance(rows, list):
        return []
    speakers = payload.get("speakers") if isinstance(payload.get("speakers"), list) else []
    labels = {str(s.get("id")): s.get("label") for s in speakers if isinstance(s, dict)}
    return [
        TranscriptSegment.from_route(row, speaker = _speaker(row.get("speaker"), labels))
        for row in rows
        if isinstance(row, dict)
    ]


def _speaker(raw: Any, labels: dict[str, Any]) -> Optional[str]:
    if raw is None or isinstance(raw, bool):
        return None
    return opt_text(labels.get(str(raw))) or opt_text(str(raw))


async def _scored(result: TranscriptResult, expected: Optional[str]) -> TranscriptResult:
    """Result with match_score vs expected; scored off the event loop the server shares."""
    if expected is None:
        return result
    from core.inference.audio_metrics import match_score

    score = await asyncio.to_thread(match_score, expected, result.text)
    return result.model_copy(update = {"match_score": score})


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
    engine = await loading.catalog_stt_engine(caller, status, model)
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
    fields: dict[str, Any] = {"response_format": "verbose_json" if timestamps else "json"}
    if timestamps and not translate:
        # The route times segments only when asked, and refuses loudly on an engine that cannot.
        fields["timestamp_granularities[]"] = "segment"
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
    return model, await loading.catalog_stt_engine(caller, status, model)


async def _transcribe_source(
    caller: Caller,
    ctx: Optional[Context],
    source: dict[str, str],
    *,
    language: Optional[str],
    timestamps: bool,
    speakers: bool = False,
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
    if speakers:
        body["speakers"] = True
    # A diarized transcript comes back with its segments.
    timestamps = timestamps or speakers

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
    record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    return TranscriptResult(
        text = payload.get("text") if isinstance(payload.get("text"), str) else "",
        transcript_id = opt_text(record.get("id")),
        language = opt_text(payload.get("language")),
        model = model,
        segments = _segments(payload) if timestamps else None,
        saved_to_history = True,
    )


@with_heartbeat("Transcribing")
async def transcribe(
    audio: AudioInput,
    language: Optional[str] = None,
    translate: bool = False,
    timestamps: bool = False,
    speakers: Annotated[
        bool,
        Field(
            description = "Label who speaks each segment (segments[].speaker); needs a model that "
            "tells speakers apart, such as MOSS-Transcribe-Diarize or VibeVoice-ASR."
        ),
    ] = False,
    expected_text: Annotated[
        Optional[str],
        Field(
            max_length = 20000,
            description = "The words the audio should say; adds match_score, 1.0 when every "
            "word matches and null when the texts are too long to align.",
        ),
    ] = None,
    model: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> TranscriptResult:
    """Transcribe speech with Unsloth Studio's speech-to-text, or translate it to English with ``translate``.
    ``audio`` is inline base64 with a filename, a path on the Unsloth Studio computer, or an Unsloth Studio id.
    Audio over 25 MB, given by id, or sent with ``speakers`` is stored with Unsloth Studio first and its transcript is saved to Audio history (``saved_to_history``); translation needs a smaller file.
    ``timestamps`` adds segments and needs a model that times them, such as Qwen3-ASR (audio.cpp) or Parakeet-TDT; others refuse it, and translation has none.
    ``speakers`` labels who speaks each segment and needs a model that tells speakers apart.
    ``expected_text`` adds match_score: 1 minus the word error rate against it, so 1.0 when every word matches; it is null when the texts are too long to align (over 50 million word pairs; in Chinese, Japanese and Thai each character counts as a word).
    A missing model is downloaded first.
    Without ``model`` the loaded speech-to-text model or Unsloth Studio's default is used."""
    if translate and (timestamps or speakers):
        raise ToolError(
            "Translation returns text only; leave out timestamps and speakers, or transcribe "
            "without translate."
        )
    caller = current_caller()
    data, name = audio_bytes(caller, audio) if audio.is_upload else (None, None)
    if speakers or data is None or len(data) > MULTIPART_LIMIT:
        if translate:
            raise ToolError("Translation needs a file under 25 MB sent as data or a path.")
        source = (
            {"input_id": await upload_audio(caller, data, name)}
            if data is not None
            else await audio_ref(caller, audio)
        )
        return await _scored(
            await _transcribe_source(
                caller,
                ctx,
                source,
                language = language,
                timestamps = timestamps,
                speakers = speakers,
                model = model,
            ),
            expected_text,
        )
    # Without a model the multipart route runs its Transformers default, not the loaded one.
    model, engine = await _resident_stt(caller, model)
    # A whisper.cpp id is shared with Transformers, so only the stored-audio route keeps that engine.
    if engine == "gguf" and not translate:
        source = {"input_id": await upload_audio(caller, data, name)}
        return await _scored(
            await _transcribe_source(
                caller, ctx, source, language = language, timestamps = timestamps, model = model
            ),
            expected_text,
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
    # Off audio.cpp the route refuses timestamps up front: 501 without language, 400 with it.
    untimed = timestamps and engine != "audiocpp"
    hints = {400: TIMESTAMPS_HINT, 501: TIMESTAMPS_HINT} if untimed else None
    payload = raise_for_route(response, hints = hints)
    if not isinstance(payload, dict):
        raise ToolError("Unsloth Studio returned no transcript")
    result = TranscriptResult(
        text = payload.get("text") if isinstance(payload.get("text"), str) else "",
        # This route answers with the text alone, so report what was asked for and what ran.
        # A translation is always English; its verbose form spells it "english".
        language = "en" if translate else (opt_text(payload.get("language")) or language),
        model = model or (await _resident_stt(caller))[0],
        segments = _segments(payload) if timestamps else None,
        saved_to_history = False,
    )
    return await _scored(result, expected_text)


async def _resident_stt(
    caller: Caller, model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """The model and the engine serving it: the named one where it is held, else the loaded one."""
    payload = await try_json(caller, loading.STT_STATUS, params = {"model": model} if model else None)
    status = payload if isinstance(payload, dict) else {}
    if model:
        return model, await loading.catalog_stt_engine(caller, status, model)
    for engine in loading.STT_ENGINES:
        state = status.get(engine)
        if isinstance(state, dict) and opt_text(state.get("loaded_model")):
            return state["loaded_model"], engine
    return None, None


ANALYZE_STT_HINT = (
    "Run transcribe once to download Unsloth Studio's speech-to-text model, or leave out "
    "expected_text."
)


async def analyze_audio(
    audio: AudioInput,
    expected_text: Annotated[
        Optional[str],
        Field(
            max_length = 20000,
            description = "The words the audio should say; adds the transcript and match_score, "
            "1.0 when every word matches and null when the texts are too long to align.",
        ),
    ] = None,
) -> AnalysisResult:
    """Measure a clip: its duration, integrated loudness (LUFS), true peak (dBTP), clipped samples and where the sound starts and ends.
    ``audio`` is an Unsloth Studio id (such as a clip_id from generate_audio), inline base64 with a filename, or a path on the Unsloth Studio computer.
    With ``expected_text`` the clip is also transcribed by the loaded or default speech-to-text model, and match_score is 1 minus the word error rate, or null when the texts are too long to align.
    Nothing is added to Audio history."""
    caller = current_caller()
    body = {**await audio_ref(caller, audio), **present(expected_text = expected_text)}
    payload = await route_json(
        "POST",
        "/api/inference/audio/analyze",
        caller = caller,
        json_body = body,
        hint_if = (NOT_DOWNLOADED, {409: ANALYZE_STT_HINT}),
    )
    return AnalysisResult.from_route(payload)


TOOLS = (
    (generate_audio, WRITES, AudioResult.model_json_schema()),
    (transcribe, WRITES),
    # Not read-only: inline audio or a path is uploaded to Unsloth Studio first.
    (analyze_audio, WRITES),
)
