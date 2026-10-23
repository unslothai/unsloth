# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio nodes: speech, music, transcription and joining WAV clips."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from core.flows.nodes._common import as_text, audio_run, ensure_model, media_bytes
from core.flows.registry import FlowNodeError, NodeContext, handler


@handler("audio.speak")
async def speak(ctx: NodeContext) -> dict:
    clip = await ctx.speak(
        model = ctx.params.get("model"),
        text = as_text(ctx.inputs.get("text")),
        voice = str(ctx.params.get("voice") or ""),
        instructions = str(ctx.params.get("instructions") or ""),
        reference = ctx.inputs.get("reference"),
        seed = ctx.seed,
    )
    return {"audio": clip}


@handler("audio.music")
async def music(ctx: NodeContext) -> dict:
    await ensure_model(ctx, "audio", ctx.params.get("model"))
    prompt = as_text(ctx.inputs.get("prompt")) or str(ctx.params.get("prompt") or "")
    if not prompt.strip():
        raise FlowNodeError("Describe the music or sound")
    body: dict[str, Any] = {
        "workflow": "music",
        "mode": str(ctx.params.get("mode") or "song"),
        "text": prompt,
        "seed": ctx.seed,
    }
    if ctx.inputs.get("lyrics"):
        body["lyrics"] = as_text(ctx.inputs["lyrics"])
    if ctx.params.get("duration_s"):
        body["duration_s"] = float(ctx.params["duration_s"])
    clips = await audio_run(ctx, body)
    return {"audio": clips[0]}


@handler("audio.transcribe")
async def transcribe(ctx: NodeContext) -> dict:
    data, mime = await media_bytes(ctx, ctx.inputs["audio"])
    path = "/v1/audio/translations" if ctx.params.get("translate") else "/v1/audio/transcriptions"
    form = {"response_format": "verbose_json"}
    if ctx.params.get("speakers") and not ctx.params.get("translate"):
        form["response_format"] = "diarized_json"
    files = {"file": ("clip" + _ext(mime), data, mime)}
    try:
        answer = await ctx.call("POST", path, files = files, data = form)
    except FlowNodeError as exc:
        # Some local engines only time segments when told the language; plain text still works there.
        if exc.status != 501:
            raise
        answer = await ctx.call("POST", path, files = files, data = {"response_format": "json"})
    if isinstance(answer, dict):
        return {"text": str(answer.get("text") or ""), "segments": answer.get("segments") or []}
    return {
        "text": as_text(answer.decode("utf-8", "replace") if isinstance(answer, bytes) else answer),
        "segments": [],
    }


def _ext(mime: str) -> str:
    return {
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }.get(mime, ".wav")


MAX_GAP_MS = 10_000
MAX_OUTPUT_BYTES = 1024**3


def concat_wav(clips: list[bytes], gap_ms: int) -> bytes:
    params = None
    frames: list[bytes] = []
    for i, data in enumerate(clips):
        try:
            with wave.open(io.BytesIO(data)) as wav:
                current = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
                if params is None:
                    params = current
                elif current != params:
                    raise FlowNodeError(
                        f"Clip {i + 1} has another sample rate or channel layout; generate the clips with one model"
                    )
                frames.append(wav.readframes(wav.getnframes()))
        except wave.Error:
            raise FlowNodeError(
                f"Clip {i + 1} is not a WAV file; Join Audio takes WAV clips"
            ) from None
    assert params is not None
    channels, width, rate = params
    gap_ms = min(max(0, gap_ms), MAX_GAP_MS)
    silence = b"\x00" * (int(rate * gap_ms / 1000) * channels * width)
    if sum(len(f) for f in frames) + len(silence) * max(0, len(frames) - 1) > MAX_OUTPUT_BYTES:
        raise FlowNodeError("The joined audio would be over 1 GB; join fewer or shorter clips")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        for i, chunk in enumerate(frames):
            if i:
                wav.writeframes(silence)
            wav.writeframes(chunk)
    return out.getvalue()


@handler("audio.concat")
async def concat(ctx: NodeContext) -> dict:
    clips_in = [c for c in ctx.inputs.get("clips") or [] if isinstance(c, dict)]
    if not clips_in:
        raise FlowNodeError("No clips to join")
    if len(clips_in) > 500:
        raise FlowNodeError("Join Audio takes at most 500 clips")
    clips = [(await media_bytes(ctx, c))[0] for c in clips_in]
    data = await asyncio.to_thread(concat_wav, clips, int(ctx.params.get("gap_ms") or 0))
    with wave.open(io.BytesIO(data)) as wav:
        duration = wav.getnframes() / float(wav.getframerate())
    return {
        "audio": await ctx.save_asset(
            data, "audio/wav", kind = "audio", extra = {"duration_s": round(duration, 3)}
        )
    }
