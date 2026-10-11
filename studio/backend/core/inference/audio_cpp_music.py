# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp music request bodies for ``/audio/generate`` and ``/audio/run``.

A strict spec (``schema_version``) refuses undeclared options, so the CLI's top-level shortcuts
(``duration_seconds``, ``lyrics``) are sent to it only as declared options. MiniMax keeps its old
body: its session reads both duration keys.
"""

from __future__ import annotations

import math
import secrets
from typing import Any, Optional

from core.inference.audio_cpp_models import (
    MUSIC_MAX_VARIATIONS,
    MUSIC_SPECS,
    AudioCppModel,
    MusicMode,
)
from loggers import get_logger

logger = get_logger(__name__)

_TIMEOUT_FLOOR = 300.0
_TIMEOUT_CEILING = 4 * 3600.0
_CPU_FACTOR = 10.0
# MiDashengLM's spec caps seeds at 2**31 - 1.
_SEED_LIMIT = 2**31 - 1


class MusicRequestError(ValueError):
    """Message shown to the user as is."""


def random_seed() -> int:
    return secrets.randbelow(_SEED_LIMIT)


def take_seed(seed: Optional[int], index: int) -> Optional[int]:
    """Seed of the ``index``-th sequential take; wraps so a valid seed never overflows the spec."""
    if seed is None:
        return None
    return (seed + index) % (_SEED_LIMIT + 1) if 0 <= seed <= _SEED_LIMIT else seed + index


def seconds_text(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def timeout_seconds(family: Optional[str], seconds: float, variations: int, cpu: bool) -> float:
    music = MUSIC_SPECS.get(str(family or ""))
    rtf = music.rtf if music is not None else 4.0
    work = rtf * max(0.0, float(seconds)) * max(1, int(variations)) * (_CPU_FACTOR if cpu else 1.0)
    return max(_TIMEOUT_FLOOR, min(_TIMEOUT_CEILING, _TIMEOUT_FLOOR + work))


def frames_for(seconds: float, variations: int = 1) -> int:
    return int(math.ceil(max(0.0, float(seconds)) * 25 * max(1, int(variations))))


def music_rules(model: AudioCppModel, max_batch: int = 1) -> Optional[dict[str, Any]]:
    music = model.music
    if music is None or model.task != "music":
        return None
    modes: list[dict[str, Any]] = []
    for mode in music.modes:
        if mode.id == "edit":
            modes.append(
                {
                    "id": "edit",
                    "actions": list(mode.actions),
                    "max_ranges": mode.max_ranges,
                    "max_source_s": mode.max_source_s,
                }
            )
            continue
        entry: dict[str, Any] = {"id": mode.id}
        if mode.id == "song":
            entry.update(
                {
                    "lyrics": mode.lyrics,
                    "description": mode.description,
                    "instrumental": mode.instrumental,
                    "section_case": mode.section_case,
                }
            )
        low, high, default = mode.duration
        entry["duration"] = {
            "min": low,
            "max": high,
            "default": default,
            "approximate": mode.approximate,
        }
        entry["variations"] = (
            {
                "max": MUSIC_MAX_VARIATIONS,
                "how": mode.variations,
                # Variations one request makes without a reload.
                "loaded": max(1, int(max_batch))
                if mode.variations == "batch"
                else MUSIC_MAX_VARIATIONS,
            }
            if mode.variations
            else None
        )
        modes.append(entry)
    return {"modes": modes}


def _declared(model: AudioCppModel, name: str) -> bool:
    return model.request_keys is None or name in model.request_keys


def _put_option(model: AudioCppModel, request: dict, options: dict, name: str, value: Any) -> None:
    if model.request_keys is None:
        request[name] = value
    elif name in model.request_keys:
        options[name] = value
    else:
        logger.info("audio.cpp: %s does not declare %s; not sending it", model.family, name)


def _put_duration(model: AudioCppModel, request: dict, options: dict, seconds: float) -> None:
    keys = model.request_keys
    if keys is None:
        request["duration_seconds"] = seconds
    elif "duration_sec" in keys:
        options["duration_sec"] = seconds
    elif "duration_seconds" in keys:
        options["duration_seconds"] = seconds
    else:
        logger.info("audio.cpp: %s declares no duration; not sending one", model.family)


def _finish(model: AudioCppModel, request: dict, options: dict, seed: Optional[int]) -> dict:
    if model.request_keys is not None:
        dropped = sorted(name for name in options if name not in model.request_keys)
        if dropped:
            logger.info(
                "audio.cpp: %s does not declare %s; not sending them", model.family, dropped
            )
            options = {k: v for k, v in options.items() if k in model.request_keys}
    if options:
        request["options"] = options
    if seed is not None:
        request["seed"] = str(int(seed))
    return request


def song_request(
    model: AudioCppModel,
    *,
    description: str,
    lyrics: str,
    seconds: float,
    options: dict,
    seed: Optional[int] = None,
    instrumental: bool = False,
    batch: int = 1,
) -> dict[str, Any]:
    description = str(description or "").strip()
    lyrics = str(lyrics or "").strip()
    music = model.music
    request: dict[str, Any] = {}
    request_options: dict[str, Any] = dict(options)
    if model.family == "minimax_music3":
        if not lyrics:
            raise MusicRequestError("MiniMax Music 3 needs lyrics.")
        # Sending duration_seconds and duration_sec together is refused as conflicting, even equal.
        request.update(
            {"text": description or lyrics, "lyrics": lyrics, "duration_seconds": seconds}
        )
        request_options["lyrics"] = lyrics
        if request_options:
            request["options"] = request_options
        if seed is not None:
            request["seed"] = str(int(seed))
        return request
    if model.family == "yue2":
        if not description:
            raise MusicRequestError("YuE2 needs a style description.")
        # Missing or empty lyrics still sing; only "[Instrumental]" comes back wordless.
        if instrumental or not lyrics:
            lyrics = (
                music.instrumental_lyrics
                if music and music.instrumental_lyrics
                else "[Instrumental]"
            )
        request["text"] = lyrics
        request_options["style"] = description
        request_options["lyrics"] = lyrics
        # Length is the semantic token budget (25 fps); the default 200-frame floor outlasts short asks.
        frames = int(round(seconds * 25))
        request_options["semantic_max_tokens"] = frames
        request_options["semantic_min_tokens"] = min(200, frames)
        return _finish(model, request, request_options, seed)
    if instrumental and music is not None and music.instrumental_lyrics is not None:
        lyrics = music.instrumental_lyrics
    request["text"] = description
    if music is not None and music.description_option:
        request_options[music.description_option] = description
    _put_duration(model, request, request_options, seconds)
    if lyrics and model.family != "stable_audio":
        _put_option(model, request, request_options, "lyrics", lyrics)
    if batch > 1:
        request_options["batch_size"] = str(int(batch))
    return _finish(model, request, request_options, seed)


def legacy_song_request(
    model: AudioCppModel,
    text: str,
    instructions: Optional[str],
    seconds: float,
    options: dict,
    seed: Optional[int],
) -> dict[str, Any]:
    """Description = instructions, lyrics = text; a lone prompt is the description."""
    description = str(instructions or "").strip()
    lyrics = str(text or "").strip()
    if model.family not in ("minimax_music3", "yue2") and not description:
        description, lyrics = lyrics, ""
    # MiDashengLM crashes past 81 s.
    mode = model.music.modes[0] if model.music is not None and model.music.modes else None
    if mode is not None:
        seconds = min(seconds, mode.duration[1])
    return song_request(
        model,
        description = description,
        lyrics = lyrics,
        seconds = seconds,
        options = options,
        seed = seed,
    )


def merge_ranges(ranges: list[tuple[float, float]], limit: float) -> list[tuple[float, float]]:
    clamped = sorted(
        (max(0.0, float(s)), min(float(limit), float(e)))
        for s, e in ranges
        if min(float(limit), float(e)) > max(0.0, float(s))
    )
    merged: list[tuple[float, float]] = []
    for start, end in clamped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def edit_request(
    model: AudioCppModel,
    *,
    text: str,
    edit: dict,
    source: str,
    source_seconds: float,
    duration_s: Optional[float],
    options: dict,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    mode = model.music.mode("edit") if model.music is not None else None
    action = str(edit.get("action") or "")
    if mode is None or action not in mode.actions:
        raise MusicRequestError(f"{model.display_name} cannot {action or 'edit'} a clip.")
    text = str(text or "").strip()
    if not text:
        raise MusicRequestError("Describe the change.")
    ranges = [(float(r["start_s"]), float(r["end_s"])) for r in edit.get("ranges") or []]
    strength = edit.get("strength")
    request: dict[str, Any] = {"text": text, "audio": str(source)}
    request_options: dict[str, Any] = dict(options)
    if model.family == "ace_step":
        if action in ("repaint", "extend"):
            if action == "extend":
                start, end = source_seconds, source_seconds + float(edit.get("extend_s") or 0.0)
            else:
                (start, end) = ranges[0]
            # The window may pass the clip's end: ACE-Step pads it, extending the song.
            request_options.update(
                {"route": "repaint", "repainting_start": start, "repainting_end": end}
            )
            if strength is not None and action == "repaint":
                request_options["repaint_strength"] = float(strength)
        elif action == "cover":
            request_options["route"] = "cover"
            if strength is not None:
                # strength is how much to change; audio_cover_strength is the share of steps
                # conditioned on the source (diffusion.cpp), so higher keeps more of it.
                request_options["audio_cover_strength"] = 1.0 - float(strength)
        else:  # continue: ACE-Step's "complete" adds parts across the whole track.
            request_options["route"] = "complete"
            request_options["duration_seconds"] = float(duration_s or source_seconds)
        return _finish(model, request, request_options, seed)
    if model.family == "stable_audio":
        if action == "inpaint":
            merged = merge_ranges(ranges, source_seconds)
            if not merged:
                raise MusicRequestError("Pick a part of the clip to change.")
            request_options.update(
                {
                    "audio_input_kind": "inpaint_audio",
                    "inpaint_mask_start_seconds": ",".join(seconds_text(s) for s, _ in merged),
                    "inpaint_mask_end_seconds": ",".join(seconds_text(e) for _, e in merged),
                }
            )
        else:  # restyle
            request_options["audio_input_kind"] = "init_audio"
            if strength is not None:
                request_options["init_noise_level"] = float(strength)
        _put_duration(model, request, request_options, source_seconds)
        return _finish(model, request, request_options, seed)
    raise MusicRequestError(f"{model.display_name} cannot edit a clip.")


def song_mode(model: AudioCppModel, mode_id: str) -> MusicMode:
    mode = model.music.mode(mode_id) if model.music is not None else None
    if mode is None:
        raise MusicRequestError(f"{model.display_name} does not offer {mode_id}.")
    return mode


def clamp_seconds(mode: MusicMode, seconds: Optional[float]) -> float:
    low, high, default = mode.duration
    value = default if seconds is None else float(seconds)
    return max(low, min(high, value))
