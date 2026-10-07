# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice conversion on audio.cpp: the session a Convert run needs and its /v1/tasks/run request."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from core.inference.audio_cpp_models import AudioCppModel

CONVERT_SOURCE_MAX_SECONDS = 300
CONVERT_TARGET_MAX_SECONDS = 30

CONVERT_MODES = ("speech", "singing")

_STYLE_ROUTES = {"vc": "style_converted_vc"}


class ConvertRequestError(ValueError):
    """A user-facing reason the loaded model cannot run this conversion."""


def convert_caps(model: AudioCppModel) -> Optional[dict[str, Any]]:
    convert = model.convert
    if convert is None:
        return None
    return {
        "modes": [mode for mode, _task in convert.modes],
        "target": convert.target,
        "builtin_voices": [{"id": vid, "label": label} for vid, label in convert.builtin_voices],
        "pitch": {
            mode: {"auto": auto, "shift_with_auto": auto and convert.pitch_options == "semitone"}
            for mode, auto in convert.pitch
        },
        "style": convert.style,
        "route_reloads": bool(convert.routes),
        "source_max_seconds": CONVERT_SOURCE_MAX_SECONDS,
    }


def workflow_tasks(model: AudioCppModel) -> dict[str, str]:
    tasks = {workflow: binding.server_task for workflow, binding in model.workflows.items()}
    if model.convert is not None:
        for mode, task in model.convert.modes[1:]:
            tasks[f"convert:{mode}"] = task
    return tasks


def served_route(model: AudioCppModel) -> Optional[str]:
    convert = model.convert
    if convert is None or not convert.routes:
        return None
    configured = ((model.model_options or {}).get("default_request_options") or {}).get("route")
    if configured:
        return str(configured)
    if model.server_task == convert.modes[0][1]:
        return convert.routes[0]
    return "v1_svc" if model.server_task == "svc" else None


def served_model(
    model: AudioCppModel, mode: str, options: dict[str, Any]
) -> tuple[AudioCppModel, dict[str, Any]]:
    """``(model as the server must load it for this run, options left for the request)``."""
    convert = model.convert
    if convert is None:
        raise ConvertRequestError(f"{model.display_name} cannot convert a voice.")
    tasks = convert.server_tasks
    if mode not in tasks:
        raise ConvertRequestError(f"{model.display_name} does not convert {mode}.")
    rest = dict(options)
    requested = rest.pop("route", None)
    model_options = {
        key: value
        for key, value in (model.model_options or {}).items()
        if key != "default_request_options"
    }
    if convert.routes and mode == convert.modes[0][0]:
        route = str(requested) if requested in convert.routes else convert.routes[0]
        if route != convert.routes[0]:
            model_options["default_request_options"] = {"route": route}
    return replace(model, server_task = tasks[mode], model_options = model_options), rest


def convert_request(
    model: AudioCppModel,
    *,
    mode: str,
    source: str,
    target: Optional[str],
    voice: Optional[str] = None,
    pitch: Optional[int] = None,
    pitch_auto: bool = False,
    style: str = "source",
    source_text: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    convert = model.convert
    if convert is None:
        raise ConvertRequestError(f"{model.display_name} cannot convert a voice.")
    if not source:
        raise ConvertRequestError("Add the recording to convert.")
    builtin = convert.target == "builtin"
    if builtin and target:
        raise ConvertRequestError(
            f"{model.display_name} converts to its built-in voices; pick one under Built-in."
        )
    if not builtin and not target:
        raise ConvertRequestError("Add the target voice.")
    task = convert.server_tasks.get(mode)
    take_style = convert.style and style == "target" and task in _STYLE_ROUTES
    transcript = str(source_text or "").strip()
    if take_style and not transcript:
        raise ConvertRequestError("Type what's said in the recording, or press Transcribe.")

    request: dict[str, Any] = {}
    if convert.fields == "source_target":
        request["source_audio"] = str(source)
        request["target_voice"] = str(target)
    else:
        request["audio"] = str(source)
        if target:
            request["voice_ref"] = str(target)
    if take_style:
        request["route"] = _STYLE_ROUTES[task]
        request["target_text"] = transcript
        request["style_ref"] = str(target)

    request_options = dict(options or {})
    if builtin:
        ids = [vid for vid, _label in convert.builtin_voices]
        request_options["voice_id"] = voice if voice in ids else ids[0]
    pitch_modes = dict(convert.pitch)
    if mode in pitch_modes and not take_style:
        auto = pitch_auto and pitch_modes[mode]
        if convert.pitch_options == "shift_steps":
            if not auto and pitch is not None:
                if int(pitch):
                    request_options["use_pitch_shift"] = "true"
                    request_options["source_shift_steps"] = str(int(pitch))
                else:
                    request_options["use_pitch_shift"] = "false"
        else:
            if auto:
                request_options["auto_f0_adjust"] = "true"
            if pitch is not None:
                request_options["semitone_shift"] = str(int(pitch))
    if request_options:
        request["options"] = request_options
    if convert.seed and seed is not None:
        request["seed"] = str(int(seed))
    return request
