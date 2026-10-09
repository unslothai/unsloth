# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``studio_status``: every model slot, training, export and the GPUs in one call."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable

from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.outputs import (
    ChatModel,
    ChatSlot,
    EmbedderSlot,
    ExportSlot,
    GpuDevice,
    HardwareSlot,
    ImageSlot,
    StudioStatus,
    SttDownload,
    SttSlot,
    TrainingSlot,
    VideoSlot,
)
from studio_mcp.tools import READ_ONLY, as_dict, integer, leaf_name, number, opt_text, try_json

SLOT_ROUTES = {
    "chat": "/api/inference/status",
    "image": "/api/inference/images/status",
    "video": "/api/inference/video/status",
    "stt": "/api/inference/audio/stt/status",
    "embedder": "/api/settings/embedding-model",
    "training": "/api/train/status",
    "export": "/api/export/status",
    # The route reads the GPUs off the event loop; never call get_gpu_utilization from a tool.
    "hardware": "/api/train/hardware",
}
STT_ENGINES = ("transformers", "mtmd", "gguf", "audiocpp")


def _chat(payload: dict) -> ChatSlot:
    active = payload.get("model_identifier")
    loaded = [
        ChatModel(
            id = model_id,
            display_name = opt_text(payload.get("active_model")) if model_id == active else None,
            is_gguf = bool(payload.get("is_gguf")) if model_id == active else None,
        )
        for model_id in payload.get("loaded") or []
        if isinstance(model_id, str)
    ]
    loading = [model for model in payload.get("loading") or [] if isinstance(model, str)]
    return ChatSlot(loaded = loaded, loading = loading)


def _media(slot: type, payload: dict) -> Any:
    if payload.get("loaded") is not True:
        return slot()
    model = opt_text(payload.get("display_repo_id")) or opt_text(payload.get("repo_id"))
    family = {"family": opt_text(payload.get("family"))} if slot is ImageSlot else {}
    return slot(loaded = True, model = model, **family)


def _stt(payload: dict) -> SttSlot:
    engine = model = None
    loading = False
    downloading = []
    for name in STT_ENGINES:
        state = payload.get(name)
        if not isinstance(state, dict):
            continue
        if model is None and opt_text(state.get("loaded_model")):
            engine, model = name, state["loaded_model"]
        loading = loading or state.get("loading") is True
        download = state.get("download")
        if (
            isinstance(download, dict)
            and download.get("downloading")
            and opt_text(download.get("model"))
        ):
            done, total = number(download.get("bytes_done")), number(download.get("bytes_total"))
            fraction = min(1.0, done / total) if done is not None and total else None
            downloading.append(SttDownload(model = download["model"], fraction = fraction))
    return SttSlot(engine = engine, model = model, loading = loading, downloading = downloading)


def _embedder(payload: dict) -> EmbedderSlot:
    custom = payload.get("is_custom") is True
    model = "custom" if custom else opt_text(payload.get("embedding_model"))
    return EmbedderSlot.from_route(payload, available = True, model = model)


def _training(payload: dict) -> TrainingSlot:
    details = as_dict(payload.get("details"))
    step, total = integer(details.get("step")), integer(details.get("total_steps"))
    return TrainingSlot.from_route(
        payload,
        step = step,
        total_steps = total,
        loss = number(details.get("loss")),
        progress_percent = round(100.0 * step / total, 1) if step is not None and total else None,
        eta_seconds = number(details.get("eta_seconds")),
    )


def _export(payload: dict) -> ExportSlot:
    path = opt_text(payload.get("last_op_output_path"))
    return ExportSlot.from_route(
        payload,
        active = payload.get("is_export_active") is True,
        op_kind = opt_text(payload.get("active_op_kind")),
        last_output = (path and leaf_name(path)) or None,
    )


def _hardware(payload: dict) -> HardwareSlot:
    devices = [
        GpuDevice.from_route(device)
        for device in payload.get("devices") or []
        if isinstance(device, dict)
    ]
    return HardwareSlot.from_route(payload, devices = devices)


BUILDERS: dict[str, Callable[[dict], Any]] = {
    "chat": _chat,
    "image": partial(_media, ImageSlot),
    "video": partial(_media, VideoSlot),
    "stt": _stt,
    "embedder": _embedder,
    "training": _training,
    "export": _export,
    "hardware": _hardware,
}


async def _read_slot(caller: Caller, slot: str) -> Any:
    response = await forward(caller, "GET", SLOT_ROUTES[slot])
    if slot == "embedder" and response.status_code == 403:
        # The embedder setting is owner-only; a managed account's key simply cannot see it.
        return EmbedderSlot(available = False)
    payload = raise_for_route(response)
    if not isinstance(payload, dict):
        raise ToolError("unexpected response")
    return BUILDERS[slot](payload)


# Whether a generation is running; read best-effort, since a missing answer only means "not known".
GENERATING_ROUTES = {
    "image": "/api/inference/images/generate-progress",
    "video": "/api/inference/video/generate-progress",
}


async def _generating(caller: Caller, slot: str) -> bool:
    payload = await try_json(caller, GENERATING_ROUTES[slot])
    return isinstance(payload, dict) and payload.get("active") is True


async def studio_status() -> StudioStatus:
    """What Unsloth Studio is doing right now: the loaded chat, image, video, speech-to-text and embedding models, any model still loading or downloading, whether an image or video is generating, the training job, the export job and GPU use. Call this before loading a model or starting GPU work. The embedding model is managed by Unsloth Studio and unload_model does not unload it. A slot that could not be read is listed under ``unavailable`` with the reason."""
    caller = current_caller()
    slots = list(SLOT_ROUTES)
    results, generating = await asyncio.gather(
        asyncio.gather(*(_read_slot(caller, slot) for slot in slots), return_exceptions = True),
        asyncio.gather(*(_generating(caller, slot) for slot in GENERATING_ROUTES)),
    )
    status: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for slot, result in zip(slots, results):
        if isinstance(result, ToolError):
            unavailable[slot] = str(result)
        elif isinstance(result, BaseException):
            unavailable[slot] = "unexpected response"
        else:
            status[slot] = result
    for slot, active in zip(GENERATING_ROUTES, generating):
        if active and slot in status:
            status[slot] = status[slot].model_copy(update = {"generating": True})
    return StudioStatus(**status, unavailable = unavailable)


TOOLS = ((studio_status, READ_ONLY),)
