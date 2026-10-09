# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``studio_status``: every model slot, training, export and the GPUs in one call."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from fastmcp import FastMCP
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
from studio_mcp.tools import READ_ONLY, integer, number, opt_text

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
    loaded = []
    for model_id in payload.get("loaded") or []:
        if not isinstance(model_id, str):
            continue
        is_active = model_id == active
        loaded.append(
            ChatModel(
                id = model_id,
                display_name = opt_text(payload.get("active_model")) if is_active else None,
                is_gguf = bool(payload.get("is_gguf")) if is_active else None,
            )
        )
    loading = [model for model in payload.get("loading") or [] if isinstance(model, str)]
    return ChatSlot(loaded = loaded, loading = loading)


def _image(payload: dict) -> ImageSlot:
    loaded = payload.get("loaded") is True
    return ImageSlot(
        loaded = loaded,
        model = (opt_text(payload.get("display_repo_id")) or opt_text(payload.get("repo_id")))
        if loaded
        else None,
        family = opt_text(payload.get("family")) if loaded else None,
    )


def _video(payload: dict) -> VideoSlot:
    loaded = payload.get("loaded") is True
    return VideoSlot(
        loaded = loaded,
        model = (opt_text(payload.get("display_repo_id")) or opt_text(payload.get("repo_id")))
        if loaded
        else None,
    )


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
    return EmbedderSlot(
        available = True,
        loaded = payload.get("loaded") is True,
        model = "custom" if custom else opt_text(payload.get("embedding_model")),
    )


def _training(payload: dict) -> TrainingSlot:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    step, total = integer(details.get("step")), integer(details.get("total_steps"))
    return TrainingSlot(
        job_id = opt_text(payload.get("job_id")),
        phase = opt_text(payload.get("phase")) or "idle",
        is_training_running = payload.get("is_training_running") is True,
        message = opt_text(payload.get("message")) or "",
        error = opt_text(payload.get("error")),
        step = step,
        total_steps = total,
        loss = number(details.get("loss")),
        progress_percent = round(100.0 * step / total, 1) if step is not None and total else None,
        eta_seconds = number(details.get("eta_seconds")),
    )


def _folder_name(value: Any) -> Any:
    if not opt_text(value):
        return None
    return re.split(r"[\\/]", value.rstrip("\\/"))[-1] or None


def _export(payload: dict) -> ExportSlot:
    return ExportSlot(
        active = payload.get("is_export_active") is True,
        op_kind = opt_text(payload.get("active_op_kind")),
        last_op_status = opt_text(payload.get("last_op_status")),
        last_output = _folder_name(payload.get("last_op_output_path")),
    )


def _hardware(payload: dict) -> HardwareSlot:
    devices = [
        GpuDevice(
            index = integer(device.get("index")),
            gpu_utilization_pct = number(device.get("gpu_utilization_pct")),
            vram_used_gb = number(device.get("vram_used_gb")),
            vram_total_gb = number(device.get("vram_total_gb")),
            temperature_c = number(device.get("temperature_c")),
        )
        for device in payload.get("devices") or []
        if isinstance(device, dict)
    ]
    return HardwareSlot(
        available = payload.get("available") is True,
        backend = opt_text(payload.get("backend")),
        devices = devices,
    )


BUILDERS: dict[str, Callable[[dict], Any]] = {
    "chat": _chat,
    "image": _image,
    "video": _video,
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


async def studio_status() -> StudioStatus:
    """What Unsloth Studio is doing right now: the loaded chat, image, video, speech-to-text and embedding models, any model still loading or downloading, the training job, the export job and GPU use. Call this before loading a model or starting GPU work. A slot that could not be read is listed under ``unavailable`` with the reason."""
    caller = current_caller()
    slots = list(SLOT_ROUTES)
    results = await asyncio.gather(
        *(_read_slot(caller, slot) for slot in slots), return_exceptions = True
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
    return StudioStatus(**status, unavailable = unavailable)


def register_status(mcp: FastMCP) -> None:
    mcp.tool(studio_status, annotations = READ_ONLY)
