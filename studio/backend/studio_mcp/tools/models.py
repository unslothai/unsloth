# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model discovery and residency for agents: ``list_models``, ``load_model``, ``unload_model``."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from studio_mcp import loading
from studio_mcp.caller import current_caller
from studio_mcp.outputs import HardwareSlot, LoadResult, ModelEntry, ModelList, UnloadResult
from studio_mcp.tools import (
    DESTRUCTIVE,
    READ_ONLY,
    WRITES,
    as_dict,
    integer,
    opt_text,
    route_json,
    strings,
    try_json,
)
from studio_mcp.tools.status import SLOT_ROUTES, _hardware

# "tts" and "audio" are what list_models reports for audio models; they load in the llm slot.
ModelKind = Literal["llm", "image", "video", "stt", "tts", "audio"]
AudioWorkflow = Literal["speak", "clone", "edit", "convert", "music", "separate", "transcribe"]

KIND_BY_TASK = {
    "text-to-image": "image",
    "text-to-video": "video",
    "automatic-speech-recognition": "stt",
    "text-to-speech": "tts",
    "audio-to-audio": "audio",
}

# Scalars a training config may start from; nothing that names a file or a folder.
TRAINING_DEFAULT_KEYS = {
    "training": (
        "max_seq_length",
        "num_epochs",
        "learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "max_steps",
        "save_steps",
        "weight_decay",
        "random_seed",
        "packing",
        "train_on_completions",
        "gradient_checkpointing",
        "optim",
        "lr_scheduler_type",
    ),
    "lora": ("lora_r", "lora_alpha", "lora_dropout", "use_rslora", "use_loftq", "use_dora"),
}


CATALOG_FIELDS = ("size_bytes", "low_vram", "fits_now", "source")


def kind_of(entry: dict) -> str:
    task = entry.get("task")
    if task is None:
        return "llm"
    return KIND_BY_TASK.get(task, str(task))


def _entry(entry: dict) -> Optional[ModelEntry]:
    if not opt_text(entry.get("id")):
        return None
    # The catalog fields stay null on local rows, whatever the route sends.
    return ModelEntry.from_route(entry, kind = kind_of(entry), **dict.fromkeys(CATALOG_FIELDS))


def _training_defaults(details: Any) -> Optional[dict[str, Any]]:
    config = as_dict(details).get("config")
    if not isinstance(config, dict):
        return None
    defaults = {}
    for section, keys in TRAINING_DEFAULT_KEYS.items():
        values = as_dict(config.get(section))
        for key in keys:
            if isinstance(values.get(key), (bool, int, float, str)):
                defaults[key] = values[key]
    return defaults


def _same_model_id(listed: str, asked: str) -> bool:
    # Hub ids are case-insensitive, and a GGUF may be asked for with its ":variant".
    return listed.lower() == asked.split(":", 1)[0].lower() or listed.lower() == asked.lower()


# The audio catalog's task -> the kind list_models reports and load_model takes.
AUDIO_KIND = {"tts": "tts", "music": "audio", "sep": "audio", "asr": "stt", "": "audio"}
MEDIA_KINDS = ("image", "video")
AUDIO_KINDS = ("tts", "audio", "stt")
# GpuDevice memory is GiB; catalog sizes are decimal GB.
GIB_TO_GB = 1.073741824


def free_gb(hardware: Optional[HardwareSlot]) -> Optional[float]:
    # One load lands on one GPU, so the most free memory on any one device counts, never the sum.
    free = [
        (device.vram_total_gb - device.vram_used_gb) * GIB_TO_GB
        for device in (hardware.devices if hardware else [])
        if device.vram_total_gb is not None and device.vram_used_gb is not None
    ]
    return max(free) if free else None


def fits(size_bytes: Optional[int], free: Optional[float]) -> Optional[bool]:
    if size_bytes is None or free is None:
        return None
    return size_bytes / 1e9 <= free


async def _catalog_part(path: str, label: str, others: Optional[str]) -> list:
    try:
        listing = await route_json("GET", path)
    except ToolError as error:
        hint = f" Pass kind={others} to list the others." if others else ""
        message = f"Unsloth Studio could not list the {label} catalog: {error}{hint}"
        raise ToolError(message) from None
    rows = as_dict(listing).get("models")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


async def _catalog(
    kind: Optional[str], loaded_only: bool, workflow: Optional[str]
) -> list[ModelEntry]:
    if loaded_only:
        raise ToolError("loaded_only lists resident models; use the default source.")
    if kind == "llm":
        return []
    want_media = kind in (None, *MEDIA_KINDS) and not workflow
    want_audio = kind in (None, *AUDIO_KINDS)
    both = want_media and want_audio

    async def nothing() -> list:
        return []

    media, audio, hardware = await asyncio.gather(
        _catalog_part("/api/inference/media/catalog", "image and video", '"tts"' if both else None)
        if want_media
        else nothing(),
        _catalog_part("/api/inference/audio/catalog", "audio", '"image"' if both else None)
        if want_audio
        else nothing(),
        try_json(current_caller(), SLOT_ROUTES["hardware"]),
    )
    free = free_gb(_hardware(hardware) if isinstance(hardware, dict) else None)
    entries = []
    for row in media:
        if not opt_text(row.get("id")) or row.get("kind") not in MEDIA_KINDS:
            continue
        size = integer(row.get("size_bytes"))
        low_vram = row.get("low_vram") if isinstance(row.get("low_vram"), bool) else None
        entries.append(
            ModelEntry(
                id = row["id"],
                kind = row["kind"],
                display_name = opt_text(row.get("display_name")),
                size_bytes = size,
                low_vram = low_vram,
                fits_now = fits(size, free),
                source = "catalog",
            )
        )
    for row in audio:
        # A row the catalog refuses (unsupported family, no Audio page) can't be loaded.
        if not opt_text(row.get("id")) or row.get("unsupported_reason"):
            continue
        size = integer(row.get("size_bytes"))
        entries.append(
            ModelEntry(
                id = row["id"],
                kind = AUDIO_KIND.get(row.get("task") or "", "audio"),
                display_name = opt_text(row.get("display_name")),
                audio_workflows = strings(row.get("workflows")),
                size_bytes = size,
                fits_now = fits(size, free),
                source = "catalog",
            )
        )
    return entries


async def _training_defaults_for(model: Optional[str]) -> Optional[dict[str, Any]]:
    if not model:
        return None
    details = await route_json(
        "GET", "/api/models/config/" + quote(model, safe = "/:"), hub_header = True
    )
    return _training_defaults(details)


async def list_models(
    kind: Optional[ModelKind] = None,
    loaded_only: bool = False,
    model: Optional[str] = None,
    workflow: Optional[AudioWorkflow] = None,
    source: Annotated[
        Literal["local", "catalog"],
        Field(
            description = '"local" lists downloaded and loaded models; "catalog" lists image, video '
            "and audio models Unsloth Studio can download."
        ),
    ] = "local",
) -> ModelList:
    """Models this Unsloth Studio can serve: loaded ones and those already downloaded.
    Each entry has the id to pass to load_model, its kind and whether it is loaded.
    ``kind`` is read from the model's task; a model with no task is reported as "llm", which also covers embedding models and unloaded text-to-speech models Unsloth Studio cannot classify yet, so pass the kind explicitly to load_model.
    ``loaded_only`` returns just the resident models.
    ``model`` narrows the list to that model (empty when it is not downloaded yet) and adds Unsloth Studio's training defaults for it, which also works for a Hugging Face repo id that is not downloaded (send X-Unsloth-HF-Token for a gated repo).
    ``workflow`` keeps only audio models that run that Unsloth Studio Audio page (their ``audio_workflows``), such as the ids to pass to generate_audio as ``model``.
    With source="catalog" it lists image, video and audio models Unsloth Studio can download, with size_bytes (decimal), low_vram and fits_now (whether it fits in the free memory of one GPU right now; null when unknown).
    LLM rows are not in the catalog: list downloaded LLMs with the default source, or load a Hugging Face id.
    Load a catalog row with load_model(model=id, kind=its kind).
    """
    if source == "catalog":
        models = await _catalog(kind, loaded_only, workflow)
        models = [
            entry
            for entry in models
            if (not kind or entry.kind == kind)
            and (not model or _same_model_id(entry.id, model))
            and (not workflow or workflow in (entry.audio_workflows or []))
        ]
        return ModelList(models = models, training_defaults = await _training_defaults_for(model))
    if loaded_only and kind == "llm" and not workflow:
        # Resident chat models without a scan of every model folder; it never lists image, video or STT.
        listing = await route_json("GET", "/api/inference/loaded-models")
    else:
        listing = await route_json("GET", "/v1/models")
    rows = as_dict(listing).get("data")
    models = []
    for row in rows if isinstance(rows, list) else []:
        entry = _entry(row) if isinstance(row, dict) else None
        if entry is None or (kind and entry.kind != kind) or (loaded_only and not entry.loaded):
            continue
        if model and not _same_model_id(entry.id, model):
            continue
        if workflow and workflow not in (entry.audio_workflows or []):
            continue
        models.append(entry)
    return ModelList(models = models, training_defaults = await _training_defaults_for(model))


async def load_model(
    model: str,
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> LoadResult:
    """Load a model into Unsloth Studio, downloading it first if needed, and wait until it is ready; progress is reported while it loads. ``kind`` "llm" covers chat, vision, embedding and audio models, and the "tts" and "audio" kinds list_models reports load the same way; "image" and "video" load the generation models and "stt" a speech-to-text model (downloaded first when needed). ``model`` is an id from list_models or a Hugging Face repo id; ``variant`` picks a GGUF quantization such as Q4_K_M. ``max_seq_length`` 0 or unset lets Unsloth Studio choose the context, and ``load_in_4bit`` picks 4-bit weights (both llm only). ``hf_token`` is for gated repos. Loading may unload other models to make room, for example a chat model to fit an image model; they are listed in ``evicted``."""
    caller = current_caller()
    if kind == "stt":
        return await loading.load_stt(caller, ctx, model = model, variant = variant, hf_token = hf_token)
    if kind in loading.MEDIA_ROUTES:
        return await loading.load_media(
            caller, ctx, kind = kind, model = model, variant = variant, hf_token = hf_token
        )
    return await loading.load_llm(
        caller,
        ctx,
        model = model,
        variant = variant,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
        kind = kind,
    )


async def unload_model(kind: ModelKind = "llm", model: Optional[str] = None) -> UnloadResult:
    """Unload a model to free memory. Without ``model`` the active one of that kind is unloaded; image and video have one slot each, so ``model`` is ignored there. The embedding model is managed by Unsloth Studio and is not unloaded here. ``unloaded`` is false when nothing matching was loaded."""
    if kind == "stt":
        return await loading.unload_stt(current_caller(), model)
    if kind in loading.MEDIA_ROUTES:
        return await loading.unload_media(current_caller(), kind)
    return await loading.unload_llm(current_caller(), model)


TOOLS = ((list_models, READ_ONLY), (load_model, WRITES), (unload_model, DESTRUCTIVE))
