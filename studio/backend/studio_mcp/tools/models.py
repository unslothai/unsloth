# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model discovery and residency for agents: ``list_models``, ``load_model``, ``unload_model``."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context, FastMCP

from studio_mcp import loading
from studio_mcp.caller import current_caller
from studio_mcp.outputs import LoadResult, ModelEntry, ModelList, UnloadResult
from studio_mcp.tools import DESTRUCTIVE, READ_ONLY, WRITES, integer, route_json, text

ModelKind = Literal["llm", "image", "video", "stt", "tts", "audio"]
LoadKind = Literal["llm"]

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


def kind_of(entry: dict) -> str:
    task = entry.get("task")
    if task is None:
        return "llm"
    return KIND_BY_TASK.get(task, str(task))


def _entry(entry: dict) -> Optional[ModelEntry]:
    if not text(entry.get("id")):
        return None
    workflows = entry.get("audio_workflows")
    return ModelEntry(
        id = entry["id"],
        kind = kind_of(entry),
        loaded = entry.get("loaded") is True,
        display_name = text(entry.get("display_name")),
        quant = text(entry.get("quant")),
        context_length = integer(entry.get("context_length")),
        audio_workflows = [w for w in workflows if isinstance(w, str)]
        if isinstance(workflows, list)
        else None,
    )


def _training_defaults(details: Any) -> Optional[dict[str, Any]]:
    config = details.get("config") if isinstance(details, dict) else None
    if not isinstance(config, dict):
        return None
    defaults = {}
    for section, keys in TRAINING_DEFAULT_KEYS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        for key in keys:
            if isinstance(values.get(key), (bool, int, float, str)):
                defaults[key] = values[key]
    return defaults


async def list_models(
    kind: Optional[ModelKind] = None,
    loaded_only: bool = False,
    model: Optional[str] = None,
) -> ModelList:
    """Models this Studio can serve: loaded ones and those already downloaded. Each entry has the id to pass to load_model, its kind and whether it is loaded. ``kind`` is read from the model's task; a model with no task is reported as "llm", which also covers embedding models and unloaded text-to-speech models Studio cannot classify yet, so pass the kind explicitly to load_model. ``loaded_only`` returns just the resident models. ``model`` adds Studio's training defaults for that model (send X-Unsloth-HF-Token for a gated repo)."""
    if loaded_only and kind in (None, "llm"):
        # Resident chat models without a scan of every model folder.
        listing = await route_json("GET", "/api/inference/loaded-models")
    else:
        listing = await route_json("GET", "/v1/models")
    rows = listing.get("data") if isinstance(listing, dict) else None
    models = []
    for row in rows if isinstance(rows, list) else []:
        entry = _entry(row) if isinstance(row, dict) else None
        if entry is None or (kind and entry.kind != kind) or (loaded_only and not entry.loaded):
            continue
        models.append(entry)
    defaults = None
    if model:
        details = await route_json(
            "GET", "/api/models/config/" + quote(model, safe = "/:"), hub_header = True
        )
        defaults = _training_defaults(details)
    return ModelList(models = models, training_defaults = defaults)


async def load_model(
    model: str,
    kind: LoadKind = "llm",
    variant: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> LoadResult:
    """Load a model into Studio, downloading it first if needed, and wait until it is ready; progress is reported while it loads. ``kind`` "llm" covers chat, vision, embedding and text-to-speech models. ``model`` is an id from list_models or a Hugging Face repo id; ``variant`` picks a GGUF quantization such as Q4_K_M. ``max_seq_length`` 0 or unset lets Studio choose the context. ``hf_token`` is for gated repos. Loading may unload other models to make room; they are listed in ``evicted``."""
    return await loading.load_llm(
        current_caller(),
        ctx,
        model = model,
        variant = variant,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
    )


async def unload_model(kind: LoadKind = "llm", model: Optional[str] = None) -> UnloadResult:
    """Unload a model to free memory. Without ``model`` the active one of that kind is unloaded. ``unloaded`` is false when nothing matching was loaded."""
    return await loading.unload_llm(current_caller(), model)


def register_models(mcp: FastMCP) -> None:
    mcp.tool(list_models, annotations = READ_ONLY)
    mcp.tool(load_model, annotations = WRITES)
    mcp.tool(unload_model, annotations = DESTRUCTIVE)
