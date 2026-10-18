# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model discovery and residency for agents: ``list_models``, ``load_model``, ``unload_model``."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import loading, placement
from studio_mcp.caller import current_caller
from studio_mcp.media import media_result
from studio_mcp.outputs import (
    LoadResult,
    ModelEntry,
    ModelList,
    PlanLoadResult,
    UnloadAllResult,
    UnloadResult,
)
from studio_mcp.tools import DESTRUCTIVE, READ_ONLY, WRITES, as_dict, opt_text, route_json

# "tts" and "audio" are what list_models reports for audio models; they load in the llm slot.
ModelKind = Literal["llm", "image", "video", "stt", "tts", "audio"]
UnloadKind = Literal["llm", "image", "video", "stt", "tts", "audio", "all"]

MemoryMode = Annotated[
    Optional[Literal["auto", "fast", "balanced", "low_vram"]],
    Field(
        description = 'Image and video. "auto" (the default) measures free VRAM and picks how much of the model to move off the GPU; "low_vram" keeps the least on the GPU, for when other GPU work is resident; "fast" and "balanced" keep more. With Keep multiple models loaded set to When they fit, an image or video model whose transformer stays on the GPU loads beside the chat model.'
    ),
]
SpeedMode = Annotated[
    Optional[Literal["off", "eager", "default", "max"]],
    Field(
        description = "Image and video: how hard Unsloth Studio compiles the model for speed; unset uses the model's default."
    ),
]
TransformerQuant = Annotated[
    Optional[Literal["auto", "none", "off", "int8", "fp8", "nvfp4", "mxfp8"]],
    Field(
        description = 'Image only: quantize the transformer on load to save VRAM; unset or "auto" lets Unsloth Studio choose.'
    ),
]
TextEncoderQuant = Annotated[
    Optional[Literal["auto", "none", "off", "fp8", "fp8_dynamic", "int8", "nvfp4"]],
    Field(
        description = 'Image only: quantize the text encoder on load; unset or "auto" lets Unsloth Studio choose.'
    ),
]
H3Task = Annotated[
    Optional[Literal["fl2va", "ref2va"]],
    Field(
        description = 'Video, H3 models only: "fl2va" animates from a first and last frame, "ref2va" from reference images.'
    ),
]
Device = Annotated[
    Optional[Literal["auto", "cpu", "gpu"]],
    Field(
        description = 'llm, tts and audio: where the model runs. "cpu" keeps the GPU free, so a resident image or video model stays loaded; to keep the chat model loaded as well, also pass alongside=true. For kind "llm" it needs a GGUF model, and a vision GGUF still uses the GPU for its projector. "gpu" is never moved to the CPU silently; placement.device reports where the model ended up. Unset leaves the choice to Unsloth Studio.'
    ),
]
Alongside = Annotated[
    Optional[bool],
    Field(
        description = 'llm, tts and audio: keep the chat and speech models already loaded instead of replacing the active one. Needs Keep multiple models loaded set to When they fit or Always; under Off it is ignored. It never keeps image or video models. Under Always, a device="cpu" load beside a serving chat model sends it automatically. alongside_applied says whether the earlier chat model is still serving.'
    ),
]
KvCacheType = Annotated[
    Optional[Literal["f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"]],
    Field(
        description = 'GGUF chat models: the KV cache type; "q8_0" roughly halves the context memory of "f16".'
    ),
]
GpuLayers = Annotated[
    Optional[int],
    Field(
        ge = 0,
        description = "GGUF models: how many layers to put on the GPU (0 runs on the CPU); unset lets Unsloth Studio fit as many as it can.",
    ),
]

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
    if not opt_text(entry.get("id")):
        return None
    return ModelEntry.from_route(entry, kind = kind_of(entry))


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


async def list_models(
    kind: Optional[ModelKind] = None,
    loaded_only: bool = False,
    model: Optional[str] = None,
) -> ModelList:
    """Models this Unsloth Studio can serve: loaded ones and those already downloaded. Each entry has the id to pass to load_model, its kind and whether it is loaded. ``kind`` is read from the model's task; a model with no task is reported as "llm", which also covers embedding models and unloaded text-to-speech models Unsloth Studio cannot classify yet, so pass the kind explicitly to load_model. ``loaded_only`` returns just the resident models. ``model`` narrows the list to that model (empty when it is not downloaded yet) and adds Unsloth Studio's training defaults for it, which also works for a Hugging Face repo id that is not downloaded (send X-Unsloth-HF-Token for a gated repo)."""
    if loaded_only and kind == "llm":
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
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    memory_mode: MemoryMode = None,
    speed_mode: SpeedMode = None,
    transformer_quant: TransformerQuant = None,
    text_encoder_quant: TextEncoderQuant = None,
    h3_task: H3Task = None,
    max_seq_length: Optional[int] = None,
    load_in_4bit: bool = True,
    device: Device = None,
    alongside: Alongside = None,
    kv_cache_type: KvCacheType = None,
    gpu_layers: GpuLayers = None,
    hf_token: Optional[str] = None,
    wait_s: Annotated[
        Optional[int],
        Field(
            description = (
                "Seconds to wait for the load, 0 to 240 (unset: 240). "
                "Raise the client's tool timeout above 240 s, or pass 50 under Codex's 60 s default. "
                "The load keeps running in Unsloth Studio either way."
            )
        ),
    ] = None,
    ctx: Optional[Context] = None,
) -> LoadResult:
    """Load a model into Unsloth Studio, downloading it first if needed, and report progress while it loads.
    ``kind`` "llm" covers chat, vision, embedding and audio models, and the "tts" and "audio" kinds list_models reports load the same way; "image" and "video" load the generation models and "stt" a speech-to-text model (downloaded first when needed).
    ``model`` is an id from list_models or a Hugging Face repo id; ``variant`` picks a GGUF quantization such as Q4_K_M.
    ``max_seq_length`` 0 or unset lets Unsloth Studio choose the context, and ``load_in_4bit`` picks 4-bit weights (both llm only).
    ``hf_token`` is for gated repos.
    The call waits up to ``wait_s`` seconds (at most 240); set the client's tool timeout above that, or pass wait_s=50 under Codex's 60 s default.
    A load that is still running then returns ``loaded`` false, ``status`` "loading" and a ``job_id``: it keeps running in Unsloth Studio, and get_job(kind="load", id=job_id, wait_s=…) follows it to the same result.
    Calling load_model again with the same arguments while that load runs follows it instead of starting another, and so does a call (without ``variant``) for a chat model Unsloth Studio is already loading.
    Loading may unload other models to make room, for example a chat model to fit an image model; ``evicted`` lists them and ``evicted_slots`` gives each one's kind."""
    caller = current_caller()
    deadline = loading.deadline_after(wait_s)
    knobs = placement.media_knobs(
        kind,
        memory_mode = memory_mode,
        speed_mode = speed_mode,
        transformer_quant = transformer_quant,
        text_encoder_quant = text_encoder_quant,
        h3_task = h3_task,
    )
    where = {
        "device": device,
        "alongside": alongside,
        "kv_cache_type": kv_cache_type,
        "gpu_layers": gpu_layers,
    }
    placement.check_placement(kind, model, variant, **where)
    # A gated repo's GGUF listing needs this call's Hub token.
    lister = dataclasses.replace(caller, hf_token = hf_token) if hf_token else caller
    gguf = await placement.require_cpu_gguf(lister, kind, model, variant, device)
    if kind == "stt":

        async def load_stt(job: loading.LoadJob, _before: Any) -> LoadResult:
            return await loading.load_stt(
                caller, job, model = model, variant = variant, hf_token = hf_token
            )

        return await loading.start_load(
            caller,
            ctx,
            slot = kind,
            model = model,
            variant = variant,
            deadline = deadline,
            run = load_stt,
        )
    if kind in loading.MEDIA_ROUTES:
        return await loading.load_media(
            caller,
            ctx,
            kind = kind,
            model = model,
            variant = variant,
            hf_token = hf_token,
            deadline = deadline,
            knobs = knobs,
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
        deadline = deadline,
        gguf = gguf,
        **where,
    )


async def unload_model(kind: UnloadKind = "llm", model: Optional[str] = None) -> ToolResult:
    """Unload a model to free memory.
    Without ``model`` the active one of that kind is unloaded; image and video have one slot each, so ``model`` is ignored there.
    A load_model of that kind still running is stopped too (a speech-to-text load past its download finishes first, then unloads); for "llm", "tts" and "audio" without ``model``, that load is the one unloaded.
    kind "all" unloads every chat, speech, image, video and speech-to-text model in one call, stops every load_model still running, and lists them under ``unloaded``.
    The embedding model is managed by Unsloth Studio and is not unloaded here.
    ``unloaded`` is false when nothing matching was loaded."""
    if kind == "all":
        if model is not None:
            raise ToolError("unload_model(kind='all') takes no model.")
        return media_result([], await loading.unload_all(current_caller()))
    if kind == "stt":
        return media_result([], await loading.unload_stt(current_caller(), model))
    if kind in loading.MEDIA_ROUTES:
        return media_result([], await loading.unload_media(current_caller(), kind))
    return media_result([], await loading.unload_llm(current_caller(), model, kind))


def _unload_schema() -> dict[str, Any]:
    # One object for both answers: ``unloaded`` is a bool for one kind and the list of models for "all".
    one, every = UnloadResult.model_json_schema(), UnloadAllResult.model_json_schema()
    unloaded = {"anyOf": [one["properties"]["unloaded"], every["properties"]["unloaded"]]}
    return {
        "type": "object",
        "$defs": every.get("$defs", {}),
        "properties": {**one["properties"], "unloaded": unloaded},
        "required": ["kind", "unloaded"],
    }


async def plan_load(
    model: str,
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    device: Device = None,
    alongside: Alongside = None,
    memory_mode: MemoryMode = None,
    speed_mode: SpeedMode = None,
    transformer_quant: TransformerQuant = None,
    text_encoder_quant: TextEncoderQuant = None,
    h3_task: H3Task = None,
    hf_token: Optional[str] = None,
) -> PlanLoadResult:
    """Preview a load_model call without loading or downloading anything.
    ``variants`` lists a GGUF repo's quantizations with their size and whether each is downloaded or only partly on disk; ``download_bytes`` is what the load would still download, and ``est_gpu_bytes`` the GPU memory a downloaded GGUF chat model is estimated to take.
    ``residents`` are the models loaded now, and ``would_evict`` the ones this load would unload; pass the same ``device``, ``alongside``, image or video options and ``hf_token`` you would pass to load_model.
    Under Keep multiple models loaded set to Off the prediction is exact; under When they fit it is an upper bound (``would_evict_exact`` false), since a model that fits stays.
    Under Always nothing is unloaded and ``would_refuse`` says the load may be refused instead; unload_model(kind="all"), alongside=true or device="cpu" avoid that."""
    return await placement.plan_load(
        current_caller(),
        model = model,
        kind = kind,
        variant = variant,
        device = device,
        alongside = alongside,
        knobs = placement.media_knobs(
            kind,
            memory_mode = memory_mode,
            speed_mode = speed_mode,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
            h3_task = h3_task,
        ),
        hf_token = hf_token,
    )


TOOLS = (
    (list_models, READ_ONLY),
    (load_model, WRITES),
    (unload_model, DESTRUCTIVE, _unload_schema()),
    (plan_load, READ_ONLY),
)
