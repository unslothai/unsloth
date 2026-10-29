# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model discovery and residency for agents: ``list_models``, ``load_model``, ``unload_model``."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import profile
from studio_mcp import loading, model_downloads, placement, trained_models
from studio_mcp.caller import current_caller
from studio_mcp.media import media_result
from studio_mcp.outputs import (
    DownloadRef,
    HardwareSlot,
    LoadResult,
    ModelEntry,
    ModelList,
    PlanLoadResult,
    UnloadAllResult,
    UnloadResult,
)
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

# plan_load's overall bound, above the 15 s each of its reads may take.
PLAN_S = 60

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
    Each entry has the id to pass <<load_model: to load_model | as a generate tool's ``model``>>, its kind and whether it is loaded.
    ``kind`` is read from the model's task; a model with no task is reported as "llm", which also covers embedding models and unloaded text-to-speech models Unsloth Studio cannot classify yet, <<load_model: , so pass the kind explicitly to load_model>>.
    ``loaded_only`` returns just the resident models.
    ``model`` narrows the list to that model (empty when it is not downloaded yet) and adds Unsloth Studio's training defaults for it, which also works for a Hugging Face repo id that is not downloaded (send X-Unsloth-HF-Token for a gated repo).
    ``workflow`` keeps only audio models that run that Unsloth Studio Audio page (their ``audio_workflows``), such as the ids to pass to generate_audio as ``model``.
    With source="catalog" it lists image, video and audio models Unsloth Studio can download, with size_bytes (decimal), low_vram and fits_now (whether it fits in the free memory of one GPU right now; null when unknown).
    LLM rows are not in the catalog: list downloaded LLMs with the default source, or load a Hugging Face id.
    <<load_model: Load a catalog row with load_model(model=id, kind=its kind).>>
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
    if not loaded_only and kind in (None, *AUDIO_KINDS):
        models += [
            entry
            for entry in await _downloaded_audio(kind, workflow)
            if not (model and not _same_model_id(entry.id, model))
            and not any(_same_model_id(listed.id, entry.id) for listed in models)
        ]
    return ModelList(models = models, training_defaults = await _training_defaults_for(model))


async def _downloaded_audio(kind: Optional[str], workflow: Optional[str]) -> list[ModelEntry]:
    """Downloaded audio.cpp models, which the /v1/models scan does not list. Best effort: the
    local listing must not fail because the audio catalog could not be read."""
    try:
        rows = await _catalog_part("/api/inference/audio/catalog", "audio", None)
    except ToolError:
        return []
    entries = []
    for row in rows:
        if (
            not opt_text(row.get("id"))
            or row.get("unsupported_reason")
            or row.get("downloaded") is not True
        ):
            continue
        entry = ModelEntry(
            id = row["id"],
            kind = AUDIO_KIND.get(row.get("task") or "", "audio"),
            display_name = opt_text(row.get("display_name")),
            audio_workflows = strings(row.get("workflows")),
        )
        if (kind and entry.kind != kind) or (workflow and workflow not in entry.audio_workflows):
            continue
        entries.append(entry)
    return entries


async def load_model(
    model: Optional[str] = None,
    checkpoint: Annotated[
        Optional[str],
        Field(
            description = "Load a training checkpoint instead of model: a name from "
            'list_training_runs(include_checkpoints=true), "<run folder>" or "<run folder>/checkpoint-N". '
            "It becomes the active model: chat with it by leaving chat's model unset."
        ),
    ] = None,
    export: Annotated[
        Optional[str],
        Field(
            description = "Load an export instead of model: its output name as export_model and "
            'get_job(kind="export") report it, relative to Unsloth Studio\'s exports folder. A merged or '
            'LoRA export is found when its save_directory has two levels, such as "<run>/<name>"; '
            "variant picks a GGUF export's quantization. "
            "It becomes the active model: chat with it by leaving chat's model unset."
        ),
    ] = None,
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
    trained = await trained_models.resolve_by(
        deadline,
        caller,
        model = model,
        checkpoint = checkpoint,
        export = export,
        kind = kind,
        variant = variant,
    )
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
    if trained:
        # The export listing knows the type: a GGUF export is placed by its variant, not the Hub's listing.
        variant = trained.gguf_variant or variant
        if trained.unplaced_gguf and (device is not None or gpu_layers is not None):
            raise ToolError(
                f"Pass variant (the export's quantization, such as Q4_K_M) to load {trained.name} with device or gpu_layers."
            )
        if kind == "llm" and device == "cpu" and not trained.gguf_variant:
            raise ToolError(placement.CPU_NEEDS_GGUF)
    placement.check_placement(kind, **where)
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
            hub_token = hf_token,
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
    result = await loading.load_llm(
        caller,
        ctx,
        model = trained.ref if trained else model,
        variant = variant,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
        kind = kind,
        is_lora = trained.is_lora if trained else False,
        deadline = deadline,
        **where,
    )
    # The route echoes the minted ref; the agent gets the name it asked for.
    if not trained:
        trained_models.forget_load(caller, model)
        return result
    trained_models.remember_load(caller, trained)
    trained_models.rename_job(caller, result.job_id, trained)
    return trained.renamed(result)


async def unload_model(kind: UnloadKind = "llm", model: Optional[str] = None) -> ToolResult:
    """Unload a model to free memory.
    Without ``model`` the active one of that kind is unloaded; image and video have one slot each, so ``model`` is ignored there.
    For "image", "video" and "stt", a load_model of that kind still running is stopped too (a speech-to-text load past its download is unloaded once it finishes, after this call if it takes longer than 240 s; an image or video load is ended by its load id, even one whose request has not reached Unsloth Studio yet, and never a newer load of the slot).
    kind "all" unloads every chat, speech, image, video and speech-to-text model in one call, stopping image, video and speech-to-text loads as those kinds do, and lists them under ``unloaded``.
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
    caller = current_caller()
    # A checkpoint or export is resident under its ref: handle, not the name load_model reported.
    target = trained_models.loaded_ref(caller, model)
    result = await loading.unload_llm(caller, target, kind)
    if target != model:
        result = result.model_copy(update = {"model": model})
    return media_result([], result)


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
    gpu_layers: GpuLayers = None,
    kv_cache_type: KvCacheType = None,
) -> PlanLoadResult:
    """Preview a load_model call without loading or downloading anything.
    ``variants`` lists a GGUF repo's quantizations with their size and whether each is downloaded or only partly on disk; ``download_bytes`` is what the load would still download, and ``est_gpu_bytes`` the GPU memory a downloaded GGUF chat model is estimated to take.
    ``residents`` are the models loaded now, and ``would_evict`` the ones this load would unload; pass the same ``device``, ``gpu_layers``, ``kv_cache_type``, ``alongside``, image or video options and ``hf_token`` you would pass to load_model.
    Under Keep multiple models loaded set to Off the prediction is exact when it does not depend on where Unsloth Studio places the load; unloading another kind of model (which happens only when the load reaches the GPU), and every prediction under When they fit, is an upper bound (``would_evict_exact`` false).
    Under Always nothing is unloaded and ``would_refuse`` says the load may be refused instead; unload_model(kind="all"), alongside=true or device="cpu" avoid that."""
    # A preview never hangs: each read inside is bounded, and so is the whole plan.
    plan = placement.plan_load(
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
        gpu_layers = gpu_layers,
        kv_cache_type = kv_cache_type,
    )
    try:
        return await asyncio.wait_for(plan, PLAN_S)
    except asyncio.TimeoutError:
        raise ToolError(f"Unsloth Studio did not answer within {PLAN_S} s; try plan_load again.")


async def download_model(
    model: str,
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    hf_token: Optional[str] = None,
    memory_mode: MemoryMode = None,
    speed_mode: SpeedMode = None,
    transformer_quant: TransformerQuant = None,
    text_encoder_quant: TextEncoderQuant = None,
    h3_task: H3Task = None,
    ctx: Optional[Context] = None,
) -> DownloadRef:
    """Download a model ahead of time without loading it, so a later load starts sooner.
    The download keeps running in Unsloth Studio's download manager after this returns.
    ``kind`` is the kind <<load_model: load_model takes | list_models reports>>: "llm", "tts" and "audio" fetch the repo (for a GGUF repo only ``variant``, by default the recommended one), "image" and "video" the files their load reads, and "stt" a speech-to-text model.
    Follow it with get_job(kind="download", id=<the returned id>)<<cancel:  and stop it with cancel(kind="model_download", id=<the same id>)>>.
    A speech-to-text download finishes before this returns, so it has no id.
    <<load_model: load_model waits for a running download of the same model.>>
    ``hf_token`` is for gated repos. For image and video, pass the options the later load will take: they pick the files it reads, such as a pre-cast text encoder."""
    knobs = placement.media_knobs(
        kind,
        memory_mode = memory_mode,
        speed_mode = speed_mode,
        transformer_quant = transformer_quant,
        text_encoder_quant = text_encoder_quant,
        h3_task = h3_task,
    )
    return await model_downloads.start(
        current_caller(),
        ctx,
        model = model,
        kind = kind,
        variant = variant,
        hf_token = hf_token,
        knobs = knobs,
    )


FACTS = {
    "load_model": {profile.SWAPS_MODELS},
    "unload_model": {profile.SWAPS_MODELS},
    "plan_load": {profile.PREVIEWS_SWAPS},
}

TOOLS = (
    (list_models, READ_ONLY),
    (load_model, WRITES),
    (unload_model, DESTRUCTIVE, _unload_schema()),
    (plan_load, READ_ONLY),
    (download_model, WRITES),
)
