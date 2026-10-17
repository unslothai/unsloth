# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model discovery and residency for agents: ``list_models``, ``load_model``, ``unload_model``."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context
from pydantic import Field

from studio_mcp import loading, model_downloads, trained_models
from studio_mcp.caller import current_caller
from studio_mcp.outputs import DownloadRef, LoadResult, ModelEntry, ModelList, UnloadResult
from studio_mcp.tools import DESTRUCTIVE, READ_ONLY, WRITES, as_dict, opt_text, route_json

# "tts" and "audio" are what list_models reports for audio models; they load in the llm slot.
ModelKind = Literal["llm", "image", "video", "stt", "tts", "audio"]

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
    model: Optional[str] = None,
    checkpoint: Annotated[
        Optional[str],
        Field(
            description = "Load a training checkpoint instead of model: a name from "
            'list_training_runs(include_checkpoints=true), "<run folder>" or "<run folder>/checkpoint-N".'
        ),
    ] = None,
    export: Annotated[
        Optional[str],
        Field(
            description = "Load an export instead of model: its output name as export_model and "
            'get_job(kind="export") report it, relative to Unsloth Studio\'s exports folder. A merged or '
            'LoRA export is found when its save_directory has two levels, such as "<run>/<name>"; '
            "variant picks a GGUF export's quantization."
        ),
    ] = None,
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> LoadResult:
    """Load a model into Unsloth Studio, downloading it first if needed, and wait until it is ready; progress is reported while it loads. ``kind`` "llm" covers chat, vision, embedding and audio models, and the "tts" and "audio" kinds list_models reports load the same way; "image" and "video" load the generation models and "stt" a speech-to-text model (downloaded first when needed). ``model`` is an id from list_models or a Hugging Face repo id; ``variant`` picks a GGUF quantization such as Q4_K_M. ``max_seq_length`` 0 or unset lets Unsloth Studio choose the context, and ``load_in_4bit`` picks 4-bit weights (both llm only). ``hf_token`` is for gated repos. Loading may unload other models to make room, for example a chat model to fit an image model; they are listed in ``evicted``."""
    caller = current_caller()
    trained = await trained_models.resolve(
        caller, model = model, checkpoint = checkpoint, export = export, kind = kind, variant = variant
    )
    if kind == "stt":
        return await loading.load_stt(caller, ctx, model = model, variant = variant, hf_token = hf_token)
    if kind in loading.MEDIA_ROUTES:
        return await loading.load_media(
            caller, ctx, kind = kind, model = model, variant = variant, hf_token = hf_token
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
    )
    # The route echoes the minted ref; the agent gets the name it asked for.
    return trained.renamed(result) if trained else result


async def unload_model(kind: ModelKind = "llm", model: Optional[str] = None) -> UnloadResult:
    """Unload a model to free memory. Without ``model`` the active one of that kind is unloaded; image and video have one slot each, so ``model`` is ignored there. The embedding model is managed by Unsloth Studio and is not unloaded here. ``unloaded`` is false when nothing matching was loaded."""
    if kind == "stt":
        return await loading.unload_stt(current_caller(), model)
    if kind in loading.MEDIA_ROUTES:
        return await loading.unload_media(current_caller(), kind)
    return await loading.unload_llm(current_caller(), model)


async def download_model(
    model: str,
    kind: ModelKind = "llm",
    variant: Optional[str] = None,
    hf_token: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> DownloadRef:
    """Download a model ahead of time without loading it, so a later load_model starts sooner.
    The download keeps running in Unsloth Studio's download manager after this returns.
    ``kind`` is the kind load_model takes: "llm", "tts" and "audio" fetch the repo (for a GGUF repo only ``variant``, by default the recommended one), "image" and "video" the files their load reads, and "stt" a speech-to-text model.
    Follow it with get_job(kind="download", id=<the returned id>) and stop it with cancel(kind="model_download", id=<the same id>).
    A speech-to-text download finishes before this returns, so it has no id.
    load_model waits for a running download of the same model.
    ``hf_token`` is for gated repos."""
    return await model_downloads.start(
        current_caller(), ctx, model = model, kind = kind, variant = variant, hf_token = hf_token
    )


TOOLS = (
    (list_models, READ_ONLY),
    (load_model, WRITES),
    (unload_model, DESTRUCTIVE),
    (download_model, WRITES),
)
