# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training tools: ``start_training`` for LLMs and image LoRAs, and ``list_training_runs``."""

from __future__ import annotations

import asyncio
import re
import secrets
from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError

from studio_mcp import checkpoints as named
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_status_field
from studio_mcp.forward import forward
from studio_mcp.outputs import (
    CheckpointName,
    TrainingRun,
    TrainingRunDetail,
    TrainingRuns,
    TrainingStarted,
)
from studio_mcp.tools import READ_ONLY, WRITES, as_dict, integer, opt_text, route_json


TRAINING_ROUTES = {"llm": "/api/train/start", "diffusion": "/api/train/diffusion/start"}
# The dataset formats the trainer knows; the route itself accepts any string and fails later.
FORMAT_TYPES = ("auto", "alpaca", "chatml", "sharegpt", "conversational", "raw")


def _checked_config(config: dict[str, Any], kind: str) -> Any:
    """Validate the config the way the route will, so a mistake is reported before any GPU work."""
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest, TrainingStartRequest

    if kind == "llm" and "format_type" in config and config["format_type"] not in FORMAT_TYPES:
        raise ToolError(f"format_type must be one of {', '.join(FORMAT_TYPES)}.")
    model = TrainingStartRequest if kind == "llm" else DiffusionTrainingStartRequest
    try:
        return model.model_validate(config)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:10]
        )
        raise ToolError(f"Invalid training config: {problems}") from None


TRAINABLE_TTS = (
    "Trainable text-to-speech models: unsloth/orpheus-3b-0.1-ft, sesame/csm-1b, "
    "unsloth/Spark-TTS-0.5B, unsloth/Llama-OuteTTS-1.0-1B."
)
# Never echoed by validate_only: secrets, and the host paths the request validators resolve refs into.
HIDDEN_CONFIG_KEYS = {
    "llm": frozenset(
        {
            "hf_token",
            "wandb_token",
            "s3_config",
            "approved_remote_code_fingerprint",
            "model_local_path",
            "model_snapshot_path",
            "dataset_local_path",
            "dataset_snapshot_path",
            "local_datasets",
            "local_eval_datasets",
            "tensorboard_dir",
            "resume_from_checkpoint",
            "model_subfolder",
        }
    ),
    "diffusion": frozenset(
        {"hf_token", "data_dir", "output_dir", "cond_cache_dir", "resume_from_checkpoint"}
    ),
}


def _hidden_key(key: str, kind: str) -> bool:
    return (
        key in HIDDEN_CONFIG_KEYS[kind]
        or key.endswith(("_path", "_dir", "_token"))
        or "secret" in key
    )


def _shown_scalar(value: Any) -> bool:
    if not isinstance(value, str):
        return isinstance(value, (bool, int, float))
    text = value.strip()
    # A host path or a ref: handle in any field, such as a model_name given as a local folder.
    return not (
        text.startswith(("/", "\\", "~", "ref:", "file:"))
        or (len(text) > 2 and text[1] == ":" and text[2] in "\\/")
    )


def _shown_config(validated: Any, kind: str) -> dict[str, Any]:
    """The config as the route will read it: scalars, lists of scalars and flat dicts, no paths or secrets."""
    shown: dict[str, Any] = {}
    for key, value in validated.model_dump(mode = "json", exclude_none = True).items():
        if _hidden_key(key, kind):
            continue
        if isinstance(value, list) and all(_shown_scalar(v) for v in value):
            shown[key] = value
        elif isinstance(value, dict) and all(
            _shown_scalar(v) and not _hidden_key(str(k), kind) for k, v in value.items()
        ):
            shown[key] = value
        elif _shown_scalar(value):
            shown[key] = value
    return shown


async def _checked_trainable(validated: Any, caller: Caller) -> tuple[Any, Optional[str]]:
    """Refuse a model the trainer can't train; a codec text-to-speech model gets is_dataset_audio."""
    from core.inference.audio_cpp_models import AUDIO_CPP_AUDIO_TYPES, looks_like_audio_cpp
    from core.inference.native_audio import NATIVE_AUDIO_MODEL_IDS
    from utils.audio_tokens import NATIVE_OUTPUT_AUDIO_TYPES, TTS_AUDIO_TYPES

    name = str(validated.model_name or "").strip()
    leaf = re.split(r"[\\/]", name.rstrip("/\\"))[-1].lower()
    # audio.cpp rows are GGUF folders too; their refusal says more.
    if looks_like_audio_cpp(name):
        raise ToolError(
            f"{name} runs on audio.cpp, which Unsloth Studio can run but not train. {TRAINABLE_TTS}"
        )
    if validated.model_format == "gguf" or leaf.endswith(("-gguf", ".gguf")):
        raise ToolError(f"GGUF models are inference-only and cannot be trained. {TRAINABLE_TTS}")
    audio_type = NATIVE_AUDIO_MODEL_IDS.get(name.lower())
    if audio_type is None:
        # Imported here so a test can patch the module attribute; any failure is no verdict.
        from utils.models import model_config
        try:
            audio_type, _known = await asyncio.to_thread(
                model_config.detect_audio_type_checked,
                name,
                hf_token = validated.hf_token or caller.hf_token,
            )
        except Exception:
            audio_type = None
    if audio_type in AUDIO_CPP_AUDIO_TYPES:
        raise ToolError(
            f"{name} runs on audio.cpp, which Unsloth Studio can run but not train. {TRAINABLE_TTS}"
        )
    if audio_type in NATIVE_OUTPUT_AUDIO_TYPES:
        raise ToolError(
            f"{name} is a text-to-speech model Unsloth Studio can run but not train. {TRAINABLE_TTS}"
        )
    if audio_type in TTS_AUDIO_TYPES and not validated.is_dataset_audio:
        return validated.model_copy(update = {"is_dataset_audio": True}), (
            f"{name} is a text-to-speech model, so is_dataset_audio is set to true in the returned "
            "config; start training with that config."
        )
    return validated, None


async def start_training(
    config: dict[str, Any],
    kind: Literal["llm", "diffusion"] = "llm",
    validate_only: bool = False,
) -> TrainingStarted:
    """Start a training job and return at once.
    This starts real GPU work, so build the config first and check it with ``validate_only`` true, which starts nothing and returns the resolved config without paths or secrets, for example {"kind": "llm", "config": {...}, "validate_only": true}.
    Training keeps running after this returns; follow it with get_job(kind="training", wait_s=50) (kind "diffusion_training" for kind "diffusion"), and an export with get_job(kind="export", wait_s=50).

    kind "llm": ``config`` is the Unsloth Studio training request.
    The minimum is {"model_name": a Hugging Face model repo id (not a GGUF; it is downloaded when training starts), "training_type": "LoRA/QLoRA" | "Full Finetuning" | "Continued Pretraining", "format_type": "auto" | "alpaca" | "chatml" | "sharegpt" | "conversational" | "raw", "hf_dataset": a Hugging Face dataset repo id}.
    Instead of hf_dataset, "local_datasets" takes the ref values datasets(action="upload") and datasets(action="list") return.
    Common options: max_steps or num_epochs, learning_rate (a string such as "2e-4"), batch_size, gradient_accumulation_steps, max_seq_length, load_in_4bit, lora_r, lora_alpha, train_split.
    list_models with ``model`` returns that model's defaults to start from, and datasets(action="check_format") detects the dataset's format: use it when it is one of the format_type values above, otherwise "auto" (image and audio datasets included).
    hf_token for a gated model goes in the config.

    Text-to-speech: Unsloth Studio trains unsloth/orpheus-3b-0.1-ft, sesame/csm-1b, unsloth/Spark-TTS-0.5B and unsloth/Llama-OuteTTS-1.0-1B, on a CUDA GPU only (not on Apple Silicon).
    The dataset needs an audio column ("audio" or "speech") and a text column ("text", "sentence", "transcript" or "transcription"), plus an optional speaker column ("source" or "speaker_id"); name other columns with "custom_format_mapping": {"<column>": "audio" | "text" | "speaker_id"}.
    Set "is_dataset_audio": true and "format_type": "auto"; validate_only sets is_dataset_audio for these models and refuses GGUF, audio.cpp and other text-to-speech models it can't train.
    GGUF and audio.cpp text-to-speech models, such as Qwen3-TTS and Breeze, can't be trained.
    An audio dataset comes in through hf_dataset; uploads take CSV, JSON, JSONL and Parquet files only.

    kind "diffusion": an image LoRA; ``config`` is the image training request, and ``data_dir`` names an image dataset in Unsloth Studio, such as the name datasets(action="upload_images") returns.

    One GPU: Unsloth Studio refuses (409) while another training run or an API inference request is in flight, so wait for it to finish.
    Starting training unloads the image and video models, and unloads chat, text-to-speech and speech-to-text models only when they don't fit beside it (speech-to-text first).
    When it finishes, load the result with load_model(checkpoint=…), or export_model it and load that with load_model(export=…), then chat or generate_audio with it.
    The result's ``start_request_id`` cancels a start that has not begun (cancel kind "training_start"); ``job_id`` stops a running job (cancel kind "training", or "diffusion_training" for kind "diffusion", whose progress is studio_status's ``diffusion_training``).
    """
    config = dict(config)
    # validate_only echoes the config, and an id it made up would cancel nothing.
    if kind == "llm" and not validate_only and not config.get("start_request_id"):
        config["start_request_id"] = f"mcp-{secrets.token_hex(8)}"
    validated = _checked_config(config, kind)
    if validate_only:
        note = None
        if kind == "llm":
            validated, note = await _checked_trainable(validated, current_caller())
        return TrainingStarted(
            status = "valid",
            message = " ".join(filter(None, ("The config is valid; nothing was started.", note))),
            config = _shown_config(validated, kind),
        )
    start_request_id = opt_text(config.get("start_request_id"))
    try:
        payload = await route_json("POST", TRAINING_ROUTES[kind], json_body = config)
        # The route refuses some starts with a 200 and status "error".
        raise_for_status_field(payload)
    except ToolError:
        if start_request_id:
            await _acknowledge_start(start_request_id)
        raise
    if not isinstance(payload, dict) or not opt_text(payload.get("job_id")):
        raise ToolError("Unsloth Studio did not start the training job")
    return TrainingStarted(
        job_id = payload["job_id"],
        status = opt_text(payload.get("status")) or "queued",
        message = opt_text(payload.get("message")),
        start_request_id = start_request_id,
    )


async def _acknowledge_start(start_request_id: str) -> None:
    # A refused start stays the training status until acknowledged, hiding the run that is (or was) going; the
    # training page acknowledges its own refusals the same way. Best effort: the refusal is what the agent needs.
    try:
        await forward(
            current_caller(),
            "POST",
            f"/api/train/start-requests/{quote(start_request_id, safe = '')}/acknowledge",
        )
    except Exception:
        pass


# Scalars from a run's saved config worth showing; nothing that names a file or a folder.
RUN_CONFIG_KEYS = (
    "model_name",
    "dataset",
    "training_type",
    "max_seq_length",
    "num_epochs",
    "max_steps",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "warmup_ratio",
    "weight_decay",
    "random_seed",
    "load_in_4bit",
    "packing",
    "train_on_completions",
    "optim",
    "lr_scheduler_type",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "use_rslora",
)


def _run_fields(row: dict, folders_by_ref: dict[str, str]) -> dict[str, Any]:
    output_ref = opt_text(row.get("output_dir"))
    return {
        "id": row["id"],
        "status": opt_text(row.get("status")) or "unknown",
        "run_folder": folders_by_ref.get(output_ref) if output_ref else None,
    }


async def _folders_by_ref(caller: Caller) -> tuple[list[named.Checkpoint], dict[str, str]]:
    """Run folders keyed by the opaque reference the runs route gives an API key for output_dir."""
    from hub.utils.host_paths import cache_reference

    checkpoints, folders = await named.list_checkpoints(caller)
    return checkpoints, {cache_reference(path): folder for folder, path in folders.items()}


def _numbers(values: Any, kind: type) -> Optional[list]:
    if not isinstance(values, list):
        return None
    return [kind(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


async def list_training_runs(
    run_id: Optional[str] = None,
    include_checkpoints: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> TrainingRuns:
    """Training runs, newest first, with each run's folder; with ``run_id``, that run's saved settings and loss curve too. ``include_checkpoints`` lists every checkpoint by the name export_model takes: "<run folder>" for the final weights or "<run folder>/checkpoint-N". ``limit`` is 1 to 200."""
    caller = current_caller()
    checkpoints, folders_by_ref = await _folders_by_ref(caller)
    limit, offset = max(1, min(limit, 200)), max(0, offset)
    listing = await route_json(
        "GET", "/api/train/runs", caller = caller, params = {"limit": limit, "offset": offset}
    )
    listing = as_dict(listing)
    runs = [
        TrainingRun.from_route(row, **_run_fields(row, folders_by_ref))
        for row in listing.get("runs") or []
        if isinstance(row, dict) and opt_text(row.get("id"))
    ]
    detail = None
    if run_id:
        found = as_dict(
            await route_json("GET", f"/api/train/runs/{quote(run_id, safe = '')}", caller = caller)
        )
        row = {"id": run_id, **as_dict(found.get("run"))}
        config, metrics = as_dict(found.get("config")), as_dict(found.get("metrics"))
        detail = TrainingRunDetail.from_route(
            row,
            **_run_fields(row, folders_by_ref),
            config = {
                k: config[k]
                for k in RUN_CONFIG_KEYS
                if isinstance(config.get(k), (bool, int, float, str))
            },
            step_history = _numbers(metrics.get("step_history"), int),
            loss_history = _numbers(metrics.get("loss_history"), float),
        )
    return TrainingRuns(
        runs = runs,
        total = integer(listing.get("total")),
        run = detail,
        checkpoints = [CheckpointName.model_validate(c, from_attributes = True) for c in checkpoints]
        if include_checkpoints
        else None,
    )


TOOLS = ((start_training, WRITES), (list_training_runs, READ_ONLY))
