# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training tools: ``start_training`` for LLMs and image LoRAs, and ``list_training_runs``."""

from __future__ import annotations

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


def _checked_config(config: dict[str, Any], kind: str) -> None:
    """Validate the config the way the route will, so a mistake is reported before any GPU work."""
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest, TrainingStartRequest

    if kind == "llm" and "format_type" in config and config["format_type"] not in FORMAT_TYPES:
        raise ToolError(f"format_type must be one of {', '.join(FORMAT_TYPES)}.")
    model = TrainingStartRequest if kind == "llm" else DiffusionTrainingStartRequest
    try:
        model.model_validate(config)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:10]
        )
        raise ToolError(f"Invalid training config: {problems}") from None


async def start_training(
    config: dict[str, Any],
    kind: Literal["llm", "diffusion"] = "llm",
    validate_only: bool = False,
) -> TrainingStarted:
    """Start a training job and return at once; follow it with studio_status. This starts real GPU work, so build the config first and check it with ``validate_only`` true, which starts nothing, for example {"kind": "llm", "config": {...}, "validate_only": true}.

    kind "llm": ``config`` is the Unsloth Studio training request. The minimum is {"model_name": a Hugging Face model repo id (not a GGUF; it is downloaded when training starts), "training_type": "LoRA/QLoRA" | "Full Finetuning" | "Continued Pretraining", "format_type": "auto" | "alpaca" | "chatml" | "sharegpt" | "conversational" | "raw", "hf_dataset": a Hugging Face dataset repo id}. Common options: max_steps or num_epochs, learning_rate (a string such as "2e-4"), batch_size, gradient_accumulation_steps, max_seq_length, load_in_4bit, lora_r, lora_alpha, train_split. list_models with ``model`` returns that model's defaults to start from, and datasets(action="check_format") detects the dataset's format: use it when it is one of the format_type values above, otherwise "auto" (image and audio datasets included). hf_token for a gated model goes in the config.

    kind "diffusion": an image LoRA; ``config`` is the image training request, and ``data_dir`` names an image dataset already in Unsloth Studio (there is no upload tool).

    Unsloth Studio refuses while another training run or an API inference request is active. The result's ``start_request_id`` cancels a start that has not begun (cancel kind "training_start"); ``job_id`` stops a running job (cancel kind "training")."""
    config = dict(config)
    if kind == "llm" and not config.get("start_request_id"):
        config["start_request_id"] = f"mcp-{secrets.token_hex(8)}"
    _checked_config(config, kind)
    if validate_only:
        return TrainingStarted(status = "valid", message = "The config is valid; nothing was started.")
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
