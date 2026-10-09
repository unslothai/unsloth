# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training tools: ``start_training`` for LLMs and image LoRAs, and ``list_training_runs``."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from studio_mcp import checkpoints as named
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_status_field
from studio_mcp.outputs import (
    CheckpointName,
    TrainingRun,
    TrainingRunDetail,
    TrainingRuns,
    TrainingStarted,
)
from studio_mcp.tools import READ_ONLY, WRITES, integer, number, opt_text, route_json


TRAINING_ROUTES = {"llm": "/api/train/start", "diffusion": "/api/train/diffusion/start"}


async def start_training(
    config: dict[str, Any], kind: Literal["llm", "diffusion"] = "llm"
) -> TrainingStarted:
    """Start a training job and return at once; follow it with studio_status. kind "llm": ``config`` is a TrainingStartRequest, the same one the Unsloth Studio UI sends (list_models with ``model`` gives its training defaults; hf_token for a gated model goes in the config). kind "diffusion": an image LoRA, where ``config`` is a DiffusionTrainingStartRequest and ``data_dir`` names an image dataset already in Unsloth Studio (there is no upload tool). Unsloth Studio refuses while another training run or an API inference request is active."""
    payload = await route_json("POST", TRAINING_ROUTES[kind], json_body = config)
    # The route refuses some starts with a 200 and status "error".
    raise_for_status_field(payload)
    if not isinstance(payload, dict) or not opt_text(payload.get("job_id")):
        raise ToolError("Unsloth Studio did not start the training job")
    return TrainingStarted(
        job_id = payload["job_id"],
        status = opt_text(payload.get("status")) or "queued",
        message = opt_text(payload.get("message")),
    )


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
        "model_name": opt_text(row.get("model_name")),
        "dataset_name": opt_text(row.get("dataset_name")),
        "display_name": opt_text(row.get("display_name")),
        "started_at": opt_text(row.get("started_at")),
        "ended_at": opt_text(row.get("ended_at")),
        "final_step": integer(row.get("final_step")),
        "final_loss": number(row.get("final_loss")),
        "can_resume": row.get("can_resume") is True,
        "error_message": opt_text(row.get("error_message")),
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
    listing = listing if isinstance(listing, dict) else {}
    runs = [
        TrainingRun(**_run_fields(row, folders_by_ref))
        for row in listing.get("runs") or []
        if isinstance(row, dict) and opt_text(row.get("id"))
    ]
    detail = None
    if run_id:
        found = await route_json(
            "GET", f"/api/train/runs/{quote(run_id, safe = '')}", caller = caller
        )
        found = found if isinstance(found, dict) else {}
        row = found.get("run") if isinstance(found.get("run"), dict) else {}
        config = found.get("config") if isinstance(found.get("config"), dict) else {}
        metrics = found.get("metrics") if isinstance(found.get("metrics"), dict) else {}
        detail = TrainingRunDetail(
            **_run_fields({"id": run_id, **row}, folders_by_ref),
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
        checkpoints = [
            CheckpointName(
                run = c.run,
                name = c.name,
                loss = c.loss,
                base_model = c.base_model,
                peft_type = c.peft_type,
                lora_rank = c.lora_rank,
                is_quantized = c.is_quantized,
            )
            for c in checkpoints
        ]
        if include_checkpoints
        else None,
    )


def register_training(mcp: FastMCP) -> None:
    mcp.tool(start_training, annotations = WRITES)
    mcp.tool(list_training_runs, annotations = READ_ONLY)
