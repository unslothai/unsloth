# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training nodes: LoRA fine-tuning, export and chatting with a checkpoint, through the routes the MCP tools use."""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any, Optional

from core.flows.nodes._common import FLOW_CHAT_CONTEXT, as_text, poll_job, strip_think
from core.flows.nodes.data import recipe_artifact
from core.flows.registry import FlowNodeError, NodeContext, handler

POLL_S = 5.0
# How long a start refused as "already active" keeps retrying before it reports the other run.
BUSY_WAIT_S = 180.0
TRAINING_TYPES = {"lora": "LoRA/QLoRA", "full": "Full Finetuning"}
# The same flags the Train page sets for each kind of model.
TASK_FLAGS = {
    "vision": {"is_dataset_image": True},
    "audio": {"is_dataset_audio": True},
    "embedding": {"is_embedding": True},
}
# Optional settings sent only when filled in, so a flow that leaves them blank starts the same run as before.
OPTIONAL_INTS = (
    "lora_alpha",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "vision_image_size",
    "dataset_slice_end",
)
EXPORT_QUANTS = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0", "f16": "F16"}


@handler("train.finetune")
async def finetune(ctx: NodeContext) -> dict:
    from studio_mcp.tools.jobs import TRAINING_STATUS, _run_folder
    from studio_mcp.tools.status import _training

    dataset = ctx.inputs["dataset"]
    if not isinstance(dataset, dict) or dataset.get("kind") != "dataset":
        raise FlowNodeError("Fine-tune needs a dataset")
    config: dict[str, Any] = {
        "model_name": str(ctx.params.get("base_model") or "").strip(),
        "training_type": TRAINING_TYPES.get(str(ctx.params.get("training_type")), "LoRA/QLoRA"),
        "format_type": str(ctx.params.get("format_type") or "auto"),
        "max_steps": int(ctx.params.get("max_steps") or 60),
        "learning_rate": str(ctx.params.get("learning_rate") or "2e-4"),
        "lora_r": int(ctx.params.get("lora_r") or 16),
        "max_seq_length": int(ctx.params.get("max_seq_length") or 2048),
        "random_seed": ctx.seed % (2**31),
        "start_request_id": f"flow-{secrets.token_hex(8)}",
    }
    if not config["model_name"]:
        raise FlowNodeError("Name the base model to fine-tune")
    config.update(TASK_FLAGS.get(str(ctx.params.get("task") or "text"), {}))
    if ctx.params.get("num_epochs") not in (None, ""):
        config["num_epochs"] = int(ctx.params["num_epochs"])
        config["max_steps"] = None
    for key in OPTIONAL_INTS:
        if ctx.params.get(key) not in (None, ""):
            config[key] = int(ctx.params[key])
    if ctx.params.get("dataset_streaming"):
        config["dataset_streaming"] = True
    # Sent only when off, so flows saved before this setting start exactly as they did.
    if ctx.params.get("load_in_4bit") is False:
        config["load_in_4bit"] = False
    if ctx.params.get("train_on_completions"):
        config["train_on_completions"] = True
    if ctx.params.get("weight_decay") not in (None, ""):
        config["weight_decay"] = float(ctx.params["weight_decay"])
    mapping = ctx.params.get("custom_format_mapping")
    if isinstance(mapping, dict) and mapping:
        config["custom_format_mapping"] = mapping
    if dataset.get("source") == "hf":
        config["hf_dataset"] = dataset["ref"]
        if dataset.get("subset"):
            config["subset"] = dataset["subset"]
        if dataset.get("split"):
            config["train_split"] = dataset["split"]
    elif dataset.get("source") == "local":
        config["local_datasets"] = [dataset["ref"]]
    elif dataset.get("source") == "recipe" and recipe_artifact(dataset.get("ref")):
        config["local_datasets"] = [dataset["ref"]]
    elif dataset.get("source") == "recipe":
        # Flows saved before recipe datasets carried their parquet path name the job instead.
        from core.flows.nodes.data import recipe_rows, upload_rows

        rows = await recipe_rows(ctx, str(dataset["ref"]))
        if not rows:
            raise FlowNodeError("The Data Recipe produced no rows to train on")
        saved = await upload_rows(ctx, rows, f"recipe-{dataset['ref']}")
        config["local_datasets"] = [saved["ref"]]
    else:
        raise FlowNodeError("Fine-tune takes a Hub dataset, a Data Recipe or Rows to Dataset")
    waited = 0.0
    while True:
        started = await ctx.call("POST", "/api/train/start", json = config)
        if not isinstance(started, dict) or started.get("status") != "error":
            break
        # A refused start answers 200 with the *running* job's id; polling that would report its checkpoint as ours.
        # A run that just finished keeps the trainer busy for a few seconds while it shuts down, so wait that out.
        if "already active" not in str(started.get("error") or "") or waited >= BUSY_WAIT_S:
            raise FlowNodeError(
                str(started.get("message") or started.get("error") or "Training did not start")
            )
        ctx.progress(phase = "Waiting for the previous training run to finish")
        await asyncio.sleep(POLL_S)
        waited += POLL_S
        ctx.check_cancelled()
    job_id = started.get("job_id") if isinstance(started, dict) else None
    if not job_id:
        raise FlowNodeError("Unsloth Studio did not start training")

    async def cancel() -> None:
        await ctx.call("POST", "/api/train/stop", json = {"save": True, "expected_job_id": job_id})

    def parse(payload: Any) -> tuple[str, Optional[dict], Optional[str]]:
        slot = _training(payload if isinstance(payload, dict) else {})
        if slot.job_id and slot.job_id != job_id:
            raise FlowNodeError("Another training run replaced this one")
        status = TRAINING_STATUS.get(slot.phase, "running")
        progress = None
        if slot.progress_percent is not None:
            progress = {
                "fraction": slot.progress_percent / 100.0,
                "phase": slot.phase,
                "step": slot.step,
                "total": slot.total_steps,
            }
        return status, progress, slot.error or f"Training {status}"

    payload = await poll_job(
        ctx,
        status = lambda: ctx.call("GET", "/api/train/status"),
        parse = parse,
        cancel = cancel,
        interval = POLL_S,
    )
    slot = _training(payload if isinstance(payload, dict) else {})
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    run, _ = await _run_folder(ctx.caller, details.get("output_dir"))
    if not run:
        raise FlowNodeError("Training finished but its run folder was not found")
    metrics = {"loss": slot.loss, "steps": slot.step, "job_id": job_id}
    return {
        "checkpoint": {
            "kind": "checkpoint",
            "name": run,
            "base_model": config["model_name"],
        },
        "metrics": metrics,
    }


@handler("train.export")
async def export(ctx: NodeContext) -> dict:
    from studio_mcp import checkpoints

    checkpoint = ctx.inputs["checkpoint"]
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if not name:
        raise FlowNodeError("Export needs a checkpoint")
    status = await ctx.call("GET", "/api/export/status")
    if isinstance(status, dict) and status.get("is_export_active"):
        raise FlowNodeError("Another export is running; try again when it ends", status = 409)
    try:
        found = await checkpoints.resolve(ctx.caller, name)
    except Exception as exc:
        raise FlowNodeError(str(exc)) from None
    fmt = str(ctx.params.get("format") or "gguf")
    save_directory = f"flows-{name.replace('/', '-')}-{fmt}-{int(time.time())}"
    op = f"flow-{secrets.token_hex(6)}"

    async def cancel() -> None:
        for step in ("load", "export"):
            try:
                await ctx.call(
                    "POST", "/api/export/cancel", json = {"expected_op_id": f"{op}-{step}"}
                )
            except FlowNodeError:
                pass

    ctx.on_cancel(cancel)
    loaded = await ctx.call(
        "POST",
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": found.path, "max_seq_length": 2048, "op_id": f"{op}-load"},
    )
    if isinstance(loaded, dict) and loaded.get("success") is False:
        raise FlowNodeError(loaded.get("message") or "Could not load the checkpoint for export")
    body: dict[str, Any] = {
        "save_directory": save_directory,
        "push_to_hub": False,
        "expected_checkpoint": found.path,
        "op_id": f"{op}-export",
    }
    if fmt == "gguf":
        body["quantization_method"] = EXPORT_QUANTS.get(str(ctx.params.get("quant")), "Q4_K_M")
    result = await ctx.call("POST", f"/api/export/export/{fmt}", json = body)
    if isinstance(result, dict) and result.get("success") is False:
        raise FlowNodeError(result.get("message") or "The export failed")
    return {
        "model": {
            "kind": "model",
            "model_kind": "chat",
            "repo": save_directory,
            "variant": body.get("quantization_method"),
        }
    }


@handler("train.chat_checkpoint")
async def chat_checkpoint(ctx: NodeContext) -> dict:
    from studio_mcp import loading, trained_models

    checkpoint = ctx.inputs["checkpoint"]
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if not name:
        raise FlowNodeError("Chat With Checkpoint needs a checkpoint")
    deadline = time.monotonic() + 3600
    ctx.set_status("loading_model", f"Loading {name}")
    try:
        trained = await trained_models.resolve_by(
            deadline, ctx.caller, model = None, checkpoint = name, export = None, kind = "llm", variant = None
        )
        await loading.load_llm(
            ctx.caller,
            None,
            model = trained.ref,
            variant = None,
            max_seq_length = FLOW_CHAT_CONTEXT,
            load_in_4bit = True,
            hf_token = None,
            kind = "llm",
            is_lora = trained.is_lora,
            deadline = deadline,
        )
    except FlowNodeError:
        raise
    except Exception as exc:
        raise FlowNodeError(f"Could not load {name}: {exc}") from None
    ctx.resident.clear()
    ctx.set_status("running")
    prompt = as_text(ctx.inputs.get("prompt")) or str(ctx.params.get("prompt") or "")
    if not prompt.strip():
        raise FlowNodeError("The prompt is empty")
    text = await ctx.llm_text(
        model = None,
        system = "",
        prompt = prompt,
        max_tokens = int(ctx.params.get("max_tokens") or 512),
        seed = ctx.seed,
    )
    return {"text": strip_think(text)}
