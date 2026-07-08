# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio training (start / status / stop).

Training runs in Studio's existing subprocess backend, so ``train_start``
returns immediately with a job id; poll ``train_status`` and stop early with
``train_stop``. This mirrors the Studio UI / ``POST /api/train/*`` exactly.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from loggers import get_logger
from mcp_server.auth import resolve_hf_token, resolve_secret

logger = get_logger(__name__)

GROUP = "train"

_STATEFUL = {"destructiveHint": False}


def _request_to_training_kwargs(request: Any, subject: str) -> dict[str, Any]:
    """Convert a validated ``TrainingStartRequest`` into backend kwargs.

    Mirrors the mapping in ``routes/training.py`` (start_training route) so the
    MCP server drives the worker subprocess with the exact same contract as the
    Studio UI. Resume, S3 sources and auto GPU selection are intentionally out
    of scope for the MCP surface.
    """
    gradient_checkpointing = request.gradient_checkpointing
    if gradient_checkpointing:
        gradient_checkpointing = gradient_checkpointing.strip()
    if not gradient_checkpointing:
        gradient_checkpointing = "unsloth"

    return {
        "model_name": request.model_name,
        "project_name": request.project_name,
        "training_type": request.training_type,
        "hf_token": request.hf_token or "",
        "load_in_4bit": request.load_in_4bit,
        "max_seq_length": request.max_seq_length,
        "vision_image_size": request.vision_image_size,
        "hf_dataset": request.hf_dataset or "",
        "local_datasets": request.local_datasets,
        "local_eval_datasets": request.local_eval_datasets,
        "format_type": request.format_type,
        "subset": request.subset,
        "train_split": request.train_split,
        "dataset_streaming": request.dataset_streaming,
        "eval_split": request.eval_split,
        "eval_steps": request.eval_steps,
        "dataset_slice_start": request.dataset_slice_start,
        "dataset_slice_end": request.dataset_slice_end,
        "custom_format_mapping": request.custom_format_mapping,
        "num_epochs": request.num_epochs,
        "learning_rate": request.learning_rate,
        "embedding_learning_rate": request.embedding_learning_rate,
        "batch_size": request.batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "warmup_steps": request.warmup_steps,
        "warmup_ratio": request.warmup_ratio,
        "max_steps": request.max_steps,
        "save_steps": request.save_steps,
        "weight_decay": request.weight_decay,
        "max_grad_norm": request.max_grad_norm,
        "max_grad_value": request.max_grad_value,
        "max_grad_leaf_norm": request.max_grad_leaf_norm,
        "cast_norm_output_to_input_dtype": request.cast_norm_output_to_input_dtype,
        "random_seed": request.random_seed,
        "packing": request.packing,
        "optim": request.optim,
        "lr_scheduler_type": request.lr_scheduler_type,
        "use_lora": request.use_lora,
        "lora_r": request.lora_r,
        "lora_alpha": request.lora_alpha,
        "lora_dropout": request.lora_dropout,
        "target_modules": request.target_modules if request.target_modules else None,
        "gradient_checkpointing": gradient_checkpointing,
        "use_rslora": request.use_rslora,
        "use_loftq": request.use_loftq,
        "train_on_completions": request.train_on_completions,
        "finetune_vision_layers": request.finetune_vision_layers,
        "finetune_language_layers": request.finetune_language_layers,
        "finetune_attention_modules": request.finetune_attention_modules,
        "finetune_mlp_modules": request.finetune_mlp_modules,
        "is_dataset_image": request.is_dataset_image,
        "is_dataset_audio": request.is_dataset_audio,
        "is_embedding": request.is_embedding,
        "enable_wandb": request.enable_wandb,
        "wandb_token": request.wandb_token or "",
        "wandb_project": request.wandb_project or "",
        "enable_tensorboard": request.enable_tensorboard,
        "tensorboard_dir": request.tensorboard_dir or "",
        "output_dir": None,
        "resume_from_checkpoint": request.resume_from_checkpoint,
        "trust_remote_code": request.trust_remote_code,
        "approved_remote_code_fingerprint": request.approved_remote_code_fingerprint,
        "subject": subject,
        "gpu_ids": request.gpu_ids,
        "s3_config": None,
    }


def train_start(
    model_name: str,
    training_type: str = "LoRA/QLoRA",
    hf_dataset: Optional[str] = None,
    local_datasets: Optional[List[str]] = None,
    format_type: str = "auto",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    max_steps: Optional[int] = None,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    packing: bool = False,
    train_on_completions: bool = False,
    warmup_steps: Optional[int] = None,
    save_steps: int = 0,
    weight_decay: float = 0.0,
    random_seed: int = 3407,
    subset: Optional[str] = None,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    eval_steps: float = 0.0,
    is_dataset_image: bool = False,
    is_dataset_audio: bool = False,
    is_embedding: bool = False,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    custom_format_mapping: Optional[Dict[str, Any]] = None,
    enable_wandb: bool = False,
    wandb_token: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    approved_remote_code_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Start a fine-tuning job. Returns immediately with a ``job_id``.

    Provide a dataset via either ``hf_dataset`` (a HuggingFace id) or
    ``local_datasets`` (one or more names/paths from ``data_list_local``).
    Defaults match Studio's recommended values; per-model YAML defaults are
    applied on top by the trainer. Poll progress with ``train_status`` and
    stop early with ``train_stop``.

    ``training_type`` is one of ``LoRA/QLoRA``, ``Full Finetuning`` or
    ``Continued Pretraining``. Set ``max_steps`` > 0 to train by step count
    instead of epochs. For datasets that ``data_check_format`` flags as
    needing a manual mapping, pass ``custom_format_mapping``. For models that
    require custom code, set ``trust_remote_code`` (and pass the
    ``approved_remote_code_fingerprint`` from Studio's remote-code scan);
    first-party (unsloth/nvidia) repos that opt in via their YAML default are
    auto-trusted. The ``finetune_*`` toggles default to True (the Studio UI
    default) so VLM/audio-VLM LoRA jobs actually train layers; the worker
    ignores vision layers for non-VLM models.
    """
    from core.training.training import get_training_backend
    from models.training import TrainingStartRequest

    backend = get_training_backend()
    if backend.is_training_active():
        return {
            "success": False,
            "status": "error",
            "job_id": backend.current_job_id or "",
            "error": "Training is already in progress. Call train_stop first.",
        }

    # Resolve/validate local datasets the same way routes/training.py does:
    # resolve_dataset_path keeps absolute paths inside Studio's dataset roots
    # (rejecting arbitrary host files) and resolves registered names. Mirrors
    # the HTTP route's _validate_local_dataset_paths.
    if local_datasets:
        try:
            from utils.paths import resolve_dataset_path
            resolved: list[str] = []
            for path in local_datasets:
                resolved_path = resolve_dataset_path(path)
                if not resolved_path.exists():
                    raise ValueError(
                        f"Local dataset not found: {path} (resolved: {resolved_path}). "
                        "Register it via data_register first."
                    )
                resolved.append(str(resolved_path))
        except ValueError as exc:
            return {"success": False, "status": "error", "error": str(exc)}
        local_datasets = resolved

    fields: dict[str, Any] = dict(
        model_name = model_name,
        training_type = training_type,
        hf_dataset = hf_dataset,
        local_datasets = local_datasets or [],
        format_type = format_type,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
        batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        max_steps = max_steps,
        use_lora = use_lora,
        lora_r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        packing = packing,
        train_on_completions = train_on_completions,
        warmup_steps = warmup_steps,
        save_steps = save_steps,
        weight_decay = weight_decay,
        random_seed = random_seed,
        subset = subset,
        train_split = train_split,
        eval_split = eval_split,
        eval_steps = eval_steps,
        is_dataset_image = is_dataset_image,
        is_dataset_audio = is_dataset_audio,
        is_embedding = is_embedding,
        finetune_vision_layers = finetune_vision_layers,
        finetune_language_layers = finetune_language_layers,
        finetune_attention_modules = finetune_attention_modules,
        finetune_mlp_modules = finetune_mlp_modules,
        custom_format_mapping = custom_format_mapping,
        enable_wandb = enable_wandb,
        wandb_token = resolve_secret("WANDB_TOKEN", wandb_token),
        hf_token = resolve_hf_token(hf_token),
        trust_remote_code = trust_remote_code,
        approved_remote_code_fingerprint = approved_remote_code_fingerprint,
    )

    try:
        request = TrainingStartRequest(**fields)
    except Exception as exc:  # Pydantic ValidationError or field validators.
        return {"success": False, "status": "error", "error": str(exc)}

    training_kwargs = _request_to_training_kwargs(request, subject = "mcp")

    # First-party (unsloth/nvidia) repos whose YAML opts into trust_remote_code
    # are auto-trusted, matching routes/training.py. Explicit user opt-in above
    # always wins; this only lifts the default for trusted repos.
    if not training_kwargs.get("trust_remote_code"):
        try:
            from utils.models.model_config import load_model_defaults
            from utils.security.trusted_org import is_trusted_org_repo

            yaml_trust = (
                load_model_defaults(model_name).get("training", {}).get("trust_remote_code", False)
            )
            if yaml_trust and is_trusted_org_repo(
                model_name, hf_token = training_kwargs.get("hf_token") or None
            ):
                training_kwargs["trust_remote_code"] = True
        except Exception as exc:  # noqa: BLE001 -- never block training start on the trust probe
            logger.warning("trust_remote_code auto-resolution failed for %s: %s", model_name, exc)

    # Free GPU memory held by the export subprocess before training spawns
    # (export and training share the GPU). The Studio UI route additionally
    # unloads chat models in-process; that doesn't apply here since the MCP
    # server runs in its own process and never hosts the UI's chat model.
    def _free_vram_for_training() -> None:
        try:
            from core.export import get_export_backend
            exp_backend = get_export_backend()
            if exp_backend.current_checkpoint or exp_backend.is_export_active():
                logger.info("Shutting down export subprocess to free GPU memory for training")
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as exc:  # noqa: BLE001 -- hook failures must never block the start
            logger.warning("Could not shut down export subprocess for training: %s", exc)

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    try:
        started = backend.start_training(
            job_id = job_id, before_spawn = _free_vram_for_training, **training_kwargs
        )
    except Exception as exc:
        logger.exception("MCP train_start failed")
        return {"success": False, "status": "error", "error": str(exc)}

    if not started:
        progress = backend.trainer.get_training_progress()
        return {
            "success": False,
            "status": "error",
            "error": progress.error or "Failed to start training subprocess",
        }

    return {
        "success": True,
        "status": "queued",
        "job_id": job_id,
        "message": "Training job queued and starting in subprocess. Poll train_status.",
    }


def train_status() -> dict[str, Any]:
    """Get the current training job's progress (or idle state).

    Returns the live progress (epoch, step, loss, learning rate, ETA, error)
    and whether a job is active. The ``job_id`` of the active/last job is
    included. When the worker blocked on a remote-code scan, ``error_kind`` and
    ``remote_code`` (carrying the approval ``fingerprint``) are included so a
    pure-MCP caller can discover the fingerprint and retry with approval.
    """
    from core.training.training import get_training_backend

    backend = get_training_backend()
    progress = backend.trainer.get_training_progress()
    result: dict[str, Any] = {
        "job_id": backend.current_job_id,
        "is_active": backend.is_training_active(),
        "progress": dataclasses.asdict(progress),
    }
    error_kind = getattr(backend, "last_error_kind", None)
    remote_code = getattr(backend, "last_remote_code", None)
    if error_kind:
        result["error_kind"] = error_kind
    if remote_code:
        result["remote_code"] = remote_code
    return result


def train_stop(save: bool = True) -> dict[str, Any]:
    """Stop the running training job.

    With ``save=True`` (default) the current checkpoint is saved first; pass
    ``save=False`` to cancel without saving.
    """
    from core.training.training import get_training_backend

    backend = get_training_backend()
    if not backend.is_training_active():
        return {"status": "idle", "message": "No training job is currently running"}
    backend.stop_training(save = save)
    return {"status": "stopping", "saved": save, "job_id": backend.current_job_id}


def register(mcp: FastMCP) -> list[str]:
    """Register the training tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    for fn in (train_start, train_status, train_stop):
        mcp.tool(fn, annotations = _STATEFUL)
        names.append(fn.__name__)
    return names
