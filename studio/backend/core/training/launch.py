# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared prepare-and-launch path for training jobs, used by POST /start and
the training queue runner."""

import threading
import uuid as _uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException

from loggers import get_logger
from models import TrainingStartRequest
from utils.models.model_config import load_model_defaults as _default_model_defaults_loader
from utils.paths import resolve_dataset_path

logger = get_logger(__name__)

# start_training's is_alive() guard and its _proc assignment are not atomic;
# serialize spawns so the queue runner and a manual /start can't double-spawn.
_start_lock = threading.Lock()


def generate_job_id() -> str:
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"


def validate_local_dataset_paths(paths: list[str], label: str = "Local dataset") -> list[str]:
    """Resolve and validate a list of local dataset paths. Returns validated absolute paths."""
    validated = []
    missing = []
    for dataset_path in paths:
        dataset_file = resolve_dataset_path(dataset_path)
        if not dataset_file.exists():
            missing.append(f"{dataset_path} (resolved: {dataset_file})")
            continue
        logger.info(f"Found {label.lower()} file: {dataset_file}")
        validated.append(str(dataset_file))

    if missing:
        missing_detail = "; ".join(missing[:3])
        raise HTTPException(
            status_code = 400,
            detail = f"{label} not found: {missing_detail}",
        )
    return validated


def validate_s3_support(request: TrainingStartRequest) -> None:
    # S3 dataset loading needs the optional boto3 dependency. Reject early
    # with a clear message so credentials are never accepted and then
    # silently dropped on a host without boto3 installed.
    if request.s3_config is not None:
        from core.training.s3_dataset import boto3_available
        if not boto3_available():
            raise HTTPException(
                status_code = 501,
                detail = "S3 dataset loading requires boto3. Install it with: pip install boto3",
            )


def validate_training_request(request: TrainingStartRequest) -> Optional[str]:
    validate_s3_support(request)

    # Validate dataset paths if provided.
    if request.local_datasets:
        request.local_datasets = validate_local_dataset_paths(
            request.local_datasets, "Local dataset"
        )
    if request.local_eval_datasets and request.eval_steps > 0:
        request.local_eval_datasets = validate_local_dataset_paths(
            request.local_eval_datasets, "Local eval dataset"
        )

    resume_output_dir: Optional[str] = None
    if request.resume_from_checkpoint:
        from core.training.resume import (
            can_resume_run,
            get_resume_checkpoint_path,
            normalize_resume_output_dir,
        )
        from storage.studio_db import get_resumable_run_by_output_dir

        try:
            resume_output_dir = normalize_resume_output_dir(request.resume_from_checkpoint)
        except ValueError as e:
            # Deliberate user-facing validation message.
            validation_message = str(e)
            raise HTTPException(status_code = 400, detail = validation_message)

        resume_run = get_resumable_run_by_output_dir(resume_output_dir)
        if not resume_run or not can_resume_run(resume_run):
            raise HTTPException(
                status_code = 400,
                detail = "Resume checkpoint must belong to a stopped run with saved trainer state.",
            )
        resume_checkpoint = get_resume_checkpoint_path(resume_output_dir)
        if not resume_checkpoint:
            raise HTTPException(
                status_code = 400,
                detail = "Resume checkpoint must include saved trainer state.",
            )
        request.resume_from_checkpoint = resume_checkpoint

    # Validate streaming-mode compatibility before any expensive work.
    # Streaming is supported only for Hugging Face text datasets.
    if request.dataset_streaming:
        if not request.hf_dataset:
            raise HTTPException(
                status_code = 400,
                detail = "dataset_streaming requires hf_dataset; streaming is not supported for local datasets.",
            )
        if request.is_dataset_image or request.is_dataset_audio:
            raise HTTPException(
                status_code = 400,
                detail = "dataset_streaming is not supported for vision or audio datasets.",
            )
        if request.is_embedding:
            raise HTTPException(
                status_code = 400,
                detail = "dataset_streaming is not supported for embedding training; the embedding loader needs the full dataset.",
            )
        from utils.hardware import hardware as _hw

        if _hw.DEVICE == _hw.DeviceType.MLX:
            raise HTTPException(
                status_code = 400,
                detail = "dataset_streaming is not yet supported on Apple Silicon (MLX); the MLX loader materializes the full dataset.",
            )
        if request.max_steps is None or request.max_steps <= 0:
            raise HTTPException(
                status_code = 422,
                detail = "dataset_streaming requires max_steps > 0 because streaming datasets have no known length.",
            )
        if request.train_on_completions:
            raise HTTPException(
                status_code = 422,
                detail = "dataset_streaming is not supported with train_on_completions yet.",
            )
        if request.eval_steps > 0:
            train_split = request.train_split or "train"
            if not request.eval_split or request.eval_split == train_split:
                raise HTTPException(
                    status_code = 422,
                    detail = "dataset_streaming with evaluation requires a separate eval_split.",
                )
        # Streaming is HF-only: reject when the request also carries a local
        # dataset path or an S3 config; those sources cannot be streamed via
        # HF's streaming loader.
        if request.local_datasets:
            raise HTTPException(
                status_code = 400,
                detail = (
                    "dataset_streaming is HF-only; remove local_datasets / S3 source. "
                    "Streaming is not supported with local file paths."
                ),
            )
        if request.s3_config is not None:
            raise HTTPException(
                status_code = 400,
                detail = (
                    "dataset_streaming is HF-only; remove local_datasets / S3 source. "
                    "Streaming is not supported with S3 datasets."
                ),
            )

    return resume_output_dir


def build_training_kwargs(
    request: TrainingStartRequest,
    resume_output_dir: Optional[str],
    subject: str,
    model_defaults_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if model_defaults_loader is None:
        model_defaults_loader = _default_model_defaults_loader

    # Convert request to backend kwargs.
    training_kwargs = {
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
        "gradient_checkpointing": request.gradient_checkpointing.strip()
        if request.gradient_checkpointing and request.gradient_checkpointing.strip()
        else "unsloth",
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
        "output_dir": resume_output_dir,
        "resume_from_checkpoint": request.resume_from_checkpoint,
        "trust_remote_code": request.trust_remote_code,
        "approved_remote_code_fingerprint": request.approved_remote_code_fingerprint,
        "subject": subject,
        "gpu_ids": request.gpu_ids,
        "s3_config": request.s3_config.model_dump() if request.s3_config else None,
    }

    # Latest-sidecar models size and train 16-bit (same flip as chat load):
    # 4-bit is disabled for brand-new architectures, so VRAM coexistence
    # checks must not underestimate against a load the worker will refuse.
    if training_kwargs["load_in_4bit"]:
        from utils.transformers_version import latest_tier_active_for
        if latest_tier_active_for(
            training_kwargs["model_name"],
            training_kwargs["hf_token"] or None,
        ):
            training_kwargs["load_in_4bit"] = False
            logger.info(
                "Latest-transformers sidecar active for %s - sizing and "
                "training in 16-bit (4-bit is disabled for brand-new "
                "architectures)",
                training_kwargs["model_name"],
            )

    # Training page has no trust_remote_code toggle, so honor the YAML default
    # -- but only for genuine first-party (unsloth/nvidia) Hub repos, never a
    # local path or a name merely starting with "unsloth/".
    if not training_kwargs["trust_remote_code"]:
        from utils.security.trusted_org import is_trusted_org_repo

        model_defaults = model_defaults_loader(request.model_name)
        yaml_trust = model_defaults.get("training", {}).get("trust_remote_code", False)
        if yaml_trust and is_trusted_org_repo(
            request.model_name, hf_token = request.hf_token or None
        ):
            logger.info(f"YAML config sets trust_remote_code=True for {request.model_name}")
            training_kwargs["trust_remote_code"] = True
        elif yaml_trust:
            logger.warning(
                "YAML sets trust_remote_code=True for %s but it is not a trusted "
                "first-party repo; leaving disabled (user can opt in explicitly).",
                request.model_name,
            )

    return training_kwargs


def make_free_vram_hook(training_kwargs: Dict[str, Any]) -> Callable[[], None]:
    # Free VRAM for training: stop export, unload chat unless it can coexist.
    # A before_spawn hook -> runs only after start_training's guards pass, so
    # we never tear down chat/export VRAM for a start that is then refused.
    def _free_vram_for_training() -> None:
        try:
            from core.export import get_export_backend
            exp_backend = get_export_backend()
            # Tear down the export subprocess whenever an export is in flight,
            # not just once a checkpoint is loaded: during the load phase
            # current_checkpoint is still unset while the worker is already
            # allocating GPU memory, so gate on is_export_active() too.
            if exp_backend.current_checkpoint or exp_backend.is_export_active():
                logger.info("Shutting down export subprocess to free GPU memory for training")
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        try:
            from routes.training_vram import (
                can_keep_chat_during_training,
                free_chat_models_for_training,
                summarize_resident_chat,
            )

            resident = summarize_resident_chat()
            if not resident["any"]:
                return
            if resident.get("loading"):
                # In-flight load can't be sized -> free rather than risk OOM.
                freed = free_chat_models_for_training(reason = "chat model still loading")
                logger.info("Freed in-flight chat load for training: %s", freed)
                return
            keep, info = can_keep_chat_during_training(
                model_name = training_kwargs["model_name"],
                hf_token = training_kwargs["hf_token"],
                training_type = training_kwargs["training_type"],
                load_in_4bit = training_kwargs["load_in_4bit"],
                batch_size = training_kwargs["batch_size"],
                max_seq_length = training_kwargs["max_seq_length"],
                lora_rank = training_kwargs["lora_r"],
                target_modules = training_kwargs["target_modules"],
                gradient_checkpointing = training_kwargs["gradient_checkpointing"],
                optimizer = training_kwargs["optim"],
                gpu_ids = training_kwargs["gpu_ids"],
            )
            if keep:
                logger.info(
                    "Keeping chat model(s) loaded during training "
                    "(free ~%s GB, needs ~%s GB): %s",
                    info.get("usable_gb"),
                    info.get("required_gb"),
                    resident,
                )
            else:
                freed = free_chat_models_for_training(
                    reason = "insufficient VRAM to run training alongside chat",
                )
                logger.info("Freed chat model(s) for training: %s", freed)
        except Exception as e:
            logger.warning("Chat/training VRAM coordination failed; proceeding: %s", e)

    return _free_vram_for_training


def launch_training(
    job_id: str,
    request: TrainingStartRequest,
    resume_output_dir: Optional[str],
    subject: str,
    backend = None,
    model_defaults_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> bool:
    if backend is None:
        from core.training import get_training_backend
        backend = get_training_backend()

    training_kwargs = build_training_kwargs(
        request, resume_output_dir, subject, model_defaults_loader = model_defaults_loader
    )

    # The hook runs only once start guards pass -> VRAM freed iff training starts.
    with _start_lock:
        return backend.start_training(
            job_id = job_id,
            before_spawn = make_free_vram_hook(training_kwargs),
            **training_kwargs,
        )
