# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training API routes
"""

import contextlib
import sys
import threading
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Any
import structlog
from loggers import get_logger
import asyncio
from datetime import datetime
import uuid as _uuid

# Add backend directory to path.
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from core.training import get_training_backend
    from core.training.resume import (
        can_resume_run,
        find_resumable_run,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
        resume_run_dir,
    )
    from utils.models.model_config import load_model_defaults
    from utils.paths import resolve_dataset_path
except ImportError:
    # Fallback: parent directory.
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.training import get_training_backend
    from core.training.resume import (
        can_resume_run,
        find_resumable_run,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
        resume_run_dir,
    )
    from utils.models.model_config import load_model_defaults
    from utils.paths import resolve_dataset_path

# Auth
from auth.authentication import authenticated_via_api_key, get_current_subject

from utils.utils import log_and_http_error

from models import (
    TrainingStartRequest,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)
from models.training import (
    DiffusionCaptionUpdateRequest,
    DiffusionDatasetExample,
    DiffusionDatasetExamplesResponse,
    DiffusionDatasetImageRecord,
    DiffusionDatasetImagesResponse,
    DiffusionDatasetImportRequest,
    DiffusionDatasetImportResponse,
    DiffusionDatasetSummary,
    DiffusionDatasetUploadResponse,
    DiffusionMetricHistory,
    DiffusionTrainableFamily,
    DiffusionTrainingInfoResponse,
    DiffusionTrainingRunDetail,
    DiffusionTrainingRunsResponse,
    DiffusionTrainingRunSummary,
    DiffusionTrainingStartRequest,
    DiffusionTrainingStartResponse,
    DiffusionTrainingStatusResponse,
    DiffusionTrainingStopRequest,
)
from models.responses import TrainingStopResponse, TrainingMetricsResponse
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError


class TrainingStopRequest(PydanticBaseModel):
    save: bool = True


router = APIRouter()
logger = get_logger(__name__)

# Consecutive 1s polls without a step update that count as a stall. Applied only
# once stepping: the pre-first-step phase (model load + tokenization) can take far
# longer, and timing out there made a healthy long-prep run look frozen.
_PROGRESS_STALL_TIMEOUT_POLLS = 1800  # ~30 min at 1 poll/sec


def _validate_local_dataset_paths(paths: list[str], label: str = "Local dataset") -> list[str]:
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


@router.get("/hardware")
async def get_hardware_utilization(current_subject: str = Depends(get_current_subject)):
    """
    Live snapshot of GPU hardware utilization for the active backend.

    Polled by the frontend during training.
    """
    from utils.hardware import get_gpu_utilization

    # Off-loop like /hardware/visible below; the first call blocks on detection while
    # the warm is importing torch.
    return await asyncio.to_thread(get_gpu_utilization)


@router.get("/hardware/visible")
async def get_visible_hardware_utilization(current_subject: str = Depends(get_current_subject)):
    from utils.hardware import get_visible_gpu_utilization

    # Off the event loop: the ROCm fallbacks shell out (Windows perf counters, sysfs) and the System view polls this route.
    return await asyncio.to_thread(get_visible_gpu_utilization)


def _background_video_generation_active() -> bool:
    """Whether a video clip is generating on the video backend's worker thread.

    POST /video/generate returns at once and generates in the background, so an
    in-flight clip is invisible to the keep-warm in-flight request count the
    API-key training guards consult; ask the backend directly. Best-effort: a
    probe failure must never block a training start."""
    try:
        from core.inference.video import get_video_backend
        return bool(get_video_backend().generate_progress().get("active"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not check video generation state for training guard: %s", e)
        return False


@router.post("/start")
async def start_training(
    request: TrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    Start a training job.

    Initiates training in the background and returns immediately. Use /status
    to check progress.
    """
    try:
        logger.info(f"Starting training job with model: {request.model_name}")

        # When Unsloth is driven as an inference API (API-key auth), refuse to start
        # training while a request is in flight: training frees VRAM by unloading
        # the chat model, which would kill the stream. The Unsloth UI (session auth)
        # still starts training and coexists/frees VRAM as before. (A mixed UI+API
        # session is not yet special-cased.)
        if via_api_key is True:
            from core.inference.llama_keepwarm import other_inference_request_count
            if (
                other_inference_request_count(current_request_counted = False) > 0
                or _background_video_generation_active()
            ):
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "Cannot start training over the API while an inference request is in "
                        "progress. Wait for it to finish, or start training from the Unsloth UI."
                    ),
                )

        # No in-process ensure_transformers_version(): the subprocess
        # (worker.py) activates the correct version before importing ML libs.

        # A consented latest-transformers install stage-and-swaps .venv_t5_latest;
        # a worker spawned mid-swap could activate a half-replaced sidecar.
        from utils.transformers_latest import is_install_in_progress

        if is_install_in_progress():
            raise HTTPException(
                status_code = 409,
                detail = ("A transformers installation is in progress. Retry when it completes."),
            )

        backend = get_training_backend()

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

        # Check before mutating state.
        if backend.is_training_active():
            existing_job_id: Optional[str] = getattr(backend, "current_job_id", "")
            return TrainingJobResponse(
                job_id = existing_job_id or "",
                status = "error",
                message = (
                    "Training is already in progress. "
                    "Stop current training before starting a new one."
                ),
                error = "Training already active",
            )

        # A diffusion LoRA job runs in its own subprocess on the same GPU, so an LLM start must refuse while one is active. Symmetric with start_diffusion_training.
        if _diffusion_training_active():
            return TrainingJobResponse(
                job_id = "",
                status = "error",
                message = (
                    "A diffusion (Images) LoRA training job is already running. "
                    "Stop it before starting an LLM training run."
                ),
                error = "Diffusion training already active",
            )

        # Job ID; start_training() sets it on the backend only after the old
        # pump thread is dead.
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

        # Validate dataset paths if provided.
        if request.local_datasets:
            request.local_datasets = _validate_local_dataset_paths(
                request.local_datasets, "Local dataset"
            )
        if request.local_eval_datasets and request.eval_steps > 0:
            request.local_eval_datasets = _validate_local_dataset_paths(
                request.local_eval_datasets, "Local eval dataset"
            )
        resume_output_dir: Optional[str] = None
        resume_run: Optional[dict] = None
        if request.resume_from_checkpoint:
            try:
                resume_output_dir = normalize_resume_output_dir(request.resume_from_checkpoint)
            except ValueError as e:
                # Deliberate user-facing validation message.
                validation_message = str(e)
                raise HTTPException(status_code = 400, detail = validation_message)

            resume_run = find_resumable_run(resume_output_dir)
            if not resume_run or not can_resume_run(resume_run):
                raise HTTPException(
                    status_code = 400,
                    detail = "Resume checkpoint must belong to a stopped or errored run with complete saved trainer state.",
                )
            resume_checkpoint = get_resume_checkpoint_path(resume_output_dir)
            if not resume_checkpoint:
                raise HTTPException(
                    status_code = 400,
                    detail = "Resume checkpoint must include saved trainer state.",
                )
            request.resume_from_checkpoint = resume_checkpoint
            # New files continue in the run dir even when a checkpoint-N was targeted.
            resume_output_dir = resume_run_dir(resume_checkpoint)

        # Validate streaming-mode compatibility before any expensive work.
        # Streaming is supported only for Hugging Face text datasets.
        from utils.hardware import hardware as _hw
        from utils.hardware import ensure_hardware_detected

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
            # The warm fills DEVICE shortly after the socket binds, so a start in that window
            # would read None, skip the MLX rejection, and hand a streaming dataset to the MLX
            # loader (which materializes it whole). Detect first, off-loop: it imports torch.
            await asyncio.to_thread(ensure_hardware_detected)
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
        else:
            # The synchronous backend start builds its worker config with get_device().
            # Detect off-loop first so an early start cannot stall login or health.
            await asyncio.to_thread(ensure_hardware_detected)

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
            "use_dora": request.use_dora,
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
            "subject": current_subject,
            "gpu_ids": request.gpu_ids,
            "s3_config": request.s3_config.model_dump() if request.s3_config else None,
        }

        # Latest-sidecar models size and train 16-bit (same flip as chat load):
        # 4-bit is disabled for brand-new architectures, so VRAM coexistence
        # checks must not underestimate against a load the worker will refuse.
        if training_kwargs["load_in_4bit"]:
            from utils.transformers_version import latest_tier_active_for
            if await asyncio.to_thread(
                latest_tier_active_for,
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

            model_defaults = load_model_defaults(request.model_name)
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
                # A resident or in-flight Images pipeline holds GPU memory the run needs and cannot be cheaply sized, so tear it down
                # unconditionally. unload() no-ops when idle and preempts an in-flight load; release the arbiter too. Must precede the chat block, which early-returns.
                from core.inference import gpu_arbiter
                from core.inference.diffusion_engine_router import (
                    get_active_diffusion_engine,
                )

                # The ACTIVE engine, not the diffusers singleton: on a native (sd_cpp) selection the diffusers backend reports unloaded while the native engine still holds state.
                diffusion = get_active_diffusion_engine()
                if diffusion.is_loaded:
                    logger.info(
                        "Unloading diffusion (Images) model to free GPU memory for training"
                    )
                diffusion.unload()
                gpu_arbiter.release(gpu_arbiter.DIFFUSION)
            except Exception as e:
                logger.warning("Could not unload diffusion model for training: %s", e)

            try:
                # A resident Video pipeline holds GPU memory too and loads under the VIDEO arbiter owner the teardown above never touches. Tear it down the same way and release VIDEO. Must precede the chat block.
                from core.inference import gpu_arbiter
                from core.inference.video import get_video_backend

                video = get_video_backend()
                if video.status().get("loaded"):
                    logger.info("Unloading Video model to free GPU memory for training")
                video.unload()
                gpu_arbiter.release(gpu_arbiter.VIDEO)
            except Exception as e:
                logger.warning("Could not unload video model for training: %s", e)

            try:
                from routes.training_vram import (
                    can_keep_chat_during_training,
                    coordinate_models_for_training,
                )

                def _can_keep_resident_models():
                    return can_keep_chat_during_training(
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

                freed = coordinate_models_for_training(_can_keep_resident_models)
                if freed:
                    logger.info("Freed models for training: %s", freed)
            except Exception as e:
                logger.warning("Inference/training memory coordination failed; proceeding: %s", e)

        # The hook runs only once start guards pass -> VRAM freed iff training starts.
        from utils.transformers_version import SidecarSwapInProgress

        try:
            # Offloaded to a worker thread: diffusion/video unload() waits on the engines' generation locks (and the export subprocess
            # teardown can take seconds), which would otherwise freeze every concurrent request. The diffusion admission is held ACROSS
            # the spawn so the decision is atomic: entering it re-tests the state under the service's own lock, and while held reserve() refuses.
            with _diffusion_gpu_admission():
                success = await asyncio.to_thread(
                    backend.start_training,
                    job_id = job_id,
                    before_spawn = _free_vram_for_training,
                    resume_source_run_id = resume_run["id"] if resume_run else None,
                    **training_kwargs,
                )
        except _DiffusionStartInFlight as exc:
            return TrainingJobResponse(
                job_id = "",
                status = "error",
                message = str(exc),
                error = "Diffusion training already active",
            )
        except SidecarSwapInProgress as exc:
            # Expected loss of the race against a sidecar install: a retryable
            # 409 matching the route-entry guard, not an internal error.
            raise HTTPException(status_code = 409, detail = str(exc))

        if not success:
            progress_error = backend.trainer.training_progress.error
            return TrainingJobResponse(
                job_id = backend.current_job_id or "",
                status = "error",
                message = progress_error or "Failed to start training subprocess",
                error = progress_error or "subprocess_start_failed",
            )

        return TrainingJobResponse(
            job_id = job_id,
            status = "queued",
            message = "Training job queued and starting in subprocess",
            error = None,
        )

    except HTTPException:
        # Deliberate rejections (S3 not implemented, resume validation) must
        # reach the client with their original status, not a generic 500.
        raise
    except ValueError as e:
        logger.warning("Rejected training GPU selection: %s", e)
        # Deliberate user-facing GPU-selection validation message.
        validation_message = str(e)
        raise HTTPException(status_code = 400, detail = validation_message)
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to start training",
            event = "training.start_failed",
            log = logger,
        )


@router.post("/stop", response_model = TrainingStopResponse)
async def stop_training(
    body: TrainingStopRequest = TrainingStopRequest(),
    current_subject: str = Depends(get_current_subject),
):
    """
    Stop the currently running training job.

    Body:
        save (bool): If True (default), save the model at the current checkpoint.
    """
    try:
        backend = get_training_backend()
        is_active = backend.is_training_active()
        logger.info("Stop requested: save=%s is_active=%s", body.save, is_active)

        if not is_active:
            return TrainingStopResponse(
                status = "idle", message = "No training job is currently running"
            )

        if not backend.stop_training(save = body.save):
            return TrainingStopResponse(
                status = "idle", message = "No training job is currently running"
            )

        return TrainingStopResponse(
            status = "stopped",
            message = "Stop requested. Training will stop at the next safe step.",
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to stop training",
            event = "training.stop_failed",
            log = logger,
        )


@router.post("/reset")
async def reset_training(current_subject: str = Depends(get_current_subject)):
    """Reset training state so the user can return to configuration."""
    try:
        backend = get_training_backend()
        is_active = backend.is_training_active()

        if is_active:
            if backend._cancel_requested:
                # Cancel (save=False) requested — force-terminate to reset immediately.
                logger.info("Force-terminating subprocess for immediate reset (cancel path)")
                backend.force_terminate()
            else:
                logger.warning("Rejected reset while training active: is_active=%s", is_active)
                raise HTTPException(
                    status_code = 409,
                    detail = "Training is still running. Stop training and wait for it to finish before resetting.",
                )

        logger.info("Reset training state: clearing runtime + metric history")
        backend._should_stop = False  # Clear stop flag so status returns to idle
        backend.trainer._update_progress(
            is_training = False,
            is_completed = False,
            error = None,
            status_message = "Ready to train",
            step = 0,
            loss = None,
            epoch = 0,
            total_steps = 0,
        )
        backend.loss_history = []
        backend.lr_history = []
        backend.step_history = []
        backend.grad_norm_history = []
        backend.grad_norm_step_history = []
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to reset training",
            event = "training.reset_failed",
            log = logger,
        )


@router.get("/status")
async def get_training_status(current_subject: str = Depends(get_current_subject)):
    """
    Get the current training status.
    """
    try:
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "") or ""

        is_active = backend.is_training_active()

        try:
            progress = backend.trainer.get_training_progress()
        except Exception:
            progress = None

        status_message = (
            getattr(progress, "status_message", None) if progress else None
        ) or "Ready to train"
        error_message = getattr(progress, "error", None) if progress else None

        trainer_stopped = getattr(backend, "_should_stop", False)

        # Derive high-level phase
        if error_message:
            phase = "error"
        elif is_active:
            msg_lower = status_message.lower()
            if "loading" in msg_lower or "importing" in msg_lower:
                phase = "loading_model"
            elif any(k in msg_lower for k in ["preparing", "initializing", "configuring"]):
                phase = "configuring"
            else:
                phase = "training"
        elif trainer_stopped:
            phase = "stopped"
        elif progress and getattr(progress, "is_completed", False):
            phase = "completed"
        else:
            phase = "idle"

        details = None
        if progress:
            details = {
                "epoch": getattr(progress, "epoch", 0),
                "step": getattr(progress, "step", 0),
                "total_steps": getattr(progress, "total_steps", 0),
                "loss": getattr(progress, "loss", None),
                "learning_rate": getattr(progress, "learning_rate", None),
            }
            # Always present: an explicit null tells the client to drop a cached
            # path (stop without save clears the run's output_dir).
            details["output_dir"] = getattr(backend, "_output_dir", None) or None

        # Metric history for chart recovery after SSE reconnection.
        metric_history = None
        if backend.step_history:
            metric_history = {
                "steps": list(backend.step_history),
                "loss": list(backend.loss_history),
                "lr": list(backend.lr_history),
                "grad_norm": list(getattr(backend, "grad_norm_history", [])),
                "grad_norm_steps": list(getattr(backend, "grad_norm_step_history", [])),
                "eval_loss": list(backend.eval_loss_history),
                "eval_steps": list(backend.eval_step_history),
            }

        return TrainingStatus(
            job_id = job_id,
            phase = phase,
            is_training_running = is_active,
            eval_enabled = backend.eval_enabled,
            message = status_message,
            error = error_message,
            details = details,
            metric_history = metric_history,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training status",
            event = "training.status_failed",
            log = logger,
        )


@router.get("/metrics", response_model = TrainingMetricsResponse)
async def get_training_metrics(current_subject: str = Depends(get_current_subject)):
    """
    Get training metrics (loss, learning rate, steps).
    """
    try:
        backend = get_training_backend()

        loss_history = backend.loss_history
        lr_history = backend.lr_history
        step_history = backend.step_history
        grad_norm_history = getattr(backend, "grad_norm_history", [])
        grad_norm_step_history = getattr(backend, "grad_norm_step_history", [])

        current_loss = loss_history[-1] if loss_history else None
        current_lr = lr_history[-1] if lr_history else None
        current_step = step_history[-1] if step_history else None

        return TrainingMetricsResponse(
            loss_history = loss_history,
            lr_history = lr_history,
            step_history = step_history,
            grad_norm_history = grad_norm_history,
            grad_norm_step_history = grad_norm_step_history,
            current_loss = current_loss,
            current_lr = current_lr,
            current_step = current_step,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training metrics",
            event = "training.metrics_failed",
            log = logger,
        )


@router.get("/progress")
async def stream_training_progress(
    request: Request, current_subject: str = Depends(get_current_subject)
):
    """
    Stream training progress via Server-Sent Events (SSE).

    Real-time progress with reconnection support per the SSE spec:
      - `id:` per event so the browser tracks position.
      - `retry:` to control reconnection interval.
      - Named `event:` types (progress, heartbeat, complete, error).
      - Reads `Last-Event-ID` on reconnect to replay missed steps.
    """
    # Read Last-Event-ID header for reconnection resume.
    last_event_id = request.headers.get("last-event-id")
    resume_from_step: Optional[int] = None
    if last_event_id is not None:
        try:
            resume_from_step = int(last_event_id)
            # Fires on every reconnect (each tab switch); the meaningful signal is
            # the "replayed N missed steps" line below, logged only when N > 0.
            logger.debug(f"SSE reconnect: resuming from step {resume_from_step}")
        except ValueError:
            logger.warning(f"Invalid Last-Event-ID: {last_event_id}")

    async def event_generator():
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "") or ""

        # ── Helpers ──────────────────────────────────────────────
        def build_progress(
            step: int,
            loss: Optional[float],
            learning_rate: Optional[float],
            total_steps: int,
            epoch: Optional[float] = None,
            progress: Optional[Any] = None,
            grad_norm_override: Optional[float] = None,
            eval_loss_override: Optional[float] = None,
        ) -> TrainingProgress:
            total = max(total_steps, 0)
            if step < 0 or total == 0:
                progress_percent = 0.0
            else:
                progress_percent = float(step) / float(total) * 100.0 if total > 0 else 0.0

            # Pull values from the progress object if available.
            elapsed_seconds = getattr(progress, "elapsed_seconds", None) if progress else None
            eta_seconds = getattr(progress, "eta_seconds", None) if progress else None
            grad_norm = grad_norm_override
            if grad_norm is None and progress:
                grad_norm = getattr(progress, "grad_norm", None)
            num_tokens = getattr(progress, "num_tokens", None) if progress else None
            eval_loss = eval_loss_override
            if eval_loss is None and progress:
                eval_loss = getattr(progress, "eval_loss", None)

            return TrainingProgress(
                job_id = job_id,
                step = step,
                total_steps = total,
                loss = loss,
                learning_rate = learning_rate,
                progress_percent = progress_percent,
                epoch = epoch,
                elapsed_seconds = elapsed_seconds,
                eta_seconds = eta_seconds,
                grad_norm = grad_norm,
                num_tokens = num_tokens,
                eval_loss = eval_loss,
            )

        def format_sse(
            data: str,
            event: str = "progress",
            event_id: Optional[int] = None,
        ) -> str:
            """Format a single SSE message with id/event/data fields."""
            lines = []
            if event_id is not None:
                lines.append(f"id: {event_id}")
            lines.append(f"event: {event}")
            lines.append(f"data: {data}")
            lines.append("")  # trailing blank line
            lines.append("")  # double newline terminates the event
            return "\n".join(lines)

        # ── Retry directive ──────────────────────────────────────
        # Reconnect after 3 seconds if the connection drops.
        yield "retry: 3000\n\n"

        # ── Replay missed steps on reconnect ─────────────────────
        if resume_from_step is not None and backend.step_history:
            replayed = 0
            grad_norm_by_step = {
                step_val: grad_val
                for step_val, grad_val in zip(
                    getattr(backend, "grad_norm_step_history", []),
                    getattr(backend, "grad_norm_history", []),
                )
            }
            for i, step_val in enumerate(backend.step_history):
                if step_val > resume_from_step:
                    loss_val = backend.loss_history[i] if i < len(backend.loss_history) else None
                    lr_val = backend.lr_history[i] if i < len(backend.lr_history) else None
                    tp_replay = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    total_replay = (
                        getattr(tp_replay, "total_steps", step_val) if tp_replay else step_val
                    )
                    epoch_replay = getattr(tp_replay, "epoch", None) if tp_replay else None
                    payload = build_progress(
                        step_val,
                        loss_val,
                        lr_val,
                        total_replay,
                        epoch_replay,
                        progress = tp_replay,
                        grad_norm_override = grad_norm_by_step.get(step_val),
                    )
                    yield format_sse(payload.model_dump_json(), event = "progress", event_id = step_val)
                    replayed += 1
            if replayed:
                logger.info(f"SSE reconnect: replayed {replayed} missed steps")

        # ── Initial status (only on fresh connections) ───────────
        if resume_from_step is None:
            is_active = backend.is_training_active()
            tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
            initial_total_steps = getattr(tp, "total_steps", 0) if tp else 0
            initial_epoch = getattr(tp, "epoch", None) if tp else None

            initial_progress = build_progress(
                step = 0,
                loss = None,
                learning_rate = None,
                total_steps = initial_total_steps,
                epoch = initial_epoch,
                progress = tp,
            )
            yield format_sse(initial_progress.model_dump_json(), event = "progress", event_id = 0)

            # If not active, send final state and exit
            if not is_active:
                _live = (getattr(tp, "step", 0) or 0) if tp else 0
                if backend.step_history or _live > 0:
                    final_step = backend.step_history[-1] if backend.step_history else 0
                    final_loss = backend.loss_history[-1] if backend.loss_history else None
                    final_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; report the live step with
                    # loss=None instead of the last finite pair.
                    if _live > final_step:
                        final_step = _live
                        final_loss = getattr(tp, "loss", None)
                        final_lr = getattr(tp, "learning_rate", final_lr)
                    final_total_steps = getattr(tp, "total_steps", final_step) if tp else final_step
                    final_epoch = getattr(tp, "epoch", None) if tp else None
                    payload = build_progress(
                        final_step,
                        final_loss,
                        final_lr,
                        final_total_steps,
                        final_epoch,
                        progress = tp,
                    )
                    yield format_sse(
                        payload.model_dump_json(), event = "complete", event_id = final_step
                    )
                else:
                    yield format_sse(
                        build_progress(-1, None, None, 0, progress = tp).model_dump_json(),
                        event = "complete",
                        event_id = 0,
                    )
                return

        # ── Live polling loop ────────────────────────────────────
        last_step = resume_from_step if resume_from_step is not None else -1
        no_update_count = 0
        # The stall timeout applies only once the run is stepping (pre-step prep
        # may legitimately emit no step for a long time). On reconnect to an
        # already-stepping run, seed from the resume point / history, else a worker
        # that hangs after step N never times out for a client that reconnects past it.
        seen_live_step = (resume_from_step is not None and resume_from_step > 0) or bool(
            backend.step_history
        )

        while backend.is_training_active():
            # Client gone: end the generator without falling through to the final
            # "complete" frame, which a buffered/proxy consumer could otherwise read
            # as a finished run while training is still active.
            if await request.is_disconnected():
                return
            try:
                tp_inner = getattr(getattr(backend, "trainer", None), "training_progress", None)
                live_step = (getattr(tp_inner, "step", 0) or 0) if tp_inner else 0
                if backend.step_history or live_step > 0:
                    current_step = backend.step_history[-1] if backend.step_history else 0
                    current_loss = backend.loss_history[-1] if backend.loss_history else None
                    current_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; follow the live progress
                    # step and report its loss (None until it recovers).
                    if live_step > current_step:
                        current_step = live_step
                        current_loss = getattr(tp_inner, "loss", None)
                        current_lr = getattr(tp_inner, "learning_rate", current_lr)
                    current_total_steps = (
                        getattr(tp_inner, "total_steps", current_step) if tp_inner else current_step
                    )
                    current_epoch = getattr(tp_inner, "epoch", None) if tp_inner else None

                    # Only send if the step changed.
                    if current_step != last_step:
                        progress_payload = build_progress(
                            current_step,
                            current_loss,
                            current_lr,
                            current_total_steps,
                            current_epoch,
                            progress = tp_inner,
                        )
                        yield format_sse(
                            progress_payload.model_dump_json(),
                            event = "progress",
                            event_id = current_step,
                        )
                        last_step = current_step
                        no_update_count = 0
                        seen_live_step = True
                    else:
                        no_update_count += 1
                        # Heartbeat every 10 seconds.
                        if no_update_count % 10 == 0:
                            heartbeat_payload = build_progress(
                                current_step,
                                current_loss,
                                current_lr,
                                current_total_steps,
                                current_epoch,
                                progress = tp_inner,
                            )
                            yield format_sse(
                                heartbeat_payload.model_dump_json(),
                                event = "heartbeat",
                                event_id = current_step,
                            )
                else:
                    # No steps yet, but training is active (model loading, etc.).
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        # Pull total_steps + status so the frontend can show
                        # "Tokenizing…" etc.
                        tp_prep = getattr(
                            getattr(backend, "trainer", None),
                            "training_progress",
                            None,
                        )
                        prep_total = getattr(tp_prep, "total_steps", 0) if tp_prep else 0
                        preparing_payload = build_progress(
                            0,
                            None,
                            None,
                            prep_total,
                            progress = tp_prep,
                        )
                        yield format_sse(
                            preparing_payload.model_dump_json(),
                            event = "heartbeat",
                            event_id = 0,
                        )

                # Fires only once stepping: a long pre-first-step prep phase is not
                # a stall, and ending the stream there made a healthy run look frozen.
                if seen_live_step and no_update_count > _PROGRESS_STALL_TIMEOUT_POLLS:
                    logger.warning("Progress stream timeout - no updates received")
                    tp_timeout = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    timeout_payload = build_progress(last_step, None, None, 0, progress = tp_timeout)
                    yield format_sse(
                        timeout_payload.model_dump_json(),
                        event = "error",
                        event_id = last_step if last_step >= 0 else 0,
                    )
                    break

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error in progress stream: {e}", exc_info = True)
                tp_error = getattr(getattr(backend, "trainer", None), "training_progress", None)
                error_payload = build_progress(0, None, None, 0, progress = tp_error)
                yield format_sse(
                    error_payload.model_dump_json(),
                    event = "error",
                    event_id = last_step if last_step >= 0 else 0,
                )
                break

        # ── Final "complete" event ───────────────────────────────
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else None
        final_lr = backend.lr_history[-1] if backend.lr_history else None
        final_tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
        # If the run ended on a non-finite stretch, report the live step with
        # loss=None instead of rolling back to the last finite pair.
        _final_live_step = (getattr(final_tp, "step", 0) or 0) if final_tp else 0
        if _final_live_step > (final_step if final_step is not None else -1):
            final_step = _final_live_step
            final_loss = getattr(final_tp, "loss", None)
            final_lr = getattr(final_tp, "learning_rate", final_lr)
        final_total_steps = getattr(final_tp, "total_steps", final_step) if final_tp else final_step
        final_epoch = getattr(final_tp, "epoch", None) if final_tp else None
        final_payload = build_progress(
            final_step,
            final_loss,
            final_lr,
            final_total_steps,
            final_epoch,
            progress = final_tp,
        )
        yield format_sse(
            final_payload.model_dump_json(),
            event = "complete",
            event_id = final_step if final_step >= 0 else 0,
        )

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Diffusion (SDXL) LoRA training ────────────────────────────────────────────
# A separate, lightweight job path: diffusion runs are driven by DiffusionTrainingService (its own subprocess + event pump), not the LLM TrainingBackend, so the two never contend.


def _diffusion_training_active() -> bool:
    """Whether a diffusion (SDXL) LoRA job is currently running. Best-effort so the
    interlock never blocks a start just because the service could not be imported."""
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        return get_diffusion_training_service().is_active()
    except Exception:  # noqa: BLE001
        return False


class _DiffusionStartInFlight(RuntimeError):
    """An LLM start lost the race to a diffusion start (route: refuse, don't spawn)."""


@contextlib.contextmanager
def _diffusion_gpu_admission():
    """Hold the diffusion service's GPU admission across the LLM spawn.

    Makes the cross-trainer admission atomic: entering re-tests the diffusion state under the
    service's own lock and raises if a diffusion run is reserved or active, and while it is held
    the diffusion ``reserve()`` refuses. So of two near-simultaneous starts of different types,
    exactly one proceeds. Fails OPEN on an import/health failure, like every other guard here: a
    chat-only install has no diffusion service and must still be able to train."""
    try:
        from core.training.diffusion_training_service import (
            TrainingActiveError,
            get_diffusion_training_service,
        )
        service = get_diffusion_training_service()
    except Exception:  # noqa: BLE001 -- no diffusion stack: nothing to coordinate with
        yield
        return
    try:
        cm = service.gpu_load_admission()
    except Exception:  # noqa: BLE001
        yield
        return
    try:
        cm.__enter__()
    except TrainingActiveError as exc:
        raise _DiffusionStartInFlight(
            "A diffusion (Images) LoRA training job is already running. "
            "Stop it before starting an LLM training run."
        ) from exc
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


def _require_diffusion_dataset_mutable() -> None:
    """Reject a dataset mutation while a diffusion run is active.

    The trainer re-opens dataset images during the loop, so mutating underneath it makes the run
    nondeterministic or raises a FileNotFoundError mid-step. Fails open (a service-import failure
    never blocks a mutation on an unknowable state), matching the start interlock."""
    if _diffusion_training_active():
        raise HTTPException(
            status_code = 409,
            detail = (
                "Training images cannot be changed while diffusion training is active. "
                "Stop the run before uploading, importing, editing captions, or deleting images."
            ),
        )


def diffusion_dataset_interlock():
    """Dependency holding the dataset interlock for a whole mutating request.

    The check above only covers the instant it runs: every one of these endpoints then hands its
    filesystem work to a thread, and a ``/diffusion/start`` reserving in that gap would move
    captions or images underneath the preflight or the running trainer. As a yield dependency the
    registration spans the endpoint, so ``reserve()`` sees it and refuses instead. Fails open on an
    import error, like the check it replaces."""
    try:
        from core.training.diffusion_training_service import (
            TrainingActiveError,
            get_diffusion_training_service,
        )
        service = get_diffusion_training_service()
    except Exception:  # noqa: BLE001 -- unknowable state never blocks a mutation
        yield
        return
    try:
        with service.dataset_mutation():
            yield
    except TrainingActiveError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


def _free_gpu_for_diffusion_training() -> None:
    """Free GPU residents before the diffusion trainer spawns its own SDXL pipeline.

    The trainer subprocess loads a full SDXL pipeline; an export worker, a resident
    Images pipeline, or loaded chat models would otherwise keep their VRAM allocated and
    OOM the run. Mirrors the LLM start path's pre-spawn cleanup (export + diffusion
    pipeline + chat). Best-effort: a failure to free one resident never blocks the start."""
    try:
        from core.export import get_export_backend
        exp_backend = get_export_backend()
        if exp_backend.current_checkpoint or exp_backend.is_export_active():
            logger.info("Shutting down export subprocess to free GPU memory for diffusion training")
            exp_backend._shutdown_subprocess()
            exp_backend.current_checkpoint = None
            exp_backend.is_vision = False
            exp_backend.is_peft = False
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not shut down export subprocess: %s", e)

    try:
        from core.inference import gpu_arbiter
        from core.inference.diffusion_engine_router import get_active_diffusion_engine

        # The ACTIVE engine, not the diffusers singleton: on a native (sd_cpp) selection the resident sd-server still holds the GPU, so unloading only the singleton is a no-op.
        diffusion = get_active_diffusion_engine()
        if diffusion.is_loaded:
            logger.info("Unloading resident Images pipeline to free GPU memory for training")
        diffusion.unload()  # no-op when nothing is loaded; also preempts an in-flight load
        gpu_arbiter.release(gpu_arbiter.DIFFUSION)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not unload Images pipeline for diffusion training: %s", e)

    try:
        # A resident Video pipeline loads under the VIDEO arbiter owner the Images teardown does not free; unload it too and release VIDEO.
        from core.inference import gpu_arbiter
        from core.inference.video import get_video_backend

        video = get_video_backend()
        if video.status().get("loaded"):
            logger.info("Unloading resident Video pipeline to free GPU memory for training")
        video.unload()  # no-op when nothing is loaded; also preempts an in-flight load
        gpu_arbiter.release(gpu_arbiter.VIDEO)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not unload Video pipeline for diffusion training: %s", e)

    try:
        # The SDXL trainer footprint cannot be cheaply sized against a resident chat model, so free chat unconditionally rather than risk an OOM.
        from routes.training_vram import free_chat_models_for_training, summarize_resident_chat
        if summarize_resident_chat()["any"]:
            freed = free_chat_models_for_training(reason = "diffusion training starting")
            logger.info("Freed chat model(s) for diffusion training: %s", freed)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not free chat models for diffusion training: %s", e)


def _preflight_gated_base(base_model: str, hf_token: Optional[str]) -> None:
    """HEAD a remote base repo's model_index.json with the caller's token; raise HTTP 400 on
    401/403 (gated / unauthorized) with an actionable message. Best-effort: a local path,
    a non-repo string, or a network hiccup passes through so the trainer can surface any real
    load error itself. Runs before GPU teardown so a doomed start never evicts a loaded model."""
    import urllib.error
    import urllib.request

    repo = (base_model or "").strip()
    # Only remote 'org/name' repos are gated; skip local paths and single-file names.
    if (
        not repo
        or repo.count("/") != 1
        or repo.startswith((".", "/", "~"))
        or repo.endswith(".gguf")
    ):
        return
    url = f"https://huggingface.co/{repo}/resolve/main/model_index.json"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    req = urllib.request.Request(url, method = "HEAD", headers = headers)
    try:
        urllib.request.urlopen(req, timeout = 5)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"Access to '{repo}' is gated or unauthorized. Accept the model's license "
                    f"on its Hugging Face page and add your HF token in Studio settings, then "
                    f"try again."
                ),
            )
        # 404 (e.g. a repo without a root model_index.json) and other codes are not an access problem; let the trainer surface any genuine load error.
    except Exception:  # noqa: BLE001 -- network/DNS hiccup must not block a start
        return


def _resolve_diffusion_data_dir(raw: str) -> Path:
    """Resolve a diffusion-training ``data_dir``. The upload/labeling routes create and
    manage image datasets directly under ``datasets_root()`` and the UI passes the bare
    folder name back as ``data_dir``, but the generic :func:`resolve_dataset_path`
    searches the LLM uploads and recipe dataset roots FIRST -- so an unrelated upload
    file or recipe folder sharing that name would shadow the just-uploaded image
    dataset (preflight 400 "not a directory", or training the wrong data). Prefer the
    image dataset root for a bare single-component name that exists there; everything
    else (explicit "uploads/..." / "recipes/..." prefixes, absolute paths, missing
    names) resolves exactly as before."""
    from utils.paths import datasets_root

    value = str(raw or "").strip()
    if value and "\x00" not in value:
        p = Path(value)
        # A single component that is not "..", so joining under datasets_root() cannot escape it.
        if not p.is_absolute() and len(p.parts) == 1 and p.parts[0] != "..":
            direct = datasets_root() / value
            # Route a bare name through the same protected resolver the CRUD routes use, so a symlink to an external directory is rejected here too. A broken symlink is included so it is rejected, not passed on.
            if direct.is_dir() or direct.is_symlink():
                return _resolve_dataset_folder(value)
    return resolve_dataset_path(raw)


@router.post("/diffusion/start", response_model = DiffusionTrainingStartResponse)
async def start_diffusion_training(
    body: DiffusionTrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Start an SDXL LoRA training job from an image + caption dataset."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    # Under API-key auth, refuse to start training while a request is in flight: _free_gpu_for_diffusion_training() below unloads the chat backends, killing the stream. Mirrors start_training.
    if via_api_key is True:
        from core.inference.llama_keepwarm import other_inference_request_count
        if (
            other_inference_request_count(current_request_counted = False) > 0
            or _background_video_generation_active()
        ):
            raise HTTPException(
                status_code = 409,
                detail = (
                    "Cannot start diffusion (Images) training over the API while an inference "
                    "request is in progress. Wait for it to finish, or start training from the "
                    "Studio UI."
                ),
            )

    # Interlock: refuse while an LLM training run holds the GPU (symmetric with the diffusion check in start_training), so the two trainers never contend for VRAM.
    try:
        if get_training_backend().is_training_active():
            raise HTTPException(
                status_code = 409,
                detail = (
                    "An LLM training job is already running. "
                    "Stop it before starting diffusion (Images) training."
                ),
            )
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 -- backend import/health issue must not block a start
        pass

    # Resolve + contain the dataset and output paths BEFORE spawning: the trainer subprocess would otherwise resolve them relative to its own cwd.
    config = body.model_dump()
    try:
        from utils.paths import outputs_root, resolve_output_dir

        config["data_dir"] = str(_resolve_diffusion_data_dir(config["data_dir"]))
        # A name that cleans away to nothing ("." / "outputs" / "./.") resolves to the outputs ROOT, where the trainer would write the adapter flat and the is_dir()-filtered listings could never see it.
        root = outputs_root().resolve()
        out_dir = resolve_output_dir(config["output_dir"])
        if Path(out_dir).resolve() == root:
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"'{config['output_dir']}' is the outputs folder itself, not a run inside it. "
                    "Pick a name for this run."
                ),
            )
        config["output_dir"] = str(out_dir)
        # The persistent conditioning cache is another trainer-written directory, so it gets the same containment. Blank/None means the in-memory cache, so it must not resolve to the outputs root.
        cond_cache = str(config.get("cond_cache_dir") or "").strip()
        cond_cache_dir = resolve_output_dir(cond_cache) if cond_cache else None
        # Same collapse, but the cache has an honest "off" to fall back to: one flat safetensors per cached latent in the trained-models directory is never what was meant.
        if cond_cache_dir is not None and Path(cond_cache_dir).resolve() == root:
            cond_cache_dir = None
        config["cond_cache_dir"] = str(cond_cache_dir) if cond_cache_dir is not None else None
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Validate the config BEFORE freeing resident GPU workloads, so a refused start never tears down the user's chat/Images model. service.start() re-runs this before spawn.
    from core.training.diffusion_lora_trainer import _config_from_dict

    try:
        normalized_cfg = _config_from_dict(config).normalized()
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Only the DiT trainer reads cond_cache_dir: the SDXL trainer builds a per-process in-memory latent cache, so accepting
    # it there promised cross-run reuse that never happened. Checked against the RESOLVED family, not the request field.
    if cond_cache and normalized_cfg.resolved_family == "sdxl":
        raise HTTPException(
            status_code = 400,
            detail = (
                "cond_cache_dir is not supported for the sdxl family: its trainer uses a "
                "per-run in-memory latent cache and would ignore the persistent one. Omit it, "
                "or train a DiT family (flux.1, flux.2-klein, flux.2-dev, qwen-image, "
                "z-image, krea-2), which reuses conditioning across runs."
            ),
        )

    # Preflight the requested DiT precision BEFORE freeing GPU residents: the trainer's own checks fire only in the child,
    # AFTER eviction. Fail fast (400) so a pre-Ampere or stub-torchao host never tears down residents for a doomed run.
    from core.training.diffusion_train_common import training_precision_preflight_error

    _precision_reason = training_precision_preflight_error(
        normalized_cfg.resolved_family, normalized_cfg.base_precision
    )
    if _precision_reason:
        raise HTTPException(status_code = 400, detail = _precision_reason)

    # Run the trainers' trust gate here too, so an untrusted/typoed base 400s BEFORE freeing GPU residents rather than failing in the child.
    from core.training.diffusion_train_common import _assert_trusted_base_model

    try:
        _assert_trusted_base_model(config.get("base_model", ""))
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Preflight access to a gated base repo with the user's token BEFORE freeing GPU residents, so a missing token fails
    # fast (400) instead of a confusing mid-load 401. In a worker thread: it does a blocking urlopen HEAD (5s timeout).
    await asyncio.to_thread(
        _preflight_gated_base, config.get("base_model", ""), config.get("hf_token")
    )

    from core.training import diffusion_train_common as _dtc

    service = get_diffusion_training_service()
    # Reserve the training slot BEFORE the dataset preflight: is_active() otherwise flips true only at service.start(), so
    # during this scan a concurrent upload/caption/delete could mutate the dataset, or a load double-allocate VRAM.
    reserved = False
    try:
        service.reserve()
        reserved = True
        # Preflight the dataset: a missing/empty/uncaptionable data_dir otherwise fails inside the trainer AFTER eviction. Same discovery the trainer runs, so the two cannot disagree.
        try:
            await asyncio.to_thread(
                _dtc.discover_image_caption_pairs,
                config["data_dir"],
                instance_prompt = config.get("instance_prompt") or None,
                caption_column = config.get("caption_column") or "text",
                # Decode-probe every image now (cheap PIL header check) so a corrupt/zero-byte upload 400s BEFORE the GPU teardown.
                verify_images = True,
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code = 400, detail = str(e))
        # Free resident GPU workloads before the trainer loads its own pipeline. Offloaded: the teardown blocks on generation locks and a subprocess join.
        await asyncio.to_thread(_free_gpu_for_diffusion_training)
        job_id = service.start(config)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except RuntimeError as e:
        # A job is already running (or a start is already reserved), or a dataset mutation is open (DatasetMutationInFlight) -- the same interlock from the other side, so also a 409.
        raise HTTPException(status_code = 409, detail = str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to start diffusion training",
            event = "diffusion_training.start_failed",
            log = logger,
        )
    finally:
        # On success the live proc keeps is_active() true; on failure this clears the reservation. Only the request that reserved clears it.
        if reserved:
            service.unreserve()
    return DiffusionTrainingStartResponse(job_id = job_id, status = "running")


@router.post("/diffusion/stop")
async def stop_diffusion_training(
    body: Optional[DiffusionTrainingStopRequest] = None,
    current_subject: str = Depends(get_current_subject),
):
    """Request a clean stop of the running diffusion training job. The optional body's
    ``save`` mirrors the LLM /stop: true (default, also for an empty POST) exports the
    partial adapter, false cancels without saving one."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    save = body.save if body is not None else True
    stopped = get_diffusion_training_service().stop(save = save)
    return {"status": "stopping" if stopped else "idle"}


@router.get("/diffusion/status", response_model = DiffusionTrainingStatusResponse)
async def diffusion_training_status(current_subject: str = Depends(get_current_subject)):
    """Poll the current diffusion training job's status/progress (JSON)."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    snap = get_diffusion_training_service().status()
    # Fold the service's flat history arrays into the nested metric_history the UI charts.
    metric_history = DiffusionMetricHistory(
        steps = snap.pop("metric_steps", []),
        loss = snap.pop("metric_loss", []),
        lr = snap.pop("metric_lr", []),
        grad_norm = snap.pop("metric_grad_norm", []),
    )
    return DiffusionTrainingStatusResponse(**snap, metric_history = metric_history)


@router.get("/diffusion/runs", response_model = DiffusionTrainingRunsResponse)
async def list_diffusion_training_runs(
    limit: int = 20, current_subject: str = Depends(get_current_subject)
):
    """Previous diffusion training runs (terminal), newest first, from the persisted
    per-run records. Summaries only; fetch one run for its config + metric logs."""
    from core.training.diffusion_training_service import list_diffusion_runs

    summaries: list[DiffusionTrainingRunSummary] = []
    for r in list_diffusion_runs(limit = limit):
        # list_diffusion_runs skips non-dict / missing-id records, but a wrong-typed field would still raise here; catch per record so one bad file never breaks the panel.
        try:
            summaries.append(DiffusionTrainingRunSummary(**r))
        except ValidationError:
            continue
    return DiffusionTrainingRunsResponse(runs = summaries)


@router.get("/diffusion/runs/{job_id}", response_model = DiffusionTrainingRunDetail)
async def get_diffusion_training_run(
    job_id: str, current_subject: str = Depends(get_current_subject)
):
    """One persisted diffusion run's full record: summary + scrubbed start config + the
    step/loss/grad-norm logs (for re-plotting a past run's charts)."""
    from core.training.diffusion_training_service import get_diffusion_run

    rec = get_diffusion_run(job_id)
    # A valid-JSON file that is not an object makes DiffusionTrainingRunDetail(**rec) raise TypeError, not the ValidationError caught below. Treat any non-dict record as absent.
    if not isinstance(rec, dict):
        raise HTTPException(status_code = 404, detail = "No such training run.")
    try:
        return DiffusionTrainingRunDetail(**rec)
    except ValidationError:
        # A malformed on-disk record (hand-edited / older shape) reads as absent rather than 500 the endpoint, like the list route skips bad records.
        raise HTTPException(status_code = 404, detail = "No such training run.")


# Extensions accepted into an image-training dataset folder: images the trainer reads, plus its caption sources (per-image sidecars and metadata/captions jsonl).
_DIFFUSION_DATASET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_DIFFUSION_DATASET_TEXT_EXTS = {".txt", ".caption", ".jsonl"}


def _resolve_dataset_caption(
    folder: Path, image_path: Path, meta_captions: dict[str, str]
) -> Optional[str]:
    """Resolve an image's caption using the same sidecar > metadata precedence the trainer
    applies in ``discover_image_caption_pairs``. A per-image .txt/.caption sidecar wins and
    is stripped, so an empty (tombstone) sidecar shadows metadata and yields "" -- the
    trainer then skips that image (``if caption:``), so it must not count as captioned."""
    caption: Optional[str] = None
    sidecar_present = False
    for ext in (".txt", ".caption"):
        sidecar = image_path.with_suffix(ext)
        if sidecar.is_file():
            sidecar_present = True
            try:
                caption = sidecar.read_text(encoding = "utf-8").strip()
            except (OSError, UnicodeError):
                # Unreadable / invalid UTF-8 sidecar: the EMPTY TOMBSTONE, not "no sidecar", matching what the trainer does with it.
                # Uploads accept raw sidecar bytes, so reading it as absent would show a metadata caption the run silently replaces.
                caption = ""
            break
    if not sidecar_present:
        try:
            rel = image_path.relative_to(folder).as_posix()
        except ValueError:
            rel = None
        caption = meta_captions.get(image_path.name) or (
            meta_captions.get(rel) if rel is not None else None
        )
    return caption


_DATASET_IMPORT_LOCKS: Dict[str, "threading.Lock"] = {}
_DATASET_IMPORT_LOCKS_GUARD = threading.Lock()


def _dataset_import_lock(folder: Path) -> "threading.Lock":
    """One lock per dataset folder, so two imports cannot fill the same empty name at once.

    Keyed by the resolved path (the same folder can be reached by different names), and kept for
    the process lifetime: there are a handful of dataset folders and a Lock is tiny, while
    dropping one while another thread holds it would defeat the point."""
    key = str(folder.resolve(strict = False))
    with _DATASET_IMPORT_LOCKS_GUARD:
        lock = _DATASET_IMPORT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DATASET_IMPORT_LOCKS[key] = lock
        return lock


def _import_response(
    entry: dict, folder: Path, *, imported: int
) -> "DiffusionDatasetImportResponse":
    summary = _diffusion_dataset_summary(folder)
    return DiffusionDatasetImportResponse(
        name = folder.name,
        path = str(folder),
        image_count = summary.image_count,
        caption_count = summary.caption_count,
        imported = imported,
        license = entry["license"],
        source_repo = entry["repo"],
    )


def _diffusion_dataset_summary(folder: Path) -> DiffusionDatasetSummary:
    # Count an image as captioned only when it resolves to a NON-EMPTY caption via the trainer's sidecar-over-metadata precedence: an empty tombstone makes the trainer skip the image.
    meta_captions = _load_metadata_captions(folder)
    images = captions = 0
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS:
            continue
        images += 1
        if _resolve_dataset_caption(folder, f, meta_captions):
            captions += 1
    return DiffusionDatasetSummary(
        name = folder.name, path = str(folder), image_count = images, caption_count = captions
    )


@router.get("/diffusion/info", response_model = DiffusionTrainingInfoResponse)
async def diffusion_training_info(current_subject: str = Depends(get_current_subject)):
    """Describe where diffusion training reads/writes, and list usable dataset folders.

    A dataset folder is any direct child of the datasets root that contains at least one
    image. The UI uses this to offer a picker instead of a blind free-text path."""
    from utils.paths import datasets_root, outputs_root

    def scan() -> DiffusionTrainingInfoResponse:
        root = datasets_root()
        found: list[DiffusionDatasetSummary] = []
        try:
            # Skip hidden dirs: never user datasets, and an in-progress example import stages into a dot-prefixed sibling.
            children = sorted(
                p
                for p in root.iterdir()
                # Skip symlinked dirs: the CRUD resolver rejects them, so discovery must not advertise one as selectable.
                if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
            )
        except OSError:
            children = []
        for child in children:
            try:
                summary = _diffusion_dataset_summary(child)
            except OSError:
                continue
            if summary.image_count > 0:
                found.append(summary)
        from core.training.diffusion_train_common import family_train_infos

        families = [DiffusionTrainableFamily(**info) for info in family_train_infos()]
        return DiffusionTrainingInfoResponse(
            datasets_root = str(root),
            outputs_root = str(outputs_root()),
            datasets = found,
            families = families,
        )

    return await asyncio.to_thread(scan)


_DATASET_NAME_RE = None  # compiled lazily; module keeps its import block torch-free


# Reserved in EVERY directory on Windows, with or without an extension (NUL.txt is NUL). The superscript COM/LPT digits count as digits to Win32 and are reserved too.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{d}" for d in "123456789¹²³"}
    | {f"lpt{d}" for d in "123456789¹²³"}
)


_DATASETS_CASE_INSENSITIVE: Optional[bool] = None


def _dataset_folder_is_case_insensitive(folder: Path) -> bool:
    """True when ``folder`` cannot hold two names differing only by case.

    Probed once per process against the real filesystem rather than keyed off ``sys.platform``:
    NTFS and the default APFS fold case, but macOS also ships case-SENSITIVE APFS volumes and a
    Linux host can keep its Studio home on an exFAT/NTFS mount. A failed probe answers False,
    which keeps the case-sensitive behaviour (two names, two files) unchanged.
    """
    global _DATASETS_CASE_INSENSITIVE
    if _DATASETS_CASE_INSENSITIVE is None:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(prefix = ".case-probe-", dir = folder) as probe:
                probe_name = Path(probe.name).name
                _DATASETS_CASE_INSENSITIVE = (folder / probe_name.upper()).exists()
        except OSError:
            return False
    return _DATASETS_CASE_INSENSITIVE


def _clean_diffusion_dataset_name(name: str) -> str:
    """Validate a dataset folder name: a single path component, no traversal, printable.

    Windows path rules are applied on EVERY platform, not just Windows: a dataset created on one
    machine is opened on another, and both failures are silent or confusing. A reserved device name
    dies in mkdir with an unhandled OSError, and a trailing period is stripped by Win32
    normalization, so an upload to the "new" dataset 'photos.' would quietly write into the
    existing 'photos'."""
    import re

    global _DATASET_NAME_RE
    if _DATASET_NAME_RE is None:
        _DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
    cleaned = (name or "").strip()
    if not _DATASET_NAME_RE.fullmatch(cleaned) or ".." in cleaned:
        raise HTTPException(
            status_code = 400,
            detail = (
                "Dataset name must be a plain folder name (letters, numbers, dots, "
                "dashes, spaces; no slashes), e.g. 'my-style-photos'."
            ),
        )
    if cleaned.endswith("."):
        raise HTTPException(
            status_code = 400,
            detail = (
                "Dataset name cannot end with a period: Windows strips it, so this name would "
                f"open the existing '{cleaned.rstrip('.')}' dataset instead of a new one."
            ),
        )
    # The stem alone is checked, since NUL.txt is the NUL device too.
    if cleaned.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"'{cleaned}' is a reserved device name on Windows and cannot be a folder. "
                "Pick another dataset name."
            ),
        )
    return cleaned


@router.post("/diffusion/dataset", response_model = DiffusionDatasetUploadResponse)
async def upload_diffusion_dataset(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Upload training images (and optional caption .txt / metadata.jsonl files) into a
    named folder under the Studio datasets root, creating it if needed. Repeat uploads
    into the same name accumulate, so large datasets can arrive in batches. The returned
    name can be passed directly as ``data_dir`` to /diffusion/start."""
    import os
    import tempfile

    from utils.upload_limits import get_upload_limit_bytes, get_upload_limit_label

    _require_diffusion_dataset_mutable()
    cleaned = _clean_diffusion_dataset_name(name)
    # Run the same symlink + root-containment check as the read/caption/delete endpoints before any write, so a symlinked name cannot make the upload write outside root.
    folder = _resolve_dataset_folder(name, must_exist = False)
    folder.mkdir(parents = True, exist_ok = True)
    # Serialize against a concurrent import into the SAME folder: the training interlock counts mutations rather than
    # excluding them, so an upload could merge into a materializing import and leave a mixed dataset. The duplicate-stem validation below reads the folder too, so it is inside the lock.
    _lock = _dataset_import_lock(folder)
    if not _lock.acquire(blocking = False):
        raise HTTPException(
            status_code = 409,
            detail = (
                f"An import into '{folder.name}' is already running. Wait for it to finish, "
                "then upload again."
            ),
        )
    try:
        limit_bytes = get_upload_limit_bytes()
        total_bytes = 0
        uploaded = 0
        allowed = _DIFFUSION_DATASET_IMAGE_EXTS | _DIFFUSION_DATASET_TEXT_EXTS
        # Validate every filename up front so a valid image ahead of a bad one is not left on disk when the 400 fires; the upload is all-or-nothing.
        names: list[str] = []
        for f in files:
            # Normalise to a safe basename. Path.name does not split on a backslash on POSIX, so fold backslashes first or a
            # Windows client's path is stored verbatim and a name still holding ".." would list an image nothing can preview.
            filename = Path((f.filename or "").replace("\\", "/")).name.strip().replace("\x00", "")
            ext = Path(filename).suffix.lower()
            if not filename or ".." in filename or ext not in allowed:
                exts = ", ".join(sorted(allowed))
                raise HTTPException(
                    status_code = 400,
                    detail = f"Unsupported file '{f.filename}'. Allowed: {exts}",
                )
            # Reject an EXACT duplicate name within THIS batch: the same-name exemption below is for SEPARATE repeat uploads, a
            # deliberate overwrite, while inside one batch the later replace would silently discard the earlier file.
            fname_cf = filename.casefold()
            if filename in names:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        f"Duplicate file '{filename}' appears more than once in this upload. "
                        "Files sharing a name would overwrite each other; rename one before "
                        "uploading."
                    ),
                )
            # Reject a second IMAGE sharing this stem but differing by extension (sample.png vs sample.jpg): both resolve to the
            # same <stem>.txt sidecar, so keeping both would silently share -- and corrupt -- one caption. Caption files are exempt.
            if ext in _DIFFUSION_DATASET_IMAGE_EXTS:
                stem = Path(filename).stem
                # Compare stems case-insensitively: on case-insensitive filesystems two stems differing only by case resolve to the
                # SAME sidecar. A same-name case variant is exempt only when its stem differs in case too; cat.PNG vs cat.png is not.
                stem_cf = stem.casefold()

                def _shares_sidecar(other_name: str) -> bool:
                    other = Path(other_name)
                    if (
                        other_name == filename
                        or other.suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS
                        or other.stem.casefold() != stem_cf
                    ):
                        return False
                    # A casefold-equal full name is exempt unless the stems match EXACTLY (extension-case variants collide on one sidecar on case-sensitive filesystems).
                    return other.stem == stem or other_name.casefold() != fname_cf

                clash = next(
                    (p.name for p in folder.iterdir() if p.is_file() and _shares_sidecar(p.name)),
                    None,
                )
                if clash is None:
                    clash = next((n for n in names if _shares_sidecar(n)), None)
                if clash is not None:
                    raise HTTPException(
                        status_code = 400,
                        detail = (
                            f"Duplicate image name '{stem}'. '{clash}' is already in this "
                            f"dataset; two images sharing a name would share one '{stem}.txt' "
                            f"caption. Rename one before uploading."
                        ),
                    )
            # Reject a CASE variant of a name already in this batch, but only where the filesystem folds case: there
            # 'Cat.png' and 'cat.png' are ONE destination, so the commit below would replace the first staged part with
            # the second while `uploaded` still counted both -- a training image (or caption) silently dropped. No single
            # folder can hold such a pair, but the open dialog's cross-folder search/Recents view can put one in a batch.
            # The same-name exemption above is for a SEPARATE repeat upload over a file already on disk, a deliberate
            # overwrite; inside one batch both parts are new, so there is nothing to overwrite. On a case-sensitive
            # filesystem the two are genuinely different files with their own sidecars, so they stay allowed.
            if any(n.casefold() == fname_cf for n in names) and _dataset_folder_is_case_insensitive(
                folder
            ):
                clash_cf = next(n for n in names if n.casefold() == fname_cf)
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        f"Duplicate file '{filename}' differs from '{clash_cf}' only by letter "
                        "case, so on this filesystem they are one file and would overwrite each "
                        "other. Rename one before uploading."
                    ),
                )
            names.append(filename)
        # Stage each file to a temp name and move it in only once the whole batch is written, so a mid-batch failure leaves the dataset untouched, including any same-name file a direct write would truncate.
        staged: list[tuple[Path, Path]] = []  # (temp, final)
        committed = False
        try:
            for f, filename in zip(files, names):
                dest = folder / filename
                # A filename-independent temp name so a long (but valid) filename cannot overflow NAME_MAX with the staging suffix.
                tmp = folder / f".upload-{_uuid.uuid4().hex}.part"
                staged.append((tmp, dest))
                with open(tmp, "wb") as out:
                    while chunk := await f.read(1024 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > limit_bytes:
                            raise HTTPException(
                                status_code = 413,
                                detail = (
                                    "Dataset upload too large. "
                                    f"Maximum is {get_upload_limit_label()} per upload; "
                                    "add the remaining images in another batch."
                                ),
                            )
                        out.write(chunk)
                # Reject a decompression bomb before commit: a small PNG can pass the byte limit yet decode to huge pixels and OOM the trainer's latent cache.
                if Path(filename).suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS:
                    _validate_uploaded_training_image(tmp, filename)
                uploaded += 1
            # Re-check the interlock immediately before the commit: the entry guard only saw the pre-upload state, so a
            # /diffusion/start could have reserved the slot while we streamed. A 409 here leaves the temps to the finally below.
            _require_diffusion_dataset_mutable()
            # Commit every staged file as one transaction: a plain replace loop is not atomic, so a mid-loop failure leaves earlier
            # destinations overwritten. Back up each pre-existing destination first and restore them all on any failure.
            backups: list[tuple[Path, Optional[Path]]] = []  # (dest, backup path or None)
            installed: list[Path] = []
            try:
                for tmp, dest in staged:
                    backup: Optional[Path] = None
                    if dest.exists():
                        backup = folder / f".upload-backup-{_uuid.uuid4().hex}.part"
                        dest.replace(backup)
                    backups.append((dest, backup))
                    tmp.replace(dest)  # atomic on the same filesystem
                    installed.append(dest)
                committed = True
            except BaseException:
                # Roll back: drop every new version, then restore every displaced original.
                for dest in reversed(installed):
                    try:
                        dest.unlink(missing_ok = True)
                    except OSError:
                        pass
                for dest, backup in reversed(backups):
                    if backup is not None and backup.exists():
                        try:
                            backup.replace(dest)
                        except OSError:
                            pass
                raise
            else:
                for _, backup in backups:
                    if backup is not None:
                        try:
                            backup.unlink(missing_ok = True)
                        except OSError:
                            pass
        finally:
            if not committed:
                for tmp, _ in staged:
                    try:
                        tmp.unlink(missing_ok = True)
                    except OSError:
                        pass

        summary = _diffusion_dataset_summary(folder)
        return DiffusionDatasetUploadResponse(
            name = cleaned,
            path = str(folder),
            image_count = summary.image_count,
            caption_count = summary.caption_count,
            uploaded = uploaded,
        )
    finally:
        _lock.release()


# ── Dataset labeling (per-image caption editing) + one-click example imports ──
# Thumbnails live in a hidden subdir so they never appear in dataset listings or the trainer's image discovery.
_THUMBS_DIRNAME = ".thumbs"
_MAX_CAPTION_CHARS = 2000


def _resolve_dataset_folder(name: str, *, must_exist: bool = True) -> Path:
    """Validate ``name`` (single component, no traversal) and resolve it under the Studio
    datasets root. 404 when a read target is missing."""
    from utils.paths import datasets_root

    cleaned = _clean_diffusion_dataset_name(name)
    root = datasets_root().resolve()
    folder = root / cleaned
    # Reject a symlinked dataset directory and prove the resolved folder stays under root: _safe_dataset_image_path only checks each image path, not the folder.
    if folder.is_symlink():
        raise HTTPException(
            status_code = 400,
            detail = f"Dataset '{cleaned}' must not be a symbolic link.",
        )
    if must_exist and not folder.is_dir():
        raise HTTPException(status_code = 404, detail = f"Dataset '{cleaned}' not found.")
    try:
        folder.resolve(strict = must_exist).relative_to(root)
    except (OSError, ValueError):
        raise HTTPException(
            status_code = 400,
            detail = f"Dataset '{cleaned}' escapes the Studio datasets directory.",
        )
    return folder


# Per-side dimension bound for uploaded training images, matching diffusion._decode_b64_image's 4096px guard, so a compressible PNG cannot smuggle huge pixels past the byte limit.
_MAX_TRAINING_IMAGE_SIDE = 4096


def _validate_uploaded_training_image(path: Path, original_name: str) -> None:
    """Reject an uploaded training image whose decoded dimensions exceed the per-side limit.

    Reads only the header (never img.load()), so a small-payload / huge-dimension file is caught
    before it spikes memory. Bytes PIL cannot identify are left as-is (the upload contract accepts
    arbitrary bytes under an image extension), so only oversized real images change behaviour."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Image.DecompressionBombError:
        # Past Pillow's ~179 MP limit Image.open() raises before .size can be read, with an error deriving straight from Exception, so letting it escape would 500 the upload.
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Image '{original_name}' is too large; maximum is "
                f"{_MAX_TRAINING_IMAGE_SIDE}px per side."
            ),
        )
    except (OSError, UnidentifiedImageError, ValueError):
        return  # not a decodable image -> not a bomb; leave the existing contract
    if width > _MAX_TRAINING_IMAGE_SIDE or height > _MAX_TRAINING_IMAGE_SIDE:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Image '{original_name}' is too large ({width}x{height}); maximum is "
                f"{_MAX_TRAINING_IMAGE_SIDE}px per side."
            ),
        )


def _safe_dataset_image_path(folder: Path, filename: str) -> Path:
    """Resolve ``filename`` to an image path strictly inside ``folder``. Rejects any path
    separators / traversal / null bytes and non-image extensions."""
    raw = filename or ""
    if "/" in raw or "\\" in raw or ".." in raw or "\x00" in raw or raw != Path(raw).name:
        raise HTTPException(status_code = 400, detail = "Invalid image filename.")
    if Path(raw).suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS:
        exts = ", ".join(sorted(_DIFFUSION_DATASET_IMAGE_EXTS))
        raise HTTPException(status_code = 400, detail = f"Not an image file. Allowed: {exts}")
    path = folder / raw
    # Defense in depth: the real path must stay under the dataset folder.
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        raise HTTPException(status_code = 400, detail = "Invalid image filename.")
    return path


def _load_metadata_captions(folder: Path) -> dict[str, str]:
    """Read metadata.jsonl / captions.jsonl into {file_name: caption}, mirroring the
    trainer's discovery (keys file_name/image/file; caption in the ``text`` column)."""
    import json

    out: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = folder / meta_name
        if not meta_path.is_file():
            continue
        # Tolerate a bad upload (invalid UTF-8, or non-object JSON): skip the record so these endpoints do not 500.
        try:
            lines = meta_path.read_text(encoding = "utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            key = row.get("file_name") or row.get("image") or row.get("file")
            value = row.get("text")
            # A JSON null is "no caption", not the string "None".
            if key and value is not None:
                out[str(key)] = str(value)
    return out


def _image_record(
    folder: Path, image_path: Path, meta_captions: dict[str, str]
) -> DiffusionDatasetImageRecord:
    """Build one image record, resolving its caption with sidecar > metadata precedence
    (the same order the trainer uses). A per-image .txt / .caption sidecar wins because
    it is the user's explicit edit from the labeling grid, which must override a
    metadata.jsonl / captions.jsonl row for the image."""
    caption: Optional[str] = None
    source = "none"
    sidecar_present = False
    for ext in (".txt", ".caption"):
        sidecar = image_path.with_suffix(ext)
        if sidecar.is_file():
            sidecar_present = True
            try:
                caption = sidecar.read_text(encoding = "utf-8").strip()
                source = "sidecar"
            except (OSError, UnicodeError):
                # Unreadable / invalid UTF-8 sidecar: UnicodeDecodeError is a ValueError, not an OSError, so an OSError-only guard let
                # it 500 the labeling grid. The trainer treats ANY existing sidecar as the empty tombstone, so show no caption here.
                caption = None
                source = "sidecar"
            break
    if caption is None and not sidecar_present:
        # Basename first, then the relative path as written in the jsonl (as_posix so Windows paths match): discover_image_caption_pairs order.
        meta = meta_captions.get(image_path.name)
        if meta is None:
            try:
                meta = meta_captions.get(image_path.relative_to(folder).as_posix())
            except ValueError:
                meta = None
        if meta is not None:
            caption = meta
            source = "metadata"
    try:
        size_bytes = image_path.stat().st_size
    except OSError:
        size_bytes = 0
    width = height = 0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 -- an unreadable image still lists (0x0) rather than 500
        pass
    return DiffusionDatasetImageRecord(
        filename = image_path.name,
        caption = caption,
        caption_source = source,  # type: ignore[arg-type]
        width = width,
        height = height,
        size_bytes = size_bytes,
    )


@router.get("/diffusion/dataset/{name}/images", response_model = DiffusionDatasetImagesResponse)
async def list_diffusion_dataset_images(
    name: str, current_subject: str = Depends(get_current_subject)
):
    """List every image in a dataset folder with its resolved caption (including
    uncaptioned images), for the labeling grid."""
    folder = _resolve_dataset_folder(name)

    def scan() -> DiffusionDatasetImagesResponse:
        meta = _load_metadata_captions(folder)
        records: list[DiffusionDatasetImageRecord] = []
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS:
                records.append(_image_record(folder, p, meta))
        return DiffusionDatasetImagesResponse(name = folder.name, path = str(folder), images = records)

    return await asyncio.to_thread(scan)


@router.get("/diffusion/dataset/{name}/image/{filename}")
async def get_diffusion_dataset_image(
    name: str,
    filename: str,
    thumb: Optional[int] = None,
    current_subject: str = Depends(get_current_subject),
):
    """Serve a dataset image. ``?thumb=<px>`` returns a cached downscaled JPEG (regenerated
    when the source is newer), used by the labeling grid to stay light."""
    from fastapi.responses import FileResponse

    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")
    if not thumb:
        return FileResponse(str(image_path))

    size = max(32, min(1024, int(thumb)))

    def make_thumb() -> Path:
        from PIL import Image

        thumbs_dir = folder / _THUMBS_DIRNAME
        thumbs_dir.mkdir(exist_ok = True)
        # Key on the full filename, not the stem: two images sharing a stem would collide on one cache file and the mtime-newer entry would be served for both.
        thumb_path = thumbs_dir / f"{image_path.name}_{size}.jpg"
        src_mtime = image_path.stat().st_mtime
        if thumb_path.is_file() and thumb_path.stat().st_mtime >= src_mtime:
            return thumb_path
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            im.save(thumb_path, format = "JPEG", quality = 85)
        return thumb_path

    try:
        thumb_path = await asyncio.to_thread(make_thumb)
    except Exception as e:  # noqa: BLE001 -- fall back to the original on any decode failure
        logger.warning("Thumbnail generation failed for %s: %s", image_path, e)
        return FileResponse(str(image_path))
    return FileResponse(str(thumb_path), media_type = "image/jpeg")


@router.put(
    "/diffusion/dataset/{name}/caption/{filename}",
    response_model = DiffusionDatasetImageRecord,
)
async def set_diffusion_dataset_caption(
    name: str,
    filename: str,
    body: DiffusionCaptionUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Write (or, when blank, clear) an image's ``.txt`` caption sidecar. Returns the
    updated image record."""
    _require_diffusion_dataset_mutable()
    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")
    caption = (body.caption or "").strip()
    if len(caption) > _MAX_CAPTION_CHARS:
        raise HTTPException(
            status_code = 400,
            detail = f"Caption too long (max {_MAX_CAPTION_CHARS} characters).",
        )

    def write() -> DiffusionDatasetImageRecord:
        sidecar = image_path.with_suffix(".txt")
        if caption:
            sidecar.write_text(caption, encoding = "utf-8")
            image_path.with_suffix(".caption").unlink(missing_ok = True)
            return _image_record(folder, image_path, _load_metadata_captions(folder))
        # Blank must actually clear. Unlinking alone would resurface this image's metadata caption, so when one exists write
        # an EMPTY sidecar instead: reader and trainer both treat an existing sidecar as authoritative, a tombstone.
        meta = _load_metadata_captions(folder)
        try:
            rel = image_path.relative_to(folder).as_posix()
        except ValueError:
            rel = image_path.name
        if image_path.name in meta or rel in meta:
            sidecar.write_text("", encoding = "utf-8")
        else:
            sidecar.unlink(missing_ok = True)
        image_path.with_suffix(".caption").unlink(missing_ok = True)
        return _image_record(folder, image_path, meta)

    return await asyncio.to_thread(write)


@router.delete("/diffusion/dataset/{name}/image/{filename}")
async def delete_diffusion_dataset_image(
    name: str,
    filename: str,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Remove an image, its caption sidecars, and any cached thumbnails."""
    _require_diffusion_dataset_mutable()
    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")

    def remove() -> dict:
        import glob as _glob

        image_path.unlink(missing_ok = True)
        # Sidecars are keyed on the STEM, so cat.jpg and cat.png share cat.txt: deleting it with one of them would strip the
        # survivor's caption. New collisions are refused at upload, but hand-made and legacy folders still have them.
        stem_still_used = any(
            p.is_file()
            and p != image_path
            and p.stem == image_path.stem
            and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS
            for p in folder.iterdir()
        )
        if not stem_still_used:
            for ext in (".txt", ".caption"):
                image_path.with_suffix(ext).unlink(missing_ok = True)
        thumbs_dir = folder / _THUMBS_DIRNAME
        if thumbs_dir.is_dir():
            # Thumbs are keyed on the full filename, so match that here; a stem-only glob would strand this image's thumbs or delete a sibling's. Escape the name so a glob metacharacter cannot match siblings.
            for t in thumbs_dir.glob(f"{_glob.escape(image_path.name)}_*.jpg"):
                t.unlink(missing_ok = True)
        return {"deleted": image_path.name}

    return await asyncio.to_thread(remove)


# Curated, license-labelled example datasets for one-click import. ``loader`` picks the materialization strategy:
# "hf_dataset" streams rows from datasets.load_dataset; "imagefolder_jsonl" snapshot-downloads a repo whose captions live in a *.jsonl (file_name/text).
_DATASET_EXAMPLES: list[dict] = [
    {
        "id": "dreambooth-dog",
        "label": "Dog (DreamBooth subject)",
        "repo": "diffusers/dog-example",
        "description": "5 photos of one dog. Teach a subject, then call it with the trigger.",
        "license": "Google, research and demos",
        "image_cap": 10,
        "suggested_trigger": "a photo of sks dog",
        "loader": "hf_dataset",
        "caption_column": None,
        "no_checks": False,
    },
    {
        "id": "tuxemon",
        "label": "Tuxemon (captioned style set)",
        "repo": "linoyts/Tuxemon",
        "description": "Captioned cartoon monster art. A style set, no trigger needed.",
        "license": "cc-by-sa-3.0",
        "image_cap": 60,
        "suggested_trigger": None,
        "loader": "hf_dataset",
        "caption_column": "prompt",
        "no_checks": True,
    },
    {
        "id": "tarot-1920",
        "label": "1920 Tarot (public domain style set)",
        "repo": "multimodalart/1920-raider-waite-tarot-public-domain",
        "description": "Captioned 1920 Raider-Waite tarot art. A permissive style set.",
        "license": "public domain",
        "image_cap": 60,
        "suggested_trigger": None,
        "loader": "imagefolder_jsonl",
        "caption_column": "text",
        "no_checks": True,
    },
    {
        "id": "smithsonian-butterflies",
        "label": "Smithsonian Butterflies",
        "repo": "huggan/smithsonian_butterflies_subset",
        "description": "100 butterfly photos. No captions, so use the trigger prompt.",
        "license": "CC0",
        "image_cap": 100,
        # The metadata columns are species names / boilerplate alt-text, not captions, so train it as a subject set with the trigger prompt instead.
        "suggested_trigger": "a photo of a sks butterfly",
        "loader": "hf_dataset",
        "caption_column": None,
        "no_checks": False,
    },
    {
        "id": "pixel-nouns",
        "label": "Nouns (pixel avatars)",
        "repo": "m1guelpf/nouns",
        "description": "100 captioned pixel-art avatars. A style set, no trigger needed.",
        "license": "cc0-1.0",
        "image_cap": 100,
        "suggested_trigger": None,
        "loader": "hf_dataset",
        "caption_column": "text",
        "no_checks": False,
    },
]


def _example_by_id(example_id: str) -> dict:
    for entry in _DATASET_EXAMPLES:
        if entry["id"] == example_id:
            return entry
    raise HTTPException(status_code = 404, detail = f"Unknown example dataset '{example_id}'.")


@router.get("/diffusion/dataset-examples", response_model = DiffusionDatasetExamplesResponse)
async def list_diffusion_dataset_examples(current_subject: str = Depends(get_current_subject)):
    """List the curated example datasets available for one-click import."""
    return DiffusionDatasetExamplesResponse(
        examples = [
            DiffusionDatasetExample(
                id = e["id"],
                label = e["label"],
                repo = e["repo"],
                description = e["description"],
                license = e["license"],
                image_cap = e["image_cap"],
                suggested_trigger = e["suggested_trigger"],
            )
            for e in _DATASET_EXAMPLES
        ]
    )


def _detect_image_column(features) -> Optional[str]:
    """Return the first datasets Image-feature column name, else None."""
    try:
        from datasets import Image as HFImage
    except Exception:  # noqa: BLE001
        HFImage = None  # type: ignore[assignment]
    for col, feat in features.items():
        if HFImage is not None and isinstance(feat, HFImage):
            return col
        if type(feat).__name__ == "Image":
            return col
    return None


def _detect_image_column_from_row(row: dict) -> Optional[str]:
    """Image column picked from one materialized row, for a streamed dataset that arrives with no
    feature metadata to inspect."""
    try:
        from PIL.Image import Image as PILImage
    except Exception:  # noqa: BLE001 -- no Pillow -> the caller reports "no image column"
        return None
    for col, value in row.items():
        if isinstance(value, PILImage):
            return col
    return None


def _detect_caption_column(entry: dict, columns: list[str]) -> Optional[str]:
    """Pick the caption column: the entry's declared one if present, else a common name."""
    declared = entry.get("caption_column")
    if declared and declared in columns:
        return declared
    for cand in ("text", "prompt", "caption", "captions"):
        if cand in columns:
            return cand
    return None


def _materialize_hf_dataset(entry: dict, dest: Path, cap: int) -> int:
    """Stream rows from datasets.load_dataset into ``dest`` as numbered images + optional
    .txt sidecars. Returns the number of images written."""
    from datasets import load_dataset

    kwargs = {"split": "train"}
    if entry.get("no_checks"):
        kwargs["verification_mode"] = "no_checks"
    # Stream rather than prepare the whole split: the loop keeps at most `cap` rows while these curated repos run to tens
    # of thousands of rows a prepared load downloads in full. A repo that cannot stream falls back to the prepared load.
    try:
        ds = load_dataset(entry["repo"], streaming = True, **kwargs)
        features = ds.features
    except Exception:  # noqa: BLE001 -- not streamable; the prepared load is the fallback
        ds = load_dataset(entry["repo"], **kwargs)
        features = ds.features
    # Streaming can hand back a dataset whose features are only known once a row is read, so resolve the columns from the first row then.
    image_col = _detect_image_column(features) if features else None
    if image_col is None and features:
        raise HTTPException(
            status_code = 502,
            detail = f"'{entry['repo']}' has no image column to import.",
        )
    caption_col = _detect_caption_column(entry, list(features.keys())) if features else None
    written = 0
    for row in ds:
        if written >= cap:
            break
        if image_col is None:
            image_col = _detect_image_column_from_row(row)
            if image_col is None:
                raise HTTPException(
                    status_code = 502,
                    detail = f"'{entry['repo']}' has no image column to import.",
                )
            caption_col = _detect_caption_column(entry, list(row.keys()))
        img = row[image_col]
        if img is None:
            continue
        img = img.convert("RGB")
        stem = f"img_{written:04d}"
        img.save(dest / f"{stem}.png", format = "PNG")
        if caption_col:
            cap_text = row.get(caption_col)
            if cap_text:
                (dest / f"{stem}.txt").write_text(str(cap_text).strip(), encoding = "utf-8")
        written += 1
    return written


def _materialize_imagefolder_jsonl(entry: dict, dest: Path, cap: int) -> int:
    """Snapshot-download a dataset repo whose captions live in *.jsonl (file_name/text),
    then copy referenced images + write .txt sidecars. Returns images written."""
    import json
    import shutil

    from huggingface_hub import snapshot_download

    caption_col = entry.get("caption_column") or "text"
    snap = Path(
        snapshot_download(
            entry["repo"],
            repo_type = "dataset",
            allow_patterns = [
                "*.jsonl",
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.webp",
                "*.bmp",
                "**/*.jpg",
                "**/*.jpeg",
                "**/*.png",
                "**/*.webp",
                "**/*.bmp",
            ],
        )
    )
    # Map basename -> caption from every jsonl carrying file_name + caption column.
    captions: dict[str, str] = {}
    for jf in sorted(snap.rglob("*.jsonl")):
        for line in jf.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = row.get("file_name") or row.get("image") or row.get("file")
            value = row.get(caption_col)
            # A JSON null is "no caption", not the string "None".
            if fn and value is not None:
                # First writer wins over sorted manifests, for deterministic results.
                captions.setdefault(Path(str(fn)).name, str(value))
    # Copy images (those with a caption first, so a cap keeps captioned pairs).
    images = sorted(
        p
        for p in snap.rglob("*")
        if p.is_file() and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS
    )
    images.sort(key = lambda p: (p.name not in captions, p.name))
    written = 0
    for src in images:
        if written >= cap:
            break
        stem = f"img_{written:04d}"
        shutil.copyfile(src, dest / f"{stem}{src.suffix.lower()}")
        cap_text = captions.get(src.name)
        if cap_text:
            (dest / f"{stem}.txt").write_text(cap_text.strip(), encoding = "utf-8")
        written += 1
    return written


@router.post("/diffusion/dataset/import-example", response_model = DiffusionDatasetImportResponse)
async def import_diffusion_dataset_example(
    body: DiffusionDatasetImportRequest,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Materialize a curated example dataset into a Studio dataset folder (images + .txt
    captions), ready to train. Idempotent: a folder that already holds images is returned
    as-is rather than re-downloaded."""
    _require_diffusion_dataset_mutable()
    entry = _example_by_id(body.id)
    folder = _resolve_dataset_folder(body.name or entry["id"], must_exist = False)

    def do_import() -> DiffusionDatasetImportResponse:
        folder.mkdir(parents = True, exist_ok = True)
        if _diffusion_dataset_summary(folder).image_count > 0:
            return _import_response(entry, folder, imported = 0)
        # One import at a time per dataset folder: the training interlock COUNTS mutations rather than excluding them, so two
        # imports into the same empty name both passed the emptiness check and merged. Refusing the second is honest.
        lock = _dataset_import_lock(folder)
        if not lock.acquire(blocking = False):
            raise HTTPException(
                status_code = 409,
                detail = (
                    f"An import into '{folder.name}' is already running. Wait for it to finish, "
                    "then reload the dataset list."
                ),
            )
        try:
            return _do_import_locked(entry, folder)
        finally:
            lock.release()

    def _do_import_locked(entry: dict, folder: Path) -> DiffusionDatasetImportResponse:
        import os
        import shutil
        import tempfile

        imported = 0
        # Re-read under the lock: a winner may have promoted its staging dir while this request was checking, so returning the folder as-is matches the idempotent path.
        existing = _diffusion_dataset_summary(folder)
        if existing.image_count == 0:
            cap = int(entry["image_cap"])
            # Materialize into a private staging dir and promote only after the whole import succeeds, so a partial materialize
            # leaves only that dir. Staged as a hidden same-filesystem sibling so promotion is an atomic rename.
            staging = Path(tempfile.mkdtemp(dir = folder.parent, prefix = f".{folder.name}.import-"))
            # Superseded same-name entries are parked here rather than deleted, so a failed promotion can put them back too.
            rescue = Path(tempfile.mkdtemp(dir = folder.parent, prefix = f".{folder.name}.rescue-"))
            # (entry's new home, where it came from) for every pre-existing entry moved out of the folder.
            folded: list[tuple[Path, Path]] = []

            def restore_folded() -> None:
                """Undo the fold-in so a failed promotion leaves the dataset as it was."""
                folder.mkdir(parents = True, exist_ok = True)
                for moved, original in folded:
                    try:
                        if moved.exists() and not original.exists():
                            shutil.move(str(moved), str(original))
                    except OSError:
                        # Best effort: one unrestorable entry must not mask the original failure.
                        pass

            try:
                try:
                    if entry["loader"] == "imagefolder_jsonl":
                        imported = _materialize_imagefolder_jsonl(entry, staging, cap)
                    else:
                        imported = _materialize_hf_dataset(entry, staging, cap)
                except HTTPException:
                    raise
                except Exception as e:  # noqa: BLE001 -- surface a readable fetch/parse failure
                    raise HTTPException(
                        status_code = 502,
                        detail = f"Could not import '{entry['repo']}': {e}",
                    )
                if imported == 0:
                    raise HTTPException(
                        status_code = 502,
                        detail = f"No images found in '{entry['repo']}'.",
                    )
                # Promote the fully-materialized staging dir as a UNIT: a same-filesystem rename is atomic, so a hard process death
                # leaves either the old folder or the finished import. rmdir needs an empty target, so fold any pre-existing files (a .thumbs cache, an older metadata.jsonl) INTO staging first and keep one atomic promotion.
                try:
                    for p in sorted(folder.iterdir()):
                        # Same name in both: the imported file wins, as the previous per-file move did. Park the old one in the rescue dir so the folder can be emptied for the rename without destroying it.
                        dest = (rescue if (staging / p.name).exists() else staging) / p.name
                        shutil.move(str(p), str(dest))
                        folded.append((dest, p))
                    os.rmdir(folder)
                    os.replace(str(staging), str(folder))
                except (OSError, shutil.Error) as e:
                    # Every step here can fail (an unmovable entry, a folder that gained a file, a rename held by antivirus), and by then the folder's entries live only in the staging/rescue dirs the finally deletes.
                    restore_folded()
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            f"Could not update '{folder.name}' with the imported example "
                            f"({getattr(e, 'strerror', None) or e}). Nothing was written; try again."
                        ),
                    )
            finally:
                shutil.rmtree(staging, ignore_errors = True)
                shutil.rmtree(rescue, ignore_errors = True)
        return _import_response(entry, folder, imported = imported)

    return await asyncio.to_thread(do_import)
