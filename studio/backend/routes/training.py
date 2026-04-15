# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training API routes
"""

import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Any
import structlog
from loggers import get_logger
import asyncio
from datetime import datetime
import uuid as _uuid

# Add backend directory to path
# The backend code should be in the same directory structure
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.training import get_training_backend
    from utils.models.model_config import load_model_defaults
    from utils.paths import resolve_dataset_path
except ImportError:
    # Fallback: try to import from parent directory
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.training import get_training_backend
    from utils.models.model_config import load_model_defaults
    from utils.paths import resolve_dataset_path

# Auth
from auth.authentication import get_current_subject

from models import (
    TrainingStartRequest,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)
from models.responses import TrainingStopResponse, TrainingMetricsResponse
from pydantic import BaseModel as PydanticBaseModel


class TrainingStopRequest(PydanticBaseModel):
    save: bool = True


router = APIRouter()
logger = get_logger(__name__)


def _validate_local_dataset_paths(
    paths: list[str], label: str = "Local dataset"
) -> list[str]:
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
async def get_hardware_utilization(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get a live snapshot of GPU hardware utilization.

    Designed to be polled by the frontend during training.
    Returns live GPU memory usage information for the active backend.
    """
    from utils.hardware import get_gpu_utilization

    return get_gpu_utilization()


@router.get("/hardware/visible")
async def get_visible_hardware_utilization(
    current_subject: str = Depends(get_current_subject),
):
    from utils.hardware import get_visible_gpu_utilization

    return get_visible_gpu_utilization()


@router.post("/start")
async def start_training(
    request: TrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Start a training job.

    This endpoint initiates training in the background and returns immediately.
    Use the /status endpoint to check training progress.
    """
    try:
        logger.info(f"Starting training job with model: {request.model_name}")

        # NOTE: No in-process ensure_transformers_version() call here.
        # The subprocess (worker.py) activates the correct version in a
        # fresh Python interpreter before importing any ML libraries.

        backend = get_training_backend()

        # Check if training is already active (before mutating any state)
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

        # Generate job ID — passed into start_training() which sets it on the
        # backend only after confirming the old pump thread is dead.
        job_id = (
            f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
        )

        # Validate dataset paths if provided
        if request.local_datasets:
            request.local_datasets = _validate_local_dataset_paths(
                request.local_datasets, "Local dataset"
            )
        if request.local_eval_datasets and request.eval_steps > 0:
            request.local_eval_datasets = _validate_local_dataset_paths(
                request.local_eval_datasets, "Local eval dataset"
            )

        # Convert request to kwargs for backend
        training_kwargs = {
            "model_name": request.model_name,
            "training_type": request.training_type,
            "hf_token": request.hf_token or "",
            "load_in_4bit": request.load_in_4bit,
            "max_seq_length": request.max_seq_length,
            "hf_dataset": request.hf_dataset or "",
            "local_datasets": request.local_datasets,
            "local_eval_datasets": request.local_eval_datasets,
            "format_type": request.format_type,
            "subset": request.subset,
            "train_split": request.train_split,
            "eval_split": request.eval_split,
            "eval_steps": request.eval_steps,
            "dataset_slice_start": request.dataset_slice_start,
            "dataset_slice_end": request.dataset_slice_end,
            "custom_format_mapping": request.custom_format_mapping,
            "num_epochs": request.num_epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "warmup_steps": request.warmup_steps,
            "warmup_ratio": request.warmup_ratio,
            "max_steps": request.max_steps,
            "save_steps": request.save_steps,
            "weight_decay": request.weight_decay,
            "random_seed": request.random_seed,
            "packing": request.packing,
            "optim": request.optim,
            "lr_scheduler_type": request.lr_scheduler_type,
            "use_lora": request.use_lora,
            "lora_r": request.lora_r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "target_modules": request.target_modules
            if request.target_modules
            else None,
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
            "trust_remote_code": request.trust_remote_code,
            "gpu_ids": request.gpu_ids,
        }

        # Training page has no trust_remote_code toggle — the value comes from
        # YAML model defaults applied when the user selects a model.  As a safety
        # net, consult the YAML directly so models that need it always get it.
        if not training_kwargs["trust_remote_code"]:
            model_defaults = load_model_defaults(request.model_name)
            yaml_trust = model_defaults.get("training", {}).get(
                "trust_remote_code", False
            )
            if yaml_trust:
                logger.info(
                    f"YAML config sets trust_remote_code=True for {request.model_name}"
                )
                training_kwargs["trust_remote_code"] = True

        # Free GPU memory: shut down any running inference/export subprocesses
        # before training starts (they'd compete for VRAM otherwise)
        try:
            from core.inference import get_inference_backend

            inf_backend = get_inference_backend()
            if inf_backend.active_model_name:
                logger.info(
                    "Unloading inference model '%s' to free GPU memory for training",
                    inf_backend.active_model_name,
                )
                inf_backend._shutdown_subprocess()
                inf_backend.active_model_name = None
                inf_backend.models.clear()
        except Exception as e:
            logger.warning("Could not unload inference model: %s", e)

        try:
            from core.export import get_export_backend

            exp_backend = get_export_backend()
            if exp_backend.current_checkpoint:
                logger.info(
                    "Shutting down export subprocess to free GPU memory for training"
                )
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        # start_training now spawns a subprocess (non-blocking)
        success = backend.start_training(job_id = job_id, **training_kwargs)

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

    except ValueError as e:
        logger.warning("Rejected training GPU selection: %s", e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to start training: {str(e)}",
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

        # Call backend stop method
        backend.stop_training(save = body.save)

        return TrainingStopResponse(
            status = "stopped",
            message = "Stop requested. Training will stop at the next safe step.",
        )

    except Exception as e:
        logger.error(f"Error stopping training: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to stop training: {str(e)}"
        )


@router.post("/reset")
async def reset_training(
    current_subject: str = Depends(get_current_subject),
):
    """
    Reset training state so the user can return to configuration.
    """
    try:
        backend = get_training_backend()
        is_active = backend.is_training_active()

        if is_active:
            if backend._cancel_requested:
                # Cancel (save=False) was requested — force-terminate so we can reset immediately
                logger.info(
                    "Force-terminating subprocess for immediate reset (cancel path)"
                )
                backend.force_terminate()
            else:
                logger.warning(
                    "Rejected reset while training active: is_active=%s", is_active
                )
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
        logger.error(f"Error resetting training: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to reset training: {str(e)}",
        )


@router.get("/status")
async def get_training_status(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get the current training status.
    """
    try:
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "") or ""

        # Check if training is active
        is_active = backend.is_training_active()

        # Get progress info from trainer
        try:
            progress = backend.trainer.get_training_progress()
        except Exception:
            progress = None

        status_message = (
            getattr(progress, "status_message", None) if progress else None
        ) or "Ready to train"
        error_message = getattr(progress, "error", None) if progress else None

        # Check if training was stopped by user
        trainer_stopped = getattr(backend, "_should_stop", False)

        # Derive high-level phase
        if error_message:
            phase = "error"
        elif is_active:
            msg_lower = status_message.lower()
            if "loading" in msg_lower or "importing" in msg_lower:
                phase = "loading_model"
            elif any(
                k in msg_lower for k in ["preparing", "initializing", "configuring"]
            ):
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

        # Build metric history for chart recovery after SSE reconnection
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
        logger.error(f"Error getting training status: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get training status: {str(e)}"
        )


@router.get("/metrics", response_model = TrainingMetricsResponse)
async def get_training_metrics(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get training metrics (loss, learning rate, steps).
    """
    try:
        backend = get_training_backend()

        # Get metrics from backend
        loss_history = backend.loss_history
        lr_history = backend.lr_history
        step_history = backend.step_history
        grad_norm_history = getattr(backend, "grad_norm_history", [])
        grad_norm_step_history = getattr(backend, "grad_norm_step_history", [])

        # Get current values
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
        logger.error(f"Error getting training metrics: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get training metrics: {str(e)}"
        )


@router.get("/progress")
async def stream_training_progress(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Stream training progress updates using Server-Sent Events (SSE).

    This endpoint provides real-time updates on training progress.
    Supports reconnection via the SSE spec:
      - Sends `id:` with each event so the browser tracks position.
      - Sends `retry:` to control reconnection interval.
      - Sends named `event:` types (progress, heartbeat, complete, error).
      - Reads `Last-Event-ID` header on reconnect to replay missed steps.
    """
    # Read Last-Event-ID header for reconnection resume
    last_event_id = request.headers.get("last-event-id")
    resume_from_step: Optional[int] = None
    if last_event_id is not None:
        try:
            resume_from_step = int(last_event_id)
            logger.info(f"SSE reconnect: resuming from step {resume_from_step}")
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
                progress_percent = (
                    float(step) / float(total) * 100.0 if total > 0 else 0.0
                )

            # Get actual values from progress object if available
            elapsed_seconds = (
                getattr(progress, "elapsed_seconds", None) if progress else None
            )
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
        # Tell the browser to reconnect after 3 seconds if the connection drops
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
                    loss_val = (
                        backend.loss_history[i]
                        if i < len(backend.loss_history)
                        else None
                    )
                    lr_val = (
                        backend.lr_history[i] if i < len(backend.lr_history) else None
                    )
                    tp_replay = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    total_replay = (
                        getattr(tp_replay, "total_steps", step_val)
                        if tp_replay
                        else step_val
                    )
                    epoch_replay = (
                        getattr(tp_replay, "epoch", None) if tp_replay else None
                    )
                    payload = build_progress(
                        step_val,
                        loss_val,
                        lr_val,
                        total_replay,
                        epoch_replay,
                        progress = tp_replay,
                        grad_norm_override = grad_norm_by_step.get(step_val),
                    )
                    yield format_sse(
                        payload.model_dump_json(), event = "progress", event_id = step_val
                    )
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
            yield format_sse(
                initial_progress.model_dump_json(), event = "progress", event_id = 0
            )

            # If not active, send final state and exit
            if not is_active:
                if backend.step_history:
                    final_step = backend.step_history[-1]
                    final_loss = (
                        backend.loss_history[-1] if backend.loss_history else None
                    )
                    final_lr = backend.lr_history[-1] if backend.lr_history else None
                    final_total_steps = (
                        getattr(tp, "total_steps", final_step) if tp else final_step
                    )
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
                        build_progress(
                            -1, None, None, 0, progress = tp
                        ).model_dump_json(),
                        event = "complete",
                        event_id = 0,
                    )
                return

        # ── Live polling loop ────────────────────────────────────
        last_step = resume_from_step if resume_from_step is not None else -1
        no_update_count = 0
        max_no_updates = (
            1800  # Timeout after 30 minutes (large models need time for compilation)
        )

        while backend.is_training_active():
            try:
                if backend.step_history:
                    current_step = backend.step_history[-1]
                    current_loss = (
                        backend.loss_history[-1] if backend.loss_history else None
                    )
                    current_lr = backend.lr_history[-1] if backend.lr_history else None
                    tp_inner = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    current_total_steps = (
                        getattr(tp_inner, "total_steps", current_step)
                        if tp_inner
                        else current_step
                    )
                    current_epoch = (
                        getattr(tp_inner, "epoch", None) if tp_inner else None
                    )

                    # Only send if step changed
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
                    else:
                        no_update_count += 1
                        # Send heartbeat every 10 seconds
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
                    # No steps yet, but training is active (model loading, etc.)
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        # Pull total_steps and status from trainer so
                        # the frontend can show "Tokenizing…" etc.
                        tp_prep = getattr(
                            getattr(backend, "trainer", None),
                            "training_progress",
                            None,
                        )
                        prep_total = (
                            getattr(tp_prep, "total_steps", 0) if tp_prep else 0
                        )
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

                # Timeout check
                if no_update_count > max_no_updates:
                    logger.warning("Progress stream timeout - no updates received")
                    tp_timeout = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    timeout_payload = build_progress(
                        last_step, None, None, 0, progress = tp_timeout
                    )
                    yield format_sse(
                        timeout_payload.model_dump_json(),
                        event = "error",
                        event_id = last_step if last_step >= 0 else 0,
                    )
                    break

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error in progress stream: {e}", exc_info = True)
                tp_error = getattr(
                    getattr(backend, "trainer", None), "training_progress", None
                )
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
        final_total_steps = (
            getattr(final_tp, "total_steps", final_step) if final_tp else final_step
        )
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
