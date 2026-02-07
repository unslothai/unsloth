"""
Training API routes
"""
import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Optional
import logging
import asyncio
from datetime import datetime
import threading

# Add backend directory to path
# The backend code should be in the same directory structure
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.training import get_training_backend
except ImportError:
    # Fallback: try to import from parent directory
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.training import get_training_backend

# Auth
from auth.authentication import get_current_subject

from models import (
    TrainingStartRequest,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure logger
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
        backend = get_training_backend()

        # Generate job ID and attach to backend for later status/progress calls
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backend.current_job_id = job_id

        # Check if training is already active
        if backend.is_training_active():
            existing_job_id: Optional[str] = getattr(backend, "current_job_id", "")
            return TrainingJobResponse(
                job_id=existing_job_id or job_id,
                status="error",
                message=(
                    "Training is already in progress. "
                    "Stop current training before starting a new one."
                ),
                error="Training already active",
            )

        # Validate dataset paths if provided
        if request.local_datasets:
            validated_datasets = []
            # Get the backend directory (where this file is located)
            backend_dir = Path(__file__).parent.parent
            assets_datasets_dir = backend_dir / "assets" / "datasets"

            for dataset_path in request.local_datasets:
                dataset_file = Path(dataset_path)

                # If not absolute, try multiple locations
                if not dataset_file.is_absolute():
                    # First try: relative to current working directory
                    candidate = Path.cwd() / dataset_path
                    if not candidate.exists():
                        # Second try: relative to assets/datasets folder
                        candidate = assets_datasets_dir / dataset_path
                    if not candidate.exists():
                        # Third try: just the filename in assets/datasets
                        candidate = assets_datasets_dir / dataset_file.name
                    dataset_file = candidate

                if not dataset_file.exists():
                    logger.warning(
                        f"Dataset file not found: {dataset_path} (resolved: {dataset_file})"
                    )
                else:
                    logger.info(f"Found dataset file: {dataset_file}")
                validated_datasets.append(str(dataset_file))
            request.local_datasets = validated_datasets

        # Convert request to kwargs for backend
        training_kwargs = {
            "model_name": request.model_name,
            "training_type": request.training_type,
            "hf_token": request.hf_token or "",
            "load_in_4bit": request.load_in_4bit,
            "max_seq_length": request.max_seq_length,
            "hf_dataset": request.hf_dataset or "",
            "local_datasets": request.local_datasets,
            "format_type": request.format_type,
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
            "enable_wandb": request.enable_wandb,
            "wandb_token": request.wandb_token or "",
            "wandb_project": request.wandb_project or "",
            "enable_tensorboard": request.enable_tensorboard,
            "tensorboard_dir": request.tensorboard_dir or "",
        }

        # Set initial "preparing" state
        try:
            backend.trainer._update_progress(
                status_message="Initializing training...",
                is_training=False,
            )
        except Exception:
            pass

        def run_training():
            try:
                logger.info(
                    f"Starting training job {job_id} with model {request.model_name}"
                )

                # Update status to show we're loading model
                try:
                    backend.trainer._update_progress(status_message="Loading model...")
                except Exception as e:
                    logger.error(f"Error updating progress: {e}")

                # Consume the generator - this actually runs the training
                update_count = 0
                for _update_tuple in backend.start_training(**training_kwargs):
                    update_count += 1
                    if update_count % 10 == 0:
                        logger.info(f"Training progress update #{update_count}")

                logger.info(f"Training job {job_id} completed successfully")

            except Exception as e:
                logger.error(f"Training error in job {job_id}: {e}", exc_info=True)
                try:
                    backend.trainer._update_progress(
                        error=str(e),
                        is_training=False,
                    )
                except Exception as update_error:
                    logger.error(f"Failed to update progress: {update_error}")

        # Start training in a daemon thread
        training_thread = threading.Thread(
            target=run_training,
            daemon=True,
            name=f"Training-{job_id}",
        )
        training_thread.start()

        # Store thread reference for status checking
        backend._training_thread = training_thread

        # Give it a moment to start
        import time

        time.sleep(0.5)

        # Verify training thread is alive
        if not training_thread.is_alive():
            logger.warning(f"Training thread died immediately for job {job_id}")
            return TrainingJobResponse(
                job_id=job_id,
                status="error",
                message=(
                    "Training thread failed to start. "
                    "Check server logs for details."
                ),
                error="Thread not alive",
            )

        return TrainingJobResponse(
            job_id=job_id,
            status="queued",
            message="Training job queued and starting in background",
            error=None,
        )

    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}",
        )


@router.post("/stop")
async def stop_training(
    current_subject: str = Depends(get_current_subject),
):
    """
    Stop the currently running training job.
    """
    try:
        backend = get_training_backend()
        
        if not backend.is_training_active():
            return {
                "status": "idle",
                "message": "No training job is currently running"
            }
        
        # Call backend stop method
        backend.stop_training()
        
        return {
            "status": "stopped",
            "message": "Training job stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop training: {str(e)}"
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
        job_id: str = getattr(backend, "current_job_id", "")

        # Check if training is active
        is_active = backend.is_training_active()

        # Check if there's a training thread running (preparation phase)
        has_thread = (
            hasattr(backend, "_training_thread")
            and backend._training_thread
            and backend._training_thread.is_alive()
        )

        # Get progress info from trainer
        try:
            progress = backend.trainer.get_training_progress()
        except Exception:
            progress = None

        status_message = (
            getattr(progress, "status_message", None) if progress else None
        ) or "Ready to train"
        error_message = getattr(progress, "error", None) if progress else None

        # Derive high-level phase
        if error_message:
            phase = "error"
        elif is_active:
            msg_lower = status_message.lower()
            if "loading" in msg_lower:
                phase = "loading_model"
            elif any(
                k in msg_lower for k in ["preparing", "initializing", "configuring"]
            ):
                phase = "configuring"
            else:
                phase = "training"
        elif progress and getattr(progress, "is_completed", False):
            phase = "completed"
        elif has_thread:
            phase = "loading_model"
        else:
            phase = "idle"

        details = None
        if progress:
            details = {
                "epoch": getattr(progress, "epoch", 0),
                "step": getattr(progress, "step", 0),
                "total_steps": getattr(progress, "total_steps", 0),
                "loss": getattr(progress, "loss", 0.0),
                "learning_rate": getattr(progress, "learning_rate", 0.0),
            }

        return TrainingStatus(
            job_id=job_id,
            phase=phase,
            is_training_running=is_active,
            message=status_message,
            error=error_message,
            details=details,
        )
            
    except Exception as e:
        logger.error(f"Error getting training status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        )


@router.get("/metrics")
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

        # Get current values
        current_loss = loss_history[-1] if loss_history else None
        current_lr = lr_history[-1] if lr_history else None
        current_step = step_history[-1] if step_history else None

        # Keep metrics as a simple JSON payload instead of a Pydantic model
        return {
            "loss_history": loss_history,
            "lr_history": lr_history,
            "step_history": step_history,
            "current_loss": current_loss,
            "current_lr": current_lr,
            "current_step": current_step,
        }
        
    except Exception as e:
        logger.error(f"Error getting training metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training metrics: {str(e)}"
        )


@router.get("/progress")
async def stream_training_progress(
    current_subject: str = Depends(get_current_subject),
):
    """
    Stream training progress updates using Server-Sent Events (SSE).
    
    This endpoint provides real-time updates on training progress.
    """
    async def event_generator():
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "")

        # Helper to build a TrainingProgress payload from raw values
        def build_progress(
            step: int,
            loss: float,
            learning_rate: float,
            total_steps: int,
            epoch: Optional[int] = None,
        ) -> TrainingProgress:
            total = max(total_steps, 0)
            if step < 0 or total == 0:
                progress_percent = 0.0
            else:
                progress_percent = (
                    float(step) / float(total) * 100.0 if total > 0 else 0.0
                )

            return TrainingProgress(
                job_id=job_id,
                step=step,
                total_steps=total,
                loss=loss,
                learning_rate=learning_rate,
                progress_percent=progress_percent,
                epoch=epoch,
                elapsed_seconds=None,
                eta_seconds=None,
                grad_norm=None,
                num_tokens=None,
            )

        # Send initial status
        is_active = backend.is_training_active()
        tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
        initial_total_steps = getattr(tp, "total_steps", 0) if tp else 0
        initial_epoch = getattr(tp, "epoch", None) if tp else None

        initial_progress = build_progress(
            step=0,
            loss=0.0,
            learning_rate=0.0,
            total_steps=initial_total_steps,
            epoch=initial_epoch,
        )
        yield f"data: {initial_progress.model_dump_json()}\n\n"

        # If not active, check if there's any history
        if not is_active:
            if backend.step_history:
                # Training completed - send final metrics
                final_step = backend.step_history[-1]
                final_loss = backend.loss_history[-1] if backend.loss_history else 0.0
                final_lr = backend.lr_history[-1] if backend.lr_history else 0.0
                final_total_steps = (
                    getattr(tp, "total_steps", final_step) if tp else final_step
                )
                final_epoch = getattr(tp, "epoch", None) if tp else None
                yield f"data: {build_progress(final_step, final_loss, final_lr, final_total_steps, final_epoch).model_dump_json()}\n\n"
            else:
                yield f"data: {build_progress(-1, 0.0, 0.0, 0).model_dump_json()}\n\n"
            return
        
        # Poll for updates while training is active
        last_step = -1
        no_update_count = 0
        max_no_updates = 300  # Timeout after 5 minutes
        
        while backend.is_training_active():
            try:
                # Get current metrics
                if backend.step_history:
                    current_step = backend.step_history[-1]
                    current_loss = backend.loss_history[-1] if backend.loss_history else 0.0
                    current_lr = backend.lr_history[-1] if backend.lr_history else 0.0
                    tp_inner = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    current_total_steps = (
                        getattr(tp_inner, "total_steps", current_step)
                        if tp_inner
                        else current_step
                    )
                    current_epoch = getattr(tp_inner, "epoch", None) if tp_inner else None

                    # Only send if step changed
                    if current_step != last_step:
                        progress_payload = build_progress(
                            current_step,
                            current_loss,
                            current_lr,
                            current_total_steps,
                            current_epoch,
                        )
                        yield f"data: {progress_payload.model_dump_json()}\n\n"
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
                            )
                            yield f"data: {heartbeat_payload.model_dump_json()}\n\n"
                else:
                    # No steps yet, but training is active
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        preparing_payload = build_progress(0, 0.0, 0.0, 0)
                        yield f"data: {preparing_payload.model_dump_json()}\n\n"
                
                # Timeout check
                if no_update_count > max_no_updates:
                    logger.warning("Progress stream timeout - no updates received")
                    timeout_payload = build_progress(last_step, 0.0, 0.0, 0)
                    yield f"data: {timeout_payload.model_dump_json()}\n\n"
                    break
                
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                logger.error(f"Error in progress stream: {e}", exc_info=True)
                error_payload = build_progress(0, 0.0, 0.0, 0)
                yield f"data: {error_payload.model_dump_json()}\n\n"
                break

        # Send final status
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else 0.0
        final_lr = backend.lr_history[-1] if backend.lr_history else 0.0
        final_tp = getattr(
            getattr(backend, "trainer", None), "training_progress", None
        )
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
        )
        yield f"data: {final_payload.model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

