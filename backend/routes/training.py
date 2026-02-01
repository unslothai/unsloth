"""
Training API routes
"""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict
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
    from backend.training import get_training_backend
except ImportError:
    # Fallback: try to import from parent directory
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from backend.training import get_training_backend

from models.training import (
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
    TrainingMetricsResponse,
    TrainingProgressResponse,
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
async def start_training(request: TrainingStartRequest):
    """
    Start a training job.
    
    This endpoint initiates training in the background and returns immediately.
    Use the /status endpoint to check training progress.
    """
    try:
        logger.info(f"Starting training job with model: {request.model_name}")
        backend = get_training_backend()
        
        # Check if training is already active
        if backend.is_training_active():
            return TrainingStartResponse(
                status="error",
                message="Training is already in progress. Stop current training before starting a new one.",
                error="Training already active"
            )
        
        # Validate dataset paths if provided
        if request.local_datasets:
            validated_datasets = []
            # Get the backend directory (where this file is located)
            backend_dir = Path(__file__).parent.parent
            utils_datasets_dir = backend_dir / "utils" / "datasets"
            
            for dataset_path in request.local_datasets:
                dataset_file = Path(dataset_path)
                
                # If not absolute, try multiple locations
                if not dataset_file.is_absolute():
                    # First try: relative to current working directory
                    candidate = Path.cwd() / dataset_path
                    if not candidate.exists():
                        # Second try: relative to utils/datasets folder
                        candidate = utils_datasets_dir / dataset_path
                    if not candidate.exists():
                        # Third try: just the filename in utils/datasets
                        candidate = utils_datasets_dir / dataset_file.name
                    dataset_file = candidate
                
                if not dataset_file.exists():
                    logger.warning(f"Dataset file not found: {dataset_path} (resolved: {dataset_file})")
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
            "use_lora": request.use_lora,
            "lora_r": request.lora_r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "target_modules": request.target_modules if request.target_modules else None,
            "gradient_checkpointing": request.gradient_checkpointing.strip() if request.gradient_checkpointing and request.gradient_checkpointing.strip() else "unsloth",
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
            "optim": request.optim,
            "lr_scheduler_type": request.lr_scheduler_type,
        }
        
        # Generate job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set initial "preparing" state
        try:
            backend.trainer._update_progress(
                status_message="Initializing training...",
                is_training=False
            )
        except:
            pass
        
        def run_training():
            try:
                logger.info(f"Starting training job {job_id} with model {request.model_name}")
                
                # Update status to show we're loading model
                try:
                    backend.trainer._update_progress(status_message="Loading model...")
                except Exception as e:
                    logger.error(f"Error updating progress: {e}")
                
                # Consume the generator - this actually runs the training
                update_count = 0
                for update_tuple in backend.start_training(**training_kwargs):
                    update_count += 1
                    if update_count % 10 == 0:
                        logger.info(f"Training progress update #{update_count}")
                
                logger.info(f"Training job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Training error in job {job_id}: {e}", exc_info=True)
                try:
                    backend.trainer._update_progress(
                        error=str(e),
                        is_training=False
                    )
                except Exception as update_error:
                    logger.error(f"Failed to update progress: {update_error}")
        
        # Start training in a daemon thread
        training_thread = threading.Thread(target=run_training, daemon=True, name=f"Training-{job_id}")
        training_thread.start()
        
        # Store thread reference for status checking
        backend._training_thread = training_thread
        
        # Give it a moment to start
        import time
        time.sleep(0.5)
        
        # Verify training thread is alive
        if not training_thread.is_alive():
            logger.warning(f"Training thread died immediately for job {job_id}")
            return TrainingStartResponse(
                status="error",
                message="Training thread failed to start. Check server logs for details.",
                error="Thread not alive"
            )
        
        return TrainingStartResponse(
            status="started",
            job_id=job_id,
            message="Training job started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post("/stop")
async def stop_training():
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
async def get_training_status():
    """
    Get the current training status.
    """
    try:
        backend = get_training_backend()
        
        # Check if training is active
        is_active = backend.is_training_active()
        
        # Check if there's a training thread running (preparation phase)
        has_thread = hasattr(backend, '_training_thread') and backend._training_thread and backend._training_thread.is_alive()
        
        # Get progress info
        try:
            progress = backend.trainer.get_training_progress()
            status_message = progress.status_message or "Ready to train"
        except:
            progress = None
            status_message = "Unknown"
        
        if is_active:
            # Actual training is running
            trainer = backend.trainer
            current_step = getattr(trainer.training_progress, 'step', None) or (progress.step if progress else None)
            total_steps = getattr(trainer.training_progress, 'total_steps', None) or (progress.total_steps if progress else None)
            
            return TrainingStatusResponse(
                status="training",
                is_active=True,
                message=status_message or "Training is in progress",
                current_step=current_step,
                total_steps=total_steps
            )
        elif has_thread or (progress and status_message and any(keyword in status_message.lower() for keyword in ["loading", "preparing", "initializing"])):
            # Training thread is running but not yet in active training phase
            return TrainingStatusResponse(
                status="preparing",
                is_active=False,
                message=status_message or "Preparing training...",
                current_step=None,
                total_steps=None
            )
        else:
            return TrainingStatusResponse(
                status="idle",
                is_active=False,
                message="No training job is currently running"
            )
            
    except Exception as e:
        logger.error(f"Error getting training status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        )


@router.get("/metrics")
async def get_training_metrics():
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
        
        return TrainingMetricsResponse(
            loss_history=loss_history,
            lr_history=lr_history,
            step_history=step_history,
            current_loss=current_loss,
            current_lr=current_lr,
            current_step=current_step
        )
        
    except Exception as e:
        logger.error(f"Error getting training metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training metrics: {str(e)}"
        )


@router.get("/progress")
async def stream_training_progress():
    """
    Stream training progress updates using Server-Sent Events (SSE).
    
    This endpoint provides real-time updates on training progress.
    """
    async def event_generator():
        backend = get_training_backend()
        
        # Send initial status
        is_active = backend.is_training_active()
        initial_message = 'Connecting...' if is_active else 'No training in progress'
        yield f"data: {TrainingProgressResponse(step=0, loss=0.0, learning_rate=0.0, status_message=initial_message).model_dump_json()}\n\n"
        
        # If not active, check if there's any history
        if not is_active:
            if backend.step_history:
                # Training completed - send final metrics
                final_step = backend.step_history[-1]
                final_loss = backend.loss_history[-1] if backend.loss_history else 0.0
                final_lr = backend.lr_history[-1] if backend.lr_history else 0.0
                yield f"data: {TrainingProgressResponse(step=final_step, loss=final_loss, learning_rate=final_lr, status_message='Training completed').model_dump_json()}\n\n"
            else:
                yield f"data: {TrainingProgressResponse(step=-1, loss=0.0, learning_rate=0.0, status_message='No training in progress').model_dump_json()}\n\n"
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
                    
                    # Only send if step changed
                    if current_step != last_step:
                        progress = TrainingProgressResponse(
                            step=current_step,
                            loss=current_loss,
                            learning_rate=current_lr,
                            status_message=f"Training step {current_step}"
                        )
                        yield f"data: {progress.model_dump_json()}\n\n"
                        last_step = current_step
                        no_update_count = 0
                    else:
                        no_update_count += 1
                        # Send heartbeat every 10 seconds
                        if no_update_count % 10 == 0:
                            progress = TrainingProgressResponse(
                                step=current_step,
                                loss=current_loss,
                                learning_rate=current_lr,
                                status_message=f"Training step {current_step} (waiting for next update...)"
                            )
                            yield f"data: {progress.model_dump_json()}\n\n"
                else:
                    # No steps yet, but training is active
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        yield f"data: {TrainingProgressResponse(step=0, loss=0.0, learning_rate=0.0, status_message='Preparing training...').model_dump_json()}\n\n"
                
                # Timeout check
                if no_update_count > max_no_updates:
                    logger.warning("Progress stream timeout - no updates received")
                    yield f"data: {TrainingProgressResponse(step=last_step, loss=0.0, learning_rate=0.0, status_message='Progress timeout - training may have stopped').model_dump_json()}\n\n"
                    break
                
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                logger.error(f"Error in progress stream: {e}", exc_info=True)
                yield f"data: {TrainingProgressResponse(step=0, loss=0.0, learning_rate=0.0, status_message=f'Error: {str(e)}').model_dump_json()}\n\n"
                break
        
        # Send final status
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else 0.0
        final_lr = backend.lr_history[-1] if backend.lr_history else 0.0
        yield f"data: {TrainingProgressResponse(step=final_step, loss=final_loss, learning_rate=final_lr, status_message='Training completed').model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

