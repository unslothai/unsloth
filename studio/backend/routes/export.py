# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Export API routes: checkpoint discovery and model export operations.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
import structlog
from loggers import get_logger

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Auth
from auth.authentication import get_current_subject

# Import backend functions
try:
    from core.export import get_export_backend
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.export import get_export_backend

# Import Pydantic models
from models import (
    LoadCheckpointRequest,
    ExportStatusResponse,
    ExportOperationResponse,
    ExportMergedModelRequest,
    ExportBaseModelRequest,
    ExportGGUFRequest,
    ExportLoRAAdapterRequest,
    ExportVllm4bitRequest,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/load-checkpoint", response_model = ExportOperationResponse)
async def load_checkpoint(
    request: LoadCheckpointRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Load a checkpoint into the export backend.

    Wraps ExportBackend.load_checkpoint.
    """
    try:
        # Version switching is handled automatically by the subprocess-based
        # export backend — no need for ensure_transformers_version() here.

        # Free GPU memory: shut down any running inference/training subprocesses
        # before loading the export checkpoint (they'd compete for VRAM).
        try:
            from core.inference import get_inference_backend

            inf = get_inference_backend()
            if inf.active_model_name:
                logger.info(
                    "Unloading inference model '%s' to free GPU memory for export",
                    inf.active_model_name,
                )
                inf._shutdown_subprocess()
                inf.active_model_name = None
                inf.models.clear()
        except Exception as e:
            logger.warning("Could not unload inference model: %s", e)

        try:
            from core.training import get_training_backend

            trn = get_training_backend()
            if trn.is_training_active():
                logger.info("Stopping active training to free GPU memory for export")
                trn.stop_training()
                # Wait for training subprocess to actually exit before proceeding,
                # otherwise it may still hold GPU memory when export tries to load.
                for _ in range(60):  # up to 30s
                    if not trn.is_training_active():
                        break
                    import time

                    time.sleep(0.5)
                else:
                    logger.warning(
                        "Training subprocess did not exit within 30s, proceeding anyway"
                    )
        except Exception as e:
            logger.warning("Could not stop training: %s", e)

        backend = get_export_backend()
        success, message = backend.load_checkpoint(
            checkpoint_path = request.checkpoint_path,
            max_seq_length = request.max_seq_length,
            load_in_4bit = request.load_in_4bit,
            trust_remote_code = request.trust_remote_code,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to load checkpoint: {str(e)}",
        )


@router.post("/cleanup", response_model = ExportOperationResponse)
async def cleanup_export_memory(
    current_subject: str = Depends(get_current_subject),
):
    """
    Cleanup export-related models from memory (GPU/CPU).

    Wraps ExportBackend.cleanup_memory.
    """
    try:
        backend = get_export_backend()
        success = backend.cleanup_memory()

        if not success:
            raise HTTPException(
                status_code = 500,
                detail = "Memory cleanup failed. See server logs for details.",
            )

        return ExportOperationResponse(
            success = True,
            message = "Memory cleanup completed successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during export memory cleanup: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to cleanup export memory: {str(e)}",
        )


@router.get("/status", response_model = ExportStatusResponse)
async def get_export_status(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get current export backend status (loaded checkpoint, model type, PEFT flag).
    """
    try:
        backend = get_export_backend()
        return ExportStatusResponse(
            current_checkpoint = backend.current_checkpoint,
            is_vision = bool(getattr(backend, "is_vision", False)),
            is_peft = bool(getattr(backend, "is_peft", False)),
        )
    except Exception as e:
        logger.error(f"Error getting export status: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to get export status: {str(e)}",
        )


@router.post("/export/merged", response_model = ExportOperationResponse)
async def export_merged_model(
    request: ExportMergedModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Export a merged PEFT model (e.g., 16-bit or 4-bit) and optionally push to Hub.

    Wraps ExportBackend.export_merged_model.
    """
    try:
        backend = get_export_backend()
        success, message = backend.export_merged_model(
            save_directory = request.save_directory,
            format_type = request.format_type,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting merged model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export merged model: {str(e)}",
        )


@router.post("/export/base", response_model = ExportOperationResponse)
async def export_base_model(
    request: ExportBaseModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Export a non-PEFT base model and optionally push to Hub.

    Wraps ExportBackend.export_base_model.
    """
    try:
        backend = get_export_backend()
        success, message = backend.export_base_model(
            save_directory = request.save_directory,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
            base_model_id = request.base_model_id,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting base model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export base model: {str(e)}",
        )


@router.post("/export/gguf", response_model = ExportOperationResponse)
async def export_gguf(
    request: ExportGGUFRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Export the current model to GGUF format and optionally push to Hub.

    Wraps ExportBackend.export_gguf.
    """
    try:
        backend = get_export_backend()
        success, message = backend.export_gguf(
            save_directory = request.save_directory,
            quantization_method = request.quantization_method,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting GGUF model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export GGUF model: {str(e)}",
        )


@router.post("/export/lora", response_model = ExportOperationResponse)
async def export_lora_adapter(
    request: ExportLoRAAdapterRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Export only the LoRA adapter (if the loaded model is PEFT).

    Wraps ExportBackend.export_lora_adapter.
    """
    try:
        backend = get_export_backend()
        success, message = backend.export_lora_adapter(
            save_directory = request.save_directory,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting LoRA adapter: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export LoRA adapter: {str(e)}",
        )


@router.post("/export/vllm4bit", response_model = ExportOperationResponse)
async def export_vllm_4bit(
    request: ExportVllm4bitRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Export the model directly to vLLM 4-bit format using Auto-Round (AWQ/GPTQ).

    Wraps ExportBackend.export_vllm_4bit.
    """
    try:
        backend = get_export_backend()
        success, message = backend.export_vllm_4bit(
            save_directory = request.save_directory,
            export_format = request.export_format,
            bits = request.bits,
            group_size = request.group_size,
            iters = request.iters,
            nsamples = request.nsamples,
            dataset = request.dataset,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting vLLM 4-bit model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export vLLM 4-bit model: {str(e)}",
        )
