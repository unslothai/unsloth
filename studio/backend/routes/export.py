# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export API routes: checkpoint discovery and model export operations."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
import structlog
from loggers import get_logger

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject

from utils.utils import safe_error_detail

try:
    from core.export import get_export_backend
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.export import get_export_backend

from models import (
    LoadCheckpointRequest,
    ExportStatusResponse,
    ExportOperationResponse,
    ExportMergedModelRequest,
    ExportBaseModelRequest,
    ExportGGUFRequest,
    ExportLoRAAdapterRequest,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/load-checkpoint", response_model = ExportOperationResponse)
async def load_checkpoint(
    request: LoadCheckpointRequest, current_subject: str = Depends(get_current_subject)
):
    """Load a checkpoint into the export backend (ExportBackend.load_checkpoint)."""
    try:
        # Free GPU memory: shut down running inference/training subprocesses
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
                # Wait for the training subprocess to exit, else it may still hold GPU memory.
                for _ in range(60):  # up to 30s
                    if not trn.is_training_active():
                        break
                    await asyncio.sleep(0.5)
                else:
                    logger.warning("Training subprocess did not exit within 30s, proceeding anyway")
        except Exception as e:
            logger.warning("Could not stop training: %s", e)

        backend = get_export_backend()
        # Run in a worker thread (spawns and waits on a subprocess, can take
        # minutes) so the event loop stays free to serve the live log SSE stream.
        success, message = await asyncio.to_thread(
            backend.load_checkpoint,
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
            detail = "Failed to load checkpoint",
        )


@router.post("/cleanup", response_model = ExportOperationResponse)
async def cleanup_export_memory(current_subject: str = Depends(get_current_subject)):
    """Cleanup export-related models from memory (ExportBackend.cleanup_memory)."""
    try:
        backend = get_export_backend()
        success = await asyncio.to_thread(backend.cleanup_memory)

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
            detail = "Failed to cleanup export memory",
        )


@router.get("/status", response_model = ExportStatusResponse)
async def get_export_status(current_subject: str = Depends(get_current_subject)):
    """Get export backend status (loaded checkpoint, model type, PEFT flag)."""
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
            detail = "Failed to get export status",
        )


def _export_details(output_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the export path relative to exports_root, hiding the install path."""
    if not output_path:
        return None
    try:
        from utils.paths.storage_roots import exports_root

        rel = os.path.relpath(output_path, exports_root())
        if rel.startswith(".."):
            rel = os.path.basename(output_path)
        return {"output_path": rel}
    except Exception:
        return {"output_path": os.path.basename(output_path)}


@router.post("/export/merged", response_model = ExportOperationResponse)
async def export_merged_model(
    request: ExportMergedModelRequest, current_subject: str = Depends(get_current_subject)
):
    """Export a merged PEFT model (16-bit or 4-bit), optionally pushing to Hub.

    Wraps ExportBackend.export_merged_model.
    """
    try:
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_merged_model,
            save_directory = request.save_directory,
            format_type = request.format_type,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(
            success = True,
            message = message,
            details = _export_details(output_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting merged model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to export merged model",
        )


@router.post("/export/base", response_model = ExportOperationResponse)
async def export_base_model(
    request: ExportBaseModelRequest, current_subject: str = Depends(get_current_subject)
):
    """Export a non-PEFT base model, optionally pushing to Hub.

    Wraps ExportBackend.export_base_model.
    """
    try:
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_base_model,
            save_directory = request.save_directory,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
            base_model_id = request.base_model_id,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(
            success = True,
            message = message,
            details = _export_details(output_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting base model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to export base model",
        )


@router.post("/export/gguf", response_model = ExportOperationResponse)
async def export_gguf(
    request: ExportGGUFRequest, current_subject: str = Depends(get_current_subject)
):
    """Export the current model to GGUF format, optionally pushing to Hub.

    Wraps ExportBackend.export_gguf.
    """
    try:
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_gguf,
            save_directory = request.save_directory,
            quantization_method = request.quantization_method,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
            gguf_shard_size = request.gguf_shard_size,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(
            success = True,
            message = message,
            details = _export_details(output_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting GGUF model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to export GGUF model",
        )


@router.post("/export/lora", response_model = ExportOperationResponse)
async def export_lora_adapter(
    request: ExportLoRAAdapterRequest, current_subject: str = Depends(get_current_subject)
):
    """Export only the LoRA adapter (if the loaded model is PEFT).

    Wraps ExportBackend.export_lora_adapter.
    """
    try:
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_lora_adapter,
            save_directory = request.save_directory,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(
            success = True,
            message = message,
            details = _export_details(output_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting LoRA adapter: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to export LoRA adapter",
        )


# Live export log stream (Server-Sent Events).
#
# The export worker's stdout/stderr is piped to the orchestrator as log
# entries (core/export/worker.py, orchestrator.py); this endpoint streams
# them to the browser for a live terminal panel during export operations.
#
# Shape follows routes/training.py::stream_training_progress: each event
# carries id/event/data, the stream starts with a `retry:` directive, and
# `Last-Event-ID` is honored on reconnect.


def _format_sse(
    data: str,
    event: str,
    event_id: Optional[int] = None,
) -> str:
    """Format a single SSE message with id/event/data fields."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


@router.get("/logs/stream")
async def stream_export_logs(
    request: Request,
    since: Optional[int] = Query(
        None,
        description = "Return log entries with seq strictly greater than this cursor.",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    Stream live stdout/stderr from the export worker subprocess as
    Server-Sent Events.

    Events:
      - `log`      : a single log line (data: {"stream","line","ts"})
      - `heartbeat`: periodic keepalive when no new lines are available
      - `complete` : once the worker is idle and no new lines arrived for
                     ~1 second. Clients should close.
      - `error`    : unrecoverable server-side error

    Each event's `id:` field is the log entry's monotonic seq number so the
    browser can resume via `Last-Event-ID` on reconnect.
    """
    backend = get_export_backend()

    # Starting cursor: explicit `since` wins, then Last-Event-ID on reconnect,
    # else the run-start snapshot so the client sees every line since the run
    # began even if the SSE connection opened after the export-kickoff POST.
    last_event_id = request.headers.get("last-event-id")
    if since is None and last_event_id is not None:
        try:
            since = int(last_event_id)
        except ValueError:
            pass

    if since is None:
        cursor = backend.get_run_start_seq()
    else:
        cursor = max(0, int(since))

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal cursor
        # Reconnect after 3 seconds if the connection drops mid-export.
        yield "retry: 3000\n\n"

        last_yield = time.monotonic()
        idle_since: Optional[float] = None
        try:
            while True:
                if await request.is_disconnected():
                    return

                entries, new_cursor = backend.get_logs_since(cursor)
                if entries:
                    for entry in entries:
                        payload = json.dumps(
                            {
                                "stream": entry.get("stream", "stdout"),
                                "line": entry.get("line", ""),
                                "ts": entry.get("ts"),
                            }
                        )
                        yield _format_sse(
                            payload,
                            event = "log",
                            event_id = int(entry.get("seq", 0)),
                        )
                    cursor = new_cursor
                    last_yield = time.monotonic()
                    idle_since = None
                else:
                    now = time.monotonic()
                    if now - last_yield > 10.0:
                        yield _format_sse("{}", event = "heartbeat")
                        last_yield = now
                    if not backend.is_export_active():
                        # Let the reader thread drain trailing lines printed just
                        # before the worker signalled done.
                        if idle_since is None:
                            idle_since = now
                        elif now - idle_since > 1.0:
                            yield _format_sse(
                                "{}",
                                event = "complete",
                                event_id = cursor,
                            )
                            return
                    else:
                        idle_since = None

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Client disconnected mid-yield: end cleanly so StreamingResponse finalizes.
            return
        except Exception as exc:
            logger.error("Export log stream failed: %s", exc, exc_info = True)
            try:
                yield _format_sse(
                    json.dumps({"error": safe_error_detail(exc)}),
                    event = "error",
                )
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
