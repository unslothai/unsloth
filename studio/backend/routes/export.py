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


def _ensure_export_supported() -> None:
    """Reject a mutating export request up front (HTTP 400) when the host can't export.

    Keeps the backend authoritative even if a client bypasses the UI gate. Read-only endpoints
    (scan/status/logs) are intentionally NOT gated so the Export page can still render the reason.
    Also refuses (409) while a latest-transformers install is swapping .venv_t5_latest: an
    export worker spawned mid-swap could activate a half-replaced sidecar.
    """
    from utils.transformers_latest import is_install_in_progress

    if is_install_in_progress():
        raise HTTPException(
            status_code = 409,
            detail = "A transformers installation is in progress. Retry when it completes.",
        )

    from utils.hardware import export_capability

    cap = export_capability()
    if not cap.get("export_supported", True):
        raise HTTPException(
            status_code = 400,
            detail = cap.get("export_unsupported_message")
            or "Export is not supported on this platform.",
        )


@router.post("/load-checkpoint", response_model = ExportOperationResponse)
async def load_checkpoint(
    request: LoadCheckpointRequest, current_subject: str = Depends(get_current_subject)
):
    """Load a checkpoint into the export backend (ExportBackend.load_checkpoint).

    Export runs in its own subprocess and is allowed to run in parallel with
    training and inference. We deliberately do NOT stop training or unload the
    chat model here -- if the GPU runs out of memory the load/export fails with
    a clear error instead of tearing down the user's other running workloads.
    """
    try:
        _ensure_export_supported()
        backend = get_export_backend()
        # Run in a worker thread (spawns and waits on a subprocess, can take
        # minutes) so the event loop stays free to serve the live log SSE stream.
        success, message = await asyncio.to_thread(
            backend.load_checkpoint,
            checkpoint_path = request.checkpoint_path,
            max_seq_length = request.max_seq_length,
            load_in_4bit = request.load_in_4bit,
            trust_remote_code = request.trust_remote_code,
            approved_remote_code_fingerprint = request.approved_remote_code_fingerprint,
            hf_token = request.hf_token,
            subject = current_subject,
        )

        if not success:
            raise HTTPException(status_code = 400, detail = message)

        return ExportOperationResponse(success = True, message = message)
    except HTTPException:
        raise
    except Exception as e:
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Expected loss of the race against a sidecar install: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
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


@router.post("/cancel", response_model = ExportOperationResponse)
async def cancel_export(current_subject: str = Depends(get_current_subject)):
    """Cancel the in-flight export by terminating its worker subprocess.

    Only the export subprocess is killed; training and inference run in their
    own subprocesses and keep going.
    """
    try:
        backend = get_export_backend()
        cancelled = await asyncio.to_thread(backend.cancel_export)
        return ExportOperationResponse(
            success = True,
            message = "Export cancelled" if cancelled else "No active export to cancel",
        )
    except Exception as e:
        logger.error(f"Error cancelling export: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to cancel export",
        )


@router.get("/status", response_model = ExportStatusResponse)
async def get_export_status(current_subject: str = Depends(get_current_subject)):
    """Get export backend status (loaded checkpoint, model type, PEFT flag)."""
    try:
        backend = get_export_backend()
        last_op = backend.get_last_op()
        # Relativise the recovered output path the same way the per-op POST response
        # does, so the success banner shows an identical path on either route.
        last_op_output_path = None
        if last_op and last_op.get("output_path"):
            details = _export_details(last_op["output_path"])
            last_op_output_path = (details or {}).get("output_path")
        return ExportStatusResponse(
            current_checkpoint = backend.current_checkpoint,
            is_vision = bool(getattr(backend, "is_vision", False)),
            is_peft = bool(getattr(backend, "is_peft", False)),
            is_export_active = bool(backend.is_export_active()),
            active_op_kind = backend.get_active_op_kind(),
            last_op_seq = int(last_op["seq"]) if last_op else 0,
            last_op_kind = last_op.get("kind") if last_op else None,
            last_op_status = last_op.get("status") if last_op else None,
            last_op_output_path = last_op_output_path,
            last_op_error = last_op.get("error") if last_op else None,
        )
    except Exception as e:
        logger.error(f"Error getting export status: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to get export status",
        )


@router.get("/logs")
async def get_export_logs(
    since: Optional[int] = Query(
        None,
        description = "Return log entries with seq strictly greater than this cursor.",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """Tunnel-safe JSON fallback for the live export log stream.

    The SSE endpoint (`/logs/stream`) is the low-latency path, but some reverse
    proxies -- notably Cloudflare quick tunnels (`*.trycloudflare.com`) used by
    `--secure` mode -- buffer `text/event-stream` responses and only flush when
    the stream closes, so over the tunnel the browser sees nothing for the whole
    export ("connecting..." with no logs). This endpoint returns the same
    ring-buffer lines as a short, complete JSON response that no proxy buffers,
    so the frontend can poll it and still show logs in near real time.

    Shares the orchestrator's monotonic `seq` cursor with the SSE stream, so the
    two transports can run together and the client de-dupes by seq.
    """
    try:
        backend = get_export_backend()
        # No cursor on the first poll of a run: start from the run-start snapshot
        # so the client gets every line since the run began (matches the SSE
        # default), not the entire historical ring buffer.
        if since is None:
            cursor = backend.get_run_start_seq()
        else:
            cursor = max(0, int(since))

        entries, new_cursor = backend.get_logs_since(cursor)
        return {
            "entries": [
                {
                    "seq": int(entry.get("seq", 0)),
                    "stream": entry.get("stream", "stdout"),
                    "line": entry.get("line", ""),
                    "ts": entry.get("ts"),
                }
                for entry in entries
            ],
            "cursor": new_cursor,
            "active": bool(backend.is_export_active()),
        }
    except Exception as e:
        logger.error(f"Error getting export logs: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to get export logs",
        )


def _try_register_external_export(path: Path) -> tuple[bool, Optional[str]]:
    """Best-effort registration so absolute exports show up in local scans."""
    try:
        from storage.studio_db import add_scan_folder
        folder = add_scan_folder(str(path))
        return True, str(folder.get("path") or path)
    except Exception as exc:
        logger.warning("Could not register export scan folder %s: %s", path, exc)
        return False, None


def _export_details(output_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return relative export paths, keeping external absolute paths visible."""
    if not output_path:
        return None
    try:
        from utils.paths.storage_roots import exports_root

        path = Path(output_path)
        # If it's outside exports_root, return the full absolute path
        # so users can find their files on a different drive.
        if path.is_absolute():
            try:
                path.resolve().relative_to(exports_root().resolve())
            except ValueError:
                registered, registered_path = _try_register_external_export(path)
                return {
                    "output_path": str(path),
                    "scan_folder_registered": registered,
                    "scan_folder_path": registered_path,
                }
        rel = os.path.relpath(output_path, exports_root())
        return {"output_path": rel}
    except Exception:
        return {"output_path": output_path}


@router.post("/export/merged", response_model = ExportOperationResponse)
async def export_merged_model(
    request: ExportMergedModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Export a merged PEFT model (16-bit or 4-bit), optionally pushing to Hub.

    Wraps ExportBackend.export_merged_model.
    """
    try:
        _ensure_export_supported()
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_merged_model,
            save_directory = request.save_directory,
            format_type = request.format_type,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
            compressed_method = request.compressed_method,
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
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Expected loss of the race against a sidecar install: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
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
        _ensure_export_supported()
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
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Expected loss of the race against a sidecar install: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
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
        _ensure_export_supported()
        backend = get_export_backend()
        # A custom path wins; otherwise the imatrix toggle requests the upstream auto-download.
        imatrix_file = request.imatrix_path or (True if request.imatrix else None)
        success, message, output_path = await asyncio.to_thread(
            backend.export_gguf,
            save_directory = request.save_directory,
            quantization_method = request.quantization_method,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            imatrix_file = imatrix_file,
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
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Expected loss of the race against a sidecar install: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
        logger.error(f"Error exporting GGUF model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to export GGUF model",
        )


@router.post("/export/lora", response_model = ExportOperationResponse)
async def export_lora_adapter(
    request: ExportLoRAAdapterRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Export only the LoRA adapter (if the loaded model is PEFT).

    Wraps ExportBackend.export_lora_adapter.
    """
    try:
        _ensure_export_supported()
        backend = get_export_backend()
        success, message, output_path = await asyncio.to_thread(
            backend.export_lora_adapter,
            save_directory = request.save_directory,
            push_to_hub = request.push_to_hub,
            repo_id = request.repo_id,
            hf_token = request.hf_token,
            private = request.private,
            gguf = request.gguf,
            gguf_outtype = request.gguf_outtype,
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
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Expected loss of the race against a sidecar install: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
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
