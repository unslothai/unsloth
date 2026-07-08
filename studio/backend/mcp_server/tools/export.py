# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio model export.

Export operations (``export_load_checkpoint``, ``export_merged`` /
``export_gguf`` / ``export_lora``) are **synchronous** and block until the
operation finishes -- they may take minutes (merged) to hours (multi-quant
GGUF). They run on the Studio export subprocess, exactly like the Studio UI.
Use ``export_status`` for the loaded-checkpoint state and ``export_cancel`` to
abort an in-flight export.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from fastmcp import FastMCP

from loggers import get_logger
from mcp_server.auth import resolve_hf_token

logger = get_logger(__name__)

GROUP = "export"

_READ_ONLY = {"readOnlyHint": True}
_STATEFUL = {"destructiveHint": False}

# Mirrors the ExportMergedModelRequest Literal in models/export.py.
_VALID_MERGED_FORMATS = (
    "16-bit (FP16)",
    "4-bit (FP4)",
    "FP8 (compressed-tensors)",
    "NVFP4 (compressed-tensors)",
)

# Mirrors LoadCheckpointRequest.max_seq_length bounds in models/export.py.
_MIN_SEQ_LEN, _MAX_SEQ_LEN = 128, 32768


def _save_dir_error(save_directory: str) -> Optional[str]:
    """Return an error string if ``save_directory`` is invalid, else None.

    Reuses models/export.py ``_validate_save_directory`` so the MCP path
    enforces the same rules as the HTTP schema (non-empty, no '..', etc.) and
    can't report success while the backend writes nothing.
    """
    try:
        from models.export import _validate_save_directory
        _validate_save_directory(save_directory)
        return None
    except (ValueError, TypeError) as exc:
        return str(exc)


def _export_supported_error() -> Optional[str]:
    """Return an error string if this host can't export, else None.

    Mirrors routes/export.py ``_ensure_export_supported`` so unsupported hosts
    (CPU-only / no-torch installs) get a clear rejection instead of a
    low-level load failure.
    """
    try:
        from utils.hardware import export_capability
        cap = export_capability()
        if not cap.get("export_supported", True):
            return (
                cap.get("export_unsupported_message") or "Export is not supported on this platform."
            )
    except Exception as exc:  # noqa: BLE001 -- never block on the capability probe
        logger.warning("export_capability probe failed: %s", exc)
    return None


def _register_external_export(output_path: Optional[str]) -> Optional[str]:
    """Best-effort register an export dir outside exports_root for local scans.

    Mirrors routes/export.py ``_try_register_external_export`` so an MCP export
    to an absolute ``save_directory`` still shows up in Studio's model
    inventory. Returns the registered path, or None if not applicable/failed.
    """
    if not output_path:
        return None
    try:
        from pathlib import Path

        from utils.paths.storage_roots import exports_root

        path = Path(output_path)
        if not path.is_absolute():
            return None
        try:
            path.resolve().relative_to(exports_root().resolve())
            return None  # inside exports_root: already discoverable
        except ValueError:
            pass
        from storage.studio_db import add_scan_folder

        folder = add_scan_folder(str(path))
        return str(folder.get("path") or path)
    except Exception as exc:  # noqa: BLE001 -- registration is best-effort
        logger.warning("Could not register export scan folder %s: %s", output_path, exc)
        return None


def export_list_checkpoints(outputs_dir: Optional[str] = None) -> dict[str, Any]:
    """List training checkpoints available for export.

    Scans Studio's outputs root (or ``outputs_dir``) for runs/checkpoints.
    Pass one of the returned checkpoint paths to ``export_load_checkpoint``.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    scanned = backend.scan_checkpoints(outputs_dir = outputs_dir)
    runs: list[dict[str, Any]] = []
    for model_name, checkpoints, metadata in scanned:
        runs.append(
            {
                "model_name": model_name,
                "checkpoints": [
                    {"name": name, "path": path, "loss": loss} for name, path, loss in checkpoints
                ],
                "metadata": metadata,
            }
        )
    return {"outputs_dir": outputs_dir, "runs": runs}


def export_load_checkpoint(
    checkpoint_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    trust_remote_code: bool = False,
    approved_remote_code_fingerprint: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Load a checkpoint into the export backend (must precede ``export_*``).

    Returns once the checkpoint is resident in the export subprocess. After a
    successful load, call one of ``export_merged`` / ``export_gguf`` /
    ``export_lora``. Pass ``trust_remote_code`` (and the
    ``approved_remote_code_fingerprint`` from Studio's remote-code scan) for
    checkpoints that require custom model code.
    """
    unsupported = _export_supported_error()
    if unsupported:
        return {"success": False, "message": unsupported, "checkpoint": checkpoint_path}
    if not (_MIN_SEQ_LEN <= max_seq_length <= _MAX_SEQ_LEN):
        return {
            "success": False,
            "message": f"max_seq_length must be in [{_MIN_SEQ_LEN}, {_MAX_SEQ_LEN}] (got {max_seq_length}).",
            "checkpoint": checkpoint_path,
        }

    from core.export import get_export_backend

    backend = get_export_backend()
    success, message = backend.load_checkpoint(
        checkpoint_path = checkpoint_path,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        trust_remote_code = trust_remote_code,
        approved_remote_code_fingerprint = approved_remote_code_fingerprint,
        hf_token = resolve_hf_token(hf_token),
        subject = "mcp",
    )
    result: dict[str, Any] = {"success": success, "message": message, "checkpoint": checkpoint_path}
    # Surface a remote-code consent block (with its approval fingerprint) so a
    # pure-MCP caller can discover the fingerprint and retry with approval.
    if not success:
        error_kind = getattr(backend, "last_error_kind", None)
        remote_code = getattr(backend, "last_remote_code", None)
        if error_kind:
            result["error_kind"] = error_kind
        if remote_code:
            result["remote_code"] = remote_code
    return result


def export_merged(
    save_directory: str,
    format_type: str = "16-bit (FP16)",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    compressed_method: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Export the loaded checkpoint as a merged model.

    ``format_type`` is one of ``16-bit (FP16)``, ``4-bit (FP4)``,
    ``FP8 (compressed-tensors)``, ``NVFP4 (compressed-tensors)``. Blocks until
    the merge writes to ``save_directory`` (and optionally pushes to the Hub).
    """
    unsupported = _export_supported_error()
    if unsupported:
        return _export_result("merged", False, unsupported, None)
    sd_error = _save_dir_error(save_directory)
    if sd_error:
        return _export_result("merged", False, sd_error, None)
    if format_type not in _VALID_MERGED_FORMATS:
        return _export_result(
            "merged",
            False,
            f"Unknown format_type {format_type!r}. Valid: {list(_VALID_MERGED_FORMATS)}",
            None,
        )

    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_merged_model(
        save_directory = save_directory,
        format_type = format_type,
        push_to_hub = push_to_hub,
        repo_id = repo_id,
        hf_token = resolve_hf_token(hf_token),
        private = private,
        compressed_method = compressed_method,
    )
    return _export_result("merged", success, message, output_path)


def export_gguf(
    save_directory: str,
    quantization_method: Union[str, List[str]] = "Q4_K_M",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    imatrix_file: Optional[str] = None,
    imatrix: bool = False,
) -> dict[str, Any]:
    """Export the loaded checkpoint in GGUF format for llama.cpp / Ollama.

    ``quantization_method`` is a llama.cpp quant (e.g. ``Q4_K_M``, ``Q8_0``,
    ``F16``) or a list of quants to produce in one pass off a single merge.
    This is the slowest export (minutes to hours for large models). For
    importance-matrix quants (``iq2_xxs``, ``iq4_xs``, ...) set ``imatrix=True``
    so Unsloth auto-downloads the upstream imatrix, or pass a local file via
    ``imatrix_file``.
    """
    unsupported = _export_supported_error()
    if unsupported:
        return _export_result("gguf", False, unsupported, None)
    sd_error = _save_dir_error(save_directory)
    if sd_error:
        return _export_result("gguf", False, sd_error, None)
    # A custom path wins; otherwise the imatrix toggle requests the upstream auto-download
    # (mirrors routes/export.py GGUF handler).
    resolved_imatrix = imatrix_file or (True if imatrix else None)

    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_gguf(
        save_directory = save_directory,
        quantization_method = quantization_method,
        push_to_hub = push_to_hub,
        repo_id = repo_id,
        hf_token = resolve_hf_token(hf_token),
        imatrix_file = resolved_imatrix,
    )
    return _export_result("gguf", success, message, output_path)


def export_lora(
    save_directory: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    gguf: bool = False,
    gguf_outtype: str = "q8_0",
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Export the loaded checkpoint as a LoRA adapter only.

    With ``gguf=True``, also writes a GGUF LoRA file (``gguf_outtype`` one of
    ``q8_0``, ``f16``, ``bf16``, ``f32``). Much smaller/faster than a full merge.
    """
    unsupported = _export_supported_error()
    if unsupported:
        return _export_result("lora", False, unsupported, None)
    sd_error = _save_dir_error(save_directory)
    if sd_error:
        return _export_result("lora", False, sd_error, None)

    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_lora_adapter(
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        repo_id = repo_id,
        hf_token = resolve_hf_token(hf_token),
        private = private,
        gguf = gguf,
        gguf_outtype = gguf_outtype,
    )
    return _export_result("lora", success, message, output_path)


def export_status() -> dict[str, Any]:
    """Get export-backend state: loaded checkpoint, model type, last operation.

    Use this after ``export_load_checkpoint`` to confirm what is resident
    before choosing an ``export_*`` format, and to inspect the last op result.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    last_op = backend.get_last_op() or {}
    return {
        "current_checkpoint": backend.current_checkpoint,
        "is_vision": bool(getattr(backend, "is_vision", False)),
        "is_peft": bool(getattr(backend, "is_peft", False)),
        "is_export_active": bool(backend.is_export_active()),
        "active_op_kind": backend.get_active_op_kind(),
        "last_op": {
            "kind": last_op.get("kind"),
            "status": last_op.get("status"),
            "output_path": last_op.get("output_path"),
            "error": last_op.get("error"),
        },
    }


def export_cancel() -> dict[str, Any]:
    """Cancel any in-flight export (terminates the export subprocess)."""
    from core.export import get_export_backend

    backend = get_export_backend()
    cancelled = backend.cancel_export()
    return {
        "success": True,
        "message": "Export cancelled" if cancelled else "No active export to cancel",
    }


def export_cleanup() -> dict[str, Any]:
    """Release the loaded checkpoint / free export GPU memory."""
    from core.export import get_export_backend

    backend = get_export_backend()
    success = backend.cleanup_memory()
    return {
        "success": success,
        "message": "Memory cleanup completed" if success else "Cleanup failed",
    }


def _export_result(
    kind: str, success: bool, message: str, output_path: Optional[str]
) -> dict[str, Any]:
    # Best-effort: make an export to an absolute dir outside exports_root
    # discoverable by Studio's local model scans (matches the HTTP route).
    scan_folder = _register_external_export(output_path) if success and output_path else None
    result = {
        "success": success,
        "kind": kind,
        "message": message,
        "output_path": output_path,
    }
    if scan_folder:
        result["scan_folder_registered"] = scan_folder
    return result


def register(mcp: FastMCP) -> list[str]:
    """Register the export tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    mcp.tool(export_list_checkpoints, annotations = _READ_ONLY)
    names.append("export_list_checkpoints")
    mcp.tool(export_status, annotations = _READ_ONLY)
    names.append("export_status")
    for fn in (
        export_load_checkpoint,
        export_merged,
        export_gguf,
        export_lora,
        export_cancel,
        export_cleanup,
    ):
        mcp.tool(fn, annotations = _STATEFUL)
        names.append(fn.__name__)
    return names
