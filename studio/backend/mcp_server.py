# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated MCP tools for driving an Unsloth Studio instance. The MCP surface deliberately wraps the existing
Unsloth services instead of duplicating training or export logic. It is opt-in because several tools can start
GPU work or write model artifacts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from studio_mcp.tools.audio import register_audio
from studio_mcp.tools.data import register_data
from studio_mcp.tools.images import register_images
from studio_mcp.tools.jobs import register_jobs
from studio_mcp.tools.models import register_models
from studio_mcp.tools.status import register_status
from studio_mcp.tools.text import register_text
from studio_mcp.tools.video import register_video


def _dump(value: Any) -> Any:
    """Convert Pydantic responses to plain JSON values for MCP clients."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode = "json")
    return value


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an MCP-supplied integer into an inclusive range. MCP tools call the Unsloth route functions
    directly, which skips FastAPI's Query(ge=, le=) validation, so we re-apply the same bounds here.
    """
    return max(low, min(value, high))


def create_studio_mcp() -> FastMCP:
    """Create the Unsloth MCP server and register the high-value tools."""
    mcp = FastMCP(
        "Unsloth Studio",
        instructions = (
            "Use read tools to inspect the local Unsloth state before starting GPU work. "
            "Training and export tools can consume substantial VRAM and write files. "
            "Never expose tokens or local paths from tool results unless the user asks."
        ),
    )

    register_status(mcp)

    register_models(mcp)
    register_text(mcp)
    register_images(mcp)
    register_audio(mcp)
    register_video(mcp)
    register_jobs(mcp)
    register_data(mcp)

    @mcp.tool
    async def start_training(config: dict[str, Any]) -> dict[str, Any]:
        """Start a validated Unsloth training job from a TrainingStartRequest-shaped object.

        The config is validated by the same Pydantic model used by the Unsloth UI.
        Call get_training_status first and do not start work while another job runs.
        """
        from models import TrainingStartRequest
        from routes.training import start_training as start

        request = TrainingStartRequest.model_validate(config)
        return _dump(await start(request, current_subject = "mcp", via_api_key = True))

    @mcp.tool
    async def stop_training(expected_job_id: str, save: bool = True) -> dict[str, Any]:
        """Stop the identified training job at its next safe checkpoint."""
        from routes.training import TrainingStopRequest, stop_training as stop
        return _dump(
            await stop(
                TrainingStopRequest(save = save, expected_job_id = expected_job_id),
                current_subject = "mcp",
            )
        )

    @mcp.tool
    async def list_training_runs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List completed and stopped training runs, newest first."""
        from routes.training_history import list_training_runs as list_runs

        # Clamp here (direct call skips Query bounds); a negative LIMIT = no limit.
        limit = _clamp(limit, 1, 200)
        offset = max(0, offset)
        return _dump(await list_runs(limit = limit, offset = offset, current_subject = "mcp"))

    @mcp.tool
    async def load_checkpoint(
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool | None = None,
        trust_remote_code: bool = False,
        approved_remote_code_fingerprint: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint into the export backend.

        Export runs in its own subprocess and coexists with training and
        inference; it does not unload them, so a load can fail with a clear
        out-of-memory error if the GPU is already full. Pass hf_token to load a
        gated checkpoint, and approved_remote_code_fingerprint to retry a
        trust_remote_code load that was blocked pending review. Leave
        load_in_4bit unset to load full fine-tunes in 16-bit and adapters in
        4-bit.
        """
        from models import LoadCheckpointRequest
        from routes.export import load_checkpoint as load

        # Omit an unset load_in_4bit so the backend can pick 16-bit for a full fine-tune.
        optional = {} if load_in_4bit is None else {"load_in_4bit": load_in_4bit}
        request = LoadCheckpointRequest(
            checkpoint_path = checkpoint_path,
            max_seq_length = max_seq_length,
            trust_remote_code = trust_remote_code,
            approved_remote_code_fingerprint = approved_remote_code_fingerprint,
            hf_token = hf_token,
            **optional,
        )
        return _dump(await load(request, current_subject = "mcp", allow_ambient = False))

    @mcp.tool
    async def export_gguf(
        save_directory: str,
        quantization_method: str | list[str] = "Q4_K_M",
        push_to_hub: bool = False,
        repo_id: str | None = None,
        hf_token: str | None = None,
        imatrix: bool = False,
        imatrix_path: str | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        """Export the loaded model to GGUF using Unsloth's existing path validation.

        quantization_method may be a single method or a list to produce several
        GGUFs from one load. Pass hf_token when push_to_hub is set (the backend
        rejects a Hub upload without it). Set imatrix (or imatrix_path) for the
        IQ low-bit quants that require an importance matrix.
        """
        from models import ExportGGUFRequest
        from routes.export import export_gguf as export

        request = ExportGGUFRequest(
            save_directory = save_directory,
            quantization_method = quantization_method,
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            imatrix = imatrix,
            imatrix_path = imatrix_path,
            private = private,
        )
        return _dump(await export(request, current_subject = "mcp", allow_ambient = False))

    return mcp
