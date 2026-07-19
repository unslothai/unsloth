# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated MCP tools for driving an Unsloth Studio instance.

The MCP surface deliberately wraps the existing Studio services instead of
duplicating training or export logic. It is opt-in because several tools can
start GPU work or write model artifacts.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastmcp import FastMCP


class BearerTokenMiddleware:
    """Require an exact bearer token when Studio MCP is exposed remotely."""

    def __init__(self, app: Any, token: str) -> None:
        if not token or not token.strip():
            raise ValueError("Studio MCP bearer token must be a non-empty value")
        if not token.isascii():
            # A non-ASCII token cannot be sent in an HTTP header; reject it here.
            raise ValueError("Studio MCP bearer token must contain ASCII characters only")
        self.app = app
        # Compare on raw header bytes: str hmac.compare_digest raises on non-ASCII
        # input, which would surface as a 500 instead of a clean 401.
        self.expected = token.encode("utf-8")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_auth = headers.get(b"authorization", b"")
        scheme, _, supplied = raw_auth.partition(b" ")
        if scheme.lower() != b"bearer" or not hmac.compare_digest(supplied, self.expected):
            await _send_unauthorized(send, scope_type)
            return

        await self.app(scope, receive, send)


async def _send_unauthorized(send: Any, scope_type: str) -> None:
    if scope_type == "websocket":
        await send({"type": "websocket.close", "code": 4401})
        return

    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b'{"detail":"MCP bearer token required"}',
        }
    )


def _dump(value: Any) -> Any:
    """Convert Pydantic responses to plain JSON values for MCP clients."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode = "json")
    return value


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an MCP-supplied integer into an inclusive range.

    MCP tools call the Studio route functions directly, which skips FastAPI's
    Query(ge=, le=) validation, so we re-apply the same bounds here.
    """
    return max(low, min(value, high))


def create_studio_mcp() -> FastMCP:
    """Create the Studio MCP server and register the high-value tools."""
    mcp = FastMCP(
        "Unsloth Studio",
        instructions = (
            "Use read tools to inspect the local Studio state before starting GPU work. "
            "Training and export tools can consume substantial VRAM and write files. "
            "Never expose tokens or local paths from tool results unless the user asks."
        ),
    )

    @mcp.tool
    async def studio_status() -> dict[str, Any]:
        """Return the current training, export, inference, and GPU state."""
        from routes.export import get_export_status
        from routes.inference import get_status as get_inference_status
        from routes.training import get_training_status

        from utils.hardware import get_gpu_utilization

        training, export, inference = await _gather_status(
            get_training_status(current_subject = "mcp"),
            get_export_status(current_subject = "mcp"),
            get_inference_status(current_subject = "mcp"),
        )
        return {
            "training": _dump(training),
            "export": _dump(export),
            "inference": _dump(inference),
            "hardware": get_gpu_utilization(),
        }

    @mcp.tool
    async def list_local_models(models_dir: str = "./models") -> dict[str, Any]:
        """List local and cached models available to Studio."""
        from routes.models import list_local_models as list_models
        return _dump(await list_models(models_dir = models_dir, current_subject = "mcp"))

    @mcp.tool
    async def get_training_status() -> dict[str, Any]:
        """Read the active training job, phase, progress, and recent metrics."""
        from routes.training import get_training_status as get_status
        return _dump(await get_status(current_subject = "mcp"))

    @mcp.tool
    async def start_training(config: dict[str, Any]) -> dict[str, Any]:
        """Start a validated Studio training job from a TrainingStartRequest-shaped object.

        The config is validated by the same Pydantic model used by the Studio UI.
        Call get_training_status first and do not start work while another job runs.
        """
        from models import TrainingStartRequest
        from routes.training import start_training as start

        request = TrainingStartRequest.model_validate(config)
        # Pass via_api_key explicitly (a direct call leaves it a Depends object).
        # MCP drives Studio like the UI session, so it coexists and frees VRAM.
        return _dump(await start(request, current_subject = "mcp", via_api_key = False))

    @mcp.tool
    async def stop_training(save: bool = True) -> dict[str, Any]:
        """Ask the active training process to stop at its next safe checkpoint."""
        from routes.training import TrainingStopRequest, stop_training as stop
        return _dump(await stop(TrainingStopRequest(save = save), current_subject = "mcp"))

    @mcp.tool
    async def list_training_runs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List completed and stopped training runs, newest first."""
        from routes.training_history import list_training_runs as list_runs

        # Clamp here (direct call skips Query bounds); a negative LIMIT = no limit.
        limit = _clamp(limit, 1, 200)
        offset = max(0, offset)
        return _dump(await list_runs(limit = limit, offset = offset, current_subject = "mcp"))

    @mcp.tool
    def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
        """Validate a Data Recipe with the same validator used by Studio."""
        from models.data_recipe import RecipePayload
        from routes.data_recipe.validate import validate

        return _dump(validate(RecipePayload(recipe = recipe)))

    @mcp.tool
    def get_recipe_job_status(job_id: str) -> dict[str, Any]:
        """Read the status of a Data Recipe job."""
        from routes.data_recipe.jobs import job_status
        return _dump(job_status(job_id))

    @mcp.tool
    def get_recipe_job_dataset(
        job_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Read a bounded page of generated Data Recipe rows."""
        from routes.data_recipe.jobs import job_dataset

        # Clamp here (direct call skips FastAPI's Query bounds).
        limit = _clamp(limit, 1, 500)
        offset = max(0, offset)
        return _dump(job_dataset(job_id, limit = limit, offset = offset))

    @mcp.tool
    async def load_checkpoint(
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        approved_remote_code_fingerprint: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint into the export backend.

        Export runs in its own subprocess and coexists with training and
        inference; it does not unload them, so a load can fail with a clear
        out-of-memory error if the GPU is already full. Pass hf_token to load a
        gated checkpoint, and approved_remote_code_fingerprint to retry a
        trust_remote_code load that was blocked pending review.
        """
        from models import LoadCheckpointRequest
        from routes.export import load_checkpoint as load

        request = LoadCheckpointRequest(
            checkpoint_path = checkpoint_path,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            trust_remote_code = trust_remote_code,
            approved_remote_code_fingerprint = approved_remote_code_fingerprint,
            hf_token = hf_token,
        )
        return _dump(await load(request, current_subject = "mcp"))

    @mcp.tool
    async def export_gguf(
        save_directory: str,
        quantization_method: str | list[str] = "Q4_K_M",
        push_to_hub: bool = False,
        repo_id: str | None = None,
        hf_token: str | None = None,
        imatrix: bool = False,
        imatrix_path: str | None = None,
    ) -> dict[str, Any]:
        """Export the loaded model to GGUF using Studio's existing path validation.

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
        )
        return _dump(await export(request, current_subject = "mcp"))

    return mcp


async def _gather_status(*coroutines: Any) -> tuple[Any, ...]:
    """Gather independent status calls without letting one optional backend fail all state."""
    import asyncio

    results = await asyncio.gather(*coroutines, return_exceptions = True)
    return tuple(
        {"error": str(result)} if isinstance(result, Exception) else result for result in results
    )
