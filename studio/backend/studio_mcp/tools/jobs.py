# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Long-running Unsloth Studio jobs: ``run_recipe`` starts a data recipe and ``get_job`` reports on any job, waiting for it to move when asked."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import profile
from studio_mcp import jobs, loading
from studio_mcp.caller import current_caller
from studio_mcp.job_kinds import RECIPE_ROUTES
from studio_mcp.media import ReturnMedia
from studio_mcp.outputs import JobStatus, RecipeError, RecipeResult
from studio_mcp.tools import READ_ONLY, WRITES, as_dict, opt_text, route_json


async def get_job(
    kind: Literal[
        "video", "recipe", "export", "training", "diffusion_training", "load", "download"
    ],
    id: Optional[str] = None,
    wait_s: Annotated[
        int,
        Field(
            ge = 0,
            le = loading.MAX_WAIT_S,
            description = (
                "Seconds to wait for the job to finish or move on, 0 to 240. "
                "50 stays under Codex's 60 s default tool timeout; above that, raise the client's timeout."
            ),
        ),
    ] = 0,
    ctx: Optional[Context] = None,
    rows: Annotated[Optional[int], Field(ge = 1, le = 500)] = None,
    offset: Annotated[int, Field(ge = 0)] = 0,
    return_media: Annotated[
        ReturnMedia,
        Field(
            description = 'For a completed video: "inline" (default) its thumbnail; "preview" its first frame '
            'at preview_px; "link" no picture. The MP4 itself is always a link.'
        ),
    ] = "inline",
    preview_px: Annotated[
        Optional[int],
        Field(ge = 32, le = 1024, description = "Long side of a video preview in pixels; default 1024."),
    ] = None,
) -> ToolResult:
    """The state of a long-running job.
    Follow every long job with ``wait_s`` instead of polling: the call checks every 2 s, reports progress, and returns once the job finishes, its status changes (for an export or a training run, also its phase), or ``wait_s`` runs out; a running job then gives ``poll_after_s``.
    kind "video": with ``id``, its status and progress, and once completed a thumbnail plus the video's URL; without ``id``, recent video jobs.
    kind "recipe": a data recipe job's status and progress (without ``id``, the current one), and with ``rows`` that many generated rows from ``offset``.
    Unsloth Studio runs one recipe at a time; a recipe job id is not tied to an account.
    kind "export": an <<export_model: export_model | export>> job by its id (without ``id``, this key's export jobs), with its output named relative to Unsloth Studio's exports folder.
    kind "training" or "diffusion_training": the current run<<start_training:  started by start_training>> (``id`` is its job id), and once completed its run folder and checkpoint names; ``loss_history`` names the list_training_runs call with the full loss curve.
    kind "load": a <<load_model: load_model call | load>> that returned ``status`` "loading", by its ``job_id``; once completed, ``load`` is the load's result with what it unloaded.
    kind "download": a download_model download by the id it returned."""
    return await jobs.follow(
        current_caller(),
        kind,
        id,
        wait_s = wait_s,
        ctx = ctx,
        rows = rows,
        offset = offset,
        return_media = return_media,
        preview_px = preview_px,
    )


async def run_recipe(
    recipe: dict[str, Any], mode: Literal["validate", "preview", "full"] = "validate"
) -> RecipeResult:
    """Validate a Data Recipe, or run it. "validate" checks it and returns any errors; "preview" generates a few rows; "full" generates the whole dataset and saves it. A run returns a job id at once: follow it with get_job(kind="recipe"). Recipes that define a local (stdio) MCP server, or take a provider key from a server environment variable (api_key_env), can only be run from the Unsloth Studio UI."""
    if mode == "validate":
        payload = await route_json(
            "POST", f"{RECIPE_ROUTES}/validate", json_body = {"recipe": recipe}
        )
        payload = as_dict(payload)
        errors = [
            RecipeError.from_route(e, message = opt_text(e.get("message")) or "invalid")
            for e in payload.get("errors") or []
            if isinstance(e, dict)
        ]
        return RecipeResult(mode = mode, valid = payload.get("valid") is True, errors = errors)
    created = await route_json(
        "POST",
        f"{RECIPE_ROUTES}/jobs",
        json_body = {"recipe": recipe, "run": {"execution_type": mode}},
    )
    job_id = opt_text(created.get("job_id")) if isinstance(created, dict) else None
    if job_id is None:
        raise ToolError("Unsloth Studio did not start the recipe")
    return RecipeResult(mode = mode, job_id = job_id)


# A recipe can send host env credentials and host files to the provider endpoint it names.
FACTS = {"run_recipe": {profile.SENDS_HOST_SECRETS}}

TOOLS = ((run_recipe, WRITES), (get_job, READ_ONLY, JobStatus.model_json_schema()))
