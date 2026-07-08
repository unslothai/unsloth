# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio Data Designer (synthetic-data recipes).

A "recipe" is a dict describing the columns/providers of a synthetic dataset
(the same shape the Data Designer UI builds). Validate or preview it cheaply,
then ``recipe_start`` spawns a generation job (subprocess); poll with
``recipe_status`` and abort with ``recipe_cancel``.

These tools require the optional ``data_designer`` package; calls return a
clear error if it is not installed.

Known limitation: recipes that reference a *local* Studio model provider
(``is_local: true``) are not wired up here. The HTTP Data Designer route rewrites
such providers to a loopback inference endpoint and mints an internal API key
against the running Studio UI server -- coordination the standalone MCP path
does not assume. Point recipes at a hosted/remote provider instead.
"""

from __future__ import annotations

from typing import Any, Optional

from fastmcp import FastMCP

from loggers import get_logger

logger = get_logger(__name__)

GROUP = "recipe"

_READ_ONLY = {"readOnlyHint": True}
_STATEFUL = {"destructiveHint": False}


def recipe_validate(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate a Data Designer recipe without running it.

    Returns ``{"valid": true}`` on success, or ``{"valid": false, "error": ...}``
    with the validation detail. ``recipe`` must include a ``columns`` list.
    """
    from core.data_recipe.service import validate_recipe

    if not recipe.get("columns"):
        return {"valid": False, "error": "Recipe must include a non-empty 'columns' list."}
    try:
        validate_recipe(recipe)
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True}


def recipe_preview(recipe: dict[str, Any], num_records: int = 5) -> dict[str, Any]:
    """Generate a small preview of a recipe (no job spawned).

    Returns up to ``num_records`` synthetic rows plus any processor artifacts
    and analysis. Useful for iterating on a recipe before ``recipe_start``.
    """
    from core.data_recipe.service import preview_recipe

    try:
        dataset, artifacts, analysis = preview_recipe(recipe, num_records = num_records)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "dataset": dataset,
        "processor_artifacts": artifacts,
        "analysis": analysis,
    }


def recipe_start(recipe: dict[str, Any], run: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Start a Data Designer generation job. Returns immediately with a job id.

    ``recipe`` must include ``columns``. ``run`` carries execution options:
    ``execution_type`` (``"full"`` default, or ``"preview"``), ``run_name``,
    ``run_config``, and provider settings. Poll with ``recipe_status``.
    """
    from core.data_recipe.jobs.manager import get_job_manager

    if not recipe.get("columns"):
        return {"success": False, "error": "Recipe must include a non-empty 'columns' list."}

    run_payload = dict(run or {})
    run_payload.pop("artifact_path", None)
    run_payload.pop("dataset_name", None)
    execution_type = str(run_payload.get("execution_type") or "full").strip().lower()
    if execution_type not in {"preview", "full"}:
        return {"success": False, "error": "execution_type must be 'preview' or 'full'."}
    run_payload["execution_type"] = execution_type

    # Validate run_config up front (matches routes/data_recipe/jobs.py). The
    # optional data_designer package is imported lazily: if it's absent we skip
    # this check and let the worker surface the failure via recipe_status.
    run_config_raw = run_payload.get("run_config")
    if run_config_raw is not None:
        try:
            from data_designer.config.run_config import RunConfig
            RunConfig.model_validate(run_config_raw)
        except ImportError:
            pass
        except Exception as exc:
            return {"success": False, "error": f"invalid run_config: {exc}"}

    mgr = get_job_manager()
    try:
        job_id = mgr.start(recipe = recipe, run = run_payload, internal_api_key_id = None)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "job_id": job_id,
        "message": "Recipe job started. Poll recipe_status for progress.",
    }


def recipe_status(job_id: Optional[str] = None) -> dict[str, Any]:
    """Get the status of a recipe job.

    With ``job_id``, reports that specific job; without it, reports the
    current/last job (or ``idle`` if none).
    """
    from core.data_recipe.jobs.manager import get_job_manager

    mgr = get_job_manager()
    if job_id is not None:
        status = mgr.get_status(job_id)
        if status is None:
            return {"idle": True, "message": f"No recipe job found with id {job_id!r}."}
        return status
    status = mgr.get_current_status()
    if status is None:
        return {"idle": True, "message": "No recipe job has been started."}
    return status


def recipe_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a running recipe job by terminating its subprocess."""
    from core.data_recipe.jobs.manager import get_job_manager

    mgr = get_job_manager()
    cancelled = mgr.cancel(job_id)
    return {
        "success": cancelled,
        "message": "Job cancelled" if cancelled else "Job not found or not running",
    }


def register(mcp: FastMCP) -> list[str]:
    """Register the recipe tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    mcp.tool(recipe_validate, annotations = _READ_ONLY)
    names.append("recipe_validate")
    mcp.tool(recipe_preview)
    names.append("recipe_preview")
    for fn in (recipe_start, recipe_status, recipe_cancel):
        mcp.tool(fn, annotations = _STATEFUL)
        names.append(fn.__name__)
    return names
