# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import json
import shutil

from fastapi import APIRouter, Depends, HTTPException, Query
from loggers import get_logger

from auth.authentication import get_current_subject
from core.training.resume import artifacts_present, can_resume_run
from models import (
    TrainingRunDeleteResponse,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunMetrics,
    TrainingRunSummary,
    TrainingRunUpdateRequest,
)
from storage.studio_db import (
    delete_run,
    get_run,
    get_run_metrics,
    list_runs,
    update_run_display_name,
)
from utils.paths import outputs_root, resolve_output_dir

logger = get_logger(__name__)

router = APIRouter()


def _summary_from_row(row: dict) -> TrainingRunSummary:
    payload = {k: v for k, v in row.items() if k != "config_json"}
    payload["can_resume"] = can_resume_run(row)
    payload["artifacts_available"] = artifacts_present(row.get("output_dir"))
    return TrainingRunSummary(**payload)


@router.get("/runs", response_model = TrainingRunListResponse)
async def list_training_runs(
    limit: int = Query(50, ge = 1, le = 200),
    offset: int = Query(0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    """List training runs, newest first."""
    result = list_runs(limit = limit, offset = offset)
    return TrainingRunListResponse(
        runs = [_summary_from_row(r) for r in result["runs"]],
        total = result["total"],
    )


@router.get("/runs/{run_id}", response_model = TrainingRunDetailResponse)
async def get_training_run_detail(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Get a single training run with full config and metrics."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")

    try:
        config = json.loads(run.get("config_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        logger.debug("Failed to parse config_json for run %s", run_id)
        config = {}

    metrics_data = get_run_metrics(run_id)

    return TrainingRunDetailResponse(
        run = _summary_from_row(run),
        config = config,
        metrics = TrainingRunMetrics(**metrics_data),
    )


@router.patch("/runs/{run_id}", response_model = TrainingRunSummary)
async def update_training_run(
    run_id: str,
    payload: TrainingRunUpdateRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Update mutable fields on a training run (currently only display_name)."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")

    if "display_name" in payload.model_fields_set:
        next_display = payload.display_name
        if next_display is not None:
            next_display = next_display.strip() or None
        update_run_display_name(run_id, next_display)

    refreshed = get_run(run_id)
    if refreshed is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")
    return _summary_from_row(refreshed)


@router.delete("/runs/{run_id}", response_model = TrainingRunDeleteResponse)
async def delete_training_run(
    run_id: str,
    delete_artifacts: bool = Query(
        False,
        description = "Also remove the run's on-disk adapter directory.",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """Delete a training run from history (optionally its on-disk adapter directory)."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")
    if run["status"] == "running":
        raise HTTPException(
            status_code = 409, detail = "Cannot delete a running training run"
        )

    logger.info(
        "Deleting training run %s (delete_artifacts=%s)", run_id, delete_artifacts
    )

    if delete_artifacts:
        output_dir = run.get("output_dir")
        if output_dir:
            _delete_run_output_dir(run_id, output_dir)

    delete_run(run_id)
    return TrainingRunDeleteResponse(
        status = "deleted",
        message = f"Run {run_id} deleted",
    )


def _delete_run_output_dir(run_id: str, output_dir: str) -> None:
    try:
        resolved = resolve_output_dir(output_dir).resolve()
        outputs_base = outputs_root().resolve()
    except (OSError, ValueError):
        logger.warning(
            "Cannot resolve output_dir for run %s; skipping disk cleanup: %s",
            run_id,
            output_dir,
        )
        return

    try:
        resolved.relative_to(outputs_base)
    except ValueError:
        logger.warning(
            "Refusing to delete output_dir outside outputs_root for run %s: %s",
            run_id,
            resolved,
        )
        return

    if not resolved.exists():
        return

    if not resolved.is_dir():
        logger.warning(
            "Run %s output path is not a directory; skipping: %s", run_id, resolved
        )
        return

    try:
        shutil.rmtree(resolved)
        logger.info("Deleted adapter directory for run %s: %s", run_id, resolved)
    except OSError:
        logger.exception(
            "Failed to delete adapter directory for run %s: %s", run_id, resolved
        )
