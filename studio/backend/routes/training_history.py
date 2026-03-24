# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from loggers import get_logger

from auth.authentication import get_current_subject
from models import (
    TrainingRunDeleteResponse,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunMetrics,
    TrainingRunSummary,
)
from storage.studio_db import delete_run, get_run, get_run_metrics, list_runs

logger = get_logger(__name__)

router = APIRouter()


@router.get("/runs", response_model = TrainingRunListResponse)
async def list_training_runs(
    limit: int = Query(50, ge = 1, le = 200),
    offset: int = Query(0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    """List training runs, newest first."""
    result = list_runs(limit = limit, offset = offset)
    return TrainingRunListResponse(
        runs = [TrainingRunSummary(**r) for r in result["runs"]],
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
        run = TrainingRunSummary(**{k: v for k, v in run.items() if k != "config_json"}),
        config = config,
        metrics = TrainingRunMetrics(**metrics_data),
    )


@router.delete("/runs/{run_id}", response_model = TrainingRunDeleteResponse)
async def delete_training_run(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Delete a training run and its metrics (CASCADE)."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")
    if run["status"] == "running":
        raise HTTPException(
            status_code = 409, detail = "Cannot delete a running training run"
        )
    logger.info("Deleting training run %s", run_id)
    delete_run(run_id)
    return TrainingRunDeleteResponse(
        status = "deleted",
        message = f"Run {run_id} deleted",
    )
