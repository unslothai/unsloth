# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loggers import get_logger

from auth.authentication import get_current_subject
from core.training.resume import can_resume_run
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
from utils.models.checkpoints import has_preview_model, preview_ref
from utils.preview_sharing_settings import get_preview_sharing_enabled
from utils.preview_token import sign_preview_ref

logger = get_logger(__name__)

router = APIRouter()


def _preview_fields(output_dir: Optional[str], sharing_on: bool) -> dict:
    """Previewability + the signed `/p` share ref for a run's output dir.

    The signature is what makes the share link a capability: these routes are
    authenticated, so only the run's owner ever receives it. When public sharing
    is switched off, omit the signature so the UI hides the copy-link affordance
    (and the link would 404 anyway). ``sharing_on`` is resolved once per request.
    """
    ref = preview_ref(output_dir)
    return {
        "has_preview_model": has_preview_model(output_dir),
        "preview_ref": ref,
        "preview_sig": sign_preview_ref(ref) if (ref and sharing_on) else None,
    }


@router.get("/runs", response_model = TrainingRunListResponse)
async def list_training_runs(
    limit: int = Query(50, ge = 1, le = 200),
    offset: int = Query(0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    """List training runs, newest first."""
    result = list_runs(limit = limit, offset = offset)
    sharing_on = get_preview_sharing_enabled()
    return TrainingRunListResponse(
        runs = [
            TrainingRunSummary(
                **{
                    **r,
                    "can_resume": can_resume_run(r),
                    **_preview_fields(r.get("output_dir"), sharing_on),
                }
            )
            for r in result["runs"]
        ],
        total = result["total"],
    )


@router.get("/runs/{run_id}", response_model = TrainingRunDetailResponse)
async def get_training_run_detail(run_id: str, current_subject: str = Depends(get_current_subject)):
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
        run = TrainingRunSummary(
            **{
                **{k: v for k, v in run.items() if k != "config_json"},
                "can_resume": can_resume_run(run),
                **_preview_fields(run.get("output_dir"), get_preview_sharing_enabled()),
            }
        ),
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
    return TrainingRunSummary(
        **{
            **{k: v for k, v in refreshed.items() if k != "config_json"},
            "can_resume": can_resume_run(refreshed),
            **_preview_fields(refreshed.get("output_dir"), get_preview_sharing_enabled()),
        }
    )


@router.delete("/runs/{run_id}", response_model = TrainingRunDeleteResponse)
async def delete_training_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    """Delete a training run and its metrics (CASCADE)."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")
    if run["status"] == "running":
        raise HTTPException(status_code = 409, detail = "Cannot delete a running training run")
    logger.info("Deleting training run %s", run_id)
    delete_run(run_id)
    return TrainingRunDeleteResponse(
        status = "deleted",
        message = f"Run {run_id} deleted",
    )
