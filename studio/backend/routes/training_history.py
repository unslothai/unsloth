# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
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
from utils.models.checkpoints import has_preview_model, preview_ref, scan_checkpoints
from utils.preview_sharing_settings import get_preview_sharing_enabled
from utils.preview_token import sign_preview_ref

logger = get_logger(__name__)

router = APIRouter()

_FILESYSTEM_RUN_PREFIX = "filesystem-"


def _filesystem_run_id(output_dir: str) -> str:
    digest = hashlib.sha256(str(Path(output_dir).resolve()).encode()).hexdigest()[:24]
    return f"{_FILESYSTEM_RUN_PREFIX}{digest}"


def _trainer_history(checkpoint_path: str) -> list[dict]:
    try:
        state = json.loads((Path(checkpoint_path) / "trainer_state.json").read_text())
        history = state.get("log_history", [])
        return history if isinstance(history, list) else []
    except (OSError, TypeError, json.JSONDecodeError):
        return []


def _filesystem_runs() -> list[dict]:
    """Build read-only history summaries for runs found only on disk."""
    runs = []
    for name, checkpoints, metadata in scan_checkpoints():
        if not checkpoints:
            continue
        first_path = Path(checkpoints[0][1])
        output_dir = first_path if first_path.name == name else first_path.parent
        numbered = [
            (int(cp_name.removeprefix("checkpoint-")), cp_path, loss)
            for cp_name, cp_path, loss in checkpoints
            if cp_name.removeprefix("checkpoint-").isdigit()
        ]
        final_step, state_path, final_loss = max(numbered, default=(0, str(first_path), None))
        history = _trainer_history(state_path)
        losses = [entry.get("loss") for entry in history if isinstance(entry.get("loss"), (int, float))]
        modified = output_dir.stat().st_mtime
        runs.append(
            {
                "id": _filesystem_run_id(str(output_dir)),
                "status": "completed" if first_path == output_dir else "stopped",
                "model_name": metadata.get("base_model") or name,
                "project_name": None,
                "dataset_name": "Recovered from checkpoint folder",
                "display_name": name,
                "started_at": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
                "ended_at": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
                "total_steps": final_step or None,
                "final_step": final_step or None,
                "final_loss": final_loss,
                "output_dir": str(output_dir),
                "duration_seconds": None,
                "error_message": None,
                "loss_sparkline": losses[-100:] or None,
                "can_resume": False,
                "resumed_later": False,
            }
        )
    return runs


def _find_filesystem_run(run_id: str) -> Optional[dict]:
    if not run_id.startswith(_FILESYSTEM_RUN_PREFIX):
        return None
    return next((run for run in _filesystem_runs() if run["id"] == run_id), None)


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
    # Merge database history with filesystem discovery. Paths already owned by
    # DB records are de-duplicated; recovered entries remain read-only.
    db_result = list_runs(limit = 1_000_000, offset = 0)
    db_runs = db_result["runs"]
    db_paths = {
        str(Path(run["output_dir"]).resolve())
        for run in db_runs
        if run.get("output_dir")
    }
    recovered = [
        run
        for run in _filesystem_runs()
        if str(Path(run["output_dir"]).resolve()) not in db_paths
    ]
    merged = sorted(db_runs + recovered, key = lambda run: run["started_at"], reverse = True)
    page = merged[offset : offset + limit]
    sharing_on = get_preview_sharing_enabled()
    return TrainingRunListResponse(
        runs = [
            TrainingRunSummary(
                **{
                    **r,
                    "can_resume": (
                        False
                        if r["id"].startswith(_FILESYSTEM_RUN_PREFIX)
                        else can_resume_run(r)
                    ),
                    **_preview_fields(r.get("output_dir"), sharing_on),
                }
            )
            for r in page
        ],
        total = len(merged),
    )


@router.get("/runs/{run_id}", response_model = TrainingRunDetailResponse)
async def get_training_run_detail(run_id: str, current_subject: str = Depends(get_current_subject)):
    """Get a single training run with full config and metrics."""
    run = get_run(run_id)
    if run is None:
        run = _find_filesystem_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")

    try:
        config = json.loads(run.get("config_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        logger.debug("Failed to parse config_json for run %s", run_id)
        config = {}

    if run_id.startswith(_FILESYSTEM_RUN_PREFIX):
        checkpoints = scan_checkpoints(outputs_dir = str(Path(run["output_dir"]).parent))
        match = next((item for item in checkpoints if item[0] == Path(run["output_dir"]).name), None)
        numbered = (
            [checkpoint for checkpoint in match[1] if checkpoint[0].startswith("checkpoint-")]
            if match
            else []
        )
        state_path = numbered[0][1] if numbered else run["output_dir"]
        history = _trainer_history(state_path)
        loss_entries = [entry for entry in history if isinstance(entry.get("loss"), (int, float))]
        metrics_data = {
            "step_history": [int(entry.get("step", 0)) for entry in loss_entries],
            "loss_history": [float(entry["loss"]) for entry in loss_entries],
            "loss_step_history": [int(entry.get("step", 0)) for entry in loss_entries],
        }
    else:
        metrics_data = get_run_metrics(run_id)

    return TrainingRunDetailResponse(
        run = TrainingRunSummary(
            **{
                **{k: v for k, v in run.items() if k != "config_json"},
                "can_resume": (
                    False
                    if run_id.startswith(_FILESYSTEM_RUN_PREFIX)
                    else can_resume_run(run)
                ),
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
