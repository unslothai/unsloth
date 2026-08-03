# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import asyncio
import json
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Optional

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
    list_other_run_output_dirs,
    list_runs,
    update_run_display_name,
)
from utils.models.checkpoints import has_preview_model, preview_ref
from utils.paths import outputs_root, resolve_output_dir
from utils.preview_sharing_settings import get_preview_sharing_enabled
from utils.preview_token import sign_preview_ref

logger = get_logger(__name__)

router = APIRouter()


def _canonical_output_dir(output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir or not str(output_dir).strip():
        return None
    raw = str(output_dir).strip()
    native_path = Path(raw).expanduser()
    if (
        PureWindowsPath(raw).is_absolute() or PurePosixPath(raw).is_absolute()
    ) and not native_path.is_absolute():
        return None
    try:
        outputs_base = outputs_root().expanduser().resolve(strict = False)
        try:
            candidate = resolve_output_dir(output_dir)
        except ValueError:
            candidate = native_path if native_path.is_absolute() else outputs_base / native_path
        resolved = candidate.resolve(strict = False)
        resolved.relative_to(outputs_base)
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


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


def _summary_from_row(
    row: dict,
    sharing_on: bool,
    resource_cache: Optional[dict[str, bool]] = None,
) -> TrainingRunSummary:
    can_resume = (
        can_resume_run(row)
        if resource_cache is None
        else can_resume_run(row, resource_cache = resource_cache)
    )
    return TrainingRunSummary(
        **{
            **{k: v for k, v in row.items() if k != "config_json"},
            "can_resume": can_resume,
            "artifacts_available": artifacts_present(row.get("output_dir")),
            **_preview_fields(row.get("output_dir"), sharing_on),
        }
    )


def _summaries_from_rows(rows: list[dict], sharing_on: bool) -> list[TrainingRunSummary]:
    resource_cache: dict[str, bool] = {}
    return [_summary_from_row(row, sharing_on, resource_cache) for row in rows]


def _delete_run_output_dir(run_id: str, output_dir: str) -> bool:
    resolved = _canonical_output_dir(output_dir)
    if resolved is None:
        logger.warning(
            "Cannot resolve output_dir for run %s; skipping disk cleanup: %s",
            run_id,
            output_dir,
        )
        return False
    outputs_base = outputs_root().expanduser().resolve(strict = False)

    if resolved == outputs_base:
        logger.warning(
            "Refusing to delete the outputs root itself for run %s: %s", run_id, resolved
        )
        return False

    if not resolved.exists():
        return True

    if not resolved.is_dir():
        logger.warning("Run %s output path is not a directory; skipping: %s", run_id, resolved)
        return False

    try:
        shutil.rmtree(resolved)
        logger.info("Deleted adapter directory for run %s: %s", run_id, resolved)
        return True
    except OSError:
        logger.exception("Failed to delete adapter directory for run %s: %s", run_id, resolved)
        return False


def _active_training_output_dir() -> Optional[str]:
    from core.training import get_training_backend
    return get_training_backend().active_output_dir()


def _same_output_dir(first: Optional[str], second: Optional[str]) -> bool:
    first_path = _canonical_output_dir(first)
    second_path = _canonical_output_dir(second)
    if first_path is None or second_path is None:
        return False
    try:
        if first_path.exists() and second_path.exists() and first_path.samefile(second_path):
            return True
    except OSError:
        pass
    return first_path == second_path


def _output_dirs_overlap(first: Optional[str], second: Optional[str]) -> bool:
    first_path = _canonical_output_dir(first)
    second_path = _canonical_output_dir(second)
    if first_path is None or second_path is None:
        return False
    if _same_output_dir(str(first_path), str(second_path)):
        return True
    return first_path in second_path.parents or second_path in first_path.parents


def _output_dir_shared(output_dir: str, run_id: str) -> bool:
    return any(
        _output_dirs_overlap(output_dir, candidate)
        for candidate in list_other_run_output_dirs(run_id)
    )


_ArtifactDeleteOutcome = Literal["deleted", "active", "shared", "failed"]


def _delete_run_output_dir_guarded(run_id: str, output_dir: str) -> _ArtifactDeleteOutcome:
    from core.training.lifecycle import training_lifecycle_guard
    with training_lifecycle_guard():
        if _output_dirs_overlap(output_dir, _active_training_output_dir()):
            return "active"
        if _output_dir_shared(output_dir, run_id):
            return "shared"
        return "deleted" if _delete_run_output_dir(run_id, output_dir) else "failed"


@router.get("/runs", response_model = TrainingRunListResponse)
async def list_training_runs(
    limit: int = Query(50, ge = 1, le = 200),
    offset: int = Query(0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    """List training runs, newest first."""
    result = list_runs(limit = limit, offset = offset)
    sharing_on = get_preview_sharing_enabled()
    runs = await asyncio.to_thread(
        _summaries_from_rows,
        result["runs"],
        sharing_on,
    )
    return TrainingRunListResponse(
        runs = runs,
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

    summary = await asyncio.to_thread(
        _summary_from_row,
        run,
        get_preview_sharing_enabled(),
    )
    return TrainingRunDetailResponse(
        run = summary,
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
    return await asyncio.to_thread(
        _summary_from_row,
        refreshed,
        get_preview_sharing_enabled(),
    )


@router.delete("/runs/{run_id}", response_model = TrainingRunDeleteResponse)
async def delete_training_run(
    run_id: str,
    delete_artifacts: bool = Query(
        False,
        description = "Also delete the run's output directory on disk",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """Delete a training run and its metrics (CASCADE)."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = f"Run {run_id} not found")
    if run["status"] == "running":
        raise HTTPException(status_code = 409, detail = "Cannot delete a running training run")
    logger.info("Deleting training run %s (delete_artifacts=%s)", run_id, delete_artifacts)
    artifacts_deleted = False
    artifacts_kept_reason: Optional[str] = None
    if delete_artifacts:
        output_dir = run.get("output_dir")
        if output_dir:
            delete_outcome = await asyncio.to_thread(
                _delete_run_output_dir_guarded,
                run_id,
                output_dir,
            )
            if delete_outcome == "active":
                raise HTTPException(
                    status_code = 409,
                    detail = {
                        "code": "training_artifacts_in_use",
                        "message": (
                            "Cannot delete artifacts while a training run is writing "
                            "to this directory"
                        ),
                    },
                )
            if delete_outcome == "shared":
                artifacts_kept_reason = "shared_output_dir"
                logger.info(
                    "Keeping artifacts for run %s; another run shares %s", run_id, output_dir
                )
            elif delete_outcome == "failed":
                raise HTTPException(
                    status_code = 409,
                    detail = {
                        "code": "training_artifact_deletion_failed",
                        "message": (
                            "Could not delete run artifacts; training history was retained"
                        ),
                    },
                )
            else:
                artifacts_deleted = True
    delete_run(run_id)
    return TrainingRunDeleteResponse(
        status = "deleted",
        message = f"Run {run_id} deleted",
        artifacts_deleted = artifacts_deleted,
        artifacts_kept_reason = artifacts_kept_reason,
    )
