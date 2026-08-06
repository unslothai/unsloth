# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training history API routes — browse, view, and delete past training runs.
"""

import asyncio
import json
import shutil
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Optional, Union

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


def _resume_blocked_reason(row: dict) -> Optional[str]:
    """The provenance gate's own explanation, for runs it refuses.

    Only consulted for a row that is already known unresumable, so the extra work is
    bounded to those. Returns None when the checkpoint is what is missing, leaving the
    client's existing wording in place for that case.
    """
    from core.training.provenance import resource_provenance_resume_blocker
    from core.training.resume import has_resume_state, training_run_config

    try:
        # A row whose checkpoint is gone is refused for that reason, not provenance.
        # Asking the gate anyway hands the client a provenance sentence for a missing
        # checkpoint, which is the wrong diagnosis on the more common case.
        if not has_resume_state(row.get("output_dir")):
            return None
        return resource_provenance_resume_blocker(training_run_config(row))
    except Exception:
        return None


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
            "resume_blocked_reason": None if can_resume else _resume_blocked_reason(row),
            "artifacts_available": artifacts_present(row.get("output_dir")),
            **_preview_fields(row.get("output_dir"), sharing_on),
        }
    )


def _summaries_from_rows(rows: list[dict], sharing_on: bool) -> list[TrainingRunSummary]:
    resource_cache: dict[str, bool] = {}
    return [_summary_from_row(row, sharing_on, resource_cache) for row in rows]


def _delete_run_output_dir(run_id: str, output_dir: str) -> Union[Path, bool]:
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

    staged = resolved.with_name(f".{resolved.name}.deleting-{uuid.uuid4().hex}")
    try:
        # A same-parent rename, not an rmtree: the run is logically gone immediately but
        # the bytes survive until the database row is committed, so a failed row delete
        # can roll the whole operation back instead of stranding a row whose artifacts
        # are silently missing. _purge_staged_output_dir does the destructive half.
        resolved.rename(staged)
        return staged
    except OSError:
        logger.exception("Failed to stage adapter directory for run %s: %s", run_id, resolved)
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


def _delete_run_output_dir_guarded(
    run_id: str, output_dir: str
) -> tuple[_ArtifactDeleteOutcome, Optional[Path], Optional[Path]]:
    """Move the run's artifacts aside, reversibly, instead of destroying them.

    Deleting the directory outright and only then removing the database row leaves an
    unrecoverable half-state if the row delete fails: the artifacts are gone and the row
    survives with ``output_dir`` still populated, which is indistinguishable from the
    legitimate "history kept, files kept" outcome. A same-parent rename is atomic and
    costs nothing, so the destructive step can wait until the row is actually gone.

    Returns the outcome plus (original, staged) paths when there is something to purge.
    """
    from core.training.lifecycle import training_lifecycle_guard
    with training_lifecycle_guard():
        if _output_dirs_overlap(output_dir, _active_training_output_dir()):
            return "active", None, None
        if _output_dir_shared(output_dir, run_id):
            return "shared", None, None

        resolved = _canonical_output_dir(output_dir)
        if resolved is None:
            logger.warning(
                "Cannot resolve output_dir for run %s; skipping disk cleanup: %s",
                run_id,
                output_dir,
            )
            return "failed", None, None
        outputs_base = outputs_root().expanduser().resolve(strict = False)
        if resolved == outputs_base:
            logger.warning(
                "Refusing to delete the outputs root itself for run %s: %s", run_id, resolved
            )
            return "failed", None, None
        if not resolved.exists():
            return "deleted", None, None
        if not resolved.is_dir():
            logger.warning("Run %s output path is not a directory; skipping: %s", run_id, resolved)
            return "failed", None, None

        staged = _delete_run_output_dir(run_id, str(resolved))
        if not staged:
            return "failed", None, None
        return "deleted", resolved, staged if isinstance(staged, Path) else None


def _restore_staged_output_dir(original: Path, staged: Path) -> None:
    """Undo the staging rename after a failed database delete."""
    try:
        staged.rename(original)
    except OSError:
        logger.exception("Failed to restore staged artifacts from %s to %s", staged, original)


def _purge_staged_output_dir(run_id: str, staged: Path) -> None:
    """Remove the staged copy once the row is gone. A failure here only leaks disk."""
    try:
        shutil.rmtree(staged)
        logger.info("Deleted adapter directory for run %s: %s", run_id, staged)
    except OSError:
        logger.exception("Failed to purge staged artifacts for run %s: %s", run_id, staged)


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
    staged_original: Optional[Path] = None
    staged_path: Optional[Path] = None
    if delete_artifacts:
        output_dir = run.get("output_dir")
        if output_dir:
            delete_outcome, staged_original, staged_path = await asyncio.to_thread(
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
    try:
        delete_run(run_id)
    except Exception:
        # The artifacts are only staged, so the whole operation rolls back and the run
        # is exactly as it was. Destroying them first would leave a row whose artifacts
        # are silently gone, looking just like a deliberate "kept history" outcome.
        if staged_original is not None and staged_path is not None:
            await asyncio.to_thread(_restore_staged_output_dir, staged_original, staged_path)
        raise
    if staged_path is not None:
        await asyncio.to_thread(_purge_staged_output_dir, run_id, staged_path)
    return TrainingRunDeleteResponse(
        status = "deleted",
        message = f"Run {run_id} deleted",
        artifacts_deleted = artifacts_deleted,
        artifacts_kept_reason = artifacts_kept_reason,
    )
