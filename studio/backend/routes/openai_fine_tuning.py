# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-compatible fine-tuning API (``/v1/files`` and ``/v1/fine_tuning/jobs``) over Studio's own training.

A thin translation layer: uploads land where ``/api/hub/datasets/upload`` puts them, jobs start through the
same ``start_training`` handler as ``/api/train/start`` (every guard and refusal included), and job state is
read back from the training history, so a job is just a Studio training run under another name. Nothing here
grants a capability an API key does not already have through ``/api/train``.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict

from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.services.datasets import local as local_datasets
from hub.utils.host_paths import host_paths_visible, redact_host_paths, redact_paths_in_text
from loggers import get_logger
from models.training import TrainingStartRequest
from storage.studio_db import get_connection, get_run, list_runs
from utils.training_runs import drop_non_finite

logger = get_logger(__name__)

router = APIRouter()

FILE_PREFIX = "file-"
JOB_PREFIX = "ft"  # "job_20261021_101500_ab12cd34" -> "ftjob_20261021_101500_ab12cd34"
# OpenAI scales its own default; Studio's default learning rate is the base the multiplier scales.
BASE_LEARNING_RATE = 2e-4
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
# Set by this layer from the OpenAI fields; extra_body may not override them.
_MANAGED_FIELDS = frozenset(
    {"model_name", "local_datasets", "local_eval_datasets", "start_request_id", "project_name"}
)
_RUN_STATUS = {
    "running": "running",
    "completed": "succeeded",
    "stopped": "cancelled",
    "error": "failed",
}
_PHASE_STATUS = {
    "idle": "validating_files",
    "loading_model": "validating_files",
    "loading_dataset": "validating_files",
    "configuring": "queued",
    "training": "running",
    "training_drafter": "running",
    "finalizing": "running",
    "completed": "succeeded",
    "stopped": "cancelled",
    "error": "failed",
}


class FineTuningJobCreate(BaseModel):
    # Unknown keys are Studio training settings sent through the SDK's extra_body.
    model_config = ConfigDict(extra = "allow")

    model: str
    training_file: str
    validation_file: Optional[str] = None
    hyperparameters: Optional[dict[str, Any]] = None
    method: Optional[dict[str, Any]] = None
    suffix: Optional[str] = None
    seed: Optional[int] = None
    integrations: Optional[list[Any]] = None
    metadata: Optional[dict[str, Any]] = None


def _bad_request(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code = status_code, detail = message)


def _unix(value: Any) -> Optional[int]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo = timezone.utc)
    return int(parsed.timestamp())


# Files


def _upload_id(name: str) -> Optional[str]:
    prefix, sep, _rest = name.partition("_")
    return prefix if sep and _UPLOAD_ID.match(prefix) else None


def _file_object(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "id": f"{FILE_PREFIX}{_upload_id(path.name)}",
        "object": "file",
        "bytes": stat.st_size,
        "created_at": int(stat.st_mtime),
        "filename": local_datasets._display_uploaded_dataset_name(path),
        "purpose": "fine-tune",
        "status": "processed",
        "status_details": None,
    }


def _uploaded_files() -> list[Path]:
    """This account's dataset uploads, newest first, as the Studio dataset picker lists them."""
    items = local_datasets.list_local_datasets_response().datasets
    paths = [Path(item.path) for item in items if item.source == "upload"]
    return [path for path in paths if _upload_id(path.name)]


def _find_file(file_id: str) -> Path:
    upload_id = file_id[len(FILE_PREFIX) :] if file_id.startswith(FILE_PREFIX) else ""
    if _UPLOAD_ID.match(upload_id):
        for path in _uploaded_files():
            if _upload_id(path.name) == upload_id:
                return path
    raise _bad_request(f"No such File object: {file_id}", 404)


def _file_id_for(path: Any) -> Optional[str]:
    upload_id = _upload_id(Path(str(path)).name) if path else None
    return f"{FILE_PREFIX}{upload_id}" if upload_id else None


@router.post("/files")
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    current_subject: str = Depends(get_current_subject),
):
    if purpose != "fine-tune":
        raise _bad_request(
            f"Unsloth Studio stores files for fine-tuning only; got purpose '{purpose}'."
        )
    saved = await local_datasets.upload_dataset_response(file)
    return await asyncio.to_thread(_file_object, Path(saved.stored_path))


@router.get("/files")
async def list_files(
    after: Optional[str] = None,
    limit: int = Query(10000, ge = 1, le = 10000),
    order: Literal["asc", "desc"] = "desc",
    current_subject: str = Depends(get_current_subject),
):
    files = await asyncio.to_thread(lambda: [_file_object(path) for path in _uploaded_files()])
    return _paged(files if order == "desc" else files[::-1], after, limit)


@router.get("/files/{file_id}")
async def retrieve_file(file_id: str, current_subject: str = Depends(get_current_subject)):
    return await asyncio.to_thread(lambda: _file_object(_find_file(file_id)))


# Jobs


def _job_id(run_id: str) -> str:
    return f"{JOB_PREFIX}{run_id}"


def _run_id(job_id: str) -> str:
    return job_id[len(JOB_PREFIX) :] if job_id.startswith(f"{JOB_PREFIX}job_") else job_id


def _hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    try:
        multiplier = round(float(config.get("learning_rate")) / BASE_LEARNING_RATE, 6)
    except (TypeError, ValueError):
        multiplier = "auto"
    return {
        "n_epochs": config.get("num_epochs") or "auto",
        "batch_size": config.get("batch_size") or "auto",
        "learning_rate_multiplier": multiplier,
    }


def _job_object(
    run_id: str,
    *,
    status: str,
    config: dict[str, Any],
    created_at: Optional[int],
    finished_at: Optional[int] = None,
    error: Optional[str] = None,
    fine_tuned_model: Optional[str] = None,
    trained_tokens: Optional[int] = None,
) -> dict[str, Any]:
    hyperparameters = _hyperparameters(config)
    datasets = config.get("local_datasets") or []
    eval_datasets = config.get("local_eval_datasets") or []
    return {
        "id": _job_id(run_id),
        "object": "fine_tuning.job",
        "created_at": created_at or int(datetime.now(timezone.utc).timestamp()),
        "error": {"code": "training_failed", "message": error, "param": None} if error else None,
        "fine_tuned_model": fine_tuned_model,
        "finished_at": finished_at,
        "hyperparameters": hyperparameters,
        "model": config.get("model_name"),
        "organization_id": "unsloth-studio",
        "result_files": [],
        "seed": config.get("random_seed"),
        "status": status,
        "trained_tokens": trained_tokens,
        "training_file": _file_id_for(datasets[0]) if datasets else config.get("hf_dataset"),
        "validation_file": _file_id_for(eval_datasets[0]) if eval_datasets else None,
        "method": {"type": "supervised", "supervised": {"hyperparameters": hyperparameters}},
        "suffix": config.get("project_name"),
        "metadata": None,
    }


def _run_config(run: dict[str, Any]) -> dict[str, Any]:
    try:
        # An older install may have stored Infinity / NaN, which the JSON response cannot carry.
        config = drop_non_finite(json.loads(run.get("config_json") or "{}"))
    except (TypeError, ValueError):
        config = {}
    return config if isinstance(config, dict) else {}


def _trained_tokens(run_id: str) -> Optional[int]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(num_tokens) FROM training_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] is not None else None


def _job_from_run(run: dict[str, Any]) -> dict[str, Any]:
    status = _RUN_STATUS.get(run.get("status"), "running")
    if status != "running":
        _CANCELLING.discard(run["id"])
    elif run["id"] in _CANCELLING:
        status = "cancelled"
    output_dir = run.get("output_dir")
    return _job_object(
        run["id"],
        status = status,
        config = _run_config(run),
        created_at = _unix(run.get("started_at")),
        finished_at = _unix(run.get("ended_at")) if status != "running" else None,
        error = (run.get("error_message") or "Training failed") if status == "failed" else None,
        # The run folder: the name Studio's chat, export and checkpoint lists show for this model.
        fine_tuned_model = Path(output_dir).name if status == "succeeded" and output_dir else None,
        trained_tokens = _trained_tokens(run["id"]),
    )


# Jobs started here, until their history row exists: a retrieve right after create keeps the job's fields.
_STARTED: dict[str, tuple[dict[str, Any], int]] = {}
_STARTED_LIMIT = 16
# Cancelled here while the worker finishes its step: the run row reads "running" until it stops.
_CANCELLING: set[str] = set()


def _remember_start(run_id: str, config: dict[str, Any], created_at: int) -> None:
    _STARTED[run_id] = (config, created_at)
    while len(_STARTED) > _STARTED_LIMIT:
        _STARTED.pop(next(iter(_STARTED)))


def _for_caller(job: dict[str, Any], via_api_key: bool) -> dict[str, Any]:
    """An API key never sees host paths: a local model becomes its handle, an error loses its paths."""
    if host_paths_visible(via_api_key):
        return job
    model = redact_host_paths({"model_name": job["model"]}, via_api_key = via_api_key)
    job = {**job, "model": model["model_name"]}
    if job["error"]:
        job["error"] = {**job["error"], "message": redact_paths_in_text(job["error"]["message"])}
    return job


def _status_field(status: Any, name: str) -> Any:
    # The status route hands back a model, or its redacted dump for an API key.
    return status.get(name) if isinstance(status, dict) else getattr(status, name, None)


async def _current_status(current_subject: str, via_api_key: bool) -> Any:
    from routes import training as training_routes
    try:
        return await training_routes.get_training_status(
            current_subject = current_subject, via_api_key = via_api_key
        )
    except HTTPException:
        return None


async def _job(run_id: str, current_subject: str, via_api_key: bool) -> dict[str, Any]:
    run = await asyncio.to_thread(get_run, run_id)
    if run is not None:
        return _for_caller(await asyncio.to_thread(_job_from_run, run), via_api_key)
    # A started job has no history row until its worker is up; the live status still knows it.
    status = await _current_status(current_subject, via_api_key)
    if status is not None and _status_field(status, "job_id") == run_id:
        phase = _status_field(status, "phase")
        failed = _status_field(status, "start_request_state") == "rejected" or phase == "error"
        message = _status_field(status, "error") or _status_field(status, "message")
        config, created_at = _STARTED.get(run_id, ({}, None))
        state = _PHASE_STATUS.get(phase, "queued")
        if failed:
            state = "failed"
        elif run_id in _CANCELLING:
            state = "cancelled"
        return _for_caller(
            _job_object(
                run_id,
                status = state,
                config = config,
                created_at = created_at,
                error = (message or "Training failed") if failed else None,
            ),
            via_api_key,
        )
    raise _bad_request(f"No such fine-tuning job: {_job_id(run_id)}", 404)


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _bad_request(f"{name} must be an object.")
    return value


def _training_config(body: FineTuningJobCreate) -> dict[str, Any]:
    method = _object(body.method, "method")
    method_type = method.get("type", "supervised")
    if method_type != "supervised":
        raise _bad_request(
            f"Fine-tuning method '{method_type}' is not supported yet; use 'supervised'."
        )
    supervised = _object(method.get("supervised"), "method.supervised")
    hyperparameters = dict(_object(body.hyperparameters, "hyperparameters"))
    hyperparameters.update(
        _object(supervised.get("hyperparameters"), "method.supervised.hyperparameters")
    )

    config: dict[str, Any] = {"training_type": "LoRA/QLoRA", "format_type": "auto"}
    extra = body.model_extra or {}
    unknown = sorted(
        key
        for key in extra
        if key not in TrainingStartRequest.model_fields or key in _MANAGED_FIELDS
    )
    if unknown:
        raise _bad_request(f"Unknown parameter(s): {', '.join(unknown)}")
    config.update(extra)

    config["model_name"] = body.model
    config["local_datasets"] = [str(_find_file(body.training_file))]
    if body.validation_file:
        config["local_eval_datasets"] = [str(_find_file(body.validation_file))]
        # Studio evaluates only when eval_steps > 0; OpenAI always validates a given validation_file.
        config.setdefault("eval_steps", 0.1)
    if body.suffix:
        config["project_name"] = body.suffix
    for integration in body.integrations or []:
        integration = _object(integration, "integrations[]")
        if integration.get("type") != "wandb":
            raise _bad_request(
                f"Integration '{integration.get('type')}' is not supported; use 'wandb'."
            )
        wandb = _object(integration.get("wandb"), "integrations[].wandb")
        # The W&B key is the server's WANDB_API_KEY, or wandb_token through extra_body.
        config.setdefault("enable_wandb", True)
        if wandb.get("project"):
            config.setdefault("wandb_project", wandb["project"])
    if body.seed is not None:
        config["random_seed"] = body.seed
    try:
        n_epochs = hyperparameters.get("n_epochs", "auto")
        if n_epochs != "auto":
            config["num_epochs"] = int(n_epochs)
        batch_size = hyperparameters.get("batch_size", "auto")
        if batch_size != "auto":
            config["batch_size"] = int(batch_size)
        multiplier = hyperparameters.get("learning_rate_multiplier", "auto")
        if multiplier != "auto":
            config["learning_rate"] = f"{BASE_LEARNING_RATE * float(multiplier):g}"
    except (TypeError, ValueError):
        raise _bad_request(
            "n_epochs and batch_size take an integer or 'auto', learning_rate_multiplier a number or 'auto'."
        ) from None
    return config


@router.post("/fine_tuning/jobs")
async def create_fine_tuning_job(
    body: FineTuningJobCreate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    from pydantic import ValidationError

    from routes import training as training_routes

    config = await asyncio.to_thread(_training_config, body)
    try:
        request = TrainingStartRequest.model_validate(config)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()))
        raise _bad_request(f"Invalid parameter {field}: {first.get('msg')}") from None
    started = await training_routes.start_training(
        request, current_subject = current_subject, via_api_key = via_api_key
    )
    if started.status == "error" or not started.job_id:
        message = started.message or started.error or "Training did not start"
        raise _bad_request(
            message if host_paths_visible(via_api_key) else redact_paths_in_text(message), 409
        )
    created_at = int(datetime.now(timezone.utc).timestamp())
    _remember_start(started.job_id, request.model_dump(), created_at)
    return _for_caller(
        _job_object(
            started.job_id,
            status = "validating_files",
            config = request.model_dump(),
            created_at = created_at,
        ),
        via_api_key,
    )


def _paged(rows: list[dict[str, Any]], after: Optional[str], limit: int) -> dict[str, Any]:
    if after:
        ids = [row["id"] for row in rows]
        rows = rows[ids.index(after) + 1 :] if after in ids else []
    return {"object": "list", "data": rows[:limit], "has_more": len(rows) > limit}


def _runs_after(after: Optional[str], count: int) -> list[dict[str, Any]]:
    """Up to ``count`` runs older than ``after`` (newest first), paging through the whole history."""
    runs: list[dict[str, Any]] = []
    found, offset = after is None, 0
    while len(runs) < count:
        batch = list_runs(200, offset)["runs"]
        if not batch:
            break
        for run in batch:
            if found:
                runs.append(run)
            found = found or run["id"] == after
        offset += len(batch)
    return runs[:count]


@router.get("/fine_tuning/jobs")
async def list_fine_tuning_jobs(
    after: Optional[str] = None,
    limit: int = Query(20, ge = 1, le = 100),
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    after_run = _run_id(after) if after else None
    runs = await asyncio.to_thread(_runs_after, after_run, limit + 1)
    if after_run and not runs:
        # The live job has no history row yet and is listed first, so the history follows it.
        status = await _current_status(current_subject, via_api_key)
        if status is not None and _status_field(status, "job_id") == after_run:
            runs = await asyncio.to_thread(_runs_after, None, limit + 1)
    jobs = [_for_caller(await asyncio.to_thread(_job_from_run, run), via_api_key) for run in runs]
    if after is None:
        status = await _current_status(current_subject, via_api_key)
        live_id = _status_field(status, "job_id") if status is not None else None
        live_phase = _status_field(status, "phase") if status is not None else None
        if live_id and live_phase != "idle" and all(run["id"] != live_id for run in runs):
            jobs.insert(0, await _job(live_id, current_subject, via_api_key))
    return {"object": "list", "data": jobs[:limit], "has_more": len(jobs) > limit}


@router.get("/fine_tuning/jobs/{job_id}")
async def retrieve_fine_tuning_job(
    job_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    return await _job(_run_id(job_id), current_subject, via_api_key)


def _events(run: dict[str, Any], via_api_key: bool) -> list[dict[str, Any]]:
    """Newest first: one metrics event per logged step, framed by the start and the outcome."""
    run_id = run["id"]
    started = _unix(run.get("started_at")) or 0
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT step, loss, learning_rate, grad_norm, eval_loss, epoch, elapsed_seconds "
            "FROM training_metrics WHERE run_id = ? AND step > 0 ORDER BY step",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    total = run.get("total_steps")

    def event(
        key: str,
        created_at: int,
        message: str,
        level: str = "info",
        data: Any = None,
    ):
        return {
            "id": f"ftevent_{run_id}_{key}",
            "object": "fine_tuning.job.event",
            "created_at": created_at,
            "level": level,
            "message": message,
            "data": data,
            "type": "metrics" if data else "message",
        }

    events = [event("start", started, "Fine-tuning job started")]
    # An fp16 overflow logs an inf grad norm, which the JSON response cannot carry.
    for step, loss, lr, grad_norm, eval_loss, epoch, elapsed in drop_non_finite(
        [tuple(row) for row in rows]
    ):
        data = {
            "step": step,
            "total_steps": total,
            "train_loss": loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "valid_loss": eval_loss,
            "epoch": epoch,
        }
        loss_text = f"training loss={loss:.4f}" if loss is not None else "no loss logged"
        events.append(
            event(
                f"step{step}",
                started + int(elapsed or 0),
                f"Step {step}/{total or '?'}: {loss_text}",
                data = data,
            )
        )
    status = _RUN_STATUS.get(run.get("status"), "running")
    ended = _unix(run.get("ended_at")) or started
    if status == "succeeded":
        events.append(event("end", ended, "The job has successfully completed"))
    elif status == "cancelled":
        events.append(event("end", ended, "The job was cancelled"))
    elif status == "failed":
        message = run.get("error_message") or "Training failed"
        if not host_paths_visible(via_api_key):
            message = redact_paths_in_text(message)
        events.append(event("end", ended, message, level = "error"))
    events.reverse()
    return events


@router.get("/fine_tuning/jobs/{job_id}/events")
async def list_fine_tuning_events(
    job_id: str,
    after: Optional[str] = None,
    limit: int = Query(20, ge = 1, le = 1000),
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    run_id = _run_id(job_id)
    run = await asyncio.to_thread(get_run, run_id)
    if run is None:
        # Known but not started yet: no events, rather than a 404 a polling client would give up on.
        await _job(run_id, current_subject, via_api_key)
        return {"object": "list", "data": [], "has_more": False}
    return _paged(await asyncio.to_thread(_events, run, via_api_key), after, limit)


def _checkpoints(run: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = run.get("output_dir")
    if not output_dir or not Path(output_dir).is_dir():
        return []
    run_id = run["id"]
    started = _unix(run.get("started_at")) or 0
    conn = get_connection()
    try:
        metrics = {
            step: (loss, eval_loss)
            for step, loss, eval_loss in drop_non_finite(
                [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT step, loss, eval_loss FROM training_metrics WHERE run_id = ?",
                        (run_id,),
                    ).fetchall()
                ]
            )
        }
    finally:
        conn.close()
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        step_text = path.name.rpartition("-")[2]
        if not path.is_dir() or not step_text.isdigit():
            continue
        step = int(step_text)
        loss, eval_loss = metrics.get(step, (None, None))
        checkpoints.append(
            {
                "id": f"ftckpt_{run_id}_{step}",
                "object": "fine_tuning.job.checkpoint",
                "created_at": int(path.stat().st_mtime) or started,
                # The "<run folder>/<checkpoint>" name Studio loads and exports a checkpoint by.
                "fine_tuned_model_checkpoint": f"{Path(output_dir).name}/{path.name}",
                "fine_tuning_job_id": _job_id(run_id),
                "metrics": {"step": step, "train_loss": loss, "valid_loss": eval_loss},
                "step_number": step,
            }
        )
    checkpoints.sort(key = lambda checkpoint: checkpoint["step_number"], reverse = True)
    return checkpoints


@router.get("/fine_tuning/jobs/{job_id}/checkpoints")
async def list_fine_tuning_checkpoints(
    job_id: str,
    after: Optional[str] = None,
    limit: int = Query(10, ge = 1, le = 100),
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    run_id = _run_id(job_id)
    run = await asyncio.to_thread(get_run, run_id)
    if run is None:
        await _job(run_id, current_subject, via_api_key)
        return {"object": "list", "data": [], "has_more": False}
    return _paged(await asyncio.to_thread(_checkpoints, run), after, limit)


@router.post("/fine_tuning/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(
    job_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    from routes import training as training_routes

    run_id = _run_id(job_id)
    job = await _job(run_id, current_subject, via_api_key)
    if job["status"] in ("succeeded", "failed", "cancelled"):
        raise _bad_request(f"Job {job['id']} has already {job['status']}.")
    stopped = await training_routes.stop_training(
        training_routes.TrainingStopRequest(save = True, expected_job_id = run_id),
        current_subject = current_subject,
    )
    if stopped.status == "idle":
        raise _bad_request(f"Job {job['id']} is not running.", 409)
    # The worker stops at its next safe step; the job reads "cancelled" from then on.
    _CANCELLING.add(run_id)
    return {**job, "status": "cancelled"}


@router.post("/fine_tuning/jobs/{job_id}/pause")
@router.post("/fine_tuning/jobs/{job_id}/resume")
async def pause_or_resume_fine_tuning_job(
    job_id: str, current_subject: str = Depends(get_current_subject)
):
    raise _bad_request("Pausing and resuming fine-tuning jobs is not supported; use cancel.")
