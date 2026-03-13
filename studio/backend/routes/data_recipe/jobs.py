# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Job lifecycle endpoints for data recipe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from core.data_recipe.jobs import get_job_manager
from models.data_recipe import JobCreateResponse, RecipePayload

router = APIRouter()


def _normalize_run_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(
            status_code = 400, detail = "invalid run_name: must be a string"
        )
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:120]


@router.post("/jobs", response_class = JSONResponse, response_model = JobCreateResponse)
def create_job(payload: RecipePayload):
    recipe = payload.recipe
    if not recipe.get("columns"):
        raise HTTPException(status_code = 400, detail = "Recipe must include columns.")

    run: dict[str, Any] = payload.run or {}
    run.pop("artifact_path", None)
    run.pop("dataset_name", None)
    execution_type = str(run.get("execution_type") or "full").strip().lower()
    if execution_type not in {"preview", "full"}:
        raise HTTPException(
            status_code = 400,
            detail = "invalid execution_type: must be 'preview' or 'full'",
        )
    run["execution_type"] = execution_type
    run["run_name"] = _normalize_run_name(run.get("run_name"))
    run_config_raw = run.get("run_config")
    if run_config_raw is not None:
        try:
            from data_designer.config.run_config import RunConfig

            RunConfig.model_validate(run_config_raw)
        except (ImportError, ValidationError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code = 400, detail = f"invalid run_config: {exc}"
            ) from exc

    mgr = get_job_manager()
    try:
        job_id = mgr.start(recipe = recipe, run = run)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    return {"job_id": job_id}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mgr = get_job_manager()
    state = mgr.get_status(job_id)
    if state is None:
        raise HTTPException(status_code = 404, detail = "job not found")
    return state


@router.get("/jobs/current")
def current_job():
    mgr = get_job_manager()
    state = mgr.get_current_status()
    if state is None:
        raise HTTPException(status_code = 404, detail = "no job")
    return state


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    mgr = get_job_manager()
    ok = mgr.cancel(job_id)
    if not ok:
        raise HTTPException(status_code = 404, detail = "job not found")
    return mgr.get_status(job_id)


@router.get("/jobs/{job_id}/analysis")
def job_analysis(job_id: str):
    mgr = get_job_manager()
    analysis = mgr.get_analysis(job_id)
    if analysis is None:
        raise HTTPException(status_code = 404, detail = "analysis not ready")
    return analysis


@router.get("/jobs/{job_id}/dataset")
def job_dataset(
    job_id: str,
    limit: int = Query(default = 20, ge = 1, le = 500),
    offset: int = Query(default = 0, ge = 0),
):
    mgr = get_job_manager()
    result = mgr.get_dataset(job_id, limit = limit, offset = offset)
    if result is None:
        raise HTTPException(status_code = 404, detail = "dataset not ready")
    if "error" in result:
        raise HTTPException(status_code = 422, detail = result["error"])
    return {
        "dataset": result["dataset"],
        "total": result["total"],
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    mgr = get_job_manager()
    last_id = request.headers.get("last-event-id")
    after_seq: int | None = None
    if last_id:
        try:
            after_seq = int(str(last_id).strip())
        except (TypeError, ValueError):
            after_seq = None

    after_q = request.query_params.get("after")
    if after_q:
        try:
            after_seq = int(str(after_q).strip())
        except (TypeError, ValueError):
            pass

    sub = mgr.subscribe(job_id, after_seq = after_seq)
    if sub is None:
        raise HTTPException(status_code = 404, detail = "job not found")

    async def gen():
        try:
            for event in sub.replay:
                yield sub.format_sse(event)

            while True:
                if await request.is_disconnected():
                    break
                event = await sub.next_event(timeout_sec = 1.0)
                if event is None:
                    continue
                yield sub.format_sse(event)
        finally:
            mgr.unsubscribe(sub)

    return StreamingResponse(gen(), media_type = "text/event-stream")
