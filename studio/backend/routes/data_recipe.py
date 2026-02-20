"""
Data Recipe routes (DataDesigner runner).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

# same thing as other files do
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.data_recipe.jobs import get_job_manager
from core.data_recipe.service import preview_recipe, validate_recipe
from models.data_recipe import JobCreateResponse, PreviewResponse, RecipePayload, ValidateError, ValidateResponse

router = APIRouter()


@router.post("/validate", response_model=ValidateResponse)
def validate(payload: RecipePayload) -> ValidateResponse:
    recipe = payload.recipe
    if not recipe.get("columns"):
        return ValidateResponse(
            valid=False,
            errors=[ValidateError(message="Recipe must include columns.")],
        )

    try:
        validate_recipe(recipe)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or "Validation failed."
        return ValidateResponse(
            valid=False,
            errors=[ValidateError(message=detail)],
            raw_detail=detail,
        )

    return ValidateResponse(valid=True)


@router.post("/preview", response_model=PreviewResponse)
def preview(payload: RecipePayload) -> PreviewResponse:
    recipe = payload.recipe
    if not recipe.get("columns"):
        raise HTTPException(status_code=400, detail="Recipe must include columns.")

    run = payload.run or {}
    num_records = int(run.get("rows") or 5)

    try:
        dataset, artifacts, analysis = preview_recipe(recipe, num_records)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PreviewResponse(dataset=dataset, processor_artifacts=artifacts, analysis=analysis)


@router.post("/jobs", response_class=JSONResponse, response_model=JobCreateResponse)
def create_job(payload: RecipePayload):
    recipe = payload.recipe
    if not recipe.get("columns"):
        raise HTTPException(status_code=400, detail="Recipe must include columns.")

    run: dict[str, Any] = payload.run or {}
    run_config_raw = run.get("run_config")
    if run_config_raw is not None:
        try:
            from data_designer.config.run_config import RunConfig

            RunConfig.model_validate(run_config_raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid run_config: {exc}") from exc

    mgr = get_job_manager()
    try:
        job_id = mgr.start(recipe=recipe, run=run)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"job_id": job_id}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mgr = get_job_manager()
    state = mgr.get_status(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state


@router.get("/jobs/current")
def current_job():
    mgr = get_job_manager()
    state = mgr.get_current_status()
    if state is None:
        raise HTTPException(status_code=404, detail="no job")
    return state


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    mgr = get_job_manager()
    ok = mgr.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return mgr.get_status(job_id)


@router.get("/jobs/{job_id}/analysis")
def job_analysis(job_id: str):
    mgr = get_job_manager()
    analysis = mgr.get_analysis(job_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not ready")
    return analysis


@router.get("/jobs/{job_id}/dataset")
def job_dataset(job_id: str, limit: int = Query(default=20, ge=1, le=500)):
    mgr = get_job_manager()
    dataset = mgr.get_dataset(job_id, limit=limit)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not ready")
    return {"dataset": dataset}


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    mgr = get_job_manager()
    last_id = request.headers.get("last-event-id")
    after_seq: int | None = None
    if last_id:
        try:
            after_seq = int(str(last_id).strip())
        except Exception:
            after_seq = None

    # EventSource can't set custom headers on first connect after a full page refresh,
    # so allow resume via query param too: /events?after=<seq>
    after_q = request.query_params.get("after")
    if after_q:
        try:
            after_seq = int(str(after_q).strip())
        except Exception:
            pass

    sub = mgr.subscribe(job_id, after_seq=after_seq)
    if sub is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        try:
            for event in sub.replay:
                yield sub.format_sse(event)

            while True:
                if await request.is_disconnected():
                    break
                event = await sub.next_event(timeout_sec=1.0)
                if event is None:
                    continue
                yield sub.format_sse(event)
        finally:
            mgr.unsubscribe(sub)

    return StreamingResponse(gen(), media_type="text/event-stream")
