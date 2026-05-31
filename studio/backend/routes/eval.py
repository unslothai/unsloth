# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loggers import get_logger

from auth.authentication import get_current_subject
from eval.dataset import DatasetRef, sample_column_values
from eval.jobs import EvalBusyError, EvalJobManager, build_eval_run_fn
from eval.json_score.schema import (ArrayNode, LeafNode, Node, ObjectNode,
                                     normalize_schema)
from eval.metrics.registry import list_metrics
from models import (EvalDatasetRef, EvalProgress, EvalResultRow, EvalRunDetail,
                    EvalRunSummary, EvalStartRequest, MetricInfo)
from pydantic import BaseModel
from storage import studio_db

logger = get_logger(__name__)
router = APIRouter()

_MANAGER: EvalJobManager | None = None


def get_eval_manager() -> EvalJobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = EvalJobManager(run_fn=build_eval_run_fn())
    return _MANAGER


@router.get("/metrics")
async def get_metrics(current_subject: str = Depends(get_current_subject)):
    return {"metrics": [MetricInfo(**m).model_dump() for m in list_metrics()]}


def _flatten_comparators(node: Node, prefix: str = "") -> list[dict]:
    """Walk a normalized schema into a flat [{path, comparator}] list, so the UI
    can show which comparator each field resolved to."""
    if isinstance(node, ObjectNode):
        out: list[dict] = []
        for key, child in node.fields.items():
            path = key if not prefix else f"{prefix}.{key}"
            out.extend(_flatten_comparators(child, path))
        return out
    if isinstance(node, ArrayNode):
        return _flatten_comparators(node.item, f"{prefix}[]")
    if isinstance(node, LeafNode):
        return [{"path": prefix or "(value)", "comparator": node.comparator}]
    return []


class InferSchemaRequest(BaseModel):
    dataset: EvalDatasetRef
    output_column: str
    samples: int = 10


@router.post("/infer-schema")
async def infer_schema(payload: InferSchemaRequest,
                       current_subject: str = Depends(get_current_subject)):
    """Infer a JSON Schema from sample values of the dataset's output column,
    so the user doesn't have to hand-write one for the json_document metric."""
    ref = DatasetRef(
        is_local=payload.dataset.is_local, path=payload.dataset.path,
        name=payload.dataset.name, split=payload.dataset.split,
        subset=payload.dataset.subset,
    )
    try:
        values = sample_column_values(
            ref, column=payload.output_column, limit=max(1, min(payload.samples, 50)),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    parsed: list = []
    parse_errors = 0
    for v in values:
        if isinstance(v, (dict, list)):
            parsed.append(v)
            continue
        if not isinstance(v, str):
            parse_errors += 1
            continue
        try:
            parsed.append(json.loads(v))
        except (ValueError, TypeError):
            try:
                from json_repair import repair_json
                r = repair_json(v, return_objects=True)
                if isinstance(r, (dict, list)) and r:
                    parsed.append(r)
                else:
                    parse_errors += 1
            except Exception:
                parse_errors += 1

    if not parsed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No parseable JSON in column {payload.output_column!r} "
                f"(checked {len(values)} sample(s))."
            ),
        )

    try:
        from genson import SchemaBuilder
    except ImportError:
        raise HTTPException(status_code=500, detail="genson is not installed")

    builder = SchemaBuilder()
    for sample in parsed:
        builder.add_object(sample)
    schema = builder.to_schema()
    return {
        "schema": schema,
        "samples_used": len(parsed),
        "parse_errors": parse_errors,
    }


@router.post("/schema-preview")
async def schema_preview(payload: dict = Body(default={}),
                         current_subject: str = Depends(get_current_subject)):
    """Preview the comparator each field resolves to for a given schema —
    accepts our field→comparator mapping OR a standard JSON Schema."""
    schema = payload.get("schema")
    if schema is None or schema == "":
        return {"fields": []}
    try:
        node = normalize_schema(schema)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"fields": _flatten_comparators(node)}


@router.post("/start")
async def start_eval(payload: EvalStartRequest,
                     current_subject: str = Depends(get_current_subject)):
    mgr = get_eval_manager()
    try:
        run_id = mgr.start(payload)
    except EvalBusyError:
        raise HTTPException(status_code=409, detail="An eval is already running.")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"run_id": run_id}


@router.post("/cancel/{run_id}")
async def cancel_eval(run_id: str, current_subject: str = Depends(get_current_subject)):
    ok = get_eval_manager().cancel(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No active eval with that id.")
    return {"cancelled": True}


@router.get("/runs")
async def list_runs(limit: int = 50, offset: int = 0,
                    current_subject: str = Depends(get_current_subject)):
    data = studio_db.list_eval_runs(limit=limit, offset=offset)
    return {"runs": [EvalRunSummary(**r).model_dump() for r in data["runs"]],
            "total": data["total"]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, limit: int = 100, offset: int = 0,
                  current_subject: str = Depends(get_current_subject)):
    run = studio_db.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found.")
    res = studio_db.get_eval_results(run_id, limit=limit, offset=offset)
    rows = []
    for r in res["results"]:
        rows.append(EvalResultRow(
            idx=r["idx"], input_text=r["input_text"],
            prediction_text=r["prediction_text"], reference_text=r["reference_text"],
            score=r["score"],
            breakdown=json.loads(r["breakdown_json"]) if r["breakdown_json"] else None,
            error=r["error"],
        ))
    return EvalRunDetail(run=EvalRunSummary(**run), results=rows,
                         total_results=res["total"]).model_dump()


@router.get("/progress/{run_id}")
async def stream_progress(run_id: str, request: Request,
                          current_subject: str = Depends(get_current_subject)):
    mgr = get_eval_manager()

    async def gen():
        yield "retry: 3000\n\n"
        last_done = -1
        log_cursor = 0

        def drain_logs():
            nonlocal log_cursor
            data = mgr.get_logs(run_id, since=log_cursor)
            if data["entries"]:
                log_cursor = data["next"]
                return f"event: log\ndata: {json.dumps({'entries': data['entries']})}\n\n"
            return None

        while True:
            if await request.is_disconnected():
                break
            chunk = drain_logs()
            if chunk:
                yield chunk
            prog = mgr.get(run_id)
            if prog is None:
                break
            if prog["done"] != last_done or prog["status"] != "running":
                payload = EvalProgress(**{
                    "run_id": run_id, "status": prog["status"],
                    "done": prog.get("done", 0), "total": prog.get("total", 0),
                    "avg_score": prog.get("avg_score", 0.0),
                    "last_result": prog.get("last_result"),
                }).model_dump_json()
                event = "complete" if prog["status"] != "running" else "progress"
                yield f"id: {prog.get('done', 0)}\nevent: {event}\ndata: {payload}\n\n"
                last_done = prog["done"]
                if prog["status"] != "running":
                    final = drain_logs()      # flush any trailing logs before closing
                    if final:
                        yield final
                    break
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")
