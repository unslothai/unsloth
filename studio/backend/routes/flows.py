# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows: saved node graphs, their runs (with replayable SSE events), player entry points and saves, sharing and
importing, and flow assets. Everything is per account: flows live in the account's studio.db, live runs are only
visible to the account that started them."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.flows import assets as asset_store
from core.flows import caller as flow_caller
from core.flows import compiler, package
from core.flows.examples import build_examples as examples
from core.flows.compiler import CompileError
from core.flows.engine import get_engine
from core.flows.registry import FlowNodeError
from core.flows.schema import MAX_ASSET_BYTES, MAX_PACKAGE_BYTES
from core.flows.specs import catalog
from loggers import get_logger
from storage import flows_db
from utils.account_context import current_account

logger = get_logger(__name__)

router = APIRouter(dependencies = [Depends(get_current_subject)])

KEEPALIVE_S = 15.0
_SLOT = r"^[A-Za-z0-9_-]{1,64}$"
_TIME = Field(default = None, ge = 0, le = 2**53)


class FlowBody(BaseModel):
    doc: dict[str, Any]
    baseUpdatedAt: Optional[int] = _TIME


class ValidateBody(BaseModel):
    doc: Any


class RunBody(BaseModel):
    params: dict[str, Any] = Field(default_factory = dict)
    target_nodes: list[str] = Field(default_factory = list, max_length = 2000)
    force_nodes: list[str] = Field(default_factory = list, max_length = 2000)
    bake: bool = False
    # Recompute nodes the flow ships pre-generated (a story's "New story" re-roll).
    ignore_baked: bool = False
    doc: Optional[dict[str, Any]] = None


class PlanBody(BaseModel):
    params: dict[str, Any] = Field(default_factory = dict)
    doc: Optional[dict[str, Any]] = None


class InvokeBody(BaseModel):
    entry: str = Field(min_length = 1, max_length = 64)
    inputs: dict[str, Any] = Field(default_factory = dict)
    params: dict[str, Any] = Field(default_factory = dict)
    session_id: Optional[str] = Field(default = None, max_length = 128)
    rev: Optional[int] = None


class SessionBody(BaseModel):
    state: Any
    label: Optional[str] = Field(default = None, max_length = 200)


def _doc_or_400(raw: Any) -> dict:
    doc, errors = compiler.parse_document(raw)
    if doc is None:
        raise HTTPException(status_code = 400, detail = {"message": "Invalid flow", "errors": errors})
    return doc


def _compile_error(exc: CompileError) -> HTTPException:
    return HTTPException(status_code = 400, detail = {"message": str(exc), "errors": exc.issues})


def _flow_or_404(flow_id: str) -> dict:
    record = flows_db.get_flow(flow_id)
    if record is None:
        raise HTTPException(status_code = 404, detail = "Flow not found")
    return record


def _save(doc: dict, base: Optional[int]) -> dict:
    try:
        return flows_db.save_flow(doc, base)
    except flows_db.FlowDeleted:
        raise HTTPException(status_code = 410, detail = "This flow was deleted")
    except flows_db.FlowConflict:
        raise HTTPException(
            status_code = 409,
            detail = "This flow was changed in another window. Reload it to keep editing.",
        )


@router.get("/nodes")
def node_catalog() -> dict:
    return catalog()


@router.post("/validate")
def validate_flow(body: ValidateBody) -> dict:
    return compiler.validate(body.doc)


def _example_summary(doc: dict) -> dict:
    meta = doc.get("meta") or {}
    cover = meta.get("cover")
    return {
        "id": doc["id"],
        "name": doc["name"],
        "kind": doc["kind"],
        "description": doc.get("description") or "",
        "createdAt": 0,
        "updatedAt": 0,
        "cover": cover.get("url")
        if isinstance(cover, dict)
        else cover
        if isinstance(cover, str)
        else None,
        "nodeCount": len(doc.get("nodes") or []),
        "example": True,
        "tags": meta.get("tags") or [],
    }


@router.get("/examples")
def list_examples() -> dict:
    return {"examples": [_example_summary(examples.example(i)) for i in examples.example_ids()]}


def _import_doc(data: bytes, name: Optional[str] = None) -> tuple[dict, dict]:
    try:
        doc, files, report = package.read_package(data)
        package.store_assets(files)
    except package.PackageError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    package.record_lineage(doc)
    doc["id"] = uuid.uuid4().hex
    if name:
        doc["name"] = name[:200]
    record = _save(doc, None)
    return record, report


@router.post("/examples/{example_id}/copy")
def copy_example(example_id: str) -> dict:
    doc = examples.example(example_id)
    if doc is None:
        raise HTTPException(status_code = 404, detail = "Example not found")
    record, _report = _import_doc(json.dumps(doc).encode())
    return record


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code = 413, detail = "The upload is too large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code = 413, detail = "The upload is too large")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/import")
async def import_flow(request: Request, name: Optional[str] = None) -> dict:
    data = await _read_body(request, MAX_PACKAGE_BYTES)
    if not data:
        raise HTTPException(status_code = 400, detail = "The upload is empty")
    record, report = await asyncio.to_thread(_import_doc, data, name)
    report["missing_models"] = await _missing_models(request, record["doc"])
    return {"flow": record, "report": report}


@router.post("/assets")
async def upload_asset(
    request: Request,
    name: Optional[str] = None,
    mime: Optional[str] = None,
) -> dict:
    data = await _read_body(request, MAX_ASSET_BYTES)
    if asset_store.kind_of(asset_store.sniff_mime(data) or "") not in ("image", "audio", "video"):
        raise HTTPException(status_code = 415, detail = "Flows take images, audio and video")
    try:
        digest, sniffed = await asyncio.to_thread(asset_store.store, data, mime)
    except ValueError as exc:
        raise HTTPException(status_code = 415, detail = str(exc))
    return {"media": asset_store.media_value(digest, sniffed)}


@router.get("/assets/{ref}")
def get_asset(ref: str) -> Response:
    digest = asset_store.parse_ref(ref)
    got = asset_store.read(digest) if digest else None
    if got is None:
        raise HTTPException(status_code = 404, detail = "Asset not found")
    data, mime = got
    return Response(
        content = data,
        media_type = mime,
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


def _stored_snapshot(run_id: str) -> Optional[dict]:
    run = flows_db.get_run(run_id)
    if run is None:
        return None
    return {
        "run": {
            k: run[k]
            for k in ("id", "flow_id", "status", "error", "outputs", "started_at", "finished_at")
        },
        "nodes": flows_db.node_results(run_id),
        "last_seq": run["last_seq"],
    }


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    rs = get_engine().live(run_id)
    if rs is not None:
        snapshot = rs.snapshot()
        stored = flows_db.get_run(run_id)
        if stored:
            snapshot["run"].update(
                {"started_at": stored["started_at"], "finished_at": stored["finished_at"]}
            )
        return snapshot
    snapshot = _stored_snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code = 404, detail = "Run not found")
    return snapshot


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    status = await get_engine().cancel(run_id)
    if status is None:
        raise HTTPException(status_code = 404, detail = "Run not found")
    return {"status": status}


def _sse(event: dict) -> str:
    return f"id: {event['seq']}\ndata: {json.dumps(event, default = str)}\n\n"


def _after(request: Request) -> Optional[int]:
    for raw in (request.query_params.get("after"), request.headers.get("last-event-id")):
        if raw is not None and str(raw).strip().lstrip("-").isdigit():
            return int(str(raw).strip())
    return None


# POST too: quick tunnels hold a streamed GET until it closes.
REPLAY_PAGE = 5000


@router.post("/runs/{run_id}/events")
@router.get("/runs/{run_id}/events", include_in_schema = False)
async def run_events(request: Request, run_id: str) -> StreamingResponse:
    engine = get_engine()
    rs = engine.live(run_id)
    stored = None if rs is not None else _stored_snapshot(run_id)
    if rs is None and stored is None:
        raise HTTPException(status_code = 404, detail = "Run not found")
    after = _after(request)
    queue = engine.subscribe(rs) if rs is not None else None
    terminal = ("run.completed", "run.failed", "run.cancelled", "run.interrupted")

    async def gen():
        try:
            sent = 0
            if after is None:
                snapshot = rs.snapshot() if rs is not None else stored
                sent = snapshot["last_seq"]
                yield _sse(
                    {
                        "seq": sent,
                        "run_id": run_id,
                        "type": "snapshot",
                        "node_id": None,
                        "ts": time.time(),
                        "data": snapshot,
                    }
                )
                if rs is None:
                    return
            else:
                sent = after
                while True:
                    page = await asyncio.to_thread(flows_db.events_after, run_id, sent)
                    for event in page:
                        sent = event["seq"]
                        yield _sse(event)
                        if event["type"] in terminal:
                            return
                    if len(page) < REPLAY_PAGE:
                        break
                if rs is None:
                    return
            last = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout = 1.0)
                except asyncio.TimeoutError:
                    if rs.finished_at is not None and queue.empty():
                        return
                    if time.monotonic() - last >= KEEPALIVE_S:
                        last = time.monotonic()
                        yield ": keepalive\n\n"
                    continue
                if event["seq"] <= sent:
                    continue
                sent = event["seq"]
                last = time.monotonic()
                yield _sse(event)
                if event["type"] in terminal:
                    return
        finally:
            if queue is not None:
                engine.unsubscribe(rs, queue)

    return StreamingResponse(
        gen(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("")
@router.get("/", include_in_schema = False)
def list_flows() -> dict:
    return {"flows": flows_db.list_flows()}


@router.post("")
@router.post("/", include_in_schema = False)
def create_flow(body: FlowBody) -> dict:
    doc = _doc_or_400(body.doc)
    if flows_db.get_flow(doc["id"]) is not None or flows_db.flow_deleted(doc["id"]):
        doc["id"] = uuid.uuid4().hex
    return _save(doc, None)


@router.get("/{flow_id}")
def get_flow(flow_id: str) -> dict:
    return _flow_or_404(flow_id)


@router.put("/{flow_id}")
def put_flow(flow_id: str, body: FlowBody) -> dict:
    doc = _doc_or_400(body.doc)
    if doc["id"] != flow_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    return _save(doc, body.baseUpdatedAt)


@router.delete("/{flow_id}", status_code = 204)
async def delete_flow(flow_id: str) -> Response:
    # A running flow would keep training or generating with no page left to stop it.
    engine = get_engine()
    for run_id in [rs.id for rs in list(engine.runs.values()) if rs.flow_id == flow_id]:
        await engine.cancel(run_id)
    flows_db.delete_flow(flow_id)
    return Response(status_code = 204)


def _hub_cached(repo: str) -> Optional[bool]:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    folder = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    snapshots = folder / "snapshots"
    try:
        return snapshots.is_dir() and any(snapshots.iterdir())
    except OSError:
        return None


async def _resident(request: Request) -> dict[str, list[str]]:
    from studio_mcp import loading

    account = current_account()
    try:
        token, key_id = await asyncio.to_thread(flow_caller.mint_run_key, account)
    except Exception:
        return {}
    try:
        caller = flow_caller.make_caller(token, account, request.app)
        return await loading.resident_slots(caller)
    except Exception:
        return {}
    finally:
        await asyncio.to_thread(flow_caller.revoke_run_key, key_id)


async def _model_report(request: Request, plan: compiler.Plan) -> list[dict]:
    from core.flows.engine import MODEL_SLOT

    needs = compiler.model_needs(plan)
    if not needs:
        return []
    resident = await _resident(request)
    report = []
    for need in needs:
        slot = MODEL_SLOT.get(need["kind"], need["kind"])
        loaded = any(name.lower() == need["repo"].lower() for name in resident.get(slot, []))
        report.append(
            {
                **need,
                "downloaded": True
                if loaded
                else await asyncio.to_thread(_hub_cached, need["repo"]),
                "loaded": loaded,
                "size_bytes": None,
                "fits": None,
            }
        )
    return report


async def _missing_models(request: Request, doc: dict) -> list[dict]:
    try:
        plan = compiler.compile_flow(doc)
    except CompileError:
        return []
    return [
        m
        for m in await _model_report(request, plan)
        if m.get("downloaded") is False and not m.get("optional")
    ]


@router.post("/{flow_id}/plan")
async def plan_flow(request: Request, flow_id: str, body: PlanBody) -> dict:
    record = _flow_or_404(flow_id)
    doc = _doc_or_400(body.doc) if body.doc is not None else record["doc"]
    try:
        plan = compiler.compile_flow(doc)
    except CompileError as exc:
        return {"models": [], "gpu_ops": 0, "cached_nodes": 0, "errors": exc.issues, "warnings": []}
    baked = set((doc.get("baked") or {}).keys())
    return {
        "models": await _model_report(request, plan),
        "gpu_ops": len(plan.gpu_nodes()),
        "cached_nodes": len([n for n in plan.order if n in baked]),
        "errors": [],
        "warnings": plan.warnings,
    }


def _hf_token(request: Request) -> Optional[str]:
    return request.headers.get(flow_caller.HF_TOKEN_HEADER) or None


@router.post("/{flow_id}/runs")
async def start_run(request: Request, flow_id: str, body: RunBody) -> dict:
    record = _flow_or_404(flow_id)
    doc = _doc_or_400(body.doc) if body.doc is not None else record["doc"]
    options = body.model_dump(exclude = {"doc"})
    try:
        rs = await get_engine().start_run(
            app = request.app, flow_id = flow_id, doc = doc, options = options, hf_token = _hf_token(request)
        )
    except CompileError as exc:
        raise _compile_error(exc)
    return {"run_id": rs.id}


@router.get("/{flow_id}/runs")
def list_runs(flow_id: str) -> dict:
    runs = flows_db.list_runs(flow_id)
    return {
        "runs": [
            {
                k: r[k]
                for k in (
                    "id",
                    "flow_id",
                    "status",
                    "error",
                    "outputs",
                    "started_at",
                    "finished_at",
                    "created_at",
                )
            }
            for r in runs
        ]
    }


def _latest_outputs(flow_id: str) -> dict[str, dict]:
    found = flows_db.latest_node_outputs(flow_id)
    engine = get_engine()
    for rs in list(engine.runs.values()):
        if (
            rs.flow_id == flow_id
            and rs.account.account_id == current_account().account_id
            and rs.persist
        ):
            for nid, state in rs.nodes.items():
                if state.status in ("done", "cached") and state.outputs is not None:
                    found[nid] = {"status": state.status, "outputs": state.outputs}
    return found


@router.get("/{flow_id}/latest-outputs")
def latest_outputs(flow_id: str) -> dict:
    _flow_or_404(flow_id)
    return {"nodes": _latest_outputs(flow_id)}


def _node_outputs(record: dict) -> dict[str, dict]:
    """Baked outputs, overridden by what this account's latest runs produced."""
    outputs = {
        nid: dict(v) for nid, v in (record["doc"].get("baked") or {}).items() if isinstance(v, dict)
    }
    for nid, entry in _latest_outputs(record["id"]).items():
        if isinstance(entry.get("outputs"), dict):
            outputs[nid] = entry["outputs"]
    return outputs


@router.post("/{flow_id}/invoke")
async def invoke(request: Request, flow_id: str, body: InvokeBody) -> dict:
    record = _flow_or_404(flow_id)
    try:
        outputs = await get_engine().invoke(
            app = request.app,
            flow_id = flow_id,
            doc = record["doc"],
            entry = body.entry,
            inputs = body.inputs,
            params = body.params,
            hf_token = _hf_token(request),
        )
    except CompileError as exc:
        raise _compile_error(exc)
    except FlowNodeError as exc:
        raise HTTPException(
            status_code = exc.status if exc.status in (409, 503) else 422, detail = str(exc)
        )
    return {"outputs": outputs, "session_id": body.session_id, "rev": body.rev}


@router.get("/{flow_id}/story")
def story(flow_id: str) -> dict:
    from core.flows.story import compile_story
    record = _flow_or_404(flow_id)
    try:
        return compile_story(record["doc"], _node_outputs(record))
    except ValueError as exc:
        raise HTTPException(status_code = 422, detail = str(exc))


@router.get("/{flow_id}/world")
def world(flow_id: str) -> dict:
    from core.flows.world import resolve_world

    record = _flow_or_404(flow_id)
    try:
        found = resolve_world(record["doc"], _node_outputs(record))
    except ValueError as exc:
        raise HTTPException(status_code = 422, detail = str(exc))
    if found is None:
        raise HTTPException(status_code = 404, detail = "Run the flow first: it has no world yet")
    return found


@router.get("/{flow_id}/sessions")
def list_sessions(flow_id: str) -> dict:
    _flow_or_404(flow_id)
    return {"sessions": flows_db.list_sessions(flow_id)}


def _slot(slot: str) -> str:
    import re
    if not re.match(_SLOT, slot):
        raise HTTPException(status_code = 400, detail = "Invalid save slot")
    return slot


@router.get("/{flow_id}/sessions/{slot}")
def get_session(flow_id: str, slot: str) -> dict:
    found = flows_db.get_session(flow_id, _slot(slot))
    if found is None:
        raise HTTPException(status_code = 404, detail = "No save in that slot")
    return found


@router.put("/{flow_id}/sessions/{slot}")
def put_session(flow_id: str, slot: str, body: SessionBody) -> dict:
    _flow_or_404(flow_id)
    if len(json.dumps(body.state, default = str)) > 4 * 1024 * 1024:
        raise HTTPException(status_code = 413, detail = "The save is too large")
    return flows_db.put_session(flow_id, _slot(slot), body.state, body.label)


@router.delete("/{flow_id}/sessions/{slot}", status_code = 204)
def delete_session(flow_id: str, slot: str) -> Response:
    flows_db.delete_session(flow_id, _slot(slot))
    return Response(status_code = 204)


@router.get("/{flow_id}/export")
async def export_flow(
    request: Request,
    flow_id: str,
    mode: str = "package",
) -> Response:
    if mode not in ("json", "package", "baked"):
        raise HTTPException(status_code = 400, detail = "mode is json, package or baked")
    record = _flow_or_404(flow_id)
    account = current_account()
    key_id: Optional[int] = None
    caller = None

    async def read_media(media: dict) -> tuple[bytes, str]:
        nonlocal key_id, caller
        from core.flows.nodes._common import media_bytes

        if caller is None:
            token, key_id = await asyncio.to_thread(flow_caller.mint_run_key, account)
            caller = flow_caller.make_caller(token, account, request.app)

        class _Ctx:
            pass

        ctx = _Ctx()

        async def call(method: str, path: str, **kwargs: Any) -> Any:
            response = await flow_caller.forward(caller, method, path)
            return flow_caller.parse_body(response)

        ctx.call = call  # type: ignore[attr-defined]
        try:
            return await media_bytes(ctx, media)
        except (FlowNodeError, flow_caller.RouteError) as exc:
            raise HTTPException(
                status_code = 409, detail = f"Could not include {media.get('ref')}: {exc}"
            )

    try:
        doc = record["doc"]
        if mode == "baked":
            # A Playable export ships what the flow last produced here, not only what an explicit bake stored.
            latest = {
                nid: entry["outputs"]
                for nid, entry in _latest_outputs(flow_id).items()
                if isinstance(entry.get("outputs"), dict)
            }
            doc = {**doc, "baked": {**(doc.get("baked") or {}), **latest}}
        data, filename, media_type = await package.export_flow(doc, mode, read_media)
    finally:
        if key_id is not None:
            await asyncio.to_thread(flow_caller.revoke_run_key, key_id)
    return Response(
        content = data,
        media_type = media_type,
        headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )
