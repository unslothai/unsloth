# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows: saved node graphs, their runs (with replayable SSE events), player entry points and saves, sharing and
importing, and flow assets. Everything is per account: flows live in the account's studio.db, live runs are only
visible to the account that started them."""

from __future__ import annotations

import asyncio
import functools
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.flows import assets as asset_store
from core.flows import caller as flow_caller
from core.flows import capabilities, compiler, notebook_import, package, replication, triggers
from core.flows.recipe_import import flow_from_recipe
from core.flows.examples import build_examples as examples
from core.flows.compiler import CompileError
from core.flows.engine import get_engine
from core.flows.nodes import automation
from core.flows.registry import FlowNodeError
from core.flows.schema import MAX_ASSET_BYTES, MAX_PACKAGE_BYTES
from core.flows.specs import catalog
from loggers import get_logger
from storage import flows_db
from utils.account_context import current_account

logger = get_logger(__name__)

router = APIRouter(dependencies = [Depends(get_current_subject)])
hooks_router = APIRouter()
WEBHOOK_MAX_BYTES = 256 * 1024

KEEPALIVE_S = 15.0
DISCONNECT_POLL_S = 0.1
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
    ignore_baked: bool = False
    pinned: dict[str, dict[str, Any]] = Field(default_factory = dict, max_length = 2000)
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
    stream: bool = False
    warm: bool = False
    background: bool = False


class RecipeImportBody(BaseModel):
    recipe_id: Optional[str] = Field(default = None, min_length = 1, max_length = 128)
    payload: Optional[dict[str, Any]] = None
    name: Optional[str] = Field(default = None, max_length = 200)


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
        record = flows_db.save_flow(doc, base)
    except flows_db.FlowDeleted:
        raise HTTPException(status_code = 410, detail = "This flow was deleted")
    except flows_db.FlowConflict:
        raise HTTPException(
            status_code = 409,
            detail = "This flow was changed in another window. Reload it to keep editing.",
        )
    if current_account().via_api_key:
        approved = flows_db.get_capabilities(record["id"])
        if approved.get("triggers"):
            flows_db.set_capabilities(record["id"], {**approved, "triggers": False})
    try:
        triggers.sync(record["id"], record["doc"])
    except Exception:
        logger.warning("flows: could not update the triggers of %s", record["id"], exc_info = True)
    return record


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
        "needs": _example_needs(doc),
    }


def _example_needs(doc: dict) -> list[str]:
    """The model kinds an example needs, in first-use order, plus "training" when it fine-tunes."""
    needs: list[str] = []
    for model in (doc.get("requirements") or {}).get("models") or []:
        kind = model.get("kind") if isinstance(model, dict) else None
        if kind and not model.get("optional") and kind not in needs:
            needs.append(kind)
    if any(str(n.get("type", "")).startswith("train.") for n in doc.get("nodes") or []):
        needs.append("training")
    return needs


@router.get("/examples")
def list_examples() -> dict:
    return {"examples": _example_summaries()}


@functools.lru_cache(maxsize = 1)
def _example_summaries() -> list[dict]:
    return [_example_summary(doc) for doc in examples.example_docs()]


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
    report["replication"] = replication.notices(doc)
    if report.get("baked_nodes") and replication.live_conversations(doc):
        report["live_note"] = replication.LIVE_NOTE
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


@router.post("/import/notebook")
async def import_notebook(request: Request, name: str = "notebook.ipynb") -> dict:
    """An Unsloth notebook read as data (never run): a new flow, or a report saying to open it in Colab."""
    data = await _read_body(request, notebook_import.MAX_NOTEBOOK_BYTES)
    if not data:
        raise HTTPException(status_code = 400, detail = "The upload is empty")
    try:
        doc, report = await asyncio.to_thread(notebook_import.read_notebook, data, name[:200])
    except notebook_import.NotebookImportError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    if doc is None:
        return {"flow": None, "report": report}
    record, _report = await asyncio.to_thread(_import_doc, json.dumps(doc).encode())
    report["missing_models"] = await _missing_models(request, record["doc"])
    return {"flow": record, "report": report}


def _recipe_flow(payload: Any, name: str, source: Optional[dict]) -> tuple[dict, list[str]]:
    try:
        doc, notices = flow_from_recipe(payload, name = name, source = source)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    doc["id"] = uuid.uuid4().hex
    return _save(_doc_or_400(doc), None), notices


def _recipe_source(record: dict) -> dict:
    return {"id": record["id"], "name": record.get("name"), "updatedAt": record.get("updatedAt")}


@router.post("/import/recipe")
def import_recipe(body: RecipeImportBody) -> dict:
    """A saved Data Recipe (by id) or recipe JSON as a new flow; the recipe itself is left alone.
    Asking again for an unchanged saved recipe opens the flow made last time."""
    from storage import data_recipes_db

    if body.recipe_id:
        record = data_recipes_db.get_recipe(body.recipe_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "Recipe not found")
        # The recipe unchanged since its newest flow was made: open that flow again.
        known = flows_db.recipe_flows().get(record["id"])
        existing = flows_db.get_flow(known) if known else None
        made_from = ((existing or {}).get("doc", {}).get("meta") or {}).get("from_recipe") or {}
        if existing and made_from.get("updatedAt") == record.get("updatedAt") and not body.name:
            return {"flow": existing, "report": {"notices": [], "reused": True}}
        payload, name, source = record["payload"], record.get("name"), _recipe_source(record)
    elif body.payload is not None:
        payload, name, source = body.payload, None, None
    else:
        raise HTTPException(status_code = 400, detail = "Send a recipe_id or a recipe payload")
    flow, notices = _recipe_flow(payload, body.name or name or "Data Recipe", source)
    return {"flow": flow, "report": {"notices": notices}}


@router.post("/import/recipes")
def import_all_recipes() -> dict:
    """A flow per saved Data Recipe not in Flows yet. Safe to repeat; recipes are left alone."""
    from storage import data_recipes_db

    known = flows_db.recipe_flows()
    created, skipped, failed = [], [], []
    for record in data_recipes_db.list_recipes():
        if record["id"] in known:
            skipped.append({"recipe_id": record["id"], "flow_id": known[record["id"]]})
            continue
        name = str(record.get("name") or "Data Recipe")
        try:
            flow, notices = _recipe_flow(record["payload"], name, _recipe_source(record))
        except HTTPException as exc:
            failed.append({"recipe_id": record["id"], "name": name, "error": str(exc.detail)})
            continue
        created.append(
            {"recipe_id": record["id"], "flow_id": flow["id"], "name": name, "notices": notices}
        )
    return {"created": created, "skipped": skipped, "failed": failed}


@router.post("/assets")
async def upload_asset(
    request: Request,
    name: Optional[str] = None,
    mime: Optional[str] = None,
) -> dict:
    data = await _read_body(request, MAX_ASSET_BYTES)
    hint = asset_store.text_hint(mime, name)
    sniffed = asset_store.sniff_mime(data) or await asyncio.to_thread(
        asset_store.sniff_text, data, hint
    )
    if asset_store.kind_of(sniffed or "") not in ("image", "audio", "video", "document", "model"):
        raise HTTPException(
            status_code = 415,
            detail = "Flows take images, audio, video, documents (PDF, DOCX, text) and GLB models",
        )
    try:
        digest, sniffed = await asyncio.to_thread(asset_store.store, data, hint or mime)
    except ValueError as exc:
        raise HTTPException(status_code = 415, detail = str(exc))
    label = (name or "")[:200] if asset_store.kind_of(sniffed) == "document" else None
    return {"media": asset_store.media_value(digest, sniffed, name = label or None)}


@router.get("/assets/{ref}")
def get_asset(ref: str) -> Response:
    digest = asset_store.parse_ref(ref)
    got = asset_store.read(digest) if digest else None
    if got is None:
        raise HTTPException(status_code = 404, detail = "Asset not found")
    data, mime = got
    if mime.startswith("text/"):
        # Never as HTML: an uploaded page served on Studio's origin would run its scripts.
        mime = "text/plain; charset=utf-8"
    return Response(
        content = data,
        media_type = mime,
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
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
                if rs is not None:
                    await engine.persist(rs)
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
                    yield _sse(
                        {
                            "seq": max(sent, stored["last_seq"]),
                            "run_id": run_id,
                            "type": "snapshot",
                            "node_id": None,
                            "ts": time.time(),
                            "data": stored,
                        }
                    )
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
    engine = get_engine()
    stopping = []
    for rs in list(engine.runs.values()):
        if rs.flow_id == flow_id and engine.live(rs.id) is rs:
            await engine.cancel(rs.id)
            if rs.task is not None and not rs.task.done():
                stopping.append(rs.task)
    if stopping:
        await asyncio.wait(stopping, timeout = 30)
    flows_db.delete_flow(flow_id)
    try:
        from core.inference.tools import remove_session_sandbox
        await asyncio.to_thread(remove_session_sandbox, automation.python_session(flow_id), True)
    except Exception:
        logger.warning("flows: could not remove the sandbox of %s", flow_id, exc_info = True)
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
    models = await _model_report(request, plan)
    return {
        "models": models,
        "gpu_ops": len(plan.gpu_nodes()),
        "cached_nodes": len([n for n in plan.order if n in baked]),
        "errors": [],
        "warnings": plan.warnings,
        "hints": await asyncio.to_thread(_residency_hints, models),
    }


SWAPPING_KINDS = ("chat", "vision", "tts", "audio", "image", "video")


def _residency_hints(models: list[dict]) -> list[dict]:
    """A flow that alternates between models while Studio evicts on every load: say how to stop the reloads."""
    repos = {
        m["repo"].lower()
        for m in models
        if m.get("kind") in SWAPPING_KINDS and not m.get("optional")
    }
    if len(repos) < 2:
        return []
    try:
        from utils.multi_model_settings import get_multi_model_mode
        mode = get_multi_model_mode()
    except Exception:
        return []
    if mode != "off":
        return []
    return [
        {
            "id": "multi_model",
            "message": f"This flow switches between {len(repos)} models. With Keep multiple models loaded set to "
            "When they fit, models that fit stay loaded instead of reloading at every switch.",
        }
    ]


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


@router.get("/{flow_id}/nodes/{node_id}/history")
def node_history(
    flow_id: str,
    node_id: str,
    limit: int = Query(10, ge = 1, le = 10),
) -> dict:
    _flow_or_404(flow_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", node_id):
        raise HTTPException(status_code = 400, detail = "Invalid node id")
    return {"entries": flows_db.node_output_history(flow_id, node_id, limit)}


@router.get("/{flow_id}/nodes/{node_id}/recipe-runs")
def recipe_runs(
    flow_id: str,
    node_id: str,
    limit: int = Query(50, ge = 1, le = 200),
) -> dict:
    """A Recipe Dataset step's runs as Recipe Studio execution records, newest first."""
    _flow_or_404(flow_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", node_id):
        raise HTTPException(status_code = 400, detail = "Invalid node id")
    return {"executions": flows_db.recipe_runs(flow_id, node_id, limit)}


@router.get("/{flow_id}/latest-outputs")
def latest_outputs(flow_id: str) -> dict:
    _flow_or_404(flow_id)
    return {"nodes": _latest_outputs(flow_id)}


def _node_outputs(record: dict) -> dict[str, dict]:
    outputs = {
        nid: dict(v) for nid, v in (record["doc"].get("baked") or {}).items() if isinstance(v, dict)
    }
    for nid, entry in _latest_outputs(record["id"]).items():
        if isinstance(entry.get("outputs"), dict):
            outputs[nid] = entry["outputs"]
    return outputs


def _invoke_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CompileError):
        return _compile_error(exc)
    status = exc.status if isinstance(exc, FlowNodeError) and exc.status in (409, 503) else 422
    return HTTPException(status_code = status, detail = str(exc))


@router.post("/{flow_id}/invoke", response_model = None)
async def invoke(request: Request, flow_id: str, body: InvokeBody) -> Any:
    record = _flow_or_404(flow_id)
    engine = get_engine()
    if body.warm:
        try:
            ready = await engine.warm(
                app = request.app,
                flow_id = flow_id,
                doc = record["doc"],
                entry = body.entry,
                hf_token = _hf_token(request),
            )
        except (CompileError, FlowNodeError) as exc:
            raise _invoke_error(exc)
        return {"warm": ready}
    call = dict(
        app = request.app,
        flow_id = flow_id,
        doc = record["doc"],
        entry = body.entry,
        inputs = body.inputs,
        params = body.params,
        hf_token = _hf_token(request),
        timing = {},
        background = body.background,
    )
    if body.stream:
        return _invoke_stream(request, engine, call, body)
    task = asyncio.create_task(engine.invoke(**call))
    try:
        while not (await asyncio.wait({task}, timeout = DISCONNECT_POLL_S))[0]:
            if await request.is_disconnected():
                task.cancel()
                raise HTTPException(
                    status_code = 499, detail = "The player left before the answer was ready"
                )
        outputs = task.result()
    except (CompileError, FlowNodeError) as exc:
        raise _invoke_error(exc)
    finally:
        if not task.done():
            task.cancel()
    return {
        "outputs": outputs,
        "session_id": body.session_id,
        "rev": body.rev,
        "timing": call["timing"],
    }


def _invoke_stream(
    request: Request, engine: Any, call: dict, body: InvokeBody
) -> StreamingResponse:
    events: asyncio.Queue = asyncio.Queue(maxsize = 10_000)
    entry = body.entry

    def frame(data: dict) -> str:
        return f"data: {json.dumps(data, default = str)}\n\n"

    async def gen():
        task = asyncio.create_task(engine.invoke(**call, events = events))
        try:
            while True:
                getter = asyncio.ensure_future(events.get())
                done, _ = await asyncio.wait(
                    {getter, task}, timeout = KEEPALIVE_S, return_when = asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    event = getter.result()
                    if event["type"] == "node.text_delta" and event.get("node_id") == entry:
                        yield frame(
                            {
                                "type": "delta",
                                "port": event["data"].get("port"),
                                "text": event["data"].get("delta", ""),
                            }
                        )
                    continue
                getter.cancel()
                if task in done:
                    break
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
            while not events.empty():
                event = events.get_nowait()
                if event["type"] == "node.text_delta" and event.get("node_id") == entry:
                    yield frame(
                        {
                            "type": "delta",
                            "port": event["data"].get("port"),
                            "text": event["data"].get("delta", ""),
                        }
                    )
            try:
                outputs = task.result()
            except (CompileError, FlowNodeError) as exc:
                error = _invoke_error(exc)
                yield frame({"type": "error", "status": error.status_code, "detail": error.detail})
                return
            except Exception as exc:
                logger.exception("flows: streamed invoke of %s failed", entry)
                yield frame(
                    {"type": "error", "status": 500, "detail": f"The entry point failed: {exc}"}
                )
                return
            yield frame(
                {
                    "type": "result",
                    "outputs": outputs,
                    "session_id": body.session_id,
                    "rev": body.rev,
                    "timing": call["timing"],
                }
            )
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        gen(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


class ApproveBody(BaseModel):
    """What the owner saw and approves; only what the flow still needs is stored."""

    python: list[str] = Field(default_factory = list, max_length = 500)
    hosts: list[str] = Field(default_factory = list, max_length = 500)
    mcp: bool = False
    providers: list[str] = Field(default_factory = list, max_length = 100)
    triggers: bool = False


def _owner_hand() -> bool:
    account = current_account()
    return account.is_owner and not account.via_api_key


def _capability_report(record: dict) -> dict:
    approved = flows_db.get_capabilities(record["id"])
    return {
        "needs": capabilities.doc_needs(record["doc"]),
        "pending": capabilities.pending(record["doc"], approved),
        "approved": approved,
        "can_approve": _owner_hand(),
    }


@router.get("/{flow_id}/capabilities")
def get_capabilities(flow_id: str) -> dict:
    return _capability_report(_flow_or_404(flow_id))


@router.post("/{flow_id}/approve")
def approve_capabilities(flow_id: str, body: ApproveBody) -> dict:
    if not _owner_hand():
        raise HTTPException(
            status_code = 403,
            detail = "Only the Studio owner can approve this, from Unsloth Studio itself",
        )
    record = _flow_or_404(flow_id)
    need = capabilities.doc_needs(record["doc"])
    before = flows_db.get_capabilities(flow_id)
    approved = {
        "python": [
            h
            for h in need["python"]
            if h in body.python or capabilities.is_approved(before, "python", h)
        ],
        "hosts": [
            h
            for h in need["hosts"]
            if h in body.hosts or capabilities.is_approved(before, "http", h)
        ],
        "mcp": need["mcp"] and (body.mcp or capabilities.is_approved(before, "mcp", "mcp")),
        "providers": [
            p
            for p in need["providers"]
            if p in body.providers or capabilities.is_approved(before, "provider", p)
        ],
        "triggers": need["triggers"]
        and (body.triggers or capabilities.is_approved(before, "trigger", "trigger")),
    }
    flows_db.set_capabilities(flow_id, approved)
    triggers.sync(flow_id, record["doc"])
    return _capability_report(record)


def _trigger_view(row: dict) -> dict:
    view = {"node_id": row["node_id"], "kind": row["kind"], **row["config"]}
    if row["kind"] == "webhook" and row.get("token"):
        view["path"] = f"/api/flows/hooks/{row['token']}"
    view["last_fired_at"] = row.get("last_fired_at")
    return view


@router.get("/{flow_id}/triggers")
def list_flow_triggers(flow_id: str) -> dict:
    _flow_or_404(flow_id)
    if not _owner_hand():
        return {"triggers": [], "owner_only": True}
    return {
        "triggers": [_trigger_view(r) for r in flows_db.list_triggers(flow_id)],
        "owner_only": False,
    }


@router.post("/{flow_id}/triggers/{node_id}/rotate")
def rotate_trigger(flow_id: str, node_id: str) -> dict:
    if not _owner_hand():
        raise HTTPException(status_code = 403, detail = "Only the Studio owner can change webhooks")
    _flow_or_404(flow_id)
    if not flows_db.rotate_trigger_token(flow_id, node_id, triggers.new_token()):
        raise HTTPException(status_code = 404, detail = "No webhook on that block; save the flow first")
    return list_flow_triggers(flow_id)


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


@hooks_router.post("/{token}")
async def call_webhook(request: Request, token: str) -> dict:
    from utils.account_context import OWNER, run_as

    if not triggers.TOKEN_RE.match(token):
        raise HTTPException(status_code = 404, detail = "Not found")
    # Resolve the secret first: unknown URLs cost no body read, and only real secrets get a rate-limit bucket.
    row = await asyncio.to_thread(run_as, OWNER, flows_db.trigger_by_token, token)
    if row is None:
        raise HTTPException(status_code = 404, detail = "Not found")
    if triggers.rate_limited(token):
        raise HTTPException(status_code = 429, detail = "Too many calls; wait a minute")
    if not row["config"].get("enabled"):
        raise HTTPException(status_code = 409, detail = "This webhook is switched off")
    data = await _read_body(request, WEBHOOK_MAX_BYTES)
    try:
        # Strict like the Python and HTTP blocks: NaN / Infinity would make the run's snapshot unreadable.
        payload = json.loads(data, parse_constant = _reject_constant) if data.strip() else {}
    except ValueError:
        raise HTTPException(status_code = 400, detail = "The body must be JSON")
    from utils.account_context import arun_as

    received = {
        "payload": payload,
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run_id = await arun_as(
        OWNER,
        triggers.fire(request.app, row["flow_id"], row["node_id"], received, kind = "webhook"),
    )
    if run_id is None:
        raise HTTPException(
            status_code = 409,
            detail = "Not started: triggers are not approved for this flow, or its last run is still going",
        )
    return {"run_id": run_id}
