# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Thin HTTP face over Apache Unforgettable operators and the B store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from loggers import get_logger
from utils.utils import log_and_http_error, safe_error_detail
from unforgettable.eyes.gate import contradictions
from unforgettable.operators import (
    ERROR_BLOCKED,
    ERROR_INVALID,
    ERROR_NO_HOST,
    ERROR_REFUSED,
    ERROR_UNKNOWN,
    ERROR_VOTER_OFF,
    STUDIO_ADMIT_REASON,
    STUDIO_REJECT_REASON,
    admit_record,
    compile_record,
    mine_store,
    promote_adapter_record,
    reject_record,
    review_proposed,
    summarize_store,
)
from unforgettable.sidecar.adapters import list_adapters, rollback_adapter
from unforgettable.sidecar.pack import list_packs
from unforgettable.store.compact import run_compact
from unforgettable.store.compile import get_compiled, list_compiled, unpin_compiled
from unforgettable.store.records import (
    deprecate_record,
    get_record,
    list_admissions,
    list_inject_stats,
    list_records,
    list_rollouts,
    update_proposed_record,
)
from unforgettable.store.search import search_records
from unforgettable.supervisor import resolve_supervisor_host
from utils.unforgettable_settings import (
    get_unforgettable_settings,
    memory_db_path,
    set_unforgettable_settings,
    supervisor_config_from_settings,
)

router = APIRouter()
logger = get_logger(__name__)


def _db() -> Path:
    return memory_db_path()


def _host():
    return resolve_supervisor_host()


def _config():
    return supervisor_config_from_settings()


def _raise_outcome(outcome) -> None:
    kind = outcome.error_kind
    if kind == ERROR_UNKNOWN:
        raise HTTPException(status_code = 404, detail = f"unknown id: {outcome.error_detail}")
    if kind == ERROR_REFUSED:
        raise HTTPException(
            status_code = 409,
            detail = f"admit refused: status is {outcome.error_detail} (use force)",
        )
    if kind == ERROR_BLOCKED:
        raise HTTPException(
            status_code = 409,
            detail = f"refused: voter deny: {outcome.error_detail}",
        )
    if kind == ERROR_VOTER_OFF:
        raise HTTPException(
            status_code = 400,
            detail = "voter off; set Unforgettable voter to advisory or binding",
        )
    if kind == ERROR_NO_HOST:
        raise HTTPException(status_code = 400, detail = "mine needs a supervisor URL")
    if kind == ERROR_INVALID:
        raise HTTPException(status_code = 400, detail = outcome.error_detail or "invalid")
    if not outcome.ok:
        raise HTTPException(status_code = 400, detail = outcome.error_detail or "failed")


class RecordPatch(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None


class AdmitBody(BaseModel):
    force: bool = False


class RejectBody(BaseModel):
    reason: Optional[str] = None


class ApplyBody(BaseModel):
    apply: bool = False
    limit: int = Field(default = 20, ge = 1, le = 100)


class CompactBody(BaseModel):
    apply: bool = False


class DeprecateBody(BaseModel):
    reason: Optional[str] = None


@router.get("/summary")
def get_summary(current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    return summarize_store(db_path = _db())


@router.get("/records")
def get_records(
    current_subject: str = Depends(get_current_subject),
    status: Optional[str] = None,
    kind: Optional[str] = None,
    q: Optional[str] = None,
    provenance: Optional[str] = None,
    limit: int = Query(default = 40, ge = 1, le = 200),
    offset: int = Query(default = 0, ge = 0),
) -> dict[str, Any]:
    statuses = (
        None
        if not status or status == "all"
        else [part.strip() for part in status.split(",") if part.strip()]
    )
    kinds = (
        None
        if not kind or kind == "all"
        else [part.strip() for part in kind.split(",") if part.strip()]
    )
    provenances = (
        None if not provenance else [part.strip() for part in provenance.split(",") if part.strip()]
    )
    if q:
        rows = search_records(
            q,
            top_k = limit,
            kinds = kinds,
            statuses = statuses,
            provenances = provenances,
            db_path = _db(),
        )
    else:
        rows = list_records(
            statuses = statuses,
            kinds = kinds,
            limit = limit,
            offset = offset,
            db_path = _db(),
        )
        if provenances:
            rows = [row for row in rows if row.get("provenance") in set(provenances)]
    return {"records": rows}


@router.get("/records/{record_id}")
def get_one_record(
    record_id: str, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    rec = get_record(record_id, db_path = _db())
    if rec is None:
        raise HTTPException(status_code = 404, detail = f"unknown id: {record_id}")
    return rec


@router.patch("/records/{record_id}")
def patch_record(
    record_id: str,
    payload: RecordPatch,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        return update_proposed_record(
            record_id,
            title = payload.title,
            body = payload.body,
            db_path = _db(),
        )
    except KeyError:
        raise HTTPException(status_code = 404, detail = f"unknown id: {record_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.post("/records/{record_id}/admit")
def post_admit(
    record_id: str,
    payload: AdmitBody,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    outcome = admit_record(
        record_id,
        force = payload.force,
        db_path = _db(),
        host = _host(),
        config = _config(),
        reason = STUDIO_ADMIT_REASON,
    )
    if outcome.vote is not None and not outcome.ok and outcome.error_kind != ERROR_BLOCKED:
        _raise_outcome(outcome)
    if not outcome.ok:
        _raise_outcome(outcome)
    result = dict(outcome.record or {})
    if outcome.vote is not None:
        result["vote"] = {
            "decision": outcome.vote.decision,
            "reason": outcome.vote.reason,
        }
    return result


@router.post("/records/{record_id}/reject")
def post_reject(
    record_id: str,
    payload: RejectBody,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    outcome = reject_record(
        record_id,
        reason = payload.reason or STUDIO_REJECT_REASON,
        db_path = _db(),
    )
    if not outcome.ok:
        _raise_outcome(outcome)
    return outcome.record or {}


@router.post("/records/{record_id}/deprecate")
def post_deprecate(
    record_id: str,
    payload: DeprecateBody,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        return deprecate_record(record_id, reason = payload.reason, db_path = _db())
    except KeyError:
        raise HTTPException(status_code = 404, detail = f"unknown id: {record_id}") from None


@router.get("/compiled")
def get_compiled_rows(current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    return {"records": list_compiled(db_path = _db())}


@router.post("/compile/{record_id}")
def post_compile(
    record_id: str, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    outcome = compile_record(
        record_id,
        db_path = _db(),
        host = _host(),
        config = _config(),
    )
    if not outcome.ok:
        _raise_outcome(outcome)
    return outcome.record or {}


@router.post("/uncompile/{record_id}")
def post_uncompile(
    record_id: str, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    row = get_compiled(record_id, db_path = _db())
    if row is None:
        raise HTTPException(status_code = 404, detail = f"unknown id: {record_id}")
    unpin_compiled(record_id, db_path = _db())
    return row


@router.get("/contradictions")
def get_contradictions(current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    rows = contradictions(db_path = _db())
    return {
        "contradictions": [
            {
                "title_key": item.title_key,
                "record_ids": list(item.record_ids),
                "reason": item.reason,
            }
            for item in rows
        ]
    }


@router.get("/admissions")
def get_admissions(
    current_subject: str = Depends(get_current_subject),
    limit: int = Query(default = 50, ge = 1, le = 200),
) -> dict[str, Any]:
    return {"admissions": list_admissions(limit = limit, db_path = _db())}


@router.get("/rollouts")
def get_rollouts(
    current_subject: str = Depends(get_current_subject),
    contact: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = Query(default = 40, ge = 1, le = 200),
) -> dict[str, Any]:
    return {
        "rollouts": list_rollouts(
            contact = contact,
            outcome = outcome,
            limit = limit,
            db_path = _db(),
        )
    }


@router.get("/load")
def get_load(
    current_subject: str = Depends(get_current_subject),
    limit: int = Query(default = 20, ge = 1, le = 100),
) -> dict[str, Any]:
    return {"inject": list_inject_stats(limit = limit, db_path = _db())}


@router.post("/compact")
def post_compact(
    payload: CompactBody, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    from dataclasses import asdict
    report = run_compact(_db(), dry_run = not payload.apply)
    return asdict(report)


@router.post("/review")
def post_review(
    payload: ApplyBody, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    outcome = review_proposed(
        apply = payload.apply,
        limit = payload.limit,
        db_path = _db(),
        host = _host(),
        config = _config(),
    )
    if not outcome.ok:
        _raise_outcome(outcome)
    return {"items": outcome.items or []}


@router.post("/mine")
def post_mine(
    payload: ApplyBody, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    outcome = mine_store(
        apply = payload.apply,
        limit = payload.limit,
        db_path = _db(),
        host = _host(),
        config = _config(),
    )
    if not outcome.ok:
        _raise_outcome(outcome)
    return {"items": outcome.items or []}


@router.get("/adapters")
def get_adapters(
    current_subject: str = Depends(get_current_subject), status: Optional[str] = None
) -> dict[str, Any]:
    return {"adapters": list_adapters(status = status, db_path = _db())}


@router.get("/packs")
def get_packs(
    current_subject: str = Depends(get_current_subject),
    limit: int = Query(default = 20, ge = 1, le = 100),
) -> dict[str, Any]:
    return {"packs": list_packs(limit = limit, db_path = _db())}


@router.post("/adapters/{adapter_id}/promote")
def post_promote(
    adapter_id: str,
    payload: AdmitBody,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    outcome = promote_adapter_record(
        adapter_id,
        force = payload.force,
        db_path = _db(),
        host = _host(),
        config = _config(),
    )
    if not outcome.ok:
        _raise_outcome(outcome)
    return outcome.record or {}


@router.post("/adapters/rollback")
def post_rollback(current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    row = rollback_adapter(db_path = _db())
    return {"promoted": row}


class UnforgettableSettingsPayload(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    planner: Optional[str] = None
    planner_model: Optional[str] = None
    filter: Optional[str] = None
    filter_model: Optional[str] = None
    judge_model: Optional[str] = None
    stakes: Optional[str] = None
    confirm_retry: Optional[bool] = None
    skip_standing: Optional[bool] = None
    adapter_id: Optional[str] = None
    test_command: Optional[str] = None
    max_clones: Optional[int] = Field(default = None, ge = 1)
    max_sim_turns: Optional[int] = Field(default = None, ge = 1)
    twin_plugin: Optional[str] = None
    voter: Optional[str] = None
    voter_model: Optional[str] = None
    supervisor_url: Optional[str] = None
    supervisor_timeout: Optional[float] = Field(default = None, gt = 0)


class UnforgettableSettingsResponse(UnforgettableSettingsPayload):
    db_path: str = ""
    namespace: str = "default"


def _settings_response() -> UnforgettableSettingsResponse:
    return UnforgettableSettingsResponse.model_validate(get_unforgettable_settings())


@router.get("/settings", response_model = UnforgettableSettingsResponse)
def get_unforgettable_settings_route(
    current_subject: str = Depends(get_current_subject),
) -> UnforgettableSettingsResponse:
    return _settings_response()


@router.put("/settings", response_model = UnforgettableSettingsResponse)
def update_unforgettable_settings_route(
    payload: UnforgettableSettingsPayload, current_subject: str = Depends(get_current_subject)
) -> UnforgettableSettingsResponse:
    try:
        set_unforgettable_settings(payload.model_dump(exclude_unset = True))
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid Unforgettable settings."),
            event = "unforgettable.update_settings_failed",
            log = logger,
        ) from exc
    return _settings_response()
