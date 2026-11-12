# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Subagents spawned by a chat: list, read and stop them (see studio/SUBAGENTS.md)."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from auth.authentication import get_current_subject
from storage import subagents_db as db

router = APIRouter()

_NOT_FOUND = "Subagent not found"
_STOPPED = "Studio stopped before this subagent finished."


def _live():
    """The in-memory registry, or None while core.inference.subagents is not available."""
    try:
        from core.inference import subagents as live
    except ImportError:
        return None
    return live


def _sort_key(record: dict) -> int:
    return record.get("startedAt") or 0


def _settle_stale(stored: dict) -> dict:
    """A stored run still "running" with no live record died with Studio: fail it, and keep that."""
    if stored.get("status") != "running":
        return stored
    full = db.get_subagent(stored["id"]) or dict(stored)
    full["status"] = "failed"
    full["error"] = _STOPPED
    full["pendingApproval"] = None
    full["finishedAt"] = full.get("finishedAt") or int(time.time() * 1000)
    full["transcript"] = [
        {**item, "status": "done", "result": item.get("result") or "Not finished."}
        if item.get("kind") == "tool" and item.get("status") != "done"
        else item
        for item in full.get("transcript") or []
        if item.get("kind") != "status"
    ]
    db.upsert_subagent(full)
    if "transcript" not in stored:
        full = {key: value for key, value in full.items() if key != "transcript"}
    return full


@router.get("")
def list_subagents(
    parent_thread_id: str = Query(..., min_length = 1),
    current_subject: str = Depends(get_current_subject),
):
    live = _live()
    live_rows = live.live_subagents_for_parent(parent_thread_id) if live is not None else []
    live_ids = {record["id"] for record in live_rows}
    merged = {
        record["id"]: record if record["id"] in live_ids else _settle_stale(record)
        for record in db.list_subagents(parent_thread_id)
    }
    for record in live_rows:
        merged[record["id"]] = record
    return {"subagents": sorted(merged.values(), key = _sort_key)}


@router.get("/{subagent_id}")
def get_subagent(subagent_id: str, current_subject: str = Depends(get_current_subject)):
    record: Optional[dict] = None
    live = _live()
    if live is not None:
        record = live.live_subagent(subagent_id)
    if record is None:
        record = db.get_subagent(subagent_id)
        if record is not None:
            record = _settle_stale(record)
    if record is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    record.setdefault("transcript", [])
    return record


@router.post("/{subagent_id}/cancel")
def cancel_subagent(subagent_id: str, current_subject: str = Depends(get_current_subject)):
    live = _live()
    return {"cancelled": bool(live.cancel_subagent(subagent_id)) if live is not None else False}
