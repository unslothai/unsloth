# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated durable inline Deep Research API."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.inference.message_content import content_to_text
from core.inference.web_access_policy import normalize_website_policy
from storage import research_runs_db as db
from storage.studio_db import get_chat_message, get_chat_thread, upsert_chat_message

router = APIRouter()
_SENSITIVE_KEY_EXACT = {
    "authorization",
    "password",
    "secret",
    "token",
    "apikey",
    "credential",
    "credentials",
}
_SENSITIVE_KEY_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesstoken",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "privatekey",
    "refreshtoken",
    "sessiontoken",
)
_MAX_PLAN_STEPS = 30
_DELTA_ONLY_EVENTS = {"reasoning.updated", "report.updated"}


class CreateResearchRun(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    threadId: str
    userMessageId: str
    assistantMessageId: str | None = Field(
        default = None,
        validation_alias = AliasChoices("unstable_assistantMessageId", "assistantMessageId"),
    )
    inferenceRequest: dict[str, Any] = Field(default_factory = dict)
    ragScope: dict[str, Any] | None = None
    budgets: dict[str, int] | None = None
    websitePolicy: dict[str, list[str]] | None = None
    instructions: str | None = Field(default = None, max_length = 32_000)


class ResearchPlanStep(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    title: str = Field(min_length = 1, max_length = 200)
    query: str = Field(min_length = 1, max_length = 500)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    title: str = Field(min_length = 1, max_length = 200)
    steps: list[ResearchPlanStep] = Field(min_length = 1, max_length = _MAX_PLAN_STEPS)


class UpdatePlan(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    plan: ResearchPlan
    expectedRevision: int = Field(ge = 0)


class ApprovePlan(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    planRevision: int = Field(ge = 1)
    planHash: str = Field(min_length = 64, max_length = 64)


def _require_run(run_id: str) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "Research run not found")
    return run


def _sync_assistant(run: dict, text: str | None = None) -> None:
    message_id = db.discover_and_bind_assistant_message(run["id"])
    if not message_id:
        if run["status"] not in db.TERMINAL_STATUSES:
            return
        fallback_text = (
            text
            or {
                "cancelled": "Research cancelled.",
                "failed": f"Research failed: {run.get('error') or 'Unknown error'}",
                "completed": "Research completed.",
            }[run["status"]]
        )
        message_id, created = db.create_and_bind_terminal_fallback(
            run["id"],
            text = fallback_text,
            status = run["status"],
        )
        if created:
            return
    message = get_chat_message(run["threadId"], message_id)
    if message is None:
        return
    content = message.get("content") if isinstance(message.get("content"), list) else []
    if text is not None:
        content = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("researchRunId") == run["id"])
        ]
        content.append({"type": "text", "text": text, "researchRunId": run["id"]})
    metadata = dict(message.get("metadata") or {})
    metadata.update(
        {
            "researchRunId": run["id"],
            "researchStatus": run["status"],
            "researchPlanRevision": run["planRevision"],
            "serverManaged": True,
        }
    )
    upsert_chat_message(
        {
            **message,
            "content": content,
            "metadata": metadata,
        },
        allow_research_update = True,
    )


def _is_sensitive_key(key: object) -> bool:
    # Match after stripping separators/case so openaiApiKey, access_token, clientSecret all hit.
    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    return normalized in _SENSITIVE_KEY_EXACT or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _contains_sensitive_key(value: object) -> bool:
    """Recursively test whether any (possibly nested) mapping key looks sensitive,
    so credentials cannot be smuggled into a durable run via a nested dict."""
    if isinstance(value, dict):
        return any(
            _is_sensitive_key(key) or _contains_sensitive_key(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _sanitize_config(payload: CreateResearchRun, thread: dict) -> dict:
    request = dict(payload.inferenceRequest)
    if _contains_sensitive_key(request):
        raise HTTPException(status_code = 400, detail = "Inference credentials cannot be persisted")
    if any(key in request for key in ("baseUrl", "endpoint", "provider", "tools", "enabledTools")):
        raise HTTPException(
            status_code = 400,
            detail = "Durable research currently supports only the selected local Studio model",
        )
    allowed = {
        "model",
        "temperature",
        "topP",
        "maxTokens",
        "enableThinking",
        "reasoningEffort",
    }
    unknown = set(request) - allowed
    if unknown:
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported inferenceRequest fields: {', '.join(sorted(unknown))}",
        )
    model = str(request.get("model") or thread.get("modelId") or "").strip()
    if not model:
        raise HTTPException(status_code = 400, detail = "A selected local model is required")
    request["model"] = model
    try:
        if "temperature" in request:
            request["temperature"] = float(request["temperature"])
            if not 0 <= request["temperature"] <= 2:
                raise ValueError
        if "topP" in request:
            request["topP"] = float(request["topP"])
            if not 0 < request["topP"] <= 1:
                raise ValueError
        if "maxTokens" in request:
            request["maxTokens"] = int(request["maxTokens"])
            if not 1 <= request["maxTokens"] <= 8192:
                raise ValueError
        if "enableThinking" in request and not isinstance(request["enableThinking"], bool):
            raise ValueError
        if "reasoningEffort" in request:
            request["reasoningEffort"] = str(request["reasoningEffort"])
            if request["reasoningEffort"] not in {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "max",
                "xhigh",
            }:
                raise ValueError
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code = 400, detail = "Invalid inferenceRequest value") from exc
    rag_scope = payload.ragScope
    if rag_scope is not None:
        allowed_rag = {
            "kb_id",
            "thread_id",
            "project_id",
            "default_top_k",
            "mode",
            "autoinject",
            "autoinject_min_score",
            "whole_doc",
        }
        unknown_rag = set(rag_scope) - allowed_rag
        # Every ragScope field is a scalar (id strings, an int, an enum, floats, bools). A nested
        # container both evades the sensitive-key scan when its inner keys are unlisted (e.g.
        # {"kb_id": {"auth": "sk-..."}}) and would reach retrieval code that expects a scalar scope
        # id, so reject any non-scalar value outright.
        non_scalar = any(isinstance(value, (dict, list, tuple)) for value in rag_scope.values())
        if unknown_rag or non_scalar or _contains_sensitive_key(rag_scope):
            raise HTTPException(status_code = 400, detail = "Unsupported or sensitive ragScope field")
    budgets = {
        "maxSteps": 12,
        "maxSources": 40,
        "modelTimeoutSeconds": 900,
        "toolTimeoutSeconds": 120,
    }
    for key, value in (payload.budgets or {}).items():
        if key not in budgets:
            raise HTTPException(status_code = 400, detail = f"Unsupported budget: {key}")
        budgets[key] = int(value)
    limits = {
        "maxSteps": (1, _MAX_PLAN_STEPS),
        "maxSources": (1, 100),
        "modelTimeoutSeconds": (10, 3600),
        "toolTimeoutSeconds": (5, 600),
    }
    for key, (minimum, maximum) in limits.items():
        if not minimum <= budgets[key] <= maximum:
            raise HTTPException(
                status_code = 400, detail = f"{key} must be between {minimum} and {maximum}"
            )
    # Server-controlled, not client tunable. OFF by default; opt in via
    # UNSLOTH_RESEARCH_AUTO_SCRAPE=1. Injected only when enabled, so a default run's budgets stay
    # byte-identical to legacy.
    from core.research_runs import _auto_scrape_default

    _auto_scrape = _auto_scrape_default()
    if _auto_scrape > 0:
        budgets["maxAutoScrape"] = _auto_scrape
    try:
        website_policy = normalize_website_policy(payload.websitePolicy)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    return {
        "model": model,
        "inferenceRequest": request,
        "ragScope": rag_scope,
        "budgets": budgets,
        "websitePolicy": website_policy,
        "instructions": (payload.instructions or "").strip(),
    }


@router.post("", status_code = 202)
async def create_research_run(
    payload: CreateResearchRun,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    thread = get_chat_thread(payload.threadId)
    if thread is None:
        raise HTTPException(status_code = 404, detail = "Thread not found")
    user_message = get_chat_message(payload.threadId, payload.userMessageId)
    if user_message is None or user_message.get("role") != "user":
        raise HTTPException(
            status_code = 400, detail = "userMessageId must identify a user message in the thread"
        )
    if not content_to_text(user_message.get("content")).strip():
        raise HTTPException(
            status_code = 400,
            detail = "Deep research requires a user message with non-empty text",
        )
    if db.has_thread_claim(payload.threadId):
        raise HTTPException(
            status_code = 409,
            detail = "This thread already has a Deep Research run",
        )
    config = _sanitize_config(payload, thread)
    run_id = uuid.uuid4().hex
    assistant_id = payload.assistantMessageId
    try:
        run = db.create_run(
            run_id = run_id,
            owner_subject = current_subject,
            thread_id = payload.threadId,
            user_message_id = payload.userMessageId,
            assistant_message_id = assistant_id,
            config = config,
        )
    except db.ResearchConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is not None:
        supervisor.note_request_port(request)
        supervisor.wake()
    return run


@router.get("/active")
async def active_research_runs(
    thread_id: str = Query(alias = "threadId"), current_subject: str = Depends(get_current_subject)
):
    return {
        "runs": db.list_active(thread_id),
        "hasRun": db.has_thread_claim(thread_id),
    }


@router.get("/{run_id}")
async def get_research_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    return _require_run(run_id)


@router.put("/{run_id}/plan")
async def update_research_plan(
    run_id: str,
    payload: UpdatePlan,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    try:
        db.set_plan(run_id, payload.plan.model_dump(), payload.expectedRevision)
    except (db.ResearchConflictError, KeyError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    run = _require_run(run_id)
    _sync_assistant(run)
    return run


@router.post("/{run_id}/approve")
async def approve_research_plan(
    run_id: str,
    payload: ApprovePlan,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    try:
        db.approve(run_id, payload.planRevision, payload.planHash)
    except (db.ResearchConflictError, KeyError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is not None:
        supervisor.note_request_port(request)
        supervisor.wake()
    run = _require_run(run_id)
    _sync_assistant(run)
    return run


@router.post("/{run_id}/cancel")
async def cancel_research_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    status = db.request_cancel(run_id)
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is not None and status == "cancelling":
        supervisor.cancel(run_id)
    run = _require_run(run_id)
    _sync_assistant(run)
    return run


@router.post("/{run_id}/retry")
async def retry_research_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    try:
        db.retry(run_id)
    except (db.ResearchConflictError, KeyError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is not None:
        supervisor.note_request_port(request)
        supervisor.wake()
    run = _require_run(run_id)
    _sync_assistant(run)
    return run


@router.get("/{run_id}/events")
async def research_events(
    run_id: str,
    request: Request,
    after: int | None = Query(None, ge = 0),
    last_event_id: str | None = Header(None, alias = "Last-Event-ID"),
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    header_after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    cursor = max(after or 0, header_after)

    async def stream():
        nonlocal cursor
        while True:
            events = await asyncio.to_thread(
                db.wait_for_events,
                run_id,
                cursor,
                15,
            )
            snapshot = await asyncio.to_thread(db.get_run, run_id)
            if snapshot is None:
                return
            for event in events:
                cursor = int(event["seq"])
                event_data = dict(event["data"])
                event_data["createdAt"] = event["createdAt"]
                if event["type"] not in _DELTA_ONLY_EVENTS:
                    event_data["run"] = snapshot
                data = json.dumps(event_data, separators = (",", ":"), ensure_ascii = False)
                yield f"id: {cursor}\nevent: {event['type']}\ndata: {data}\n\n"
            if snapshot["status"] in db.TERMINAL_STATUSES and cursor >= int(
                snapshot["lastEventSeq"]
            ):
                return
            if await request.is_disconnected():
                return
            if not events:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
