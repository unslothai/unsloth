# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated durable inline Deep Research API."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from auth.authentication import get_current_subject
from core.inference.message_content import message_text_with_pastes
from core.inference.web_access_policy import normalize_website_policy
from storage import research_runs_db as db
from core.inference.providers import provider_runs_local_tools
from storage import providers_db
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
# Zero is the unlimited sentinel, so a finite value only has to cover the longest run anyone
# would set: a year reads back in the 400, unlike a float-max ceiling.
_MIN_FINITE_MODEL_TIMEOUT_SECONDS = 10
_MAX_FINITE_MODEL_TIMEOUT_SECONDS = 365 * 24 * 3600
_DELTA_ONLY_EVENTS = {
    "reasoning.updated",
    "report.updated",
    "phase.progress",
    "phase.started",
    "phase.ended",
}
# Dedicated to the blocking event wait so open streams cannot exhaust the default executor.
_EVENT_WAIT_EXECUTOR = ThreadPoolExecutor(max_workers = 32, thread_name_prefix = "research-events")


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
    question: str | None = Field(default = None, max_length = 2000)

    @field_validator("budgets", mode = "before")
    @classmethod
    def _reject_boolean_budgets(cls, value: Any) -> Any:
        # bool is an int subclass, so False would coerce to the 0 "unlimited" sentinel and
        # silently drop a deadline. Reject it here: by the time the field is typed it is 0.
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, bool):
                    raise ValueError(f"{key} must be an integer, not a boolean")
        return value


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
            expected_attempt = int(run.get("retryCount") or 0),
        )
        if created:
            return
        if not message_id:
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
        expected_research_run_id = run["id"],
        expected_research_attempt = int(run.get("retryCount") or 0),
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
            detail = "Research inference routing cannot override endpoints or tool catalogs",
        )
    allowed = {
        "model",
        "providerId",
        "providerType",
        "externalModel",
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
    provider_type = request.get("providerType")
    provider_id = request.get("providerId")
    external_model = request.get("externalModel")
    external_requested = any(
        value is not None for value in (provider_type, provider_id, external_model)
    )
    if external_requested:
        # A saved connection is still mandatory: the run is durable, so an
        # inline key would have to be persisted, and _is_sensitive_key exists to
        # stop exactly that. Only the provider-type allowlist is widened.
        if (
            not provider_runs_local_tools(provider_type)
            or not isinstance(provider_id, str)
            or not provider_id.strip()
            or not isinstance(external_model, str)
            or not external_model.strip()
        ):
            raise HTTPException(
                status_code = 400,
                detail = "Durable research requires a saved connection whose provider supports Unsloth tools",
            )
        provider = providers_db.get_provider(provider_id)
        if provider is None:
            raise HTTPException(status_code = 404, detail = "Provider config not found")
        # The saved row is the source of truth for routing, so validate against
        # it rather than against the type the client sent. A self-hosted
        # connection is stored under the backend "openai" type but surfaced to
        # the UI as "custom" / "vllm" / "ollama" / "llama_cpp", and the composer
        # offers research for those aliases because their registry entries
        # declare Unsloth tools. Comparing the two for equality therefore 400s
        # exactly the connections this path exists to serve, while the ordinary
        # inference route already overrides the type from the row.
        saved_provider_type = provider["provider_type"]
        if not provider_runs_local_tools(saved_provider_type) or not provider["is_enabled"]:
            raise HTTPException(
                status_code = 400,
                detail = "Durable research requires an enabled connection whose provider supports Unsloth tools",
            )
        request["providerType"] = saved_provider_type

    # Mirrors the ragScope guard below. Every allowed field is a scalar, but "model" is
    # stringified, so {"auth": "sk-..."} would slip past the sensitive-key scan (inner key
    # unlisted) into the durable config as the model id.
    if any(isinstance(value, (dict, list, tuple)) for value in request.values()):
        raise HTTPException(status_code = 400, detail = "Invalid inferenceRequest value")
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
        # Every ragScope field is a scalar. A nested container evades the sensitive-key scan when
        # its inner keys are unlisted (e.g. {"kb_id": {"auth": "sk-..."}}) and would reach
        # retrieval code expecting a scalar scope id, so reject non-scalars outright.
        non_scalar = any(isinstance(value, (dict, list, tuple)) for value in rag_scope.values())
        if unknown_rag or non_scalar or _contains_sensitive_key(rag_scope):
            raise HTTPException(status_code = 400, detail = "Unsupported or sensitive ragScope field")
    budgets = {
        "maxSteps": 12,
        "maxSources": 40,
        "modelTimeoutSeconds": 900,
        "toolTimeoutSeconds": 120,
        "firstOutputTimeoutSeconds": 120,
    }
    for key, value in (payload.budgets or {}).items():
        if key not in budgets:
            raise HTTPException(status_code = 400, detail = f"Unsupported budget: {key}")
        budgets[key] = int(value)
    limits = {
        "maxSteps": (1, _MAX_PLAN_STEPS),
        "maxSources": (1, 100),
        # Zero disables the total wall-clock deadline. Per-output stall deadlines still apply.
        "modelTimeoutSeconds": (
            _MIN_FINITE_MODEL_TIMEOUT_SECONDS,
            _MAX_FINITE_MODEL_TIMEOUT_SECONDS,
        ),
        "toolTimeoutSeconds": (5, 600),
        # Same range as its parent: slow CPU and offloaded models need minutes to first token.
        "firstOutputTimeoutSeconds": (10, 3600),
    }
    for key, (minimum, maximum) in limits.items():
        # The sentinel is not a short timeout, so it skips the floor rather than lowering it.
        if key == "modelTimeoutSeconds" and budgets[key] == 0:
            continue
        if not minimum <= budgets[key] <= maximum:
            allowed = f"between {minimum} and {maximum}"
            if key == "modelTimeoutSeconds":
                allowed = f"0 (unlimited) or {allowed}"
            raise HTTPException(status_code = 400, detail = f"{key} must be {allowed}")
    # Server-controlled, not client tunable. OFF unless UNSLOTH_RESEARCH_AUTO_SCRAPE=1, and
    # injected only when enabled, so a default run's budgets stay byte-identical to legacy.
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
        "question": (payload.question or "").strip(),
    }


@router.post("", status_code = 202)
def create_research_run(
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
    # A handed-off question counts as the text. An image-, audio- or video-only send is a
    # normal composer turn, and a multimodal model that reads one and calls deep_research
    # passes the question it wrote; the worker researches config.question, so refusing here
    # on the message's own (empty) text ends an otherwise complete handoff in a toast.
    if not message_text_with_pastes(user_message).strip() and not (payload.question or "").strip():
        raise HTTPException(
            status_code = 400,
            detail = "Deep research requires a user message with non-empty text",
        )
    config = _sanitize_config(payload, thread)
    try:
        if db.has_thread_claim(payload.threadId):
            # The thread's one run was stopped, so it is re-pointed at this question rather
            # than refusing every later one in the chat.
            run = db.rebind_cancelled(
                thread_id = payload.threadId,
                user_message_id = payload.userMessageId,
                assistant_message_id = payload.assistantMessageId,
                config = config,
            )
            if run is None:
                raise HTTPException(
                    status_code = 409,
                    detail = "This thread already has a Deep Research run",
                )
        else:
            run = db.create_run(
                run_id = uuid.uuid4().hex,
                owner_subject = current_subject,
                thread_id = payload.threadId,
                user_message_id = payload.userMessageId,
                assistant_message_id = payload.assistantMessageId,
                config = config,
            )
    except db.ResearchConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        # The thread can be deleted between the check above and this insert, and the foreign key
        # then fails. Report it gone rather than as a server fault.
        raise HTTPException(status_code = 404, detail = "Thread not found") from exc
    if run is None:
        raise HTTPException(status_code = 404, detail = "Thread not found")
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is not None:
        supervisor.note_request_port(request)
        supervisor.wake()
    return run


@router.get("/active")
def active_research_runs(
    thread_id: str = Query(alias = "threadId"), current_subject: str = Depends(get_current_subject)
):
    return {
        "runs": db.list_active(thread_id),
        "hasRun": db.research_spent(thread_id),
    }


@router.get("/{run_id}")
def get_research_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    return _require_run(run_id)


@router.put("/{run_id}/plan")
def update_research_plan(
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
def approve_research_plan(
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
def cancel_research_run(
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
def retry_research_run(
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


# POST too: proxies that stream /v1/chat/completions still buffer a streamed GET until it closes.
@router.post("/{run_id}/events")
# Separate registration, out of the schema: one api_route would give both verbs one operationId.
@router.get("/{run_id}/events", include_in_schema = False)
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
        loop = asyncio.get_running_loop()
        while True:
            # off the default executor: parked followers there starved the run's own db writes.
            events = await loop.run_in_executor(
                _EVENT_WAIT_EXECUTOR,
                db.wait_for_events,
                run_id,
                cursor,
                15,
            )
            # Not the wait executor: this read is short, and queueing it behind parked waits
            # would delay every follower once the pool is full.
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
