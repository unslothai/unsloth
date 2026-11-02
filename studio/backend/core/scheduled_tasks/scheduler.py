# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs scheduled tasks while Studio is up: one loop, one run at a time, each saved as a chat."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from starlette.requests import Request

from core.scheduled_tasks.schedule import next_run_after, parse_schedule, resolve_timezone
from core.training.account_jobs import job_accounts
from loggers import get_logger
from storage import scheduled_tasks_db as db
from utils.account_context import AccountContext, arun_as, current_account_id, run_as

logger = get_logger(__name__)

# Longest sleep: asyncio's clock stops while a computer sleeps, so a run due meanwhile starts at most this late.
_POLL_SECONDS = 60.0
# Floor after a pass that claimed nothing, so an unclaimable due task cannot spin the loop.
_MIN_SLEEP_SECONDS = 1.0
_HEARTBEAT_SECONDS = 30.0
_STALE_RUN_MS = 2 * 60 * 1000
_RUN_TIMEOUT_SECONDS = 20 * 60
_CANCEL_GRACE_SECONDS = 10.0
_STOP_GRACE_SECONDS = 5.0
_SETTLE_ATTEMPTS = 3
_SETTLE_RETRY_SECONDS = 1.0
_ERROR_TEXT_LIMIT = 500
# Studio counts as in use this long after its last inference request; a run then asks before swapping.
_AWAY_SECONDS = 5 * 60
_YIELD_SECONDS = 5 * 60
_YIELD_POLL_SECONDS = 2.0

_schedulers: "weakref.WeakSet[ScheduledTaskScheduler]" = weakref.WeakSet()


class RunFailed(Exception):
    """A run that ends without a chat. ``code`` is a reason the UI words itself."""

    def __init__(
        self,
        code: Optional[str] = None,
        message: Optional[str] = None,
    ):
        super().__init__(message or code or "failed")
        self.code = code
        self.message = message


@dataclass
class Answer:
    content: str
    reasoning: str
    model: Optional[str]
    # What Chat selects to continue the conversation; `model` is the reply's display name.
    checkpoint: Optional[str] = None
    gguf_variant: Optional[str] = None
    provider: Optional[dict] = None


@dataclass
class _Swap:
    """A run that loads its task's model in place of what was loaded, put back afterwards."""

    previous: Optional[tuple]
    model: Optional[str] = None
    subject: Optional[str] = None
    timezone: str = "UTC"
    task_id: Optional[str] = None
    # llama_keepwarm.attended_stamp() at the swap: a different one later means someone came back.
    attended: Optional[float] = None
    # A Transformers or MLX model in the primary has no GGUF identity: it is loaded back by name.
    previous_model: Optional[str] = None


@dataclass
class SwapHolder:
    """A swap left for the next run, so back-to-back runs on one model load it once."""

    pending: Optional[_Swap] = None


@dataclass
class _ActiveRun:
    account_id: str
    task_id: str
    run_id: str
    cancel_event: threading.Event


def cancel_account_runs(account_id: str) -> None:
    for scheduler in list(_schedulers):
        scheduler.cancel_account(account_id)


def retire_account_scheduled_tasks(account: AccountContext) -> None:
    cancel_account_runs(account.account_id)


def next_run_for(task: dict, after_ms: int) -> Optional[int]:
    return next_run_after(
        parse_schedule(task["schedule"]),
        task["timezone"],
        after_ms,
        anchor_ms = task.get("nextRunAt"),
    )


def system_prompt(task: dict, now_ms: int) -> str:
    try:
        zone = resolve_timezone(task["timezone"])
    except ValueError:
        zone = timezone.utc
    local = datetime.fromtimestamp(now_ms / 1000, tz = zone)
    when = f"{local:%A}, {local:%B} {local.day}, {local.year} at {local:%H:%M}"
    return (
        f"This is a scheduled task the user set up in Unsloth Studio, running on {when} "
        f"({task['timezone']}). The user is not watching this chat and will read your reply "
        "later, so complete the task directly and do not ask follow-up questions."
    )


def _run_instructions(task: dict, now_ms: int) -> str:
    from core.agents_md import agents_md_text
    from storage import studio_db

    project = studio_db.get_chat_project(task["projectId"]) if task.get("projectId") else None
    agents_md = agents_md_text(project).strip()
    instructions = project["instructions"].strip() if project and not project["archived"] else ""
    parts = [
        f"<agents_md>\n{agents_md}\n</agents_md>" if agents_md else "",
        f"<project_instructions>\n{instructions}\n</project_instructions>" if instructions else "",
        system_prompt(task, now_ms),
    ]
    return "\n\n".join(part for part in parts if part)


def _loop_cancelled(current: Optional[asyncio.Task], stopping: bool) -> bool:
    """Whether the scheduler's own task is being cancelled. ``Task.cancelling()`` is 3.11+; before it only stop() does."""
    if current is None:
        return False
    cancelling = getattr(current, "cancelling", None)
    return bool(cancelling()) if cancelling is not None else stopping


def _producer_request(
    app: Any,
    run_id: str,
    cancel_event: threading.Event,
    tz: str,
    loaded_only: bool = True,
) -> Request:
    from utils.current_date_prompt_settings import CURRENT_DATE_TIMEZONE_HEADER
    from utils.keyless_api_access import KEYLESS_ADMISSION_STATE_KEY
    from routes.inference import (
        LOADED_MODELS_ONLY_SCOPE_KEY,
        REQUIRE_TEXT_HEADER,
        disable_auto_download_for_request,
    )

    path = f"/api/scheduled-tasks/runs/{run_id}"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii", "ignore"),
        "query_string": b"",
        "headers": [
            (CURRENT_DATE_TIMEZONE_HEADER.encode("latin-1"), tz.encode("latin-1", "ignore")),
            (REQUIRE_TEXT_HEADER.lower().encode("latin-1"), b"1"),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {
            "generation_cancel_event": cancel_event,
            "api_monitor_via_api_key": False,
            # Created by a signed-in user, so never classified as keyless loopback traffic.
            KEYLESS_ADMISSION_STATE_KEY: False,
        },
    }

    if loaded_only:
        # The kept model it names may be unloaded before routing: never load it back.
        scope[LOADED_MODELS_ONLY_SCOPE_KEY] = True
    disable_auto_download_for_request(scope)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _error_text(detail: Any) -> str:
    if isinstance(detail, dict):
        nested = detail.get("error")
        if isinstance(nested, dict):
            detail = nested
        detail = detail.get("message") or detail.get("detail") or detail.get("error") or detail
    text = detail if isinstance(detail, str) else json.dumps(detail, default = str)
    return text.strip()[:_ERROR_TEXT_LIMIT] or "The model could not answer."


def _chat_model_loaded() -> bool:
    from routes.inference import _loaded_slot_ident
    return _loaded_slot_ident() is not None


def _speech_only(llama: Any, orchestrator: Any) -> bool:
    """A loaded audio model a text-only run cannot use: TTS answers with speech, Whisper refuses a request without audio."""
    if getattr(llama, "is_loaded", False):
        return bool(getattr(llama, "_is_audio", False))
    active = getattr(orchestrator, "active_model_name", None)
    info = (getattr(orchestrator, "models", None) or {}).get(active) or {}
    return bool(info.get("is_audio"))


def _slot_model(slot: Any) -> Optional[str]:
    if _speech_only(slot.llama, slot.orchestrator):
        return None
    if getattr(slot.llama, "is_loaded", False) and getattr(slot.llama, "model_identifier", None):
        return str(slot.llama.model_identifier)
    name = getattr(slot.orchestrator, "active_model_name", None)
    return str(name) if name else None


def _sole_kept_model() -> Optional[str]:
    """With no primary this account may use for text, its one model kept alongside: named, a run routes to it without a load."""
    from core.inference import model_slots
    from hub.services.models import account_access
    from routes.inference import _loaded_slot_ident, _peek_inference_backend, get_llama_cpp_backend

    primary = _loaded_slot_ident()
    if (
        primary is not None
        and not (
            account_access.managed_account() and account_access.resident_hidden("chat", primary)
        )
        and not _speech_only(get_llama_cpp_backend(), _peek_inference_backend())
    ):
        return None
    names = [name for name in map(_slot_model, model_slots.visible()) if name]
    return names[0] if len(names) == 1 else None


def _primary_serves(model: str) -> bool:
    from routes.inference import _loaded_identity_satisfies, _loaded_satisfies
    return _loaded_satisfies(model) or _loaded_identity_satisfies(model)


def _model_is_loaded(model: str) -> bool:
    """Loaded as the primary or alongside it; the producer routes to whichever serves it."""
    from core.inference import model_slots

    if _primary_serves(model):
        return True
    candidates = model_slots.visible()
    return (
        bool(candidates)
        and model_slots.serving_slot(model, candidates, _primary_serves) is not None
    )


# Levels the request accepts; a template may list others, which a run never sends.
_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "max", "xhigh")


def _reasoning_fields(effort: Optional[str], connection: bool, on_loaded_model: bool) -> dict:
    """The request fields for a task's thinking effort; None keeps the model's default."""
    if effort is None:
        return {}
    relative = {"off": "none", "lowest": "low", "highest": "high"}
    if connection:
        return {"reasoning_effort": relative.get(effort, effort)}
    if effort not in relative:
        return {"enable_thinking": True, "reasoning_effort": effort}
    backend = None
    if on_loaded_model:
        from routes.inference import get_llama_cpp_backend
        backend = get_llama_cpp_backend()
        if not getattr(backend, "is_loaded", False):
            backend = None
    if backend is None:
        if effort == "off":
            return {"enable_thinking": False}
        return {
            "enable_thinking": True,
            "reasoning_effort": "low" if effort == "lowest" else "high",
        }
    if not backend.supports_reasoning:
        return {}
    style = backend.reasoning_style
    if effort == "off":
        if backend.reasoning_always_on or style == "reasoning_effort":
            return {}
        return {"enable_thinking": False}
    if style == "enable_thinking":
        return {"enable_thinking": True}
    levels = [
        level
        for level in (backend.reasoning_effort_levels or ["low", "medium", "high"])
        if level in _EFFORT_LEVELS
    ] or ["low", "high"]
    fields = {"reasoning_effort": levels[0] if effort == "lowest" else levels[-1]}
    if style == "enable_thinking_effort":
        fields["enable_thinking"] = True
    return fields


def _model_fields(model: Optional[str], may_load: bool) -> tuple[dict, Optional[dict], bool]:
    """The request fields that pick the task's model, and whether it must be loaded; raises RunFailed."""
    from routes.inference import _connection_model_ref
    from storage import providers_db
    from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled

    if not model:
        return {}, None, False
    connection = _connection_model_ref(model)
    if connection is not None:
        provider_id, external_model = connection
        config = providers_db.get_provider(provider_id)
        if config is None or not config["is_enabled"]:
            raise RunFailed("connection_unavailable")
        # The proxy relays a stream:false provider body as is, so stream and collect the chunks.
        return (
            {
                "model": model,
                "provider_id": provider_id,
                "external_model": external_model,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            config,
            False,
        )
    if _model_is_loaded(model):
        return {"model": model}, None, False
    # Loading replaces Chat's model, so only with the user's yes; otherwise RunFailed asks for it.
    if may_load or get_openai_auto_switch_enabled():
        return {"model": model}, None, True
    raise RunFailed("model_not_loaded")


async def _plan_swap(task: dict, model: str, unattended: bool, carried: Optional[_Swap]) -> _Swap:
    """What a load must put back; an unattended run swaps only a restorable GGUF while Studio is idle."""
    from core.inference.llama_keepwarm import attended_stamp, idle_for, loaded_identity
    from routes.inference import (
        _loaded_slot_ident,
        _resident_npu_model,
        _training_is_active,
        get_inference_backend,
    )
    from hub.services.models import account_access

    if unattended:
        unrestorable = _resident_npu_model() is not None or getattr(
            await asyncio.to_thread(get_inference_backend), "active_model_name", None
        )
        # Another account's model, which this account could not load back afterwards.
        resident = _loaded_slot_ident()
        foreign = resident is not None and account_access.resident_hidden("chat", resident)
        if (
            unrestorable
            or foreign
            or not idle_for(_AWAY_SECONDS)
            or await asyncio.to_thread(_training_is_active)
        ):
            raise RunFailed("model_busy")
    # A model loaded since the carried swap is the user's latest choice, and the one to put back.
    keep_carried = carried is not None and carried.model and _model_is_loaded(carried.model)
    previous = carried.previous if keep_carried else loaded_identity()
    previous_model = carried.previous_model if keep_carried else None
    if not keep_carried and previous is None and _resident_npu_model() is None:
        previous_model = getattr(
            await asyncio.to_thread(get_inference_backend), "active_model_name", None
        )
    return _Swap(
        previous,
        model = model,
        subject = task["ownerSubject"],
        timezone = task["timezone"],
        task_id = task["id"],
        attended = attended_stamp(),
        previous_model = previous_model,
    )


def _require_still_away(swap: _Swap) -> None:
    """Raise model_busy once someone came back since the swap was planned: an inference request ended,
    a model load landed, or a request is running beside this run."""
    from core.inference.llama_keepwarm import attended_stamp, other_inference_request_count
    if attended_stamp() != swap.attended or other_inference_request_count(
        current_request_counted = True
    ):
        raise RunFailed("model_busy")


async def _restore_after_swap(swap: _Swap, app: Any) -> None:
    """Put back what the run's load replaced, unless someone used or loaded a model since. Best effort."""
    from core.inference.llama_keepwarm import (
        attended_stamp,
        end_in_process_request,
        idle_for,
        loaded_identity,
        stash_for_reload,
        unattended_activity,
    )
    from models.inference import UnloadRequest
    from routes.inference import (
        _RELOAD_ONLY_MODEL,
        _maybe_auto_switch_model,
        _unload_model_impl,
        allow_model_switch_for_request,
        get_inference_backend,
    )

    try:
        if not idle_for(0) or attended_stamp() != swap.attended:
            return
        with unattended_activity():
            current = loaded_identity()
            if not _primary_serves(swap.model) and _model_is_loaded(swap.model):
                # Loaded beside the primary (Keep multiple models loaded with a speech model in it): the primary
                # was never replaced, so only the task model goes.
                # Its own task: unloading routes to the slot through a context variable, which must not stick to the loop.
                await asyncio.ensure_future(
                    _unload_model_impl(UnloadRequest(model_path = swap.model), swap.subject)
                )
                return
            if _model_is_loaded(swap.model):
                # A Transformers or MLX task model has no GGUF identity; it holds the primary all the same.
                target = (
                    current[0]
                    if current is not None
                    else getattr(
                        await asyncio.to_thread(get_inference_backend), "active_model_name", None
                    )
                )
                if target:
                    await _unload_model_impl(UnloadRequest(model_path = target), swap.subject)
            elif current is not None:
                return
            if swap.previous is None and not swap.previous_model:
                return
            if swap.previous is not None:
                stash_for_reload(swap.previous)
            # Named for a non-GGUF model, which must not be held to what is loaded now.
            request = _producer_request(
                app,
                f"{swap.task_id}:restore",
                threading.Event(),
                swap.timezone,
                loaded_only = swap.previous is not None,
            )
            allow_model_switch_for_request(request.scope)
            restored = False
            try:
                await _maybe_auto_switch_model(
                    _RELOAD_ONLY_MODEL if swap.previous is not None else swap.previous_model,
                    request,
                    swap.subject,
                )
                restored = True
            finally:
                # No middleware ends this in-process request; its admission would block previews.
                end_in_process_request(request.scope, answered = restored)
    except Exception as exc:  # noqa: BLE001 - the run's own outcome stands
        logger.warning(
            "Scheduled task %s could not restore the earlier model: %s", swap.task_id, exc
        )


def _pins_local_model(model: Optional[str]) -> bool:
    from routes.inference import _connection_model_ref
    return bool(model) and _connection_model_ref(model) is None


async def generate_answer(
    app: Any,
    task: dict,
    run: dict,
    cancel_event: threading.Event,
    holder: Optional[SwapHolder] = None,
) -> Answer:
    """Ask the chat model to do the task, text only (tools never run unattended)."""
    from fastapi import HTTPException

    from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError
    from core.inference.llama_keepwarm import (
        InferenceActivityReservation,
        end_in_process_request,
        unattended_activity,
    )
    from fastapi.responses import StreamingResponse

    from models.inference import ChatCompletionRequest
    from routes.inference import (
        _chat_completion_from_connection_stream,
        _relay_split_thinking,
        allow_model_switch_for_request,
        load_alongside_for_request,
        produce_openai_chat_completions,
        require_for_model_switch,
    )

    carried = holder.pending if holder is not None else None
    if carried is not None and not _pins_local_model(task.get("model")):
        holder.pending = None
        await _restore_after_swap(carried, app)
        carried = None
    model_fields, provider, allow_switch = _model_fields(
        task.get("model"), bool(task.get("allowModelLoad") or run.get("allowModelLoad"))
    )
    unattended = run.get("source") == "schedule" and not run.get("allowModelLoad")
    swap = await _plan_swap(task, task["model"], unattended, carried) if allow_switch else None
    if holder is not None:
        holder.pending = swap or carried
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RUN_TIMEOUT_SECONDS
    request = _producer_request(
        app, run["id"], cancel_event, task["timezone"], loaded_only = not task.get("model")
    )
    if allow_switch:
        allow_model_switch_for_request(request.scope)
        # Keep multiple models loaded: the task model goes beside the user's when it fits, as from Chat's picker.
        load_alongside_for_request(request.scope)
    away_swap = swap if unattended else None
    refused: list[bool] = []
    if away_swap is not None:
        # The switch rechecks under its gates; it can only answer with an HTTP error, read back as model_busy.
        def still_away() -> None:
            try:
                _require_still_away(away_swap)
            except RunFailed:
                refused.append(True)
                raise HTTPException(status_code = 409, detail = "model_busy") from None

        require_for_model_switch(request.scope, still_away)
    with unattended_activity():
        activity = InferenceActivityReservation()
        activity.reserve()
        producer: Optional[asyncio.Future] = None
        response = None
        try:
            # The deadline covers the wait for a model load or unload to let go too: one that never
            # does must not keep the scheduler from every other task.
            try:
                await asyncio.wait_for(activity.start(cancel_event), timeout = _RUN_TIMEOUT_SECONDS)
            except TimeoutError:
                raise RunFailed("timeout") from None
            if cancel_event.is_set():
                raise RunFailed("cancelled")
            if away_swap is not None:
                # Admission can wait out a load the user started, so the idle check before it may be stale.
                _require_still_away(away_swap)
            # Built once admitted: a model kept alongside is chosen as it stands now, not before a load or unload.
            if not task.get("model") and (kept := _sole_kept_model()):
                model_fields = {"model": kept}
            # A project folder on a stalled mount must not keep the scheduler from every other task.
            try:
                instructions = await asyncio.wait_for(
                    asyncio.to_thread(_run_instructions, task, db.now_ms()),
                    timeout = max(0.0, deadline - loop.time()),
                )
            except TimeoutError:
                raise RunFailed("timeout") from None
            payload = ChatCompletionRequest.model_validate(
                {
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    "stream": False,
                    "cancel_id": run["id"],
                    # Both opt-outs: --enable-tools overrides enable_tools, and an omitted enabled_tools
                    # means every built-in, python and terminal included.
                    "tool_choice": "none",
                    "enabled_tools": [],
                    **model_fields,
                    **_reasoning_fields(
                        task.get("reasoningEffort"),
                        provider is not None,
                        # Only the primary's levels are known here: a model kept beside it gets the generic fields.
                        on_loaded_model = not allow_switch
                        and (
                            not model_fields.get("model") or _primary_serves(model_fields["model"])
                        ),
                    ),
                }
            )

            serving: dict = {}

            async def produce() -> Any:
                # Settled in this task: slot routing lives in its context, which the caller never sees.
                reply = None
                try:
                    reply = await produce_openai_chat_completions(
                        payload,
                        request,
                        task["ownerSubject"],
                        cancel_on_disconnect = False,
                    )
                    if provider is not None and isinstance(reply, StreamingResponse):
                        reply.body_iterator = _relay_split_thinking(
                            reply.body_iterator,
                            payload.stop,
                            parse_thinking = payload.enable_thinking is not False,
                        )
                        reply = await _chat_completion_from_connection_stream(reply, payload.model)
                    elif provider is None:
                        serving.update(_serving_checkpoint())
                    # Checked before the request ends, while nobody else can have loaded a model yet.
                    if allow_switch and int(getattr(reply, "status_code", 200)) < 400:
                        serving["loaded"] = _model_is_loaded(payload.model)
                    return reply
                finally:
                    end_in_process_request(
                        request.scope,
                        answered = reply is not None
                        and int(getattr(reply, "status_code", 200)) < 400,
                    )

            producer = asyncio.ensure_future(produce())
            done, _ = await asyncio.wait({producer}, timeout = max(0.0, deadline - loop.time()))
            if not done:
                raise RunFailed("timeout")
            try:
                response = producer.result()
            except HTTPException as exc:
                if refused:
                    raise RunFailed("model_busy") from None
                if allow_switch and not _model_is_loaded(payload.model):
                    raise RunFailed("model_load_failed", _error_text(exc.detail)) from None
                # A named kept model answered for itself, so its refusal is not "nothing loaded",
                # unless it was unloaded before the request reached it.
                if provider is None and not task.get("model"):
                    named = "model" in payload.model_fields_set
                    if (not named and not _chat_model_loaded()) or (
                        named and _sole_kept_model() is None
                    ):
                        raise RunFailed("no_model") from None
                raise RunFailed(None, _error_text(exc.detail)) from None
            except GpuBusyForAnotherAccountError as exc:
                raise RunFailed(None, _error_text(exc.as_http_exception().detail)) from None
            if refused:
                raise RunFailed("model_busy")
            if serving.get("loaded") is False:
                raise RunFailed("model_load_failed")
        finally:
            if producer is not None and not producer.done():
                # A timeout or shutdown: stop the generation itself, not just this wait.
                cancel_event.set()
                producer.cancel()
                await asyncio.wait({producer}, timeout = _CANCEL_GRACE_SECONDS)

            def release(*_: Any) -> None:
                activity.finish(request.scope)
                if producer is None:
                    end_in_process_request(request.scope, answered = False)

            if producer is not None and not producer.done():
                # Its worker has not unwound yet: the model stays busy until it does.
                producer.add_done_callback(release)
            else:
                release()
            if holder is None and swap is not None and not cancel_event.is_set():
                await _restore_after_swap(swap, app)
            elif (
                holder is not None
                and carried is not None
                and holder.pending is swap
                and not _model_is_loaded(swap.model)
                and _model_is_loaded(carried.model)
            ):
                # This load failed and left the carried model in place: that swap still has to be put back.
                holder.pending = carried
    if cancel_event.is_set():
        raise RunFailed("cancelled")
    body = getattr(response, "body", None)
    if body is None:
        raise RunFailed(None, "The model did not return a reply.")
    data = json.loads(bytes(body))
    if int(getattr(response, "status_code", 200)) >= 400:
        raise RunFailed(None, _error_text(data))
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content.strip():
        raise RunFailed("empty")
    return Answer(
        content = content,
        reasoning = message.get("reasoning_content") or "",
        model = data.get("model") or None,
        checkpoint = serving.get("checkpoint"),
        gguf_variant = serving.get("gguf_variant"),
        provider = provider,
    )


def _serving_checkpoint() -> dict:
    """The model that answered, as Chat names it: read in the producer's task, where its slot was routed."""
    from routes.inference import _loaded_slot_ident, get_llama_cpp_backend

    llama = get_llama_cpp_backend()
    variant = getattr(llama, "hf_variant", None) if getattr(llama, "is_loaded", False) else None
    return {"checkpoint": _loaded_slot_ident(), "gguf_variant": variant or None}


def _project_or_none(project_id: Optional[str]) -> Optional[str]:
    from storage import studio_db
    return project_id if project_id and studio_db.get_chat_project(project_id) else None


def save_chat(task: dict, run: dict, answer: Answer, started_at: int, finished_at: int) -> str:
    from storage import studio_db

    thread_id = db.chat_id_for_run(run["id"])
    user_message_id = str(uuid.uuid4())
    # After the prompt even within one millisecond, which the message order relies on.
    answered_at = max(finished_at, started_at + 1)
    model = answer.model or ""
    checkpoint = answer.checkpoint or model
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": task["title"],
            "modelType": "base",
            "modelId": checkpoint,
            "modelGgufVariant": answer.gguf_variant,
            "createdAt": started_at,
            "updatedAt": answered_at,
            "scheduledTaskId": task["id"],
            "projectId": _project_or_none(task.get("projectId")),
            "archived": bool(task.get("archiveSuccessful")),
        }
    )
    parts: list[dict] = []
    source = {"taskId": task["id"], "runId": run["id"]}
    metadata: dict = {
        "scheduledTask": source,
        "responseDetails": {
            "modelId": checkpoint,
            "modelLabel": model,
            "responseModelId": model,
            **(
                {
                    "providerId": answer.provider["id"],
                    "providerName": answer.provider["display_name"],
                    "providerType": answer.provider["provider_type"],
                }
                if answer.provider
                else {"providerName": "Local model", "providerType": "local"}
            ),
            "startedAt": started_at,
            "finishedAt": answered_at,
            "durationMs": answered_at - started_at,
            "toolCalls": [],
        },
    }
    if answer.reasoning.strip():
        parts.append({"type": "reasoning", "text": answer.reasoning})
        seconds = max(0, round((answered_at - started_at) / 1000))
        metadata.update(reasoningDuration = seconds, reasoningDurations = [seconds])
    parts.append({"type": "text", "text": answer.content})
    try:
        studio_db.sync_chat_messages(
            thread_id,
            [
                {
                    "id": user_message_id,
                    "threadId": thread_id,
                    "parentId": None,
                    "role": "user",
                    "content": [{"type": "text", "text": task["prompt"]}],
                    "metadata": {"scheduledTask": source},
                    "createdAt": started_at,
                },
                {
                    "id": str(uuid.uuid4()),
                    "threadId": thread_id,
                    "parentId": user_message_id,
                    "role": "assistant",
                    "content": parts,
                    "metadata": metadata,
                    "createdAt": answered_at,
                },
            ],
        )
    except Exception:
        # Never leave an empty chat behind.
        _discard_chat(thread_id)
        raise
    return thread_id


def _discard_chat(thread_id: str) -> None:
    from storage import studio_db
    try:
        studio_db.delete_chat_threads([thread_id])
    except Exception:
        logger.exception("scheduled_tasks.discard_chat_failed")


class ScheduledTaskScheduler:
    def __init__(self, app: Any) -> None:
        _schedulers.add(self)
        self.app = app
        self.worker_id = uuid.uuid4().hex
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Made in start(): an asyncio.Event is bound to its loop, and a second lifespan may run on another.
        self._wake: Optional[asyncio.Event] = None
        self._stopping = False
        self._active: Optional[_ActiveRun] = None
        self._last_account: Optional[str] = None
        self._due: Optional[int] = None
        self._swap = SwapHolder()
        self._swap_account: Optional[AccountContext] = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name = "scheduled-task-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        self.wake()
        active = self._active
        if active is not None:
            active.cancel_event.set()
        task, self._task = self._task, None
        if task is None:
            return
        done, _ = await asyncio.wait({task}, timeout = _STOP_GRACE_SECONDS)
        if not done:
            task.cancel()
            await asyncio.wait({task}, timeout = _STOP_GRACE_SECONDS)

    def wake(self) -> None:
        """Re-check what is due now. Safe from any thread: sync routes run in a pool."""
        loop, event = self._loop, self._wake
        if loop is None or event is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(event.set)

    def cancel_task_run(
        self,
        task_id: str,
        account_id: Optional[str] = None,
    ) -> None:
        active = self._active
        account_id = current_account_id() if account_id is None else account_id
        if active is not None and active.task_id == task_id and active.account_id == account_id:
            active.cancel_event.set()

    def cancel_account(self, account_id: str) -> None:
        active = self._active
        if active is not None and active.account_id == account_id:
            active.cancel_event.set()

    async def _run_loop(self) -> None:
        while not self._stopping:
            # Cleared before looking, so a wake that lands while this pass runs is not lost.
            self._wake.clear()
            try:
                claimed = await asyncio.to_thread(self._claim)
                if claimed is not None:
                    account, task, run = claimed
                    # Shutdown began after the claim: give it back. Past here nothing awaits before the run is active.
                    if self._stopping:
                        await asyncio.to_thread(self._release, account, task, run)
                        return
                    swapped_by = self._swap_account
                    if swapped_by is not None and swapped_by.account_id != account.account_id:
                        await self._restore_swap()
                    await arun_as(account, self._execute(account, task, run))
                    if self._swap.pending is not None:
                        self._swap_account = account
                    continue
                await self._restore_swap()
                # Sleep until the next run is due (APScheduler, Celery beat); wake() cuts it short on any change.
                delay = _POLL_SECONDS
                if self._due is not None:
                    delay = min(delay, max(_MIN_SLEEP_SECONDS, (self._due - db.now_ms()) / 1000))
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never escape the loop: that would stop every schedule for the life of the process.
                logger.exception("scheduled_tasks.loop_iteration_failed")
                delay = 5.0
            if self._stopping:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout = delay)

    async def _restore_swap(self) -> None:
        swap, account = self._swap.pending, self._swap_account
        self._swap.pending = self._swap_account = None
        if swap is None or self._stopping:
            return
        from core.inference.llama_keepwarm import unattended_activity

        with unattended_activity():
            if account is None:
                await _restore_after_swap(swap, self.app)
            else:
                await arun_as(account, _restore_after_swap(swap, self.app))

    async def _yield_to_inference(self, cancel_event: threading.Event) -> None:
        from core.inference.llama_keepwarm import inference_in_flight
        deadline = asyncio.get_running_loop().time() + _YIELD_SECONDS
        while (
            inference_in_flight()
            and not cancel_event.is_set()
            and not self._stopping
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(_YIELD_POLL_SECONDS)

    def _accounts(self) -> list[AccountContext]:
        accounts = job_accounts()
        for index, account in enumerate(accounts):
            if account.account_id == self._last_account:
                return accounts[index + 1 :] + accounts[: index + 1]
        return accounts

    def _claim(self) -> Optional[tuple[AccountContext, dict, dict]]:
        """One pass, one read per account: claim the next due run, else note in ``_due`` when one is."""
        now = db.now_ms()
        self._due = None
        for account in self._accounts():
            if self._stopping:
                return None
            try:
                claimed, due = run_as(
                    account, db.poll, now, now - _STALE_RUN_MS, self.worker_id, next_run_for
                )
            except Exception:
                logger.exception("scheduled_tasks.claim_failed_for_account")
                continue
            if claimed is not None:
                # stop() cannot cancel this thread, so give the claim back here; its occurrence runs at next start.
                if self._stopping:
                    self._release(account, *claimed)
                    return None
                self._last_account = account.account_id
                return (account, *claimed)
            if due is not None and (self._due is None or due < self._due):
                self._due = due
        return None

    def _release(self, account: AccountContext, task: dict, run: dict) -> None:
        try:
            run_as(account, db.release_claim, run["id"], self.worker_id, task["nextRunAt"])
        except Exception:
            logger.exception("scheduled_tasks.release_failed")

    async def _heartbeat(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(db.heartbeat, run_id, self.worker_id)

    async def _execute(self, account: AccountContext, task: dict, run: dict) -> None:
        cancel_event = threading.Event()
        self._active = _ActiveRun(account.account_id, task["id"], run["id"], cancel_event)
        heartbeat = asyncio.create_task(self._heartbeat(run["id"]))
        started_at = run["startedAt"] or db.now_ms()
        outcome: dict[str, Any] = {"status": "failed"}
        saving = False
        try:
            if not await asyncio.to_thread(self._still_wanted, account, task["id"]):
                outcome = {"status": "failed", "error_code": "cancelled"}
                return
            if run.get("source") == "schedule":
                await self._yield_to_inference(cancel_event)
                if cancel_event.is_set():
                    raise RunFailed("cancelled")
            answer = await generate_answer(self.app, task, run, cancel_event, self._swap)
            if not await asyncio.to_thread(db.task_exists, task["id"]):
                return
            saving = True
            thread_id = await asyncio.to_thread(
                save_chat, task, run, answer, started_at, db.now_ms()
            )
            outcome = {"status": "succeeded", "thread_id": thread_id, "model": answer.model}
        except RunFailed as failure:
            if self._stopping and failure.code == "cancelled":
                failure = RunFailed("interrupted")
            outcome = {
                "status": "failed",
                "error_code": failure.code,
                "error": failure.message,
            }
        except asyncio.CancelledError:
            if _loop_cancelled(asyncio.current_task(), self._stopping):
                outcome = {"status": "failed", "error_code": "interrupted"}
                raise
            # This run's own cancel (the unload gate raises it while waiting): ends the run, not the loop.
            outcome = {
                "status": "failed",
                "error_code": "interrupted" if self._stopping else "cancelled",
            }
        except Exception as exc:
            logger.exception("scheduled_tasks.run_failed")
            outcome = {"status": "failed", "error": _error_text(str(exc))}
        finally:
            heartbeat.cancel()
            self._active = None
            # A chat stands only for a succeeded run; a failure removes it in the same transaction.
            discard = saving and outcome["status"] != "succeeded"
            settled = await self._settle(run["id"], {**outcome, "discard_chat": discard})
            if saving and settled is not True:
                await asyncio.to_thread(_discard_chat, db.chat_id_for_run(run["id"]))

    @staticmethod
    def _still_wanted(account: AccountContext, task_id: str) -> bool:
        return db.task_exists(task_id) and any(
            entry.account_id == account.account_id for entry in job_accounts()
        )

    async def _settle(self, run_id: str, outcome: dict[str, Any]) -> Optional[bool]:
        for attempt in range(_SETTLE_ATTEMPTS):
            try:
                return await asyncio.to_thread(db.finish_run, run_id, self.worker_id, **outcome)
            except Exception:
                logger.exception("scheduled_tasks.finish_failed")
            if attempt + 1 < _SETTLE_ATTEMPTS:
                await asyncio.sleep(_SETTLE_RETRY_SECONDS)
        return None
