# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs scheduled tasks while Studio is up.

One loop serves every account. It sleeps until the next task is due (or a route wakes it),
claims one run, sends the task's prompt to whichever chat model is loaded and saves the exchange
as a new chat. Runs go one at a time, so a burst of due tasks never stacks up on a local model.

A new chat per run, rather than one per task: an open chat never picks up messages written on the
server, so a result appended to it would stay invisible, and each run starts from a clean context
that a small model's window can always hold.
"""

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

# The longest the loop sleeps without being woken; due times are minute-precise.
_POLL_SECONDS = 60.0
# The shortest, after a pass that claimed nothing: a due task that cannot be claimed (its database
# keeps failing) must not turn the loop into a spin.
_MIN_SLEEP_SECONDS = 1.0
_HEARTBEAT_SECONDS = 30.0
# Several missed heartbeats: no one is working on the run any more.
_STALE_RUN_MS = 2 * 60 * 1000
# A run that has not answered by then is stopped, so one stuck run cannot hold up the rest.
_RUN_TIMEOUT_SECONDS = 20 * 60
_CANCEL_GRACE_SECONDS = 10.0
_STOP_GRACE_SECONDS = 5.0
# Settling a run is tried again over a locked database, all within the stop grace above.
_SETTLE_ATTEMPTS = 3
_SETTLE_RETRY_SECONDS = 1.0
_ERROR_TEXT_LIMIT = 500

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


@dataclass
class _ActiveRun:
    account_id: str
    task_id: str
    run_id: str
    cancel_event: threading.Event


def retire_account_scheduled_tasks(account: AccountContext) -> None:
    """Stop the account's run in progress. Retired accounts are already out of job_accounts()."""
    for scheduler in list(_schedulers):
        scheduler.cancel_account(account.account_id)


def next_run_for(task: dict, after_ms: int) -> Optional[int]:
    """The task's next run time after ``after_ms``. Raises when its schedule cannot resolve."""
    return next_run_after(parse_schedule(task["schedule"]), task["timezone"], after_ms)


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


def _producer_request(app: Any, run_id: str, cancel_event: threading.Event, tz: str) -> Request:
    """The in-process request a run is produced under, shaped like a durable chat run's."""
    from utils.current_date_prompt_settings import CURRENT_DATE_TIMEZONE_HEADER
    from utils.keyless_api_access import KEYLESS_ADMISSION_STATE_KEY
    from routes.inference import REQUIRE_TEXT_HEADER

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
            # Dates the prompt in the user's zone, not the server's.
            (CURRENT_DATE_TIMEZONE_HEADER.encode("latin-1"), tz.encode("latin-1", "ignore")),
            # A spoken reply has nowhere to go in a saved chat.
            (REQUIRE_TEXT_HEADER.lower().encode("latin-1"), b"1"),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {
            "generation_cancel_event": cancel_event,
            # Studio's own work, not someone using it as an API server.
            "api_monitor_via_api_key": False,
            # Created by a signed-in user, so never classified as keyless loopback traffic.
            KEYLESS_ADMISSION_STATE_KEY: False,
        },
    }

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


async def generate_answer(app: Any, task: dict, run: dict, cancel_event: threading.Event) -> Answer:
    """Ask the loaded chat model to do the task. Text only: tools never run unattended."""
    from fastapi import HTTPException

    from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError
    from core.inference.llama_keepwarm import InferenceActivityReservation
    from models.inference import ChatCompletionRequest
    from routes.inference import produce_openai_chat_completions

    payload = ChatCompletionRequest.model_validate(
        {
            # No model: the loaded one answers, and an idle-unloaded one is brought back. Naming
            # one could switch models under the user.
            "messages": [
                {"role": "system", "content": system_prompt(task, db.now_ms())},
                {"role": "user", "content": task["prompt"]},
            ],
            "stream": False,
            "cancel_id": run["id"],
            # Both opt-outs: --enable-tools overrides enable_tools, and an omitted enabled_tools
            # means every built-in, python and terminal included.
            "tool_choice": "none",
            "enabled_tools": [],
        }
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RUN_TIMEOUT_SECONDS
    activity = InferenceActivityReservation()
    activity.reserve()
    producer: Optional[asyncio.Future] = None
    try:
        # The deadline covers the wait for a model load or unload to let go too: one that never
        # does must not keep the scheduler from every other task.
        try:
            await asyncio.wait_for(activity.start(cancel_event), timeout = _RUN_TIMEOUT_SECONDS)
        except TimeoutError:
            raise RunFailed("timeout") from None
        if cancel_event.is_set():
            raise RunFailed("cancelled")
        producer = asyncio.ensure_future(
            produce_openai_chat_completions(
                payload,
                _producer_request(app, run["id"], cancel_event, task["timezone"]),
                task["ownerSubject"],
                cancel_on_disconnect = False,
            )
        )
        done, _ = await asyncio.wait({producer}, timeout = max(0.0, deadline - loop.time()))
        if not done:
            raise RunFailed("timeout")
        try:
            response = producer.result()
        except HTTPException as exc:
            if not _chat_model_loaded():
                raise RunFailed("no_model") from None
            raise RunFailed(None, _error_text(exc.detail)) from None
        except GpuBusyForAnotherAccountError as exc:
            raise RunFailed(None, _error_text(exc.as_http_exception().detail)) from None
    finally:
        if producer is not None and not producer.done():
            # A timeout or shutdown: stop the generation itself, not just this wait.
            cancel_event.set()
            producer.cancel()
            await asyncio.wait({producer}, timeout = _CANCEL_GRACE_SECONDS)
        activity.finish()
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
    )


def save_chat(task: dict, run: dict, answer: Answer, started_at: int, finished_at: int) -> str:
    """Write the run as a new chat: the task's prompt, then the answer. Returns the chat id."""
    from storage import studio_db

    thread_id = str(uuid.uuid4())
    user_message_id = str(uuid.uuid4())
    # After the prompt even within one millisecond, which the message order relies on.
    answered_at = max(finished_at, started_at + 1)
    model = answer.model or ""
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": task["title"],
            "modelType": "base",
            "modelId": model,
            "createdAt": started_at,
            "updatedAt": answered_at,
            "scheduledTaskId": task["id"],
        }
    )
    parts: list[dict] = []
    source = {"taskId": task["id"], "runId": run["id"]}
    metadata: dict = {
        "scheduledTask": source,
        # The shape a server-produced local reply carries, so the chat shows which model answered.
        "responseDetails": {
            "modelId": model,
            "modelLabel": model,
            "responseModelId": model,
            "providerName": "Local model",
            "providerType": "local",
            "startedAt": started_at,
            "finishedAt": answered_at,
            "durationMs": answered_at - started_at,
            "toolCalls": [],
        },
    }
    if answer.reasoning.strip():
        parts.append({"type": "reasoning", "text": answer.reasoning})
        # Whole seconds, as the reasoning header shows them. An answer that was not streamed has no
        # separate thinking time, so this is the run's.
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
        # Made in start(): an asyncio.Event belongs to the loop it was created on, and a second
        # lifespan can run on another one.
        self._wake: Optional[asyncio.Event] = None
        self._stopping = False
        self._active: Optional[_ActiveRun] = None
        self._last_account: Optional[str] = None

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
                    await arun_as(account, self._execute(account, task, run))
                    continue
                delay = await asyncio.to_thread(self._seconds_until_due)
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

    def _accounts(self) -> list[AccountContext]:
        accounts = job_accounts()
        # Round robin from the account after the last claim, so one busy account cannot starve the rest.
        for index, account in enumerate(accounts):
            if account.account_id == self._last_account:
                return accounts[index + 1 :] + accounts[: index + 1]
        return accounts

    def _claim(self) -> Optional[tuple[AccountContext, dict, dict]]:
        now = db.now_ms()
        for account in self._accounts():
            try:
                run_as(account, db.recover_stale_runs, now - _STALE_RUN_MS)
                claimed = run_as(account, db.claim_next, now, self.worker_id, next_run_for)
            except Exception:
                # One unreadable database must not hide the accounts behind it.
                logger.exception("scheduled_tasks.claim_failed_for_account")
                continue
            if claimed is not None:
                self._last_account = account.account_id
                return (account, *claimed)
        return None

    def _seconds_until_due(self) -> float:
        due: Optional[int] = None
        for account in job_accounts():
            try:
                at = run_as(account, db.next_due_at)
            except Exception:
                logger.exception("scheduled_tasks.next_due_failed_for_account")
                continue
            if at is not None and (due is None or at < due):
                due = at
        if due is None:
            return _POLL_SECONDS
        return min(_POLL_SECONDS, max(_MIN_SLEEP_SECONDS, (due - db.now_ms()) / 1000))

    async def _heartbeat(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            # A contended write misses one beat; the stale threshold allows for several.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(db.heartbeat, run_id, self.worker_id)

    async def _execute(self, account: AccountContext, task: dict, run: dict) -> None:
        cancel_event = threading.Event()
        self._active = _ActiveRun(account.account_id, task["id"], run["id"], cancel_event)
        heartbeat = asyncio.create_task(self._heartbeat(run["id"]))
        started_at = run["startedAt"] or db.now_ms()
        outcome: dict[str, Any] = {"status": "failed"}
        try:
            answer = await generate_answer(self.app, task, run, cancel_event)
            # Deleted while it ran: the answer has nowhere to belong.
            if not await asyncio.to_thread(db.task_exists, task["id"]):
                return
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
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # This loop itself is being cancelled: Studio is shutting down.
                outcome = {"status": "failed", "error_code": "interrupted"}
                raise
            # Raised for this run's own cancel event (the model unload gate raises it while the
            # run waits on a load or unload): it ends the run, not the loop serving every account.
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
            settled = await self._settle(run["id"], outcome)
            # A chat stands only for a run recorded as succeeded. Deleting a task deletes its runs
            # in the same transaction, so a run gone when it settles lost to a deletion, even one
            # that landed after the check above; one that cannot be settled at all is left to the
            # stale-run sweep, which records it as interrupted.
            if settled is not True and outcome.get("thread_id"):
                await asyncio.to_thread(_discard_chat, outcome["thread_id"])

    async def _settle(self, run_id: str, outcome: dict[str, Any]) -> Optional[bool]:
        """finish_run, tried again over a transient error such as a locked database. None when it
        never went through."""
        for attempt in range(_SETTLE_ATTEMPTS):
            try:
                return await asyncio.to_thread(db.finish_run, run_id, self.worker_id, **outcome)
            except Exception:
                logger.exception("scheduled_tasks.finish_failed")
            if attempt + 1 < _SETTLE_ATTEMPTS:
                await asyncio.sleep(_SETTLE_RETRY_SECONDS)
        return None
