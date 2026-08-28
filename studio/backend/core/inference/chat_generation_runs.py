# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background producer for durable local Studio chat generations."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from typing import Any, AsyncIterator

from starlette.requests import Request

from core.inference.llama_keepwarm import InferenceActivityReservation
from loggers import get_logger
from models.inference import ChatCompletionRequest
from state import active_generations
from storage import chat_generation_runs_db as db

logger = get_logger(__name__)
_EVENT_BATCH_SIZE = 16
_EVENT_BATCH_MIN_SIZE = 2
_EVENT_BATCH_SECONDS = 0.1
_EVENT_SINGLE_FLUSH_SECONDS = 1.0
_SHUTDOWN_GRACE_SECONDS = 10.0
# Second budget, after task.cancel(). Shorter than the grace period: by this point the
# run is already being abandoned, and the only question is whether shutdown returns.
_SHUTDOWN_CANCEL_SECONDS = 5.0


class _SSEDecoder:
    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text.replace("\r\n", "\n")
        values: list[str] = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            data = "\n".join(
                line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
            )
            if data:
                values.append(data)
        return values


def _background_request(app: Any, run_id: str, cancel_event: threading.Event) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/chat-runs/producer",
        "raw_path": b"/api/inference/chat-runs/producer",
        "query_string": b"",
        "headers": [(b"x-unsloth-generation-run", run_id.encode("ascii", "ignore"))],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {"generation_cancel_event": cancel_event},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _chunk_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
            return str(choice["finish_reason"])
    return None


def _chunk_error(chunk: dict[str, Any]) -> str | None:
    error = chunk.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("detail") or "Generation failed")[:1000]
    if error:
        return str(error)[:1000]
    return None


async def _close_iterator(iterator: AsyncIterator[Any] | None) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass


class ChatGenerationSupervisor:
    def __init__(
        self,
        app: Any,
        *,
        wal_keeper: sqlite3.Connection | None = None,
    ) -> None:
        self.app = app
        # A quiet connection keeps SQLite's WAL alive between the short-lived
        # event-batch connections.  Without it, each batch can be the last open
        # connection and SQLite checkpoints/deletes the WAL on close, turning the
        # 100 ms stream cadence into repeated writes back to studio.db.
        self._wal_keeper = wal_keeper
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_registrations: dict[str, active_generations.ActiveGeneration] = {}
        self._activities: dict[str, InferenceActivityReservation] = {}
        self._shutdown_runs: set[str] = set()
        self._stopping = False

    def _ensure_reservation(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model: str | None = None,
    ) -> bool:
        if self._stopping:
            return False
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            # Enrich an early run-only reservation with authoritative identity.
            with active_generations.ActiveGeneration(
                cancel_event,
                run_id = run_id,
                thread_id = thread_id,
                model = model,
            ):
                pass
            return True
        cancel_event = threading.Event()
        activity = InferenceActivityReservation()
        activity.reserve()
        registration = active_generations.ActiveGeneration(
            cancel_event,
            run_id = run_id,
            thread_id = thread_id,
            model = model,
        )
        registration.__enter__()
        self._cancel_events[run_id] = cancel_event
        self._activities[run_id] = activity
        self._active_registrations[run_id] = registration
        return True

    def start(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model: str | None = None,
    ) -> None:
        if (
            self._stopping
            or run_id in self._tasks
            or not self._ensure_reservation(run_id, thread_id = thread_id, model = model)
        ):
            return
        cancel_event = self._cancel_events[run_id]
        activity = self._activities[run_id]
        try:
            task = asyncio.create_task(
                self._produce(run_id, cancel_event, activity),
                name = f"chat-generation-{run_id}",
            )
        except BaseException:
            self._cleanup_registration(run_id)
            raise
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed, rid = run_id: self._task_done(rid, completed))

    def _cleanup_registration(self, run_id: str) -> None:
        self._cancel_events.pop(run_id, None)
        activity = self._activities.pop(run_id, None)
        if activity is not None:
            activity.finish()
        registration = self._active_registrations.pop(run_id, None)
        if registration is not None:
            registration.__exit__(None, None, None)

    def _task_done(self, run_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(run_id, None)
        self._cleanup_registration(run_id)
        self._shutdown_runs.discard(run_id)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            logger.error("Durable chat generation %s crashed: %s", run_id, exc)

    def cancel(self, run_id: str) -> None:
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()
        else:
            task = self._tasks.get(run_id)
            if task is not None and not task.done():
                # Defensive compatibility for a task registered by a caller
                # other than start(); production tasks always own an event.
                task.cancel()
        active_generations.cancel_run(run_id)
        # The inference cancel registry closes the narrow gap where registration is imminent
        # but this supervisor has not yet observed it.
        from routes.inference import _cancel_by_cancel_id_or_stash

        _cancel_by_cancel_id_or_stash(run_id)

    async def stop(self) -> None:
        self._stopping = True
        try:
            tasks = list(self._tasks.items())
            self._shutdown_runs.update(run_id for run_id, _task in tasks)
            for run_id, _task in tasks:
                self.cancel(run_id)
            if not tasks:
                return
            # asyncio.wait, not wait_for(gather(...)): on timeout wait_for cancels the inner
            # future and then awaits it, so a producer that does not unwind on cancellation --
            # an engine draining its subprocess inside the generator's aclose -- makes the
            # wait itself unbounded, and takes the whole uvicorn shutdown down with it. wait
            # returns the pending set instead and leaves those tasks alone.
            pending = {task for _run_id, task in tasks}
            _done, pending = await asyncio.wait(pending, timeout = _SHUTDOWN_GRACE_SECONDS)
            if not pending:
                return
            for task in pending:
                task.cancel()
            _done, pending = await asyncio.wait(pending, timeout = _SHUTDOWN_CANCEL_SECONDS)
            if pending:
                stuck = [run_id for run_id, task in tasks if task in pending]
                # Abandoned, not leaked: the run is already fenced and reconcile_orphaned_runs
                # settles it on the next boot. Process exit reclaims the rest.
                logger.warning(
                    "Durable chat generations did not stop within the shutdown budget: %s",
                    ", ".join(stuck),
                )
        finally:
            if self._wal_keeper is not None:
                self._wal_keeper.close()
                self._wal_keeper = None

    async def _produce(
        self,
        run_id: str,
        cancel_event: threading.Event | None = None,
        activity: InferenceActivityReservation | None = None,
    ) -> None:
        cancel_event = cancel_event or threading.Event()
        if activity is None:
            activity = InferenceActivityReservation()
            activity.reserve()
        pending: list[tuple[str, dict[str, Any], int]] = []
        iterator: AsyncIterator[Any] | None = None
        last_flush = time.monotonic()
        finish_reason: str | None = None
        error: str | None = None
        saw_done = False
        worker_token: str | None = None
        next_raw_task: asyncio.Task | None = None
        try:
            worker_run = await asyncio.to_thread(db.get_worker_run, run_id)
            if worker_run is None:
                return
            run, owner, worker_token = worker_run
            if cancel_event.is_set():
                shutting_down = run_id in self._shutdown_runs
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "failed" if shutting_down else "cancelled",
                    finish_reason = "interrupted" if shutting_down else "cancelled",
                    error = "Studio shut down during generation" if shutting_down else None,
                )
                return
            await activity.start(cancel_event)
            if cancel_event.is_set():
                shutting_down = run_id in self._shutdown_runs
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "failed" if shutting_down else "cancelled",
                    finish_reason = "interrupted" if shutting_down else "cancelled",
                    error = "Studio shut down during generation" if shutting_down else None,
                )
                return
            if not await asyncio.to_thread(db.mark_running, run_id, worker_token):
                return
            worker_run = await asyncio.to_thread(db.get_worker_run, run_id, worker_token)
            if worker_run is None:
                return
            run, owner, worker_token = worker_run
            if run["status"] != "running" or run["cancelRequested"]:
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "cancelled",
                )
                return

            from routes.inference import produce_openai_chat_completions

            payload = ChatCompletionRequest.model_validate(run["requestPayload"])
            response = await produce_openai_chat_completions(
                payload,
                _background_request(self.app, run_id, cancel_event),
                owner,
                cancel_on_disconnect = False,
            )
            if int(getattr(response, "status_code", 200)) >= 400:
                raise RuntimeError(f"Local generation returned HTTP {response.status_code}")
            iterator = getattr(response, "body_iterator", None)
            if iterator is None:
                raise RuntimeError("Local generation did not return an event stream")
            decoder = _SSEDecoder()
            next_raw_task = asyncio.create_task(iterator.__anext__())
            while True:
                timeout = (
                    max(
                        0.0,
                        (
                            _EVENT_BATCH_SECONDS
                            if len(pending) >= _EVENT_BATCH_MIN_SIZE
                            else _EVENT_SINGLE_FLUSH_SECONDS
                        )
                        - (time.monotonic() - last_flush),
                    )
                    if pending
                    else None
                )
                ready, _waiting = await asyncio.wait({next_raw_task}, timeout = timeout)
                if not ready:
                    await asyncio.to_thread(
                        db.append_events,
                        run_id,
                        worker_token,
                        pending,
                    )
                    pending = []
                    last_flush = time.monotonic()
                    continue
                try:
                    raw = next_raw_task.result()
                except StopAsyncIteration:
                    next_raw_task = None
                    break
                next_raw_task = asyncio.create_task(iterator.__anext__())
                text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                for encoded in decoder.feed(text):
                    if encoded == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(encoded)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    pending.append(("chunk", chunk, db.now_ms()))
                    finish_reason = _chunk_finish_reason(chunk) or finish_reason
                    error = _chunk_error(chunk) or error
                    now = time.monotonic()
                    if len(pending) >= _EVENT_BATCH_SIZE or (
                        len(pending) >= _EVENT_BATCH_MIN_SIZE
                        and now - last_flush >= _EVENT_BATCH_SECONDS
                    ):
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                        pending = []
                        last_flush = now
                if saw_done:
                    break

            current = await asyncio.to_thread(db.get_run, run_id)
            if current is None:
                return
            if run_id in self._shutdown_runs:
                status = "failed"
                finish_reason = "interrupted"
                error = "Studio shut down during generation"
            elif current["cancelRequested"] or (cancel_event.is_set() and error is None):
                # A bare event is not proof of a user stop: the streaming paths set this
                # same event from their cleanup after emitting an in-band error, so a
                # parsed failure outranks it. An explicit cancelRequested still wins,
                # and a real Stop carries no error chunk, so neither loses its identity.
                status = "cancelled"
                finish_reason = "cancelled"
            elif error is not None:
                status = "failed"
            elif not saw_done and finish_reason is None:
                status = "failed"
                finish_reason = "interrupted"
                error = "Generation stream ended before completion"
            else:
                status = "completed"
            await asyncio.to_thread(
                db.finish_run,
                run_id,
                worker_token = worker_token,
                status = status,
                finish_reason = finish_reason,
                error = error,
                pending_events = pending,
            )
            pending = []
        except asyncio.CancelledError:
            if worker_token is not None:
                shutting_down = run_id in self._shutdown_runs
                cancelled = cancel_event.is_set() and not shutting_down
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "cancelled" if cancelled else "failed",
                    finish_reason = "cancelled" if cancelled else "interrupted",
                    error = None if cancelled else "Generation worker stopped unexpectedly",
                    pending_events = pending,
                )
            pending = []
            raise
        except Exception as exc:
            if worker_token is not None:
                shutting_down = run_id in self._shutdown_runs
                cancelled = cancel_event.is_set() and not shutting_down
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "cancelled" if cancelled else "failed",
                    finish_reason = (
                        "cancelled" if cancelled else "interrupted" if shutting_down else "error"
                    ),
                    error = (
                        None
                        if cancelled
                        else "Studio shut down during generation"
                        if shutting_down
                        else str(exc)[:1000]
                    ),
                    pending_events = pending,
                )
            pending = []
        finally:
            if next_raw_task is not None:
                if not next_raw_task.done():
                    next_raw_task.cancel()
                await asyncio.gather(next_raw_task, return_exceptions = True)
            await _close_iterator(iterator)
            activity.finish()
