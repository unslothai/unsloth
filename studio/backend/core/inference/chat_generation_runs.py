# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background producer for durable local Studio chat generations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
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
# Progress lease defaults. A durable run sets cancel_on_disconnect=False so a closed
# browser cannot kill a long generation, and nothing else bounds it: a producer that stops
# producing leaves the thread "generating" forever, hiding Send and holding the engine
# slot. Reaping is therefore keyed on progress, never on connectedness.
#
# The default matches llama_cpp._DEFAULT_FIRST_TOKEN_TIMEOUT_S, the request path's own
# first-token budget, so a run whose lease has not moved for longer than that cannot be
# legitimately prefilling. Slow decode is safe at any speed.
_LEASE_TIMEOUT_SECONDS = 1200.0
_LEASE_SWEEP_INTERVAL_SECONDS = 60.0
_LEASE_ERROR = "Generation stopped making progress"


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


def _env_seconds(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class ChatGenerationLeaseSweeper:
    """Periodically settle durable runs whose progress lease has expired.

    reconcile_orphaned_runs used to run exactly once, at process boot, so a run that
    wedged while Studio kept running was never repaired and browser reloads could not
    clear it. This runs the same reconciliation on an interval, bounded to runs that
    have made no progress for the lease timeout so a live generation is never reaped.
    """

    def __init__(
        self,
        app: Any,
        *,
        interval_s: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.app = app
        self._interval = max(
            1.0,
            interval_s
            if interval_s is not None
            else _env_seconds(
                "UNSLOTH_STUDIO_CHAT_RUN_LEASE_SWEEP_INTERVAL_S", _LEASE_SWEEP_INTERVAL_SECONDS
            ),
        )
        # 0 disables the sweep entirely, matching UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S.
        self._timeout = max(
            0.0,
            timeout_s
            if timeout_s is not None
            else _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS),
        )
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._timeout > 0.0

    def start(self) -> None:
        if self._task is not None or not self.enabled:
            return
        # A second lifespan reuses the instance parked on app.state, and stop() left the
        # event set. Recreated rather than cleared, because the second lifespan can also
        # be a different event loop (repeated TestClient contexts, an embedded server
        # restart), and an asyncio.Event stays bound to the loop it was made on: clearing
        # it would leave the new task failing its first wait with "bound to a different
        # event loop", silently disabling reaping for that whole lifespan.
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name = "chat-generation-lease-sweeper")

    async def _run(self) -> None:
        while True:
            waiter = asyncio.ensure_future(self._stop_event.wait())
            done, _pending = await asyncio.wait({waiter}, timeout = self._interval)
            if done:
                return
            waiter.cancel()
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One failed sweep (a locked database, a torn-down home in tests) must
                # not retire the watchdog for the life of the process.
                logger.warning("chat_generation_lease_sweep_failed", error = repr(exc))

    async def sweep_once(self) -> list[str]:
        if not self.enabled:
            return []
        settled = await asyncio.to_thread(
            db.reconcile_runs,
            error = _LEASE_ERROR,
            stale_after_ms = int(self._timeout * 1000),
        )
        if not settled:
            return []
        supervisor = getattr(getattr(self.app, "state", None), "chat_generation_supervisor", None)
        for run_id in settled:
            logger.warning(
                "chat_generation_run_lease_expired",
                run_id = run_id,
                idle_s = round(self._timeout, 1),
            )
            if supervisor is None:
                continue
            # The row is settled, but a producer wedged inside the engine is still
            # holding its slot and activity reservation; cancel unwinds it.
            try:
                supervisor.cancel(run_id)
            except Exception as exc:
                logger.warning(
                    "chat_generation_lease_cancel_failed", run_id = run_id, error = repr(exc)
                )
        return settled

    async def stop(self) -> None:
        self._stop_event.set()
        task, self._task = self._task, None
        if task is None:
            return
        # Same idiom as ChatGenerationSupervisor.stop(): asyncio.wait, never
        # wait_for(gather(...)), so a sweep parked in the database cannot make
        # shutdown itself unbounded.
        _done, pending = await asyncio.wait({task}, timeout = _SHUTDOWN_GRACE_SECONDS)
        if not pending:
            return
        task.cancel()
        _done, pending = await asyncio.wait({task}, timeout = _SHUTDOWN_CANCEL_SECONDS)
        if pending:
            logger.warning(
                "The chat generation lease sweeper did not stop within the shutdown budget"
            )


def start_lease_sweeper(app: Any) -> ChatGenerationLeaseSweeper | None:
    """Attach one lease sweeper to the app and start it. Idempotent per app."""
    state = getattr(app, "state", None)
    sweeper = getattr(state, "chat_generation_lease_sweeper", None)
    if sweeper is None:
        sweeper = ChatGenerationLeaseSweeper(app)
        if state is not None:
            state.chat_generation_lease_sweeper = sweeper
    sweeper.start()
    return sweeper


# The admission stream's own comment, matched rather than imported to keep this module
# free of a routes import at module scope. Pinned by a test against the constant there.
_ADMISSION_WAIT_MARKER = ": admission-wait"


def _renew_interval_seconds() -> float:
    """How often a lease may be renewed by something other than streamed output.

    Derived from the configured lease rather than fixed, because both are configurable and
    a fixed 30 or 60 seconds is longer than a lease set below that: the first renewal would
    then arrive after the sweep had already settled the run. A quarter of the lease gives
    three renewals inside every window, and the floor keeps a very short lease from turning
    into a write per second.
    """
    lease = _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS)
    if lease <= 0.0:  # sweeping disabled, so cadence only controls write volume
        return 30.0
    # The floor has to stay UNDER the lease, not at a round number: a one second floor
    # against a one second lease schedules the first renewal no earlier than expiry, and
    # the sweeper may also be running at one second. A quarter of the lease throughout
    # keeps three renewals inside every window however short it is configured.
    return min(30.0, max(0.25, lease / 4.0))


class ChatGenerationSupervisor:
    def __init__(self, app: Any) -> None:
        self.app = app
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
        # Before the runs, so the sweeper cannot settle a run as stalled while shutdown
        # is already settling it as interrupted.
        sweeper = getattr(getattr(self.app, "state", None), "chat_generation_lease_sweeper", None)
        if sweeper is not None:
            await sweeper.stop()
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

    # Total time renewals may cover, not a count: the interval is derived from the
    # configured lease, so a fixed count would mean very different durations. Bounded at
    # all because an unbounded heartbeat keeps a preparation that never returns alive
    # forever, which is the failure this whole file exists to end. Generous enough to
    # outlast a real download; after it the ordinary lease takes over.
    _PREPARE_RENEW_MAX_SECONDS = 2 * 60 * 60

    @contextlib.asynccontextmanager
    async def _lease_heartbeat(self, run_id: str):
        """Hold the progress lease open across work that produces no output.

        Covers the lifecycle gate as well as model preparation. A run waiting on the gate
        is still queued, so its lease ages from created_at with nothing renewing it, and a
        slow preview swap on the other side of the gate could outlast the lease.

        Cancelled on exit, including an early return, so this never overlaps the streaming
        phase where real output is the lease.
        """
        task = asyncio.create_task(
            self._renew_lease_while_preparing(run_id),
            name = f"chat-generation-prepare-lease:{run_id}",
        )
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _try_touch_progress(self, run_id: str) -> None:
        """Renew the lease, treating contention as a missed stamp rather than a failure.

        Every renewal that is not streamed output goes through here. A history transaction
        can hold SQLite's writer lock past the busy timeout, and letting that escape would
        abort an otherwise healthy generation over a lock about to be released. Missing one
        stamp costs one interval; the next renewal takes it.
        """
        try:
            await asyncio.to_thread(db.touch_progress, run_id)
        except Exception:
            return

    async def _renew_lease_while_preparing(self, run_id: str) -> None:
        interval = _renew_interval_seconds()
        for _ in range(max(1, int(self._PREPARE_RENEW_MAX_SECONDS / interval))):
            await asyncio.sleep(interval)
            # Skips a contended stamp rather than abandoning the rest: one lost stamp is
            # ordinary, and giving up here would let a healthy long load be reaped once
            # the last successful stamp aged out.
            await self._try_touch_progress(run_id)

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
            # Spans the lifecycle gate as well as preparation. A run waiting on the gate
            # is still queued, so its lease ages from created_at with nothing renewing it,
            # and a preview swap on the other side of the gate can outlast the lease.
            async with self._lease_heartbeat(run_id):
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
                # Automatic switching, idle reload and auto-download all happen inside the
                # call below, and llama.cpp's own first-token budget only starts after it,
                # so a lease aged from mark_running would reap a legitimate preparation. One
                # touch afterwards covers load-plus-prefill but not a single preparation that
                # is itself longer than the lease, which a large GGUF over a slow link is.
                response = await produce_openai_chat_completions(
                    payload,
                    _background_request(self.app, run_id, cancel_event),
                    owner,
                    cancel_on_disconnect = False,
                )
            # The load is behind us: the stream is open, so streamed output is the
            # lease from here. Stamped once more so the handover carries no gap.
            await self._try_touch_progress(run_id)
            if int(getattr(response, "status_code", 200)) >= 400:
                raise RuntimeError(f"Local generation returned HTTP {response.status_code}")
            iterator = getattr(response, "body_iterator", None)
            if iterator is None:
                raise RuntimeError("Local generation did not return an event stream")
            decoder = _SSEDecoder()
            last_keepalive = time.monotonic()
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
                # Admission-wait comments are progress; plain keep-alives are not.
                #
                # A run queued behind another generation waits in the admission stream,
                # which emits `: admission-wait` while the server is healthy and simply
                # busy, and _SSEDecoder drops those, so nothing renewed the lease and a
                # normally functioning queue got its own runs reaped.
                #
                # `: keep-alive` is the opposite signal. routes/inference.py emits it when
                # the generator has produced NOTHING for a stall interval, which is exactly
                # the wedge this file exists to reap. Renewing on any byte would keep a
                # wedged run alive forever. Rate limited because chunk traffic already
                # renews through append_events.
                if _ADMISSION_WAIT_MARKER in text:
                    now_s = time.monotonic()
                    if now_s - last_keepalive >= _renew_interval_seconds():
                        last_keepalive = now_s
                        await self._try_touch_progress(run_id)
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
