# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background producer for durable local Studio chat generations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
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
# The sweeper's own shutdown budget, far below the producers'. Its work is redundant
# at shutdown and Desktop force-kills the backend after five seconds.
_SWEEP_SHUTDOWN_SECONDS = 0.5
# A durable run sets cancel_on_disconnect=False, so reaping is keyed on progress rather
# than on connectedness. The default matches llama_cpp._DEFAULT_FIRST_TOKEN_TIMEOUT_S, the
# request path's own first-token budget: a lease older than that cannot be legitimate
# prefill, and slow decode is safe at any speed.
# A century: clear of any real lease, far below where integer milliseconds overflow.
_MAX_ENV_SECONDS = 100.0 * 365.0 * 24.0 * 60.0 * 60.0
# The longest admission keep-alive cadence worth deriving a lease from. A day already
# means the queue never reports, and tripling it stays far inside _MAX_ENV_SECONDS.
_MAX_ADMISSION_INTERVAL_SECONDS = 24.0 * 60.0 * 60.0
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
    """Seconds from the environment, falling back on anything unusable.

    Non-finite values parse cleanly and fail silently downstream: `inf` raises on
    `int(timeout * 1000)` every sweep, and `max(0.0, nan)` returns 0.0, disabling reaping.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        logger.warning(
            "chat_generation_lease_env_ignored",
            variable = name,
            value = raw,
            reason = "not a finite number",
        )
        return default
    if value > _MAX_ENV_SECONDS:
        # Finite is not usable: past ~1.8e305 the multiply to milliseconds overflows and
        # every sweep raises. Clamped, not rejected, since this already meant "never reap".
        logger.warning(
            "chat_generation_lease_env_clamped",
            variable = name,
            value = raw,
            applied_s = _MAX_ENV_SECONDS,
            reason = "larger than can be converted to integer milliseconds",
        )
        return _MAX_ENV_SECONDS
    return value


async def _sweep_in_daemon_thread(fn, /, *args, **kwargs):
    """Run one blocking sweep on a daemon thread.

    Not asyncio.to_thread: its executor threads are non-daemon and joined by an atexit
    hook, so a sweep parked on SQLite's writer lock keeps the whole process alive long
    after shutdown gave up waiting for it. Studio Desktop allows five seconds for a
    graceful backend exit before force-killing, and stops the backend this way before it
    updates, so an unbounded exit is a user-visible hang. A daemon thread abandoned here
    cannot hold the interpreter open.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _settle(setter, value):
        # The loop can be closed already: this thread outlived the shutdown that
        # abandoned it, which is exactly the case the daemon thread exists to make safe.
        try:
            loop.call_soon_threadsafe(lambda: future.done() or setter(value))
        except RuntimeError:
            pass

    def _runner():
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - relayed to the awaiting caller
            _settle(future.set_exception, exc)
        else:
            _settle(future.set_result, result)

    threading.Thread(target = _runner, name = "chat-lease-sweep", daemon = True).start()
    return await future


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
        configured = max(
            0.0,
            timeout_s
            if timeout_s is not None
            else _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS),
        )
        self._timeout = _clamped_lease_timeout(configured)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # Grace for a settled producer to notice the cooperative cancel. Generous: unwinding
    # cleanly beats being cancelled mid-teardown, and the run is already declared dead.
    _FORCE_CANCEL_GRACE_S = 30.0

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
        settled = await _sweep_in_daemon_thread(
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
                continue
            asyncio.create_task(
                self._force_cancel_after_grace(supervisor, run_id),
                name = f"chat-generation-lease-force-cancel:{run_id}",
            )
        return settled

    async def _force_cancel_after_grace(self, supervisor: Any, run_id: str) -> None:
        """Escalate from the cooperative cancel to cancelling the producer task.

        supervisor.cancel() only sets a threading.Event, which a producer blocked inside
        next(gen) never looks at, so it goes on holding its activity reservation and engine
        slot. Cancelling the task releases that bookkeeping. It cannot unblock a thread
        already inside the engine, and the warning says so rather than implying otherwise.
        """
        await asyncio.sleep(self._FORCE_CANCEL_GRACE_S)
        task = getattr(supervisor, "_tasks", {}).get(run_id)
        if task is None or task.done():
            return
        logger.warning(
            "chat_generation_run_force_cancelled",
            run_id = run_id,
            grace_s = self._FORCE_CANCEL_GRACE_S,
            note = "producer ignored the cooperative cancel; any engine thread it left "
            "blocked cannot be reclaimed from here",
        )
        task.cancel()

    async def stop(self) -> None:
        self._stop_event.set()
        task, self._task = self._task, None
        if task is None:
            return
        # A short wait, not the producer grace. Letting a sweep finish at shutdown buys
        # nothing: the boot reconcile settles every active run anyway, so its work is
        # redundant here, while a sweep parked on the writer lock would otherwise spend
        # Studio Desktop's whole graceful-exit budget before producers are even signalled.
        # asyncio.wait, never wait_for(gather(...)); see ChatGenerationSupervisor.stop.
        _done, pending = await asyncio.wait({task}, timeout = _SWEEP_SHUTDOWN_SECONDS)
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
# Leaving the queue. Renewed unconditionally: wait renewals are rate limited, and the
# lease equals the first-token timeout, so any age carried in is negative margin.
_ADMISSION_DONE_MARKER = ": admission-done"


def _minimum_lease_seconds() -> float:
    """The shortest lease every renewal source can keep up with.

    The admission keep-alive interval is upstream and not ours to speed up, so a queued run
    only produces a renewable marker that often. A shorter lease expires between markers
    however fast we poll, reaping a healthy queued run.
    """
    from core.inference.llama_admission import DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S

    try:
        from core.inference.llama_admission import llama_admission_config_from_env
        interval = float(llama_admission_config_from_env().keepalive_interval_s)
    except Exception:
        interval = float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)
    # That parser is not ours and only checks the value is positive, so `inf` arrives
    # intact and makes the applied lease infinite, which the sweeper cannot convert to
    # milliseconds. An oversized finite cadence stretches it past any horizon instead.
    if not math.isfinite(interval) or interval > _MAX_ADMISSION_INTERVAL_SECONDS:
        logger.warning(
            "chat_generation_admission_cadence_ignored",
            value = interval,
            applied_s = DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S,
            reason = "not a usable keep-alive cadence",
        )
        interval = float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)
    return max(1.0, interval) * 3.0


def _applied_lease_timeout(configured: float) -> float:
    """The lease actually in force, without the warning. Zero still means disabled.

    Separate from the logging wrapper because the renewal cadence consults this on every
    keep-alive, and warning once per keep-alive would bury the one line that matters.
    """
    if configured <= 0.0:
        return 0.0
    return max(configured, _minimum_lease_seconds())


def _clamped_lease_timeout(configured: float) -> float:
    """Raise a lease that no renewal source could satisfy, and say so.

    Silently honouring it would reap healthy queued runs, and silently ignoring it would
    hide that the setting did nothing. Zero still means disabled.
    """
    applied = _applied_lease_timeout(configured)
    if applied == configured:
        return applied
    logger.warning(
        "chat_generation_lease_timeout_clamped",
        configured_s = round(configured, 2),
        applied_s = round(applied, 2),
        reason = "shorter than the admission keep-alive cadence could renew",
    )
    return applied


def _renew_interval_seconds() -> float:
    """How often a lease may be renewed by something other than streamed output.

    Derived from the lease rather than fixed: a fixed 30s against a shorter lease would
    first renew after the sweep had already settled the run. A quarter gives three renewals
    per window, and the floor keeps a short lease from becoming a write per second.
    """
    lease = _applied_lease_timeout(
        _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS)
    )
    if lease <= 0.0:  # sweeping disabled, so cadence only controls write volume
        return 30.0
    # The floor must stay UNDER the lease: a one second floor against a one second lease
    # first renews no earlier than expiry. A quarter keeps three renewals per window.
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

    # Total time, not a count: the interval derives from the lease, so a count would mean
    # very different durations. Bounded because an unbounded heartbeat would keep a
    # preparation that never returns alive forever, the failure this file exists to end.
    _PREPARE_RENEW_MAX_SECONDS = 2 * 60 * 60

    @contextlib.asynccontextmanager
    async def _lease_heartbeat(self, run_id: str):
        """Hold the progress lease open across work that produces no output.

        Covers the lifecycle gate as well as model preparation: a run waiting on the gate is
        still queued, so its lease ages from created_at with nothing renewing it. Cancelled
        on exit, including an early return, so it never overlaps the streaming phase.
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
        holding SQLite's writer lock past the busy timeout would otherwise abort a healthy
        generation; a missed stamp costs one interval instead.
        """
        try:
            await asyncio.to_thread(db.touch_progress, run_id)
        except Exception:
            return

    async def _renew_lease_while_preparing(self, run_id: str) -> None:
        interval = _renew_interval_seconds()
        for _ in range(max(1, int(self._PREPARE_RENEW_MAX_SECONDS / interval))):
            await asyncio.sleep(interval)
            # Skip a contended stamp rather than abandon the rest: giving up here would let a
            # healthy long load be reaped once the last stamp aged out.
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
            # Spans the lifecycle gate as well as preparation: a run waiting on the gate is
            # still queued, so its lease ages from created_at with nothing renewing it.
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
                # Switching, idle reload and auto-download all happen in the call below, and
                # llama.cpp's first-token budget only starts after it. One touch afterwards
                # cannot cover a preparation longer than the lease itself.
                response = await produce_openai_chat_completions(
                    payload,
                    _background_request(self.app, run_id, cancel_event),
                    owner,
                    cancel_on_disconnect = False,
                )
            # Streamed output is the lease from here; stamped so the handover has no gap.
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
                # Admission comments are progress; plain keep-alives are not. A queued run only
                # emits `: admission-wait`, which _SSEDecoder drops, so nothing renewed the lease
                # and a healthy queue reaped its own runs. `: keep-alive` is the opposite signal,
                # emitted when the generator has produced NOTHING, so renewing on any byte would
                # keep a wedged run alive forever. Rate limited because chunk traffic already
                # renews through append_events.
                if _ADMISSION_DONE_MARKER in text:
                    # Once per run, so no rate limit; the two budgets must start together.
                    last_keepalive = time.monotonic()
                    await self._try_touch_progress(run_id)
                elif _ADMISSION_WAIT_MARKER in text:
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
