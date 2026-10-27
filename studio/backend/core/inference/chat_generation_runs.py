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
from core.training.account_jobs import sweepable_job_accounts
from utils.account_context import current_account_id, run_as
from utils.first_token_timeout import FirstTokenWait, track_first_token_wait
from loggers import get_logger
from models.inference import ChatCompletionRequest
from state import active_generations
from storage import chat_generation_runs_db as db

logger = get_logger(__name__)
# Followers only see appended chunks, so this is the chat's text frame rate: at most one commit per display frame.
_EVENT_FLUSH_SECONDS = 1 / 60
# Reading pauses past this many events behind a running commit, so a wedged database cannot buffer a generation.
_MAX_PENDING_EVENTS = 4096
_SHUTDOWN_GRACE_SECONDS = 10.0
# Post-cancel budget, shorter than the grace period: the run is abandoned, only prompt return matters
_SHUTDOWN_CANCEL_SECONDS = 5.0
# Desktop force-kills the backend after five seconds.
_SWEEP_SHUTDOWN_SECONDS = 0.5
# Durable runs set cancel_on_disconnect=False, so reaping keys on progress, not connectedness. A century: clear of real leases, far below ms overflow.
_MAX_ENV_SECONDS = 100.0 * 365.0 * 24.0 * 60.0 * 60.0
# Admission keep-alive cap: a day means the queue never reports; 3x still fits _MAX_ENV_SECONDS
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


def _background_request(
    app: Any,
    run_id: str,
    cancel_event: threading.Event,
    timezone_headers: dict[str, str] | None = None,
    *,
    via_api_key: bool | None = None,
) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/chat-runs/producer",
        "raw_path": b"/api/inference/chat-runs/producer",
        "query_string": b"",
        "headers": [
            (b"x-unsloth-generation-run", run_id.encode("ascii", "ignore")),
            # Durable runs replay their event log to the UI, which needs the Unsloth control frames (see routes.inference).
            (b"x-unsloth-events", b"1"),
            *(
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in (timezone_headers or {}).items()
            ),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {
            "generation_cancel_event": cancel_event,
            "api_monitor_via_api_key": via_api_key,
        },
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
    """Non-finite values are rejected: inf raises every sweep and nan disables reaping."""
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
        # Past ~1.8e305 the multiply to ms overflows and every sweep raises; clamp rather than reject, since huge already meant never reap.
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
    """A daemon thread, since to_thread's non-daemon threads would keep a sweep alive past shutdown."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _settle(setter, value):
        # Loop may already be closed: this daemon thread can outlive the shutdown that abandoned it
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


def _run_id_held_by_another_account(account: Any, run_id: str) -> bool:
    """Client-chosen ids are per account but the supervisor keys by bare id, so a live
    registration under another account is theirs. No registration still cancels."""
    account_id = getattr(account, "account_id", None)
    return any(
        entry.get("run_id") == run_id and entry.get("account_id") != account_id
        for entry in active_generations.snapshot()
    )


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

    # Generous grace for a settled producer to unwind; cancelling mid-teardown is worse
    _FORCE_CANCEL_GRACE_S = 30.0

    @property
    def enabled(self) -> bool:
        return self._timeout > 0.0

    def start(self) -> None:
        if self._task is not None or not self.enabled:
            return
        # Recreated, not cleared: an asyncio.Event is bound to the loop it was made on.
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
                # A failed sweep (locked DB, torn-down home in tests) must not retire the watchdog for process life
                logger.warning("chat_generation_lease_sweep_failed", error = repr(exc))

    async def sweep_once(self) -> list[str]:
        if not self.enabled:
            return []
        settled: list[tuple[Any, str]] = []
        # Deactivated accounts too: their wedged producer never sees the cancel event.
        for account in sweepable_job_accounts():
            settled.extend(
                (account, run_id)
                for run_id in await _sweep_in_daemon_thread(
                    run_as,
                    account,
                    db.reconcile_runs,
                    error = _LEASE_ERROR,
                    stale_after_ms = int(self._timeout * 1000),
                )
            )
        if not settled:
            return []
        supervisor = getattr(getattr(self.app, "state", None), "chat_generation_supervisor", None)
        for account, run_id in settled:
            logger.warning(
                "chat_generation_run_lease_expired",
                run_id = run_id,
                idle_s = round(self._timeout, 1),
            )
            if supervisor is None:
                continue
            if _run_id_held_by_another_account(account, run_id):
                # Another account started a live run under this id; the slot is theirs.
                logger.warning(
                    "chat_generation_lease_cancel_skipped",
                    run_id = run_id,
                    reason = "another account holds the live registration for this id",
                )
                continue
            # Settled row, but a wedged producer still holds its slot; cancel it inside the owner's namespace
            try:
                run_as(account, supervisor.cancel, run_id)
            except Exception as exc:
                logger.warning(
                    "chat_generation_lease_cancel_failed", run_id = run_id, error = repr(exc)
                )
                continue
            asyncio.create_task(
                self._force_cancel_after_grace(supervisor, run_id, account),
                name = f"chat-generation-lease-force-cancel:{run_id}",
            )
        return [run_id for _account, run_id in settled]

    async def _force_cancel_after_grace(
        self,
        supervisor: Any,
        run_id: str,
        account: Any = None,
    ) -> None:
        """Cancels the task: supervisor.cancel() only sets an Event that a blocked producer never checks."""
        await asyncio.sleep(self._FORCE_CANCEL_GRACE_S)
        task = getattr(supervisor, "_tasks", {}).get(run_id)
        if task is None or task.done():
            return
        if account is not None and _run_id_held_by_another_account(account, run_id):
            # The slot changed hands during the grace period.
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
        # Short wait, not the producer grace: a sweep parked on the writer lock would use up the Desktop exit budget. asyncio.wait, never wait_for(gather(...)).
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


# Pinned by a test against the constant in routes.
_ADMISSION_WAIT_MARKER = ": admission-wait"
# Renew unconditionally: wait renewals are rate limited and the lease equals the first-token timeout
_ADMISSION_DONE_MARKER = ": admission-done"
# A server-side tool still running. Rate limited like the wait marker.
_TOOL_HEARTBEAT_MARKER = ": tool-heartbeat"
# Prefill advancing while prompt_progress is dropped; renews the lease like the tool heartbeat.
_PREFILL_PROGRESS_MARKER = ": prefill-progress"


def _minimum_lease_seconds() -> float:
    """A shorter lease expires between the upstream keep-alive markers and reaps healthy queued runs."""
    from core.inference.llama_admission import DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S

    try:
        from core.inference.llama_admission import llama_admission_config_from_env
        interval = float(llama_admission_config_from_env().keepalive_interval_s)
    except Exception:
        interval = float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)
    # inf passes the upstream parser and makes the lease infinite; an oversized finite cadence stretches it past any horizon.
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
    """Kept separate from the warning wrapper, since renewal consults it on every keep-alive."""
    if configured <= 0.0:
        return 0.0
    return max(configured, _minimum_lease_seconds())


def _clamped_lease_timeout(configured: float) -> float:
    """Raises a lease no renewal can satisfy and logs it; silently keeping it would reap healthy runs."""
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
    """Derived from the lease, not a fixed 30s that renews after a shorter lease expired; a quarter gives 3 renewals, floor 0.25s avoids write per second."""
    lease = _applied_lease_timeout(
        _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS)
    )
    if lease <= 0.0:  # sweeping disabled, so cadence only controls write volume
        return 30.0
    # The floor must stay under the lease.
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
            with active_generations.ActiveGeneration(
                cancel_event,
                run_id = run_id,
                thread_id = thread_id,
                model = model,
            ):
                pass
            return True
        cancel_event = threading.Event()
        # Read by wait_tool_decision. The UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S ceiling ends one wait, not the run; the lease sweeper bounds a stalled producer.
        cancel_event.durable = True
        cancel_event.durable_run_id = run_id
        # Same scope ActiveGeneration captures below: the id alone is not unique across accounts.
        cancel_event.durable_account_id = current_account_id() or ""
        # Parking makes no progress, so without renewal the sweeper caps an attended approval at the lease.
        cancel_event.renew_lease = lambda: db.touch_progress(run_id)
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
                task.cancel()
        active_generations.cancel_run(run_id)
        # Inference cancel registry closes the gap where registration is imminent but not yet observed here
        from routes.inference import _cancel_by_cancel_id_or_stash

        _cancel_by_cancel_id_or_stash(run_id)

    async def stop(self) -> None:
        self._stopping = True
        # before the runs, so the sweeper cannot settle a run as stalled while shutdown is settling it as interrupted
        sweeper = getattr(getattr(self.app, "state", None), "chat_generation_lease_sweeper", None)
        if sweeper is not None:
            await sweeper.stop()
        tasks = list(self._tasks.items())
        self._shutdown_runs.update(run_id for run_id, _task in tasks)
        for run_id, _task in tasks:
            self.cancel(run_id)
        if not tasks:
            return
        # asyncio.wait, not wait_for(gather(...)): wait_for awaits the cancelled inner future, which can hang.
        pending = {task for _run_id, task in tasks}
        _done, pending = await asyncio.wait(pending, timeout = _SHUTDOWN_GRACE_SECONDS)
        if not pending:
            return
        for task in pending:
            task.cancel()
        _done, pending = await asyncio.wait(pending, timeout = _SHUTDOWN_CANCEL_SECONDS)
        if pending:
            stuck = [run_id for run_id, task in tasks if task in pending]
            # Abandoned, not leaked: reconcile_orphaned_runs settles it on the next boot.
            logger.warning(
                "Durable chat generations did not stop within the shutdown budget: %s",
                ", ".join(stuck),
            )

    # Total time, not a count: the interval derives from the lease. Capped so a preparation that never returns cannot keep its lease alive forever.
    _PREPARE_RENEW_MAX_SECONDS = 2 * 60 * 60

    @contextlib.asynccontextmanager
    async def _lease_heartbeat(self, run_id: str):
        """Keeps the lease alive while queued or preparing; nothing else renews it before streaming."""
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
        """Lock contention costs one missed stamp, not a failure, so a busy SQLite writer cannot
        abort it."""
        try:
            await asyncio.to_thread(db.touch_progress, run_id)
        except Exception:
            return

    async def _renew_lease_while_preparing(self, run_id: str) -> None:
        interval = _renew_interval_seconds()
        for _ in range(max(1, int(self._PREPARE_RENEW_MAX_SECONDS / interval))):
            await asyncio.sleep(interval)
            # Skip a contended stamp, not the rest, or a healthy long load is reaped when stamps age out
            await self._try_touch_progress(run_id)

    async def _produce(
        self,
        run_id: str,
        cancel_event: threading.Event | None = None,
        activity: InferenceActivityReservation | None = None,
    ) -> None:
        with track_first_token_wait() as first_token_wait:
            await self._produce_run(run_id, cancel_event, activity, first_token_wait)

    async def _produce_run(
        self,
        run_id: str,
        cancel_event: threading.Event | None,
        activity: InferenceActivityReservation | None,
        first_token_wait: FirstTokenWait,
    ) -> None:
        cancel_event = cancel_event or threading.Event()
        if activity is None:
            activity = InferenceActivityReservation()
            activity.reserve()
        pending: list[tuple[str, dict[str, Any], int]] = []
        iterator: AsyncIterator[Any] | None = None
        # Never await a commit inline: the GGUF tool loop only advances as fast as this producer pulls.
        writing: list[tuple[str, dict[str, Any], int]] = []
        write_task: asyncio.Task | None = None
        held_cancel: asyncio.CancelledError | None = None
        # When the last commit STARTED: a commit slower than a frame must not leave each later chunk its own commit.
        last_flush = -math.inf
        finish_reason: str | None = None
        error: str | None = None
        # Sent on its own chunk, since the finish chunk still reads `stop`.
        quote_cut = False
        saw_done = False
        worker_token: str | None = None
        next_raw_task: asyncio.Task | None = None

        def start_write() -> None:
            nonlocal writing, pending, write_task, last_flush
            writing, pending = pending, []
            last_flush = time.monotonic()
            write_task = asyncio.create_task(
                asyncio.to_thread(db.append_events, run_id, worker_token, writing)
            )

        async def settle_write() -> BaseException | None:
            """Wait out the in-flight commit and return its failure; a failed batch goes back in front of pending.
            Never cancelled: finish_run must not overtake the thread, so a cancel is held until the run settles."""
            nonlocal writing, pending, write_task, held_cancel
            if write_task is None:
                return None
            while not write_task.done():
                try:
                    await asyncio.wait({write_task})
                except asyncio.CancelledError as exc:
                    held_cancel = exc
            task, write_task = write_task, None
            failure = asyncio.CancelledError() if task.cancelled() else task.exception()
            if failure is not None:
                pending = writing + pending
            writing = []
            return failure

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
                    error = "Unsloth shut down during generation" if shutting_down else None,
                )
                return
            # Covers the lifecycle gate too: a queued run's lease ages with nothing renewing it.
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
                        error = "Unsloth shut down during generation" if shutting_down else None,
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

                request_payload = dict(run["requestPayload"])
                timezone_headers = request_payload.pop(db.TIMEZONE_HEADERS_FIELD, None)
                via_api_key = request_payload.pop(db.API_MONITOR_ORIGIN_FIELD, None)
                payload = ChatCompletionRequest.model_validate(request_payload)
                # Prep runs before the first-token budget; one later touch cannot cover prep longer than the lease
                response = await produce_openai_chat_completions(
                    payload,
                    _background_request(
                        self.app, run_id, cancel_event, timezone_headers, via_api_key = via_api_key
                    ),
                    owner,
                    cancel_on_disconnect = False,
                )
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
                timeout = None
                if pending and write_task is None:
                    since_flush = time.monotonic() - last_flush
                    if since_flush >= _EVENT_FLUSH_SECONDS or len(pending) >= _MAX_PENDING_EVENTS:
                        start_write()
                    else:
                        timeout = _EVENT_FLUSH_SECONDS - since_flush
                # Wake to renew the lease across a first-token wait the backend is still inside its own deadline for.
                renew_interval = _renew_interval_seconds()
                timeout = renew_interval if timeout is None else min(timeout, renew_interval)
                waiting = {next_raw_task} if len(pending) < _MAX_PENDING_EVENTS else set()
                if write_task is not None:
                    waiting.add(write_task)
                ready, _waiting = await asyncio.wait(
                    waiting, timeout = timeout, return_when = asyncio.FIRST_COMPLETED
                )
                if (
                    first_token_wait.active()
                    and time.monotonic() - last_keepalive >= renew_interval
                ):
                    await self._try_touch_progress(run_id)
                    last_keepalive = time.monotonic()
                if write_task is not None and write_task.done():
                    failure = await settle_write()
                    if failure is not None:
                        raise failure
                if next_raw_task not in ready:
                    continue
                try:
                    raw = next_raw_task.result()
                except StopAsyncIteration:
                    next_raw_task = None
                    break
                next_raw_task = asyncio.create_task(iterator.__anext__())
                text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                # `: keep-alive` means nothing was produced, so renewing on it would keep a wedged run alive.
                if _ADMISSION_DONE_MARKER in text:
                    last_keepalive = time.monotonic()
                    await self._try_touch_progress(run_id)
                elif (
                    _ADMISSION_WAIT_MARKER in text
                    or _TOOL_HEARTBEAT_MARKER in text
                    or _PREFILL_PROGRESS_MARKER in text
                ):
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
                    quote_cut = quote_cut or chunk.get("quote_cut") is True
                if saw_done:
                    break

            failure = await settle_write()
            if held_cancel is not None:
                raise held_cancel
            if failure is not None:
                raise failure
            current = await asyncio.to_thread(db.get_run, run_id)
            if current is None:
                return
            if run_id in self._shutdown_runs:
                status = "failed"
                finish_reason = "interrupted"
                error = "Unsloth shut down during generation"
            elif current["cancelRequested"] or (cancel_event.is_set() and error is None):
                # The streaming paths set this event after an in-band error too, so a parsed failure outranks it.
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
                quote_cut = quote_cut,
            )
            pending = []
        except asyncio.CancelledError:
            await settle_write()
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
            await settle_write()
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
                        else "Unsloth shut down during generation"
                        if shutting_down
                        else str(exc)[:1000]
                    ),
                    pending_events = pending,
                )
            pending = []
            if held_cancel is not None:
                raise held_cancel
        finally:
            if next_raw_task is not None:
                if not next_raw_task.done():
                    next_raw_task.cancel()
                await asyncio.gather(next_raw_task, return_exceptions = True)
            await _close_iterator(iterator)
            activity.finish()
