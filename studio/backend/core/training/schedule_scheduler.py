# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Starts a training run at the time the user scheduled it, while Studio is up."""

from __future__ import annotations

import asyncio
import contextlib
import weakref
from typing import Any, Optional

from fastapi import HTTPException

from core.training.account_jobs import job_accounts
from core.training.schedules import request_from_payload
from loggers import get_logger
from storage import training_schedules_db as db
from utils.account_context import AccountContext, arun_as, run_as

logger = get_logger(__name__)

# Longest sleep: asyncio's clock stops while a computer sleeps, so a start due meanwhile runs at most this late.
_POLL_SECONDS = 30.0
# Floor after a pass that claimed nothing, so an unclaimable due start cannot spin the loop.
_MIN_SLEEP_SECONDS = 1.0
_STOP_GRACE_SECONDS = 5.0
_SETTLE_ATTEMPTS = 3
_SETTLE_RETRY_SECONDS = 1.0
_ERROR_TEXT_LIMIT = 500

_schedulers: "weakref.WeakSet[TrainingScheduleScheduler]" = weakref.WeakSet()


def retire_account_training_schedules(account: AccountContext) -> None:
    """The account is going away: drop its waiting start rather than fire it for a removed account."""
    run_as(account, db.cancel_all_active)
    for scheduler in list(_schedulers):
        scheduler.wake()


def _error_text(message: Any) -> str:
    text = str(message or "").strip() or "Training could not be started."
    return text if len(text) <= _ERROR_TEXT_LIMIT else f"{text[:_ERROR_TEXT_LIMIT]}…"


def _account_is_active(account: AccountContext) -> bool:
    return any(entry.account_id == account.account_id for entry in job_accounts())


class TrainingScheduleScheduler:
    """One loop, one scheduled start at a time; the start itself runs the normal training route."""

    def __init__(self, app: Any) -> None:
        _schedulers.add(self)
        self.app = app
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Made in start(): an asyncio.Event is bound to its loop, and a second lifespan may run on another.
        self._wake: Optional[asyncio.Event] = None
        self._stopping = False
        self._active: Optional[str] = None
        self._last_account: Optional[str] = None
        self._due: Optional[int] = None
        # Accounts whose previous process's claims have been settled by this one.
        self._reconciled: set[str] = set()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(
            self._run_loop(), name = "training-schedule-scheduler"
        )

    async def stop(self) -> None:
        self._stopping = True
        self.wake()
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

    async def _run_loop(self) -> None:
        while not self._stopping:
            # Cleared before looking, so a wake that lands while this pass runs is not lost.
            self._wake.clear()
            try:
                claimed = await asyncio.to_thread(self._claim)
                if claimed is not None:
                    account, schedule = claimed
                    # Shutdown began after the claim: give it back, it runs at the next start.
                    if self._stopping:
                        await asyncio.to_thread(self._release, account, schedule)
                        return
                    await arun_as(account, self._execute(account, schedule))
                    continue
                # Sleep until the next start is due; wake() cuts it short on any change.
                delay = _POLL_SECONDS
                if self._due is not None:
                    delay = min(
                        delay, max(_MIN_SLEEP_SECONDS, (self._due - db.now_ms()) / 1000)
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never escape the loop: that would stop every scheduled start for the life of the process.
                logger.exception("training_schedules.loop_iteration_failed")
                delay = 5.0
            if self._stopping:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout = delay)

    def _accounts(self) -> list[AccountContext]:
        accounts = job_accounts()
        for index, account in enumerate(accounts):
            if account.account_id == self._last_account:
                return accounts[index + 1 :] + accounts[: index + 1]
        return accounts

    def _claim(self) -> Optional[tuple[AccountContext, dict]]:
        """One pass, one read per account: claim the next due start, else note in ``_due`` when one is."""
        now = db.now_ms()
        self._due = None
        accounts = self._accounts()
        self._reconciled &= {account.account_id for account in accounts}
        for account in accounts:
            if self._stopping:
                return None
            if account.account_id not in self._reconciled:
                try:
                    # A claim a previous process left behind never began; give it back before claiming.
                    run_as(account, db.reconcile_startup, now)
                    self._reconciled.add(account.account_id)
                except Exception:
                    logger.exception("training_schedules.reconcile_failed_for_account")
                    continue
            try:
                schedule, due = run_as(account, db.poll, now)
            except Exception:
                logger.exception("training_schedules.claim_failed_for_account")
                continue
            if schedule is not None:
                if self._stopping:
                    self._release(account, schedule)
                    return None
                self._last_account = account.account_id
                return account, schedule
            if due is not None and (self._due is None or due < self._due):
                self._due = due
        return None

    def _release(self, account: AccountContext, schedule: dict) -> None:
        try:
            run_as(account, db.release_claim, schedule["id"])
        except Exception:
            logger.exception("training_schedules.release_failed")

    async def _execute(self, account: AccountContext, schedule: dict) -> None:
        schedule_id = schedule["id"]
        self._active = schedule_id
        try:
            if not await asyncio.to_thread(_account_is_active, account):
                await self._settle(
                    schedule_id,
                    db.MISSED,
                    error = "The account was removed before the scheduled run could start.",
                )
                return
            payload = await asyncio.to_thread(db.get_payload, schedule_id)
            if payload is None:
                await self._settle(
                    schedule_id,
                    db.FAILED,
                    error = "The saved training configuration could not be read. Schedule the run again.",
                )
                return
            try:
                request = request_from_payload(payload)
            except ValueError:
                logger.warning("training_schedules.invalid_payload", schedule = schedule_id)
                await self._settle(
                    schedule_id,
                    db.FAILED,
                    error = "The saved training configuration is no longer valid. Schedule the run again.",
                )
                return

            from routes.training import start_training

            try:
                response = await start_training(
                    request,
                    current_subject = account.username,
                    via_api_key = False,
                )
            except HTTPException as exc:
                await self._settle(schedule_id, db.FAILED, error = _error_text(exc.detail))
                return
            except Exception as exc:  # noqa: BLE001 - the schedule records the failure
                logger.exception("training_schedules.start_failed", schedule = schedule_id)
                await self._settle(schedule_id, db.FAILED, error = _error_text(exc))
                return

            if response.status == "error":
                await self._settle(
                    schedule_id,
                    db.FAILED,
                    job_id = response.job_id or None,
                    error = _error_text(response.message),
                )
            else:
                await self._settle(
                    schedule_id,
                    db.STARTED,
                    job_id = response.job_id or None,
                )
        except asyncio.CancelledError:
            # Studio stopping is not the run's failure: hand the occurrence to the next start unless
            # the spawn had already got past the point where training is active.
            if await asyncio.to_thread(self._training_active):
                await self._settle(schedule_id, db.STARTED)
            else:
                await asyncio.to_thread(db.release_claim, schedule_id)
            raise
        finally:
            self._active = None

    @staticmethod
    def _training_active() -> bool:
        try:
            from core.training import get_training_backend

            return bool(get_training_backend().is_training_active())
        except Exception:
            return False

    async def _settle(
        self,
        schedule_id: str,
        status: str,
        *,
        job_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        for attempt in range(_SETTLE_ATTEMPTS):
            try:
                settled = await asyncio.to_thread(
                    db.finish_claim,
                    schedule_id,
                    status,
                    job_id = job_id,
                    error = error,
                )
                if settled:
                    return
            except Exception:
                logger.exception("training_schedules.finish_failed")
            if attempt + 1 < _SETTLE_ATTEMPTS:
                await asyncio.sleep(_SETTLE_RETRY_SECONDS)
