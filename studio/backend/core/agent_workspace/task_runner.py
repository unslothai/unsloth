# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded execution lanes for durable project tasks.

The executor is trusted server code. It must enforce the captured runtime and
workspace policy, check ownership before each operation, pass cancellation to
active operations, and release any model admission lease before waiting for a
child. This engine owns scheduling; it does not confer tool or filesystem access.
"""

from __future__ import annotations

import copy
import logging
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from . import task_state as state

log = logging.getLogger(__name__)


class TaskCancelled(state.TaskStateError):
    pass


@dataclass(frozen = True)
class TaskContext:
    """One attempt's private capability, never a renderer-supplied session ID."""

    _runner: ProjectTaskRunner = field(repr = False)
    _work: _Work = field(repr = False)

    @property
    def task(self) -> dict:
        return copy.deepcopy(self._work.task)

    @property
    def cancel_event(self) -> threading.Event:
        return self._work.cancel

    @property
    def deadline(self) -> float:
        return self._work.deadline

    def check(self) -> dict:
        if self.cancel_event.is_set() or time.monotonic() >= self.deadline:
            raise TaskCancelled("Task cancellation or deadline reached.")
        return state.validate_owner(self._work.task["id"], self._work.owner)

    def delegate(
        self,
        instruction: str,
        *,
        role: str,
        max_output_tokens: int = 4096,
    ) -> dict:
        self.check()
        return self._runner._delegate(self._work, instruction, role, max_output_tokens)

    def wait_child(
        self,
        task_id: str,
        *,
        timeout: float = 60,
    ) -> dict | None:
        self.check()
        child = state.get_task(self._work.task["projectId"], task_id)
        if child["parentId"] != self._work.task["id"]:
            raise state.TaskStateError("Only this attempt's children can be awaited.")
        return self._runner.wait(child["projectId"], task_id, timeout = timeout, check = self.check)

    def retry_child(self, task_id: str) -> dict:
        self.check()
        child = state.get_task(self._work.task["projectId"], task_id)
        if child["parentId"] != self._work.task["id"]:
            raise state.TaskStateError("Only this attempt's children can be retried.")
        return self._runner._retry(child, parent = self._work)


@dataclass
class _Work:
    task: dict
    owner: str
    deadline: float
    cancel: threading.Event = field(default_factory = threading.Event)


def _seconds(value: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise state.TaskStateError("Invalid task time bound.")
    return float(value)


class ProjectTaskRunner:
    """Explicit submission only: opening a runner never replays stored work."""

    def __init__(
        self,
        executor: Callable[[TaskContext], dict],
        *,
        root_workers: int = 2,
        child_workers: int = 2,
        heartbeat_seconds: float = 1,
        drain_seconds: float = 10,
    ):
        if not callable(executor):
            raise TypeError("A server task executor is required.")
        state._integer(root_workers, 1, 4, "root-worker bound")
        state._integer(child_workers, 1, 8, "child-worker bound")
        self._heartbeat_seconds = _seconds(heartbeat_seconds, state.LEASE_MS / 3000)
        self._drain_seconds = _seconds(drain_seconds, 30)
        self._executor = executor
        self._condition = threading.Condition(threading.RLock())
        self._work: dict[str, _Work] = {}
        self._closing = False
        self._lanes = {
            "root": queue.Queue(maxsize = state.MAX_ACTIVE_TASKS),
            "child": queue.Queue(maxsize = state.MAX_ACTIVE_TASKS),
        }
        self._workers = []
        state.reconcile_expired_tasks()
        for lane, count in (("root", root_workers), ("child", child_workers)):
            for index in range(count):
                thread = threading.Thread(
                    target = self._worker,
                    args = (lane,),
                    daemon = True,
                    name = f"studio-task-{lane}-{index}",
                )
                self._workers.append(thread)
                thread.start()
        self._watcher = threading.Thread(target = self._watch, daemon = True, name = "studio-task-leases")
        self._watcher.start()

    def _open(self):
        if self._closing:
            raise state.TaskStateError("The task runner is shutting down.")

    def _enqueue(self, task: dict, deadline: float) -> dict:
        owner = str(uuid.uuid4())
        state.reserve_dispatch(task["projectId"], task["id"], owner)
        work = _Work(task = task, owner = owner, deadline = deadline)
        lane = "child" if task["parentId"] else "root"
        try:
            with self._condition:
                self._open()
                self._work[task["id"]] = work
                try:
                    self._lanes[lane].put_nowait(work)
                except queue.Full:
                    self._work.pop(task["id"], None)
                    raise state.TaskStateError("The task worker queue is full.") from None
                self._condition.notify_all()
        except state.TaskStateError:
            state.cancel_task(task["projectId"], task["id"])
            raise
        return task

    def submit(
        self,
        project_id: str,
        instruction: str,
        snapshot: dict,
        *,
        max_output_tokens: int = 8192,
        child_limit: int = 0,
        child_budget: int = 0,
        timeout: float = 900,
    ) -> dict:
        duration = _seconds(timeout, 3600)
        with self._condition:
            self._open()
        task = state.create_task(
            project_id,
            instruction,
            snapshot,
            max_output_tokens = max_output_tokens,
            child_limit = child_limit,
            child_budget = child_budget,
        )
        return self._enqueue(task, time.monotonic() + duration)

    def _delegate(self, parent: _Work, instruction: str, role: str, max_output_tokens: int) -> dict:
        with self._condition:
            self._open()
        task = state.create_child(
            parent.task["id"],
            parent.owner,
            instruction,
            parent.task["snapshot"],
            role = role,
            max_output_tokens = max_output_tokens,
        )
        return self._enqueue(task, parent.deadline)

    def _retry(
        self,
        task: dict,
        *,
        parent: _Work | None = None,
        timeout: float = 900,
    ) -> dict:
        duration = _seconds(timeout, 3600)
        with self._condition:
            self._open()
            if task["id"] in self._work or (
                parent is None
                and any(w.task["rootId"] == task["rootId"] for w in self._work.values())
            ):
                raise state.TaskStateError("The previous attempt's workers have not stopped.")
        retried = state.retry_task(
            task["projectId"], task["id"], parent_owner = parent.owner if parent else None
        )
        return self._enqueue(retried, parent.deadline if parent else time.monotonic() + duration)

    def retry(
        self,
        project_id: str,
        task_id: str,
        *,
        timeout: float = 900,
    ) -> dict:
        task = state.get_task(project_id, task_id)
        if task["parentId"]:
            raise state.TaskStateError("Child retries require the active parent worker.")
        return self._retry(task, timeout = timeout)

    def cancel(self, project_id: str, task_id: str) -> dict:
        with self._condition:
            for work in self._work.values():
                if work.task["projectId"] == project_id and (
                    work.task["id"] == task_id or work.task["parentId"] == task_id
                ):
                    work.cancel.set()
            self._condition.notify_all()
        return state.cancel_task(project_id, task_id)

    def project_is_idle(self, project_id: str) -> bool:
        """Durable expiry alone cannot prove an in-process executor has stopped."""
        with self._condition:
            if any(work.task["projectId"] == project_id for work in self._work.values()):
                return False
        return not state.project_has_active_tasks(project_id)

    def wait(
        self,
        project_id: str,
        task_id: str,
        *,
        timeout: float = 60,
        check: Callable[[], object] | None = None,
    ) -> dict | None:
        deadline = time.monotonic() + _seconds(timeout, 3600)
        while True:
            if check is not None:
                check()
            row = state.get_task(project_id, task_id)
            with self._condition:
                if row["status"] in state.TERMINAL and task_id not in self._work:
                    return row
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.05))

    def _watch(self):
        while True:
            with self._condition:
                if self._closing and not self._work:
                    return
                work_items = list(self._work.values())
            for work in work_items:
                try:
                    if work.cancel.is_set() or time.monotonic() >= work.deadline:
                        self.cancel(work.task["projectId"], work.task["id"])
                    if not state.heartbeat(work.task["id"], work.owner):
                        work.cancel.set()
                except state.TaskStateError:
                    work.cancel.set()
                except Exception:
                    work.cancel.set()
                    log.exception("Failed to renew project task ownership")
            with self._condition:
                self._condition.wait(self._heartbeat_seconds)

    def _settle_children(self, work: _Work) -> bool:
        children = state.list_children(work.task["projectId"], work.task["id"])
        for child in children:
            if child["status"] not in state.TERMINAL:
                self.cancel(child["projectId"], child["id"])
        deadline = time.monotonic() + self._drain_seconds
        for child in children:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or self.wait(child["projectId"], child["id"], timeout = remaining) is None
            ):
                return False
        return True

    def _worker(self, lane: str):
        while True:
            try:
                work = self._lanes[lane].get(timeout = 0.05)
            except queue.Empty:
                with self._condition:
                    if self._closing:
                        return
                continue
            try:
                work.task = state.claim_task(work.task["projectId"], work.task["id"], work.owner)
                context = TaskContext(self, work)
                result, error, status = None, None, "completed"
                try:
                    context.check()
                    result = self._executor(context)
                    if not isinstance(result, dict):
                        raise state.TaskStateError("A task executor must return an object.")
                    state._json(result, 1024 * 1024, "Task result")
                    context.check()
                except TaskCancelled:
                    status = "cancelled"
                except BaseException as exc:
                    # Exceptions can contain provider credentials. Persist a
                    # bounded generic diagnostic; the executor owns safe detail.
                    # SystemExit from an executor must not kill this worker lane.
                    status = "failed"
                    result = None
                    error = f"Task executor failed ({type(exc).__name__})."
                if not self._settle_children(work):
                    state.interrupt_task(work.task["id"], work.owner)
                else:
                    state.finish_task(
                        work.task["id"], work.owner, status, result = result, error = error
                    )
            except state.TaskStateError:
                # A cancellation/expiry may win before claim or finalization.
                # Never resurrect ownership or publish a stale result.
                pass
            except Exception:
                work.cancel.set()
                log.exception("Project task worker could not finalize its attempt")
            finally:
                self._lanes[lane].task_done()
                with self._condition:
                    self._work.pop(work.task["id"], None)
                    self._condition.notify_all()

    def shutdown(self, *, timeout: float = 10) -> bool:
        """Cancel and drain; False means an executor still has not returned."""
        deadline = time.monotonic() + _seconds(timeout, 60)
        with self._condition:
            self._closing = True
            for work in self._work.values():
                work.cancel.set()
            self._condition.notify_all()
        for thread in [*self._workers, self._watcher]:
            thread.join(max(0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in [*self._workers, self._watcher])
