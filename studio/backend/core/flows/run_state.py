# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A run's in-memory state: node states, the scope environment, and its buffered events, results and cache rows."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from core.flows.compiler import Plan
from utils.account_context import AccountContext

PROGRESS_MIN_INTERVAL_S = 0.2
MEMORY_EVENTS = 4000


def _now() -> float:
    return time.time()


@dataclass(slots = True)
class NodeState:
    status: str = "idle"
    outputs: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None
    message: Optional[str] = None
    partial: Optional[dict] = None

    def snapshot(self) -> dict:
        found: dict[str, Any] = {"status": self.status}
        if self.outputs is not None:
            found["outputs"] = self.outputs
        if self.error:
            found["error"] = self.error
        if self.progress:
            found["progress"] = self.progress
        if self.message:
            found["message"] = self.message
        if self.partial:
            found["partial"] = self.partial
        return found


class Env:
    """Node outputs visible to a scope; a For Each iteration layers its own over the run's."""

    __slots__ = ("parent", "values")

    def __init__(self, parent: Optional["Env"] = None):
        self.parent = parent
        self.values: dict[str, Any] = {}

    def has(self, node_id: str) -> bool:
        return node_id in self.values or (self.parent is not None and self.parent.has(node_id))

    def get(self, node_id: str) -> Any:
        if node_id in self.values:
            return self.values[node_id]
        if self.parent is not None:
            return self.parent.get(node_id)
        return None


@dataclass
class RunState:
    id: str
    flow_id: str
    account: AccountContext
    doc: dict
    plan: Plan
    options: dict
    persist: bool = True
    status: str = "queued"
    seq: int = 0
    events: deque = field(default_factory = lambda: deque(maxlen = MEMORY_EVENTS))
    _unsaved_events: list[tuple[int, dict]] = field(default_factory = list, repr = False)
    _unsaved_results: dict[str, tuple] = field(default_factory = dict, repr = False)
    _unsaved_cache: dict[str, tuple[str, dict]] = field(default_factory = dict, repr = False)
    # The batch being written: still a cache hit until it commits.
    _writing_cache: dict[str, tuple[str, dict]] = field(default_factory = dict, repr = False)
    _unsaved_touch: set[str] = field(default_factory = set, repr = False)
    _persist_lock: asyncio.Lock = field(default_factory = asyncio.Lock, repr = False)
    subscribers: set[asyncio.Queue] = field(default_factory = set)
    cancel_event: asyncio.Event = field(default_factory = asyncio.Event)
    nodes: dict[str, NodeState] = field(default_factory = dict)
    outputs: dict[str, Any] = field(default_factory = dict)
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    caller: Any = None
    key_id: Optional[int] = None
    app: Any = None
    resident: dict[str, Optional[str]] = field(default_factory = dict)
    initial_resident: dict[str, list[str]] = field(default_factory = dict)
    resident_variant: dict[str, Optional[str]] = field(default_factory = dict)
    fresh: set[str] = field(default_factory = set)
    prefetch_task: Optional[asyncio.Task] = field(default = None, repr = False)
    prefetched: int = 0
    prefetch_load: Optional[tuple[str, str]] = None
    loaded: dict[tuple[str, str], str] = field(default_factory = dict)
    gpu_ops: int = 0
    gpu_parent: Optional["RunState"] = field(default = None, repr = False)
    cancel_hooks: dict[tuple, list[Callable[[], Awaitable[None]]]] = field(default_factory = dict)
    node_tasks: dict[str, asyncio.Task] = field(default_factory = dict)
    _last_progress: dict[str, float] = field(default_factory = dict)
    _pending_progress: dict[str, dict] = field(default_factory = dict)
    _pending_text: dict[tuple[str, str], list[str]] = field(default_factory = dict)
    _pending_partial: dict[tuple, dict] = field(default_factory = dict)
    executed: set[str] = field(default_factory = set)
    ever_failed: set[str] = field(default_factory = set)
    input_overrides: dict[str, dict] = field(default_factory = dict)
    finished_at: Optional[float] = None
    hf_token: Optional[str] = field(default = None, repr = False)
    origin: dict = field(default_factory = dict)
    approvals: dict = field(default_factory = dict)
    marks: dict[str, float] = field(default_factory = dict)
    background: bool = False
    load_keys: dict[tuple, Optional[str]] = field(default_factory = dict, repr = False)

    def node(self, node_id: str) -> NodeState:
        state = self.nodes.get(node_id)
        if state is None:
            state = self.nodes[node_id] = NodeState()
        return state

    def emit(
        self,
        type_: str,
        node_id: Optional[str] = None,
        data: Optional[dict] = None,
    ) -> dict:
        self.seq += 1
        event = {
            "seq": self.seq,
            "run_id": self.id,
            "type": type_,
            "node_id": node_id,
            "ts": _now(),
            "data": data or {},
        }
        self.events.append(event)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if self.persist:
            stored = event
            if "preview" in event["data"]:
                stored = {
                    **event,
                    "data": {k: v for k, v in event["data"].items() if k != "preview"},
                }
            elif type_ == "node.partial":
                # Partials often repeat everything so far: store the position only, or the table grows quadratically.
                stored = {**event, "data": {k: v for k, v in event["data"].items() if k != "value"}}
            self._unsaved_events.append((self.seq, stored))
        return event

    def pending_cache(self, key: str) -> Optional[tuple[str, dict]]:
        return self._unsaved_cache.get(key) or self._writing_cache.get(key)

    def take_unsaved(self) -> Optional[tuple[list, list, list, tuple]]:
        if not (
            self._unsaved_events
            or self._unsaved_results
            or self._unsaved_cache
            or self._unsaved_touch
        ):
            return None
        batch = (
            self._unsaved_events,
            [(nid, *row) for nid, row in self._unsaved_results.items()],
            [(key, *row) for key, row in self._unsaved_cache.items()],
            tuple(self._unsaved_touch),
        )
        self._writing_cache = self._unsaved_cache
        self._unsaved_events, self._unsaved_results, self._unsaved_cache = [], {}, {}
        self._unsaved_touch = set()
        return batch

    def set_status(
        self,
        node_id: str,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        state = self.node(node_id)
        state.status = status
        state.message = message
        if status == "failed":
            self.ever_failed.add(node_id)
        data: dict[str, Any] = {"status": status}
        if message:
            data["message"] = message
        if state.error and status == "failed":
            data["error"] = state.error
        self.emit("node.status", node_id, data)

    def progress(self, node_id: str, data: dict) -> None:
        now = time.monotonic()
        last = self._last_progress.get(node_id)
        self.node(node_id).progress = {k: v for k, v in data.items() if k != "preview"}
        if last is not None and now - last < PROGRESS_MIN_INTERVAL_S:
            self._pending_progress[node_id] = data
            return
        self._last_progress[node_id] = now
        self._pending_progress.pop(node_id, None)
        self.emit("node.progress", node_id, data)

    def text_delta(self, node_id: str, port: str, delta: str) -> None:
        self.marks.setdefault("first_text", time.monotonic())
        self._pending_text.setdefault((node_id, port), []).append(delta)

    def partial(self, node_id: str, data: dict) -> None:
        """Keep the newest value per (node, port, iteration); the flusher stores one event per interval."""
        state = self.node(node_id)
        state.partial = {**(state.partial or {}), data["port"]: data}
        key = (node_id, data["port"], tuple(data.get("iteration") or ()))
        self._pending_partial[key] = data

    def flush(self, node_id: Optional[str] = None) -> None:
        for nid in list(self._pending_progress):
            if node_id is None or nid == node_id:
                self._last_progress[nid] = time.monotonic()
                self.emit("node.progress", nid, self._pending_progress.pop(nid))
        for key in list(self._pending_text):
            if node_id is None or key[0] == node_id:
                parts = self._pending_text.pop(key)
                if parts:
                    self.emit("node.text_delta", key[0], {"port": key[1], "delta": "".join(parts)})
        for key in list(self._pending_partial):
            if node_id is None or key[0] == node_id:
                self.emit("node.partial", key[0], self._pending_partial.pop(key))

    def snapshot(self) -> dict:
        return {
            "run": self.summary(),
            "nodes": {nid: state.snapshot() for nid, state in self.nodes.items()},
            "last_seq": self.seq,
        }

    def summary(self) -> dict:
        return {
            "id": self.id,
            "flow_id": self.flow_id,
            "status": self.status,
            "error": self.error,
            "outputs": self.outputs,
        }

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()
