# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Flows run supervisor.

One asyncio task per run, bound to the run's account. Nodes run in data order: CPU nodes up to ``CPU_CONCURRENCY`` at a
time, GPU nodes one at a time process-wide (the GPU lock) and, among ready GPU nodes, the one that keeps the resident
model first, so a flow swaps models as rarely as it can. If and Switch skip the branch they do not take: a node whose
required input was skipped is skipped too, and First Available joins branches. Outputs are cached by content key;
side-effecting nodes never are. Every event carries a per-run ``seq`` and is stored, so a client replays from any
point. Cancelling stops admitting nodes, asks in-flight nodes to stop through the routes they called, and keeps what
already finished. Each run reaches Studio as its own internal API key, revoked when the run ends."""

from __future__ import annotations

import asyncio
import contextvars
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from core.flows import assets as asset_store
from core.flows import capabilities
from core.flows import caller as flow_caller
from core.flows.cache import cache_key
from core.flows.compiler import FOR_EACH, CompileError, Plan, compile_flow
from core.flows.registry import FlowCancelled, FlowNodeError, load_handlers
from core.flows.schema import MAX_FOREACH_ITEMS, MAX_GPU_OPS_PER_RUN, MAX_RUN_SECONDS
from storage import flows_db
from utils.account_context import AccountContext, arun_as, current_account

logger = logging.getLogger(__name__)

CPU_CONCURRENCY = 4
PROGRESS_MIN_INTERVAL_S = 0.2
# A streamed player answer is flushed this often, so its text reads as typed.
INVOKE_FLUSH_S = 0.05
CANCEL_GRACE_S = 10.0
LEASE_RETRY_S = 2.0
# Run Flow nests at most this many flows below the one the user started.
MAX_CALL_DEPTH = 3
MEMORY_EVENTS = 4000
FINISHED_RUNS_KEPT = 64

# The slot each model kind loads into; models sharing a slot evict each other.
MODEL_SLOT = {
    "chat": "llm",
    "vision": "llm",
    "tts": "llm",
    "audio": "llm",
    "image": "image",
    "video": "video",
    "stt": "stt",
    "embedding": "embedding",
}
JOIN_TYPES = ("logic.merge",)
LOAD_KIND = {"chat": "llm", "vision": "llm", "tts": "tts", "audio": "audio"}


class _Skip:
    def __repr__(self) -> str:
        return "SKIPPED"


SKIPPED = _Skip()
# Blocks a schedule or webhook fires; in a run one of them started, the others do not run.
TRIGGER_TYPES = ("trigger.schedule", "trigger.webhook")


def _now() -> float:
    return time.time()


def _derive_seed(
    flow_seed: int,
    node_id: str,
    iteration: Optional[tuple] = None,
) -> int:
    material = f"{flow_seed}:{node_id}:{iteration or ''}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**31)


# Set while cancel hooks run: their own Studio calls (the cancel endpoints) must go through.
_IN_CANCEL_HOOK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "flows_in_cancel_hook", default = False
)
_LATE_HOOKS: set[asyncio.Task] = set()


async def _run_cancel_hook(fn: Callable[[], Awaitable[None]]) -> None:
    token = _IN_CANCEL_HOOK.set(True)
    try:
        await asyncio.wait_for(fn(), timeout = 15)
    except Exception:
        logger.debug("flows: cancel hook failed", exc_info = True)
    finally:
        _IN_CANCEL_HOOK.reset(token)


def _clamp(spec_param: dict, value: Any) -> Any:
    """Numbers stay inside the catalog's min/max: documents come from imports and the API, not only the editor."""
    if (
        spec_param.get("kind") not in ("number", "int")
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        return value
    low, high = spec_param.get("min"), spec_param.get("max")
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return int(value) if spec_param["kind"] == "int" else value


def param_values(spec: dict, node: dict) -> dict:
    by_id = {p["id"]: p for p in spec["params"]}
    params = {p["id"]: p.get("default") for p in spec["params"]}
    for key, value in (node.get("params") or {}).items():
        params[key] = _clamp(by_id[key], value) if key in by_id else value
    return params


@dataclass
class NodeState:
    status: str = "idle"
    outputs: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None
    message: Optional[str] = None
    # The last value a long node published per port before it finished; never fed to other nodes.
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
    events: list[dict] = field(default_factory = list)
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
    gpu_ops: int = 0
    # A Run Flow child counts its GPU steps against the run that called it.
    gpu_parent: Optional["RunState"] = field(default = None, repr = False)
    # Keyed by (node id, For Each iteration): parallel iterations of one node each keep their own hooks.
    cancel_hooks: dict[tuple, list[Callable[[], Awaitable[None]]]] = field(default_factory = dict)
    node_tasks: dict[str, asyncio.Task] = field(default_factory = dict)
    _last_progress: dict[str, float] = field(default_factory = dict)
    _pending_progress: dict[str, dict] = field(default_factory = dict)
    _pending_text: dict[tuple[str, str], list[str]] = field(default_factory = dict)
    _pending_partial: dict[tuple, dict] = field(default_factory = dict)
    executed: set[str] = field(default_factory = set)
    # Every node that failed at least once; a later For Each iteration succeeding must not hide it.
    ever_failed: set[str] = field(default_factory = set)
    input_overrides: dict[str, dict] = field(default_factory = dict)
    finished_at: Optional[float] = None
    hf_token: Optional[str] = field(default = None, repr = False)
    # Who started the run ({"kind": "ui" | "api_key" | "trigger" | "player", "owner": bool}) and what the owner
    # approved this flow to do; together they gate Python, HTTP, MCP, provider and trigger blocks.
    origin: dict = field(default_factory = dict)
    approvals: dict = field(default_factory = dict)

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
        if len(self.events) > MEMORY_EVENTS:
            del self.events[: len(self.events) - MEMORY_EVENTS]
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
                # A partial often repeats everything so far (rows collected), so storing each copy grows the table
                # quadratically. Store the position only; live clients and the run snapshot carry the value.
                stored = {**event, "data": {k: v for k, v in event["data"].items() if k != "value"}}
            try:
                flows_db.append_event(self.id, self.seq, stored)
            except Exception:
                logger.warning(
                    "flows: could not store event %s of run %s", self.seq, self.id, exc_info = True
                )
        return event

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


class NodeContextImpl:
    """What a handler sees. See ``core.flows.registry.NodeContext`` for the contract."""

    def __init__(
        self,
        engine: "FlowEngine",
        rs: RunState,
        node_id: str,
        spec: dict,
        params: dict,
        inputs: dict,
        seed: int,
        iteration: Optional[tuple] = None,
    ):
        self._engine = engine
        self._rs = rs
        self.run_id = rs.id
        self.flow_id = rs.flow_id
        self.node_id = node_id
        self.spec = spec
        self.params = params
        self.inputs = inputs
        self.seed = seed
        self.flow_params = rs.options.get("params") or {}
        self.iteration = iteration
        self.doc = rs.doc
        self.origin = dict(rs.origin)
        self.outputs_extra: dict[str, Any] = {}

    def capability_refusal(
        self,
        cap: str,
        detail: Optional[str] = None,
    ) -> Optional[str]:
        """Why this run may not use ``cap`` ("python", "http" + host, "mcp", "provider" + id), or None."""
        return capabilities.refusal(
            self._rs.origin, self._rs.approvals, cap, detail or cap, self.spec["title"]
        )

    def capability_allowed(
        self,
        cap: str,
        detail: Optional[str] = None,
    ) -> bool:
        """For handlers that use a capability only sometimes, e.g. an agent turning on MCP tools."""
        return self.capability_refusal(cap, detail) is None

    async def call(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[dict] = None,
        files: Any = None,
        data: Optional[dict] = None,
        content: Optional[bytes] = None,
        content_type: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> Any:
        if not _IN_CANCEL_HOOK.get():
            self.check_cancelled()
        try:
            response = await flow_caller.forward(
                self._rs.caller,
                method,
                path,
                params = params,
                json_body = json,
                content = content,
                content_type = content_type,
                files = files,
                data = data,
                timeout = timeout_s,
            )
            return flow_caller.parse_body(response)
        except flow_caller.RouteError as exc:
            raise FlowNodeError(
                exc.detail, status = exc.status, retryable = exc.status in (409, 503)
            ) from None

    async def ensure_model(
        self,
        model_kind: str,
        choice: Optional[dict],
        options: Optional[dict] = None,
    ) -> Optional[dict]:
        from core.flows.nodes import _common
        return await _common.ensure_model(self, model_kind, choice, options)

    async def llm_text(self, **kwargs: Any) -> str:
        from core.flows.nodes import _common
        return await _common.llm_text(self, **kwargs)

    async def llm_json(self, **kwargs: Any) -> Any:
        from core.flows.nodes import _common
        return await _common.llm_json(self, **kwargs)

    async def generate_image(self, **kwargs: Any) -> list[dict]:
        from core.flows.nodes import _common
        return await _common.generate_image(self, **kwargs)

    async def remove_background(
        self,
        image: dict,
        subject: str,
        trim: bool = True,
    ) -> tuple[dict, dict]:
        from core.flows.nodes import _common
        return await _common.remove_background(self, image, subject, trim)

    async def segment(self, image: dict, subject: str) -> bytes:
        from core.flows.nodes import _common
        return await _common.segment(self, image, subject)

    async def speak(self, **kwargs: Any) -> dict:
        from core.flows.nodes import _common
        return await _common.speak(self, **kwargs)

    async def run_flow(self, flow_id: str, values: dict) -> dict:
        return await self._engine.run_child(self._rs, self, flow_id, values)

    async def media_bytes(self, media: dict) -> tuple[bytes, str]:
        from core.flows.nodes import _common
        return await _common.media_bytes(self, media)

    async def media_data_url(self, media: dict) -> str:
        data, mime = await self.media_bytes(media)
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    async def save_asset(
        self,
        data: bytes,
        mime: str,
        *,
        kind: str,
        name: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        digest, sniffed = await asyncio.to_thread(asset_store.store, data, mime)
        return asset_store.media_value(digest, sniffed, **(extra or {}))

    def progress(
        self,
        fraction: Optional[float] = None,
        *,
        phase: Optional[str] = None,
        step: Optional[int] = None,
        total: Optional[int] = None,
        preview: Optional[str] = None,
    ) -> None:
        data = {
            k: v
            for k, v in (
                ("fraction", fraction),
                ("phase", phase),
                ("step", step),
                ("total", total),
                ("preview", preview),
            )
            if v is not None
        }
        if self.iteration is not None:
            data["iteration"] = list(self.iteration)
        self._rs.progress(self.node_id, data)

    def log(
        self,
        message: str,
        level: str = "info",
    ) -> None:
        self._rs.emit("node.log", self.node_id, {"level": level, "message": str(message)[:4000]})

    def text_delta(self, port: str, delta: str) -> None:
        self._rs.text_delta(self.node_id, port, delta)

    def partial(
        self,
        port: str,
        value: Any,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        data: dict[str, Any] = {"port": port, "value": value}
        if step is not None:
            data["step"] = step
        if total is not None:
            data["total"] = total
        if self.iteration is not None:
            data["iteration"] = list(self.iteration)
        self._rs.partial(self.node_id, data)

    def set_status(
        self,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        self._rs.set_status(self.node_id, status, message)

    @property
    def cancelled(self) -> bool:
        return self._rs.cancelled

    def check_cancelled(self) -> None:
        if self._rs.cancelled:
            raise FlowCancelled()

    def on_cancel(self, fn: Callable[[], Awaitable[None]]) -> None:
        from core.inference import gpu_arbiter

        holder = gpu_arbiter._lease_holder.get()
        if holder is not None:
            # Stop runs from the cancel request's task; its cleanup (a restore load) still acts as the lease holder.
            inner = fn

            async def fn() -> None:
                token = gpu_arbiter._lease_holder.set(holder)
                try:
                    await inner()
                finally:
                    gpu_arbiter._lease_holder.reset(token)

        if self._rs.cancelled:
            # The run was cancelled while the job was being started: stop the job now.
            task = asyncio.get_running_loop().create_task(_run_cancel_hook(fn))
            _LATE_HOOKS.add(task)
            task.add_done_callback(_LATE_HOOKS.discard)
            return
        self._rs.cancel_hooks.setdefault((self.node_id, self.iteration), []).append(fn)

    @property
    def resident(self) -> dict[str, Optional[str]]:
        return self._rs.resident

    @property
    def caller(self) -> Any:
        return self._rs.caller

    def count_gpu_op(self) -> None:
        rs = self._rs
        while rs.gpu_parent is not None:
            rs = rs.gpu_parent
        rs.gpu_ops += 1
        if rs.gpu_ops > MAX_GPU_OPS_PER_RUN:
            raise FlowNodeError(f"This run passed the limit of {MAX_GPU_OPS_PER_RUN} GPU steps")


class FlowEngine:
    def __init__(self) -> None:
        self.runs: dict[str, RunState] = {}
        self._gpu_lock: Optional[asyncio.Lock] = None
        self._warming: dict[tuple, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.handlers: Optional[dict] = None
        # Tests swap these in.
        self.mint_key: Callable[[AccountContext], tuple[str, Optional[int]]] = (
            flow_caller.mint_run_key
        )
        self.revoke_key: Callable[[Optional[int]], None] = flow_caller.revoke_run_key

    def gpu_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._gpu_lock is None or self._loop is not loop:
            self._gpu_lock = asyncio.Lock()
            self._loop = loop
        return self._gpu_lock

    def _handlers(self) -> dict:
        if self.handlers is None:
            self.handlers = load_handlers()
        return self.handlers

    async def start_run(
        self,
        *,
        app: Any,
        flow_id: str,
        doc: dict,
        options: Optional[dict] = None,
        hf_token: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> RunState:
        """Compile (raises CompileError) and start a run for the current account; returns at once."""
        plan = compile_flow(doc)
        options = dict(options or {})
        account = current_account()
        run_origin = _origin(account, origin)
        options["origin"] = run_origin["kind"]
        run_id = uuid.uuid4().hex
        flows_db.create_run(run_id, flow_id, _stored_options(options))
        rs = RunState(
            id = run_id,
            flow_id = flow_id,
            account = account,
            doc = doc,
            plan = plan,
            options = options,
            app = app,
            origin = run_origin,
            approvals = flows_db.get_capabilities(flow_id),
        )
        rs.hf_token = hf_token
        for nid in plan.order:
            rs.node(nid)
        self.runs[run_id] = rs
        rs.emit("run.queued")
        rs.task = asyncio.create_task(arun_as(account, self._run(rs)), name = f"flow-run-{run_id}")
        self._trim_finished()
        return rs

    def live(self, run_id: str) -> Optional[RunState]:
        rs = self.runs.get(run_id)
        if rs is None or rs.account.account_id != current_account().account_id:
            return None
        return rs

    async def cancel(self, run_id: str) -> Optional[str]:
        rs = self.live(run_id)
        if rs is None:
            stored = flows_db.get_run(run_id)
            return stored["status"] if stored else None
        if rs.status in ("completed", "failed", "cancelled", "interrupted"):
            return rs.status
        if not rs.cancel_event.is_set():
            rs.cancel_event.set()
            flows_db.update_run(run_id, cancel_requested = True, status = "cancelling")
            rs.status = "cancelling"
            rs.emit("run.cancelling")
            await self._fire_cancel_hooks(rs)
        return rs.status

    def subscribe(self, rs: RunState) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize = 10_000)
        rs.subscribers.add(queue)
        return queue

    def unsubscribe(self, rs: RunState, queue: asyncio.Queue) -> None:
        rs.subscribers.discard(queue)

    async def invoke(
        self,
        *,
        app: Any,
        flow_id: str,
        doc: dict,
        entry: str,
        inputs: Optional[dict] = None,
        params: Optional[dict] = None,
        hf_token: Optional[str] = None,
        origin: Optional[str] = "player",
        events: Optional[asyncio.Queue] = None,
    ) -> dict:
        """Run one entrypoint node (and what it needs, cache first) now and return its outputs.
        ``events`` receives the run's events as they happen, text deltas included, for a streamed answer."""
        plan, spec = self._entry(doc, entry)
        account = current_account()
        rs = RunState(
            id = "invoke-" + uuid.uuid4().hex,
            flow_id = flow_id,
            account = account,
            doc = doc,
            plan = plan,
            options = {"params": params or {}, "force_nodes": [entry]},
            persist = False,
            app = app,
            origin = _origin(account, origin),
            approvals = flows_db.get_capabilities(flow_id),
        )
        rs.hf_token = hf_token
        overrides = {
            k: v for k, v in (inputs or {}).items() if any(p["id"] == k for p in spec["inputs"])
        }
        rs.input_overrides[entry] = overrides
        # Ports supplied by the player do not need their upstream nodes.
        need = {entry}
        for port, sources in plan.inputs.get(entry, {}).items():
            if port not in overrides:
                need |= plan.ancestors(src for src, _ in sources)
        token, key_id = await asyncio.to_thread(self.mint_key, account)
        rs.caller = flow_caller.make_caller(token, account, app, hf_token)
        rs.key_id = key_id
        flusher = None
        if events is not None:
            rs.subscribers.add(events)
            flusher = asyncio.create_task(self._flusher(rs, INVOKE_FLUSH_S))
        try:
            env = Env()
            await self._run_scope(
                rs, [n for n in plan.order if n in need and plan.owner.get(n) is None], env
            )
            if flusher is not None:
                rs.flush()
            state = rs.node(entry)
            if state.status == "failed":
                raise FlowNodeError(state.error or "The entry point failed")
            if state.status not in ("done", "cached"):
                raise FlowNodeError(state.message or "The entry point did not run")
            return state.outputs or {}
        finally:
            if flusher is not None:
                flusher.cancel()
                rs.subscribers.discard(events)
            await asyncio.to_thread(self.revoke_key, key_id)

    def _entry(self, doc: dict, entry: str) -> tuple[Any, dict]:
        plan = compile_flow(doc)
        spec = plan.specs.get(entry)
        if spec is None or entry not in plan.executable:
            raise FlowNodeError(f"No runnable node {entry!r} in this flow")
        if not spec.get("entrypoint"):
            raise FlowNodeError(f"{spec['title']} is not an entry point a player can call")
        if plan.owner.get(entry):
            raise FlowNodeError("An entry point cannot sit inside a For Each group")
        return plan, spec

    async def warm(
        self,
        *,
        app: Any,
        flow_id: str,
        doc: dict,
        entry: str,
        hf_token: Optional[str] = None,
    ) -> bool:
        """Load the model an entry point answers with, so a player's first call does not wait for it.
        Calls for the same entry share one load; True once the model is resident (or the entry needs none)."""
        plan, spec = self._entry(doc, entry)
        model_kind = spec.get("model_kind")
        if not model_kind:
            return True
        account = current_account()
        key = (account.account_id, flow_id, entry)
        task = self._warming.get(key)
        if task is None or task.done():
            node = next(n for n in doc.get("nodes") or [] if n.get("id") == entry)
            params = param_values(spec, node)
            task = asyncio.create_task(
                self._warm(app, flow_id, doc, plan, entry, spec, params, account, hf_token)
            )
            self._warming[key] = task
            task.add_done_callback(lambda t, k = key: self._warming.pop(k, None))
        # A player leaving mid-load does not abort it: the next conversation still wants the model.
        return await asyncio.shield(task)

    async def _warm(
        self,
        app: Any,
        flow_id: str,
        doc: dict,
        plan: Any,
        entry: str,
        spec: dict,
        params: dict,
        account: Any,
        hf_token: Optional[str],
    ) -> bool:
        rs = RunState(
            id = "warm-" + uuid.uuid4().hex,
            flow_id = flow_id,
            account = account,
            doc = doc,
            plan = plan,
            options = {"params": {}},
            persist = False,
            app = app,
        )
        rs.hf_token = hf_token
        token, key_id = await asyncio.to_thread(self.mint_key, account)
        rs.caller = flow_caller.make_caller(token, account, app, hf_token)
        rs.key_id = key_id
        ctx = NodeContextImpl(self, rs, entry, spec, params, {}, 0)
        try:
            # Same lane as the entry point itself, so a warm-up never loads under another GPU step.
            async with self.gpu_lock():
                await ctx.ensure_model(spec["model_kind"], params.get("model"))
            return True
        except Exception as exc:
            logger.info("flows: warming %s of %s did not load a model: %s", entry, flow_id, exc)
            return False
        finally:
            await asyncio.to_thread(self.revoke_key, key_id)

    async def run_child(
        self, parent: RunState, ctx: "NodeContextImpl", flow_id: str, values: dict
    ) -> dict:
        """Run another saved flow of the same account inside ``parent`` (Run Flow); returns its Flow Outputs.

        The child shares the parent's run key, cancel signal and GPU lock, so Stop and the one-GPU-step-at-a-time
        rule cover it; it is not stored as a run of its own."""
        chain = list(parent.options.get("call_chain") or [parent.flow_id])
        if flow_id in chain:
            raise FlowNodeError("Run Flow would call a flow that is already running in this chain")
        if len(chain) > MAX_CALL_DEPTH:
            raise FlowNodeError(f"Run Flow nests at most {MAX_CALL_DEPTH} flows deep")
        record = await asyncio.to_thread(flows_db.get_flow, flow_id)
        if record is None:
            raise FlowNodeError(
                "The flow this step runs is not in your Flows; it may have been deleted, or this flow was "
                "shared without it"
            )
        from core.flows.nodes.inputs import apply_input_values

        doc, params = apply_input_values(record["doc"], values)
        # A declared param the caller leaves out keeps its default, as in the Run dialog.
        for declared in doc.get("params") or []:
            if (
                isinstance(declared, dict)
                and declared.get("id")
                and declared.get("default") is not None
            ):
                params.setdefault(str(declared["id"]), declared["default"])
        try:
            plan = compile_flow(doc)
        except CompileError as exc:
            raise FlowNodeError(f"The flow this step runs has a problem: {exc}") from None
        child = RunState(
            id = "call-" + uuid.uuid4().hex,
            flow_id = flow_id,
            account = parent.account,
            doc = doc,
            plan = plan,
            options = {"params": params, "call_chain": chain + [flow_id]},
            persist = False,
            app = parent.app,
            cancel_event = parent.cancel_event,
            gpu_parent = parent,
        )
        if ctx.node_id in set(parent.options.get("force_nodes") or []):
            # Re-roll on the Run Flow block re-runs the called flow too, cache and baked outputs included.
            child.options["force_nodes"] = list(plan.order)
        child.hf_token = parent.hf_token
        child.caller = parent.caller
        child.key_id = parent.key_id
        # The called flow runs for whoever started the parent, but only with what its owner approved for that flow.
        child.origin = parent.origin
        child.approvals = await asyncio.to_thread(flows_db.get_capabilities, flow_id)
        ctx.on_cancel(lambda: self._fire_cancel_hooks(child))
        top = [
            n
            for n in plan.order
            if plan.owner.get(n) is None and not plan.specs[n].get("entrypoint")
        ]
        name = str(doc.get("name") or "the flow")
        scope = asyncio.ensure_future(self._run_scope(child, top, Env()))
        try:
            while not scope.done():
                await asyncio.wait({scope}, timeout = 1.0)
                ctx.progress(
                    step = min(len(child.executed), len(top)), total = len(top), phase = f"Running {name}"
                )
            await scope
        finally:
            if not scope.done():
                scope.cancel()
        ctx.check_cancelled()
        failed = sorted(child.ever_failed)
        if failed:
            state = child.nodes.get(failed[0])
            title = plan.specs[failed[0]]["title"]
            raise FlowNodeError(
                f"{name}: {title} failed: {state.error if state else 'unknown error'}"
            )
        return dict(child.outputs)

    async def _run(self, rs: RunState) -> None:
        try:
            token, key_id = await asyncio.to_thread(self.mint_key, rs.account)
            rs.caller = flow_caller.make_caller(token, rs.account, rs.app, rs.hf_token)
            rs.key_id = key_id
            rs.status = "running"
            flows_db.update_run(rs.id, status = "running", started_at = flows_db.now_ms())
            rs.emit("run.started", data = {"order": rs.plan.order})
            targets = rs.options.get("target_nodes") or []
            # Re-rolling a For Each re-rolls what it runs: its body, nested loops included.
            forced = set(rs.options.get("force_nodes") or [])
            stack = [n for n in forced if n in rs.plan.bodies]
            while stack:
                for member in rs.plan.bodies.get(stack.pop(), ()):
                    if member not in forced:
                        forced.add(member)
                        stack.append(member)
            rs.options["force_nodes"] = sorted(forced)
            # Entrypoints (Next Scene, Scene Art, NPC Reply) answer the player through /invoke, not a full run.
            need = (
                rs.plan.ancestors(targets)
                if targets
                else {n for n in rs.plan.order if not rs.plan.specs[n].get("entrypoint")}
            )
            top = [n for n in rs.plan.order if n in need and rs.plan.owner.get(n) is None]
            for nid in rs.plan.order:
                if nid not in need:
                    rs.node(nid).status = "idle"
            env = Env()
            flusher = asyncio.create_task(self._flusher(rs))
            scope = asyncio.create_task(self._run_scope(rs, top, env))
            try:
                done, _ = await asyncio.wait({scope}, timeout = MAX_RUN_SECONDS)
                if not done:
                    # Stop external jobs while their hooks are registered, then the node tasks themselves.
                    rs.cancel_event.set()
                    await self._fire_cancel_hooks(rs)
                    raise asyncio.TimeoutError
                scope.result()
            finally:
                if not scope.done():
                    scope.cancel()
                    await asyncio.gather(scope, return_exceptions = True)
                flusher.cancel()
                rs.flush()
            if rs.options.get("bake") and not rs.cancelled:
                await self._bake(rs)
            failed = sorted(rs.ever_failed)
            if rs.cancelled:
                rs.status = "cancelled"
            elif failed:
                rs.status = "failed"
                rs.error = rs.error or f"{len(failed)} step(s) failed"
            else:
                rs.status = "completed"
        except asyncio.TimeoutError:
            rs.status, rs.error = "failed", "The run passed its time limit"
        except asyncio.CancelledError:
            rs.status = "cancelled"
            raise
        except Exception as exc:
            logger.exception("flows: run %s crashed", rs.id)
            rs.status, rs.error = "failed", f"The run stopped: {exc}"
        finally:
            await asyncio.to_thread(self.revoke_key, rs.key_id)
            rs.finished_at = time.monotonic()
            flows_db.update_run(
                rs.id,
                status = rs.status,
                outputs = rs.outputs,
                error = rs.error,
                finished_at = flows_db.now_ms(),
            )
            rs.emit(f"run.{rs.status}", data = {"outputs": rs.outputs, "error": rs.error})

    async def _flusher(
        self,
        rs: RunState,
        interval: float = PROGRESS_MIN_INTERVAL_S,
    ) -> None:
        while True:
            await asyncio.sleep(interval)
            rs.flush()

    async def _fire_cancel_hooks(self, rs: RunState) -> None:
        hooks = [fn for fns in rs.cancel_hooks.values() for fn in fns]
        rs.cancel_hooks.clear()
        for fn in hooks:
            await _run_cancel_hook(fn)

        async def hard_stop() -> None:
            await asyncio.sleep(CANCEL_GRACE_S)
            for task in list(rs.node_tasks.values()):
                if not task.done():
                    task.cancel()

        asyncio.create_task(hard_stop())

    async def _run_scope(
        self,
        rs: RunState,
        node_ids: list[str],
        env: Env,
        iteration: Optional[tuple] = None,
    ) -> None:
        pending = list(node_ids)
        running: dict[asyncio.Task, str] = {}
        try:
            await self._scope_loop(rs, pending, running, env, iteration)
        except asyncio.CancelledError:
            # Node tasks are not children of this one: cancel and await them, or they outlive the run.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions = True)
            raise

    async def _scope_loop(
        self,
        rs: RunState,
        pending: list[str],
        running: dict[asyncio.Task, str],
        env: Env,
        iteration: Optional[tuple],
    ) -> None:
        plan = rs.plan
        gpu_running = 0
        cpu_running = 0
        while pending or running:
            if rs.cancelled:
                for nid in pending:
                    rs.set_status(nid, "cancelled")
                pending = []
            ready = [n for n in pending if self._resolved(rs, n, env)]
            launch: list[str] = [n for n in ready if plan.specs[n]["lane"] != "gpu"][
                : max(0, CPU_CONCURRENCY - cpu_running)
            ]
            if gpu_running == 0:
                gpu_ready = [n for n in ready if plan.specs[n]["lane"] == "gpu"]
                if gpu_ready:
                    launch.append(self._pick_gpu(rs, gpu_ready))
            for nid in launch:
                pending.remove(nid)
                task = asyncio.create_task(self._execute(rs, nid, env, iteration))
                if iteration is None:
                    rs.node_tasks[nid] = task
                running[task] = nid
                if plan.specs[nid]["lane"] == "gpu":
                    gpu_running += 1
                else:
                    cpu_running += 1
            if not running:
                if pending:
                    # Nothing can start: the remaining nodes wait on nodes outside this scope that never ran.
                    for nid in pending:
                        env.values[nid] = SKIPPED
                        rs.set_status(nid, "skipped", "Its inputs never arrived")
                    pending = []
                break
            done, _ = await asyncio.wait(running.keys(), return_when = asyncio.FIRST_COMPLETED)
            for task in done:
                nid = running.pop(task)
                rs.node_tasks.pop(nid, None)
                if plan.specs[nid]["lane"] == "gpu":
                    gpu_running -= 1
                else:
                    cpu_running -= 1
                if task.cancelled() and nid not in env.values:
                    env.values[nid] = SKIPPED
                    rs.set_status(nid, "cancelled")
                elif task.exception() is not None:
                    env.values[nid] = SKIPPED
                    logger.error("flows: node %s crashed", nid, exc_info = task.exception())
                    # Outside its handler's own error path: still a failed step, so the run cannot end completed.
                    rs.node(nid).error = (
                        rs.node(nid).error or f"Unexpected error: {task.exception()}"
                    )
                    rs.set_status(nid, "failed")

    def _resolved(self, rs: RunState, nid: str, env: Env) -> bool:
        overrides = rs.input_overrides.get(nid, {})
        for port, sources in rs.plan.inputs.get(nid, {}).items():
            if port in overrides:
                continue
            for src, _ in sources:
                if src in rs.plan.executable and not env.has(src):
                    return False
        for src in rs.plan.waits.get(nid, ()):
            if src in rs.plan.executable and not env.has(src):
                return False
        owner = rs.plan.owner.get(nid)
        if owner is not None and not env.has(owner):
            return False
        return True

    def _pick_gpu(self, rs: RunState, ready: list[str]) -> str:
        """The ready GPU node that keeps the resident model; else the model most ready nodes share; else data order."""
        plan = rs.plan

        def model_of(nid: str) -> tuple[str, str]:
            kind = plan.specs[nid].get("model_kind") or ""
            choice = (plan.nodes[nid].get("params") or {}).get("model")
            repo = choice.get("repo", "") if isinstance(choice, dict) else ""
            return MODEL_SLOT.get(kind, kind), repo.lower()

        for nid in ready:
            slot, repo = model_of(nid)
            if not repo or (rs.resident.get(slot) or "").lower() == repo:
                return nid
        counts: dict[tuple, int] = {}
        for nid in ready:
            counts[model_of(nid)] = counts.get(model_of(nid), 0) + 1
        best = max(counts.values())
        return next(n for n in ready if counts[model_of(n)] == best)

    def _gather_inputs(self, rs: RunState, nid: str, env: Env) -> tuple[dict, Optional[str]]:
        """(inputs, why it must be skipped or None)."""
        plan = rs.plan
        spec = plan.specs[nid]
        overrides = rs.input_overrides.get(nid, {})
        inputs: dict[str, Any] = {}
        for port in spec["inputs"]:
            pid = port["id"]
            if pid in overrides:
                inputs[pid] = overrides[pid]
                continue
            sources = plan.inputs.get(nid, {}).get(pid)
            if not sources:
                continue
            values = []
            upstream_failed = False
            for src, sport in sources:
                produced = env.get(src)
                if produced is SKIPPED or not isinstance(produced, dict) or sport not in produced:
                    if rs.node(src).status == "failed":
                        upstream_failed = True
                    continue
                values.append(produced[sport])
            if values:
                if port["type"] == "list":
                    values = [v if isinstance(v, list) else [v] for v in values]
                inputs[pid] = values if port.get("multi") else values[0]
            elif spec["type"] not in JOIN_TYPES:
                # A connected input that got nothing (a branch not taken, a failed step) skips the node; only joins
                # treat it as absent.
                return (
                    inputs,
                    "An upstream step failed" if upstream_failed else "Its input was skipped",
                )
        return inputs, None

    async def _execute(self, rs: RunState, nid: str, env: Env, iteration: Optional[tuple]) -> None:
        plan = rs.plan
        spec = plan.specs[nid]
        node = plan.nodes[nid]
        state = rs.node(nid)
        state.error = None
        inputs, skip = self._gather_inputs(rs, nid, env)
        if skip:
            env.values[nid] = SKIPPED
            rs.set_status(nid, "skipped", skip)
            self._record(rs, nid, "skipped", None, None, iteration)
            return
        fired = (rs.options.get("params") or {}).get("__trigger")
        if (
            rs.origin.get("kind") == "trigger"
            and isinstance(fired, dict)
            and spec["type"] in TRIGGER_TYPES
            and nid not in fired
        ):
            # Another trigger started this run: this one's branch stays still.
            env.values[nid] = SKIPPED
            rs.set_status(nid, "skipped", "Another trigger started this run")
            self._record(rs, nid, "skipped", None, None, iteration)
            return
        params = param_values(spec, node)
        _bind_params(spec, params, inputs)
        seed = self._seed(rs, nid, params, inputs, iteration)
        forced = nid in set(rs.options.get("force_nodes") or [])
        pinned = (rs.options.get("pinned") or {}).get(nid)
        if isinstance(pinned, dict) and iteration is None and not forced:
            from core.flows.nodes._common import gallery_media_present

            # The editor pinned an earlier output of this node: downstream gets it as if it were cached. One holding
            # media the user has since deleted runs the node again instead, as a stale cache hit does.
            try:
                present = await asyncio.to_thread(gallery_media_present, pinned)
            except Exception:
                logger.debug("flows: pinned media check failed", exc_info = True)
                present = False
            if present:
                self._finish(rs, nid, env, dict(pinned), "cached", iteration, message = "Pinned")
                return
        baked = (rs.doc.get("baked") or {}).get(nid)
        if (
            baked
            and iteration is None
            and not forced
            and not rs.options.get("bake")
            and not rs.options.get("ignore_baked")
        ):
            self._finish(rs, nid, env, dict(baked), "cached", iteration, message = "Baked")
            return

        if spec["type"] == FOR_EACH:
            await self._for_each(rs, nid, env, inputs, params, iteration)
            return

        ctx = NodeContextImpl(self, rs, nid, spec, params, inputs, seed, iteration)
        key = None
        choice = params.get("model")
        # "Use the loaded model" can change under the node (another run or page swaps it), so it is never cached.
        cacheable = spec["cacheable"] and not (
            spec.get("model_kind") and not (isinstance(choice, dict) and choice.get("repo"))
        )
        if cacheable and not forced:
            try:
                model_key = await self._model_key(ctx, spec, params)
                key = cache_key(
                    spec["type"],
                    spec["v"],
                    params,
                    inputs,
                    model = model_key,
                    seed = seed,
                    flow_params = ctx.flow_params,
                )
                hit = flows_db.cache_get(key)
                # A cached image, clip or video the user has since deleted is regenerated, not served as a 404.
                if hit is not None:
                    from core.flows.nodes._common import gallery_media_present
                    if not await asyncio.to_thread(gallery_media_present, hit):
                        hit = None
            except Exception:
                logger.debug("flows: cache lookup failed", exc_info = True)
                hit = None
            if hit is not None:
                self._finish(rs, nid, env, hit, "cached", iteration)
                return
        elif cacheable:
            try:
                key = cache_key(
                    spec["type"], spec["v"], params, inputs,
                    model = await self._model_key(ctx, spec, params), seed = seed,
                    flow_params = ctx.flow_params,
                )  # fmt: skip
            except Exception:
                key = None

        handler = self._handlers().get(spec["type"])
        if handler is None:
            state.error = f"{spec['title']} cannot run in this version of Unsloth Studio"
            env.values[nid] = SKIPPED
            rs.set_status(nid, "failed")
            self._record(rs, nid, "failed", None, state.error, iteration)
            return
        try:
            for cap, detail in capabilities.node_needs(spec, params):
                refused = capabilities.refusal(rs.origin, rs.approvals, cap, detail, spec["title"])
                if refused:
                    raise FlowNodeError(refused)
            if spec["lane"] == "gpu":
                lock = self.gpu_lock()
                if lock.locked():
                    rs.set_status(nid, "waiting", "Waiting for the GPU")
                async with lock:
                    ctx.check_cancelled()
                    ctx.count_gpu_op()
                    if spec.get("exclusive_gpu"):
                        outputs = await self._exclusive(rs, nid, spec, handler, ctx)
                    else:
                        rs.set_status(nid, "running")
                        outputs = await handler(ctx)
            else:
                rs.set_status(nid, "running")
                outputs = await handler(ctx)
        except (FlowCancelled, asyncio.CancelledError):
            env.values[nid] = SKIPPED
            rs.flush(nid)
            rs.set_status(nid, "cancelled")
            self._record(rs, nid, "cancelled", None, None, iteration)
            return
        except Exception as exc:
            env.values[nid] = SKIPPED
            rs.flush(nid)
            if rs.cancelled:
                # Stop aborted the request in flight, so its error is the stop itself, not a failure.
                rs.set_status(nid, "cancelled")
                self._record(rs, nid, "cancelled", None, None, iteration)
                return
            if isinstance(exc, FlowNodeError):
                state.error = str(exc)
            else:
                logger.exception("flows: %s (%s) failed", nid, spec["type"])
                state.error = f"{spec['title']} failed: {exc}"
            rs.set_status(nid, "failed")
            self._record(rs, nid, "failed", None, state.error, iteration)
            return
        finally:
            rs.cancel_hooks.pop((nid, iteration), None)
        outputs = dict(outputs or {})
        if key is not None and spec["cacheable"]:
            try:
                flows_db.cache_put(key, spec["type"], outputs)
            except Exception:
                logger.debug("flows: cache write failed", exc_info = True)
        rs.flush(nid)
        self._finish(rs, nid, env, outputs, "done", iteration)

    async def _exclusive(self, rs: RunState, nid: str, spec: dict, handler: Any, ctx: Any) -> dict:
        """Run under the Studio-wide GPU lease, so chat, training and other runs cannot swap the model meanwhile."""
        from core.inference import gpu_arbiter
        while True:
            lease = gpu_arbiter.operation_lease(
                "flow", f"flow:{rs.id}:{nid}", MAX_RUN_SECONDS, label = spec["title"]
            )
            try:
                lease.__enter__()
            except gpu_arbiter.GpuLeaseHeldError as exc:
                rs.set_status(nid, "waiting", str(exc))
                await asyncio.sleep(LEASE_RETRY_S)
                ctx.check_cancelled()
                continue
            try:
                rs.set_status(nid, "running")
                return await handler(ctx)
            finally:
                lease.__exit__(None, None, None)

    def _finish(
        self,
        rs: RunState,
        nid: str,
        env: Env,
        outputs: dict,
        status: str,
        iteration: Optional[tuple],
        message: Optional[str] = None,
    ) -> None:
        spec = rs.plan.specs[nid]
        declared = {p["id"] for p in spec["outputs"]}
        # If/Switch leave out the branches they do not take; consumers of those see them skipped.
        env.values[nid] = {k: v for k, v in outputs.items() if k in declared}
        state = rs.node(nid)
        state.outputs = outputs
        rs._pending_partial = {k: v for k, v in rs._pending_partial.items() if k[0] != nid}
        state.partial = None
        if spec["type"] == "output.result":
            name = str((rs.plan.nodes[nid].get("params") or {}).get("name") or nid)
            if "value" in outputs:
                rs.outputs[name] = outputs["value"]
        data: dict[str, Any] = {"outputs": outputs, "cached": status == "cached"}
        if iteration is not None:
            data["iteration"] = list(iteration)
        rs.emit("node.output", nid, data)
        rs.executed.add(nid)
        rs.set_status(nid, status, message)
        self._record(rs, nid, status, outputs, None, iteration)

    def _record(
        self,
        rs: RunState,
        nid: str,
        status: str,
        outputs: Optional[dict],
        error: Optional[str],
        iteration: Optional[tuple],
    ) -> None:
        if not rs.persist:
            return
        try:
            flows_db.save_node_result(rs.id, nid, status, outputs, error)
        except Exception:
            logger.debug("flows: could not store a node result", exc_info = True)

    def _seed(
        self, rs: RunState, nid: str, params: dict, inputs: dict, iteration: Optional[tuple]
    ) -> int:
        value = params.get("seed")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value) % (2**32)
        incoming = inputs.get("seed")
        if isinstance(incoming, (int, float)) and not isinstance(incoming, bool):
            return int(incoming) % (2**32)
        return _derive_seed(int(rs.doc.get("seed") or 0), nid, iteration)

    async def _model_key(self, ctx: NodeContextImpl, spec: dict, params: dict) -> Optional[dict]:
        kind = spec.get("model_kind")
        if not kind:
            return None
        choice = params.get("model") or {}
        key = {"repo": choice.get("repo"), "variant": choice.get("variant")}
        repo = choice.get("repo")
        if isinstance(repo, str) and os.path.isabs(repo):
            # A local export can be rewritten in place: its files name the weights, not the path.
            key["files"] = await asyncio.to_thread(_local_fingerprint, repo)
        return key

    async def _for_each(
        self,
        rs: RunState,
        nid: str,
        env: Env,
        inputs: dict,
        params: dict,
        iteration: Optional[tuple],
    ) -> None:
        plan = rs.plan
        items = inputs.get("items")
        if not isinstance(items, list):
            items = [] if items is None else [items]
        limit = max(1, min(int(params.get("max_items") or 16), MAX_FOREACH_ITEMS))
        if len(items) > limit:
            rs.emit(
                "node.log",
                nid,
                {
                    "level": "warning",
                    "message": f"Only the first {limit} of {len(items)} items run",
                },
            )
            items = items[:limit]
        body = plan.body_order(nid)
        collect = plan.collect.get(nid)
        rs.set_status(nid, "running")
        results: list[Any] = []
        parallel = bool(params.get("parallel")) and not any(
            plan.specs[b]["lane"] == "gpu" for b in body
        )

        async def one(index: int, item: Any) -> Any:
            child = Env(env)
            child.values[nid] = {"item": item, "index": index}
            key = (*(iteration or ()), index)
            await self._run_scope(rs, body, child, key)
            if collect is None:
                return None
            produced = child.values.get(collect[0])
            if isinstance(produced, dict) and collect[1] in produced:
                return produced[collect[1]]
            return SKIPPED

        if parallel:
            # Each iteration has its own scope, so the CPU limit is applied across iterations here.
            gate = asyncio.Semaphore(CPU_CONCURRENCY)

            async def bounded(index: int, item: Any) -> Any:
                async with gate:
                    return await one(index, item)

            gathered = await asyncio.gather(*(bounded(i, item) for i, item in enumerate(items)))
        else:
            gathered = []
            for i, item in enumerate(items):
                if rs.cancelled:
                    break
                rs.progress(
                    nid, {"fraction": i / max(1, len(items)), "step": i, "total": len(items)}
                )
                gathered.append(await one(i, item))
        # Only skipped iterations drop out: a body may legitimately produce null.
        results = [] if collect is None else [r for r in gathered if r is not SKIPPED]
        if rs.cancelled:
            env.values[nid] = SKIPPED
            rs.set_status(nid, "cancelled")
            return
        failed_body = [b for b in body if b in rs.ever_failed]
        last = items[-1] if items else None
        outputs = {"results": results, "item": last, "index": max(0, len(items) - 1)}
        self._finish(
            rs,
            nid,
            env,
            outputs,
            "done",
            iteration,
            message = f"{len(failed_body)} step(s) failed" if failed_body else None,
        )
        # Body nodes keep their own status; the run reports their failures.

    async def _bake(self, rs: RunState) -> None:
        """Copy every finished output into the flow as embedded values, media included, so it plays with no models."""
        baked: dict[str, dict] = {}
        refs: dict[str, dict] = {}
        for nid in rs.plan.order:
            spec = rs.plan.specs[nid]
            state = rs.node(nid)
            if (
                nid not in rs.executed
                or state.status not in ("done", "cached")
                or spec["side_effect"]
                or spec["type"] in ("output.preview", "output.result", "input.seed")
                or rs.plan.owner.get(nid)
            ):
                continue
            try:
                baked[nid] = await self._embed(rs, state.outputs or {}, refs)
            except Exception as exc:
                rs.emit("node.log", nid, {"level": "warning", "message": f"Could not bake: {exc}"})
        for _ in range(3):
            stored = flows_db.get_flow(rs.flow_id)
            if stored is None:
                return
            doc = stored["doc"]
            doc["baked"] = {**(doc.get("baked") or {}), **baked}
            known = {a["id"] for a in doc.get("assets") or []}
            doc["assets"] = [
                *(doc.get("assets") or []),
                *(a for a in refs.values() if a["id"] not in known),
            ]
            try:
                flows_db.save_flow(doc, stored["updatedAt"])
                rs.emit(
                    "node.log",
                    None,
                    {"level": "info", "message": f"Baked {len(baked)} step(s) into the flow"},
                )
                return
            except flows_db.FlowConflict:
                continue
            except flows_db.FlowDeleted:
                return

    async def _embed(self, rs: RunState, value: Any, refs: dict[str, dict]) -> Any:
        if isinstance(value, list):
            return [await self._embed(rs, v, refs) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get("kind") in ("image", "audio", "video") and isinstance(value.get("ref"), str):
            ref = value["ref"]
            digest = asset_store.parse_ref(ref)
            if digest is None:
                from core.flows.nodes import _common

                ctx = NodeContextImpl(self, rs, "bake", {"type": "bake"}, {}, {}, 0)
                data, mime = await _common.media_bytes(ctx, value)
                digest, mime = await asyncio.to_thread(asset_store.store, data, mime)
            else:
                got = asset_store.read(digest)
                if got is None:
                    return value
                data, mime = got
            refs[digest] = {
                "id": f"sha256:{digest}",
                "mime": mime,
                "bytes": len(data),
                "name": None,
            }
            embedded = {
                **value,
                "ref": f"sha256:{digest}",
                "url": asset_store.url_for(digest),
                "mime": mime,
            }
            return embedded
        if value.get("kind") in ("table", "heatmap", "tensor") and isinstance(
            value.get("ref"), str
        ):
            # A large structured value's full data is a JSON flow asset: the baked package carries it too.
            digest = asset_store.parse_ref(value["ref"])
            got = asset_store.read(digest) if digest else None
            if got is not None:
                refs[digest] = {
                    "id": f"sha256:{digest}",
                    "mime": got[1],
                    "bytes": len(got[0]),
                    "name": None,
                }
            return value
        return {k: await self._embed(rs, v, refs) for k, v in value.items()}

    def _trim_finished(self) -> None:
        finished = [
            rs for rs in self.runs.values() if rs.finished_at is not None and not rs.subscribers
        ]
        if len(finished) <= FINISHED_RUNS_KEPT:
            return
        finished.sort(key = lambda r: r.finished_at or 0)
        for rs in finished[: len(finished) - FINISHED_RUNS_KEPT]:
            self.runs.pop(rs.id, None)


def _bind_params(spec: dict, params: dict, inputs: dict) -> None:
    """A value connected to a bindable param's input replaces the field, clamped like a typed one."""
    by_id = {p["id"]: p for p in spec["params"]}
    for port in spec["inputs"]:
        pid = port.get("param")
        if pid and port["id"] in inputs:
            value = inputs.pop(port["id"])
            params[pid] = _clamp(by_id[pid], value) if pid in by_id else value


def _origin(account: AccountContext, kind: Optional[str]) -> dict:
    """An API key (MCP included) is never the owner's own hand, whatever account it acts for."""
    if account.via_api_key:
        return {"kind": "api_key", "owner": False}
    return {"kind": kind or "ui", "owner": account.is_owner}


def _local_fingerprint(path: str) -> Optional[list]:
    """(name, size, mtime) of a local model file; a folder folds every file directly in it into one digest."""
    try:
        if os.path.isdir(path):
            entries = sorted(os.scandir(path), key = lambda e: e.name)
            digest = hashlib.sha256()
            for e in entries:
                if e.is_file():
                    st = e.stat()
                    digest.update(f"{e.name}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
            return [[os.path.basename(path.rstrip(os.sep)), digest.hexdigest()]]
        st = os.stat(path)
        return [[os.path.basename(path), st.st_size, st.st_mtime_ns]]
    except OSError:
        return None


def _stored_options(options: dict) -> dict:
    stored = {k: v for k, v in options.items() if k not in ("doc", "pinned")}
    return json.loads(json.dumps(stored, default = str))


ENGINE = FlowEngine()


def get_engine() -> FlowEngine:
    return ENGINE
