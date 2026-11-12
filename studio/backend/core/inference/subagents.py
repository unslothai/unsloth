# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Subagents: a ``spawn_subagent`` call runs a child Studio tool loop and returns its final answer.

The child is the same ``stream_with_studio_tools`` loop the parent runs, on a fresh two-message
conversation (a fixed system prompt and the parent's brief), through a transport the route builds
for the same provider connection. It inherits the parent's tool policy minus ``spawn_subagent``
(depth 1), shares the parent's session id (same sandbox folder; approvals resolve through the
normal ``/tool-confirm``), and stops when the parent's cancel event fires.

Live state is kept here in memory for the side panel (see studio/SUBAGENTS.md); finished runs are
saved as an archived chat thread plus a ``storage.subagents_db`` record. This module must not
import ``studio_tool_loop`` at import time: the loop imports this module.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loggers import get_logger
from utils.account_context import current_account_id

logger = get_logger(__name__)

SUBAGENT_TOOL_NAME = "spawn_subagent"

SUBAGENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_NAME,
        "description": (
            "Delegate a well-scoped task to a subagent: a separate agent with its own fresh "
            "context and the same tools as you (minus this one). It works on the brief alone, "
            "cannot see this conversation or ask questions, and returns one final report as "
            "this tool's result. Use it for independent, self-contained pieces of work such as "
            "investigating one area of a codebase, researching one question, or writing one "
            "file, especially to keep long tool output out of your own context. Call it several "
            "times in the same turn to run independent tasks in parallel. Do not use it for "
            "trivial steps you can do with one tool call yourself. Review what a subagent reports "
            "before relying on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A 3-6 word label for the task, shown to the user.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The complete, self-contained brief: the goal, all context the subagent "
                        "needs (paths, names, constraints, what was already tried), and exactly "
                        "what its final report should contain."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of your tool names the subagent may use. Omit to give "
                        "it all of your tools."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional model id on the same provider for this subagent, e.g. a "
                        "faster or cheaper model for simple work. Omit to use your own model."
                    ),
                },
            },
            "required": ["description", "prompt"],
        },
    },
}

SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent in Unsloth Studio. Another AI agent (the main agent) delegated one task "
    "to you. You cannot see its conversation and cannot ask it or the user anything; work from "
    "the brief alone and use your tools as needed. When you are done, reply with your final "
    "report. It is returned to the main agent verbatim as the result of its tool call, so make "
    "it self-contained and concise: lead with the answer, include the concrete details it needs "
    "(file paths, commands, numbers, sources), and say plainly what you could not do or verify. "
    "Do not ask follow-up questions. Today's date is {today}."
)

_MAX_RESULT_CHARS = 24_000
_PREVIEW_CHARS = 600
_TRANSCRIPT_RESULT_CHARS = 4_000
_TRANSCRIPT_MAX_ITEMS = 2_000
# Finished runs stay in memory this long for the panel; the stored record serves them afterwards.
_KEEP_FINISHED_S = 15 * 60
_CANCEL_POLL_S = 0.1


def _max_parallel() -> int:
    raw = os.environ.get("UNSLOTH_SUBAGENT_MAX_PARALLEL", "")
    try:
        value = int(raw) if raw.strip() else 4
    except ValueError:
        value = 4
    return max(1, min(value, 16))


def _now_ms() -> int:
    return int(time.time() * 1000)


def subagents_requested(enabled_tools: Optional[list[str]]) -> bool:
    return bool(enabled_tools) and SUBAGENT_TOOL_NAME in enabled_tools


# ---------------------------------------------------------------------------------------------
# Live registry
# ---------------------------------------------------------------------------------------------


@dataclass(eq = False)
class _Run:
    record: dict[str, Any]
    account_id: Optional[str]
    cancel_event: threading.Event
    done_event: threading.Event = field(default_factory = threading.Event)
    result_text: str = ""
    finished_monotonic: Optional[float] = None
    # Tool card id -> index into record["transcript"].
    tool_items: dict[str, int] = field(default_factory = dict)


_lock = threading.Lock()
_runs: dict[str, _Run] = {}


def _evict_finished_locked() -> None:
    now = time.monotonic()
    for run_id in [
        run_id
        for run_id, run in _runs.items()
        if run.finished_monotonic is not None and now - run.finished_monotonic > _KEEP_FINISHED_S
    ]:
        del _runs[run_id]


def _visible(run: _Run) -> bool:
    return run.account_id == current_account_id()


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "transcript"}


def _snapshot(record: dict[str, Any]) -> dict[str, Any]:
    copy = dict(record)
    copy["tools"] = list(record.get("tools") or [])
    copy["transcript"] = [dict(item) for item in record.get("transcript") or []]
    if isinstance(record.get("pendingApproval"), dict):
        copy["pendingApproval"] = dict(record["pendingApproval"])
    return copy


def live_subagent(subagent_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        _evict_finished_locked()
        run = _runs.get(subagent_id)
        if run is None or not _visible(run):
            return None
        return _snapshot(run.record)


def live_subagents_for_parent(parent_thread_id: str) -> list[dict[str, Any]]:
    with _lock:
        _evict_finished_locked()
        rows = [
            _summary(_snapshot(run.record))
            for run in _runs.values()
            if run.record.get("parentThreadId") == parent_thread_id and _visible(run)
        ]
    rows.sort(key = lambda row: row.get("startedAt") or 0)
    return rows


def cancel_subagent(subagent_id: str) -> bool:
    with _lock:
        run = _runs.get(subagent_id)
        if run is None or not _visible(run) or run.record.get("status") != "running":
            return False
        run.cancel_event.set()
        return True


# ---------------------------------------------------------------------------------------------
# Transcript building from the loop's SSE lines
# ---------------------------------------------------------------------------------------------


def _truncate(text: Any, limit: int) -> Any:
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters]"


class _TranscriptWriter:
    """Folds the loop's SSE lines into the record's transcript; every mutation holds ``_lock``."""

    def __init__(self, run: _Run) -> None:
        self.run = run
        self.error: Optional[str] = None

    @property
    def _items(self) -> list[dict[str, Any]]:
        return self.run.record["transcript"]

    def _append_text(self, kind: str, text: str) -> None:
        items = self._items
        if items and items[-1].get("kind") == kind:
            items[-1]["text"] += text
        elif len(items) < _TRANSCRIPT_MAX_ITEMS:
            items.append({"kind": kind, "text": text})

    def _drop_status(self) -> None:
        items = self._items
        if items and items[-1].get("kind") == "status":
            items.pop()

    def feed(self, line: str) -> None:
        if not isinstance(line, str) or not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        with _lock:
            self._feed_locked(payload)

    def _feed_locked(self, payload: dict[str, Any]) -> None:
        record = self.run.record
        event_type = payload.get("type")
        if event_type == "tool_status":
            text = payload.get("content")
            self._drop_status()
            if isinstance(text, str) and text.strip():
                self._items.append({"kind": "status", "text": text})
            return
        if event_type == "tool_start":
            self._drop_status()
            card_id = str(payload.get("tool_call_id") or uuid.uuid4().hex)
            awaiting = bool(payload.get("awaiting_confirmation"))
            item = {
                "kind": "tool",
                "toolCallId": card_id,
                "toolName": payload.get("tool_name") or "",
                "arguments": payload.get("arguments")
                if isinstance(payload.get("arguments"), dict)
                else {},
                "status": "awaiting_approval" if awaiting else "running",
                "approvalId": payload.get("approval_id") or None,
                "result": None,
            }
            if card_id in self.run.tool_items:
                self._items[self.run.tool_items[card_id]].update(item)
            elif len(self._items) < _TRANSCRIPT_MAX_ITEMS:
                self.run.tool_items[card_id] = len(self._items)
                self._items.append(item)
            if awaiting and item["approvalId"]:
                record["pendingApproval"] = {
                    "approvalId": item["approvalId"],
                    "toolName": item["toolName"],
                    "toolCallId": card_id,
                    "arguments": item["arguments"],
                }
            return
        if event_type == "tool_approved":
            card_id = str(payload.get("tool_call_id") or "")
            index = self.run.tool_items.get(card_id)
            if index is not None:
                self._items[index]["status"] = "running"
            self._clear_approval(card_id)
            return
        if event_type == "tool_end":
            self._drop_status()
            card_id = str(payload.get("tool_call_id") or "")
            index = self.run.tool_items.get(card_id)
            result = payload.get("result")
            if not isinstance(result, str):
                result = json.dumps(result, default = str) if result is not None else ""
            if index is not None:
                self._items[index]["status"] = "done"
                self._items[index]["result"] = _truncate(result, _TRANSCRIPT_RESULT_CHARS)
            record["toolCallCount"] = int(record.get("toolCallCount") or 0) + 1
            self._clear_approval(card_id)
            return
        error = payload.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            self.error = str(message or "The provider returned an error.")
            self._items.append({"kind": "error", "text": self.error})
            return
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self._drop_status()
            self._append_text("reasoning", reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._drop_status()
            self._append_text("text", content)

    def _clear_approval(self, card_id: str) -> None:
        pending = self.run.record.get("pendingApproval")
        if isinstance(pending, dict) and pending.get("toolCallId") == card_id:
            self.run.record["pendingApproval"] = None

    def final_answer(self) -> str:
        """The prose after the last tool call: the report. Falls back to all prose."""
        with _lock:
            items = list(self._items)
        tail: list[str] = []
        for item in reversed(items):
            if item.get("kind") == "tool":
                break
            if item.get("kind") == "text":
                tail.append(item["text"])
        answer = "".join(reversed(tail)).strip()
        if answer:
            return answer
        return "".join(item["text"] for item in items if item.get("kind") == "text").strip()


# ---------------------------------------------------------------------------------------------
# Spawner: one per parent run, built by the route
# ---------------------------------------------------------------------------------------------


@dataclass(eq = False)
class SubagentSpawner:
    """How a parent run makes children: a transport factory for its provider connection."""

    # model override (None = parent's model) -> a fresh ToolLoopTransport.
    make_transport: Callable[[Optional[str]], Any]
    parent_model: str
    parent_thread_id: Optional[str]
    session_id: Optional[str]
    provider: Optional[dict[str, Any]] = None
    _semaphore: Optional[asyncio.Semaphore] = None

    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(_max_parallel())
        return self._semaphore

    def start(
        self,
        arguments: dict[str, Any],
        *,
        parent_run: Any,
        parent_policy: Any,
        parent_cancel: threading.Event,
        parent_permissions: Any = None,
    ) -> "SubagentHandle":
        """Start a child on the running loop now; the returned handle yields its result."""
        description = str(arguments.get("description") or "").strip()[:120] or "Subagent task"
        prompt = str(arguments.get("prompt") or "").strip()
        model_override = arguments.get("model")
        model = (
            model_override.strip()
            if isinstance(model_override, str) and model_override.strip()
            else None
        )
        if model is not None and (len(model) > 200 or model == self.parent_model):
            model = None if model == self.parent_model else model[:200]

        available = [
            tool
            for tool in parent_policy.tools
            if tool.get("function", {}).get("name") != SUBAGENT_TOOL_NAME
        ]
        requested = arguments.get("tools")
        child_tools = available
        if isinstance(requested, list) and requested:
            wanted = {str(name).strip() for name in requested if isinstance(name, str)}
            picked = [tool for tool in available if tool.get("function", {}).get("name") in wanted]
            if picked:
                child_tools = picked
        tool_names = [tool.get("function", {}).get("name") for tool in child_tools]

        subagent_id = f"sa_{uuid.uuid4().hex[:16]}"
        record: dict[str, Any] = {
            "id": subagent_id,
            "parentThreadId": self.parent_thread_id,
            "childThreadId": str(uuid.uuid4()),
            "sessionId": self.session_id,
            "description": description,
            "prompt": prompt,
            "model": model or self.parent_model,
            "tools": tool_names,
            "status": "running",
            "error": None,
            "resultPreview": None,
            "toolCallCount": 0,
            "pendingApproval": None,
            "startedAt": _now_ms(),
            "finishedAt": None,
            "transcript": [],
        }
        run = _Run(record = record, account_id = current_account_id(), cancel_event = threading.Event())
        with _lock:
            _evict_finished_locked()
            _runs[subagent_id] = run
        task = asyncio.ensure_future(
            self._run(
                run,
                prompt = prompt,
                model = model,
                child_tools = child_tools,
                parent_run = parent_run,
                parent_policy = parent_policy,
                parent_cancel = parent_cancel,
                parent_permissions = parent_permissions,
            )
        )
        return SubagentHandle(run = run, task = task, arguments = arguments)

    async def _run(
        self,
        run: _Run,
        *,
        prompt: str,
        model: Optional[str],
        child_tools: list[dict[str, Any]],
        parent_run: Any,
        parent_policy: Any,
        parent_cancel: threading.Event,
        parent_permissions: Any = None,
    ) -> None:
        from core.inference.studio_tool_loop import ToolLoopRun, stream_with_studio_tools

        record = run.record
        writer = _TranscriptWriter(run)
        watcher = asyncio.ensure_future(_link_cancel(parent_cancel, run.cancel_event))
        status = "failed"
        error: Optional[str] = None
        await _persist(record)
        try:
            if not prompt:
                raise ValueError("spawn_subagent needs a non-empty prompt.")
            semaphore = self.semaphore()
            if semaphore.locked():
                with _lock:
                    record["transcript"].append(
                        {"kind": "status", "text": "Waiting for a free subagent slot"}
                    )
            async with semaphore:
                if run.cancel_event.is_set():
                    raise asyncio.CancelledError
                transport = self.make_transport(model)
                child_policy = dataclasses.replace(
                    parent_policy,
                    tools = child_tools,
                    on_withheld_tool_call = None,
                    on_provider_turn_end = None,
                    subagent_spawner = None,
                    # The parent's live grant (Full access) holds here, and a grant made here reaches the parent.
                    shared_live_permissions = parent_permissions,
                )
                child_run = ToolLoopRun(
                    messages = [
                        {
                            "role": "system",
                            "content": SUBAGENT_SYSTEM_PROMPT.format(
                                today = time.strftime("%Y-%m-%d")
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    session_id = self.session_id,
                    thread_id = record["childThreadId"],
                    model = model or self.parent_model,
                    supports_vision = bool(getattr(parent_run, "supports_vision", False))
                    and model is None,
                )
                with _lock:
                    writer._drop_status()
                stream = stream_with_studio_tools(
                    transport,
                    run = child_run,
                    policy = child_policy,
                    cancel_event = run.cancel_event,
                )
                try:
                    async for line in stream:
                        writer.feed(line)
                finally:
                    await stream.aclose()
            if run.cancel_event.is_set():
                status = "cancelled"
            elif writer.error and not writer.final_answer():
                status, error = "failed", writer.error
            else:
                status = "completed"
        except asyncio.CancelledError:
            status = "cancelled"
            run.cancel_event.set()
        except Exception as exc:  # noqa: BLE001 - reported to the parent as the tool result
            logger.exception("subagent.run_failed id=%s", record["id"])
            status, error = "failed", str(exc) or exc.__class__.__name__
        finally:
            watcher.cancel()
        answer = writer.final_answer()
        run.result_text = _result_for_parent(record["description"], status, answer, error)
        with _lock:
            if error and not any(item.get("kind") == "error" for item in record["transcript"]):
                record["transcript"].append({"kind": "error", "text": error})
            record["status"] = status
            record["error"] = error
            record["resultPreview"] = (answer[:_PREVIEW_CHARS] or None) if answer else None
            record["pendingApproval"] = None
            record["finishedAt"] = _now_ms()
            for item in record["transcript"]:
                if item.get("kind") == "tool" and item.get("status") != "done":
                    item["status"] = "done"
                    item["result"] = item.get("result") or "Not finished."
            record["transcript"] = [
                item for item in record["transcript"] if item.get("kind") != "status"
            ]
            run.finished_monotonic = time.monotonic()
            snapshot = _snapshot(record)
        run.done_event.set()
        # Saving never decides the result the parent gets.
        await _persist(snapshot)
        await asyncio.to_thread(_save_child_chat, snapshot, self.provider)


@dataclass(eq = False)
class SubagentHandle:
    run: _Run
    task: "asyncio.Future[None]"
    arguments: dict[str, Any]

    @property
    def subagent_id(self) -> str:
        return self.run.record["id"]

    def serves(self, arguments: Any) -> bool:
        return arguments == self.arguments

    def cancel(self) -> None:
        self.run.cancel_event.set()

    def tool_stream(self) -> "_SubagentToolStream":
        return _SubagentToolStream(self)


class _SubagentToolStream:
    """The loop's tool-stream protocol over a running child: heartbeats until it ends, then its result."""

    def __init__(self, handle: SubagentHandle) -> None:
        self._handle = handle
        self._finished = False

    def __iter__(self) -> "_SubagentToolStream":
        return self

    def __next__(self) -> Any:
        from core.inference.tool_stream_exec import TOOL_HEARTBEAT_INTERVAL_S

        if self._finished:
            raise StopIteration
        if self._handle.run.done_event.wait(TOOL_HEARTBEAT_INTERVAL_S):
            self._finished = True
            raise StopIteration(self._handle.run.result_text)
        return {"type": "heartbeat"}

    def close(self) -> None:
        # Closed after use as well; only an unfinished child is stopped.
        if not self._handle.run.done_event.is_set():
            self._handle.cancel()


async def _link_cancel(parent: threading.Event, child: threading.Event) -> None:
    while not child.is_set():
        if parent.is_set():
            child.set()
            return
        await asyncio.sleep(_CANCEL_POLL_S)


def _result_for_parent(description: str, status: str, answer: str, error: Optional[str]) -> str:
    if len(answer) > _MAX_RESULT_CHARS:
        answer = answer[:_MAX_RESULT_CHARS] + "\n… [report truncated]"
    if status == "completed":
        return answer or f"Subagent '{description}' finished without a report."
    if status == "cancelled":
        suffix = f"\nPartial report:\n{answer}" if answer else ""
        return f"Subagent '{description}' was stopped before it finished.{suffix}"
    suffix = f"\nPartial report:\n{answer}" if answer else ""
    return f"Error: subagent '{description}' failed: {error or 'unknown error'}{suffix}"


# ---------------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------------


async def _persist(record: dict[str, Any]) -> None:
    stored = dict(record)
    stored["pendingApproval"] = None
    try:
        from storage import subagents_db
    except ImportError:
        return
    try:
        await asyncio.to_thread(subagents_db.upsert_subagent, stored)
    except Exception:  # noqa: BLE001 - the panel falls back to live data
        logger.exception("subagent.persist_failed id=%s", record.get("id"))


def _save_child_chat(record: dict[str, Any], provider: Optional[dict[str, Any]]) -> None:
    """Saves the child as an archived chat thread the panel's "Open chat" opens."""
    try:
        from storage import studio_db

        thread_id = record["childThreadId"]
        started_at = int(record["startedAt"])
        finished_at = max(int(record.get("finishedAt") or started_at), started_at + 1)
        project_id = None
        if record.get("parentThreadId"):
            parent = studio_db.get_chat_thread(record["parentThreadId"])
            project_id = (parent or {}).get("projectId")
        studio_db.upsert_chat_thread(
            {
                "id": thread_id,
                "title": f"Subagent · {record['description']}",
                "modelType": "base",
                "modelId": record.get("model") or "",
                "createdAt": started_at,
                "updatedAt": finished_at,
                "projectId": project_id,
                "archived": True,
            }
        )
        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        text_cursor = 0
        for item in record.get("transcript") or []:
            kind = item.get("kind")
            if kind == "reasoning" and item.get("text"):
                parts.append({"type": "reasoning", "text": item["text"]})
            elif kind == "text" and item.get("text"):
                parts.append({"type": "text", "text": item["text"]})
                text_cursor += len(item["text"])
            elif kind == "tool":
                arguments = item.get("arguments") or {}
                parts.append(
                    {
                        "type": "tool-call",
                        "toolCallId": item.get("toolCallId") or uuid.uuid4().hex,
                        "toolName": item.get("toolName") or "",
                        "args": arguments,
                        "argsText": json.dumps(arguments, default = str),
                        "result": item.get("result") or "",
                        "textCursor": text_cursor,
                    }
                )
                tool_calls.append({"name": item.get("toolName") or ""})
            elif kind == "error" and item.get("text"):
                parts.append({"type": "text", "text": f"\n\n**Error:** {item['text']}"})
        if not parts:
            parts.append({"type": "text", "text": "(no output)"})
        details: dict[str, Any] = {
            "modelId": record.get("model") or "",
            "modelLabel": record.get("model") or "",
            "responseModelId": record.get("model") or "",
            "startedAt": started_at,
            "finishedAt": finished_at,
            "durationMs": finished_at - started_at,
            "toolCalls": tool_calls,
        }
        if provider:
            details.update(
                providerId = provider.get("id"),
                providerName = provider.get("display_name"),
                providerType = provider.get("provider_type"),
            )
        source = {"subagentId": record["id"], "parentThreadId": record.get("parentThreadId")}
        user_message_id = str(uuid.uuid4())
        studio_db.sync_chat_messages(
            thread_id,
            [
                {
                    "id": user_message_id,
                    "threadId": thread_id,
                    "parentId": None,
                    "role": "user",
                    "content": [{"type": "text", "text": record.get("prompt") or ""}],
                    "metadata": {"subagent": source},
                    "createdAt": started_at,
                },
                {
                    "id": str(uuid.uuid4()),
                    "threadId": thread_id,
                    "parentId": user_message_id,
                    "role": "assistant",
                    "content": parts,
                    "metadata": {"subagent": source, "responseDetails": details},
                    "createdAt": finished_at,
                },
            ],
        )
    except Exception:  # noqa: BLE001 - the record and the parent's result stand without it
        logger.exception("subagent.save_chat_failed id=%s", record.get("id"))
