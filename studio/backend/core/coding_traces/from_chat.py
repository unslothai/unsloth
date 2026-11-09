# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from typing import Any, Optional

import storage.studio_db as studio_db
from core.coding_traces.schema import (
    ChatTrace,
    ChatTraceError,
    ToolCall,
    TraceMessage,
    TraceWarning,
)

# Same order as compareStoredMessages in frontend/src/features/chat/utils/message-order.ts.
_ROLE_ORDER = {"system": 0, "user": 1, "assistant": 2}


def _sort_key(message: dict) -> tuple:
    return (
        message.get("createdAt") or 0,
        _ROLE_ORDER.get(message.get("role"), 99),
        str(message["id"]),
    )


def _resolve_parents(ordered: list[dict]) -> dict[str, Optional[str]]:
    # Mirrors createParentResolver: rows saved before parent links existed chain to the
    # previous row, while a null parent after any recorded parent starts a new root.
    parents: dict[str, Optional[str]] = {}
    previous_id = None
    saw_recorded_parent = False
    for message in ordered:
        parent_id = message.get("parentId")
        inferred = "parentId" not in message or (parent_id is None and not saw_recorded_parent)
        parents[message["id"]] = previous_id if inferred else parent_id
        if parent_id is not None:
            saw_recorded_parent = True
        previous_id = message["id"]
    return parents


def _newest_leaf(head_id: str, ordered: list[dict], parents: dict[str, Optional[str]]) -> str:
    # The UI saves a branch head and then shows its newest descendant (resolveSavedBranchHead).
    position = {message["id"]: index for index, message in enumerate(ordered)}
    children: dict[str, list[str]] = {}
    for message_id, parent_id in parents.items():
        if parent_id is not None:
            children.setdefault(parent_id, []).append(message_id)
    best, pending, seen = head_id, [head_id], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if position[current] > position[best]:
            best = current
        pending.extend(children.get(current, []))
    return best


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"]
    return json.dumps(result, ensure_ascii = False)


def _append_text(current: str, addition: str) -> str:
    if not addition:
        return current
    return f"{current}\n\n{addition}" if current else addition


def _branch(
    messages: list[dict], head_id: Optional[str], warnings: list[TraceWarning]
) -> list[dict]:
    ordered = sorted(messages, key = _sort_key)
    by_id = {message["id"]: message for message in ordered}
    parents = _resolve_parents(ordered)
    if head_id is None:
        current = ordered[-1]["id"]
    elif head_id in by_id:
        current = _newest_leaf(head_id, ordered, parents)
    else:
        raise ChatTraceError(f"Message {head_id!r} is not in this thread")

    chain: list[dict] = []
    seen: set[str] = set()
    while current is not None and current not in seen:
        message = by_id.get(current)
        if message is None:
            warnings.append(
                TraceWarning(
                    "missing_parent",
                    f"Parent {current!r} is not saved; the trace starts after it",
                    chain[-1]["id"],
                )
            )
            break
        seen.add(current)
        chain.append(message)
        current = parents[current]
    chain.reverse()

    dropped = len(messages) - len(chain)
    if dropped:
        warnings.append(
            TraceWarning(
                "other_branches_dropped", f"Left out {dropped} saved message(s) from other branches"
            )
        )
    return chain


def _convert(chain: list[dict], warnings: list[TraceWarning]) -> list[TraceMessage]:
    out: list[TraceMessage] = []
    call_ids: set[str] = set()
    for message in chain:
        role, message_id = message.get("role"), message["id"]
        if role not in ("system", "user", "assistant"):
            warnings.append(TraceWarning("unknown_role", f"Skipped role {role!r}", message_id))
            continue
        if role != "assistant":
            current = TraceMessage(id = message_id, role = role, created_at = message.get("createdAt"))
            out.append(current)
        for part in message.get("content") or []:
            kind = part.get("type") if isinstance(part, dict) else None
            if role == "assistant":
                # Importers split one reply into a row per block; merge rows until a tool result.
                if not out or out[-1].role != "assistant":
                    out.append(
                        TraceMessage(
                            id = message_id, role = "assistant", created_at = message.get("createdAt")
                        )
                    )
                current = out[-1]
            if kind == "text":
                current.text = _append_text(current.text, str(part.get("text") or ""))
            elif kind in ("reasoning", "thinking") and role == "assistant":
                reasoning = part.get("text") or part.get("thinking") or ""
                current.reasoning = _append_text(current.reasoning, str(reasoning))
            elif kind == "tool-call" and role == "assistant":
                call_id = str(part.get("toolCallId") or f"{message_id}:call")
                if call_id in call_ids:
                    suffix = 2
                    while f"{call_id}#{suffix}" in call_ids:
                        suffix += 1
                    renamed = f"{call_id}#{suffix}"
                    warnings.append(
                        TraceWarning(
                            "duplicate_tool_call_id",
                            f"Tool call id {call_id!r} repeats; renamed to {renamed!r}",
                            message_id,
                        )
                    )
                    call_id = renamed
                call_ids.add(call_id)
                args = part.get("args")
                current.tool_calls.append(
                    ToolCall(
                        id = call_id,
                        name = str(part.get("toolName") or "unknown"),
                        arguments = args if isinstance(args, dict) else {},
                    )
                )
                if part.get("result") is None:
                    warnings.append(
                        TraceWarning(
                            "unanswered_tool_call",
                            f"Tool call {call_id!r} has no result",
                            message_id,
                        )
                    )
                    continue
                out.append(
                    TraceMessage(
                        id = f"{message_id}:{call_id}",
                        role = "tool",
                        text = _result_text(part["result"]),
                        tool_call_id = call_id,
                        is_error = bool(part.get("isError")),
                        created_at = message.get("createdAt"),
                    )
                )
            else:
                warnings.append(
                    TraceWarning("unsupported_part", f"Dropped a {kind!r} part", message_id)
                )
    return [m for m in out if m.text or m.reasoning or m.tool_calls or m.role == "tool"]


def chat_to_trace(
    thread: dict,
    messages: list[dict],
    *,
    project: Optional[dict] = None,
    head_id: Optional[str] = None,
) -> ChatTrace:
    """Turn a saved Studio thread and its message rows into one coding trace branch."""
    if not messages:
        raise ChatTraceError(f"Thread {thread.get('id')!r} has no messages")
    warnings: list[TraceWarning] = []
    trace_messages = _convert(_branch(messages, head_id, warnings), warnings)
    if not trace_messages:
        raise ChatTraceError(f"Thread {thread.get('id')!r} has no usable content on this branch")
    if not any(m.role == "user" for m in trace_messages):
        warnings.append(TraceWarning("no_user_message", "The branch has no user message"))
    return ChatTrace(
        thread_id = thread["id"],
        title = thread.get("title") or "",
        model_id = thread.get("modelId") or "",
        created_at = thread.get("createdAt"),
        updated_at = thread.get("updatedAt"),
        messages = trace_messages,
        warnings = warnings,
        project_id = thread.get("projectId"),
        project_name = (project or {}).get("name"),
        working_folder = (project or {}).get("workingFolder"),
    )


def load_chat_trace(thread_id: str, *, head_id: Optional[str] = None) -> ChatTrace:
    """Load a saved Studio thread from the database and turn it into a coding trace."""
    thread = studio_db.get_chat_thread(thread_id)
    if thread is None:
        raise ChatTraceError(f"Thread {thread_id!r} does not exist")
    project_id = thread.get("projectId")
    project = studio_db.get_chat_project(project_id) if project_id else None
    return chat_to_trace(
        thread, studio_db.list_chat_messages(thread_id), project = project, head_id = head_id
    )
