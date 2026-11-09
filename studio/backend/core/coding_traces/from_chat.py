# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Builds a coding trace from one saved Studio chat thread, in the order the UI shows."""

from typing import Optional

from core.coding_traces.schema import (
    ChatTrace,
    ChatTraceError,
    ToolCall,
    TraceMessage,
    TraceWarning,
)
from core.rag.conversation_archive import (
    _flushes_local_pair,
    _local_round_id,
    _server_builtin,
    _tool_result_content,
    _with_attachments,
)
from storage import studio_db

# Same order as compareStoredMessages in frontend/src/features/chat/utils/message-order.ts.
_ROLE_ORDER = {"system": 0, "user": 1, "assistant": 2}


def _sort_key(message: dict) -> tuple:
    return (
        message.get("createdAt") or 0,
        _ROLE_ORDER.get(message.get("role"), 99),
        str(message["id"]),
    )


def _resolve_parents(ordered: list[dict]) -> dict[str, Optional[str]]:
    # Mirrors createParentResolver (and the parent loop in studio_db.fork_chat_thread): rows saved
    # before parent links existed chain to the previous row, while a null parent after any recorded
    # parent starts a new root.
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
    # The UI saves a branch head and then shows its newest leaf (resolveSavedBranchHead); inner
    # nodes never win, even when an edited timestamp sorts them after the leaf.
    position = {message["id"]: index for index, message in enumerate(ordered)}
    children: dict[str, list[str]] = {}
    for message_id, parent_id in parents.items():
        if parent_id is not None:
            children.setdefault(parent_id, []).append(message_id)
    best, best_position, pending, seen = head_id, -1, [head_id], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        below = children.get(current)
        if below:
            pending.extend(below)
        elif position[current] > best_position:
            best, best_position = current, position[current]
    return best


def _model_parts(message: dict) -> list:
    """Content plus attachments, minus mcpToolOnly parts, as modelVisibleMessage (mcp-image.ts) sends them."""
    content = _with_attachments(message) or []
    return [p for p in content if not (isinstance(p, dict) and p.get("mcpToolOnly") is True)]


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
                "other_branches_dropped",
                f"Left out {dropped} saved message(s) that are not on the selected branch",
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
        # Tool messages wait until their call group closes, as serializeAssistantReplayMessages
        # (chat-adapter.ts) and conversation_archive._as_wire group one row's calls.
        pending: list[TraceMessage] = []
        pending_round = None
        pieces = 0
        for position, part in enumerate(_model_parts(message)):
            kind = part.get("type") if isinstance(part, dict) else None
            if role == "assistant":
                if kind == "tool-call" and _server_builtin(part)[0]:
                    # The provider ran it; the model never got it back as a tool message.
                    warnings.append(
                        TraceWarning(
                            "server_tool_skipped",
                            f"Skipped provider-run tool {part.get('toolName')!r}",
                            message_id,
                        )
                    )
                    continue
                round_id = _local_round_id(part) if kind == "tool-call" else None
                flush_pair = kind == "tool-call" and round_id is None and _flushes_local_pair(part)
                if pending and (
                    kind != "tool-call"
                    or flush_pair
                    or (round_id is not None and pending_round not in (None, round_id))
                ):
                    out.extend(pending)
                    pending, pending_round = [], None
                if round_id is not None:
                    pending_round = round_id
                # Importers split one reply into a row per block; merge rows until a tool result.
                if not out or out[-1].role != "assistant":
                    pieces += 1
                    out.append(
                        TraceMessage(
                            id = message_id if pieces == 1 else f"{message_id}#{pieces}",
                            role = "assistant",
                            created_at = message.get("createdAt"),
                        )
                    )
                current = out[-1]
            if kind == "text":
                current.text = _append_text(current.text, str(part.get("text") or ""))
            elif kind in ("reasoning", "thinking") and role == "assistant":
                reasoning = part.get("text") or part.get("thinking") or ""
                current.reasoning = _append_text(current.reasoning, str(reasoning))
            elif kind == "tool-call" and role == "assistant":
                call_id = str(part.get("toolCallId") or f"{message_id}-{position}")
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
                name = str(part.get("toolName") or "")
                current.tool_calls.append(
                    ToolCall(
                        id = call_id,
                        name = name or "unknown",
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
                pending.append(
                    TraceMessage(
                        id = f"{message_id}:{call_id}",
                        role = "tool",
                        text = _tool_result_content(part["result"], name),
                        tool_call_id = call_id,
                        is_error = bool(part.get("isError")),
                        created_at = message.get("createdAt"),
                    )
                )
                if flush_pair:
                    out.extend(pending)
                    pending, pending_round = [], None
            else:
                warnings.append(
                    TraceWarning("unsupported_part", f"Dropped a {kind!r} part", message_id)
                )
        out.extend(pending)
    return [m for m in out if m.text or m.reasoning or m.tool_calls or m.role == "tool"]


def _system_message(
    thread: dict, project: Optional[dict], warnings: list[TraceWarning]
) -> Optional[TraceMessage]:
    """The system turn as composeChatInstructions (agents-md.ts) builds it from stored fields."""
    settings = thread.get("settings")
    prompt = settings.get("systemPrompt") if isinstance(settings, dict) else None
    if not isinstance(prompt, str):
        # No thread snapshot means the chat ran on the browser's global prompt, which is not saved.
        warnings.append(
            TraceWarning(
                "system_prompt_not_stored",
                "The thread stores no system prompt; the trace may lack the one the model saw",
            )
        )
        prompt = ""
    elif "{{" in prompt:
        warnings.append(
            TraceWarning(
                "system_prompt_variables",
                "System prompt template variables are left as stored, not resolved",
            )
        )
    instructions = ""
    if project and not project.get("archived"):
        instructions = str(project.get("instructions") or "").strip()
    # AGENTS.md is read live at send time and not kept with the thread, so it is not added.
    text = "\n\n".join(
        part
        for part in (
            f"<project_instructions>\n{instructions}\n</project_instructions>"
            if instructions
            else "",
            prompt.strip(),
        )
        if part
    )
    if not text:
        return None
    return TraceMessage(id = f"{thread['id']}:system", role = "system", text = text)


def chat_to_trace(
    thread: dict,
    messages: list[dict],
    *,
    project: Optional[dict] = None,
    head_id: Optional[str] = None,
) -> ChatTrace:
    """Turn a saved Studio thread and its message rows into one coding trace branch.

    head_id picks the branch like a saved UI branch head (its newest leaf); None follows the newest
    row. Unlike the UI, an id that is not saved raises ChatTraceError instead of falling back.
    """
    if not messages:
        raise ChatTraceError(f"Thread {thread.get('id')!r} has no messages")
    warnings: list[TraceWarning] = []
    trace_messages = _convert(_branch(messages, head_id, warnings), warnings)
    if not trace_messages:
        raise ChatTraceError(f"Thread {thread.get('id')!r} has no usable content on this branch")
    if not any(m.role == "user" for m in trace_messages):
        warnings.append(TraceWarning("no_user_message", "The branch has no user message"))
    system = _system_message(thread, project, warnings)
    if system is not None:
        trace_messages.insert(0, system)
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
    """Load a saved Studio thread from the database and turn it into a coding trace; head_id works as in chat_to_trace."""
    thread = studio_db.get_chat_thread(thread_id)
    if thread is None:
        raise ChatTraceError(f"Thread {thread_id!r} does not exist")
    project_id = thread.get("projectId")
    project = studio_db.get_chat_project(project_id) if project_id else None
    return chat_to_trace(
        thread, studio_db.list_chat_messages(thread_id), project = project, head_id = head_id
    )
