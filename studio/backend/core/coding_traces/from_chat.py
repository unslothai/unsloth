# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Builds a coding trace from one saved Studio chat thread, in the order the UI shows.

The system text uses the thread's latest saved settings and the project's current instructions;
Studio keeps no history of either, so an earlier turn may have seen different text.
"""

import json
import re
from typing import Optional

from core.agents_md import agents_md_enabled
from core.coding_traces.schema import (
    ChatTrace,
    ChatTraceError,
    ToolCall,
    TraceMessage,
    TraceWarning,
)
from core.inference.tool_loop_controller import (
    TOOL_MODEL_SUFFIX_KEY,
    restore_replayed_tool_results,
)
from core.rag.conversation_archive import (
    _flushes_local_pair,
    _local_round_id,
    _mcp_image_result,
    _sanitised_assistant_text,
    _search_images_result,
    _server_builtin,
    _tool_result_content,
    _with_attachments,
)
from storage import studio_db

# Same order as compareStoredMessages in frontend/src/features/chat/utils/message-order.ts.
_ROLE_ORDER = {"system": 0, "user": 1, "assistant": 2}

# INCOMPLETE_LABELS in frontend/src/features/chat/utils/continuation.ts.
_STOP_LABELS = {
    "length": "Response hit the Max Tokens limit",
    "cancelled": "Response stopped",
    "interrupted": "Response interrupted",
    "context_window": "Response filled the model's context window",
    "empty": "The model returned an empty response",
    "quote_cut": "This response may have ended early",
}

# resolveSystemPromptVariables in frontend/src/features/chat/api/chat-adapter.ts.
_PROMPT_VARIABLE = re.compile(r"{{\s*([a-zA-Z_$][a-zA-Z0-9_$.-]*)\s*}}")


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _sort_key(message: dict) -> tuple:
    return (
        message.get("createdAt") or 0,
        _ROLE_ORDER.get(message.get("role"), 99),
        str(message["id"]),
    )


def _resolve_parents(ordered: list[dict]) -> dict[str, Optional[str]]:
    # Mirrors createParentResolver (and studio_db.fork_chat_thread): rows saved before parent links
    # chain to the previous row; a null parent after any recorded parent starts a new root.
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
    """Content and attachments minus mcpToolOnly parts, as modelVisibleMessage (mcp-image.ts)."""
    content = _with_attachments(message) or []
    return [p for p in content if not (isinstance(p, dict) and p.get("mcpToolOnly") is True)]


def _append_text(current: str, addition: str, joined: bool) -> str:
    return f"{current}\n{addition}" if joined or (current and addition) else current or addition


def _branch(
    messages: list[dict], head_id: Optional[str], warnings: list[TraceWarning]
) -> list[dict]:
    ordered = sorted(messages, key = _sort_key)
    by_id = {message["id"]: message for message in ordered}
    parents = _resolve_parents(ordered)
    if head_id is None:
        current = _newest_leaf(ordered[-1]["id"], ordered, parents)
    elif head_id in by_id:
        current = _newest_leaf(head_id, ordered, parents)
    else:
        raise ChatTraceError(f"Message {head_id!r} is not in this thread")

    chain: list[dict] = []
    seen: set[str] = set()
    while current is not None and current not in seen:
        message = by_id.get(current)
        if message is None:
            note = f"Parent {current!r} is not saved; the trace starts after it"
            warnings.append(TraceWarning("missing_parent", note, chain[-1]["id"]))
            break
        seen.add(current)
        chain.append(message)
        current = parents[current]
    chain.reverse()

    dropped = len(messages) - len(chain)
    if dropped:
        note = f"Left out {dropped} saved message(s) not on the selected branch"
        warnings.append(TraceWarning("other_branches_dropped", note))
    return chain


def _convert(chain: list[dict], warnings: list[TraceWarning]) -> list[TraceMessage]:
    out: list[TraceMessage] = []
    call_ids: set[str] = set()
    for message in chain:
        role, message_id, start, kept = message.get("role"), message["id"], len(out), []
        if role not in ("system", "user", "assistant"):
            warnings.append(TraceWarning("unknown_role", f"Skipped role {role!r}", message_id))
            continue
        if role != "assistant":
            current = TraceMessage(id = message_id, role = role, created_at = message.get("createdAt"))
            out.append(current)
        # Tool messages wait for their call group to close (serializeAssistantReplayMessages).
        pending: list[TraceMessage] = []
        pending_round = reasoned = imaged = None
        # collectTextParts joins all text parts, even "", and toThreadMessage loads [] as one "".
        pieces, texted = 0, (current if role == "user" and not message.get("content") else None)
        for position, part in enumerate(_model_parts(message)):
            kind = part.get("type") if isinstance(part, dict) else None
            if role == "assistant" and kind in ("text", "reasoning", "tool-call"):
                skipped = _skipped_call(part, message_id, position) if kind == "tool-call" else None
                if skipped:
                    warnings.append(skipped)
                    kept += out[start:][-1:] if _server_builtin(part)[1] else []
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
                    piece = message_id if pieces == 1 else f"{message_id}#{pieces}"
                    at = message.get("createdAt")
                    out.append(TraceMessage(piece, "assistant", created_at = at))
                current = out[-1]
            if kind == "text":
                text = str(part.get("text") or "")
                current.text, texted = _append_text(current.text, text, current is texted), current
            elif kind == "reasoning" and role == "assistant":
                reasoning = str(part.get("text") or "")
                current.reasoning = _append_text(current.reasoning, reasoning, current is reasoned)
                reasoned = current  # pendingReasoningParts.join("\n") also keeps "" parts
            elif kind == "tool-call" and role == "assistant":
                call_id = str(part.get("toolCallId") or f"{message_id}-{position}")
                if call_id in call_ids:
                    names = (f"{call_id}#{n}" for n in range(2, len(call_ids) + 3))
                    renamed = next(name for name in names if name not in call_ids)
                    note = f"Tool call id {call_id!r} repeats; renamed to {renamed!r}"
                    warnings.append(TraceWarning("duplicate_tool_call_id", note, message_id))
                    call_id = renamed
                call_ids.add(call_id)
                args, raw = part.get("args"), part.get("argsText")
                try:  # toolCallReplayArguments sends argsText as is when it is one JSON object
                    exact = json.loads(raw, parse_constant = _reject_constant)
                except (TypeError, ValueError):
                    exact = None
                marker = isinstance(raw, str) and raw and args == {"_raw": raw}  # replays as {}
                args = exact if isinstance(exact, dict) else {} if marker else args
                name = str(part.get("toolName") or "")
                current.tool_calls.append(
                    ToolCall(
                        id = call_id,
                        name = name or "unknown",
                        arguments = args if isinstance(args, dict) else {},
                    )
                )
                wire = {"role": "tool", "name": name}
                result = part["result"]  # isMcpUiToolResult, then the gated isMcpImageToolResult
                ui = _as_dict(result.get("ui")) if isinstance(result, dict) else {}
                if name.startswith("mcp__") and isinstance(ui.get("resourceUri"), str):
                    result = result["text"] if isinstance(result.get("text"), str) else result
                elif _mcp_image_result(result) and not _search_images_result(result):
                    image_tool = name == "view_image" or name.startswith("mcp__")  # isImageToolName
                    if "ui" in result or not (image_tool or set(result) <= {"text", "images"}):
                        result = json.dumps(result, ensure_ascii = False, separators = (",", ":"))
                wire["content"] = _tool_result_content(result, name)
                suffix = _as_dict(part.get("provenance")).get("model_suffix")
                if isinstance(suffix, str) and suffix:
                    wire["extra_content"] = {TOOL_MODEL_SUFFIX_KEY: suffix}
                pending.append(
                    TraceMessage(
                        id = f"{message_id}:{call_id}",
                        role = "tool",
                        text = restore_replayed_tool_results([wire])[0]["content"],
                        tool_call_id = call_id,
                        is_error = bool(part.get("isError")),
                        created_at = message.get("createdAt"),
                    )
                )
                if flush_pair:
                    out.extend(pending)
                    pending, pending_round = [], None
            else:
                imaged = imaged or (kind == "image" and part.get("image"))
                note = f"Dropped a {kind!r} part"
                warnings.append(TraceWarning("unsupported_part", note, message_id))
        out.extend(pending)
        kept += out[start:][:1] if imaged else []  # collectImageParts: the row's first UI group
        reason = _as_dict(_as_dict(message.get("metadata")).get("incomplete")).get("reason")
        if isinstance(reason, str) and reason in _STOP_LABELS:
            for m in out[start:]:
                if not (_sanitised_assistant_text(m.text) or m.tool_calls or m in kept):
                    m.reasoning = ""  # shouldReplayAssistantReasoning: the text check is untrimmed
    for m in out:
        if m.role == "assistant":
            m.text = _sanitised_assistant_text(m.text)
    # hasReplayContent trims assistant text; a user's text is sent as it is.
    return [
        m
        for m in out
        if (m.text.strip() if m.role == "assistant" else m.text)
        or m.reasoning
        or m.tool_calls
        or m.role == "tool"
    ]


def _skipped_call(part: dict, message_id: str, position: int) -> Optional[TraceWarning]:
    """The warning for a tool-call part the UI never replays as a call, or None to keep it."""
    name = part.get("toolName")
    if name == "studio_load_skill":
        # A request-local system-prompt preload (skill_mentions.py); chat-adapter.ts drops the card.
        skill = _as_dict(part.get("args")).get("name")
        return TraceWarning(
            "skill_preload_not_stored",
            f"Skill @{skill} was added to that request's system prompt; its text is not saved",
            message_id,
        )
    builtin, native = _server_builtin(part)
    if builtin:
        # A native part would be a call with no tool message; the provider ran it either way.
        detail = "; the UI replays its code and output to Gemini" if native else ""
        return TraceWarning(
            "server_tool_skipped", f"Skipped provider-run tool {name!r}{detail}", message_id
        )
    if part.get("result") is None:
        # serializeToolResultPart (chat-adapter.ts) and conversation_archive._replayable drop it.
        call_id = part.get("toolCallId") or f"{message_id}-{position}"
        note = f"Dropped tool call {call_id!r} ({name!r}); it has no stored result"
        return TraceWarning("unanswered_tool_call_dropped", note, message_id)
    return None


def _prune(chain: list[dict]) -> list[dict]:
    """Drop refused and abandoned turns as pruneOutboundHistory (chat-adapter.ts) does, and give an
    empty stopped turn its label, as fillStoppedAssistantReplay does. An unsettled run's row gets no
    "Response interrupted" label: the UI asks the server if the run is live, and rows lack that."""

    def replayed(message, reasoning):
        # The UI sends an image (collectImageParts) and a native Gemini call; the trace drops both.
        return any(
            p.get("image") if p.get("type") == "image" else _server_builtin(p)[1]
            for p in _model_parts(message)
            if isinstance(p, dict) and p.get("type") in ("image", "tool-call")
        ) or any(
            m.text.strip() or m.tool_calls or (m.reasoning and (reasoning or m.text))
            for m in _convert([message], [])
        )

    def abandoned(message):
        # assistantTurnCarriesPayload reads the raw parts, before the replay sanitiser, and
        # isAbandonedAssistantTurn keeps any turn with a reasoning part, even an empty one.
        payload = any(
            isinstance(p, dict)
            and (
                p.get("type") == "reasoning"
                or (p.get("type") == "image" and p.get("image"))
                or (p.get("type") == "text" and str(p.get("text") or "").strip())
            )
            for p in _model_parts(message)
        )
        return not payload and not replayed(message, reasoning = True)

    meta = [_as_dict(m.get("metadata")) for m in chain]  # runtime-provider.tsx: metadata.custom
    assistant = [m.get("role") == "assistant" for m in chain]
    refused = [a and d.get("anthropicRefusal") is True for d, a in zip(meta, assistant)]
    reasons = [_as_dict(d.get("incomplete")).get("reason") for d in meta]
    stopped = [isinstance(r, str) and r in _STOP_LABELS for r in reasons]
    dropped = [
        r or (a and not s and abandoned(m))
        for m, a, r, s in zip(chain, assistant, refused, stopped)
    ]
    last = max((i for i, d in enumerate(dropped) if not d), default = -1)
    kept: list[dict] = []
    for index, message in enumerate(chain):
        if dropped[index]:
            if (refused[index] or index < last) and kept and kept[-1].get("role") == "user":
                kept.pop()
            continue
        # The UI never replays the reasoning of a stopped turn with no text and no calls.
        if assistant[index] and stopped[index] and not replayed(message, reasoning = False):
            label = _STOP_LABELS[reasons[index]]
            message = {**message, "content": [{"type": "text", "text": label}], "attachments": []}
        kept.append(message)
    return kept


def _js_float(text: str) -> float | int:
    value = float(text)
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def _reject_constant(text: str) -> None:
    raise ValueError(f"{text} is not JSON")


def _fill_variables(prompt: str, raw: object) -> str:
    """Fill custom systemVariables as resolveSystemPromptVariables does; $date, $time and $now
    used the send-time clock, which is not stored, so they stay as written."""
    try:
        # parseSystemVariablesMap uses JSON.parse, so String() prints integral floats as ints,
        # and a NaN fills nothing. Numbers below 1e-4 or above 2**53 can still print unlike JS.
        values = (
            json.loads(raw, parse_float = _js_float, parse_constant = _reject_constant)
            if isinstance(raw, str) and raw.strip()
            else {}
        )
    except ValueError:
        values = {}

    def fill(match):
        current = values if isinstance(values, dict) else {}
        if match.group(1) in ("$date", "$time", "$now"):
            return match.group(0)
        for key in filter(None, (p.strip() for p in match.group(1).split("."))):
            if not isinstance(current, dict) or key not in current:
                return match.group(0)
            current = current[key]
        if current is None:
            return ""
        if isinstance(current, str):
            return current
        return json.dumps(current, separators = (",", ":"), ensure_ascii = False)

    return _PROMPT_VARIABLE.sub(fill, prompt)


def _system_message(
    thread: dict, project: Optional[dict], warnings: list[TraceWarning]
) -> Optional[TraceMessage]:
    """The system turn as composeChatInstructions (agents-md.ts) builds it from stored fields."""
    settings = _as_dict(thread.get("settings"))
    prompt = settings.get("systemPrompt")
    if not isinstance(prompt, str):
        # No thread snapshot means the chat ran on the browser's global prompt, which is not saved.
        note = "The thread stores no system prompt; the trace may lack the one the model saw"
        warnings.append(TraceWarning("system_prompt_not_stored", note))
        prompt = ""
    prompt = _fill_variables(prompt, settings.get("systemVariables"))
    if _PROMPT_VARIABLE.search(prompt):
        note = "Send-time or unknown system prompt variables are left as stored"
        warnings.append(TraceWarning("system_prompt_variables", note))
    instructions = ""
    if project and not project.get("archived"):
        instructions = str(project.get("instructions") or "").strip()
    if agents_md_enabled():
        note = "AGENTS.md is read at send time and not stored; it may have come before this text"
        warnings.append(TraceWarning("agents_md_not_stored", note))
    if instructions:
        instructions = f"<project_instructions>\n{instructions}\n</project_instructions>"
    text = "\n\n".join(part for part in (instructions, prompt.strip()) if part)
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

    head_id picks the branch like a saved UI branch head (its newest leaf); None uses the newest
    row's leaf. Unlike the UI, an unsaved id raises ChatTraceError instead of falling back.
    Consecutive assistant rows merge (importers split one reply into rows; the UI sends each row).
    Reasoning is kept as for a local model; the UI drops it for most external providers, which rows
    do not record. Tool text is the restored replay text; MCP image envelopes are left out.
    """
    if not messages:
        raise ChatTraceError(f"Thread {thread.get('id')!r} has no messages")
    warnings: list[TraceWarning] = []
    trace_messages = _convert(_prune(_branch(messages, head_id, warnings)), warnings)
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
    """Load a saved Studio thread from the database and turn it into a coding trace; head_id
    works as in chat_to_trace."""
    thread = studio_db.get_chat_thread(thread_id)
    if thread is None:
        raise ChatTraceError(f"Thread {thread_id!r} does not exist")
    project_id = thread.get("projectId")
    project = studio_db.get_chat_project(project_id) if project_id else None
    return chat_to_trace(
        thread, studio_db.list_chat_messages(thread_id), project = project, head_id = head_id
    )
